#include <metal_stdlib>
using namespace metal;

#define REDUCE_GROUP_SIZE 256

struct DeviceExpr {
  float c0;
  float x;
  float y;

  float eval(float px, float py) const {
    return c0 + x * px + y * py;
  }
};

struct DeviceBC {
  int kind;
  DeviceExpr value;
  DeviceExpr alpha;
  DeviceExpr beta;
  DeviceExpr gamma;
};

struct DomainInfo {
  int nx;
  int ny;
  float xmin;
  float xmax;
  float ymin;
  float ymax;
  float dx;
  float dy;
};

inline int Index(int i, int j, int nx) {
  return j * nx + i;
}

kernel void apply_boundaries(device float* grid [[buffer(0)]],
                             constant DomainInfo* domain_ptr [[buffer(1)]],
                             constant DeviceBC* left_ptr [[buffer(2)]],
                             constant DeviceBC* right_ptr [[buffer(3)]],
                             constant DeviceBC* bottom_ptr [[buffer(4)]],
                             constant DeviceBC* top_ptr [[buffer(5)]],
                             uint2 tid [[thread_position_in_grid]]) {
  const DomainInfo domain = *domain_ptr;
  const DeviceBC left_bc = *left_ptr;
  const DeviceBC right_bc = *right_ptr;
  const DeviceBC bottom_bc = *bottom_ptr;
  const DeviceBC top_bc = *top_ptr;

  const int i = static_cast<int>(tid.x);
  const int j = static_cast<int>(tid.y);
  if (i >= domain.nx || j >= domain.ny) {
    return;
  }

  const float x = domain.xmin + i * domain.dx;
  const float y = domain.ymin + j * domain.dy;

  if (i == 0) {
    if (left_bc.kind == 0) {
      grid[Index(0, j, domain.nx)] = left_bc.value.eval(domain.xmin, y);
    } else if (left_bc.kind == 1) {
      const float g = left_bc.value.eval(domain.xmin, y);
      grid[Index(0, j, domain.nx)] = grid[Index(1, j, domain.nx)] - domain.dx * g;
    } else if (left_bc.kind == 2) {
      const float alpha = left_bc.alpha.eval(domain.xmin, y);
      const float beta = left_bc.beta.eval(domain.xmin, y);
      const float gamma = left_bc.gamma.eval(domain.xmin, y);
      const float denom = alpha + beta / domain.dx;
      if (fabs(denom) > 1e-6f) {
        grid[Index(0, j, domain.nx)] = (gamma + (beta / domain.dx) * grid[Index(1, j, domain.nx)]) / denom;
      }
    }
  }

  if (i == domain.nx - 1) {
    if (right_bc.kind == 0) {
      grid[Index(domain.nx - 1, j, domain.nx)] = right_bc.value.eval(domain.xmax, y);
    } else if (right_bc.kind == 1) {
      const float g = right_bc.value.eval(domain.xmax, y);
      grid[Index(domain.nx - 1, j, domain.nx)] = grid[Index(domain.nx - 2, j, domain.nx)] + domain.dx * g;
    } else if (right_bc.kind == 2) {
      const float alpha = right_bc.alpha.eval(domain.xmax, y);
      const float beta = right_bc.beta.eval(domain.xmax, y);
      const float gamma = right_bc.gamma.eval(domain.xmax, y);
      const float denom = alpha + beta / domain.dx;
      if (fabs(denom) > 1e-6f) {
        grid[Index(domain.nx - 1, j, domain.nx)] =
          (gamma + (beta / domain.dx) * grid[Index(domain.nx - 2, j, domain.nx)]) / denom;
      }
    }
  }

  if (j == 0) {
    if (bottom_bc.kind == 0) {
      grid[Index(i, 0, domain.nx)] = bottom_bc.value.eval(x, domain.ymin);
    } else if (bottom_bc.kind == 1) {
      const float g = bottom_bc.value.eval(x, domain.ymin);
      grid[Index(i, 0, domain.nx)] = grid[Index(i, 1, domain.nx)] - domain.dy * g;
    } else if (bottom_bc.kind == 2) {
      const float alpha = bottom_bc.alpha.eval(x, domain.ymin);
      const float beta = bottom_bc.beta.eval(x, domain.ymin);
      const float gamma = bottom_bc.gamma.eval(x, domain.ymin);
      const float denom = alpha + beta / domain.dy;
      if (fabs(denom) > 1e-6f) {
        grid[Index(i, 0, domain.nx)] = (gamma + (beta / domain.dy) * grid[Index(i, 1, domain.nx)]) / denom;
      }
    }
  }

  if (j == domain.ny - 1) {
    if (top_bc.kind == 0) {
      grid[Index(i, domain.ny - 1, domain.nx)] = top_bc.value.eval(x, domain.ymax);
    } else if (top_bc.kind == 1) {
      const float g = top_bc.value.eval(x, domain.ymax);
      grid[Index(i, domain.ny - 1, domain.nx)] = grid[Index(i, domain.ny - 2, domain.nx)] + domain.dy * g;
    } else if (top_bc.kind == 2) {
      const float alpha = top_bc.alpha.eval(x, domain.ymax);
      const float beta = top_bc.beta.eval(x, domain.ymax);
      const float gamma = top_bc.gamma.eval(x, domain.ymax);
      const float denom = alpha + beta / domain.dy;
      if (fabs(denom) > 1e-6f) {
        grid[Index(i, domain.ny - 1, domain.nx)] =
          (gamma + (beta / domain.dy) * grid[Index(i, domain.ny - 2, domain.nx)]) / denom;
      }
    }
  }
}

kernel void jacobi_step(const device float* grid [[buffer(0)]],
                        device float* next [[buffer(1)]],
                        device float* delta [[buffer(2)]],
                        constant DomainInfo* domain_ptr [[buffer(3)]],
                        constant DeviceBC* left_ptr [[buffer(4)]],
                        constant DeviceBC* right_ptr [[buffer(5)]],
                        constant DeviceBC* bottom_ptr [[buffer(6)]],
                        constant DeviceBC* top_ptr [[buffer(7)]],
                        constant float& ax [[buffer(8)]],
                        constant float& by [[buffer(9)]],
                        constant float& cx [[buffer(10)]],
                        constant float& dyc [[buffer(11)]],
                        constant float& center [[buffer(12)]],
                        constant float& fval [[buffer(13)]],
                        uint2 tid [[thread_position_in_grid]]) {
  const DomainInfo domain = *domain_ptr;
  const DeviceBC left_bc = *left_ptr;
  const DeviceBC right_bc = *right_ptr;
  const DeviceBC bottom_bc = *bottom_ptr;
  const DeviceBC top_bc = *top_ptr;

  const int i = static_cast<int>(tid.x);
  const int j = static_cast<int>(tid.y);
  if (i >= domain.nx || j >= domain.ny) {
    return;
  }

  const int idx = Index(i, j, domain.nx);
  if (i == 0 || j == 0 || i == domain.nx - 1 || j == domain.ny - 1) {
    const float x = domain.xmin + i * domain.dx;
    const float y = domain.ymin + j * domain.dy;
    float value = grid[idx];

    if (i == 0) {
      if (left_bc.kind == 0) {
        value = left_bc.value.eval(domain.xmin, y);
      } else if (left_bc.kind == 1) {
        const float g = left_bc.value.eval(domain.xmin, y);
        value = grid[Index(1, j, domain.nx)] - domain.dx * g;
      } else if (left_bc.kind == 2) {
        const float alpha = left_bc.alpha.eval(domain.xmin, y);
        const float beta = left_bc.beta.eval(domain.xmin, y);
        const float gamma = left_bc.gamma.eval(domain.xmin, y);
        const float denom = alpha + beta / domain.dx;
        if (fabs(denom) > 1e-6f) {
          value = (gamma + (beta / domain.dx) * grid[Index(1, j, domain.nx)]) / denom;
        }
      }
    }

    if (i == domain.nx - 1) {
      if (right_bc.kind == 0) {
        value = right_bc.value.eval(domain.xmax, y);
      } else if (right_bc.kind == 1) {
        const float g = right_bc.value.eval(domain.xmax, y);
        value = grid[Index(domain.nx - 2, j, domain.nx)] + domain.dx * g;
      } else if (right_bc.kind == 2) {
        const float alpha = right_bc.alpha.eval(domain.xmax, y);
        const float beta = right_bc.beta.eval(domain.xmax, y);
        const float gamma = right_bc.gamma.eval(domain.xmax, y);
        const float denom = alpha + beta / domain.dx;
        if (fabs(denom) > 1e-6f) {
          value = (gamma + (beta / domain.dx) * grid[Index(domain.nx - 2, j, domain.nx)]) / denom;
        }
      }
    }

    if (j == 0) {
      if (bottom_bc.kind == 0) {
        value = bottom_bc.value.eval(x, domain.ymin);
      } else if (bottom_bc.kind == 1) {
        const float g = bottom_bc.value.eval(x, domain.ymin);
        value = grid[Index(i, 1, domain.nx)] - domain.dy * g;
      } else if (bottom_bc.kind == 2) {
        const float alpha = bottom_bc.alpha.eval(x, domain.ymin);
        const float beta = bottom_bc.beta.eval(x, domain.ymin);
        const float gamma = bottom_bc.gamma.eval(x, domain.ymin);
        const float denom = alpha + beta / domain.dy;
        if (fabs(denom) > 1e-6f) {
          value = (gamma + (beta / domain.dy) * grid[Index(i, 1, domain.nx)]) / denom;
        }
      }
    }

    if (j == domain.ny - 1) {
      if (top_bc.kind == 0) {
        value = top_bc.value.eval(x, domain.ymax);
      } else if (top_bc.kind == 1) {
        const float g = top_bc.value.eval(x, domain.ymax);
        value = grid[Index(i, domain.ny - 2, domain.nx)] + domain.dy * g;
      } else if (top_bc.kind == 2) {
        const float alpha = top_bc.alpha.eval(x, domain.ymax);
        const float beta = top_bc.beta.eval(x, domain.ymax);
        const float gamma = top_bc.gamma.eval(x, domain.ymax);
        const float denom = alpha + beta / domain.dy;
        if (fabs(denom) > 1e-6f) {
          value = (gamma + (beta / domain.dy) * grid[Index(i, domain.ny - 2, domain.nx)]) / denom;
        }
      }
    }

    next[idx] = value;
    delta[idx] = 0.0f;
    return;
  }

  const float u_left = grid[Index(i - 1, j, domain.nx)];
  const float u_right = grid[Index(i + 1, j, domain.nx)];
  const float u_down = grid[Index(i, j - 1, domain.nx)];
  const float u_up = grid[Index(i, j + 1, domain.nx)];

  const float rhs = -fval
    - (ax + cx) * u_right
    - (ax - cx) * u_left
    - (by + dyc) * u_up
    - (by - dyc) * u_down;

  const float updated = rhs / center;
  next[idx] = updated;
  delta[idx] = fabs(updated - grid[idx]);
}

kernel void jacobi_step_coeff(const device float* grid [[buffer(0)]],
                              device float* next [[buffer(1)]],
                              device float* delta [[buffer(2)]],
                              constant DomainInfo* domain_ptr [[buffer(3)]],
                              constant DeviceBC* left_ptr [[buffer(4)]],
                              constant DeviceBC* right_ptr [[buffer(5)]],
                              constant DeviceBC* bottom_ptr [[buffer(6)]],
                              constant DeviceBC* top_ptr [[buffer(7)]],
                              const device float* ax_field [[buffer(8)]],
                              const device float* by_field [[buffer(9)]],
                              const device float* cx_field [[buffer(10)]],
                              const device float* dyc_field [[buffer(11)]],
                              const device float* center_field [[buffer(12)]],
                              const device float* rhs_field [[buffer(13)]],
                              uint2 tid [[thread_position_in_grid]]) {
  const DomainInfo domain = *domain_ptr;
  const DeviceBC left_bc = *left_ptr;
  const DeviceBC right_bc = *right_ptr;
  const DeviceBC bottom_bc = *bottom_ptr;
  const DeviceBC top_bc = *top_ptr;

  const int i = static_cast<int>(tid.x);
  const int j = static_cast<int>(tid.y);
  if (i >= domain.nx || j >= domain.ny) {
    return;
  }

  const int idx = Index(i, j, domain.nx);
  if (i == 0 || j == 0 || i == domain.nx - 1 || j == domain.ny - 1) {
    const float x = domain.xmin + i * domain.dx;
    const float y = domain.ymin + j * domain.dy;
    float value = grid[idx];

    if (i == 0) {
      if (left_bc.kind == 0) {
        value = left_bc.value.eval(domain.xmin, y);
      } else if (left_bc.kind == 1) {
        const float g = left_bc.value.eval(domain.xmin, y);
        value = grid[Index(1, j, domain.nx)] - domain.dx * g;
      } else if (left_bc.kind == 2) {
        const float alpha = left_bc.alpha.eval(domain.xmin, y);
        const float beta = left_bc.beta.eval(domain.xmin, y);
        const float gamma = left_bc.gamma.eval(domain.xmin, y);
        const float denom = alpha + beta / domain.dx;
        if (fabs(denom) > 1e-6f) {
          value = (gamma + (beta / domain.dx) * grid[Index(1, j, domain.nx)]) / denom;
        }
      }
    }

    if (i == domain.nx - 1) {
      if (right_bc.kind == 0) {
        value = right_bc.value.eval(domain.xmax, y);
      } else if (right_bc.kind == 1) {
        const float g = right_bc.value.eval(domain.xmax, y);
        value = grid[Index(domain.nx - 2, j, domain.nx)] + domain.dx * g;
      } else if (right_bc.kind == 2) {
        const float alpha = right_bc.alpha.eval(domain.xmax, y);
        const float beta = right_bc.beta.eval(domain.xmax, y);
        const float gamma = right_bc.gamma.eval(domain.xmax, y);
        const float denom = alpha + beta / domain.dx;
        if (fabs(denom) > 1e-6f) {
          value = (gamma + (beta / domain.dx) * grid[Index(domain.nx - 2, j, domain.nx)]) / denom;
        }
      }
    }

    if (j == 0) {
      if (bottom_bc.kind == 0) {
        value = bottom_bc.value.eval(x, domain.ymin);
      } else if (bottom_bc.kind == 1) {
        const float g = bottom_bc.value.eval(x, domain.ymin);
        value = grid[Index(i, 1, domain.nx)] - domain.dy * g;
      } else if (bottom_bc.kind == 2) {
        const float alpha = bottom_bc.alpha.eval(x, domain.ymin);
        const float beta = bottom_bc.beta.eval(x, domain.ymin);
        const float gamma = bottom_bc.gamma.eval(x, domain.ymin);
        const float denom = alpha + beta / domain.dy;
        if (fabs(denom) > 1e-6f) {
          value = (gamma + (beta / domain.dy) * grid[Index(i, 1, domain.nx)]) / denom;
        }
      }
    }

    if (j == domain.ny - 1) {
      if (top_bc.kind == 0) {
        value = top_bc.value.eval(x, domain.ymax);
      } else if (top_bc.kind == 1) {
        const float g = top_bc.value.eval(x, domain.ymax);
        value = grid[Index(i, domain.ny - 2, domain.nx)] + domain.dy * g;
      } else if (top_bc.kind == 2) {
        const float alpha = top_bc.alpha.eval(x, domain.ymax);
        const float beta = top_bc.beta.eval(x, domain.ymax);
        const float gamma = top_bc.gamma.eval(x, domain.ymax);
        const float denom = alpha + beta / domain.dy;
        if (fabs(denom) > 1e-6f) {
          value = (gamma + (beta / domain.dy) * grid[Index(i, domain.ny - 2, domain.nx)]) / denom;
        }
      }
    }

    next[idx] = value;
    delta[idx] = 0.0f;
    return;
  }

  const float u_left = grid[Index(i - 1, j, domain.nx)];
  const float u_right = grid[Index(i + 1, j, domain.nx)];
  const float u_down = grid[Index(i, j - 1, domain.nx)];
  const float u_up = grid[Index(i, j + 1, domain.nx)];

  const float ax = ax_field[idx];
  const float by = by_field[idx];
  const float cx = cx_field[idx];
  const float dyc = dyc_field[idx];
  const float center = center_field[idx];
  const float fval = rhs_field[idx];

  const float rhs = -fval
    - (ax + cx) * u_right
    - (ax - cx) * u_left
    - (by + dyc) * u_up
    - (by - dyc) * u_down;

  const float updated = rhs / center;
  next[idx] = updated;
  delta[idx] = fabs(updated - grid[idx]);
}

kernel void rbgs_step(device float* grid [[buffer(0)]],
                      device float* delta [[buffer(1)]],
                      constant DomainInfo* domain_ptr [[buffer(2)]],
                      constant float& ax [[buffer(3)]],
                      constant float& by [[buffer(4)]],
                      constant float& cx [[buffer(5)]],
                      constant float& dyc [[buffer(6)]],
                      constant float& center [[buffer(7)]],
                      constant float& fval [[buffer(8)]],
                      constant float& omega [[buffer(9)]],
                      constant uint& parity [[buffer(10)]],
                      uint2 tid [[thread_position_in_grid]]) {
  const DomainInfo domain = *domain_ptr;
  const int i = static_cast<int>(tid.x);
  const int j = static_cast<int>(tid.y);
  if (i >= domain.nx || j >= domain.ny) {
    return;
  }
  const int idx = Index(i, j, domain.nx);
  if (i == 0 || j == 0 || i == domain.nx - 1 || j == domain.ny - 1) {
    delta[idx] = 0.0f;
    return;
  }
  if (((static_cast<uint>(i + j)) & 1u) != (parity & 1u)) {
    delta[idx] = 0.0f;
    return;
  }

  const float old_u = grid[idx];
  const float u_left = grid[Index(i - 1, j, domain.nx)];
  const float u_right = grid[Index(i + 1, j, domain.nx)];
  const float u_down = grid[Index(i, j - 1, domain.nx)];
  const float u_up = grid[Index(i, j + 1, domain.nx)];
  const float rhs = -fval
    - (ax + cx) * u_right
    - (ax - cx) * u_left
    - (by + dyc) * u_up
    - (by - dyc) * u_down;
  const float gs = rhs / center;
  const float updated = (1.0f - omega) * old_u + omega * gs;
  grid[idx] = updated;
  delta[idx] = fabs(updated - old_u);
}

kernel void rbgs_step_coeff(device float* grid [[buffer(0)]],
                            device float* delta [[buffer(1)]],
                            constant DomainInfo* domain_ptr [[buffer(2)]],
                            const device float* ax_field [[buffer(3)]],
                            const device float* by_field [[buffer(4)]],
                            const device float* cx_field [[buffer(5)]],
                            const device float* dyc_field [[buffer(6)]],
                            const device float* center_field [[buffer(7)]],
                            const device float* rhs_field [[buffer(8)]],
                            constant float& omega [[buffer(9)]],
                            constant uint& parity [[buffer(10)]],
                            uint2 tid [[thread_position_in_grid]]) {
  const DomainInfo domain = *domain_ptr;
  const int i = static_cast<int>(tid.x);
  const int j = static_cast<int>(tid.y);
  if (i >= domain.nx || j >= domain.ny) {
    return;
  }
  const int idx = Index(i, j, domain.nx);
  if (i == 0 || j == 0 || i == domain.nx - 1 || j == domain.ny - 1) {
    delta[idx] = 0.0f;
    return;
  }
  if (((static_cast<uint>(i + j)) & 1u) != (parity & 1u)) {
    delta[idx] = 0.0f;
    return;
  }

  const float old_u = grid[idx];
  const float u_left = grid[Index(i - 1, j, domain.nx)];
  const float u_right = grid[Index(i + 1, j, domain.nx)];
  const float u_down = grid[Index(i, j - 1, domain.nx)];
  const float u_up = grid[Index(i, j + 1, domain.nx)];

  const float ax = ax_field[idx];
  const float by = by_field[idx];
  const float cx = cx_field[idx];
  const float dyc = dyc_field[idx];
  const float center = center_field[idx];
  const float fval = rhs_field[idx];

  const float rhs = -fval
    - (ax + cx) * u_right
    - (ax - cx) * u_left
    - (by + dyc) * u_up
    - (by - dyc) * u_down;
  const float gs = rhs / center;
  const float updated = (1.0f - omega) * old_u + omega * gs;
  grid[idx] = updated;
  delta[idx] = fabs(updated - old_u);
}

kernel void ax_apply(const device float* x [[buffer(0)]],
                     device float* y [[buffer(1)]],
                     constant DomainInfo* domain_ptr [[buffer(2)]],
                     constant float& ax [[buffer(3)]],
                     constant float& by [[buffer(4)]],
                     constant float& cx [[buffer(5)]],
                     constant float& dyc [[buffer(6)]],
                     constant float& center [[buffer(7)]],
                     uint2 tid [[thread_position_in_grid]]) {
  const DomainInfo domain = *domain_ptr;
  const int i = static_cast<int>(tid.x);
  const int j = static_cast<int>(tid.y);
  if (i >= domain.nx || j >= domain.ny) {
    return;
  }
  const int idx = Index(i, j, domain.nx);
  if (i == 0 || j == 0 || i == domain.nx - 1 || j == domain.ny - 1) {
    y[idx] = x[idx];
    return;
  }
  const float u_c = x[idx];
  const float u_left = x[Index(i - 1, j, domain.nx)];
  const float u_right = x[Index(i + 1, j, domain.nx)];
  const float u_down = x[Index(i, j - 1, domain.nx)];
  const float u_up = x[Index(i, j + 1, domain.nx)];
  y[idx] = center * u_c
    + (ax + cx) * u_right
    + (ax - cx) * u_left
    + (by + dyc) * u_up
    + (by - dyc) * u_down;
}

kernel void vec_set(device float* v [[buffer(0)]],
                    constant float& value [[buffer(1)]],
                    constant uint& count [[buffer(2)]],
                    uint gid [[thread_position_in_grid]]) {
  if (gid >= count) {
    return;
  }
  v[gid] = value;
}

kernel void vec_copy(const device float* src [[buffer(0)]],
                     device float* dst [[buffer(1)]],
                     constant uint& count [[buffer(2)]],
                     uint gid [[thread_position_in_grid]]) {
  if (gid >= count) {
    return;
  }
  dst[gid] = src[gid];
}

kernel void vec_axpy(device float* y [[buffer(0)]],
                     const device float* x [[buffer(1)]],
                     constant float& a [[buffer(2)]],
                     constant uint& count [[buffer(3)]],
                     uint gid [[thread_position_in_grid]]) {
  if (gid >= count) {
    return;
  }
  y[gid] += a * x[gid];
}

kernel void vec_scale(device float* v [[buffer(0)]],
                      constant float& a [[buffer(1)]],
                      constant uint& count [[buffer(2)]],
                      uint gid [[thread_position_in_grid]]) {
  if (gid >= count) {
    return;
  }
  v[gid] *= a;
}

kernel void vec_lincomb2(device float* out [[buffer(0)]],
                         const device float* a [[buffer(1)]],
                         const device float* b [[buffer(2)]],
                         constant float& alpha [[buffer(3)]],
                         constant float& beta [[buffer(4)]],
                         constant uint& count [[buffer(5)]],
                         uint gid [[thread_position_in_grid]]) {
  if (gid >= count) {
    return;
  }
  out[gid] = alpha * a[gid] + beta * b[gid];
}

kernel void reduce_dot(const device float* a [[buffer(0)]],
                       const device float* b [[buffer(1)]],
                       device float* output [[buffer(2)]],
                       constant uint& count [[buffer(3)]],
                       uint tid [[thread_index_in_threadgroup]],
                       uint gid [[thread_position_in_grid]],
                       uint tg [[threadgroup_position_in_grid]]) {
  threadgroup float shared[REDUCE_GROUP_SIZE];
  float value = 0.0f;
  if (gid < count) {
    value = a[gid] * b[gid];
  }
  shared[tid] = value;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint stride = REDUCE_GROUP_SIZE / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      shared[tid] += shared[tid + stride];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (tid == 0) {
    output[tg] = shared[0];
  }
}

kernel void reduce_sum(const device float* input [[buffer(0)]],
                       device float* output [[buffer(1)]],
                       constant uint& count [[buffer(2)]],
                       uint tid [[thread_index_in_threadgroup]],
                       uint gid [[thread_position_in_grid]],
                       uint tg [[threadgroup_position_in_grid]]) {
  threadgroup float shared[REDUCE_GROUP_SIZE];
  float value = 0.0f;
  if (gid < count) {
    value = input[gid];
  }
  shared[tid] = value;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint stride = REDUCE_GROUP_SIZE / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      shared[tid] += shared[tid + stride];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (tid == 0) {
    output[tg] = shared[0];
  }
}

kernel void reduce_max(const device float* input [[buffer(0)]],
                       device float* output [[buffer(1)]],
                       constant uint& count [[buffer(2)]],
                       uint tid [[thread_index_in_threadgroup]],
                       uint gid [[thread_position_in_grid]],
                       uint tg [[threadgroup_position_in_grid]]) {
  threadgroup float shared[REDUCE_GROUP_SIZE];
  float value = 0.0f;
  if (gid < count) {
    value = input[gid];
  }
  shared[tid] = value;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint stride = REDUCE_GROUP_SIZE / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      shared[tid] = max(shared[tid], shared[tid + stride]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  if (tid == 0) {
    output[tg] = shared[0];
  }
}

kernel void rbgs_rhs(device float* x [[buffer(0)]],
                     const device float* b [[buffer(1)]],
                     constant DomainInfo* domain_ptr [[buffer(2)]],
                     constant float& ax [[buffer(3)]],
                     constant float& by [[buffer(4)]],
                     constant float& center [[buffer(5)]],
                     constant float& omega [[buffer(6)]],
                     constant uint& parity [[buffer(7)]],
                     uint2 tid [[thread_position_in_grid]]) {
  const DomainInfo domain = *domain_ptr;
  const int i = static_cast<int>(tid.x);
  const int j = static_cast<int>(tid.y);
  if (i <= 0 || j <= 0 || i >= domain.nx - 1 || j >= domain.ny - 1) {
    return;
  }
  if (((static_cast<uint>(i + j)) & 1u) != (parity & 1u)) {
    return;
  }
  const int idx = Index(i, j, domain.nx);
  const float old_u = x[idx];
  const float u_left = x[Index(i - 1, j, domain.nx)];
  const float u_right = x[Index(i + 1, j, domain.nx)];
  const float u_down = x[Index(i, j - 1, domain.nx)];
  const float u_up = x[Index(i, j + 1, domain.nx)];
  const float rhs = b[idx] - ax * (u_left + u_right) - by * (u_down + u_up);
  const float gs = rhs / center;
  x[idx] = (1.0f - omega) * old_u + omega * gs;
}

kernel void residual_poisson(const device float* b [[buffer(0)]],
                             const device float* x [[buffer(1)]],
                             device float* r [[buffer(2)]],
                             constant DomainInfo* domain_ptr [[buffer(3)]],
                             constant float& ax [[buffer(4)]],
                             constant float& by [[buffer(5)]],
                             constant float& center [[buffer(6)]],
                             uint2 tid [[thread_position_in_grid]]) {
  const DomainInfo domain = *domain_ptr;
  const int i = static_cast<int>(tid.x);
  const int j = static_cast<int>(tid.y);
  if (i >= domain.nx || j >= domain.ny) {
    return;
  }
  const int idx = Index(i, j, domain.nx);
  if (i == 0 || j == 0 || i == domain.nx - 1 || j == domain.ny - 1) {
    r[idx] = 0.0f;
    return;
  }
  const float u_c = x[idx];
  const float u_left = x[Index(i - 1, j, domain.nx)];
  const float u_right = x[Index(i + 1, j, domain.nx)];
  const float u_down = x[Index(i, j - 1, domain.nx)];
  const float u_up = x[Index(i, j + 1, domain.nx)];
  const float Au = center * u_c + ax * (u_left + u_right) + by * (u_down + u_up);
  r[idx] = b[idx] - Au;
}

kernel void restrict_full_weighting(const device float* fine [[buffer(0)]],
                                    device float* coarse [[buffer(1)]],
                                    constant uint& nx_f [[buffer(2)]],
                                    constant uint& ny_f [[buffer(3)]],
                                    constant uint& nx_c [[buffer(4)]],
                                    constant uint& ny_c [[buffer(5)]],
                                    uint2 tid [[thread_position_in_grid]]) {
  const int ic = static_cast<int>(tid.x);
  const int jc = static_cast<int>(tid.y);
  if (ic >= static_cast<int>(nx_c) || jc >= static_cast<int>(ny_c)) {
    return;
  }
  const int idx_c = Index(ic, jc, static_cast<int>(nx_c));
  if (ic == 0 || jc == 0 || ic == static_cast<int>(nx_c) - 1 || jc == static_cast<int>(ny_c) - 1) {
    coarse[idx_c] = 0.0f;
    return;
  }
  const int ifx = 2 * ic;
  const int jfy = 2 * jc;
  const int nxf = static_cast<int>(nx_f);
  const float c = fine[Index(ifx, jfy, nxf)] * 0.25f;
  const float e = (fine[Index(ifx - 1, jfy, nxf)] +
                   fine[Index(ifx + 1, jfy, nxf)] +
                   fine[Index(ifx, jfy - 1, nxf)] +
                   fine[Index(ifx, jfy + 1, nxf)]) * 0.125f;
  const float d = (fine[Index(ifx - 1, jfy - 1, nxf)] +
                   fine[Index(ifx + 1, jfy - 1, nxf)] +
                   fine[Index(ifx - 1, jfy + 1, nxf)] +
                   fine[Index(ifx + 1, jfy + 1, nxf)]) * 0.0625f;
  coarse[idx_c] = c + e + d;
}

kernel void prolong_bilinear_add(const device float* coarse [[buffer(0)]],
                                 device float* fine [[buffer(1)]],
                                 constant uint& nx_c [[buffer(2)]],
                                 constant uint& ny_c [[buffer(3)]],
                                 constant uint& nx_f [[buffer(4)]],
                                 constant uint& ny_f [[buffer(5)]],
                                 uint2 tid [[thread_position_in_grid]]) {
  const int i = static_cast<int>(tid.x);
  const int j = static_cast<int>(tid.y);
  if (i >= static_cast<int>(nx_f) || j >= static_cast<int>(ny_f)) {
    return;
  }
  if (i == 0 || j == 0 || i == static_cast<int>(nx_f) - 1 || j == static_cast<int>(ny_f) - 1) {
    return;
  }
  const int ic = i / 2;
  const int jc = j / 2;
  const int idx_f = Index(i, j, static_cast<int>(nx_f));
  const int nxc = static_cast<int>(nx_c);
  if ((i % 2 == 0) && (j % 2 == 0)) {
    fine[idx_f] += coarse[Index(ic, jc, nxc)];
  } else if ((i % 2 == 1) && (j % 2 == 0)) {
    fine[idx_f] += 0.5f * (coarse[Index(ic, jc, nxc)] + coarse[Index(ic + 1, jc, nxc)]);
  } else if ((i % 2 == 0) && (j % 2 == 1)) {
    fine[idx_f] += 0.5f * (coarse[Index(ic, jc, nxc)] + coarse[Index(ic, jc + 1, nxc)]);
  } else {
    fine[idx_f] += 0.25f * (coarse[Index(ic, jc, nxc)] +
                            coarse[Index(ic + 1, jc, nxc)] +
                            coarse[Index(ic, jc + 1, nxc)] +
                            coarse[Index(ic + 1, jc + 1, nxc)]);
  }
}

// ============================================================================
// Time-dependent PDE kernels
// ============================================================================

// First-order time stepping: u_t = -L[u] / ut_coeff
// Updates: u_next = u + dt * u_t
kernel void time_step_first_order(const device float* grid [[buffer(0)]],
                                   device float* next [[buffer(1)]],
                                   constant DomainInfo* domain_ptr [[buffer(2)]],
                                   constant DeviceBC* left_ptr [[buffer(3)]],
                                   constant DeviceBC* right_ptr [[buffer(4)]],
                                   constant DeviceBC* bottom_ptr [[buffer(5)]],
                                   constant DeviceBC* top_ptr [[buffer(6)]],
                                   constant float& ax [[buffer(7)]],
                                   constant float& by [[buffer(8)]],
                                   constant float& cx [[buffer(9)]],
                                   constant float& dyc [[buffer(10)]],
                                   constant float& fval [[buffer(11)]],
                                   constant float& ut_coeff [[buffer(12)]],
                                   constant float& dt [[buffer(13)]],
                                   uint2 tid [[thread_position_in_grid]]) {
  const DomainInfo domain = *domain_ptr;
  const DeviceBC left_bc = *left_ptr;
  const DeviceBC right_bc = *right_ptr;
  const DeviceBC bottom_bc = *bottom_ptr;
  const DeviceBC top_bc = *top_ptr;

  const int i = static_cast<int>(tid.x);
  const int j = static_cast<int>(tid.y);
  if (i >= domain.nx || j >= domain.ny) {
    return;
  }

  const int idx = Index(i, j, domain.nx);
  const float x = domain.xmin + i * domain.dx;
  const float y = domain.ymin + j * domain.dy;

  // Handle boundaries
  if (i == 0 || j == 0 || i == domain.nx - 1 || j == domain.ny - 1) {
    float value = grid[idx];
    if (i == 0 && left_bc.kind == 0) {
      value = left_bc.value.eval(domain.xmin, y);
    } else if (i == domain.nx - 1 && right_bc.kind == 0) {
      value = right_bc.value.eval(domain.xmax, y);
    } else if (j == 0 && bottom_bc.kind == 0) {
      value = bottom_bc.value.eval(x, domain.ymin);
    } else if (j == domain.ny - 1 && top_bc.kind == 0) {
      value = top_bc.value.eval(x, domain.ymax);
    }
    next[idx] = value;
    return;
  }

  // Compute spatial operator L[u] = a*u_xx + b*u_yy + c*u_x + d*u_y + f
  // Using central differences: u_xx ≈ (u_right - 2u + u_left)/dx²
  const float u = grid[idx];
  const float u_left = grid[Index(i - 1, j, domain.nx)];
  const float u_right = grid[Index(i + 1, j, domain.nx)];
  const float u_down = grid[Index(i, j - 1, domain.nx)];
  const float u_up = grid[Index(i, j + 1, domain.nx)];

  // Include center term: -2*ax*u - 2*by*u for the Laplacian
  const float laplacian = (ax + cx) * u_right + (ax - cx) * u_left +
                          (by + dyc) * u_up + (by - dyc) * u_down
                          - 2.0f * ax * u - 2.0f * by * u;
  const float L_u = laplacian + fval;

  // u_t = -L[u] / ut_coeff
  const float u_t = -L_u / ut_coeff;

  // Forward Euler: u_next = u + dt * u_t
  next[idx] = u + dt * u_t;
}

// Second-order time stepping: u_tt = -(L[u] + ut_coeff * velocity) / utt_coeff
// Updates velocity and position
kernel void time_step_second_order(const device float* grid [[buffer(0)]],
                                    device float* next [[buffer(1)]],
                                    device float* velocity [[buffer(2)]],
                                    constant DomainInfo* domain_ptr [[buffer(3)]],
                                    constant DeviceBC* left_ptr [[buffer(4)]],
                                    constant DeviceBC* right_ptr [[buffer(5)]],
                                    constant DeviceBC* bottom_ptr [[buffer(6)]],
                                    constant DeviceBC* top_ptr [[buffer(7)]],
                                    constant float& ax [[buffer(8)]],
                                    constant float& by [[buffer(9)]],
                                    constant float& cx [[buffer(10)]],
                                    constant float& dyc [[buffer(11)]],
                                    constant float& fval [[buffer(12)]],
                                    constant float& ut_coeff [[buffer(13)]],
                                    constant float& utt_coeff [[buffer(14)]],
                                    constant float& dt [[buffer(15)]],
                                    uint2 tid [[thread_position_in_grid]]) {
  const DomainInfo domain = *domain_ptr;
  const DeviceBC left_bc = *left_ptr;
  const DeviceBC right_bc = *right_ptr;
  const DeviceBC bottom_bc = *bottom_ptr;
  const DeviceBC top_bc = *top_ptr;

  const int i = static_cast<int>(tid.x);
  const int j = static_cast<int>(tid.y);
  if (i >= domain.nx || j >= domain.ny) {
    return;
  }

  const int idx = Index(i, j, domain.nx);
  const float x = domain.xmin + i * domain.dx;
  const float y = domain.ymin + j * domain.dy;

  // Handle boundaries
  if (i == 0 || j == 0 || i == domain.nx - 1 || j == domain.ny - 1) {
    float value = grid[idx];
    if (i == 0 && left_bc.kind == 0) {
      value = left_bc.value.eval(domain.xmin, y);
    } else if (i == domain.nx - 1 && right_bc.kind == 0) {
      value = right_bc.value.eval(domain.xmax, y);
    } else if (j == 0 && bottom_bc.kind == 0) {
      value = bottom_bc.value.eval(x, domain.ymin);
    } else if (j == domain.ny - 1 && top_bc.kind == 0) {
      value = top_bc.value.eval(x, domain.ymax);
    }
    next[idx] = value;
    velocity[idx] = 0.0f;
    return;
  }

  // Compute spatial operator L[u] = a*u_xx + b*u_yy + c*u_x + d*u_y + f
  // Using central differences: u_xx ≈ (u_right - 2u + u_left)/dx²
  const float u = grid[idx];
  const float u_left = grid[Index(i - 1, j, domain.nx)];
  const float u_right = grid[Index(i + 1, j, domain.nx)];
  const float u_down = grid[Index(i, j - 1, domain.nx)];
  const float u_up = grid[Index(i, j + 1, domain.nx)];

  // Include center term: -2*ax*u - 2*by*u for the Laplacian
  const float laplacian = (ax + cx) * u_right + (ax - cx) * u_left +
                          (by + dyc) * u_up + (by - dyc) * u_down
                          - 2.0f * ax * u - 2.0f * by * u;
  const float L_u = laplacian + fval;

  // acceleration = -(L[u] + ut_coeff * velocity) / utt_coeff
  const float v = velocity[idx];
  const float accel = -(L_u + ut_coeff * v) / utt_coeff;

  // Update velocity and position
  const float v_new = v + dt * accel;
  velocity[idx] = v_new;
  next[idx] = u + dt * v_new;
}
