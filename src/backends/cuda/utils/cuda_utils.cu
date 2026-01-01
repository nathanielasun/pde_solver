#include "cuda_utils.h"

namespace {
__device__ int Index(int i, int j, int nx) {
  return j * nx + i;
}
}

// Error handling
std::string CudaErrorToString(cudaError_t err) {
  return std::string(cudaGetErrorString(err));
}

bool CudaOk(cudaError_t err, std::string* error, const char* message) {
  if (err == cudaSuccess) {
    return true;
  }
  if (error) {
    *error = std::string(message) + ": " + CudaErrorToString(err);
  }
  return false;
}

// Conversion functions
DeviceExpr ToDeviceExpr(const BoundaryCondition::Expression& expr) {
  DeviceExpr out;
  out.constant = expr.constant;
  out.x = expr.x;
  out.y = expr.y;
  out.z = expr.z;
  return out;
}

DeviceBC ToDeviceBC(const BoundaryCondition& bc) {
  DeviceBC out;
  out.kind = static_cast<int>(bc.kind);
  out.value = ToDeviceExpr(bc.value);
  out.alpha = ToDeviceExpr(bc.alpha);
  out.beta = ToDeviceExpr(bc.beta);
  out.gamma = ToDeviceExpr(bc.gamma);
  return out;
}


// Kernels
__global__ void ApplyDirichletVertical(double* grid, int nx, int ny,
                                       double xmin, double xmax,
                                       double ymin, double dy,
                                       DeviceBC left, DeviceBC right) {
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j >= ny) {
    return;
  }
  const double y = ymin + j * dy;
  if (left.kind == 0) {
    grid[Index(0, j, nx)] = left.value.Eval(xmin, y);
  }
  if (right.kind == 0) {
    grid[Index(nx - 1, j, nx)] = right.value.Eval(xmax, y);
  }
}

__global__ void ApplyDirichletHorizontal(double* grid, int nx, int ny,
                                         double xmin, double xmax,
                                         double ymin, double dy,
                                         DeviceBC bottom, DeviceBC top) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= nx) {
    return;
  }
  const double x = xmin + i * ((xmax - xmin) / static_cast<double>(nx - 1));
  if (bottom.kind == 0) {
    grid[Index(i, 0, nx)] = bottom.value.Eval(x, ymin);
  }
  if (top.kind == 0) {
    grid[Index(i, ny - 1, nx)] = top.value.Eval(x, ymin + (ny - 1) * dy);
  }
}

__global__ void ApplyLeftBoundary(double* grid, int nx, int ny,
                                  double xmin, double ymin, double dy, double dx,
                                  DeviceBC left) {
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j >= ny) {
    return;
  }
  if (left.kind == 1) {
    const double y = ymin + j * dy;
    const double g = left.value.Eval(xmin, y);
    grid[Index(0, j, nx)] = grid[Index(1, j, nx)] - dx * g;
  } else if (left.kind == 2) {
    const double y = ymin + j * dy;
    const double alpha = left.alpha.Eval(xmin, y);
    const double beta = left.beta.Eval(xmin, y);
    const double gamma = left.gamma.Eval(xmin, y);
    const double denom = alpha + beta / dx;
    if (fabs(denom) > 1e-12) {
      grid[Index(0, j, nx)] = (gamma + (beta / dx) * grid[Index(1, j, nx)]) / denom;
    }
  }
}

__global__ void ApplyRightBoundary(double* grid, int nx, int ny,
                                   double xmax, double ymin, double dy, double dx,
                                   DeviceBC right) {
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j >= ny) {
    return;
  }
  if (right.kind == 1) {
    const double y = ymin + j * dy;
    const double g = right.value.Eval(xmax, y);
    grid[Index(nx - 1, j, nx)] = grid[Index(nx - 2, j, nx)] + dx * g;
  } else if (right.kind == 2) {
    const double y = ymin + j * dy;
    const double alpha = right.alpha.Eval(xmax, y);
    const double beta = right.beta.Eval(xmax, y);
    const double gamma = right.gamma.Eval(xmax, y);
    const double denom = alpha + beta / dx;
    if (fabs(denom) > 1e-12) {
      grid[Index(nx - 1, j, nx)] = (gamma + (beta / dx) * grid[Index(nx - 2, j, nx)]) / denom;
    }
  }
}

__global__ void ApplyBottomBoundary(double* grid, int nx, int ny,
                                    double xmin, double ymin, double dx, double dy,
                                    DeviceBC bottom) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= nx) {
    return;
  }
  if (bottom.kind == 1) {
    const double x = xmin + i * dx;
    const double g = bottom.value.Eval(x, ymin);
    grid[Index(i, 0, nx)] = grid[Index(i, 1, nx)] - dy * g;
  } else if (bottom.kind == 2) {
    const double x = xmin + i * dx;
    const double alpha = bottom.alpha.Eval(x, ymin);
    const double beta = bottom.beta.Eval(x, ymin);
    const double gamma = bottom.gamma.Eval(x, ymin);
    const double denom = alpha + beta / dy;
    if (fabs(denom) > 1e-12) {
      grid[Index(i, 0, nx)] = (gamma + (beta / dy) * grid[Index(i, 1, nx)]) / denom;
    }
  }
}

__global__ void ApplyTopBoundary(double* grid, int nx, int ny,
                                 double xmin, double ymax, double dx, double dy,
                                 DeviceBC top) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= nx) {
    return;
  }
  if (top.kind == 1) {
    const double x = xmin + i * dx;
    const double g = top.value.Eval(x, ymax);
    grid[Index(i, ny - 1, nx)] = grid[Index(i, ny - 2, nx)] + dy * g;
  } else if (top.kind == 2) {
    const double x = xmin + i * dx;
    const double alpha = top.alpha.Eval(x, ymax);
    const double beta = top.beta.Eval(x, ymax);
    const double gamma = top.gamma.Eval(x, ymax);
    const double denom = alpha + beta / dy;
    if (fabs(denom) > 1e-12) {
      grid[Index(i, ny - 1, nx)] = (gamma + (beta / dy) * grid[Index(i, ny - 2, nx)]) / denom;
    }
  }
}

bool ApplyBoundaryKernels(double* d_grid, int nx, int ny,
                          const Domain& domain,
                          const DeviceBoundarySet& bc,
                          std::string* error) {
  const double dx = (domain.xmax - domain.xmin) / static_cast<double>(nx - 1);
  const double dy = (domain.ymax - domain.ymin) / static_cast<double>(ny - 1);

  const int block1 = 256;
  const int grid_y = (ny + block1 - 1) / block1;
  const int grid_x = (nx + block1 - 1) / block1;

  ApplyLeftBoundary<<<grid_y, block1>>>(d_grid, nx, ny, domain.xmin, domain.ymin, dy, dx, bc.left);
  ApplyRightBoundary<<<grid_y, block1>>>(d_grid, nx, ny, domain.xmax, domain.ymin, dy, dx, bc.right);
  ApplyBottomBoundary<<<grid_x, block1>>>(d_grid, nx, ny, domain.xmin, domain.ymin, dx, dy, bc.bottom);
  ApplyTopBoundary<<<grid_x, block1>>>(d_grid, nx, ny, domain.xmin, domain.ymax, dx, dy, bc.top);

  ApplyDirichletVertical<<<grid_y, block1>>>(d_grid, nx, ny, domain.xmin, domain.xmax,
                                             domain.ymin, dy, bc.left, bc.right);
  ApplyDirichletHorizontal<<<grid_x, block1>>>(d_grid, nx, ny, domain.xmin, domain.xmax,
                                               domain.ymin, dy, bc.bottom, bc.top);

  return CudaOk(cudaGetLastError(), error, "boundary kernel failed");
}


// Vector operation kernels
__global__ void VecCopyKernel(const double* src, double* dst, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) return;
  dst[idx] = src[idx];
}

__global__ void VecSetKernel(double* dst, double value, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) return;
  dst[idx] = value;
}

__global__ void ScaleKernel(double* v, double a, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) return;
  v[idx] *= a;
}

__global__ void AxpyKernel(double* y, const double* x, double a, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) return;
  y[idx] += a * x[idx];
}

__global__ void LinComb2Kernel(double* out, const double* a, double alpha, const double* b, double beta, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) return;
  out[idx] = alpha * a[idx] + beta * b[idx];
}

__global__ void ReduceDotKernel(const double* a, const double* b, double* block_sums, int n) {
  extern __shared__ double sdata[];
  const int tid = threadIdx.x;
  int idx = blockIdx.x * blockDim.x + tid;
  double sum = 0.0;
  const int stride = blockDim.x * gridDim.x;
  while (idx < n) {
    sum += a[idx] * b[idx];
    idx += stride;
  }
  sdata[tid] = sum;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
  }
  if (tid == 0) {
    block_sums[blockIdx.x] = sdata[0];
  }
}

// Krylov buffer management
bool AllocKrylovBuffers(int n, CudaKrylovBuffers* buf, std::string* error) {
  if (!buf) return false;
  buf->n = n;
  const int block = 256;
  buf->partial_count = std::max(1, (n + block - 1) / block);
  if (!CudaOk(cudaMalloc(&buf->d_partials, sizeof(double) * buf->partial_count),
              error, "cudaMalloc partials")) {
    buf->d_partials = nullptr;
    return false;
  }
  buf->h_partials.reserve(buf->partial_count);
  return true;
}

void FreeKrylovBuffers(CudaKrylovBuffers* buf) {
  if (!buf) return;
  if (buf->d_partials) cudaFree(buf->d_partials);
  buf->d_partials = nullptr;
  buf->partial_count = 0;
  buf->n = 0;
  buf->h_partials.clear();
}

// BLAS-like operations
namespace {
double HostSum(const std::vector<double>& v) {
  double sum = 0.0;
  for (double x : v) {
    sum += x;
  }
  return sum;
}
}

bool DeviceDot(double* d_a, double* d_b, int n,
               CudaKrylovBuffers* buf,
               double* out, std::string* error) {
  if (!out || !buf || !buf->d_partials) return false;
  
  const int block = 256;
  ReduceDotKernel<<<buf->partial_count, block, sizeof(double) * block>>>(d_a, d_b, buf->d_partials, n);
  if (!CudaOk(cudaGetLastError(), error, "reduce dot kernel failed")) return false;

  buf->h_partials.assign(buf->partial_count, 0.0);
  if (!CudaOk(cudaMemcpy(buf->h_partials.data(), buf->d_partials,
                         sizeof(double) * buf->partial_count,
                         cudaMemcpyDeviceToHost),
              error, "copy dot partials")) {
    return false;
  }
  *out = HostSum(buf->h_partials);
  return true;
}

bool DeviceNorm2(double* d_v, int n, CudaKrylovBuffers* buf, double* out, std::string* error) {
  if (!out) return false;
  double dot = 0.0;
  if (!DeviceDot(d_v, d_v, n, buf, &dot, error)) {
    return false;
  }
  *out = std::sqrt(std::max(0.0, dot));
  return true;
}

void DeviceVecSet(double* d_v, double value, int n) {
    const int block = 256;
    const int grid = (n + block - 1) / block;
    VecSetKernel<<<grid, block>>>(d_v, value, n);
}

void DeviceVecCopy(const double* d_src, double* d_dst, int n) {
    const int block = 256;
    const int grid = (n + block - 1) / block;
    VecCopyKernel<<<grid, block>>>(d_src, d_dst, n);
}

void DeviceAxpy(double* d_y, const double* d_x, double a, int n) {
    const int block = 256;
    const int grid = (n + block - 1) / block;
    AxpyKernel<<<grid, block>>>(d_y, d_x, a, n);
}

void DeviceScale(double* d_v, double a, int n) {
    const int block = 256;
    const int grid = (n + block - 1) / block;
    ScaleKernel<<<grid, block>>>(d_v, a, n);
}

void DeviceLinComb2(double* d_out, const double* d_a, double alpha, const double* d_b, double beta, int n) {
    const int block = 256;
    const int grid = (n + block - 1) / block;
    LinComb2Kernel<<<grid, block>>>(d_out, d_a, alpha, d_b, beta, n);
}
