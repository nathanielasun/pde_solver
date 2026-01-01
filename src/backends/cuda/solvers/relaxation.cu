#include "relaxation.h"
#include "../utils/cuda_utils.h"
#include "expression_eval.h"
#include "coefficient_evaluator.h"
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>

namespace {

constexpr int kBlockX = 16;
constexpr int kBlockY = 16;

__device__ int Index(int i, int j, int nx) {
  return j * nx + i;
}

__device__ double EvalNonlinear(const DeviceNonlinearTerm* terms, int num_terms, double u) {
  double value = 0.0;
  for (int i = 0; i < num_terms; ++i) {
    const DeviceNonlinearTerm& term = terms[i];
    switch (term.kind) {
      case 0:  // Power
        value += term.coeff * pow(u, static_cast<double>(term.power));
        break;
      case 1:  // Sin
        value += term.coeff * sin(u);
        break;
      case 2:  // Cos
        value += term.coeff * cos(u);
        break;
      case 3:  // Exp
        value += term.coeff * exp(u);
        break;
      case 4:  // Abs
        value += term.coeff * fabs(u);
        break;
    }
  }
  return value;
}

__global__ void JacobiKernel(const double* grid, double* next, int nx, int ny,
                             double ax_const, double by_const, double cx_const, double dyc_const,
                             double center_const, const double* f_rhs,
                             const double* ax_field, const double* by_field,
                             const double* cx_field, const double* dyc_field,
                             const double* center_field,
                             const DeviceNonlinearTerm* nonlinear_terms, int num_nonlinear,
                             double* block_max) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
  const int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
  if (i >= nx - 1 || j >= ny - 1) {
    return;
  }

  const double u_left = grid[Index(i - 1, j, nx)];
  const double u_right = grid[Index(i + 1, j, nx)];
  const double u_down = grid[Index(i, j - 1, nx)];
  const double u_up = grid[Index(i, j + 1, nx)];

  const int idx = Index(i, j, nx);
  const double ax = ax_field ? ax_field[idx] : ax_const;
  const double by = by_field ? by_field[idx] : by_const;
  const double cx = cx_field ? cx_field[idx] : cx_const;
  const double dyc = dyc_field ? dyc_field[idx] : dyc_const;
  const double center = center_field ? center_field[idx] : center_const;
  const double u = grid[idx];
  const double f = f_rhs ? f_rhs[idx] : 0.0;
  const double nonlinear = (num_nonlinear > 0 && nonlinear_terms) ? EvalNonlinear(nonlinear_terms, num_nonlinear, u) : 0.0;

  const double rhs = -f - nonlinear
    - (ax + cx) * u_right
    - (ax - cx) * u_left
    - (by + dyc) * u_up
    - (by - dyc) * u_down;

  const double updated = rhs / center;
  next[idx] = updated;

  const double delta = fabs(updated - grid[idx]);
  extern __shared__ double sdata[];
  const int tid = threadIdx.y * blockDim.x + threadIdx.x;
  sdata[tid] = delta;
  __syncthreads();

  for (int stride = (blockDim.x * blockDim.y) / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      sdata[tid] = fmax(sdata[tid], sdata[tid + stride]);
    }
    __syncthreads();
  }

  if (tid == 0) {
    const int block_idx = blockIdx.y * gridDim.x + blockIdx.x;
    block_max[block_idx] = sdata[0];
  }
}

__global__ void RbGaussSeidelKernel(double* grid, int nx, int ny,
                                   double ax_const, double by_const, double cx_const, double dyc_const,
                                   double center_const, const double* f_rhs,
                                   const double* ax_field, const double* by_field,
                                   const double* cx_field, const double* dyc_field,
                                   const double* center_field,
                                   const DeviceNonlinearTerm* nonlinear_terms, int num_nonlinear,
                                   double omega, int parity,
                                   double* block_max) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
  const int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
  if (i >= nx - 1 || j >= ny - 1) {
    return;
  }
  if (((i + j) & 1) != (parity & 1)) {
    return;
  }
  const int idx = Index(i, j, nx);
  const double ax = ax_field ? ax_field[idx] : ax_const;
  const double by = by_field ? by_field[idx] : by_const;
  const double cx = cx_field ? cx_field[idx] : cx_const;
  const double dyc = dyc_field ? dyc_field[idx] : dyc_const;
  const double center = center_field ? center_field[idx] : center_const;
  const double old_u = grid[idx];
  const double u_left = grid[Index(i - 1, j, nx)];
  const double u_right = grid[Index(i + 1, j, nx)];
  const double u_down = grid[Index(i, j - 1, nx)];
  const double u_up = grid[Index(i, j + 1, nx)];

  const double f = f_rhs ? f_rhs[idx] : 0.0;
  const double nonlinear = (num_nonlinear > 0 && nonlinear_terms) ? EvalNonlinear(nonlinear_terms, num_nonlinear, old_u) : 0.0;

  const double rhs = -f - nonlinear
    - (ax + cx) * u_right
    - (ax - cx) * u_left
    - (by + dyc) * u_up
    - (by - dyc) * u_down;

  const double gs_update = rhs / center;
  const double updated = (1.0 - omega) * old_u + omega * gs_update;
  grid[idx] = updated;

  const double delta = fabs(updated - old_u);
  extern __shared__ double sdata[];
  const int tid = threadIdx.y * blockDim.x + threadIdx.x;
  sdata[tid] = delta;
  __syncthreads();

  for (int stride = (blockDim.x * blockDim.y) / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      sdata[tid] = fmax(sdata[tid], sdata[tid + stride]);
    }
    __syncthreads();
  }
  if (tid == 0) {
    const int block_idx = blockIdx.y * gridDim.x + blockIdx.x;
    block_max[block_idx] = sdata[0];
  }
}

} // namespace

SolveOutput CudaSolveRelaxation(const SolveInput& input) {
    const Domain& d = input.domain;
    std::string error;

    const int nx = d.nx;
    const int ny = d.ny;
    const int total = nx * ny;
    const double dx = (d.xmax - d.xmin) / static_cast<double>(nx - 1);
    const double dy = (d.ymax - d.ymin) / static_cast<double>(ny - 1);

    const double a = input.pde.a;
    const double b = input.pde.b;
    const double c = input.pde.c;
    const double dcoef = input.pde.d;
    const double e = input.pde.e;
    const double f_const = input.pde.f;

    CoefficientEvaluator coeff_eval = BuildCoefficientEvaluator(input.pde);
    if (!coeff_eval.ok) {
        return {coeff_eval.error, {}};
    }
    const bool has_var_coeff = coeff_eval.has_variable;

    std::vector<double> rhs_values(total, f_const);
    if (!input.pde.rhs_latex.empty()) {
        ExpressionEvaluator evaluator = ExpressionEvaluator::ParseLatex(input.pde.rhs_latex);
        if (!evaluator.ok()) {
        return {"invalid RHS expression: " + evaluator.error(), {}};
        }
        for (int j = 0; j < ny; ++j) {
            const double y = d.ymin + j * dy;
            for (int i = 0; i < nx; ++i) {
                const double x = d.xmin + i * dx;
                rhs_values[Index(i, j, nx)] = evaluator.Eval(x, y, 0.0, 0.0);
            }
        }
    }

    std::vector<DeviceNonlinearTerm> device_nonlinear;
    for (const auto& term : input.nonlinear) {
        device_nonlinear.push_back({static_cast<int>(term.kind), term.coeff, term.power});
    }

    const double ax_const = a / (dx * dx);
    const double by_const = b / (dy * dy);
    const double cx_const = c / (2.0 * dx);
    const double dyc_const = dcoef / (2.0 * dy);
    const double center_const = -2.0 * ax_const - 2.0 * by_const + e;

    std::vector<double> ax_values;
    std::vector<double> by_values;
    std::vector<double> cx_values;
    std::vector<double> dyc_values;
    std::vector<double> center_values;
    if (has_var_coeff) {
        ax_values.resize(total, ax_const);
        by_values.resize(total, by_const);
        cx_values.resize(total, cx_const);
        dyc_values.resize(total, dyc_const);
        center_values.resize(total, center_const);
        auto IndexHost = [nx](int i, int j) { return j * nx + i; };
        for (int j = 0; j < ny; ++j) {
            const double y = d.ymin + j * dy;
            for (int i = 0; i < nx; ++i) {
                const double x = d.xmin + i * dx;
                const double a_val = EvalCoefficient(coeff_eval.a, a, x, y, 0.0, 0.0);
                const double b_val = EvalCoefficient(coeff_eval.b, b, x, y, 0.0, 0.0);
                const double c_val = EvalCoefficient(coeff_eval.c, c, x, y, 0.0, 0.0);
                const double d_val = EvalCoefficient(coeff_eval.d, dcoef, x, y, 0.0, 0.0);
                const double e_val = EvalCoefficient(coeff_eval.e, e, x, y, 0.0, 0.0);
                const double ax = a_val / (dx * dx);
                const double by = b_val / (dy * dy);
                const double cx = c_val / (2.0 * dx);
                const double dyc = d_val / (2.0 * dy);
                const double center = -2.0 * ax - 2.0 * by + e_val;
                if (fabs(center) < 1e-12) {
                    return {"degenerate PDE center coefficient", {}};
                }
                const int idx = IndexHost(i, j);
                ax_values[idx] = ax;
                by_values[idx] = by;
                cx_values[idx] = cx;
                dyc_values[idx] = dyc;
                center_values[idx] = center;
            }
        }
    } else {
        if (fabs(center_const) < 1e-12) {
            return {"degenerate PDE center coefficient", {}};
        }
    }

    DeviceBoundarySet bc;
    bc.left = ToDeviceBC(input.bc.left);
    bc.right = ToDeviceBC(input.bc.right);
    bc.bottom = ToDeviceBC(input.bc.bottom);
    bc.top = ToDeviceBC(input.bc.top);

    double* d_grid = nullptr;
    double* d_next = nullptr;
    double* d_block_max = nullptr;
    double* d_rhs = nullptr;
    double* d_ax = nullptr;
    double* d_by = nullptr;
    double* d_cx = nullptr;
    double* d_dyc = nullptr;
    double* d_center = nullptr;
    DeviceNonlinearTerm* d_nonlinear = nullptr;

    auto cleanup = [&]() {
        if(d_grid) cudaFree(d_grid);
        if(d_next) cudaFree(d_next);
        if(d_block_max) cudaFree(d_block_max);
        if(d_rhs) cudaFree(d_rhs);
        if(d_ax) cudaFree(d_ax);
        if(d_by) cudaFree(d_by);
        if(d_cx) cudaFree(d_cx);
        if(d_dyc) cudaFree(d_dyc);
        if(d_center) cudaFree(d_center);
        if(d_nonlinear) cudaFree(d_nonlinear);
    };

    if (!CudaOk(cudaMalloc(&d_grid, total * sizeof(double)), &error, "cudaMalloc grid")) { cleanup(); return {error, {}}; }
    if (!CudaOk(cudaMalloc(&d_next, total * sizeof(double)), &error, "cudaMalloc next")) { cleanup(); return {error, {}}; }
    if (!CudaOk(cudaMalloc(&d_rhs, total * sizeof(double)), &error, "cudaMalloc rhs")) { cleanup(); return {error, {}}; }
    if (has_var_coeff) {
        if (!CudaOk(cudaMalloc(&d_ax, total * sizeof(double)), &error, "cudaMalloc ax")) { cleanup(); return {error, {}}; }
        if (!CudaOk(cudaMalloc(&d_by, total * sizeof(double)), &error, "cudaMalloc by")) { cleanup(); return {error, {}}; }
        if (!CudaOk(cudaMalloc(&d_cx, total * sizeof(double)), &error, "cudaMalloc cx")) { cleanup(); return {error, {}}; }
        if (!CudaOk(cudaMalloc(&d_dyc, total * sizeof(double)), &error, "cudaMalloc dyc")) { cleanup(); return {error, {}}; }
        if (!CudaOk(cudaMalloc(&d_center, total * sizeof(double)), &error, "cudaMalloc center")) { cleanup(); return {error, {}}; }
    }
    if (device_nonlinear.size() > 0) {
        if (!CudaOk(cudaMalloc(&d_nonlinear, device_nonlinear.size() * sizeof(DeviceNonlinearTerm)), &error, "cudaMalloc nonlinear")) { cleanup(); return {error, {}}; }
        if (!CudaOk(cudaMemcpy(d_nonlinear, device_nonlinear.data(), device_nonlinear.size() * sizeof(DeviceNonlinearTerm), cudaMemcpyHostToDevice), &error, "copy nonlinear")) { cleanup(); return {error, {}}; }
    }

    if (!CudaOk(cudaMemset(d_grid, 0, total * sizeof(double)), &error, "cudaMemset grid")) { cleanup(); return {error, {}}; }
    if (!CudaOk(cudaMemset(d_next, 0, total * sizeof(double)), &error, "cudaMemset next")) { cleanup(); return {error, {}}; }
    if (!CudaOk(cudaMemcpy(d_rhs, rhs_values.data(), total * sizeof(double), cudaMemcpyHostToDevice), &error, "copy rhs")) { cleanup(); return {error, {}}; }
    if (has_var_coeff) {
        if (!CudaOk(cudaMemcpy(d_ax, ax_values.data(), total * sizeof(double), cudaMemcpyHostToDevice), &error, "copy ax")) { cleanup(); return {error, {}}; }
        if (!CudaOk(cudaMemcpy(d_by, by_values.data(), total * sizeof(double), cudaMemcpyHostToDevice), &error, "copy by")) { cleanup(); return {error, {}}; }
        if (!CudaOk(cudaMemcpy(d_cx, cx_values.data(), total * sizeof(double), cudaMemcpyHostToDevice), &error, "copy cx")) { cleanup(); return {error, {}}; }
        if (!CudaOk(cudaMemcpy(d_dyc, dyc_values.data(), total * sizeof(double), cudaMemcpyHostToDevice), &error, "copy dyc")) { cleanup(); return {error, {}}; }
        if (!CudaOk(cudaMemcpy(d_center, center_values.data(), total * sizeof(double), cudaMemcpyHostToDevice), &error, "copy center")) { cleanup(); return {error, {}}; }
    }

    if (!ApplyBoundaryKernels(d_grid, nx, ny, d, bc, &error)) { cleanup(); return {error, {}}; }

    dim3 block(kBlockX, kBlockY);
    dim3 grid((nx - 2 + block.x - 1) / block.x, (ny - 2 + block.y - 1) / block.y);
    const size_t block_count = static_cast<size_t>(grid.x * grid.y);
    if (!CudaOk(cudaMalloc(&d_block_max, block_count * sizeof(double)), &error, "cudaMalloc block max")) { cleanup(); return {error, {}}; }

    std::vector<double> block_max(block_count, 0.0);

    for (int iter = 0; iter < input.solver.max_iter; ++iter) {
        if (!ApplyBoundaryKernels(d_grid, nx, ny, d, bc, &error)) { cleanup(); return {error, {}}; }

        const size_t shared_bytes = block.x * block.y * sizeof(double);
        if (input.solver.method == SolveMethod::Jacobi) {
            JacobiKernel<<<grid, block, shared_bytes>>>(d_grid, d_next, nx, ny,
                ax_const, by_const, cx_const, dyc_const, center_const,
                d_rhs, d_ax, d_by, d_cx, d_dyc, d_center,
                d_nonlinear, static_cast<int>(device_nonlinear.size()), d_block_max);
        } else { // SOR or GaussSeidel
            const double omega = input.solver.method == SolveMethod::SOR ? input.solver.sor_omega : 1.0;
            if (omega <= 0.0 || omega >= 2.0) { cleanup(); return {"SOR omega must be in (0,2)", {}}; }
            
            RbGaussSeidelKernel<<<grid, block, shared_bytes>>>(d_grid, nx, ny,
                ax_const, by_const, cx_const, dyc_const, center_const,
                d_rhs, d_ax, d_by, d_cx, d_dyc, d_center,
                d_nonlinear, static_cast<int>(device_nonlinear.size()), omega, 0, d_block_max);
            if (!CudaOk(cudaGetLastError(), &error, "rbgs red kernel failed")) { cleanup(); return {error, {}}; }
            
            RbGaussSeidelKernel<<<grid, block, shared_bytes>>>(d_grid, nx, ny,
                ax_const, by_const, cx_const, dyc_const, center_const,
                d_rhs, d_ax, d_by, d_cx, d_dyc, d_center,
                d_nonlinear, static_cast<int>(device_nonlinear.size()), omega, 1, d_block_max);
        }

        if (!CudaOk(cudaGetLastError(), &error, "solver kernel failed")) { cleanup(); return {error, {}}; }
        if (!CudaOk(cudaMemcpy(block_max.data(), d_block_max, block_count * sizeof(double), cudaMemcpyDeviceToHost), &error, "copy block max")) { cleanup(); return {error, {}}; }

        double max_delta = 0.0;
        for (double value : block_max) max_delta = std::max(max_delta, value);

        if (input.solver.method == SolveMethod::Jacobi) std::swap(d_grid, d_next);
        if (max_delta < input.solver.tol) break;
    }

    if (!ApplyBoundaryKernels(d_grid, nx, ny, d, bc, &error)) { cleanup(); return {error, {}}; }

    std::vector<double> grid_host(total, 0.0);
    if (!CudaOk(cudaMemcpy(grid_host.data(), d_grid, total * sizeof(double), cudaMemcpyDeviceToHost), &error, "copy grid")) { cleanup(); return {error, {}}; }
    
    cleanup();
    return {"", grid_host};
}
