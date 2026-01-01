#include "multigrid.h"
#include "../utils/cuda_utils.h"
#include "expression_eval.h"
#include <optional>
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>

namespace {
bool IsAllDirichlet(const BoundarySet& bc) {
  return bc.left.kind == BCKind::Dirichlet &&
         bc.right.kind == BCKind::Dirichlet &&
         bc.bottom.kind == BCKind::Dirichlet &&
         bc.top.kind == BCKind::Dirichlet;
}

double EvalExprHost(const BoundaryCondition::Expression& expr, double x, double y) {
  return expr.constant + expr.x * x + expr.y * y + expr.z * 0.0;
}

bool BuildDirichletRhs(const Domain& d, const BoundarySet& bc, double f,
                       const ExpressionEvaluator* rhs_eval,
                       std::vector<double>* b, std::string* error) {
  if (!b) {
    return false;
  }
  if (!IsAllDirichlet(bc)) {
    if (error) {
      *error = "Dirichlet boundaries required for this solver.";
    }
    return false;
  }
  const int nx = d.nx;
  const int ny = d.ny;
  b->assign(nx * ny, 0.0);
  const double dx = (d.xmax - d.xmin) / (nx - 1);
  const double dy = (d.ymax - d.ymin) / (ny - 1);

  auto Index = [nx](int i, int j) { return j * nx + i; };

  for (int j = 0; j < ny; ++j) {
    const double y = d.ymin + j * dy;
    for (int i = 0; i < nx; ++i) {
      const double x = d.xmin + i * dx;
      const int idx = Index(i, j);
      if (i == 0 || i == nx - 1 || j == 0 || j == ny - 1) {
        if (i == 0) (*b)[idx] = EvalExprHost(bc.left.value, d.xmin, y);
        else if (i == nx - 1) (*b)[idx] = EvalExprHost(bc.right.value, d.xmax, y);
        else if (j == 0) (*b)[idx] = EvalExprHost(bc.bottom.value, x, d.ymin);
        else (*b)[idx] = EvalExprHost(bc.top.value, x, d.ymax);
      } else {
        const double f_val = rhs_eval ? rhs_eval->Eval(x, y, 0.0, 0.0) : f;
        (*b)[idx] = -f_val;
      }
    }
  }
  return true;
}
} // namespace

namespace {

__device__ int Index(int i, int j, int nx) {
  return j * nx + i;
}

__global__ void ResidualAxKernel(const double* b, const double* x, double* r,
                                int nx, int ny,
                                double ax, double by, double center) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i >= nx || j >= ny) return;

  const int idx = Index(i, j, nx);
  if (i == 0 || j == 0 || i == nx - 1 || j == ny - 1) {
    r[idx] = 0.0;
    return;
  }
  const double u_c = x[idx];
  const double u_left = x[Index(i - 1, j, nx)];
  const double u_right = x[Index(i + 1, j, nx)];
  const double u_down = x[Index(i, j - 1, nx)];
  const double u_up = x[Index(i, j + 1, nx)];
  const double Au = center * u_c + ax * (u_left + u_right) + by * (u_down + u_up);
  r[idx] = b[idx] - Au;
}

__global__ void RbGsRhsKernel(double* x, const double* b, int nx, int ny,
                             double ax, double by, double center,
                             double omega, int parity) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i <= 0 || j <= 0 || i >= nx - 1 || j >= ny - 1) return;
  if (((i + j) & 1) != (parity & 1)) return;
  
  const int idx = Index(i, j, nx);
  const double old_u = x[idx];
  const double u_left = x[Index(i - 1, j, nx)];
  const double u_right = x[Index(i + 1, j, nx)];
  const double u_down = x[Index(i, j - 1, nx)];
  const double u_up = x[Index(i, j + 1, nx)];
  const double rhs = b[idx] - ax * (u_left + u_right) - by * (u_down + u_up);
  const double gs = rhs / center;
  x[idx] = (1.0 - omega) * old_u + omega * gs;
}

__global__ void RestrictFullWeightingKernel(const double* fine, int nx_f, int ny_f,
                                           double* coarse, int nx_c, int ny_c) {
  int ic = blockIdx.x * blockDim.x + threadIdx.x;
  int jc = blockIdx.y * blockDim.y + threadIdx.y;
  if (ic >= nx_c || jc >= ny_c) return;

  const int idx_c = Index(ic, jc, nx_c);
  if (ic == 0 || jc == 0 || ic == nx_c - 1 || jc == ny_c - 1) {
    coarse[idx_c] = 0.0;
    return;
  }
  const int ifx = 2 * ic;
  const int jfy = 2 * jc;
  const double c = fine[Index(ifx, jfy, nx_f)] * 0.25;
  const double e = (fine[Index(ifx - 1, jfy, nx_f)] + fine[Index(ifx + 1, jfy, nx_f)] +
                    fine[Index(ifx, jfy - 1, nx_f)] + fine[Index(ifx, jfy + 1, nx_f)]) * 0.125;
  const double d = (fine[Index(ifx - 1, jfy - 1, nx_f)] + fine[Index(ifx + 1, jfy - 1, nx_f)] +
                    fine[Index(ifx - 1, jfy + 1, nx_f)] + fine[Index(ifx + 1, jfy + 1, nx_f)]) * 0.0625;
  coarse[idx_c] = c + e + d;
}

__global__ void ProlongBilinearAddKernel(const double* coarse, int nx_c, int ny_c,
                                        double* fine, int nx_f, int ny_f) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i >= nx_f || j >= ny_f) return;
  if (i == 0 || j == 0 || i == nx_f - 1 || j == ny_f - 1) return;

  const int ic = i / 2;
  const int jc = j / 2;
  const int idx_f = Index(i, j, nx_f);
  if ((i % 2 == 0) && (j % 2 == 0)) {
    fine[idx_f] += coarse[Index(ic, jc, nx_c)];
  } else if ((i % 2 == 1) && (j % 2 == 0)) {
    fine[idx_f] += 0.5 * (coarse[Index(ic, jc, nx_c)] + coarse[Index(ic + 1, jc, nx_c)]);
  } else if ((i % 2 == 0) && (j % 2 == 1)) {
    fine[idx_f] += 0.5 * (coarse[Index(ic, jc, nx_c)] + coarse[Index(ic, jc + 1, nx_c)]);
  } else {
    fine[idx_f] += 0.25 * (coarse[Index(ic, jc, nx_c)] + coarse[Index(ic + 1, jc, nx_c)] +
                           coarse[Index(ic, jc + 1, nx_c)] + coarse[Index(ic + 1, jc + 1, nx_c)]);
  }
}

bool CanCoarsenOdd(int n) { return n >= 5 && (n % 2 == 1); }

struct MgLevelCuda {
    Domain d;
    double ax = 0.0, by = 0.0, center = 0.0;
    double* d_u = nullptr;
    double* d_b = nullptr;
    double* d_r = nullptr;
};

void FreeMgLevels(std::vector<MgLevelCuda>& levels) {
    for (auto& lvl : levels) {
        if (lvl.d_u) cudaFree(lvl.d_u);
        if (lvl.d_b) cudaFree(lvl.d_b);
        if (lvl.d_r) cudaFree(lvl.d_r);
    }
    levels.clear();
}

bool AllocMgLevels(const Domain& fine_d, double a, double b, double e, const std::vector<double>& b0_host, int max_levels, std::vector<MgLevelCuda>& levels, std::string* error) {
    Domain cur = fine_d;
    for (int L = 0; L < max_levels; ++L) {
        MgLevelCuda lvl;
        lvl.d = cur;
        const double dx = (cur.xmax - cur.xmin) / (cur.nx - 1);
        const double dy = (cur.ymax - cur.ymin) / (cur.ny - 1);
        lvl.ax = a / (dx * dx);
        lvl.by = b / (dy * dy);
        lvl.center = -2.0 * lvl.ax - 2.0 * lvl.by + e;
        const int n = cur.nx * cur.ny;
        
        if (!CudaOk(cudaMalloc(&lvl.d_u, sizeof(double)*n), error, "mg alloc") ||
            !CudaOk(cudaMalloc(&lvl.d_b, sizeof(double)*n), error, "mg alloc") ||
            !CudaOk(cudaMalloc(&lvl.d_r, sizeof(double)*n), error, "mg alloc")) {
            FreeMgLevels(levels);
            return false;
        }

        DeviceVecSet(lvl.d_u, 0.0, n);
        if (L == 0) {
            if (!CudaOk(cudaMemcpy(lvl.d_b, b0_host.data(), sizeof(double)*n, cudaMemcpyHostToDevice), error, "mg copy b0")) {
                FreeMgLevels(levels);
                return false;
            }
        } else {
            DeviceVecSet(lvl.d_b, 0.0, n);
        }
        levels.push_back(lvl);

        if (!CanCoarsenOdd(cur.nx) || !CanCoarsenOdd(cur.ny) || cur.nx <= 5 || cur.ny <= 5) break;
        
        cur.nx = (cur.nx + 1) / 2;
        cur.ny = (cur.ny + 1) / 2;
    }
    return true;
}


bool MultigridVcycleCuda(const Domain& d, double a, double bcoef, double e, const std::vector<double>& b_host, std::vector<double>* x_host, int cycles, int pre_smooth, int post_smooth, int coarse_iters, int max_levels, double tol, std::string* error) {
    if (!x_host) return false;
    if (!CanCoarsenOdd(d.nx) || !CanCoarsenOdd(d.ny)) {
        *error = "grid must have odd dimensions for this multigrid implementation";
        return false;
    }

    std::vector<MgLevelCuda> levels;
    if (!AllocMgLevels(d, a, bcoef, e, b_host, max_levels, levels, error)) return false;
    
    auto cleanup = [&]() { FreeMgLevels(levels); };
    dim3 block2(16, 16);

    for (int cyc = 0; cyc < cycles; ++cyc) {
        // Down cycle
        for (size_t ell = 0; ell < levels.size() - 1; ++ell) {
            MgLevelCuda& fine = levels[ell];
            MgLevelCuda& coarse = levels[ell + 1];
            dim3 grid_fine((fine.d.nx + block2.x - 1) / block2.x, (fine.d.ny + block2.y - 1) / block2.y);
            
            for (int s = 0; s < pre_smooth; ++s) {
                RbGsRhsKernel<<<grid_fine, block2>>>(fine.d_u, fine.d_b, fine.d.nx, fine.d.ny, fine.ax, fine.by, fine.center, 1.0, 0);
                RbGsRhsKernel<<<grid_fine, block2>>>(fine.d_u, fine.d_b, fine.d.nx, fine.d.ny, fine.ax, fine.by, fine.center, 1.0, 1);
            }
            ResidualAxKernel<<<grid_fine, block2>>>(fine.d_b, fine.d_u, fine.d_r, fine.d.nx, fine.d.ny, fine.ax, fine.by, fine.center);
            
            dim3 grid_coarse((coarse.d.nx + block2.x - 1) / block2.x, (coarse.d.ny + block2.y - 1) / block2.y);
            RestrictFullWeightingKernel<<<grid_coarse, block2>>>(fine.d_r, fine.d.nx, fine.d.ny, coarse.d_b, coarse.d.nx, coarse.d.ny);
            DeviceVecSet(coarse.d_u, 0.0, coarse.d.nx * coarse.d.ny);
        }

        // Coarsest solve
        MgLevelCuda& coarsest = levels.back();
        dim3 grid_coarsest((coarsest.d.nx + block2.x - 1) / block2.x, (coarsest.d.ny + block2.y - 1) / block2.y);
        for (int s = 0; s < coarse_iters; ++s) {
            RbGsRhsKernel<<<grid_coarsest, block2>>>(coarsest.d_u, coarsest.d_b, coarsest.d.nx, coarsest.d.ny, coarsest.ax, coarsest.by, coarsest.center, 1.0, 0);
            RbGsRhsKernel<<<grid_coarsest, block2>>>(coarsest.d_u, coarsest.d_b, coarsest.d.nx, coarsest.d.ny, coarsest.ax, coarsest.by, coarsest.center, 1.0, 1);
        }

        // Up cycle
        for (int ell = levels.size() - 2; ell >= 0; --ell) {
            MgLevelCuda& fine = levels[ell];
            MgLevelCuda& coarse = levels[ell + 1];
            dim3 grid_fine((fine.d.nx + block2.x - 1) / block2.x, (fine.d.ny + block2.y - 1) / block2.y);
            ProlongBilinearAddKernel<<<grid_fine, block2>>>(coarse.d_u, coarse.d.nx, coarse.d.ny, fine.d_u, fine.d.nx, fine.d.ny);
            for (int s = 0; s < post_smooth; ++s) {
                RbGsRhsKernel<<<grid_fine, block2>>>(fine.d_u, fine.d_b, fine.d.nx, fine.d.ny, fine.ax, fine.by, fine.center, 1.0, 0);
                RbGsRhsKernel<<<grid_fine, block2>>>(fine.d_u, fine.d_b, fine.d.nx, fine.d.ny, fine.ax, fine.by, fine.center, 1.0, 1);
            }
        }

        if (!CudaOk(cudaGetLastError(), error, "multigrid kernels failed")) { cleanup(); return false; }
    }

    const int n0 = d.nx * d.ny;
    x_host->assign(n0, 0.0);
    if (!CudaOk(cudaMemcpy(x_host->data(), levels.front().d_u, sizeof(double) * n0, cudaMemcpyDeviceToHost), error, "copy mg solution")) {
        cleanup();
        return false;
    }
    
    cleanup();
    return true;
}


} // namespace


SolveOutput CudaSolveMultigrid(const SolveInput& input) {
    const Domain& d = input.domain;
    if (fabs(input.pde.c) > 1e-12 || fabs(input.pde.d) > 1e-12) return {"CUDA multigrid currently requires no convection terms", {}};
    if (HasImplicitShape(input) || !input.integrals.empty() || !input.nonlinear.empty()) return {"CUDA Multigrid requires CPU fallback: unsupported features", {}};
    if (!IsAllDirichlet(input.bc)) return {"CUDA Multigrid currently require Dirichlet boundaries on all sides", {}};

    std::optional<ExpressionEvaluator> rhs_eval;
    if (!input.pde.rhs_latex.empty()) {
        ExpressionEvaluator eval = ExpressionEvaluator::ParseLatex(input.pde.rhs_latex);
        if (!eval.ok()) {
            return {"invalid RHS expression: " + eval.error(), {}};
        }
        rhs_eval.emplace(std::move(eval));
    }

    std::vector<double> bvec;
    std::string rhs_error;
    if (!BuildDirichletRhs(d, input.bc, input.pde.f,
                           rhs_eval ? &(*rhs_eval) : nullptr, &bvec, &rhs_error)) {
        return {rhs_error.empty() ? "failed to build rhs" : rhs_error, {}};
    }

    std::vector<double> xvec;
    std::string solve_error;
    bool ok = MultigridVcycleCuda(d, input.pde.a, input.pde.b, input.pde.e, bvec, &xvec,
                           std::max(1, input.solver.max_iter),
                           input.solver.mg_pre_smooth,
                           input.solver.mg_post_smooth,
                           input.solver.mg_coarse_iters,
                           input.solver.mg_max_levels,
                           input.solver.tol,
                           &solve_error);

    if (!ok) {
        return {solve_error.empty() ? "CUDA multigrid solver failed" : solve_error, {}};
    }
    return {"", xvec};
}
