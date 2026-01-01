#include "krylov.h"
#include "../utils/cuda_utils.h"
#include "expression_eval.h"
#include <optional>
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>

namespace {

__global__ void AxKernel(const double* x, double* y, int nx, int ny,
                         double ax, double by, double cx, double dyc, double center) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= nx || j >= ny) return;

    auto Index = [nx](int i, int j) { return j * nx + i; };

    const int idx = Index(i, j);
    if (i == 0 || j == 0 || i == nx - 1 || j == ny - 1) {
        y[idx] = x[idx];
        return;
    }
    const double u_c = x[idx];
    const double u_left = x[Index(i - 1, j)];
    const double u_right = x[Index(i + 1, j)];
    const double u_down = x[Index(i, j - 1)];
    const double u_up = x[Index(i, j + 1)];
    y[idx] = center * u_c
        + (ax + cx) * u_right
        + (ax - cx) * u_left
        + (by + dyc) * u_up
        + (by - dyc) * u_down;
}

struct CudaLinOp2D {
    int nx = 0, ny = 0;
    double ax = 0.0, by = 0.0, cx = 0.0, dyc = 0.0, center = 0.0;

    bool apply(double* d_x, double* d_y, std::string* error) const {
        dim3 block(16, 16);
        dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);
        AxKernel<<<grid, block>>>(d_x, d_y, nx, ny, ax, by, cx, dyc, center);
        return CudaOk(cudaGetLastError(), error, "Ax kernel failed");
    }
};

bool IsAllDirichlet(const BoundarySet& bc) {
    return bc.left.kind == BCKind::Dirichlet && bc.right.kind == BCKind::Dirichlet &&
           bc.bottom.kind == BCKind::Dirichlet && bc.top.kind == BCKind::Dirichlet;
}

double EvalExprHost(const BoundaryCondition::Expression& expr, double x, double y) {
    return expr.constant + expr.x * x + expr.y * y + expr.z * 0.0;
}

bool BuildDirichletRhs(const Domain& d, const BoundarySet& bc, double f,
                       const ExpressionEvaluator* rhs_eval,
                       std::vector<double>* b, std::string* error) {
    if (!b) return false;
    if (!IsAllDirichlet(bc)) {
        if (error) *error = "Dirichlet boundaries required for this solver.";
        return false;
    }
    const int nx = d.nx, ny = d.ny;
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


// SOLVER IMPLEMENTATIONS (CG, BiCGStab, GMRES)
// ===========================================

__global__ void BiCGStabPUpdateKernel(double* p, const double* r, const double* v, double beta, double omega, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) return;
  p[idx] = r[idx] + beta * (p[idx] - omega * v[idx]);
}

__global__ void XpayKernel(double* y, const double* x, double a, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) return;
  y[idx] = x[idx] + a * y[idx];
}


bool KrylovBiCGStabCuda(const CudaLinOp2D& op, const std::vector<double>& b_host, std::vector<double>* x_host, int max_iter, double tol, std::string* error) {
    const int n = op.nx * op.ny;
    if (n <= 0) return false;

    CudaKrylovBuffers red;
    if (!AllocKrylovBuffers(n, &red, error)) return false;

    double *d_x = nullptr, *d_b = nullptr, *d_r = nullptr, *d_rhat = nullptr;
    double *d_p = nullptr, *d_v = nullptr, *d_s = nullptr, *d_t = nullptr, *d_Ax = nullptr;
    
    auto cleanup = [&]() {
        cudaFree(d_x); cudaFree(d_b); cudaFree(d_r); cudaFree(d_rhat);
        cudaFree(d_p); cudaFree(d_v); cudaFree(d_s); cudaFree(d_t); cudaFree(d_Ax);
        FreeKrylovBuffers(&red);
    };

    if (!CudaOk(cudaMalloc(&d_x, sizeof(double)*n), error, "cudaMalloc") || !CudaOk(cudaMalloc(&d_b, sizeof(double)*n), error, "cudaMalloc") ||
        !CudaOk(cudaMalloc(&d_r, sizeof(double)*n), error, "cudaMalloc") || !CudaOk(cudaMalloc(&d_rhat, sizeof(double)*n), error, "cudaMalloc") ||
        !CudaOk(cudaMalloc(&d_p, sizeof(double)*n), error, "cudaMalloc") || !CudaOk(cudaMalloc(&d_v, sizeof(double)*n), error, "cudaMalloc") ||
        !CudaOk(cudaMalloc(&d_s, sizeof(double)*n), error, "cudaMalloc") || !CudaOk(cudaMalloc(&d_t, sizeof(double)*n), error, "cudaMalloc") ||
        !CudaOk(cudaMalloc(&d_Ax, sizeof(double)*n), error, "cudaMalloc")) {
        cleanup(); return false;
    }

    if (!CudaOk(cudaMemcpy(d_b, b_host.data(), sizeof(double)*n, cudaMemcpyHostToDevice), error, "copy b")) { cleanup(); return false; }
    
    DeviceVecSet(d_x, 0.0, n);
    DeviceVecSet(d_p, 0.0, n);
    DeviceVecSet(d_v, 0.0, n);
    
    if (!op.apply(d_x, d_Ax, error)) { cleanup(); return false; }
    DeviceLinComb2(d_r, d_b, 1.0, d_Ax, -1.0, n);
    DeviceVecCopy(d_r, d_rhat, n);
    
    double resid = 0.0;
    if (!DeviceNorm2(d_r, n, &red, &resid, error)) { cleanup(); return false; }
    if (resid < tol) { x_host->assign(n, 0.0); cleanup(); return true; }

    double rho_prev = 1.0, alpha = 1.0, omega = 1.0;

    for (int iter = 0; iter < max_iter; ++iter) {
        double rho = 0.0;
        if (!DeviceDot(d_rhat, d_r, n, &red, &rho, error)) { cleanup(); return false; }
        if (std::abs(rho) < 1e-30) { *error = "bicgstab breakdown (rho)"; cleanup(); return false; }
        
        if (iter == 0) {
            DeviceVecCopy(d_r, d_p, n);
        } else {
            const double beta = (rho / rho_prev) * (alpha / omega);
            const int block = 256, grid = (n + block - 1) / block;
            BiCGStabPUpdateKernel<<<grid, block>>>(d_p, d_r, d_v, beta, omega, n);
        }
        
        if (!op.apply(d_p, d_v, error)) { cleanup(); return false; }
        double denom = 0.0;
        if (!DeviceDot(d_rhat, d_v, n, &red, &denom, error)) { cleanup(); return false; }
        if (std::abs(denom) < 1e-30) { *error = "bicgstab breakdown (alpha denom)"; cleanup(); return false; }
        alpha = rho / denom;

        DeviceLinComb2(d_s, d_r, 1.0, d_v, -alpha, n);
        
        double s_norm = 0.0;
        if (!DeviceNorm2(d_s, n, &red, &s_norm, error)) { cleanup(); return false; }
        if (s_norm < tol) { DeviceAxpy(d_x, d_p, alpha, n); break; }

        if (!op.apply(d_s, d_t, error)) { cleanup(); return false; }
        double tt = 0.0, ts = 0.0;
        if (!DeviceDot(d_t, d_t, n, &red, &tt, error) || !DeviceDot(d_t, d_s, n, &red, &ts, error)) { cleanup(); return false; }
        if (std::abs(tt) < 1e-30) { *error = "bicgstab breakdown (omega denom)"; cleanup(); return false; }
        omega = ts / tt;
        if (std::abs(omega) < 1e-30) { *error = "bicgstab breakdown (omega)"; cleanup(); return false; }

        DeviceAxpy(d_x, d_p, alpha, n);
        DeviceAxpy(d_x, d_s, omega, n);
        DeviceLinComb2(d_r, d_s, 1.0, d_t, -omega, n);

        if (!DeviceNorm2(d_r, n, &red, &resid, error)) { cleanup(); return false; }
        if (resid < tol) break;
        rho_prev = rho;
    }

    x_host->assign(n, 0.0);
    if (!CudaOk(cudaMemcpy(x_host->data(), d_x, sizeof(double)*n, cudaMemcpyDeviceToHost), error, "copy x")) { cleanup(); return false; }
    
    cleanup();
    return true;
}


bool KrylovCGCuda(const CudaLinOp2D& op, const std::vector<double>& b_host, std::vector<double>* x_host, int max_iter, double tol, std::string* error) {
    const int n = op.nx * op.ny;
    CudaKrylovBuffers red;
    if (!AllocKrylovBuffers(n, &red, error)) return false;

    double *d_x = nullptr, *d_b = nullptr, *d_r = nullptr, *d_p = nullptr, *d_Ap = nullptr;
    auto cleanup = [&]() { cudaFree(d_x); cudaFree(d_b); cudaFree(d_r); cudaFree(d_p); cudaFree(d_Ap); FreeKrylovBuffers(&red); };
    
    if (!CudaOk(cudaMalloc(&d_x, sizeof(double)*n), error, "CG malloc") || !CudaOk(cudaMalloc(&d_b, sizeof(double)*n), error, "CG malloc") ||
        !CudaOk(cudaMalloc(&d_r, sizeof(double)*n), error, "CG malloc") || !CudaOk(cudaMalloc(&d_p, sizeof(double)*n), error, "CG malloc") ||
        !CudaOk(cudaMalloc(&d_Ap, sizeof(double)*n), error, "CG malloc")) {
        cleanup(); return false;
    }

    if (!CudaOk(cudaMemcpy(d_b, b_host.data(), sizeof(double)*n, cudaMemcpyHostToDevice), error, "copy b")) { cleanup(); return false; }
    
    DeviceVecSet(d_x, 0.0, n);
    DeviceVecCopy(d_b, d_r, n);
    DeviceVecCopy(d_r, d_p, n);
    
    double rr = 0.0;
    if (!DeviceDot(d_r, d_r, n, &red, &rr, error)) { cleanup(); return false; }
    if (std::sqrt(rr) < tol) { x_host->assign(n, 0.0); cleanup(); return true; }

    for (int iter = 0; iter < max_iter; ++iter) {
        if (!op.apply(d_p, d_Ap, error)) { cleanup(); return false; }
        double pAp = 0.0;
        if (!DeviceDot(d_p, d_Ap, n, &red, &pAp, error)) { cleanup(); return false; }
        if (std::abs(pAp) < 1e-30) { *error = "cg breakdown"; cleanup(); return false; }
        
        const double alpha = rr / pAp;
        DeviceAxpy(d_x, d_p, alpha, n);
        DeviceAxpy(d_r, d_Ap, -alpha, n);
        
        double rr_new = 0.0;
        if (!DeviceDot(d_r, d_r, n, &red, &rr_new, error)) { cleanup(); return false; }
        if (std::sqrt(rr_new) < tol) break;
        
        const double beta = rr_new / rr;
        const int block = 256, grid = (n + block - 1) / block;
        XpayKernel<<<grid, block>>>(d_p, d_r, beta, n); // p = r + beta*p
        if (!CudaOk(cudaGetLastError(), error, "cg p update")) { cleanup(); return false; }
        rr = rr_new;
    }
    
    x_host->assign(n, 0.0);
    if (!CudaOk(cudaMemcpy(x_host->data(), d_x, sizeof(double)*n, cudaMemcpyDeviceToHost), error, "copy x")) { cleanup(); return false; }
    
    cleanup();
    return true;
}


bool KrylovGMRESCuda(const CudaLinOp2D& op, const std::vector<double>& b_host, std::vector<double>* x_host, int max_iter, int restart, double tol, std::string* error) {
    const int n = op.nx * op.ny;
    CudaKrylovBuffers red;
    if (!AllocKrylovBuffers(n, &red, error)) return false;

    double *d_x = nullptr, *d_b = nullptr, *d_r = nullptr, *d_Ax = nullptr, *d_w = nullptr;
    std::vector<double*> d_V;
    
    auto cleanup = [&]() {
        for (double* ptr : d_V) if (ptr) cudaFree(ptr);
        cudaFree(d_x); cudaFree(d_b); cudaFree(d_r); cudaFree(d_Ax); cudaFree(d_w);
        FreeKrylovBuffers(&red);
    };

    restart = std::max(1, restart);
    if (!CudaOk(cudaMalloc(&d_x, sizeof(double)*n), error, "GMRES malloc") || !CudaOk(cudaMalloc(&d_b, sizeof(double)*n), error, "GMRES malloc") ||
        !CudaOk(cudaMalloc(&d_r, sizeof(double)*n), error, "GMRES malloc") || !CudaOk(cudaMalloc(&d_Ax, sizeof(double)*n), error, "GMRES malloc") ||
        !CudaOk(cudaMalloc(&d_w, sizeof(double)*n), error, "GMRES malloc")) {
        cleanup(); return false;
    }

    if (!CudaOk(cudaMemcpy(d_b, b_host.data(), sizeof(double)*n, cudaMemcpyHostToDevice), error, "copy b")) { cleanup(); return false; }
    DeviceVecSet(d_x, 0.0, n);

    int iter_total = 0;
    while (iter_total < max_iter) {
        if (!op.apply(d_x, d_Ax, error)) { cleanup(); return false; }
        DeviceLinComb2(d_r, d_b, 1.0, d_Ax, -1.0, n);
        
        double beta = 0.0;
        if (!DeviceNorm2(d_r, n, &red, &beta, error)) { cleanup(); return false; }
        if (beta < tol) break;

        const int m = std::min(restart, max_iter - iter_total);
        d_V.assign(m + 1, nullptr);
        for (int i = 0; i < m + 1; ++i) {
            if (!CudaOk(cudaMalloc(&d_V[i], sizeof(double)*n), error, "cudaMalloc V")) { cleanup(); return false; }
        }

        DeviceVecCopy(d_r, d_V[0], n);
        DeviceScale(d_V[0], 1.0 / beta, n);

        std::vector<std::vector<double>> H(m + 1, std::vector<double>(m, 0.0));
        std::vector<double> cs(m, 0.0), sn(m, 0.0), g(m + 1, 0.0);
        g[0] = beta;
        
        int k_final = -1;
        for (int k = 0; k < m; ++k) {
            if (!op.apply(d_V[k], d_w, error)) { cleanup(); return false; }
            for (int j = 0; j <= k; ++j) {
                double h = 0.0;
                if (!DeviceDot(d_w, d_V[j], n, &red, &h, error)) { cleanup(); return false; }
                H[j][k] = h;
                DeviceAxpy(d_w, d_V[j], -h, n);
            }
            double h_next = 0.0;
            if (!DeviceNorm2(d_w, n, &red, &h_next, error)) { cleanup(); return false; }
            H[k + 1][k] = h_next;
            
            if (h_next > 1e-30) {
                DeviceVecCopy(d_w, d_V[k + 1], n);
                DeviceScale(d_V[k + 1], 1.0 / h_next, n);
            }

            for (int j = 0; j < k; ++j) {
                const double h0 = H[j][k], h1 = H[j+1][k];
                H[j][k] = cs[j] * h0 + sn[j] * h1;
                H[j+1][k] = -sn[j] * h0 + cs[j] * h1;
            }
            
            const double h0 = H[k][k], h1 = H[k+1][k];
            const double d_denom = std::sqrt(h0*h0 + h1*h1);
            cs[k] = d_denom < 1e-30 ? 1.0 : h0 / d_denom;
            sn[k] = d_denom < 1e-30 ? 0.0 : h1 / d_denom;
            H[k][k] = cs[k]*h0 + sn[k]*h1;
            H[k+1][k] = 0.0;
            
            const double g0 = g[k];
            g[k] = cs[k] * g0;
            g[k+1] = -sn[k] * g0;
            
            ++iter_total;
            if (std::abs(g[k+1]) < tol) { k_final = k; break; }
            if (h_next < 1e-30) { k_final = k; break; }
        }

        const int k_use = (k_final >= 0) ? (k_final + 1) : m;
        std::vector<double> y(k_use, 0.0);
        for (int i = k_use - 1; i >= 0; --i) {
            double sum = g[i];
            for (int j = i + 1; j < k_use; ++j) sum -= H[i][j] * y[j];
            const double diag = H[i][i];
            if (std::abs(diag) < 1e-30) { *error = "gmres breakdown"; cleanup(); return false; }
            y[i] = sum / diag;
        }

        for (int j = 0; j < k_use; ++j) DeviceAxpy(d_x, d_V[j], y[j], n);
        
        for (double* ptr : d_V) cudaFree(ptr);
        d_V.clear();
    }
    
    x_host->assign(n, 0.0);
    if (!CudaOk(cudaMemcpy(x_host->data(), d_x, sizeof(double)*n, cudaMemcpyDeviceToHost), error, "copy x")) { cleanup(); return false; }
    
    cleanup();
    return true;
}

} // namespace

SolveOutput CudaSolveKrylov(const SolveInput& input) {
    const Domain& d = input.domain;
    if (HasImplicitShape(input)) return {"CUDA Krylov requires CPU fallback: implicit domain shape unsupported", {}};
    if (!input.integrals.empty()) return {"CUDA Krylov requires CPU fallback: integral terms unsupported", {}};
    if (!input.nonlinear.empty()) return {"CUDA Krylov requires CPU fallback: nonlinear terms unsupported", {}};
    if (!IsAllDirichlet(input.bc)) return {"CUDA Krylov currently require Dirichlet boundaries on all sides", {}};

    const double dx = (d.xmax - d.xmin) / (d.nx - 1);
    const double dy = (d.ymax - d.ymin) / (d.ny - 1);

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

    CudaLinOp2D op;
    op.nx = d.nx;
    op.ny = d.ny;
    op.ax = input.pde.a / (dx * dx);
    op.by = input.pde.b / (dy * dy);
    op.cx = input.pde.c / (2.0 * dx);
    op.dyc = input.pde.d / (2.0 * dy);
    op.center = -2.0 * op.ax - 2.0 * op.by + input.pde.e;

    std::vector<double> xvec;
    std::string solve_error;
    const int max_iter = std::max(1, input.solver.max_iter);
    const double tol = input.solver.tol;
    bool ok = false;

    if (input.solver.method == SolveMethod::CG) {
        if (fabs(op.cx) > 1e-12 || fabs(op.dyc) > 1e-12) return {"CUDA CG requires symmetric operator (no convection terms)", {}};
        ok = KrylovCGCuda(op, bvec, &xvec, max_iter, tol, &solve_error);
    } else if (input.solver.method == SolveMethod::BiCGStab) {
        ok = KrylovBiCGStabCuda(op, bvec, &xvec, max_iter, tol, &solve_error);
    } else if (input.solver.method == SolveMethod::GMRES) {
        ok = KrylovGMRESCuda(op, bvec, &xvec, max_iter, input.solver.gmres_restart, tol, &solve_error);
    } else {
        return {"Unknown Krylov method for CUDA", {}};
    }

    if (!ok) {
        return {solve_error.empty() ? "CUDA Krylov solver failed to converge" : solve_error, {}};
    }
    return {"", xvec};
}
