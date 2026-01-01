#import "krylov.h"
#import "../utils/metal_utils.h"

#include <algorithm>
#include <cmath>
#include <optional>
#include <string>
#include <vector>

#include "expression_eval.h"
namespace {

bool IsAllDirichlet(const BoundarySet& bc) {
  return bc.left.kind == BCKind::Dirichlet &&
         bc.right.kind == BCKind::Dirichlet &&
         bc.bottom.kind == BCKind::Dirichlet &&
         bc.top.kind == BCKind::Dirichlet;
}

float EvalExprHost(const BoundaryCondition::Expression& expr, double x, double y) {
  return static_cast<float>(expr.constant + expr.x * x + expr.y * y + expr.z * 0.0);
}

bool BuildDirichletRhs(const Domain& d, const BoundarySet& bc, float f,
                       const ExpressionEvaluator* rhs_eval, std::vector<float>* b,
                       std::string* error) {
  if (!b) return false;
  if (!IsAllDirichlet(bc)) {
    if (error) *error = "Dirichlet boundaries required";
    return false;
  }
  const int nx = d.nx, ny = d.ny;
  b->assign(nx * ny, 0.0f);
  const double dx = (d.xmax - d.xmin) / (nx - 1);
  const double dy = (d.ymax - d.ymin) / (ny - 1);
  for (int j = 0; j < ny; ++j) {
    const double y = d.ymin + j * dy;
    for (int i = 0; i < nx; ++i) {
      const double x = d.xmin + i * dx;
      const int idx = j * nx + i;
      if (i == 0 || i == nx - 1 || j == 0 || j == ny - 1) {
        if (i == 0) (*b)[idx] = EvalExprHost(bc.left.value, d.xmin, y);
        else if (i == nx - 1) (*b)[idx] = EvalExprHost(bc.right.value, d.xmax, y);
        else if (j == 0) (*b)[idx] = EvalExprHost(bc.bottom.value, x, d.ymin);
        else (*b)[idx] = EvalExprHost(bc.top.value, x, d.ymax);
      } else {
        const float f_val = rhs_eval ? static_cast<float>(rhs_eval->Eval(x, y, 0.0, 0.0)) : f;
        (*b)[idx] = -f_val;
      }
    }
  }
  return true;
}

} // namespace


SolveOutput MetalSolveKrylov(const SolveInput& input) {
    const Domain& d = input.domain;
    std::string error;

    if (d.nz > 1) return {"Metal Krylov supports 2D domains only", {}};
    if (!IsAllDirichlet(input.bc)) return {"Metal Krylov solvers currently require Dirichlet boundaries", {}};
    if (HasImplicitShape(input)) return {"Metal Krylov does not support implicit domain shapes", {}};
    if (!input.integrals.empty()) return {"Metal Krylov does not support integral terms", {}};
    if (!input.nonlinear.empty() || !input.nonlinear_derivatives.empty()) {
        return {"Metal Krylov does not support nonlinear terms", {}};
    }
    std::optional<ExpressionEvaluator> rhs_eval;
    if (!input.pde.rhs_latex.empty()) {
        ExpressionEvaluator eval = ExpressionEvaluator::ParseLatex(input.pde.rhs_latex);
        if (!eval.ok()) {
            return {"invalid RHS expression: " + eval.error(), {}};
        }
        rhs_eval.emplace(std::move(eval));
    }

    std::string device_note;
    id<MTLDevice> device = MetalCreateDevice(&device_note);
    if (!device) {
        const std::string msg = device_note.empty()
            ? "Metal device unavailable"
            : "Metal device unavailable: " + device_note;
        return {msg, {}};
    }
    id<MTLCommandQueue> queue = [device newCommandQueue];
    if (!queue) return {"Failed to create Metal command queue", {}};
    id<MTLLibrary> library = LoadLibrary(device, &error);
    if (!library) return {"Failed to load metal library: " + error, {}};

    MetalVectorOps vecOps(device, queue, library);
    if (!vecOps.isReady(&error)) return {"Failed to initialize Metal vector operations: " + error, {}};

    NSError* ns_error = nil;
    id<MTLFunction> ax_apply_fn = [library newFunctionWithName:@"ax_apply"];
    if (!ax_apply_fn) return {"ax_apply kernel not found", {}};
    id<MTLComputePipelineState> ax_apply_pipeline = [device newComputePipelineStateWithFunction:ax_apply_fn error:&ns_error];
    if (!ax_apply_pipeline) return {"Failed to create ax_apply pipeline", {}};

    const int nx = d.nx, ny = d.ny, total = nx * ny;
    const float dx = (d.xmax - d.xmin) / (nx - 1);
    const float dy = (d.ymax - d.ymin) / (ny - 1);
    const float ax = input.pde.a / (dx * dx);
    const float by = input.pde.b / (dy * dy);
    const float cx = input.pde.c / (2.0f * dx);
    const float dyc = input.pde.d / (2.0f * dy);
    const float center = -2.0f * ax - 2.0f * by + input.pde.e;

    std::vector<float> b_host;
    if (!BuildDirichletRhs(d, input.bc, input.pde.f,
                           rhs_eval ? &(*rhs_eval) : nullptr, &b_host, &error)) {
        return {error, {}};
    }
    
    id<MTLBuffer> b_buf = [device newBufferWithBytes:b_host.data() length:sizeof(float) * total options:MTLResourceStorageModeShared];
    id<MTLBuffer> x_buf = [device newBufferWithLength:sizeof(float) * total options:MTLResourceStorageModePrivate];
    if (!b_buf || !x_buf) return {"Failed to allocate Metal buffers", {}};

    if (!vecOps.set(x_buf, 0.0f, total)) return {"Metal vector op failed: set", {}};

    auto ax_apply_device = [&](id<MTLBuffer> x_in, id<MTLBuffer> y_out) -> bool {
        @autoreleasepool {
            id<MTLCommandBuffer> cmd = [queue commandBuffer];
            id<MTLComputeCommandEncoder> encoder = [cmd computeCommandEncoder];
            [encoder setComputePipelineState:ax_apply_pipeline];
            [encoder setBuffer:x_in offset:0 atIndex:0];
            [encoder setBuffer:y_out offset:0 atIndex:1];
            DomainInfo domain_info = {nx, ny, (float)d.xmin, (float)d.xmax, (float)d.ymin, (float)d.ymax, dx, dy};
            [encoder setBytes:&domain_info length:sizeof(DomainInfo) atIndex:2];
            [encoder setBytes:&ax length:sizeof(float) atIndex:3];
            [encoder setBytes:&by length:sizeof(float) atIndex:4];
            [encoder setBytes:&cx length:sizeof(float) atIndex:5];
            [encoder setBytes:&dyc length:sizeof(float) atIndex:6];
            [encoder setBytes:&center length:sizeof(float) atIndex:7];
            ThreadgroupSize tg = ChooseThreadgroupSize(ax_apply_pipeline, 0, 0);
            Dispatch2D(encoder, nx, ny, tg);
            [encoder endEncoding];
            [cmd commit];
            [cmd waitUntilCompleted];
            return true;
        }
    };

    error.clear();
    const int max_iter = std::max(1, input.solver.max_iter);
    const float tol = static_cast<float>(input.solver.tol);
    bool ok = false;

    if (input.solver.method == SolveMethod::CG) {
        if (std::abs(cx) > 1e-6f || std::abs(dyc) > 1e-6f) {
            return {"Metal CG requires symmetric operator", {}};
        }
        id<MTLBuffer> r = [device newBufferWithLength:sizeof(float) * total options:MTLResourceStorageModePrivate];
        id<MTLBuffer> p = [device newBufferWithLength:sizeof(float) * total options:MTLResourceStorageModePrivate];
        id<MTLBuffer> Ap = [device newBufferWithLength:sizeof(float) * total options:MTLResourceStorageModePrivate];
        if (!r || !p || !Ap) return {"Failed to allocate Metal Krylov buffers", {}};

        if (!vecOps.copy(b_buf, r, total)) return {"Metal vector op failed: copy", {}};
        if (!vecOps.copy(r, p, total)) return {"Metal vector op failed: copy", {}};

        float rr = 0.0f;
        if (!vecOps.dot(r, r, total, &rr)) return {"Metal vector op failed: dot", {}};

        for (int i = 0; i < max_iter; ++i) {
            if (std::sqrt(rr) < tol) break;

            if (!ax_apply_device(p, Ap)) return {"Metal ax_apply failed", {}};
            float pAp = 0.0f;
            if (!vecOps.dot(p, Ap, total, &pAp)) return {"Metal vector op failed: dot", {}};

            if (std::abs(pAp) < 1e-20f) { error = "CG breakdown"; break; }

            const float alpha = rr / pAp;
            if (!vecOps.axpy(x_buf, p, alpha, total)) return {"Metal vector op failed: axpy", {}};
            if (!vecOps.axpy(r, Ap, -alpha, total)) return {"Metal vector op failed: axpy", {}};

            float rr_new = 0.0f;
            if (!vecOps.dot(r, r, total, &rr_new)) return {"Metal vector op failed: dot", {}};

            const float beta = rr_new / rr;
            if (!vecOps.lincomb2(p, r, p, 1.0f, beta, total)) {
                return {"Metal vector op failed: lincomb2", {}};
            }
            rr = rr_new;
        }
        ok = error.empty();
    } else if (input.solver.method == SolveMethod::BiCGStab) {
        id<MTLBuffer> r = [device newBufferWithLength:sizeof(float) * total options:MTLResourceStorageModePrivate];
        id<MTLBuffer> rhat = [device newBufferWithLength:sizeof(float) * total options:MTLResourceStorageModePrivate];
        id<MTLBuffer> p = [device newBufferWithLength:sizeof(float) * total options:MTLResourceStorageModePrivate];
        id<MTLBuffer> v = [device newBufferWithLength:sizeof(float) * total options:MTLResourceStorageModePrivate];
        id<MTLBuffer> s = [device newBufferWithLength:sizeof(float) * total options:MTLResourceStorageModePrivate];
        id<MTLBuffer> t = [device newBufferWithLength:sizeof(float) * total options:MTLResourceStorageModePrivate];
        id<MTLBuffer> Ax = [device newBufferWithLength:sizeof(float) * total options:MTLResourceStorageModePrivate];
        if (!r || !rhat || !p || !v || !s || !t || !Ax) {
            return {"Failed to allocate Metal Krylov buffers", {}};
        }

        if (!ax_apply_device(x_buf, Ax)) return {"Metal ax_apply failed", {}};
        if (!vecOps.lincomb2(r, b_buf, Ax, 1.0f, -1.0f, total)) {
            return {"Metal vector op failed: lincomb2", {}};
        }
        if (!vecOps.copy(r, rhat, total)) return {"Metal vector op failed: copy", {}};
        if (!vecOps.set(p, 0.0f, total)) return {"Metal vector op failed: set", {}};
        if (!vecOps.set(v, 0.0f, total)) return {"Metal vector op failed: set", {}};

        float resid = 0.0f;
        if (!vecOps.norm2(r, total, &resid)) return {"Metal vector op failed: norm2", {}};
        if (resid < tol) {
            ok = true;
        } else {
            float rho_prev = 1.0f;
            float alpha = 1.0f;
            float omega = 1.0f;

            for (int iter = 0; iter < max_iter; ++iter) {
                float rho = 0.0f;
                if (!vecOps.dot(rhat, r, total, &rho)) return {"Metal vector op failed: dot", {}};
                if (std::abs(rho) < 1e-30f) { error = "bicgstab breakdown (rho)"; break; }

                if (iter == 0) {
                    if (!vecOps.copy(r, p, total)) return {"Metal vector op failed: copy", {}};
                } else {
                    const float beta = (rho / rho_prev) * (alpha / omega);
                    if (!vecOps.lincomb2(p, p, v, 1.0f, -omega, total)) {
                        return {"Metal vector op failed: lincomb2", {}};
                    }
                    if (!vecOps.scale(p, beta, total)) return {"Metal vector op failed: scale", {}};
                    if (!vecOps.axpy(p, r, 1.0f, total)) return {"Metal vector op failed: axpy", {}};
                }

                if (!ax_apply_device(p, v)) return {"Metal ax_apply failed", {}};
                float denom = 0.0f;
                if (!vecOps.dot(rhat, v, total, &denom)) return {"Metal vector op failed: dot", {}};
                if (std::abs(denom) < 1e-30f) { error = "bicgstab breakdown (alpha denom)"; break; }
                alpha = rho / denom;

                if (!vecOps.lincomb2(s, r, v, 1.0f, -alpha, total)) {
                    return {"Metal vector op failed: lincomb2", {}};
                }
                float s_norm = 0.0f;
                if (!vecOps.norm2(s, total, &s_norm)) return {"Metal vector op failed: norm2", {}};
                if (s_norm < tol) {
                    if (!vecOps.axpy(x_buf, p, alpha, total)) return {"Metal vector op failed: axpy", {}};
                    break;
                }

                if (!ax_apply_device(s, t)) return {"Metal ax_apply failed", {}};
                float tt = 0.0f;
                float ts = 0.0f;
                if (!vecOps.dot(t, t, total, &tt)) return {"Metal vector op failed: dot", {}};
                if (!vecOps.dot(t, s, total, &ts)) return {"Metal vector op failed: dot", {}};
                if (std::abs(tt) < 1e-30f) { error = "bicgstab breakdown (omega denom)"; break; }
                omega = ts / tt;
                if (std::abs(omega) < 1e-30f) { error = "bicgstab breakdown (omega)"; break; }

                if (!vecOps.axpy(x_buf, p, alpha, total)) return {"Metal vector op failed: axpy", {}};
                if (!vecOps.axpy(x_buf, s, omega, total)) return {"Metal vector op failed: axpy", {}};
                if (!vecOps.lincomb2(r, s, t, 1.0f, -omega, total)) {
                    return {"Metal vector op failed: lincomb2", {}};
                }
                if (!vecOps.norm2(r, total, &resid)) return {"Metal vector op failed: norm2", {}};
                if (resid < tol) break;
                rho_prev = rho;
            }
            ok = error.empty();
        }
    } else if (input.solver.method == SolveMethod::GMRES) {
        const int restart = std::max(1, input.solver.gmres_restart);
        id<MTLBuffer> r = [device newBufferWithLength:sizeof(float) * total options:MTLResourceStorageModePrivate];
        id<MTLBuffer> Ax = [device newBufferWithLength:sizeof(float) * total options:MTLResourceStorageModePrivate];
        id<MTLBuffer> w = [device newBufferWithLength:sizeof(float) * total options:MTLResourceStorageModePrivate];
        if (!r || !Ax || !w) return {"Failed to allocate Metal Krylov buffers", {}};

        int iter_total = 0;
        while (iter_total < max_iter) {
            if (!ax_apply_device(x_buf, Ax)) return {"Metal ax_apply failed", {}};
            if (!vecOps.lincomb2(r, b_buf, Ax, 1.0f, -1.0f, total)) {
                return {"Metal vector op failed: lincomb2", {}};
            }
            float beta = 0.0f;
            if (!vecOps.norm2(r, total, &beta)) return {"Metal vector op failed: norm2", {}};
            if (beta < tol) break;

            const int m = std::min(restart, max_iter - iter_total);
            std::vector<id<MTLBuffer>> V(static_cast<size_t>(m + 1), nil);
            for (int i = 0; i < m + 1; ++i) {
                V[i] = [device newBufferWithLength:sizeof(float) * total options:MTLResourceStorageModePrivate];
                if (!V[i]) return {"Failed to allocate Metal GMRES buffers", {}};
            }

            if (!vecOps.copy(r, V[0], total)) return {"Metal vector op failed: copy", {}};
            if (!vecOps.scale(V[0], 1.0f / beta, total)) return {"Metal vector op failed: scale", {}};

            std::vector<std::vector<float>> H(static_cast<size_t>(m + 1), std::vector<float>(static_cast<size_t>(m), 0.0f));
            std::vector<float> cs(static_cast<size_t>(m), 0.0f);
            std::vector<float> sn(static_cast<size_t>(m), 0.0f);
            std::vector<float> g(static_cast<size_t>(m + 1), 0.0f);
            g[0] = beta;

            int k_final = -1;
            for (int k = 0; k < m; ++k) {
                if (!ax_apply_device(V[k], w)) return {"Metal ax_apply failed", {}};
                for (int j = 0; j <= k; ++j) {
                    float h = 0.0f;
                    if (!vecOps.dot(w, V[j], total, &h)) return {"Metal vector op failed: dot", {}};
                    H[j][k] = h;
                    if (!vecOps.axpy(w, V[j], -h, total)) return {"Metal vector op failed: axpy", {}};
                }
                float h_next = 0.0f;
                if (!vecOps.norm2(w, total, &h_next)) return {"Metal vector op failed: norm2", {}};
                H[k + 1][k] = h_next;

                if (h_next > 1e-30f) {
                    if (!vecOps.copy(w, V[k + 1], total)) return {"Metal vector op failed: copy", {}};
                    if (!vecOps.scale(V[k + 1], 1.0f / h_next, total)) return {"Metal vector op failed: scale", {}};
                }

                for (int j = 0; j < k; ++j) {
                    const float h0 = H[j][k];
                    const float h1 = H[j + 1][k];
                    H[j][k] = cs[j] * h0 + sn[j] * h1;
                    H[j + 1][k] = -sn[j] * h0 + cs[j] * h1;
                }

                const float h0 = H[k][k];
                const float h1 = H[k + 1][k];
                const float denom = std::sqrt(h0 * h0 + h1 * h1);
                cs[k] = denom < 1e-30f ? 1.0f : h0 / denom;
                sn[k] = denom < 1e-30f ? 0.0f : h1 / denom;
                H[k][k] = cs[k] * h0 + sn[k] * h1;
                H[k + 1][k] = 0.0f;

                const float g0 = g[k];
                g[k] = cs[k] * g0;
                g[k + 1] = -sn[k] * g0;

                ++iter_total;
                if (std::abs(g[k + 1]) < tol) { k_final = k; break; }
                if (h_next < 1e-30f) { k_final = k; break; }
            }

            const int k_use = (k_final >= 0) ? (k_final + 1) : m;
            std::vector<float> y(static_cast<size_t>(k_use), 0.0f);
            for (int i = k_use - 1; i >= 0; --i) {
                float sum = g[i];
                for (int j = i + 1; j < k_use; ++j) sum -= H[i][j] * y[j];
                const float diag = H[i][i];
                if (std::abs(diag) < 1e-30f) { error = "gmres breakdown"; break; }
                y[i] = sum / diag;
            }
            if (!error.empty()) break;

            for (int j = 0; j < k_use; ++j) {
                if (!vecOps.axpy(x_buf, V[j], y[j], total)) return {"Metal vector op failed: axpy", {}};
            }
        }
        ok = error.empty();
    } else {
        return {"Unknown Krylov method for Metal", {}};
    }

    if (!ok) {
        return {error.empty() ? "Metal Krylov solver failed to converge" : error, {}};
    }


    std::vector<double> grid_host(total);
    id<MTLBuffer> host_res_buf = [device newBufferWithLength:sizeof(float) * total options:MTLResourceStorageModeShared];
    @autoreleasepool {
        id<MTLCommandBuffer> cmd = [queue commandBuffer];
        id<MTLBlitCommandEncoder> blit = [cmd blitCommandEncoder];
        [blit copyFromBuffer:x_buf sourceOffset:0 toBuffer:host_res_buf destinationOffset:0 size:sizeof(float) * total];
        [blit endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
    }
    const float* grid_ptr = static_cast<const float*>([host_res_buf contents]);
    for(int i = 0; i < total; ++i) grid_host[i] = grid_ptr[i];

    return {"", grid_host};
}
