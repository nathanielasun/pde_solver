#import "multigrid.h"
#import "../utils/metal_utils.h"

#include <algorithm>
#include <cmath>
#include <optional>
#include <string>
#include <vector>

#include "expression_eval.h"
namespace {

struct MgLevel {
  Domain d;
  DomainInfo info{};
  float ax = 0.0f;
  float by = 0.0f;
  float center = 0.0f;
  id<MTLBuffer> u = nil;
  id<MTLBuffer> b = nil;
  id<MTLBuffer> r = nil;
};

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
                       const ExpressionEvaluator* rhs_eval,
                       std::vector<float>* b, std::string* error) {
  if (!b) {
    return false;
  }
  if (!IsAllDirichlet(bc)) {
    if (error) {
      *error = "Dirichlet boundaries required";
    }
    return false;
  }
  const int nx = d.nx;
  const int ny = d.ny;
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

bool CanCoarsenOdd(int n) {
  return n >= 5 && (n % 2 == 1);
}

}  // namespace

SolveOutput MetalSolveMultigrid(const SolveInput& input) {
  const Domain& d = input.domain;
  if (d.nz > 1) {
    return {"Metal multigrid supports 2D domains only", {}};
  }
  if (d.nx < 5 || d.ny < 5 || (d.nx % 2) == 0 || (d.ny % 2) == 0) {
    return {"Metal multigrid requires odd grid dimensions of at least 5", {}};
  }
  if (std::abs(input.pde.c) > 1e-6 || std::abs(input.pde.d) > 1e-6 ||
      std::abs(input.pde.ab) > 1e-6 || std::abs(input.pde.ac) > 1e-6 ||
      std::abs(input.pde.bc) > 1e-6) {
    return {"Metal multigrid requires no convection or mixed-derivative terms", {}};
  }
  if (!IsAllDirichlet(input.bc)) {
    return {"Metal multigrid requires Dirichlet boundaries", {}};
  }
  if (HasImplicitShape(input)) {
    return {"Metal multigrid does not support implicit domain shapes", {}};
  }
  if (!input.integrals.empty()) {
    return {"Metal multigrid does not support integral terms", {}};
  }
  if (!input.nonlinear.empty() || !input.nonlinear_derivatives.empty()) {
    return {"Metal multigrid does not support nonlinear terms", {}};
  }

  std::optional<ExpressionEvaluator> rhs_eval;
  if (!input.pde.rhs_latex.empty()) {
    ExpressionEvaluator eval = ExpressionEvaluator::ParseLatex(input.pde.rhs_latex);
    if (!eval.ok()) {
      return {"invalid RHS expression: " + eval.error(), {}};
    }
    rhs_eval.emplace(std::move(eval));
  }

  std::string error;
  std::string device_note;
  id<MTLDevice> device = MetalCreateDevice(&device_note);
  if (!device) {
    const std::string msg = device_note.empty()
        ? "Metal device unavailable"
        : "Metal device unavailable: " + device_note;
    return {msg, {}};
  }
  id<MTLCommandQueue> queue = [device newCommandQueue];
  if (!queue) {
    return {"Failed to create Metal command queue", {}};
  }
  id<MTLLibrary> library = LoadLibrary(device, &error);
  if (!library) {
    return {"Failed to load metal library: " + error, {}};
  }

  MetalVectorOps vecOps(device, queue, library);
  if (!vecOps.isReady(&error)) {
    return {"Failed to initialize Metal vector operations: " + error, {}};
  }

  NSError* ns_error = nil;
  id<MTLFunction> rbgs_fn = [library newFunctionWithName:@"rbgs_rhs"];
  id<MTLFunction> residual_fn = [library newFunctionWithName:@"residual_poisson"];
  id<MTLFunction> restrict_fn = [library newFunctionWithName:@"restrict_full_weighting"];
  id<MTLFunction> prolong_fn = [library newFunctionWithName:@"prolong_bilinear_add"];
  if (!rbgs_fn || !residual_fn || !restrict_fn || !prolong_fn) {
    return {"Required Metal multigrid kernels not found", {}};
  }

  id<MTLComputePipelineState> rbgs_pipeline = [device newComputePipelineStateWithFunction:rbgs_fn error:&ns_error];
  id<MTLComputePipelineState> residual_pipeline = [device newComputePipelineStateWithFunction:residual_fn error:&ns_error];
  id<MTLComputePipelineState> restrict_pipeline = [device newComputePipelineStateWithFunction:restrict_fn error:&ns_error];
  id<MTLComputePipelineState> prolong_pipeline = [device newComputePipelineStateWithFunction:prolong_fn error:&ns_error];
  if (!rbgs_pipeline || !residual_pipeline || !restrict_pipeline || !prolong_pipeline) {
    return {"Failed to create Metal multigrid pipeline states", {}};
  }

  std::vector<float> b_host;
  if (!BuildDirichletRhs(d, input.bc, static_cast<float>(input.pde.f),
                         rhs_eval ? &(*rhs_eval) : nullptr, &b_host, &error)) {
    return {error, {}};
  }

  const int max_levels = std::max(1, input.solver.mg_max_levels);
  std::vector<MgLevel> levels;
  levels.reserve(static_cast<size_t>(max_levels));

  Domain cur = d;
  for (int level = 0; level < max_levels; ++level) {
    MgLevel lvl;
    lvl.d = cur;
    const double dx = (cur.xmax - cur.xmin) / (cur.nx - 1);
    const double dy = (cur.ymax - cur.ymin) / (cur.ny - 1);
    lvl.info = {cur.nx, cur.ny, static_cast<float>(cur.xmin), static_cast<float>(cur.xmax),
                static_cast<float>(cur.ymin), static_cast<float>(cur.ymax),
                static_cast<float>(dx), static_cast<float>(dy)};
    lvl.ax = static_cast<float>(input.pde.a / (dx * dx));
    lvl.by = static_cast<float>(input.pde.b / (dy * dy));
    lvl.center = static_cast<float>(-2.0 * lvl.ax - 2.0 * lvl.by + input.pde.e);
    if (std::abs(lvl.center) < 1e-6f) {
      return {"degenerate PDE center coefficient", {}};
    }

    const int n = cur.nx * cur.ny;
    lvl.u = [device newBufferWithLength:sizeof(float) * n options:MTLResourceStorageModePrivate];
    lvl.b = [device newBufferWithLength:sizeof(float) * n options:MTLResourceStorageModePrivate];
    lvl.r = [device newBufferWithLength:sizeof(float) * n options:MTLResourceStorageModePrivate];
    if (!lvl.u || !lvl.b || !lvl.r) {
      return {"Failed to allocate Metal multigrid buffers", {}};
    }
    levels.push_back(lvl);

    if (!CanCoarsenOdd(cur.nx) || !CanCoarsenOdd(cur.ny) || cur.nx <= 5 || cur.ny <= 5) {
      break;
    }
    cur.nx = (cur.nx + 1) / 2;
    cur.ny = (cur.ny + 1) / 2;
  }

  if (levels.empty()) {
    return {"Metal multigrid failed to build levels", {}};
  }

  // Initialize buffers.
  {
    MgLevel& fine = levels.front();
    const int n = fine.d.nx * fine.d.ny;
    std::vector<float> u_host(static_cast<size_t>(n), 0.0f);
    for (int j = 0; j < fine.d.ny; ++j) {
      for (int i = 0; i < fine.d.nx; ++i) {
        if (i == 0 || j == 0 || i == fine.d.nx - 1 || j == fine.d.ny - 1) {
          const int idx = j * fine.d.nx + i;
          u_host[static_cast<size_t>(idx)] = b_host[static_cast<size_t>(idx)];
        }
      }
    }
    id<MTLBuffer> b_stage = [device newBufferWithBytes:b_host.data()
                                               length:sizeof(float) * n
                                              options:MTLResourceStorageModeShared];
    id<MTLBuffer> u_stage = [device newBufferWithBytes:u_host.data()
                                               length:sizeof(float) * n
                                              options:MTLResourceStorageModeShared];
    if (!b_stage) {
      return {"Failed to allocate Metal staging buffer", {}};
    }
    if (!u_stage) {
      return {"Failed to allocate Metal staging buffer", {}};
    }
    @autoreleasepool {
      id<MTLCommandBuffer> cmd = [queue commandBuffer];
      id<MTLBlitCommandEncoder> blit = [cmd blitCommandEncoder];
      [blit copyFromBuffer:b_stage sourceOffset:0 toBuffer:fine.b destinationOffset:0 size:sizeof(float) * n];
      [blit copyFromBuffer:u_stage sourceOffset:0 toBuffer:fine.u destinationOffset:0 size:sizeof(float) * n];
      [blit endEncoding];
      [cmd commit];
      [cmd waitUntilCompleted];
    }

    if (!vecOps.set(fine.r, 0.0f, n)) return {"Metal vector op failed: set", {}};
  }

  for (size_t i = 1; i < levels.size(); ++i) {
    MgLevel& lvl = levels[i];
    const int n = lvl.d.nx * lvl.d.ny;
    if (!vecOps.set(lvl.u, 0.0f, n)) return {"Metal vector op failed: set", {}};
    if (!vecOps.set(lvl.b, 0.0f, n)) return {"Metal vector op failed: set", {}};
    if (!vecOps.set(lvl.r, 0.0f, n)) return {"Metal vector op failed: set", {}};
  }

  ThreadgroupSize tg_rbgs = ChooseThreadgroupSize(rbgs_pipeline, input.solver.metal_tg_x, input.solver.metal_tg_y);
  ThreadgroupSize tg_residual = ChooseThreadgroupSize(residual_pipeline, 0, 0);
  ThreadgroupSize tg_restrict = ChooseThreadgroupSize(restrict_pipeline, 0, 0);
  ThreadgroupSize tg_prolong = ChooseThreadgroupSize(prolong_pipeline, 0, 0);

  auto Smooth = [&](MgLevel& lvl, int iterations) {
    const float omega = 1.0f;
    for (int iter = 0; iter < iterations; ++iter) {
      @autoreleasepool {
        id<MTLCommandBuffer> cmd = [queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [cmd computeCommandEncoder];
        uint parity = 0;
        [encoder setComputePipelineState:rbgs_pipeline];
        [encoder setBuffer:lvl.u offset:0 atIndex:0];
        [encoder setBuffer:lvl.b offset:0 atIndex:1];
        [encoder setBytes:&lvl.info length:sizeof(DomainInfo) atIndex:2];
        [encoder setBytes:&lvl.ax length:sizeof(float) atIndex:3];
        [encoder setBytes:&lvl.by length:sizeof(float) atIndex:4];
        [encoder setBytes:&lvl.center length:sizeof(float) atIndex:5];
        [encoder setBytes:&omega length:sizeof(float) atIndex:6];
        [encoder setBytes:&parity length:sizeof(uint) atIndex:7];
        Dispatch2D(encoder, lvl.d.nx, lvl.d.ny, tg_rbgs);

        parity = 1;
        [encoder setBytes:&parity length:sizeof(uint) atIndex:7];
        Dispatch2D(encoder, lvl.d.nx, lvl.d.ny, tg_rbgs);

        [encoder endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
      }
    }
  };

  auto ComputeResidual = [&](MgLevel& lvl) {
    @autoreleasepool {
      id<MTLCommandBuffer> cmd = [queue commandBuffer];
      id<MTLComputeCommandEncoder> encoder = [cmd computeCommandEncoder];
      [encoder setComputePipelineState:residual_pipeline];
      [encoder setBuffer:lvl.b offset:0 atIndex:0];
      [encoder setBuffer:lvl.u offset:0 atIndex:1];
      [encoder setBuffer:lvl.r offset:0 atIndex:2];
      [encoder setBytes:&lvl.info length:sizeof(DomainInfo) atIndex:3];
      [encoder setBytes:&lvl.ax length:sizeof(float) atIndex:4];
      [encoder setBytes:&lvl.by length:sizeof(float) atIndex:5];
      [encoder setBytes:&lvl.center length:sizeof(float) atIndex:6];
      Dispatch2D(encoder, lvl.d.nx, lvl.d.ny, tg_residual);
      [encoder endEncoding];
      [cmd commit];
      [cmd waitUntilCompleted];
    }
  };

  auto Restrict = [&](MgLevel& fine, MgLevel& coarse) {
    const uint32_t nx_f = static_cast<uint32_t>(fine.d.nx);
    const uint32_t ny_f = static_cast<uint32_t>(fine.d.ny);
    const uint32_t nx_c = static_cast<uint32_t>(coarse.d.nx);
    const uint32_t ny_c = static_cast<uint32_t>(coarse.d.ny);
    @autoreleasepool {
      id<MTLCommandBuffer> cmd = [queue commandBuffer];
      id<MTLComputeCommandEncoder> encoder = [cmd computeCommandEncoder];
      [encoder setComputePipelineState:restrict_pipeline];
      [encoder setBuffer:fine.r offset:0 atIndex:0];
      [encoder setBuffer:coarse.b offset:0 atIndex:1];
      [encoder setBytes:&nx_f length:sizeof(uint32_t) atIndex:2];
      [encoder setBytes:&ny_f length:sizeof(uint32_t) atIndex:3];
      [encoder setBytes:&nx_c length:sizeof(uint32_t) atIndex:4];
      [encoder setBytes:&ny_c length:sizeof(uint32_t) atIndex:5];
      Dispatch2D(encoder, coarse.d.nx, coarse.d.ny, tg_restrict);
      [encoder endEncoding];
      [cmd commit];
      [cmd waitUntilCompleted];
    }
  };

  auto Prolong = [&](MgLevel& coarse, MgLevel& fine) {
    const uint32_t nx_c = static_cast<uint32_t>(coarse.d.nx);
    const uint32_t ny_c = static_cast<uint32_t>(coarse.d.ny);
    const uint32_t nx_f = static_cast<uint32_t>(fine.d.nx);
    const uint32_t ny_f = static_cast<uint32_t>(fine.d.ny);
    @autoreleasepool {
      id<MTLCommandBuffer> cmd = [queue commandBuffer];
      id<MTLComputeCommandEncoder> encoder = [cmd computeCommandEncoder];
      [encoder setComputePipelineState:prolong_pipeline];
      [encoder setBuffer:coarse.u offset:0 atIndex:0];
      [encoder setBuffer:fine.u offset:0 atIndex:1];
      [encoder setBytes:&nx_c length:sizeof(uint32_t) atIndex:2];
      [encoder setBytes:&ny_c length:sizeof(uint32_t) atIndex:3];
      [encoder setBytes:&nx_f length:sizeof(uint32_t) atIndex:4];
      [encoder setBytes:&ny_f length:sizeof(uint32_t) atIndex:5];
      Dispatch2D(encoder, fine.d.nx, fine.d.ny, tg_prolong);
      [encoder endEncoding];
      [cmd commit];
      [cmd waitUntilCompleted];
    }
  };

  const int cycles = std::max(1, input.solver.max_iter);
  const int pre_smooth = std::max(0, input.solver.mg_pre_smooth);
  const int post_smooth = std::max(0, input.solver.mg_post_smooth);
  const int coarse_iters = std::max(1, input.solver.mg_coarse_iters);

  for (int cyc = 0; cyc < cycles; ++cyc) {
    for (size_t ell = 0; ell + 1 < levels.size(); ++ell) {
      Smooth(levels[ell], pre_smooth);
      ComputeResidual(levels[ell]);
      Restrict(levels[ell], levels[ell + 1]);
      const int n_coarse = levels[ell + 1].d.nx * levels[ell + 1].d.ny;
      if (!vecOps.set(levels[ell + 1].u, 0.0f, n_coarse)) {
        return {"Metal vector op failed: set", {}};
      }
    }

    Smooth(levels.back(), coarse_iters);

    for (int ell = static_cast<int>(levels.size()) - 2; ell >= 0; --ell) {
      Prolong(levels[static_cast<size_t>(ell + 1)], levels[static_cast<size_t>(ell)]);
      Smooth(levels[static_cast<size_t>(ell)], post_smooth);
    }
  }

  // Copy result back to host.
  const MgLevel& fine = levels.front();
  const int total = fine.d.nx * fine.d.ny;
  id<MTLBuffer> host_buf = [device newBufferWithLength:sizeof(float) * total options:MTLResourceStorageModeShared];
  if (!host_buf) {
    return {"Failed to allocate Metal host buffer", {}};
  }
  @autoreleasepool {
    id<MTLCommandBuffer> cmd = [queue commandBuffer];
    id<MTLBlitCommandEncoder> blit = [cmd blitCommandEncoder];
    [blit copyFromBuffer:fine.u sourceOffset:0 toBuffer:host_buf destinationOffset:0 size:sizeof(float) * total];
    [blit endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];
  }

  std::vector<double> grid_host(static_cast<size_t>(total), 0.0);
  const float* grid_ptr = static_cast<const float*>([host_buf contents]);
  for (int i = 0; i < total; ++i) {
    grid_host[static_cast<size_t>(i)] = grid_ptr[i];
  }

  return {"", grid_host};
}
