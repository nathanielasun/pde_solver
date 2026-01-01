#import "MetalSolve.h"
#import "solvers/relaxation.h"
#import "solvers/krylov.h"
#import "solvers/multigrid.h"
#import "utils/metal_utils.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <string>

bool MetalIsAvailable(std::string* note) {
  id<MTLDevice> device = MetalCreateDevice(note);
  return device != nil;
}

SolveOutput SolvePDEMetal(const SolveInput& input) {
    const Domain& d = input.domain;
    if (d.nx < 3 || d.ny < 3) {
        return {"grid must be at least 3x3", {}};
    }
    if (d.nz > 1) {
        return {"Metal backend supports 2D domains only", {}};
    }
    if (d.xmax <= d.xmin || d.ymax <= d.ymin) {
        return {"domain bounds are invalid", {}};
    }

    std::string error;
    if (!MetalIsAvailable(&error)) {
        return {"Metal unavailable: " + error, {}};
    }

    switch (input.solver.method) {
        case SolveMethod::Jacobi:
        case SolveMethod::GaussSeidel:
        case SolveMethod::SOR:
            return MetalSolveRelaxation(input);

        case SolveMethod::CG:
        case SolveMethod::BiCGStab:
        case SolveMethod::GMRES:
            return MetalSolveKrylov(input);

        case SolveMethod::MultigridVcycle:
            return MetalSolveMultigrid(input);
        
        default:
            return {"Selected solver method not implemented for Metal backend", {}};
    }
}
