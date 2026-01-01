#ifndef METAL_SOLVERS_MULTIGRID_H
#define METAL_SOLVERS_MULTIGRID_H

#include "pde_types.h"

// Solves a PDE using a multigrid V-cycle method on the GPU (Metal).
SolveOutput MetalSolveMultigrid(const SolveInput& input);

#endif // METAL_SOLVERS_MULTIGRID_H
