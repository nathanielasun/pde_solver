#ifndef METAL_SOLVERS_RELAXATION_H
#define METAL_SOLVERS_RELAXATION_H

#include "pde_types.h"

// Solves a PDE using Jacobi or SOR relaxation methods on the GPU (Metal).
SolveOutput MetalSolveRelaxation(const SolveInput& input);

#endif // METAL_SOLVERS_RELAXATION_H
