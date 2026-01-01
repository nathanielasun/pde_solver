#ifndef METAL_SOLVERS_KRYLOV_H
#define METAL_SOLVERS_KRYLOV_H

#include "pde_types.h"

// Solves a PDE using Krylov subspace methods on the GPU (Metal).
SolveOutput MetalSolveKrylov(const SolveInput& input);

#endif // METAL_SOLVERS_KRYLOV_H
