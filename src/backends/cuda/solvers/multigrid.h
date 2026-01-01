#ifndef CUDA_SOLVERS_MULTIGRID_H
#define CUDA_SOLVERS_MULTIGRID_H

#include "pde_types.h"
#include <string>
#include <vector>

// Solves a PDE using a multigrid V-cycle method on the GPU.
SolveOutput CudaSolveMultigrid(const SolveInput& input);

#endif // CUDA_SOLVERS_MULTIGRID_H
