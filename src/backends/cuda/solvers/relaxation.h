#ifndef CUDA_SOLVERS_RELAXATION_H
#define CUDA_SOLVERS_RELAXATION_H

#include "pde_types.h"
#include <string>

// Solves a PDE using Jacobi or SOR relaxation methods on the GPU.
SolveOutput CudaSolveRelaxation(const SolveInput& input);

#endif // CUDA_SOLVERS_RELAXATION_H
