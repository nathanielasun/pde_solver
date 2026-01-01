#ifndef CUDA_SOLVERS_KRYLOV_H
#define CUDA_SOLVERS_KRYLOV_H

#include "pde_types.h"
#include <string>
#include <vector>

// Solves a PDE using Krylov subspace methods on the GPU.
SolveOutput CudaSolveKrylov(const SolveInput& input);

#endif // CUDA_SOLVERS_KRYLOV_H
