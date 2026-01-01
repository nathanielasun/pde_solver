#ifndef CUDA_SOLVE_H
#define CUDA_SOLVE_H

#include "pde_types.h"

// Checks if a CUDA-capable device is available.
bool CudaIsAvailable(std::string* note);

// Main dispatch function for the CUDA backend.
SolveOutput SolvePDECuda(const SolveInput& input);

#endif // CUDA_SOLVE_H
