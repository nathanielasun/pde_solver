#ifndef CUDA_SOLVER_H
#define CUDA_SOLVER_H

#include <string>

#include "pde_types.h"

bool CudaIsAvailable(std::string* note);
SolveOutput SolvePDECuda(const SolveInput& input);

#endif
