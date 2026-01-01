#include "CudaSolve.h"
#include "solvers/relaxation.h"
#include "solvers/krylov.h"
#include "solvers/multigrid.h"
#include "utils/cuda_utils.h"

#include <string>

bool CudaIsAvailable(std::string* note) {
  int count = 0;
  cudaError_t err = cudaGetDeviceCount(&count);
  if (err != cudaSuccess || count == 0) {
    if (note) {
      if (err != cudaSuccess) {
        *note = "cudaGetDeviceCount failed: " + CudaErrorToString(err);
      } else {
        *note = "no CUDA devices found";
      }
    }
    return false;
  }
  cudaDeviceProp prop;
  if (cudaGetDeviceProperties(&prop, 0) == cudaSuccess) {
    if (note) {
      *note = std::string("device: ") + prop.name;
    }
  }
  return true;
}

SolveOutput SolvePDECuda(const SolveInput& input) {
    const Domain& d = input.domain;
    if (d.nx < 3 || d.ny < 3) {
        return {"grid must be at least 3x3", {}};
    }
    if (d.nz > 1) {
        return {"CUDA backend supports 2D domains only", {}};
    }
    if (d.xmax <= d.xmin || d.ymax <= d.ymin) {
        return {"domain bounds are invalid", {}};
    }

    std::string error;
    if (!CudaIsAvailable(&error)) {
        return {"CUDA unavailable: " + error, {}};
    }

    switch (input.solver.method) {
        case SolveMethod::Jacobi:
        case SolveMethod::GaussSeidel:
        case SolveMethod::SOR:
            return CudaSolveRelaxation(input);

        case SolveMethod::CG:
        case SolveMethod::BiCGStab:
        case SolveMethod::GMRES:
            return CudaSolveKrylov(input);

        case SolveMethod::MultigridVcycle:
            return CudaSolveMultigrid(input);
        
        default:
            return {"Selected solver method not implemented for CUDA backend", {}};
    }
}
