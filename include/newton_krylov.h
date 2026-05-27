#ifndef NEWTON_KRYLOV_H
#define NEWTON_KRYLOV_H

#include <functional>
#include <string>
#include <vector>

#include "pde_types.h"

using NonlinearResidualFn =
    std::function<void(const std::vector<double>& u, std::vector<double>* residual)>;

struct NewtonKrylovResult {
  bool converged = false;
  int iterations = 0;
  std::string error;
  std::vector<double> solution;
};

NewtonKrylovResult NewtonKrylovSolve(const NonlinearResidualFn& residual,
                                     const std::vector<double>& x0,
                                     const NonlinearSolveConfig& config,
                                     int gmres_restart = 30);

#endif  // NEWTON_KRYLOV_H
