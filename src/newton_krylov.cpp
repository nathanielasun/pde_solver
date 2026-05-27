#include "newton_krylov.h"

#include "backends/cpu/cpu_utils.h"

NewtonKrylovResult NewtonKrylovSolve(const NonlinearResidualFn& residual,
                                     const std::vector<double>& x0,
                                     const NonlinearSolveConfig& config,
                                     int /*gmres_restart*/) {
  NewtonKrylovResult result;
  result.solution = x0;

  std::vector<double> r;
  residual(x0, &r);
  if (Norm2(r) < config.newton_tol) {
    result.converged = true;
    return result;
  }

  const double eps = 1e-7;
  for (int iter = 0; iter < config.max_newton_iter; ++iter) {
    result.iterations = iter + 1;
    residual(result.solution, &r);
    const double rnorm = Norm2(r);
    if (rnorm < config.newton_tol) {
      result.converged = true;
      return result;
    }

    std::vector<double> delta(result.solution.size(), 0.0);
    for (size_t i = 0; i < result.solution.size(); ++i) {
      std::vector<double> u_pert = result.solution;
      u_pert[i] += eps;
      std::vector<double> r_pert;
      residual(u_pert, &r_pert);
      double diag = (r_pert[i] - r[i]) / eps;
      if (std::abs(diag) < 1e-12) {
        diag = (diag >= 0.0) ? 1e-12 : -1e-12;
      }
      delta[i] = -r[i] / diag;
    }

    double alpha = 1.0;
    if (config.line_search) {
      for (int ls = 0; ls < 10; ++ls) {
        std::vector<double> trial = result.solution;
        for (size_t k = 0; k < trial.size(); ++k) {
          trial[k] += alpha * delta[k];
        }
        std::vector<double> r_trial;
        residual(trial, &r_trial);
        if (Norm2(r_trial) < rnorm) {
          result.solution = std::move(trial);
          break;
        }
        alpha *= 0.5;
      }
    } else {
      for (size_t k = 0; k < result.solution.size(); ++k) {
        result.solution[k] += alpha * delta[k];
      }
    }
  }

  result.error = "Newton iteration limit reached";
  return result;
}
