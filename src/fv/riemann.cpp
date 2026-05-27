#include "fv/riemann.h"

#include <algorithm>
#include <cmath>

double RiemannFlux(double u_left, double u_right, const FluxEvaluator& flux,
                   ConservationLawConfig::RiemannSolver solver) {
  const ScalarFlux fl = flux.Eval(u_left);
  const ScalarFlux fr = flux.Eval(u_right);

  switch (solver) {
    case ConservationLawConfig::RiemannSolver::LaxFriedrichs: {
      const double alpha = std::max(std::abs(fl.df), std::abs(fr.df));
      return 0.5 * (fl.f + fr.f) - 0.5 * alpha * (u_right - u_left);
    }
    case ConservationLawConfig::RiemannSolver::HLL: {
      const double sl = std::min(fl.df, 0.0);
      const double sr = std::max(fr.df, 0.0);
      if (sl >= 0.0) {
        return fl.f;
      }
      if (sr <= 0.0) {
        return fr.f;
      }
      return (sr * fl.f - sl * fr.f + sl * sr * (u_right - u_left)) / (sr - sl);
    }
  }
  const double alpha = std::max(std::abs(fl.df), std::abs(fr.df));
  return 0.5 * (fl.f + fr.f) - 0.5 * alpha * (u_right - u_left);
}
