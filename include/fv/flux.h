#ifndef FV_FLUX_H
#define FV_FLUX_H

#include <string>

// Scalar flux f(u) and derivative df/du for Riemann solvers.
struct ScalarFlux {
  double f = 0.0;
  double df = 0.0;
};

// Burgers flux f(u) = 0.5*u^2
ScalarFlux BurgersFlux(double u);

// Evaluate flux from LaTeX expression (supports simple polynomials in u).
class FluxEvaluator {
public:
  static FluxEvaluator Parse(const std::string& latex, std::string* error);
  bool ok() const { return ok_; }
  const std::string& error() const { return error_; }
  ScalarFlux Eval(double u) const;

private:
  bool ok_ = false;
  std::string error_;
  enum class Kind { Burgers, Linear, Constant } kind_ = Kind::Burgers;
  double coeff_ = 0.5;
  double linear_ = 0.0;
  double constant_ = 0.0;
};

#endif  // FV_FLUX_H
