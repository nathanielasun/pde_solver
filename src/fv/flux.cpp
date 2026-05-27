#include "fv/flux.h"

#include <algorithm>
#include <cmath>
#include <cctype>

namespace {
std::string NormalizeFluxLatex(std::string s) {
  for (char& c : s) {
    if (c == ' ') {
      c = '\0';
    }
  }
  s.erase(std::remove(s.begin(), s.end(), '\0'), s.end());
  return s;
}
}  // namespace

ScalarFlux BurgersFlux(double u) {
  ScalarFlux out;
  out.f = 0.5 * u * u;
  out.df = u;
  return out;
}

FluxEvaluator FluxEvaluator::Parse(const std::string& latex, std::string* error) {
  FluxEvaluator eval;
  const std::string norm = NormalizeFluxLatex(latex);
  if (norm.empty() || norm == "0.5*u^2" || norm == "0.5u^2" || norm == "u^2/2") {
    eval.ok_ = true;
    eval.kind_ = Kind::Burgers;
    eval.coeff_ = 0.5;
    return eval;
  }
  if (norm == "u" || norm == "1*u") {
    eval.ok_ = true;
    eval.kind_ = Kind::Linear;
    eval.linear_ = 1.0;
    return eval;
  }
  eval.error_ = "unsupported flux expression: " + latex;
  if (error) {
    *error = eval.error_;
  }
  return eval;
}

ScalarFlux FluxEvaluator::Eval(double u) const {
  switch (kind_) {
    case Kind::Burgers:
      return BurgersFlux(u);
    case Kind::Linear:
      return {linear_ * u, linear_};
    case Kind::Constant:
      return {constant_, 0.0};
  }
  return BurgersFlux(u);
}
