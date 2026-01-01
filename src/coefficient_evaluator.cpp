#include "coefficient_evaluator.h"

#include <string>
#include <utility>

namespace {
bool ParseCoefficient(const std::string& latex,
                      std::optional<ExpressionEvaluator>* out,
                      std::string* error,
                      const char* label) {
  if (latex.empty()) {
    return true;
  }
  ExpressionEvaluator evaluator = ExpressionEvaluator::ParseLatex(latex);
  if (!evaluator.ok()) {
    if (error) {
      *error = std::string("invalid coefficient for ") + label + ": " + evaluator.error();
    }
    return false;
  }
  if (out) {
    out->emplace(std::move(evaluator));
  }
  return true;
}
}  // namespace

CoefficientEvaluator BuildCoefficientEvaluator(const PDECoefficients& pde) {
  CoefficientEvaluator out;
  out.has_variable = HasVariableCoefficients(pde);

  if (!ParseCoefficient(pde.a_latex, &out.a, &out.error, "u_xx")) {
    out.ok = false;
    return out;
  }
  if (!ParseCoefficient(pde.b_latex, &out.b, &out.error, "u_yy")) {
    out.ok = false;
    return out;
  }
  if (!ParseCoefficient(pde.az_latex, &out.az, &out.error, "u_zz")) {
    out.ok = false;
    return out;
  }
  if (!ParseCoefficient(pde.c_latex, &out.c, &out.error, "u_x")) {
    out.ok = false;
    return out;
  }
  if (!ParseCoefficient(pde.d_latex, &out.d, &out.error, "u_y")) {
    out.ok = false;
    return out;
  }
  if (!ParseCoefficient(pde.dz_latex, &out.dz, &out.error, "u_z")) {
    out.ok = false;
    return out;
  }
  if (!ParseCoefficient(pde.e_latex, &out.e, &out.error, "u")) {
    out.ok = false;
    return out;
  }
  if (!ParseCoefficient(pde.ab_latex, &out.ab, &out.error, "u_xy")) {
    out.ok = false;
    return out;
  }
  if (!ParseCoefficient(pde.ac_latex, &out.ac, &out.error, "u_xz")) {
    out.ok = false;
    return out;
  }
  if (!ParseCoefficient(pde.bc_latex, &out.bc, &out.error, "u_yz")) {
    out.ok = false;
    return out;
  }
  if (!ParseCoefficient(pde.a3_latex, &out.a3, &out.error, "u_xxx")) {
    out.ok = false;
    return out;
  }
  if (!ParseCoefficient(pde.b3_latex, &out.b3, &out.error, "u_yyy")) {
    out.ok = false;
    return out;
  }
  if (!ParseCoefficient(pde.az3_latex, &out.az3, &out.error, "u_zzz")) {
    out.ok = false;
    return out;
  }
  if (!ParseCoefficient(pde.a4_latex, &out.a4, &out.error, "u_xxxx")) {
    out.ok = false;
    return out;
  }
  if (!ParseCoefficient(pde.b4_latex, &out.b4, &out.error, "u_yyyy")) {
    out.ok = false;
    return out;
  }
  if (!ParseCoefficient(pde.az4_latex, &out.az4, &out.error, "u_zzzz")) {
    out.ok = false;
    return out;
  }
  return out;
}

double EvalCoefficient(const std::optional<ExpressionEvaluator>& evaluator,
                       double constant, double x, double y, double z, double t) {
  if (!evaluator) {
    return constant;
  }
  return evaluator->Eval(x, y, z, t);
}
