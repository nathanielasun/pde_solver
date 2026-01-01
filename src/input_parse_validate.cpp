#include "input_parse.h"

#include <algorithm>
#include <cmath>
#include <string>

#include "expression_eval.h"

namespace {
bool EvalBCExpr(const BoundaryCondition::Expression& expr, double x, double y, double z, std::string* error) {
  // Try LaTeX expression if provided
  if (!expr.latex.empty()) {
    ExpressionEvaluator eval = ExpressionEvaluator::ParseLatex(expr.latex);
    if (!eval.ok()) {
      if (error) *error = eval.error();
      return false;
    }
    const double v = eval.Eval(x, y, z, 0.0);
    if (!std::isfinite(v)) {
      if (error) *error = "non-finite value";
      return false;
    }
    return true;
  }
  // Linear fallback
  const double v = expr.constant + expr.x * x + expr.y * y + expr.z * z;
  if (!std::isfinite(v)) {
    if (error) *error = "non-finite value";
    return false;
  }
  return true;
}

bool ValidateFace(const BoundaryCondition& bc, double x, double y, double z, double h, std::string* error) {
  switch (bc.kind) {
    case BCKind::Dirichlet:
      return EvalBCExpr(bc.value, x, y, z, error);
    case BCKind::Neumann:
      return EvalBCExpr(bc.value, x, y, z, error);
    case BCKind::Robin: {
      if (!EvalBCExpr(bc.alpha, x, y, z, error)) return false;
      if (!EvalBCExpr(bc.beta, x, y, z, error)) return false;
      if (!EvalBCExpr(bc.gamma, x, y, z, error)) return false;
      // Check denom != 0 for Robin approximation (beta/h + alpha)
      const double denom = bc.alpha.constant + bc.beta.constant / std::max(1e-12, h);
      if (std::abs(denom) < 1e-12) {
        if (error) *error = "robin denominator near zero";
        return false;
      }
      return true;
    }
    default:
      if (error) *error = "unsupported boundary kind";
      return false;
  }
}
}  // namespace

ParseResult ValidateBoundaryConditions(const BoundarySet& bc, const Domain& d) {
  ParseResult result;
  result.ok = false;
  std::string err;

  // Characteristic spacing
  const double dx = (d.xmax - d.xmin) / static_cast<double>(std::max(1, d.nx - 1));
  const double dy = (d.ymax - d.ymin) / static_cast<double>(std::max(1, d.ny - 1));
  const double dz = (d.nz > 1)
      ? (d.zmax - d.zmin) / static_cast<double>(std::max(1, d.nz - 1))
      : std::max(dx, dy);
  const double h = std::max({dx, dy, dz, 1e-12});

  auto check_face = [&](const BoundaryCondition& face_bc, double x, double y, double z,
                        const char* label) -> bool {
    if (!ValidateFace(face_bc, x, y, z, h, &err)) {
      result.error = std::string(label) + ": " + err;
      return false;
    }
    return true;
  };

  // Sample corners for stability; this is not exhaustive but catches common issues.
  const double xs[2] = {d.xmin, d.xmax};
  const double ys[2] = {d.ymin, d.ymax};
  const double zs[2] = {d.zmin, d.zmax};

  for (double y : ys) {
    for (double z : zs) {
      if (!check_face(bc.left, d.xmin, y, z, "left")) return result;
      if (!check_face(bc.right, d.xmax, y, z, "right")) return result;
    }
  }
  for (double x : xs) {
    for (double z : zs) {
      if (!check_face(bc.bottom, x, d.ymin, z, "bottom")) return result;
      if (!check_face(bc.top, x, d.ymax, z, "top")) return result;
    }
  }
  for (double x : xs) {
    for (double y : ys) {
      if (!check_face(bc.front, x, y, d.zmin, "front")) return result;
      if (!check_face(bc.back, x, y, d.zmax, "back")) return result;
    }
  }

  result.ok = true;
  return result;
}


