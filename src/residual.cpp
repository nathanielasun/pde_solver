#include "residual.h"

#include <algorithm>
#include <cmath>
#include <optional>
#include <utility>

#include "coefficient_evaluator.h"
#include "expression_eval.h"
#include "finite_differences.h"
#include "shape_utils.h"

namespace {
int Index(int i, int j, int nx) {
  return j * nx + i;
}

int Index3D(int i, int j, int k, int nx, int ny) {
  return (k * ny + j) * nx + i;
}

double EvalExpr(const BoundaryCondition::Expression& expr, double x, double y) {
  return expr.constant + expr.x * x + expr.y * y + expr.z * 0.0;
}

double EvalExpr(const BoundaryCondition::Expression& expr, double x, double y, double z) {
  return expr.constant + expr.x * x + expr.y * y + expr.z * z;
}

double EvalNonlinear(const std::vector<NonlinearTerm>& terms, double u) {
  double value = 0.0;
  for (const auto& term : terms) {
    switch (term.kind) {
      case NonlinearKind::Power:
        value += term.coeff * std::pow(u, static_cast<double>(term.power));
        break;
      case NonlinearKind::Sin:
        value += term.coeff * std::sin(u);
        break;
      case NonlinearKind::Cos:
        value += term.coeff * std::cos(u);
        break;
      case NonlinearKind::Exp:
        value += term.coeff * std::exp(u);
        break;
      case NonlinearKind::Abs:
        value += term.coeff * std::abs(u);
        break;
    }
  }
  return value;
}

struct KahanSum {
  double sum = 0.0;
  double c = 0.0;
  void Add(double value) {
    const double y = value - c;
    const double t = sum + y;
    c = (t - sum) - y;
    sum = t;
  }
};

bool BuildIntegralWeights2D(const SolveInput& input,
                            const Domain& d,
                            double dx,
                            double dy,
                            const std::vector<unsigned char>* active,
                            std::vector<double>* weights,
                            std::string* error) {
  if (input.integrals.empty()) {
    if (weights) {
      weights->clear();
    }
    return true;
  }
  if (!weights) {
    return false;
  }
  const int nx = d.nx;
  const int ny = d.ny;
  weights->assign(static_cast<size_t>(nx * ny), 0.0);
  auto is_active = [&](int i, int j) -> bool {
    if (!active || active->empty()) {
      return true;
    }
    return (*active)[static_cast<size_t>(Index(i, j, nx))] != 0;
  };
  for (const auto& term : input.integrals) {
    if (std::abs(term.coeff) < 1e-12) {
      continue;
    }
    if (term.kernel_latex.empty()) {
      for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
          if (!is_active(i, j)) {
            continue;
          }
          (*weights)[static_cast<size_t>(Index(i, j, nx))] += term.coeff;
        }
      }
      continue;
    }
    ExpressionEvaluator kernel = ExpressionEvaluator::ParseLatex(term.kernel_latex);
    if (!kernel.ok()) {
      if (error) {
        *error = "invalid kernel expression: " + kernel.error();
      }
      return false;
    }
    for (int j = 0; j < ny; ++j) {
      const double y = d.ymin + j * dy;
      for (int i = 0; i < nx; ++i) {
        if (!is_active(i, j)) {
          continue;
        }
        const double x = d.xmin + i * dx;
        (*weights)[static_cast<size_t>(Index(i, j, nx))] += term.coeff * kernel.Eval(x, y);
      }
    }
  }
  return true;
}

double ComputeIntegralValue2D(const std::vector<double>& grid,
                              int nx, int ny,
                              const std::vector<unsigned char>* active,
                              double dx, double dy) {
  KahanSum sum;
  for (int j = 0; j < ny; ++j) {
    for (int i = 0; i < nx; ++i) {
      const int idx = Index(i, j, nx);
      if (active && !active->empty() && (*active)[static_cast<size_t>(idx)] == 0) {
        continue;
      }
      sum.Add(grid[static_cast<size_t>(idx)]);
    }
  }
  return sum.sum * dx * dy;
}

void AccumulateResidual(double r, KahanSum* sum_sq, double* max_abs) {
  if (sum_sq) {
    sum_sq->Add(r * r);
  }
  if (max_abs) {
    *max_abs = std::max(*max_abs, std::abs(r));
  }
}
}

bool ComputeResidualNorms(const SolveInput& input,
                          const std::vector<double>& grid,
                          ResidualNorms* out,
                          std::string* error) {
  if (!out) {
    return false;
  }
  *out = {};
  const Domain& d = input.domain;
  if (grid.empty()) {
    if (error) {
      *error = "empty grid";
    }
    return false;
  }
  if (std::abs(input.pde.ut) > 1e-12 || std::abs(input.pde.utt) > 1e-12) {
    if (error) {
      *error = "residual norms for time-dependent solves are not implemented yet";
    }
    return false;
  }
  const bool has_mixed = HasMixedDerivatives(input.pde);
  const bool has_high_order = HasHigherOrderDerivatives(input.pde);
  if (d.coord_system != CoordinateSystem::Cartesian && (has_mixed || has_high_order)) {
    if (error) {
      *error = "mixed or higher-order derivatives require Cartesian coordinates";
    }
    return false;
  }

  CoefficientEvaluator coeff_eval = BuildCoefficientEvaluator(input.pde);
  if (!coeff_eval.ok) {
    if (error) {
      *error = coeff_eval.error;
    }
    return false;
  }
  const bool has_var_coeff = coeff_eval.has_variable;

  const bool has_rhs_expr = !input.pde.rhs_latex.empty();
  std::optional<ExpressionEvaluator> rhs_eval;
  if (has_rhs_expr) {
    ExpressionEvaluator test = ExpressionEvaluator::ParseLatex(input.pde.rhs_latex);
    if (!test.ok()) {
      if (error) {
        *error = "invalid RHS expression: " + test.error();
      }
      return false;
    }
    rhs_eval.emplace(std::move(test));
  }
  const double f_const = input.pde.f;
  auto eval_rhs_2d = [&](double x, double y) -> double {
    if (rhs_eval) {
      return rhs_eval->Eval(x, y, 0.0, 0.0);
    }
    return f_const;
  };
  auto eval_rhs_3d = [&](double x, double y, double z) -> double {
    if (rhs_eval) {
      return rhs_eval->Eval(x, y, z, 0.0);
    }
    return f_const;
  };

  // 2D steady residual (supports shape/integrals/nonlinear with current feature set).
  if (d.nz <= 1) {
    const int nx = d.nx;
    const int ny = d.ny;
    if (static_cast<int>(grid.size()) != nx * ny) {
      if (error) {
        *error = "grid size mismatch";
      }
      return false;
    }
    const double dx = (d.xmax - d.xmin) / static_cast<double>(nx - 1);
    const double dy = (d.ymax - d.ymin) / static_cast<double>(ny - 1);

    std::vector<unsigned char> active;
  if (HasImplicitShape(input)) {
    const bool use_mask = HasShapeMask(input.shape_mask);
    std::optional<ExpressionEvaluator> evaluator;
    if (!use_mask) {
      ExpressionEvaluator parsed = ExpressionEvaluator::ParseLatex(input.domain_shape);
      if (!parsed.ok()) {
        if (error) {
          *error = "invalid domain shape: " + parsed.error();
        }
        return false;
      }
      evaluator.emplace(std::move(parsed));
    }
    auto eval_phi = [&](double x, double y) -> double {
      double tx = x;
      double ty = y;
      double tz = 0.0;
        ApplyShapeTransform(input.shape_transform, x, y, 0.0, &tx, &ty, &tz);
      if (use_mask) {
        return SampleShapeMaskPhi(input.shape_mask, tx, ty, tz,
                                  input.shape_mask_threshold, input.shape_mask_invert);
      }
      return evaluator->Eval(tx, ty, 0.0);
    };
      active.assign(static_cast<size_t>(nx * ny), 0);
      for (int j = 0; j < ny; ++j) {
        const double y = d.ymin + j * dy;
        for (int i = 0; i < nx; ++i) {
          const double x = d.xmin + i * dx;
          active[static_cast<size_t>(Index(i, j, nx))] = (eval_phi(x, y) <= 0.0) ? 1 : 0;
        }
      }
    }
    auto is_active = [&](int i, int j) -> bool {
      if (active.empty()) {
        return true;
      }
      return active[static_cast<size_t>(Index(i, j, nx))] != 0;
    };

    std::vector<double> weights;
    std::string werr;
    if (!BuildIntegralWeights2D(input, d, dx, dy, active.empty() ? nullptr : &active, &weights, &werr)) {
      if (error) {
        *error = werr.empty() ? "failed to build integral weights" : werr;
      }
      return false;
    }
    const bool has_integrals = !input.integrals.empty();
    const bool has_nonlinear = !input.nonlinear.empty();
    const double integral_value = has_integrals
        ? ComputeIntegralValue2D(grid, nx, ny, active.empty() ? nullptr : &active, dx, dy)
        : 0.0;

    const double a = input.pde.a;
    const double b = input.pde.b;
    const double c = input.pde.c;
    const double dcoef = input.pde.d;
    const double e = input.pde.e;
    const double ab = input.pde.ab;
    const double a3 = input.pde.a3;
    const double b3 = input.pde.b3;
    const double a4 = input.pde.a4;
    const double b4 = input.pde.b4;

    const double ax_const = a / (dx * dx);
    const double by_const = b / (dy * dy);
    const double cx_const = c / (2.0 * dx);
    const double dyc_const = dcoef / (2.0 * dy);
    const double center_const = -2.0 * ax_const - 2.0 * by_const + e;
    const bool has_fourth =
        std::abs(a4) > 1e-12 || std::abs(b4) > 1e-12 ||
        !input.pde.a4_latex.empty() || !input.pde.b4_latex.empty();
    if (!has_var_coeff && !has_fourth && std::abs(center_const) < 1e-12) {
      if (error) {
        *error = "degenerate PDE center coefficient";
      }
      return false;
    }

    KahanSum sum_sq;
    double max_abs = 0.0;
    for (int j = 0; j < ny; ++j) {
      const double y = d.ymin + j * dy;
      const double* row = grid.data() + static_cast<size_t>(Index(0, j, nx));
      for (int i = 0; i < nx; ++i) {
        if (!is_active(i, j)) {
          continue;
        }
        const double x = d.xmin + i * dx;
        const int idx = Index(i, j, nx);
        const double u = grid[static_cast<size_t>(idx)];

        const bool boundary = (i == 0 || j == 0 || i == nx - 1 || j == ny - 1);
        if (boundary) {
          // Boundary residuals: match the same discrete closures used by the solver.
          if (i == 0) {
            if (input.bc.left.kind == BCKind::Dirichlet) {
              AccumulateResidual(u - EvalExpr(input.bc.left.value, d.xmin, y), &sum_sq, &max_abs);
            } else if (input.bc.left.kind == BCKind::Neumann) {
              const double g = EvalExpr(input.bc.left.value, d.xmin, y);
              const double target = grid[static_cast<size_t>(Index(1, j, nx))] - dx * g;
              AccumulateResidual(u - target, &sum_sq, &max_abs);
            } else {
              const double alpha = EvalExpr(input.bc.left.alpha, d.xmin, y);
              const double beta = EvalExpr(input.bc.left.beta, d.xmin, y);
              const double gamma = EvalExpr(input.bc.left.gamma, d.xmin, y);
              const double denom = alpha + beta / dx;
              if (std::abs(denom) > 1e-12) {
                const double target =
                    (gamma + (beta / dx) * grid[static_cast<size_t>(Index(1, j, nx))]) / denom;
                AccumulateResidual(u - target, &sum_sq, &max_abs);
              }
            }
            continue;
          }
          if (i == nx - 1) {
            if (input.bc.right.kind == BCKind::Dirichlet) {
              AccumulateResidual(u - EvalExpr(input.bc.right.value, d.xmax, y), &sum_sq, &max_abs);
            } else if (input.bc.right.kind == BCKind::Neumann) {
              const double g = EvalExpr(input.bc.right.value, d.xmax, y);
              const double target = grid[static_cast<size_t>(Index(nx - 2, j, nx))] + dx * g;
              AccumulateResidual(u - target, &sum_sq, &max_abs);
            } else {
              const double alpha = EvalExpr(input.bc.right.alpha, d.xmax, y);
              const double beta = EvalExpr(input.bc.right.beta, d.xmax, y);
              const double gamma = EvalExpr(input.bc.right.gamma, d.xmax, y);
              const double denom = alpha + beta / dx;
              if (std::abs(denom) > 1e-12) {
                const double target =
                    (gamma + (beta / dx) * grid[static_cast<size_t>(Index(nx - 2, j, nx))]) / denom;
                AccumulateResidual(u - target, &sum_sq, &max_abs);
              }
            }
            continue;
          }
          if (j == 0) {
            if (input.bc.bottom.kind == BCKind::Dirichlet) {
              AccumulateResidual(u - EvalExpr(input.bc.bottom.value, x, d.ymin), &sum_sq, &max_abs);
            } else if (input.bc.bottom.kind == BCKind::Neumann) {
              const double g = EvalExpr(input.bc.bottom.value, x, d.ymin);
              const double target = grid[static_cast<size_t>(Index(i, 1, nx))] - dy * g;
              AccumulateResidual(u - target, &sum_sq, &max_abs);
            } else {
              const double alpha = EvalExpr(input.bc.bottom.alpha, x, d.ymin);
              const double beta = EvalExpr(input.bc.bottom.beta, x, d.ymin);
              const double gamma = EvalExpr(input.bc.bottom.gamma, x, d.ymin);
              const double denom = alpha + beta / dy;
              if (std::abs(denom) > 1e-12) {
                const double target =
                    (gamma + (beta / dy) * grid[static_cast<size_t>(Index(i, 1, nx))]) / denom;
                AccumulateResidual(u - target, &sum_sq, &max_abs);
              }
            }
            continue;
          }
          if (j == ny - 1) {
            if (input.bc.top.kind == BCKind::Dirichlet) {
              AccumulateResidual(u - EvalExpr(input.bc.top.value, x, d.ymax), &sum_sq, &max_abs);
            } else if (input.bc.top.kind == BCKind::Neumann) {
              const double g = EvalExpr(input.bc.top.value, x, d.ymax);
              const double target = grid[static_cast<size_t>(Index(i, ny - 2, nx))] + dy * g;
              AccumulateResidual(u - target, &sum_sq, &max_abs);
            } else {
              const double alpha = EvalExpr(input.bc.top.alpha, x, d.ymax);
              const double beta = EvalExpr(input.bc.top.beta, x, d.ymax);
              const double gamma = EvalExpr(input.bc.top.gamma, x, d.ymax);
              const double denom = alpha + beta / dy;
              if (std::abs(denom) > 1e-12) {
                const double target =
                    (gamma + (beta / dy) * grid[static_cast<size_t>(Index(i, ny - 2, nx))]) / denom;
                AccumulateResidual(u - target, &sum_sq, &max_abs);
              }
            }
            continue;
          }
        }

        // Interior residual: center*u + neighbor terms + (f + integral + nonlinear) == 0.
        const double u_left = grid[static_cast<size_t>(Index(i - 1, j, nx))];
        const double u_right = grid[static_cast<size_t>(Index(i + 1, j, nx))];
        const double u_down = grid[static_cast<size_t>(Index(i, j - 1, nx))];
        const double u_up = grid[static_cast<size_t>(Index(i, j + 1, nx))];
        const double f_val = eval_rhs_2d(x, y);
        const double a_val = has_var_coeff ? EvalCoefficient(coeff_eval.a, a, x, y, 0.0, 0.0) : a;
        const double b_val = has_var_coeff ? EvalCoefficient(coeff_eval.b, b, x, y, 0.0, 0.0) : b;
        const double c_val = has_var_coeff ? EvalCoefficient(coeff_eval.c, c, x, y, 0.0, 0.0) : c;
        const double d_val = has_var_coeff ? EvalCoefficient(coeff_eval.d, dcoef, x, y, 0.0, 0.0) : dcoef;
        const double e_val = has_var_coeff ? EvalCoefficient(coeff_eval.e, e, x, y, 0.0, 0.0) : e;
        const double ab_val = has_var_coeff ? EvalCoefficient(coeff_eval.ab, ab, x, y, 0.0, 0.0) : ab;
        const double a3_val = has_var_coeff ? EvalCoefficient(coeff_eval.a3, a3, x, y, 0.0, 0.0) : a3;
        const double b3_val = has_var_coeff ? EvalCoefficient(coeff_eval.b3, b3, x, y, 0.0, 0.0) : b3;
        const double a4_val = has_var_coeff ? EvalCoefficient(coeff_eval.a4, a4, x, y, 0.0, 0.0) : a4;
        const double b4_val = has_var_coeff ? EvalCoefficient(coeff_eval.b4, b4, x, y, 0.0, 0.0) : b4;
        const double ax = has_var_coeff ? a_val / (dx * dx) : ax_const;
        const double by = has_var_coeff ? b_val / (dy * dy) : by_const;
        const double cx = has_var_coeff ? c_val / (2.0 * dx) : cx_const;
        const double dyc = has_var_coeff ? d_val / (2.0 * dy) : dyc_const;
        double center = has_var_coeff ? (-2.0 * ax - 2.0 * by + e_val) : center_const;
        double c4x = 0.0;
        double c4y = 0.0;
        if (std::abs(a4_val) > 1e-12) {
          c4x = FourthDerivativeCenterCoeff(i, nx, dx);
          center += a4_val * c4x;
        }
        if (std::abs(b4_val) > 1e-12) {
          c4y = FourthDerivativeCenterCoeff(j, ny, dy);
          center += b4_val * c4y;
        }
        if (std::abs(center) < 1e-12) {
          if (error) {
            *error = "degenerate PDE center coefficient";
          }
          return false;
        }
        const double integral_term =
            has_integrals ? weights[static_cast<size_t>(idx)] * integral_value : 0.0;
        const double nonlinear_term =
            has_nonlinear ? EvalNonlinear(input.nonlinear, u) : 0.0;

        const double base_contrib =
          (ax + cx) * u_right +
          (ax - cx) * u_left +
          (by + dyc) * u_up +
          (by - dyc) * u_down;

        double mixed_contrib = 0.0;
        if (std::abs(ab_val) > 1e-12) {
          const double u_pp = grid[static_cast<size_t>(Index(i + 1, j + 1, nx))];
          const double u_pm = grid[static_cast<size_t>(Index(i + 1, j - 1, nx))];
          const double u_mp = grid[static_cast<size_t>(Index(i - 1, j + 1, nx))];
          const double u_mm = grid[static_cast<size_t>(Index(i - 1, j - 1, nx))];
          mixed_contrib = ab_val * ComputeMixedDerivativeXY(u_pp, u_pm, u_mp, u_mm, dx, dy);
        }

        double third_contrib = 0.0;
        if (std::abs(a3_val) > 1e-12) {
          third_contrib += a3_val * ComputeThirdDerivativeX(row, i, nx, dx);
        }
        if (std::abs(b3_val) > 1e-12) {
          third_contrib += b3_val * ComputeThirdDerivativeY(grid.data(), i, j, nx, ny, dy);
        }

        double fourth_contrib = 0.0;
        if (std::abs(a4_val) > 1e-12) {
          const double u_xxxx = ComputeFourthDerivativeX(row, i, nx, dx);
          fourth_contrib += a4_val * (u_xxxx - c4x * u);
        }
        if (std::abs(b4_val) > 1e-12) {
          const double u_yyyy = ComputeFourthDerivativeY(grid.data(), i, j, nx, ny, dy);
          fourth_contrib += b4_val * (u_yyyy - c4y * u);
        }

        const double r = center * u +
          base_contrib + mixed_contrib + third_contrib + fourth_contrib +
          (f_val + integral_term + nonlinear_term);
        AccumulateResidual(r, &sum_sq, &max_abs);
      }
    }
    out->l2 = std::sqrt(std::max(0.0, sum_sq.sum));
    out->linf = max_abs;
    return true;
  }

  // 3D steady residual.
  const int nx = d.nx;
  const int ny = d.ny;
  const int nz = d.nz;
  if (static_cast<int>(grid.size()) != nx * ny * nz) {
    if (error) {
      *error = "grid size mismatch";
    }
    return false;
  }
  const double dx = (d.xmax - d.xmin) / static_cast<double>(nx - 1);
  const double dy = (d.ymax - d.ymin) / static_cast<double>(ny - 1);
  const double dz = (d.zmax - d.zmin) / static_cast<double>(nz - 1);

  std::vector<unsigned char> active;
  if (HasImplicitShape(input)) {
    const bool use_mask = HasShapeMask(input.shape_mask);
    std::optional<ExpressionEvaluator> evaluator;
    if (!use_mask) {
      ExpressionEvaluator parsed = ExpressionEvaluator::ParseLatex(input.domain_shape);
      if (!parsed.ok()) {
        if (error) {
          *error = "invalid domain shape: " + parsed.error();
        }
        return false;
      }
      evaluator.emplace(std::move(parsed));
    }
    auto eval_phi = [&](double x, double y, double z) -> double {
      double tx = x;
      double ty = y;
      double tz = z;
      ApplyShapeTransform(input.shape_transform, x, y, z, &tx, &ty, &tz);
      if (use_mask) {
        return SampleShapeMaskPhi(input.shape_mask, tx, ty, tz,
                                  input.shape_mask_threshold, input.shape_mask_invert);
      }
      return evaluator->Eval(tx, ty, tz);
    };
    active.assign(static_cast<size_t>(nx * ny * nz), 0);
    for (int k = 0; k < nz; ++k) {
      const double z = d.zmin + k * dz;
      for (int j = 0; j < ny; ++j) {
        const double y = d.ymin + j * dy;
        for (int i = 0; i < nx; ++i) {
          const double x = d.xmin + i * dx;
          active[static_cast<size_t>(Index3D(i, j, k, nx, ny))] =
              (eval_phi(x, y, z) <= 0.0) ? 1 : 0;
        }
      }
    }
  }
  auto is_active = [&](int i, int j, int k) -> bool {
    if (active.empty()) {
      return true;
    }
    return active[static_cast<size_t>(Index3D(i, j, k, nx, ny))] != 0;
  };

  const double a = input.pde.a;
  const double b = input.pde.b;
  const double az = input.pde.az;
  const double c = input.pde.c;
  const double dcoef = input.pde.d;
  const double dzcoef = input.pde.dz;
  const double e = input.pde.e;
  const double ab = input.pde.ab;
  const double ac = input.pde.ac;
  const double bc = input.pde.bc;
  const double a3 = input.pde.a3;
  const double b3 = input.pde.b3;
  const double az3 = input.pde.az3;
  const double a4 = input.pde.a4;
  const double b4 = input.pde.b4;
  const double az4 = input.pde.az4;

  const double ax_const = a / (dx * dx);
  const double by_const = b / (dy * dy);
  const double cz_const = az / (dz * dz);
  const double cx_const = c / (2.0 * dx);
  const double dyc_const = dcoef / (2.0 * dy);
  const double dzc_const = dzcoef / (2.0 * dz);
  const double center_const = -2.0 * ax_const - 2.0 * by_const - 2.0 * cz_const + e;
  const bool has_fourth =
      std::abs(a4) > 1e-12 || std::abs(b4) > 1e-12 || std::abs(az4) > 1e-12 ||
      !input.pde.a4_latex.empty() || !input.pde.b4_latex.empty() ||
      !input.pde.az4_latex.empty();
  if (!has_var_coeff && !has_fourth && std::abs(center_const) < 1e-12) {
    if (error) {
      *error = "degenerate PDE center coefficient";
    }
    return false;
  }

  KahanSum sum_sq;
  double max_abs = 0.0;
  for (int k = 1; k < nz - 1; ++k) {
    const double* slice = grid.data() + static_cast<size_t>(Index3D(0, 0, k, nx, ny));
    for (int j = 1; j < ny - 1; ++j) {
      const double* row = slice + static_cast<size_t>(Index(0, j, nx));
      for (int i = 1; i < nx - 1; ++i) {
        const int idx = Index3D(i, j, k, nx, ny);
        if (!is_active(i, j, k)) {
          continue;
        }
        const double u = grid[static_cast<size_t>(idx)];
        const double u_left = grid[static_cast<size_t>(Index3D(i - 1, j, k, nx, ny))];
        const double u_right = grid[static_cast<size_t>(Index3D(i + 1, j, k, nx, ny))];
        const double u_down = grid[static_cast<size_t>(Index3D(i, j - 1, k, nx, ny))];
        const double u_up = grid[static_cast<size_t>(Index3D(i, j + 1, k, nx, ny))];
        const double u_back = grid[static_cast<size_t>(Index3D(i, j, k - 1, nx, ny))];
        const double u_front = grid[static_cast<size_t>(Index3D(i, j, k + 1, nx, ny))];
        const double x = d.xmin + i * dx;
        const double y = d.ymin + j * dy;
        const double z = d.zmin + k * dz;
        const double f_val = eval_rhs_3d(x, y, z);
        const double a_val = has_var_coeff ? EvalCoefficient(coeff_eval.a, a, x, y, z, 0.0) : a;
        const double b_val = has_var_coeff ? EvalCoefficient(coeff_eval.b, b, x, y, z, 0.0) : b;
        const double az_val = has_var_coeff ? EvalCoefficient(coeff_eval.az, az, x, y, z, 0.0) : az;
        const double c_val = has_var_coeff ? EvalCoefficient(coeff_eval.c, c, x, y, z, 0.0) : c;
        const double d_val = has_var_coeff ? EvalCoefficient(coeff_eval.d, dcoef, x, y, z, 0.0) : dcoef;
        const double dz_val = has_var_coeff ? EvalCoefficient(coeff_eval.dz, dzcoef, x, y, z, 0.0) : dzcoef;
        const double e_val = has_var_coeff ? EvalCoefficient(coeff_eval.e, e, x, y, z, 0.0) : e;
        const double ab_val = has_var_coeff ? EvalCoefficient(coeff_eval.ab, ab, x, y, z, 0.0) : ab;
        const double ac_val = has_var_coeff ? EvalCoefficient(coeff_eval.ac, ac, x, y, z, 0.0) : ac;
        const double bc_val = has_var_coeff ? EvalCoefficient(coeff_eval.bc, bc, x, y, z, 0.0) : bc;
        const double a3_val = has_var_coeff ? EvalCoefficient(coeff_eval.a3, a3, x, y, z, 0.0) : a3;
        const double b3_val = has_var_coeff ? EvalCoefficient(coeff_eval.b3, b3, x, y, z, 0.0) : b3;
        const double az3_val = has_var_coeff ? EvalCoefficient(coeff_eval.az3, az3, x, y, z, 0.0) : az3;
        const double a4_val = has_var_coeff ? EvalCoefficient(coeff_eval.a4, a4, x, y, z, 0.0) : a4;
        const double b4_val = has_var_coeff ? EvalCoefficient(coeff_eval.b4, b4, x, y, z, 0.0) : b4;
        const double az4_val = has_var_coeff ? EvalCoefficient(coeff_eval.az4, az4, x, y, z, 0.0) : az4;
        const double ax = has_var_coeff ? a_val / (dx * dx) : ax_const;
        const double by = has_var_coeff ? b_val / (dy * dy) : by_const;
        const double cz = has_var_coeff ? az_val / (dz * dz) : cz_const;
        const double cx = has_var_coeff ? c_val / (2.0 * dx) : cx_const;
        const double dyc = has_var_coeff ? d_val / (2.0 * dy) : dyc_const;
        const double dzc = has_var_coeff ? dz_val / (2.0 * dz) : dzc_const;
        double center = has_var_coeff
            ? (-2.0 * ax - 2.0 * by - 2.0 * cz + e_val)
            : center_const;
        double c4x = 0.0;
        double c4y = 0.0;
        double c4z = 0.0;
        if (std::abs(a4_val) > 1e-12) {
          c4x = FourthDerivativeCenterCoeff(i, nx, dx);
          center += a4_val * c4x;
        }
        if (std::abs(b4_val) > 1e-12) {
          c4y = FourthDerivativeCenterCoeff(j, ny, dy);
          center += b4_val * c4y;
        }
        if (std::abs(az4_val) > 1e-12) {
          c4z = FourthDerivativeCenterCoeff(k, nz, dz);
          center += az4_val * c4z;
        }
        if (std::abs(center) < 1e-12) {
          if (error) {
            *error = "degenerate PDE center coefficient";
          }
          return false;
        }
        const double base_contrib =
          (ax + cx) * u_right +
          (ax - cx) * u_left +
          (by + dyc) * u_up +
          (by - dyc) * u_down +
          (cz + dzc) * u_front +
          (cz - dzc) * u_back;

        double mixed_contrib = 0.0;
        if (std::abs(ab_val) > 1e-12) {
          const double u_pp = grid[static_cast<size_t>(Index3D(i + 1, j + 1, k, nx, ny))];
          const double u_pm = grid[static_cast<size_t>(Index3D(i + 1, j - 1, k, nx, ny))];
          const double u_mp = grid[static_cast<size_t>(Index3D(i - 1, j + 1, k, nx, ny))];
          const double u_mm = grid[static_cast<size_t>(Index3D(i - 1, j - 1, k, nx, ny))];
          mixed_contrib += ab_val * ComputeMixedDerivativeXY(u_pp, u_pm, u_mp, u_mm, dx, dy);
        }
        if (std::abs(ac_val) > 1e-12) {
          const double u_pp = grid[static_cast<size_t>(Index3D(i + 1, j, k + 1, nx, ny))];
          const double u_pm = grid[static_cast<size_t>(Index3D(i + 1, j, k - 1, nx, ny))];
          const double u_mp = grid[static_cast<size_t>(Index3D(i - 1, j, k + 1, nx, ny))];
          const double u_mm = grid[static_cast<size_t>(Index3D(i - 1, j, k - 1, nx, ny))];
          mixed_contrib += ac_val * ComputeMixedDerivativeXZ(u_pp, u_pm, u_mp, u_mm, dx, dz);
        }
        if (std::abs(bc_val) > 1e-12) {
          const double u_pp = grid[static_cast<size_t>(Index3D(i, j + 1, k + 1, nx, ny))];
          const double u_pm = grid[static_cast<size_t>(Index3D(i, j + 1, k - 1, nx, ny))];
          const double u_mp = grid[static_cast<size_t>(Index3D(i, j - 1, k + 1, nx, ny))];
          const double u_mm = grid[static_cast<size_t>(Index3D(i, j - 1, k - 1, nx, ny))];
          mixed_contrib += bc_val * ComputeMixedDerivativeYZ(u_pp, u_pm, u_mp, u_mm, dy, dz);
        }

        double third_contrib = 0.0;
        if (std::abs(a3_val) > 1e-12) {
          third_contrib += a3_val * ComputeThirdDerivativeX(row, i, nx, dx);
        }
        if (std::abs(b3_val) > 1e-12) {
          third_contrib += b3_val * ComputeThirdDerivativeY(slice, i, j, nx, ny, dy);
        }
        if (std::abs(az3_val) > 1e-12) {
          third_contrib += az3_val * ComputeThirdDerivativeZ(grid.data(), i, j, k, nx, ny, nz, dz);
        }

        double fourth_contrib = 0.0;
        if (std::abs(a4_val) > 1e-12) {
          const double u_xxxx = ComputeFourthDerivativeX(row, i, nx, dx);
          fourth_contrib += a4_val * (u_xxxx - c4x * u);
        }
        if (std::abs(b4_val) > 1e-12) {
          const double u_yyyy = ComputeFourthDerivativeY(slice, i, j, nx, ny, dy);
          fourth_contrib += b4_val * (u_yyyy - c4y * u);
        }
        if (std::abs(az4_val) > 1e-12) {
          const double u_zzzz = ComputeFourthDerivativeZ(grid.data(), i, j, k, nx, ny, nz, dz);
          fourth_contrib += az4_val * (u_zzzz - c4z * u);
        }

        const double r = center * u +
          base_contrib + mixed_contrib + third_contrib + fourth_contrib + f_val;
        AccumulateResidual(r, &sum_sq, &max_abs);
      }
    }
  }
  out->l2 = std::sqrt(std::max(0.0, sum_sq.sum));
  out->linf = max_abs;
  return true;
}
