#include "cpu_utils.h"

#include <algorithm>
#include <cmath>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "coefficient_evaluator.h"
#include "coordinate_metrics.h"
#include "finite_differences.h"
#include "pde_types.h"
#include "boundary_utils.h"
#include "expression_eval.h"

namespace {
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
}

double Dot(const std::vector<double>& a, const std::vector<double>& b) {
  const size_t n = std::min(a.size(), b.size());
  double sum = 0.0;
#ifdef _OPENMP
  #pragma omp parallel for reduction(+:sum) schedule(static)
#endif
  for (size_t i = 0; i < n; ++i) {
    sum += a[i] * b[i];
  }
  return sum;
}

double Norm2(const std::vector<double>& v) {
  return std::sqrt(std::max(0.0, Dot(v, v)));
}

double NormInf(const std::vector<double>& v) {
  double max_abs = 0.0;
  const size_t n = v.size();
#ifdef _OPENMP
  #pragma omp parallel for reduction(max:max_abs) schedule(static)
#endif
  for (size_t i = 0; i < n; ++i) {
    const double abs_val = std::abs(v[i]);
    if (abs_val > max_abs) max_abs = abs_val;
  }
  return max_abs;
}

double ComputeIntegralValue(const std::vector<double>& grid, int nx, int ny,
                            const std::vector<unsigned char>* active, double dx, double dy) {
  KahanSum sum;
  for (int j = 0; j < ny; ++j) {
    for (int i = 0; i < nx; ++i) {
      const size_t idx = static_cast<size_t>(Index(i, j, nx));
      if (active && !active->empty() && (*active)[idx] == 0) {
        continue;
      }
      sum.Add(grid[idx]);
    }
  }
  return sum.sum * dx * dy;
}

double ComputeIntegralValue3D(const std::vector<double>& grid, int nx, int ny, int nz,
                              const std::vector<unsigned char>* active, double dx, double dy,
                              double dz) {
  KahanSum sum;
  for (int k = 0; k < nz; ++k) {
    for (int j = 0; j < ny; ++j) {
      for (int i = 0; i < nx; ++i) {
        const size_t idx = static_cast<size_t>(Index3D(i, j, k, nx, ny));
        if (active && !active->empty() && (*active)[idx] == 0) {
          continue;
        }
        sum.Add(grid[idx]);
      }
    }
  }
  return sum.sum * dx * dy * dz;
}

void Ax2D(const Domain& d,
          const BoundarySet& bc,
          const std::vector<unsigned char>* active,
          const std::vector<double>* integral_weights,
          double ax, double by, double cx, double dyc, double center,
          double ab,
          const std::vector<double>& x,
          std::vector<double>* y_out) {
  if (!y_out) {
    return;
  }
  const int nx = d.nx;
  const int ny = d.ny;
  y_out->assign(static_cast<size_t>(nx * ny), 0.0);

  const double dx = (d.xmax - d.xmin) / static_cast<double>(nx - 1);
  const double dy = (d.ymax - d.ymin) / static_cast<double>(ny - 1);
  const double a = ax * dx * dx;
  const double b = by * dy * dy;
  const double c = cx * 2.0 * dx;
  const double dcoef = dyc * 2.0 * dy;

  double integral_value = 0.0;
  if (integral_weights && !integral_weights->empty()) {
    // compute on-the-fly; caller provides weights
    KahanSum integral_sum;
    for (int j = 0; j < ny; ++j) {
      for (int i = 0; i < nx; ++i) {
        const size_t idx = static_cast<size_t>(Index(i, j, nx));
        if (active && !active->empty() && (*active)[idx] == 0) {
          continue;
        }
        integral_sum.Add(x[idx]);
      }
    }
    integral_value = integral_sum.sum * dx * dy;
  }

  auto is_active = [&](int i, int j) -> bool {
    if (!active || active->empty()) {
      return true;
    }
    return (*active)[static_cast<size_t>(Index(i, j, nx))] != 0;
  };

  auto get_neighbor = [&](int i, int j, int di, int dj) -> double {
    int ni = i + di;
    int nj = j + dj;
    if (IsPeriodicBoundary(d.coord_system, 0) && ni < 0) ni = nx - 2;
    if (IsPeriodicBoundary(d.coord_system, 1) && ni >= nx) ni = 1;
    if (IsPeriodicBoundary(d.coord_system, 2) && nj < 0) nj = ny - 2;
    if (IsPeriodicBoundary(d.coord_system, 3) && nj >= ny) nj = 1;
    if (ni < 0 || ni >= nx || nj < 0 || nj >= ny) {
      return 0.0;
    }
    return x[static_cast<size_t>(Index(ni, nj, nx))];
  };

  for (int j = 0; j < ny; ++j) {
    for (int i = 0; i < nx; ++i) {
      const int idx = Index(i, j, nx);
      if (!is_active(i, j)) {
        (*y_out)[static_cast<size_t>(idx)] = x[static_cast<size_t>(idx)];
        continue;
      }
      const bool boundary = (i == 0 || j == 0 || i == nx - 1 || j == ny - 1);
      if (boundary) {
        if (IsPeriodicBoundary(d.coord_system, 0) && i == 0) {
          (*y_out)[static_cast<size_t>(idx)] = get_neighbor(i, j, 1, 0);
        } else if (IsPeriodicBoundary(d.coord_system, 1) && i == nx - 1) {
          (*y_out)[static_cast<size_t>(idx)] = get_neighbor(i, j, -1, 0);
        } else if (IsPeriodicBoundary(d.coord_system, 2) && j == 0) {
          (*y_out)[static_cast<size_t>(idx)] = get_neighbor(i, j, 0, 1);
        } else if (IsPeriodicBoundary(d.coord_system, 3) && j == ny - 1) {
          (*y_out)[static_cast<size_t>(idx)] = get_neighbor(i, j, 0, -1);
        } else {
          (*y_out)[static_cast<size_t>(idx)] = x[static_cast<size_t>(idx)];
        }
        continue;
      }
      const double u_c = x[static_cast<size_t>(idx)];
      const double u_left = x[static_cast<size_t>(Index(i - 1, j, nx))];
      const double u_right = x[static_cast<size_t>(Index(i + 1, j, nx))];
      const double u_down = x[static_cast<size_t>(Index(i, j - 1, nx))];
      const double u_up = x[static_cast<size_t>(Index(i, j + 1, nx))];

      double y = 0.0;
      if (d.coord_system == CoordinateSystem::Cartesian) {
        y = center * u_c
          + (ax + cx) * u_right
          + (ax - cx) * u_left
          + (by + dyc) * u_up
          + (by - dyc) * u_down;

        if (std::abs(ab) > 1e-12 && i > 0 && i < nx - 1 && j > 0 && j < ny - 1) {
          const double u_pp = x[static_cast<size_t>(Index(i + 1, j + 1, nx))];
          const double u_pm = x[static_cast<size_t>(Index(i + 1, j - 1, nx))];
          const double u_mp = x[static_cast<size_t>(Index(i - 1, j + 1, nx))];
          const double u_mm = x[static_cast<size_t>(Index(i - 1, j - 1, nx))];
          const double u_xy = ComputeMixedDerivativeXY(u_pp, u_pm, u_mp, u_mm, dx, dy);
          y += ab * u_xy;
        }
      } else {
        const double x_coord = d.xmin + i * dx;
        const double y_coord = d.ymin + j * dy;
        MetricDerivatives2D derivs = ComputeMetricDerivatives2D(
            d.coord_system, x_coord, y_coord, u_c, u_left, u_right, u_down, u_up, dx, dy);

        double ar = a / (dx * dx);
        double bs = b * derivs.metric_factor / (dy * dy);
        double cr = c / (2.0 * dx);
        double cs = dcoef * std::sqrt(derivs.metric_factor) / (2.0 * dy);

        if (d.coord_system == CoordinateSystem::Polar || 
            d.coord_system == CoordinateSystem::Axisymmetric) {
          if (std::abs(x_coord) > 1e-12) {
            cr += a / x_coord / (2.0 * dx);
          }
        } else if (d.coord_system == CoordinateSystem::SphericalSurface) {
          if (std::abs(std::sin(y_coord)) > 1e-12) {
            const double cot_theta = std::cos(y_coord) / std::sin(y_coord);
            cs += b * cot_theta / (2.0 * dy);
          }
        }

        double center_metric = -2.0 * ar - 2.0 * bs + (center + 2.0 * ax + 2.0 * by);
        if (d.coord_system == CoordinateSystem::Polar || 
            d.coord_system == CoordinateSystem::Axisymmetric) {
          if (std::abs(x_coord) > 1e-12) {
            center_metric -= a / (x_coord * dx);
          }
        }

        y = center_metric * u_c
          + (ar + cr) * u_right
          + (ar - cr) * u_left
          + (bs + cs) * u_up
          + (bs - cs) * u_down;
      }

      if (integral_weights && !integral_weights->empty()) {
        y += (*integral_weights)[static_cast<size_t>(idx)] * integral_value;
      }
      (*y_out)[static_cast<size_t>(idx)] = y;
    }
  }

  for (int j = 0; j < ny; ++j) {
    if (is_active(0, j) && !IsPeriodicBoundary(d.coord_system, 0)) {
      (*y_out)[static_cast<size_t>(Index(0, j, nx))] = x[static_cast<size_t>(Index(0, j, nx))];
    }
    if (is_active(nx - 1, j) && !IsPeriodicBoundary(d.coord_system, 1)) {
      (*y_out)[static_cast<size_t>(Index(nx - 1, j, nx))] = x[static_cast<size_t>(Index(nx - 1, j, nx))];
    }
  }
  for (int i = 0; i < nx; ++i) {
    if (is_active(i, 0) && !IsPeriodicBoundary(d.coord_system, 2)) {
      (*y_out)[static_cast<size_t>(Index(i, 0, nx))] = x[static_cast<size_t>(Index(i, 0, nx))];
    }
    if (is_active(i, ny - 1) && !IsPeriodicBoundary(d.coord_system, 3)) {
      (*y_out)[static_cast<size_t>(Index(i, ny - 1, nx))] = x[static_cast<size_t>(Index(i, ny - 1, nx))];
    }
  }
}

LinearOperator2D BuildLinearOperator2D(const SolveInput& input,
                                       const Domain& d,
                                       const std::vector<unsigned char>* active,
                                       const std::vector<double>* integral_weights) {
  LinearOperator2D op;
  op.domain = &d;
  op.active = active;
  op.integral_weights = integral_weights;
  op.nx = d.nx;
  op.ny = d.ny;
  op.dx = (d.xmax - d.xmin) / static_cast<double>(std::max(1, d.nx - 1));
  op.dy = (d.ymax - d.ymin) / static_cast<double>(std::max(1, d.ny - 1));

  op.a = input.pde.a;
  op.b = input.pde.b;
  op.c = input.pde.c;
  op.dcoef = input.pde.d;
  op.e = input.pde.e;
  op.ab = input.pde.ab;
  op.a3 = input.pde.a3;
  op.b3 = input.pde.b3;
  op.a4 = input.pde.a4;
  op.b4 = input.pde.b4;

  op.has_mixed = HasMixedDerivatives(input.pde);
  op.has_high_order = HasHigherOrderDerivatives(input.pde);
  op.has_integrals = integral_weights && !integral_weights->empty();

  op.coeff_eval = BuildCoefficientEvaluator(input.pde);
  if (!op.coeff_eval.ok) {
    op.ok = false;
    op.error = op.coeff_eval.error;
    return op;
  }
  op.has_var_coeff = op.coeff_eval.has_variable;

  op.ax_const = op.a / (op.dx * op.dx);
  op.by_const = op.b / (op.dy * op.dy);
  op.cx_const = op.c / (2.0 * op.dx);
  op.dyc_const = op.dcoef / (2.0 * op.dy);
  op.center_const = -2.0 * op.ax_const - 2.0 * op.by_const + op.e;

  if (d.coord_system != CoordinateSystem::Cartesian && (op.has_mixed || op.has_high_order)) {
    op.ok = false;
    op.error = "mixed or higher-order derivatives require Cartesian coordinates";
    return op;
  }
  if (op.has_high_order) {
    const bool x_high = std::abs(input.pde.a3) > 1e-12 || std::abs(input.pde.a4) > 1e-12 ||
                        !input.pde.a3_latex.empty() || !input.pde.a4_latex.empty();
    const bool y_high = std::abs(input.pde.b3) > 1e-12 || std::abs(input.pde.b4) > 1e-12 ||
                        !input.pde.b3_latex.empty() || !input.pde.b4_latex.empty();
    if (x_high && d.nx < 5) {
      op.ok = false;
      op.error = "higher-order x-derivatives require nx >= 5";
      return op;
    }
    if (y_high && d.ny < 5) {
      op.ok = false;
      op.error = "higher-order y-derivatives require ny >= 5";
      return op;
    }
  }

  return op;
}

void LinearOperator2D::Apply(const std::vector<double>& x, std::vector<double>* y_out) const {
  if (!y_out || !domain) {
    return;
  }
  const Domain& d = *domain;
  y_out->assign(static_cast<size_t>(nx * ny), 0.0);

  double integral_value = 0.0;
  if (has_integrals) {
    integral_value = ComputeIntegralValue(x, nx, ny, active, dx, dy);
  }

  auto is_active = [&](int i, int j) -> bool {
    if (!active || active->empty()) {
      return true;
    }
    return (*active)[static_cast<size_t>(Index(i, j, nx))] != 0;
  };

  auto get_neighbor = [&](int i, int j, int di, int dj) -> double {
    int ni = i + di;
    int nj = j + dj;
    if (IsPeriodicBoundary(d.coord_system, 0) && ni < 0) ni = nx - 2;
    if (IsPeriodicBoundary(d.coord_system, 1) && ni >= nx) ni = 1;
    if (IsPeriodicBoundary(d.coord_system, 2) && nj < 0) nj = ny - 2;
    if (IsPeriodicBoundary(d.coord_system, 3) && nj >= ny) nj = 1;
    if (ni < 0 || ni >= nx || nj < 0 || nj >= ny) {
      return 0.0;
    }
    return x[static_cast<size_t>(Index(ni, nj, nx))];
  };

#ifdef _OPENMP
  #pragma omp parallel for schedule(static)
#endif
  for (int j = 0; j < ny; ++j) {
    const double y_coord = d.ymin + j * dy;
    const double* row = x.data() + static_cast<size_t>(Index(0, j, nx));
    for (int i = 0; i < nx; ++i) {
      const int idx = Index(i, j, nx);
      if (!is_active(i, j)) {
        (*y_out)[static_cast<size_t>(idx)] = x[static_cast<size_t>(idx)];
        continue;
      }
      const bool boundary = (i == 0 || j == 0 || i == nx - 1 || j == ny - 1);
      if (boundary) {
        if (IsPeriodicBoundary(d.coord_system, 0) && i == 0) {
          (*y_out)[static_cast<size_t>(idx)] = get_neighbor(i, j, 1, 0);
        } else if (IsPeriodicBoundary(d.coord_system, 1) && i == nx - 1) {
          (*y_out)[static_cast<size_t>(idx)] = get_neighbor(i, j, -1, 0);
        } else if (IsPeriodicBoundary(d.coord_system, 2) && j == 0) {
          (*y_out)[static_cast<size_t>(idx)] = get_neighbor(i, j, 0, 1);
        } else if (IsPeriodicBoundary(d.coord_system, 3) && j == ny - 1) {
          (*y_out)[static_cast<size_t>(idx)] = get_neighbor(i, j, 0, -1);
        } else {
          (*y_out)[static_cast<size_t>(idx)] = x[static_cast<size_t>(idx)];
        }
        continue;
      }

      const double x_coord = d.xmin + i * dx;
      const double u = x[static_cast<size_t>(idx)];
      const double u_left = x[static_cast<size_t>(Index(i - 1, j, nx))];
      const double u_right = x[static_cast<size_t>(Index(i + 1, j, nx))];
      const double u_down = x[static_cast<size_t>(Index(i, j - 1, nx))];
      const double u_up = x[static_cast<size_t>(Index(i, j + 1, nx))];

      const double a_val = has_var_coeff ? EvalCoefficient(coeff_eval.a, a, x_coord, y_coord, 0.0, 0.0) : a;
      const double b_val = has_var_coeff ? EvalCoefficient(coeff_eval.b, b, x_coord, y_coord, 0.0, 0.0) : b;
      const double c_val = has_var_coeff ? EvalCoefficient(coeff_eval.c, c, x_coord, y_coord, 0.0, 0.0) : c;
      const double d_val = has_var_coeff ? EvalCoefficient(coeff_eval.d, dcoef, x_coord, y_coord, 0.0, 0.0) : dcoef;
      const double e_val = has_var_coeff ? EvalCoefficient(coeff_eval.e, e, x_coord, y_coord, 0.0, 0.0) : e;
      const double ab_val = has_var_coeff ? EvalCoefficient(coeff_eval.ab, ab, x_coord, y_coord, 0.0, 0.0) : ab;
      const double a3_val = has_var_coeff ? EvalCoefficient(coeff_eval.a3, a3, x_coord, y_coord, 0.0, 0.0) : a3;
      const double b3_val = has_var_coeff ? EvalCoefficient(coeff_eval.b3, b3, x_coord, y_coord, 0.0, 0.0) : b3;
      const double a4_val = has_var_coeff ? EvalCoefficient(coeff_eval.a4, a4, x_coord, y_coord, 0.0, 0.0) : a4;
      const double b4_val = has_var_coeff ? EvalCoefficient(coeff_eval.b4, b4, x_coord, y_coord, 0.0, 0.0) : b4;

      double value = 0.0;
      if (d.coord_system == CoordinateSystem::Cartesian) {
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

        const double base_contrib =
            (ax + cx) * u_right +
            (ax - cx) * u_left +
            (by + dyc) * u_up +
            (by - dyc) * u_down;

        double mixed_contrib = 0.0;
        if (std::abs(ab_val) > 1e-12) {
          const double u_pp = x[static_cast<size_t>(Index(i + 1, j + 1, nx))];
          const double u_pm = x[static_cast<size_t>(Index(i + 1, j - 1, nx))];
          const double u_mp = x[static_cast<size_t>(Index(i - 1, j + 1, nx))];
          const double u_mm = x[static_cast<size_t>(Index(i - 1, j - 1, nx))];
          mixed_contrib = ab_val * ComputeMixedDerivativeXY(u_pp, u_pm, u_mp, u_mm, dx, dy);
        }

        double third_contrib = 0.0;
        if (std::abs(a3_val) > 1e-12) {
          third_contrib += a3_val * ComputeThirdDerivativeX(row, i, nx, dx);
        }
        if (std::abs(b3_val) > 1e-12) {
          third_contrib += b3_val * ComputeThirdDerivativeY(x.data(), i, j, nx, ny, dy);
        }

        double fourth_contrib = 0.0;
        if (std::abs(a4_val) > 1e-12) {
          const double u_xxxx = ComputeFourthDerivativeX(row, i, nx, dx);
          fourth_contrib += a4_val * (u_xxxx - c4x * u);
        }
        if (std::abs(b4_val) > 1e-12) {
          const double u_yyyy = ComputeFourthDerivativeY(x.data(), i, j, nx, ny, dy);
          fourth_contrib += b4_val * (u_yyyy - c4y * u);
        }

        value = center * u + base_contrib + mixed_contrib + third_contrib + fourth_contrib;
      } else {
        MetricDerivatives2D derivs = ComputeMetricDerivatives2D(
            d.coord_system, x_coord, y_coord, u, u_left, u_right, u_down, u_up, dx, dy);

        double ar = a_val / (dx * dx);
        double bs = b_val * derivs.metric_factor / (dy * dy);
        double cr = c_val / (2.0 * dx);
        double cs = d_val * std::sqrt(derivs.metric_factor) / (2.0 * dy);

        if (d.coord_system == CoordinateSystem::Polar ||
            d.coord_system == CoordinateSystem::Axisymmetric) {
          if (std::abs(x_coord) > 1e-12) {
            cr += a_val / x_coord / (2.0 * dx);
          }
        } else if (d.coord_system == CoordinateSystem::SphericalSurface) {
          if (std::abs(std::sin(y_coord)) > 1e-12) {
            const double cot_theta = std::cos(y_coord) / std::sin(y_coord);
            cs += b_val * cot_theta / (2.0 * dy);
          }
        }

        double center_metric = -2.0 * ar - 2.0 * bs + e_val;
        if (d.coord_system == CoordinateSystem::Polar ||
            d.coord_system == CoordinateSystem::Axisymmetric) {
          if (std::abs(x_coord) > 1e-12) {
            center_metric -= a_val / (x_coord * dx);
          }
        }

        value = center_metric * u
            + (ar + cr) * u_right
            + (ar - cr) * u_left
            + (bs + cs) * u_up
            + (bs - cs) * u_down;
      }

      if (has_integrals) {
        value += (*integral_weights)[static_cast<size_t>(idx)] * integral_value;
      }
      (*y_out)[static_cast<size_t>(idx)] = value;
    }
  }

  for (int j = 0; j < ny; ++j) {
    if (is_active(0, j) && !IsPeriodicBoundary(d.coord_system, 0)) {
      (*y_out)[static_cast<size_t>(Index(0, j, nx))] = x[static_cast<size_t>(Index(0, j, nx))];
    }
    if (is_active(nx - 1, j) && !IsPeriodicBoundary(d.coord_system, 1)) {
      (*y_out)[static_cast<size_t>(Index(nx - 1, j, nx))] = x[static_cast<size_t>(Index(nx - 1, j, nx))];
    }
  }
  for (int i = 0; i < nx; ++i) {
    if (is_active(i, 0) && !IsPeriodicBoundary(d.coord_system, 2)) {
      (*y_out)[static_cast<size_t>(Index(i, 0, nx))] = x[static_cast<size_t>(Index(i, 0, nx))];
    }
    if (is_active(i, ny - 1) && !IsPeriodicBoundary(d.coord_system, 3)) {
      (*y_out)[static_cast<size_t>(Index(i, ny - 1, nx))] = x[static_cast<size_t>(Index(i, ny - 1, nx))];
    }
  }
}

namespace {

bool IsDirichlet(const BoundaryCondition& bc) { return bc.kind == BCKind::Dirichlet; }
bool IsNeumann(const BoundaryCondition& bc) { return bc.kind == BCKind::Neumann; }
bool IsRobin(const BoundaryCondition& bc) { return bc.kind == BCKind::Robin; }

double EvalExpr(const BoundaryCondition::Expression& expr, double x, double y, double z, double t) {
  if (!expr.latex.empty()) {
    ExpressionEvaluator evaluator = ExpressionEvaluator::ParseLatex(expr.latex);
    if (evaluator.ok()) {
      const double v = evaluator.Eval(x, y, z, t);
      if (std::isfinite(v)) return v;
    }
  }
  return expr.constant + expr.x * x + expr.y * y + expr.z * z;
}

double EvalExpr(const BoundaryCondition::Expression& expr, double x, double y, double t) {
  return EvalExpr(expr, x, y, 0.0, t);
}

} // namespace

bool EvalCondition(const std::string& condition_latex, double x, double y, double z, double t) {
  if (condition_latex.empty()) {
    return true;
  }
  ExpressionEvaluator evaluator = ExpressionEvaluator::ParseLatex(condition_latex);
  if (!evaluator.ok()) {
    return false;
  }
  const double result = evaluator.Eval(x, y, z, t);
  return result > 0.0;
}

bool EvalCondition(const std::string& condition_latex, double x, double y) {
  return EvalCondition(condition_latex, x, y, 0.0, 0.0);
}

const BoundaryCondition& GetBCForPoint(
    const BoundaryCondition& default_bc,
    const std::vector<PiecewiseBoundaryCondition>& piecewise,
    double x, double y, double z, double t) {
  if (piecewise.empty()) {
    if (!default_bc.condition_latex.empty() && !EvalCondition(default_bc.condition_latex, x, y, z, t)) {
      return default_bc;
    }
    return default_bc;
  }
  for (const auto& pw : piecewise) {
    for (const auto& seg : pw.segments) {
      if (EvalCondition(seg.first, x, y, z, t)) {
        return seg.second;
      }
    }
    return pw.default_bc;
  }
  return default_bc;
}

void ApplyDirichletCPU(const SolveInput& input, const Domain& d, double dx, double dy,
                       std::vector<double>* grid,
                       const std::function<bool(int, int)>& is_active, double t) {
    const int nx = d.nx;
    const int ny = d.ny;
    auto& grid_ref = *grid;

    for (int j = 0; j < ny; ++j) {
        if (!is_active(0, j) && !is_active(nx - 1, j)) continue;
        const double y = d.ymin + j * dy;
        
        const BoundaryCondition& left_bc = GetBCForPoint(input.bc.left, input.bc.left_piecewise, d.xmin, y, 0.0, t);
        if (is_active(0, j) && IsDirichlet(left_bc)) {
            grid_ref[Index(0, j, nx)] = EvalExpr(left_bc.value, d.xmin, y, t);
        }

        const BoundaryCondition& right_bc = GetBCForPoint(input.bc.right, input.bc.right_piecewise, d.xmax, y, 0.0, t);
        if (is_active(nx - 1, j) && IsDirichlet(right_bc)) {
            grid_ref[Index(nx - 1, j, nx)] = EvalExpr(right_bc.value, d.xmax, y, t);
        }
    }

    for (int i = 0; i < nx; ++i) {
        if (!is_active(i, 0) && !is_active(i, ny - 1)) continue;
        const double x = d.xmin + i * dx;

        const BoundaryCondition& bottom_bc = GetBCForPoint(input.bc.bottom, input.bc.bottom_piecewise, x, d.ymin, 0.0, t);
        if (is_active(i, 0) && IsDirichlet(bottom_bc)) {
            grid_ref[Index(i, 0, nx)] = EvalExpr(bottom_bc.value, x, d.ymin, t);
        }

        const BoundaryCondition& top_bc = GetBCForPoint(input.bc.top, input.bc.top_piecewise, x, d.ymax, 0.0, t);
        if (is_active(i, ny - 1) && IsDirichlet(top_bc)) {
            grid_ref[Index(i, ny - 1, nx)] = EvalExpr(top_bc.value, x, d.ymax, t);
        }
    }

    if (IsPeriodicBoundary(d.coord_system, 0) && IsPeriodicBoundary(d.coord_system, 1)) {
        for (int j = 0; j < ny; ++j) {
            if (is_active(0, j) && is_active(nx - 1, j)) {
                grid_ref[Index(0, j, nx)] = grid_ref[Index(nx - 2, j, nx)];
                grid_ref[Index(nx - 1, j, nx)] = grid_ref[Index(1, j, nx)];
            }
        }
    }
    if (IsPeriodicBoundary(d.coord_system, 2) && IsPeriodicBoundary(d.coord_system, 3)) {
        for (int i = 0; i < nx; ++i) {
            if (is_active(i, 0) && is_active(i, ny - 1)) {
                grid_ref[Index(i, 0, nx)] = grid_ref[Index(i, ny - 2, nx)];
                grid_ref[Index(i, ny - 1, nx)] = grid_ref[Index(i, 1, nx)];
            }
        }
    }
}

void ApplyNeumannRobinCPU(const SolveInput& input, const Domain& d, double dx, double dy,
                          std::vector<double>* grid,
                          const std::function<bool(int, int)>& is_active, double t) {
    const int nx = d.nx;
    const int ny = d.ny;
    auto& grid_ref = *grid;

    const BoundaryCondition& left_bc_check = GetBCForPoint(input.bc.left, input.bc.left_piecewise, d.xmin, d.ymin + (d.ymax-d.ymin)/2.0, 0.0, t);
    if (!IsDirichlet(left_bc_check) && !IsPeriodicBoundary(d.coord_system, 0)) {
        for (int j = 0; j < ny; ++j) {
            if (!is_active(0, j)) continue;
            const double y = d.ymin + j * dy;
            const BoundaryCondition& bc = GetBCForPoint(input.bc.left, input.bc.left_piecewise, d.xmin, y, 0.0, t);
            if (IsNeumann(bc)) {
                const double g = EvalExpr(bc.value, d.xmin, y, t);
                grid_ref[Index(0, j, nx)] = grid_ref[Index(1, j, nx)] - dx * g;
            } else if (IsRobin(bc)) {
                const double alpha = EvalExpr(bc.alpha, d.xmin, y, t);
                const double beta = EvalExpr(bc.beta, d.xmin, y, t);
                const double gamma = EvalExpr(bc.gamma, d.xmin, y, t);
                const double denom = alpha + beta / dx;
                if (std::abs(denom) > 1e-12) {
                    grid_ref[Index(0, j, nx)] = (gamma + (beta / dx) * grid_ref[Index(1, j, nx)]) / denom;
                }
            }
        }
    }
    
    const BoundaryCondition& right_bc_check = GetBCForPoint(input.bc.right, input.bc.right_piecewise, d.xmax, d.ymin + (d.ymax-d.ymin)/2.0, 0.0, t);
    if (!IsDirichlet(right_bc_check) && !IsPeriodicBoundary(d.coord_system, 1)) {
        for (int j = 0; j < ny; ++j) {
            if (!is_active(nx - 1, j)) continue;
            const double y = d.ymin + j * dy;
            const BoundaryCondition& bc = GetBCForPoint(input.bc.right, input.bc.right_piecewise, d.xmax, y, 0.0, t);
            if (IsNeumann(bc)) {
                const double g = EvalExpr(bc.value, d.xmax, y, t);
                grid_ref[Index(nx - 1, j, nx)] = grid_ref[Index(nx - 2, j, nx)] + dx * g;
            } else if (IsRobin(bc)) {
                const double alpha = EvalExpr(bc.alpha, d.xmax, y, t);
                const double beta = EvalExpr(bc.beta, d.xmax, y, t);
                const double gamma = EvalExpr(bc.gamma, d.xmax, y, t);
                const double denom = alpha - beta / dx;
                if (std::abs(denom) > 1e-12) {
                    grid_ref[Index(nx - 1, j, nx)] = (gamma - (beta / dx) * grid_ref[Index(nx - 2, j, nx)]) / denom;
                }
            }
        }
    }

    const BoundaryCondition& bottom_bc_check = GetBCForPoint(input.bc.bottom, input.bc.bottom_piecewise, d.xmin + (d.xmax-d.xmin)/2.0, d.ymin, 0.0, t);
    if (!IsDirichlet(bottom_bc_check) && !IsPeriodicBoundary(d.coord_system, 2)) {
        for (int i = 0; i < nx; ++i) {
            if (!is_active(i, 0)) continue;
            const double x = d.xmin + i * dx;
            const BoundaryCondition& bc = GetBCForPoint(input.bc.bottom, input.bc.bottom_piecewise, x, d.ymin, 0.0, t);
            if (IsNeumann(bc)) {
                const double g = EvalExpr(bc.value, x, d.ymin, t);
                grid_ref[Index(i, 0, nx)] = grid_ref[Index(i, 1, nx)] - dy * g;
            } else if (IsRobin(bc)) {
                const double alpha = EvalExpr(bc.alpha, x, d.ymin, t);
                const double beta = EvalExpr(bc.beta, x, d.ymin, t);
                const double gamma = EvalExpr(bc.gamma, x, d.ymin, t);
                const double denom = alpha + beta / dy;
                if (std::abs(denom) > 1e-12) {
                    grid_ref[Index(i, 0, nx)] = (gamma + (beta / dy) * grid_ref[Index(i, 1, nx)]) / denom;
                }
            }
        }
    }

    const BoundaryCondition& top_bc_check = GetBCForPoint(input.bc.top, input.bc.top_piecewise, d.xmin + (d.xmax-d.xmin)/2.0, d.ymax, 0.0, t);
    if (!IsDirichlet(top_bc_check) && !IsPeriodicBoundary(d.coord_system, 3)) {
        for (int i = 0; i < nx; ++i) {
            if (!is_active(i, ny - 1)) continue;
            const double x = d.xmin + i * dx;
            const BoundaryCondition& bc = GetBCForPoint(input.bc.top, input.bc.top_piecewise, x, d.ymax, 0.0, t);
            if (IsNeumann(bc)) {
                const double g = EvalExpr(bc.value, x, d.ymax, t);
                grid_ref[Index(i, ny - 1, nx)] = grid_ref[Index(i, ny - 2, nx)] + dy * g;
            } else if (IsRobin(bc)) {
                const double alpha = EvalExpr(bc.alpha, x, d.ymax, t);
                const double beta = EvalExpr(bc.beta, x, d.ymax, t);
                const double gamma = EvalExpr(bc.gamma, x, d.ymax, t);
                const double denom = alpha - beta / dy;
                if (std::abs(denom) > 1e-12) {
                    grid_ref[Index(i, ny - 1, nx)] = (gamma - (beta / dy) * grid_ref[Index(i, ny - 2, nx)]) / denom;
                }
            }
        }
    }
}

void ApplyDirichletCPU3D(const SolveInput& input, const Domain& d, double dx, double dy, double dz,
                         std::vector<double>* grid,
                         const std::function<bool(int, int, int)>& is_active, double t) {
    const int nx = d.nx;
    const int ny = d.ny;
    const int nz = d.nz;
    auto& grid_ref = *grid;

    for (int k = 0; k < nz; ++k) {
        const double z = d.zmin + k * dz;
        for (int j = 0; j < ny; ++j) {
            const double y = d.ymin + j * dy;
            if (is_active(0, j, k)) {
                const auto& bc = GetBCForPoint(input.bc.left, input.bc.left_piecewise, d.xmin, y, z, t);
                if(IsDirichlet(bc)) grid_ref[Index3D(0, j, k, nx, ny)] = EvalExpr(bc.value, d.xmin, y, z, t);
            }
            if (is_active(nx - 1, j, k)) {
                const auto& bc = GetBCForPoint(input.bc.right, input.bc.right_piecewise, d.xmax, y, z, t);
                if(IsDirichlet(bc)) grid_ref[Index3D(nx - 1, j, k, nx, ny)] = EvalExpr(bc.value, d.xmax, y, z, t);
            }
        }
    }
    for (int k = 0; k < nz; ++k) {
        const double z = d.zmin + k * dz;
        for (int i = 0; i < nx; ++i) {
            const double x = d.xmin + i * dx;
            if (is_active(i, 0, k)) {
                const auto& bc = GetBCForPoint(input.bc.bottom, input.bc.bottom_piecewise, x, d.ymin, z, t);
                if(IsDirichlet(bc)) grid_ref[Index3D(i, 0, k, nx, ny)] = EvalExpr(bc.value, x, d.ymin, z, t);
            }
            if (is_active(i, ny - 1, k)) {
                const auto& bc = GetBCForPoint(input.bc.top, input.bc.top_piecewise, x, d.ymax, z, t);
                if(IsDirichlet(bc)) grid_ref[Index3D(i, ny - 1, k, nx, ny)] = EvalExpr(bc.value, x, d.ymax, z, t);
            }
        }
    }
    for (int j = 0; j < ny; ++j) {
        const double y = d.ymin + j * dy;
        for (int i = 0; i < nx; ++i) {
            const double x = d.xmin + i * dx;
            if (is_active(i, j, 0)) {
                const auto& bc = GetBCForPoint(input.bc.front, input.bc.front_piecewise, x, y, d.zmin, t);
                if(IsDirichlet(bc)) grid_ref[Index3D(i, j, 0, nx, ny)] = EvalExpr(bc.value, x, y, d.zmin, t);
            }
            if (is_active(i, j, nz - 1)) {
                const auto& bc = GetBCForPoint(input.bc.back, input.bc.back_piecewise, x, y, d.zmax, t);
                if(IsDirichlet(bc)) grid_ref[Index3D(i, j, nz - 1, nx, ny)] = EvalExpr(bc.value, x, y, d.zmax, t);
            }
        }
    }
}

void ApplyNeumannRobinCPU3D(const SolveInput& input, const Domain& d, double dx, double dy, double dz,
                            std::vector<double>* grid,
                            const std::function<bool(int, int, int)>& is_active, double t) {
    const int nx = d.nx;
    const int ny = d.ny;
    const int nz = d.nz;
    std::vector<double>& grid_ref = *grid;

    // Left (x = xmin) and Right (x = xmax) boundaries
    for (int k = 0; k < nz; ++k) {
        const double z = d.zmin + k * dz;
        for (int j = 0; j < ny; ++j) {
            const double y = d.ymin + j * dy;
            // Left boundary
            if (is_active(0, j, k)) {
                const auto& bc = GetBCForPoint(input.bc.left, input.bc.left_piecewise, d.xmin, y, z, t);
                if (IsNeumann(bc)) {
                    double g = EvalExpr(bc.value, d.xmin, y, z, t);
                    grid_ref[Index3D(0, j, k, nx, ny)] = grid_ref[Index3D(1, j, k, nx, ny)] - dx * g;
                } else if (IsRobin(bc)) {
                    const double alpha = EvalExpr(bc.alpha, d.xmin, y, z, t);
                    const double beta = EvalExpr(bc.beta, d.xmin, y, z, t);
                    const double gamma = EvalExpr(bc.gamma, d.xmin, y, z, t);
                    const double denom = alpha + beta / dx;
                    if (std::abs(denom) > 1e-12) {
                        grid_ref[Index3D(0, j, k, nx, ny)] = (gamma + (beta / dx) * grid_ref[Index3D(1, j, k, nx, ny)]) / denom;
                    }
                }
            }
            // Right boundary
            if (is_active(nx - 1, j, k)) {
                const auto& bc = GetBCForPoint(input.bc.right, input.bc.right_piecewise, d.xmax, y, z, t);
                if (IsNeumann(bc)) {
                    double g = EvalExpr(bc.value, d.xmax, y, z, t);
                    grid_ref[Index3D(nx - 1, j, k, nx, ny)] = grid_ref[Index3D(nx - 2, j, k, nx, ny)] + dx * g;
                } else if (IsRobin(bc)) {
                    const double alpha = EvalExpr(bc.alpha, d.xmax, y, z, t);
                    const double beta = EvalExpr(bc.beta, d.xmax, y, z, t);
                    const double gamma = EvalExpr(bc.gamma, d.xmax, y, z, t);
                    const double denom = alpha - beta / dx;
                    if (std::abs(denom) > 1e-12) {
                        grid_ref[Index3D(nx - 1, j, k, nx, ny)] = (gamma - (beta / dx) * grid_ref[Index3D(nx - 2, j, k, nx, ny)]) / denom;
                    }
                }
            }
        }
    }

    // Bottom (y = ymin) and Top (y = ymax) boundaries
    for (int k = 0; k < nz; ++k) {
        const double z = d.zmin + k * dz;
        for (int i = 0; i < nx; ++i) {
            const double x = d.xmin + i * dx;
            // Bottom boundary
            if (is_active(i, 0, k)) {
                const auto& bc = GetBCForPoint(input.bc.bottom, input.bc.bottom_piecewise, x, d.ymin, z, t);
                if (IsNeumann(bc)) {
                    double g = EvalExpr(bc.value, x, d.ymin, z, t);
                    grid_ref[Index3D(i, 0, k, nx, ny)] = grid_ref[Index3D(i, 1, k, nx, ny)] - dy * g;
                } else if (IsRobin(bc)) {
                    const double alpha = EvalExpr(bc.alpha, x, d.ymin, z, t);
                    const double beta = EvalExpr(bc.beta, x, d.ymin, z, t);
                    const double gamma = EvalExpr(bc.gamma, x, d.ymin, z, t);
                    const double denom = alpha + beta / dy;
                    if (std::abs(denom) > 1e-12) {
                        grid_ref[Index3D(i, 0, k, nx, ny)] = (gamma + (beta / dy) * grid_ref[Index3D(i, 1, k, nx, ny)]) / denom;
                    }
                }
            }
            // Top boundary
            if (is_active(i, ny - 1, k)) {
                const auto& bc = GetBCForPoint(input.bc.top, input.bc.top_piecewise, x, d.ymax, z, t);
                if (IsNeumann(bc)) {
                    double g = EvalExpr(bc.value, x, d.ymax, z, t);
                    grid_ref[Index3D(i, ny - 1, k, nx, ny)] = grid_ref[Index3D(i, ny - 2, k, nx, ny)] + dy * g;
                } else if (IsRobin(bc)) {
                    const double alpha = EvalExpr(bc.alpha, x, d.ymax, z, t);
                    const double beta = EvalExpr(bc.beta, x, d.ymax, z, t);
                    const double gamma = EvalExpr(bc.gamma, x, d.ymax, z, t);
                    const double denom = alpha - beta / dy;
                    if (std::abs(denom) > 1e-12) {
                        grid_ref[Index3D(i, ny - 1, k, nx, ny)] = (gamma - (beta / dy) * grid_ref[Index3D(i, ny - 2, k, nx, ny)]) / denom;
                    }
                }
            }
        }
    }

    // Front (z = zmin) and Back (z = zmax) boundaries
    for (int j = 0; j < ny; ++j) {
        const double y = d.ymin + j * dy;
        for (int i = 0; i < nx; ++i) {
            const double x = d.xmin + i * dx;
            // Front boundary
            if (is_active(i, j, 0)) {
                const auto& bc = GetBCForPoint(input.bc.front, input.bc.front_piecewise, x, y, d.zmin, t);
                if (IsNeumann(bc)) {
                    double g = EvalExpr(bc.value, x, y, d.zmin, t);
                    grid_ref[Index3D(i, j, 0, nx, ny)] = grid_ref[Index3D(i, j, 1, nx, ny)] - dz * g;
                } else if (IsRobin(bc)) {
                    const double alpha = EvalExpr(bc.alpha, x, y, d.zmin, t);
                    const double beta = EvalExpr(bc.beta, x, y, d.zmin, t);
                    const double gamma = EvalExpr(bc.gamma, x, y, d.zmin, t);
                    const double denom = alpha + beta / dz;
                    if (std::abs(denom) > 1e-12) {
                        grid_ref[Index3D(i, j, 0, nx, ny)] = (gamma + (beta / dz) * grid_ref[Index3D(i, j, 1, nx, ny)]) / denom;
                    }
                }
            }
            // Back boundary
            if (is_active(i, j, nz - 1)) {
                const auto& bc = GetBCForPoint(input.bc.back, input.bc.back_piecewise, x, y, d.zmax, t);
                if (IsNeumann(bc)) {
                    double g = EvalExpr(bc.value, x, y, d.zmax, t);
                    grid_ref[Index3D(i, j, nz - 1, nx, ny)] = grid_ref[Index3D(i, j, nz - 2, nx, ny)] + dz * g;
                } else if (IsRobin(bc)) {
                    const double alpha = EvalExpr(bc.alpha, x, y, d.zmax, t);
                    const double beta = EvalExpr(bc.beta, x, y, d.zmax, t);
                    const double gamma = EvalExpr(bc.gamma, x, y, d.zmax, t);
                    const double denom = alpha - beta / dz;
                    if (std::abs(denom) > 1e-12) {
                        grid_ref[Index3D(i, j, nz - 1, nx, ny)] = (gamma - (beta / dz) * grid_ref[Index3D(i, j, nz - 2, nx, ny)]) / denom;
                    }
                }
            }
        }
    }
}
