#include "relaxation.h"

#include <algorithm>
#include <cmath>
#include <mutex>
#include <optional>
#include <utility>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "../cpu_utils.h"
#include "coefficient_evaluator.h"
#include "coordinate_metrics.h"
#include "embedded_boundary.h"
#include "expression_eval.h"
#include "residual.h"

namespace {
bool IsDirichlet(const BoundaryCondition& bc) { return bc.kind == BCKind::Dirichlet; }
bool IsNeumann(const BoundaryCondition& bc) { return bc.kind == BCKind::Neumann; }
bool IsRobin(const BoundaryCondition& bc) { return bc.kind == BCKind::Robin; }

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
}  // namespace

SolveOutput SolveGaussSeidelSor2D(const SolveInput& input,
                                  const Domain& d,
                                  double dx, double dy,
                                  const std::vector<unsigned char>* active,
                                  const std::vector<double>* integral_weights,
                                  bool use_shape,
                                  const std::vector<CellBoundaryInfo>& boundary_info,
                                  const ProgressCallback& progress,
                                  const std::function<bool()>& cancelled) {
  const int nx = d.nx;
  const int ny = d.ny;
  std::vector<double> grid(static_cast<size_t>(nx * ny), 0.0);

  auto is_active = [&](int i, int j) -> bool {
    if (!active || active->empty()) {
      return true;
    }
    return (*active)[static_cast<size_t>(Index(i, j, nx))] != 0;
  };

  // Apply Dirichlet conditions once at the beginning.
  ApplyDirichletCPU(input, d, dx, dy, &grid, is_active);

  const double a = input.pde.a;
  const double b = input.pde.b;
  const double c = input.pde.c;
  const double dcoef = input.pde.d;
  const double e = input.pde.e;
  const double f_const = input.pde.f;
  const double ab = input.pde.ab;
  const double a3 = input.pde.a3;
  const double b3 = input.pde.b3;
  const double a4 = input.pde.a4;
  const double b4 = input.pde.b4;
  CoefficientEvaluator coeff_eval = BuildCoefficientEvaluator(input.pde);
  if (!coeff_eval.ok) {
    return {coeff_eval.error, {}};
  }
  const bool has_var_coeff = coeff_eval.has_variable;
  const bool has_fourth =
      std::abs(a4) > 1e-12 || std::abs(b4) > 1e-12 ||
      !input.pde.a4_latex.empty() || !input.pde.b4_latex.empty();

  const bool has_rhs_expr = !input.pde.rhs_latex.empty();
  std::optional<ExpressionEvaluator> rhs_eval;
  if (has_rhs_expr) {
    ExpressionEvaluator test = ExpressionEvaluator::ParseLatex(input.pde.rhs_latex);
    if (!test.ok()) {
      return {"invalid RHS expression: " + test.error(), {}};
    }
    rhs_eval.emplace(std::move(test));
  }
  auto eval_rhs = [f_const, &rhs_eval](double x, double y) -> double {
    if (rhs_eval) {
      return rhs_eval->Eval(x, y, 0.0, 0.0);
    }
    return f_const;
  };

  const double ax_const = a / (dx * dx);
  const double by_const = b / (dy * dy);
  const double cx_const = c / (2.0 * dx);
  const double dyc_const = dcoef / (2.0 * dy);
  const double center_const = -2.0 * ax_const - 2.0 * by_const + e;
  if (!has_var_coeff && !has_fourth && std::abs(center_const) < 1e-12) {
    return {"degenerate PDE center coefficient", {}};
  }

  const int max_iter = std::max(1, input.solver.max_iter);
  const double omega = input.solver.method == SolveMethod::SOR ? input.solver.sor_omega : 1.0;
  if (omega <= 0.0 || omega >= 2.0) {
    return {"SOR omega must be in (0,2)", {}};
  }

  const bool has_integrals = !input.integrals.empty();
  const bool has_nonlinear = !input.nonlinear.empty();

  std::mutex progress_mutex;
  auto emit_progress = [&](const std::string& phase, double value) {
    if (!progress) return;
    std::lock_guard<std::mutex> lock(progress_mutex);
    progress(phase, value);
  };

  emit_progress("threads_total", 1.0);
  emit_progress("threads_active", 1.0);
  for (int iter = 0; iter < max_iter; ++iter) {
    if (cancelled && cancelled()) {
      emit_progress("threads_active", 0.0);
      return {"solve cancelled", {}};
    }
    
    // Apply Neumann/Robin conditions at the start of each iteration.
    ApplyNeumannRobinCPU(input, d, dx, dy, &grid, is_active);

    if (use_shape && !boundary_info.empty()) {
      for (int j = 1; j < ny - 1; ++j) {
        for (int i = 1; i < nx - 1; ++i) {
          const size_t idx = static_cast<size_t>(Index(i, j, nx));
          const CellBoundaryInfo& cell_info = boundary_info[idx];
          if (cell_info.is_cut_cell && !is_active(i, j)) {
            ApplyEmbeddedBoundaryBC2D(input.embedded_bc, cell_info, i, j, d, dx, dy, &grid);
          }
        }
      }
    }

    double integral_value = 0.0;
    if (has_integrals) {
      const std::vector<unsigned char>* active_ptr = use_shape ? active : nullptr;
      integral_value = ComputeIntegralValue(grid, nx, ny, active_ptr, dx, dy);
    }

    double max_delta = 0.0;
    for (int j = 1; j < ny - 1; ++j) {
      const double* row = grid.data() + static_cast<size_t>(Index(0, j, nx));
      for (int i = 1; i < nx - 1; ++i) {
        const int idx = Index(i, j, nx);
        if (!is_active(i, j)) {
          if (use_shape && !boundary_info.empty() &&
              boundary_info[static_cast<size_t>(idx)].is_cut_cell) {
            continue;
          }
          grid[idx] = 0.0;
          continue;
        }
        const double old_u = grid[idx];
        const double u_left = grid[Index(i - 1, j, nx)];
        const double u_right = grid[Index(i + 1, j, nx)];
        const double u_down = grid[Index(i, j - 1, nx)];
        const double u_up = grid[Index(i, j + 1, nx)];
        const double x = d.xmin + i * dx;
        const double y = d.ymin + j * dy;
        const double f_val = eval_rhs(x, y);
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
          return {"degenerate PDE center coefficient", {}};
        }
        const double integral_term =
            has_integrals && integral_weights ? integral_weights->at(static_cast<size_t>(idx)) * integral_value : 0.0;
        const double nonlinear_term =
            has_nonlinear ? EvalNonlinear(input.nonlinear, old_u) : 0.0;

        double laplacian_contrib;
        if (d.coord_system == CoordinateSystem::Cartesian) {
          laplacian_contrib = (ax + cx) * u_right + (ax - cx) * u_left +
                             (by + dyc) * u_up + (by - dyc) * u_down;
        } else {
          MetricDerivatives2D derivs = ComputeMetricDerivatives2D(
              d.coord_system, x, y, old_u, u_left, u_right, u_down, u_up, dx, dy);

          double ar = a_val / (dx * dx);
          double bs = b_val * derivs.metric_factor / (dy * dy);
          double cr = c_val / (2.0 * dx);
          double cs = d_val * std::sqrt(derivs.metric_factor) / (2.0 * dy);

          if (d.coord_system == CoordinateSystem::Polar || 
              d.coord_system == CoordinateSystem::Axisymmetric) {
            if (std::abs(x) > 1e-12) {
              const double r_inv = 1.0 / x;
              cr += a_val * r_inv / (2.0 * dx);
            }
          } else if (d.coord_system == CoordinateSystem::SphericalSurface) {
            if (std::abs(std::sin(y)) > 1e-12) {
              const double cot_theta = std::cos(y) / std::sin(y);
              cs += b_val * cot_theta / (2.0 * dy);
            }
          }

          laplacian_contrib = (ar + cr) * u_right + (ar - cr) * u_left +
                             (bs + cs) * u_up + (bs - cs) * u_down;
        }

        double mixed_contrib = 0.0;
        if (std::abs(ab_val) > 1e-12) {
          const double u_pp = grid[Index(i + 1, j + 1, nx)];
          const double u_pm = grid[Index(i + 1, j - 1, nx)];
          const double u_mp = grid[Index(i - 1, j + 1, nx)];
          const double u_mm = grid[Index(i - 1, j - 1, nx)];
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
          fourth_contrib += a4_val * (u_xxxx - c4x * old_u);
        }
        if (std::abs(b4_val) > 1e-12) {
          const double u_yyyy = ComputeFourthDerivativeY(grid.data(), i, j, nx, ny, dy);
          fourth_contrib += b4_val * (u_yyyy - c4y * old_u);
        }

        const double rhs = -(f_val + integral_term + nonlinear_term) -
                           (laplacian_contrib + mixed_contrib + third_contrib + fourth_contrib);
        const double jacobi_update = rhs / center;
        const double updated = (1.0 - omega) * old_u + omega * jacobi_update;
        grid[idx] = updated;
        max_delta = std::max(max_delta, std::abs(updated - old_u));
      }
      emit_progress("solve", static_cast<double>(j) / static_cast<double>(std::max(1, ny - 2)));
    }
    emit_progress("solve_total", static_cast<double>(iter + 1) / static_cast<double>(max_iter));
    if (max_delta < input.solver.tol) {
      break;
    }
  }
  emit_progress("threads_active", 0.0);

  SolveOutput out;
  out.grid = std::move(grid);
  if (input.solver.residual_interval > 0) {
    ResidualNorms norms;
    std::string res_err;
    if (ComputeResidualNorms(input, out.grid, &norms, &res_err)) {
      out.residual_l2 = norms.l2;
      out.residual_linf = norms.linf;
    }
  }
  return out;
}

SolveOutput SolveJacobi2D(const SolveInput& input,
                          const Domain& d,
                          double dx, double dy,
                          const std::vector<unsigned char>* active,
                          const std::vector<double>* integral_weights,
                          bool use_shape,
                          const std::vector<CellBoundaryInfo>& boundary_info,
                          const ProgressCallback& progress,
                          const std::function<bool()>& cancelled) {
  const int nx = d.nx;
  const int ny = d.ny;
  std::vector<double> grid(static_cast<size_t>(nx * ny), 0.0);
  std::vector<double> next(static_cast<size_t>(nx * ny), 0.0);

  auto is_active = [&](int i, int j) -> bool {
    if (!active || active->empty()) {
      return true;
    }
    return (*active)[static_cast<size_t>(Index(i, j, nx))] != 0;
  };

  ApplyDirichletCPU(input, d, dx, dy, &grid, is_active);

  const double a = input.pde.a;
  const double b = input.pde.b;
  const double c = input.pde.c;
  const double dcoef = input.pde.d;
  const double e = input.pde.e;
  const double f_const = input.pde.f;
  const double ab = input.pde.ab;
  const double a3 = input.pde.a3;
  const double b3 = input.pde.b3;
  const double a4 = input.pde.a4;
  const double b4 = input.pde.b4;
  CoefficientEvaluator coeff_eval = BuildCoefficientEvaluator(input.pde);
  if (!coeff_eval.ok) {
    return {coeff_eval.error, {}};
  }
  const bool has_var_coeff = coeff_eval.has_variable;
  const bool has_fourth =
      std::abs(a4) > 1e-12 || std::abs(b4) > 1e-12 ||
      !input.pde.a4_latex.empty() || !input.pde.b4_latex.empty();

  const bool has_rhs_expr = !input.pde.rhs_latex.empty();
  std::optional<ExpressionEvaluator> rhs_eval;
  if (has_rhs_expr) {
    ExpressionEvaluator test = ExpressionEvaluator::ParseLatex(input.pde.rhs_latex);
    if (!test.ok()) {
      return {"invalid RHS expression: " + test.error(), {}};
    }
    rhs_eval.emplace(std::move(test));
  }
  auto eval_rhs = [f_const, &rhs_eval](double x, double y) -> double {
    if (rhs_eval) {
      return rhs_eval->Eval(x, y, 0.0, 0.0);
    }
    return f_const;
  };

  const double ax_const = a / (dx * dx);
  const double by_const = b / (dy * dy);
  const double cx_const = c / (2.0 * dx);
  const double dyc_const = dcoef / (2.0 * dy);
  const double center_const = -2.0 * ax_const - 2.0 * by_const + e;
  if (!has_var_coeff && !has_fourth && std::abs(center_const) < 1e-12) {
    return {"degenerate PDE center coefficient", {}};
  }

  const int max_iter = std::max(1, input.solver.max_iter);
  const bool has_integrals = !input.integrals.empty();
  const bool has_nonlinear = !input.nonlinear.empty();

  std::mutex progress_mutex;
  auto emit_progress = [&](const std::string& phase, double value) {
    if (!progress) return;
    std::lock_guard<std::mutex> lock(progress_mutex);
    progress(phase, value);
  };

#ifdef _OPENMP
  const int num_threads = omp_get_max_threads();
  emit_progress("threads_total", static_cast<double>(num_threads));
  emit_progress("threads_active", static_cast<double>(num_threads));
#else
  emit_progress("threads_total", 1.0);
  emit_progress("threads_active", 1.0);
#endif
  for (int iter = 0; iter < max_iter; ++iter) {
    if (cancelled && cancelled()) {
      emit_progress("threads_active", 0.0);
      return {"solve cancelled", {}};
    }

    ApplyNeumannRobinCPU(input, d, dx, dy, &grid, is_active);

    if (use_shape && !boundary_info.empty()) {
      for (int j = 1; j < ny - 1; ++j) {
        for (int i = 1; i < nx - 1; ++i) {
          const size_t idx = static_cast<size_t>(Index(i, j, nx));
          const CellBoundaryInfo& cell_info = boundary_info[idx];
          if (cell_info.is_cut_cell && !is_active(i, j)) {
            ApplyEmbeddedBoundaryBC2D(input.embedded_bc, cell_info, i, j, d, dx, dy, &grid);
          }
        }
      }
    }

    double integral_value = 0.0;
    if (has_integrals) {
      const std::vector<unsigned char>* active_ptr = use_shape ? active : nullptr;
      integral_value = ComputeIntegralValue(grid, nx, ny, active_ptr, dx, dy);
    }

    double max_delta = 0.0;
    bool degenerate_error = false;
#ifdef _OPENMP
    #pragma omp parallel for reduction(max:max_delta) reduction(||:degenerate_error) schedule(static)
#endif
    for (int j = 1; j < ny - 1; ++j) {
      const double* row = grid.data() + static_cast<size_t>(Index(0, j, nx));
      for (int i = 1; i < nx - 1; ++i) {
        if (!is_active(i, j)) {
          next[Index(i, j, nx)] = 0.0;
          continue;
        }
        const int idx = Index(i, j, nx);
        const double old_u = grid[idx];
        const double u_left = grid[Index(i - 1, j, nx)];
        const double u_right = grid[Index(i + 1, j, nx)];
        const double u_down = grid[Index(i, j - 1, nx)];
        const double u_up = grid[Index(i, j + 1, nx)];
        const double x = d.xmin + i * dx;
        const double y = d.ymin + j * dy;
        const double f_val = eval_rhs(x, y);
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
          degenerate_error = true;
          continue;
        }
        const double integral_term =
            has_integrals && integral_weights ? integral_weights->at(static_cast<size_t>(idx)) * integral_value : 0.0;
        const double nonlinear_term =
            has_nonlinear ? EvalNonlinear(input.nonlinear, old_u) : 0.0;

        double laplacian_contrib;
        if (d.coord_system == CoordinateSystem::Cartesian) {
          laplacian_contrib = (ax + cx) * u_right + (ax - cx) * u_left +
                             (by + dyc) * u_up + (by - dyc) * u_down;
        } else {
          MetricDerivatives2D derivs = ComputeMetricDerivatives2D(
              d.coord_system, x, y, old_u, u_left, u_right, u_down, u_up, dx, dy);

          double ar = a_val / (dx * dx);
          double bs = b_val * derivs.metric_factor / (dy * dy);
          double cr = c_val / (2.0 * dx);
          double cs = d_val * std::sqrt(derivs.metric_factor) / (2.0 * dy);

          if (d.coord_system == CoordinateSystem::Polar ||
              d.coord_system == CoordinateSystem::Axisymmetric) {
            if (std::abs(x) > 1e-12) {
              const double r_inv = 1.0 / x;
              cr += a_val * r_inv / (2.0 * dx);
            }
          } else if (d.coord_system == CoordinateSystem::SphericalSurface) {
            if (std::abs(std::sin(y)) > 1e-12) {
              const double cot_theta = std::cos(y) / std::sin(y);
              cs += b_val * cot_theta / (2.0 * dy);
            }
          }

          laplacian_contrib = (ar + cr) * u_right + (ar - cr) * u_left +
                             (bs + cs) * u_up + (bs - cs) * u_down;
        }

        double mixed_contrib = 0.0;
        if (std::abs(ab_val) > 1e-12) {
          const double u_pp = grid[Index(i + 1, j + 1, nx)];
          const double u_pm = grid[Index(i + 1, j - 1, nx)];
          const double u_mp = grid[Index(i - 1, j + 1, nx)];
          const double u_mm = grid[Index(i - 1, j - 1, nx)];
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
          fourth_contrib += a4_val * (u_xxxx - c4x * old_u);
        }
        if (std::abs(b4_val) > 1e-12) {
          const double u_yyyy = ComputeFourthDerivativeY(grid.data(), i, j, nx, ny, dy);
          fourth_contrib += b4_val * (u_yyyy - c4y * old_u);
        }

        const double rhs = -(f_val + integral_term + nonlinear_term) -
                           (laplacian_contrib + mixed_contrib + third_contrib + fourth_contrib);
        const double updated = rhs / center;
        next[idx] = updated;
        max_delta = std::max(max_delta, std::abs(updated - old_u));
      }
    }
    if (degenerate_error) {
      return {"degenerate PDE center coefficient", {}};
    }

    ApplyNeumannRobinCPU(input, d, dx, dy, &next, is_active);
    ApplyDirichletCPU(input, d, dx, dy, &next, is_active);

    grid.swap(next);

    emit_progress("solve_total", static_cast<double>(iter + 1) / static_cast<double>(max_iter));
    if (max_delta < input.solver.tol) {
      break;
    }
  }
  emit_progress("threads_active", 0.0);

  SolveOutput out;
  out.grid = std::move(grid);
  if (input.solver.residual_interval > 0) {
    ResidualNorms norms;
    std::string res_err;
    if (ComputeResidualNorms(input, out.grid, &norms, &res_err)) {
      out.residual_l2 = norms.l2;
      out.residual_linf = norms.linf;
    }
  }
  return out;
}
