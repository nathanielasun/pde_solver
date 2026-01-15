#include "solver.h"

#include <algorithm>
#include <cmath>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "coefficient_evaluator.h"
#include "coordinate_metrics.h"
#include "cpu_utils.h"
#include "embedded_boundary.h"
#include "expression_eval.h"
#include "residual.h"
#include "safe_math.h"
#include "solvers/krylov.h"
#include "solvers/multigrid.h"
#include "solvers/relaxation.h"

SolveOutput SolvePDE3D(const SolveInput& input, const ProgressCallback& progress);
SolveOutput SolvePDETimeSeries(const SolveInput& input, const FrameCallback& on_frame, const ProgressCallback& progress);
SolveOutput SolvePDETimeSeries3D(const SolveInput& input, const FrameCallback& on_frame, const ProgressCallback& progress);

namespace {
bool IsDirichlet(const BoundaryCondition& bc) {
  return bc.kind == BCKind::Dirichlet;
}

bool PiecewiseAllDirichlet(const std::vector<PiecewiseBoundaryCondition>& piecewise) {
  for (const auto& pw : piecewise) {
    if (!IsDirichlet(pw.default_bc)) {
      return false;
    }
    for (const auto& seg : pw.segments) {
      if (!IsDirichlet(seg.second)) {
        return false;
      }
    }
  }
  return true;
}

bool AllDirichlet2D(const SolveInput& input) {
  auto side_ok = [](const BoundaryCondition& bc,
                    const std::vector<PiecewiseBoundaryCondition>& piecewise) {
    if (!piecewise.empty()) {
      return PiecewiseAllDirichlet(piecewise);
    }
    return IsDirichlet(bc);
  };
  return side_ok(input.bc.left, input.bc.left_piecewise) &&
         side_ok(input.bc.right, input.bc.right_piecewise) &&
         side_ok(input.bc.bottom, input.bc.bottom_piecewise) &&
         side_ok(input.bc.top, input.bc.top_piecewise);
}

bool HasPeriodicBoundary2D(const Domain& d) {
  return IsPeriodicBoundary(d.coord_system, 0) ||
         IsPeriodicBoundary(d.coord_system, 1) ||
         IsPeriodicBoundary(d.coord_system, 2) ||
         IsPeriodicBoundary(d.coord_system, 3);
}

bool BuildActiveMask2D(const SolveInput& input,
                       const Domain& d,
                       std::vector<unsigned char>* active,
                       std::vector<CellBoundaryInfo>* boundary_info,
                       std::string* error) {
  if (!HasImplicitShape(input)) {
    if (active) active->clear();
    if (boundary_info) boundary_info->clear();
    return true;
  }
  if (!boundary_info) {
    if (error) *error = "missing boundary storage for embedded domain";
    return false;
  }
  if (!BuildEmbeddedBoundary2D(d, input.domain_shape, input.shape_mask,
                               input.shape_transform, input.shape_mask_threshold,
                               input.shape_mask_invert, boundary_info, error)) {
    if (error && error->empty()) {
      *error = "invalid embedded domain shape";
    }
    return false;
  }
  if (active) {
    const int nx = d.nx;
    const int ny = d.ny;
    active->assign(static_cast<size_t>(nx * ny), 1);
    for (int j = 0; j < ny; ++j) {
      for (int i = 0; i < nx; ++i) {
        const size_t idx = static_cast<size_t>(Index(i, j, nx));
        (*active)[idx] = (*boundary_info)[idx].center_inside ? 1 : 0;
      }
    }
  }
  return true;
}

bool BuildIntegralWeights2D(const SolveInput& input,
                            const Domain& d,
                            double dx,
                            double dy,
                            const std::vector<unsigned char>* active,
                            std::vector<double>* weights,
                            std::string* error) {
  if (input.integrals.empty()) {
    if (weights) weights->clear();
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

bool BuildRhs2D(const SolveInput& input,
                const Domain& d,
                double dx,
                double dy,
                const std::vector<unsigned char>* active,
                std::vector<double>* b,
                std::string* error) {
  if (!b) {
    return false;
  }
  const int nx = d.nx;
  const int ny = d.ny;
  b->assign(static_cast<size_t>(nx * ny), 0.0);

  const bool has_rhs_expr = !input.pde.rhs_latex.empty();
  if (has_rhs_expr) {
    ExpressionEvaluator test = ExpressionEvaluator::ParseLatex(input.pde.rhs_latex);
    if (!test.ok()) {
      if (error) {
        *error = "invalid RHS expression: " + test.error();
      }
      return false;
    }
  }
  const std::string rhs_latex = input.pde.rhs_latex;
  const double f_const = input.pde.f;
  auto eval_rhs = [rhs_latex, has_rhs_expr, f_const](double x, double y) -> double {
    if (!has_rhs_expr) {
      return f_const;
    }
    ExpressionEvaluator evaluator = ExpressionEvaluator::ParseLatex(rhs_latex);
    if (evaluator.ok()) {
      return evaluator.Eval(x, y, 0.0, 0.0);
    }
    return f_const;
  };

  auto is_active = [&](int i, int j) -> bool {
    if (!active || active->empty()) {
      return true;
    }
    return (*active)[static_cast<size_t>(Index(i, j, nx))] != 0;
  };

  std::vector<double> boundary(static_cast<size_t>(nx * ny), 0.0);
  ApplyDirichletCPU(input, d, dx, dy, &boundary, is_active);

  for (int j = 0; j < ny; ++j) {
    const double y = d.ymin + j * dy;
    for (int i = 0; i < nx; ++i) {
      const int idx = Index(i, j, nx);
      if (!is_active(i, j)) {
        (*b)[static_cast<size_t>(idx)] = 0.0;
        continue;
      }
      if (i == 0 || i == nx - 1 || j == 0 || j == ny - 1) {
        (*b)[static_cast<size_t>(idx)] = boundary[static_cast<size_t>(idx)];
        continue;
      }
      const double x = d.xmin + i * dx;
      const double f_val = eval_rhs(x, y);
      (*b)[static_cast<size_t>(idx)] = -f_val;
    }
  }
  return true;
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

bool BuildActiveMask3D(const SolveInput& input,
                       const Domain& d,
                       std::vector<unsigned char>* active,
                       std::vector<CellBoundaryInfo>* boundary_info,
                       std::string* error) {
  if (!HasImplicitShape(input)) {
    if (active) active->clear();
    if (boundary_info) boundary_info->clear();
    return true;
  }
  if (!boundary_info) {
    if (error) *error = "missing boundary storage for embedded domain";
    return false;
  }
  if (!BuildEmbeddedBoundary3D(d, input.domain_shape, input.shape_mask,
                               input.shape_transform, input.shape_mask_threshold,
                               input.shape_mask_invert, boundary_info, error)) {
    if (error && error->empty()) {
      *error = "invalid embedded domain shape";
    }
    return false;
  }
  if (active) {
    const int nx = d.nx;
    const int ny = d.ny;
    const int nz = d.nz;
    active->assign(static_cast<size_t>(nx * ny * nz), 1);
    for (int k = 0; k < nz; ++k) {
      for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
          const size_t idx = static_cast<size_t>(Index3D(i, j, k, nx, ny));
          (*active)[idx] = (*boundary_info)[idx].center_inside ? 1 : 0;
        }
      }
    }
  }
  return true;
}

bool BuildIntegralWeights3D(const SolveInput& input,
                            const Domain& d,
                            double dx,
                            double dy,
                            double dz,
                            const std::vector<unsigned char>* active,
                            std::vector<double>* weights,
                            std::string* error) {
  if (input.integrals.empty()) {
    if (weights) weights->clear();
    return true;
  }
  if (!weights) {
    return false;
  }
  const int nx = d.nx;
  const int ny = d.ny;
  const int nz = d.nz;
  weights->assign(static_cast<size_t>(nx * ny * nz), 0.0);
  auto is_active = [&](int i, int j, int k) -> bool {
    if (!active || active->empty()) {
      return true;
    }
    return (*active)[static_cast<size_t>(Index3D(i, j, k, nx, ny))] != 0;
  };
  for (const auto& term : input.integrals) {
    if (std::abs(term.coeff) < 1e-12) {
      continue;
    }
    if (term.kernel_latex.empty()) {
      for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
          for (int i = 0; i < nx; ++i) {
            if (!is_active(i, j, k)) {
              continue;
            }
            (*weights)[static_cast<size_t>(Index3D(i, j, k, nx, ny))] += term.coeff;
          }
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
    for (int k = 0; k < nz; ++k) {
      const double z = d.zmin + k * dz;
      for (int j = 0; j < ny; ++j) {
        const double y = d.ymin + j * dy;
        for (int i = 0; i < nx; ++i) {
          if (!is_active(i, j, k)) {
            continue;
          }
          const double x = d.xmin + i * dx;
          (*weights)[static_cast<size_t>(Index3D(i, j, k, nx, ny))] += term.coeff * kernel.Eval(x, y, z);
        }
      }
    }
  }
  return true;
}

// Validate variable coefficients by sampling across the domain to detect
// potential degenerate values before starting the solve
struct CoefficientValidationResult {
  bool ok = true;
  std::string warning;  // Non-fatal warning about coefficient values
};

CoefficientValidationResult ValidateVariableCoefficients3D(
    const CoefficientEvaluator& coeff_eval,
    const Domain& d,
    double dx, double dy, double dz,
    double a, double b, double az, double e) {
  CoefficientValidationResult result;
  if (!coeff_eval.has_variable) {
    return result;  // No variable coefficients to validate
  }

  const int nx = d.nx;
  const int ny = d.ny;
  const int nz = d.nz;

  // Sample at a subset of interior points to avoid excessive computation
  const int sample_stride = std::max(1, std::min({nx, ny, nz}) / 10);
  int degenerate_count = 0;
  int sample_count = 0;

  for (int k = 1; k < nz - 1; k += sample_stride) {
    const double z = d.zmin + k * dz;
    for (int j = 1; j < ny - 1; j += sample_stride) {
      const double y = d.ymin + j * dy;
      for (int i = 1; i < nx - 1; i += sample_stride) {
        const double x = d.xmin + i * dx;
        ++sample_count;

        // Evaluate coefficients at this point
        const double a_val = EvalCoefficient(coeff_eval.a, a, x, y, z, 0.0);
        const double b_val = EvalCoefficient(coeff_eval.b, b, x, y, z, 0.0);
        const double az_val = EvalCoefficient(coeff_eval.az, az, x, y, z, 0.0);
        const double e_val = EvalCoefficient(coeff_eval.e, e, x, y, z, 0.0);

        // Compute center coefficient
        const double ax = a_val / (dx * dx);
        const double by = b_val / (dy * dy);
        const double cz = az_val / (dz * dz);
        const double center = -2.0 * ax - 2.0 * by - 2.0 * cz + e_val;

        if (std::abs(center) < 1e-12) {
          ++degenerate_count;
        }
      }
    }
  }

  if (degenerate_count > 0) {
    const double pct = 100.0 * static_cast<double>(degenerate_count) / static_cast<double>(sample_count);
    result.warning = "variable coefficients produce near-zero center coefficient at " +
                     std::to_string(degenerate_count) + "/" + std::to_string(sample_count) +
                     " sampled points (" + std::to_string(static_cast<int>(pct)) + "%)";
    // If more than 10% of points are degenerate, treat as error
    if (pct > 10.0) {
      result.ok = false;
    }
  }

  return result;
}
} // namespace

SolveOutput SolvePDE(const SolveInput& input, const ProgressCallback& progress) {
  if (std::abs(input.pde.ut) > 1e-12 || std::abs(input.pde.utt) > 1e-12) {
    return SolvePDETimeSeries(input, nullptr, progress);
  }
  const Domain& d = input.domain;

  // Validate grid dimensions first to prevent division by zero
  auto grid_check = pde::ValidateGridSize(d.nx, d.ny, d.nz);
  if (!grid_check.ok) {
    return {grid_check.error, {}};
  }

  // Validate domain bounds
  auto domain_check = pde::ValidateDomainBounds(d.xmin, d.xmax, d.ymin, d.ymax, d.zmin, d.zmax);
  if (!domain_check.ok) {
    return {domain_check.error, {}};
  }

  if (d.nz > 1) {
    return SolvePDE3D(input, progress);
  }
  const bool has_mixed = HasMixedDerivatives(input.pde);
  const bool has_high_order = HasHigherOrderDerivatives(input.pde);
  if (d.coord_system != CoordinateSystem::Cartesian && (has_mixed || has_high_order)) {
    return {"mixed or higher-order derivatives require Cartesian coordinates", {}};
  }

  const int nx = d.nx;
  const int ny = d.ny;

  // Compute grid spacing with validation
  auto spacing = pde::ComputeGridSpacing(d.xmin, d.xmax, nx, d.ymin, d.ymax, ny);
  if (!spacing.ok) {
    return {spacing.error, {}};
  }
  const double dx = spacing.dx;
  const double dy = spacing.dy;

  std::vector<unsigned char> active;
  std::vector<CellBoundaryInfo> boundary_info;
  const bool use_shape = HasImplicitShape(input);
  std::string shape_error;
  if (!BuildActiveMask2D(input, d, &active, &boundary_info, &shape_error)) {
    return {shape_error.empty() ? "failed to build embedded boundary" : shape_error, {}};
  }

  std::vector<double> integral_weights;
  std::string weight_error;
  if (!BuildIntegralWeights2D(input, d, dx, dy,
                              active.empty() ? nullptr : &active,
                              &integral_weights, &weight_error)) {
    return {weight_error.empty() ? "failed to build integral weights" : weight_error, {}};
  }

  const std::vector<unsigned char>* active_ptr = active.empty() ? nullptr : &active;
  const std::vector<double>* weights_ptr = integral_weights.empty() ? nullptr : &integral_weights;
  auto cancelled = [&]() -> bool {
    return input.cancel && input.cancel->load();
  };

  switch (input.solver.method) {
    case SolveMethod::Jacobi:
      return SolveJacobi2D(input, d, dx, dy, active_ptr, weights_ptr,
                           use_shape, boundary_info, progress, cancelled);
    case SolveMethod::GaussSeidel:
    case SolveMethod::SOR:
      return SolveGaussSeidelSor2D(input, d, dx, dy, active_ptr, weights_ptr,
                                   use_shape, boundary_info, progress, cancelled);
    case SolveMethod::CG:
    case SolveMethod::BiCGStab:
    case SolveMethod::GMRES: {
      if (!input.nonlinear.empty() || !input.nonlinear_derivatives.empty()) {
        return {"Krylov solvers require linear PDEs", {}};
      }
      if (use_shape) {
        return {"Krylov solvers do not support implicit domain shapes", {}};
      }
      if (!AllDirichlet2D(input) || HasPeriodicBoundary2D(d)) {
        return {"Krylov solvers require Dirichlet boundaries on all sides", {}};
      }
      std::vector<double> bvec;
      std::string rhs_error;
      if (!BuildRhs2D(input, d, dx, dy, active_ptr, &bvec, &rhs_error)) {
        return {rhs_error.empty() ? "failed to build RHS" : rhs_error, {}};
      }

      LinearOperator2D op = BuildLinearOperator2D(input, d, active_ptr, weights_ptr);
      if (!op.ok) {
        return {op.error.empty() ? "failed to build linear operator" : op.error, {}};
      }

      std::vector<double> xvec;
      const int max_iter = std::max(1, input.solver.max_iter);
      const double tol = input.solver.tol;
      const int residual_interval = input.solver.residual_interval;
      bool ok = false;

      if (input.solver.method == SolveMethod::CG) {
        const bool has_var_coeff = HasVariableCoefficients(input.pde);
        const bool has_mixed = HasMixedDerivatives(input.pde);
        const bool has_high_order = HasHigherOrderDerivatives(input.pde);
        if (has_var_coeff || std::abs(input.pde.c) > 1e-12 || std::abs(input.pde.d) > 1e-12 ||
            has_mixed || has_high_order) {
          return {"CG requires symmetric operators with constant coefficients (no convection/mixed/higher-order terms)", {}};
        }
        ok = CgSolve(op, bvec, &xvec, max_iter, tol, residual_interval, progress);
      } else if (input.solver.method == SolveMethod::BiCGStab) {
        ok = BiCGStabSolve(op, bvec, &xvec, max_iter, tol, residual_interval, progress);
      } else {
        ok = GmresSolve(op, bvec, &xvec, max_iter, input.solver.gmres_restart, tol,
                        residual_interval, progress);
      }

      if (!ok) {
        return {"CPU Krylov solver failed to converge", {}};
      }
      return {"", xvec};
    }
    case SolveMethod::MultigridVcycle: {
      if (!input.nonlinear.empty() || !input.nonlinear_derivatives.empty()) {
        return {"Multigrid requires linear PDEs", {}};
      }
      if (!input.integrals.empty()) {
        return {"Multigrid does not support integral terms", {}};
      }
      if (use_shape) {
        return {"Multigrid does not support implicit domain shapes", {}};
      }
      if (!AllDirichlet2D(input) || HasPeriodicBoundary2D(d)) {
        return {"Multigrid requires Dirichlet boundaries on all sides", {}};
      }
      if (std::abs(input.pde.c) > 1e-12 || std::abs(input.pde.d) > 1e-12 ||
          std::abs(input.pde.ab) > 1e-12 || HasHigherOrderDerivatives(input.pde)) {
        return {"Multigrid requires no convection, mixed, or higher-order derivatives", {}};
      }

      std::vector<double> bvec;
      std::string rhs_error;
      if (!BuildRhs2D(input, d, dx, dy, active_ptr, &bvec, &rhs_error)) {
        return {rhs_error.empty() ? "failed to build RHS" : rhs_error, {}};
      }

      std::vector<double> xvec(static_cast<size_t>(nx * ny), 0.0);
      auto is_active = [&](int i, int j) -> bool {
        if (active.empty()) {
          return true;
        }
        return active[static_cast<size_t>(Index(i, j, nx))] != 0;
      };
      ApplyDirichletCPU(input, d, dx, dy, &xvec, is_active);

      const bool ok = MultigridVcyclePoisson(d, input.pde.a, input.pde.b, input.pde.e,
                                             bvec, &xvec,
                                             input.solver.mg_pre_smooth,
                                             input.solver.mg_post_smooth,
                                             input.solver.mg_coarse_iters,
                                             input.solver.mg_max_levels);
      if (!ok) {
        return {"CPU multigrid solver failed", {}};
      }
      return {"", xvec};
    }
  }

  return {"Unknown CPU solver method", {}};
}

SolveOutput SolvePDE3D(const SolveInput& input, const ProgressCallback& progress) {
  const Domain& d = input.domain;

  // Validate grid dimensions first to prevent division by zero
  auto grid_check = pde::ValidateGridSize(d.nx, d.ny, d.nz);
  if (!grid_check.ok) {
    return {grid_check.error, {}};
  }

  // Validate domain bounds
  auto domain_check = pde::ValidateDomainBounds(d.xmin, d.xmax, d.ymin, d.ymax, d.zmin, d.zmax);
  if (!domain_check.ok) {
    return {domain_check.error, {}};
  }

  if (d.coord_system != CoordinateSystem::Cartesian) {
    return {"3D solver supports Cartesian coordinates only", {}};
  }
  if (!input.nonlinear_derivatives.empty()) {
    return {"3D solver does not support nonlinear derivative terms", {}};
  }
  if (input.solver.method != SolveMethod::Jacobi &&
      input.solver.method != SolveMethod::GaussSeidel &&
      input.solver.method != SolveMethod::SOR) {
    return {"3D solver supports Jacobi/Gauss-Seidel/SOR only", {}};
  }

  const int nx = d.nx;
  const int ny = d.ny;
  const int nz = d.nz;

  // Compute grid spacing with validation
  auto spacing = pde::ComputeGridSpacing(d.xmin, d.xmax, nx, d.ymin, d.ymax, ny, d.zmin, d.zmax, nz);
  if (!spacing.ok) {
    return {spacing.error, {}};
  }
  const double dx = spacing.dx;
  const double dy = spacing.dy;
  const double dz = spacing.dz;

  std::vector<unsigned char> active;
  std::vector<CellBoundaryInfo> boundary_info;
  const bool use_shape = HasImplicitShape(input);
  std::string shape_error;
  if (!BuildActiveMask3D(input, d, &active, &boundary_info, &shape_error)) {
    return {shape_error.empty() ? "failed to build embedded boundary" : shape_error, {}};
  }

  std::vector<double> integral_weights;
  std::string weight_error;
  if (!BuildIntegralWeights3D(input, d, dx, dy, dz,
                              active.empty() ? nullptr : &active,
                              &integral_weights, &weight_error)) {
    return {weight_error.empty() ? "failed to build integral weights" : weight_error, {}};
  }

  const std::vector<unsigned char>* active_ptr = active.empty() ? nullptr : &active;
  const std::vector<double>* weights_ptr = integral_weights.empty() ? nullptr : &integral_weights;

  auto is_active = [&](int i, int j, int k) -> bool {
    if (!active_ptr || active_ptr->empty()) {
      return true;
    }
    return (*active_ptr)[static_cast<size_t>(Index3D(i, j, k, nx, ny))] != 0;
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

  CoefficientEvaluator coeff_eval = BuildCoefficientEvaluator(input.pde);
  if (!coeff_eval.ok) {
    return {coeff_eval.error, {}};
  }
  const bool has_var_coeff = coeff_eval.has_variable;
  const bool has_fourth =
      std::abs(a4) > 1e-12 || std::abs(b4) > 1e-12 || std::abs(az4) > 1e-12 ||
      !input.pde.a4_latex.empty() || !input.pde.b4_latex.empty() ||
      !input.pde.az4_latex.empty();

  // Pre-validate variable coefficients to catch degenerate cases early
  if (has_var_coeff) {
    auto coeff_check = ValidateVariableCoefficients3D(coeff_eval, d, dx, dy, dz, a, b, az, e);
    if (!coeff_check.ok) {
      return {"variable coefficient validation failed: " + coeff_check.warning, {}};
    }
    // Log warning if non-fatal issues detected
    if (!coeff_check.warning.empty() && progress) {
      progress("coefficient_warning", 1.0);
    }
  }

  const bool has_rhs_expr = !input.pde.rhs_latex.empty();
  std::optional<ExpressionEvaluator> rhs_eval;
  if (has_rhs_expr) {
    ExpressionEvaluator test = ExpressionEvaluator::ParseLatex(input.pde.rhs_latex);
    if (!test.ok()) {
      return {"invalid RHS expression: " + test.error(), {}};
    }
    rhs_eval.emplace(std::move(test));
  }
  const double f_const = input.pde.f;
  auto eval_rhs = [f_const, &rhs_eval](double x, double y, double z) -> double {
    if (rhs_eval) {
      return rhs_eval->Eval(x, y, z, 0.0);
    }
    return f_const;
  };

  const double ax_const = a / (dx * dx);
  const double by_const = b / (dy * dy);
  const double cz_const = az / (dz * dz);
  const double cx_const = c / (2.0 * dx);
  const double dyc_const = dcoef / (2.0 * dy);
  const double dzc_const = dzcoef / (2.0 * dz);
  const double center_const = -2.0 * ax_const - 2.0 * by_const - 2.0 * cz_const + e;
  if (!has_var_coeff && !has_fourth && std::abs(center_const) < 1e-12) {
    return {"degenerate PDE center coefficient", {}};
  }

  const double omega =
      input.solver.method == SolveMethod::SOR ? input.solver.sor_omega : 1.0;
  if (input.solver.method == SolveMethod::SOR && (omega <= 0.0 || omega >= 2.0)) {
    return {"SOR omega must be in (0,2)", {}};
  }

  const int max_iter = std::max(1, input.solver.max_iter);
  const bool has_integrals = !input.integrals.empty();
  const bool has_nonlinear = !input.nonlinear.empty();

  auto emit_progress = [&](const std::string& phase, double value) {
    if (progress) {
      progress(phase, value);
    }
  };

  std::vector<double> grid(static_cast<size_t>(nx * ny * nz), 0.0);
  std::vector<double> next;
  if (input.solver.method == SolveMethod::Jacobi) {
    next.assign(static_cast<size_t>(nx * ny * nz), 0.0);
  }

  ApplyDirichletCPU3D(input, d, dx, dy, dz, &grid, is_active);
  ApplyNeumannRobinCPU3D(input, d, dx, dy, dz, &grid, is_active);
  ApplyDirichletCPU3D(input, d, dx, dy, dz, &grid, is_active);

#ifdef _OPENMP
  const int num_threads_3d = omp_get_max_threads();
  emit_progress("threads_total", static_cast<double>(num_threads_3d));
  emit_progress("threads_active", static_cast<double>(num_threads_3d));
#else
  emit_progress("threads_total", 1.0);
  emit_progress("threads_active", 1.0);
#endif
  for (int iter = 0; iter < max_iter; ++iter) {
    if (input.cancel && input.cancel->load()) {
      emit_progress("threads_active", 0.0);
      return {"solve cancelled", {}};
    }

    ApplyNeumannRobinCPU3D(input, d, dx, dy, dz, &grid, is_active);

    if (use_shape && !boundary_info.empty()) {
      for (int k = 1; k < nz - 1; ++k) {
        for (int j = 1; j < ny - 1; ++j) {
          for (int i = 1; i < nx - 1; ++i) {
            const size_t idx = static_cast<size_t>(Index3D(i, j, k, nx, ny));
            const CellBoundaryInfo& cell_info = boundary_info[idx];
            if (cell_info.is_cut_cell && !is_active(i, j, k)) {
              ApplyEmbeddedBoundaryBC3D(input.embedded_bc, cell_info, i, j, k, d, dx, dy, dz, &grid);
            }
          }
        }
      }
    }

    // Compute integral value BEFORE parallel section to ensure thread-safe read-only access
    // The integral is computed once from the current grid state and used as a constant
    // during this iteration. This must be computed serially before any parallel updates.
    const double integral_value = [&]() -> double {
      if (!has_integrals) return 0.0;
      const std::vector<unsigned char>* active_integral = use_shape ? active_ptr : nullptr;
      return ComputeIntegralValue3D(grid, nx, ny, nz, active_integral, dx, dy, dz);
    }();

    double max_delta = 0.0;
    bool degenerate_error = false;
    // Only Jacobi can be parallelized because it reads from grid and writes to next.
    // GS/SOR update in-place and have data dependencies between cells.
    const bool can_parallelize = (input.solver.method == SolveMethod::Jacobi);
#ifdef _OPENMP
    #pragma omp parallel for reduction(max:max_delta) reduction(||:degenerate_error) schedule(static) if(can_parallelize)
#endif
    for (int k = 1; k < nz - 1; ++k) {
      const double* slice = grid.data() + static_cast<size_t>(Index3D(0, 0, k, nx, ny));
      for (int j = 1; j < ny - 1; ++j) {
        const double* row = slice + static_cast<size_t>(Index(0, j, nx));
        for (int i = 1; i < nx - 1; ++i) {
          const int idx = Index3D(i, j, k, nx, ny);
          if (!is_active(i, j, k)) {
            if (input.solver.method != SolveMethod::Jacobi &&
                use_shape && !boundary_info.empty() &&
                boundary_info[static_cast<size_t>(idx)].is_cut_cell) {
              continue;
            }
            if (input.solver.method == SolveMethod::Jacobi) {
              next[static_cast<size_t>(idx)] = 0.0;
            } else {
              grid[static_cast<size_t>(idx)] = 0.0;
            }
            continue;
          }
          const double old_u = grid[static_cast<size_t>(idx)];
          const double u_left = grid[static_cast<size_t>(Index3D(i - 1, j, k, nx, ny))];
          const double u_right = grid[static_cast<size_t>(Index3D(i + 1, j, k, nx, ny))];
          const double u_down = grid[static_cast<size_t>(Index3D(i, j - 1, k, nx, ny))];
          const double u_up = grid[static_cast<size_t>(Index3D(i, j + 1, k, nx, ny))];
          const double u_back = grid[static_cast<size_t>(Index3D(i, j, k - 1, nx, ny))];
          const double u_front = grid[static_cast<size_t>(Index3D(i, j, k + 1, nx, ny))];
          const double x = d.xmin + i * dx;
          const double y = d.ymin + j * dy;
          const double z = d.zmin + k * dz;
          const double f_val = eval_rhs(x, y, z);
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
            degenerate_error = true;
            continue;
          }
          const double integral_term =
              has_integrals && weights_ptr ? (*weights_ptr)[static_cast<size_t>(idx)] * integral_value : 0.0;
          const double nonlinear_term =
              has_nonlinear ? EvalNonlinear(input.nonlinear, old_u) : 0.0;

          const double laplacian_contrib =
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
            fourth_contrib += a4_val * (u_xxxx - c4x * old_u);
          }
          if (std::abs(b4_val) > 1e-12) {
            const double u_yyyy = ComputeFourthDerivativeY(slice, i, j, nx, ny, dy);
            fourth_contrib += b4_val * (u_yyyy - c4y * old_u);
          }
          if (std::abs(az4_val) > 1e-12) {
            const double u_zzzz = ComputeFourthDerivativeZ(grid.data(), i, j, k, nx, ny, nz, dz);
            fourth_contrib += az4_val * (u_zzzz - c4z * old_u);
          }

          const double rhs = -(f_val + integral_term + nonlinear_term) -
                             (laplacian_contrib + mixed_contrib + third_contrib + fourth_contrib);
          const double jacobi_update = rhs / center;
          if (input.solver.method == SolveMethod::Jacobi) {
            next[static_cast<size_t>(idx)] = jacobi_update;
          } else {
            grid[static_cast<size_t>(idx)] = (1.0 - omega) * old_u + omega * jacobi_update;
          }
          const double updated = (input.solver.method == SolveMethod::Jacobi)
                                     ? next[static_cast<size_t>(idx)]
                                     : grid[static_cast<size_t>(idx)];
          max_delta = std::max(max_delta, std::abs(updated - old_u));
        }
      }
    }

    if (degenerate_error) {
      return {"degenerate PDE center coefficient", {}};
    }

    if (input.solver.method == SolveMethod::Jacobi) {
      ApplyNeumannRobinCPU3D(input, d, dx, dy, dz, &next, is_active);
      ApplyDirichletCPU3D(input, d, dx, dy, dz, &next, is_active);
      grid.swap(next);
    } else {
      ApplyNeumannRobinCPU3D(input, d, dx, dy, dz, &grid, is_active);
      ApplyDirichletCPU3D(input, d, dx, dy, dz, &grid, is_active);
    }

    emit_progress("solve_total", static_cast<double>(iter + 1) / static_cast<double>(max_iter));
    if (max_delta < input.solver.tol) {
      break;
    }
  }
  emit_progress("threads_active", 0.0);

  SolveOutput out;
  out.grid = std::move(grid);
  return out;
}

SolveOutput SolvePDETimeSeries(const SolveInput& input, const FrameCallback& on_frame, const ProgressCallback& progress) {
  const Domain& d = input.domain;

  // Validate grid dimensions first
  auto grid_check = pde::ValidateGridSize(d.nx, d.ny, d.nz);
  if (!grid_check.ok) {
    return {grid_check.error, {}};
  }

  if (d.nz > 1) {
    return SolvePDETimeSeries3D(input, on_frame, progress);
  }
  if (!input.nonlinear_derivatives.empty()) {
    return {"time-dependent solver does not support nonlinear derivative terms", {}};
  }

  const int nx = d.nx;
  const int ny = d.ny;

  // Compute grid spacing with validation
  auto spacing = pde::ComputeGridSpacing(d.xmin, d.xmax, nx, d.ymin, d.ymax, ny);
  if (!spacing.ok) {
    return {spacing.error, {}};
  }
  const double dx = spacing.dx;
  const double dy = spacing.dy;

  const int frames = std::max(1, input.time.frames);
  const double t_start = input.time.t_start;
  const double dt = input.time.dt;

  // Validate dt is positive
  if (dt <= 0.0) {
    return {"time step dt must be positive", {}};
  }

  const bool has_ut = std::abs(input.pde.ut) > 1e-12;
  const bool has_utt = std::abs(input.pde.utt) > 1e-12;
  if (!has_ut && !has_utt) {
    return {"time-dependent solve requires u_t or u_tt term", {}};
  }

  // CFL stability check for explicit time stepping
  // For diffusion: dt < C * min(dx,dy)^2 / max_diffusion_coeff
  // For advection: dt < C * min(dx,dy) / max_advection_coeff
  const double max_diffusion = std::max(std::abs(input.pde.a), std::abs(input.pde.b));
  const double max_advection = std::max(std::abs(input.pde.c), std::abs(input.pde.d));
  auto cfl_check = pde::CheckCFL(dt, dx, dy, 0.0, max_diffusion, max_advection, 0.5);
  if (!cfl_check.stable && progress) {
    // Warn but don't fail - let user proceed at their own risk
    progress("cfl_warning", cfl_check.max_dt);
  }

  std::vector<unsigned char> active;
  std::vector<CellBoundaryInfo> boundary_info;
  const bool use_shape = HasImplicitShape(input);
  std::string shape_error;
  if (!BuildActiveMask2D(input, d, &active, &boundary_info, &shape_error)) {
    return {shape_error.empty() ? "failed to build embedded boundary" : shape_error, {}};
  }

  std::vector<double> integral_weights;
  std::string weight_error;
  if (!BuildIntegralWeights2D(input, d, dx, dy,
                              active.empty() ? nullptr : &active,
                              &integral_weights, &weight_error)) {
    return {weight_error.empty() ? "failed to build integral weights" : weight_error, {}};
  }

  const std::vector<unsigned char>* active_ptr = active.empty() ? nullptr : &active;
  const std::vector<double>* weights_ptr = integral_weights.empty() ? nullptr : &integral_weights;
  LinearOperator2D op = BuildLinearOperator2D(input, d, active_ptr, weights_ptr);
  if (!op.ok) {
    return {op.error.empty() ? "failed to build linear operator" : op.error, {}};
  }

  const bool has_rhs_expr = !input.pde.rhs_latex.empty();
  std::optional<ExpressionEvaluator> rhs_eval;
  if (has_rhs_expr) {
    ExpressionEvaluator test = ExpressionEvaluator::ParseLatex(input.pde.rhs_latex);
    if (!test.ok()) {
      return {"invalid RHS expression: " + test.error(), {}};
    }
    rhs_eval.emplace(std::move(test));
  }
  const double f_const = input.pde.f;
  auto eval_rhs = [&](double x, double y, double t) -> double {
    if (rhs_eval) {
      return rhs_eval->Eval(x, y, 0.0, t);
    }
    return f_const;
  };

  const bool has_nonlinear = !input.nonlinear.empty();
  auto is_active = [&](int i, int j) -> bool {
    if (!active_ptr || active_ptr->empty()) {
      return true;
    }
    return (*active_ptr)[static_cast<size_t>(Index(i, j, nx))] != 0;
  };
  auto apply_embedded = [&](std::vector<double>* grid) {
    if (!use_shape || boundary_info.empty() || !grid) {
      return;
    }
    for (int j = 1; j < ny - 1; ++j) {
      for (int i = 1; i < nx - 1; ++i) {
        const size_t idx = static_cast<size_t>(Index(i, j, nx));
        const CellBoundaryInfo& cell_info = boundary_info[idx];
        if (cell_info.is_cut_cell && !is_active(i, j)) {
          ApplyEmbeddedBoundaryBC2D(input.embedded_bc, cell_info, i, j, d, dx, dy, grid);
        }
      }
    }
  };

  std::vector<double> grid(static_cast<size_t>(nx * ny), 0.0);
  std::vector<double> next(static_cast<size_t>(nx * ny), 0.0);
  std::vector<double> velocity;
  if (has_utt) {
    velocity.assign(static_cast<size_t>(nx * ny), 0.0);
  }
  if (!input.initial_grid.empty()) {
    if (input.initial_grid.size() != grid.size()) {
      return {"initial grid size mismatch", {}};
    }
    grid = input.initial_grid;
  }
  if (has_utt && !input.initial_velocity.empty()) {
    if (input.initial_velocity.size() != grid.size()) {
      return {"initial velocity size mismatch", {}};
    }
    velocity = input.initial_velocity;
  } else if (has_utt && !input.initial_grid.empty() && input.initial_velocity.empty()) {
    return {"checkpoint restart for u_tt requires velocity data", {}};
  }

  double t = t_start;
  ApplyDirichletCPU(input, d, dx, dy, &grid, is_active, t);
  ApplyNeumannRobinCPU(input, d, dx, dy, &grid, is_active, t);
  ApplyDirichletCPU(input, d, dx, dy, &grid, is_active, t);
  apply_embedded(&grid);

  std::vector<double> op_values;
  if (progress) {
#ifdef _OPENMP
    const int num_threads = omp_get_max_threads();
    progress("threads_total", static_cast<double>(num_threads));
    progress("threads_active", static_cast<double>(num_threads));
#else
    progress("threads_total", 1.0);
    progress("threads_active", 1.0);
#endif
    progress("solve_total", static_cast<double>(frames));
    progress("time", 0.0);
  }

  for (int frame = 0; frame < frames; ++frame) {
    if (input.cancel && input.cancel->load()) {
      return {"solve cancelled", {}};
    }
    t = t_start + frame * dt;
    if (progress) {
      const double frac = (frames <= 1) ? 1.0 : static_cast<double>(frame) / static_cast<double>(frames - 1);
      progress("time", frac);
      progress("iteration", static_cast<double>(frame + 1));
    }

    if (on_frame) {
      const std::vector<double>* velocity_ptr = has_utt ? &velocity : nullptr;
      if (!on_frame(frame, t, grid, velocity_ptr)) {
        break;
      }
    }
    if (frame + 1 >= frames) {
      break;
    }

    op.Apply(grid, &op_values);
    next = grid;
    const double t_next = t + dt;
#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (int j = 1; j < ny - 1; ++j) {
      const double y = d.ymin + j * dy;
      for (int i = 1; i < nx - 1; ++i) {
        if (!is_active(i, j)) {
          next[static_cast<size_t>(Index(i, j, nx))] = 0.0;
          if (has_utt) {
            velocity[static_cast<size_t>(Index(i, j, nx))] = 0.0;
          }
          continue;
        }
        const int idx = Index(i, j, nx);
        const double x = d.xmin + i * dx;
        const double rhs = op_values[static_cast<size_t>(idx)] +
                           eval_rhs(x, y, t) +
                           (has_nonlinear ? EvalNonlinear(input.nonlinear, grid[static_cast<size_t>(idx)]) : 0.0);
        if (has_utt) {
          // Safe division for acceleration calculation
          const double numer = -(rhs + input.pde.ut * velocity[static_cast<size_t>(idx)]);
          const double accel = pde::SafeDivClamped(numer, input.pde.utt, 0.0);
          velocity[static_cast<size_t>(idx)] += dt * accel;
          next[static_cast<size_t>(idx)] = grid[static_cast<size_t>(idx)] + dt * velocity[static_cast<size_t>(idx)];
        } else {
          // Safe division for time derivative calculation
          const double u_t = pde::SafeDivClamped(-rhs, input.pde.ut, 0.0);
          next[static_cast<size_t>(idx)] = grid[static_cast<size_t>(idx)] + dt * u_t;
        }
      }
    }

    ApplyNeumannRobinCPU(input, d, dx, dy, &next, is_active, t_next);
    ApplyDirichletCPU(input, d, dx, dy, &next, is_active, t_next);
    apply_embedded(&next);
    grid.swap(next);
  }

  SolveOutput out;
  out.grid = std::move(grid);
  return out;
}

SolveOutput SolvePDETimeSeries3D(const SolveInput& input, const FrameCallback& on_frame, const ProgressCallback& progress) {
    // Stub for compilation
    return {"3D Time Series Solver not yet refactored.", {}};
}
