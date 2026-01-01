#include "backend_capability_matrix.h"

#include <cmath>
#include <string>
#include <vector>

#include "coordinate_metrics.h"

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

bool AllDirichlet(const SolveInput& input) {
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
         side_ok(input.bc.top, input.bc.top_piecewise) &&
         side_ok(input.bc.front, input.bc.front_piecewise) &&
         side_ok(input.bc.back, input.bc.back_piecewise);
}

bool HasPiecewiseBC(const SolveInput& input) {
  return !input.bc.left_piecewise.empty() ||
         !input.bc.right_piecewise.empty() ||
         !input.bc.bottom_piecewise.empty() ||
         !input.bc.top_piecewise.empty() ||
         !input.bc.front_piecewise.empty() ||
         !input.bc.back_piecewise.empty();
}

bool HasPeriodicBoundary(const Domain& d) {
  const int max_index = d.nz > 1 ? 6 : 4;
  for (int i = 0; i < max_index; ++i) {
    if (IsPeriodicBoundary(d.coord_system, i)) {
      return true;
    }
  }
  return false;
}

bool IsLinear(const SolveInput& input) {
  return input.nonlinear.empty() && input.nonlinear_derivatives.empty();
}
}  // namespace

BackendCapabilities GetBackendCapabilities(BackendKind kind) {
  BackendCapabilities caps;
  switch (kind) {
    case BackendKind::CPU:
      caps.supported_methods = {
          SolveMethod::Jacobi,
          SolveMethod::GaussSeidel,
          SolveMethod::SOR,
          SolveMethod::CG,
          SolveMethod::BiCGStab,
          SolveMethod::GMRES,
          SolveMethod::MultigridVcycle};
      caps.supports_3d = true;
      caps.supports_spatial_rhs = true;
      caps.supports_nonlinear = true;
      caps.supports_integrals = true;
      caps.supports_shapes = true;
      caps.supports_time_dependent = true;
      caps.supports_piecewise_bc = true;
      // CPU supports all coupling patterns
      caps.supports_coupled_fields = true;
      caps.supports_explicit_coupling = true;
      caps.supports_implicit_coupling = true;
      return caps;
    case BackendKind::CUDA:
      caps.supported_methods = {
          SolveMethod::Jacobi,
          SolveMethod::GaussSeidel,
          SolveMethod::SOR,
          SolveMethod::CG,
          SolveMethod::BiCGStab,
          SolveMethod::GMRES,
          SolveMethod::MultigridVcycle};
      caps.supports_3d = false;
      caps.supports_spatial_rhs = true;
      caps.supports_nonlinear = true;
      caps.supports_integrals = false;
      caps.supports_shapes = false;
      caps.supports_time_dependent = false;
      caps.supports_piecewise_bc = false;
      // CUDA: no coupling support yet
      caps.supports_coupled_fields = false;
      caps.supports_explicit_coupling = false;
      caps.supports_implicit_coupling = false;
      return caps;
    case BackendKind::Metal:
      caps.supported_methods = {
          SolveMethod::Jacobi,
          SolveMethod::GaussSeidel,
          SolveMethod::SOR,
          SolveMethod::CG,
          SolveMethod::BiCGStab,
          SolveMethod::GMRES,
          SolveMethod::MultigridVcycle};
      caps.supports_3d = false;
      caps.supports_spatial_rhs = true;
      caps.supports_nonlinear = false;
      caps.supports_integrals = false;
      caps.supports_shapes = false;
      caps.supports_time_dependent = false;
      caps.supports_piecewise_bc = false;
      // Metal: no coupling support
      caps.supports_coupled_fields = false;
      caps.supports_explicit_coupling = false;
      caps.supports_implicit_coupling = false;
      return caps;
    case BackendKind::TPU:
      caps.supported_methods = {SolveMethod::Jacobi};
      caps.supports_3d = false;
      caps.supports_spatial_rhs = false;
      caps.supports_nonlinear = false;
      caps.supports_integrals = false;
      caps.supports_shapes = false;
      caps.supports_time_dependent = false;
      caps.supports_piecewise_bc = false;
      // TPU: no coupling support
      caps.supports_coupled_fields = false;
      caps.supports_explicit_coupling = false;
      caps.supports_implicit_coupling = false;
      return caps;
    case BackendKind::Auto:
    default:
      return caps;
  }
}

bool BackendSupportsMethod(BackendKind kind, SolveMethod method) {
  return GetBackendCapabilities(kind).SupportsMethod(method);
}

bool BackendSupportsInput(BackendKind kind, const SolveInput& input, std::string* reason) {
  const BackendCapabilities caps = GetBackendCapabilities(kind);
  if (input.domain.nz > 1 && !caps.supports_3d) {
    if (reason) *reason = "3D domains are not supported";
    return false;
  }
  if (!input.pde.rhs_latex.empty() && !caps.supports_spatial_rhs) {
    if (reason) *reason = "spatial RHS expressions are not supported";
    return false;
  }
  if (HasVariableCoefficients(input.pde) && kind == BackendKind::TPU) {
    if (reason) *reason = "variable coefficients are not supported on this backend";
    return false;
  }
  if ((HasMixedDerivatives(input.pde) || HasHigherOrderDerivatives(input.pde)) &&
      kind != BackendKind::CPU) {
    if (reason) *reason = "mixed or higher-order derivatives are not supported on this backend";
    return false;
  }
  if (input.domain.coord_system != CoordinateSystem::Cartesian &&
      (HasMixedDerivatives(input.pde) || HasHigherOrderDerivatives(input.pde))) {
    if (reason) *reason = "mixed or higher-order derivatives require Cartesian coordinates";
    return false;
  }
  if (HasHigherOrderDerivatives(input.pde)) {
    const bool x_high = std::abs(input.pde.a3) > 1e-12 || std::abs(input.pde.a4) > 1e-12 ||
                        !input.pde.a3_latex.empty() || !input.pde.a4_latex.empty();
    const bool y_high = std::abs(input.pde.b3) > 1e-12 || std::abs(input.pde.b4) > 1e-12 ||
                        !input.pde.b3_latex.empty() || !input.pde.b4_latex.empty();
    const bool z_high = std::abs(input.pde.az3) > 1e-12 || std::abs(input.pde.az4) > 1e-12 ||
                        !input.pde.az3_latex.empty() || !input.pde.az4_latex.empty();
    if (x_high && input.domain.nx < 5) {
      if (reason) *reason = "higher-order x-derivatives require nx >= 5";
      return false;
    }
    if (y_high && input.domain.ny < 5) {
      if (reason) *reason = "higher-order y-derivatives require ny >= 5";
      return false;
    }
    if (z_high && input.domain.nz < 5) {
      if (reason) *reason = "higher-order z-derivatives require nz >= 5";
      return false;
    }
  }
  if ((!input.nonlinear.empty() || !input.nonlinear_derivatives.empty()) && !caps.supports_nonlinear) {
    if (reason) *reason = "nonlinear terms are not supported";
    return false;
  }
  if (!input.integrals.empty() && !caps.supports_integrals) {
    if (reason) *reason = "integral terms are not supported";
    return false;
  }
  if (HasImplicitShape(input) && !caps.supports_shapes) {
    if (reason) *reason = "implicit domain shapes are not supported";
    return false;
  }
  if ((input.time.enabled || std::abs(input.pde.ut) > 1e-12 || std::abs(input.pde.utt) > 1e-12) &&
      !caps.supports_time_dependent) {
    if (reason) *reason = "time-dependent PDEs are not supported";
    return false;
  }
  if (input.domain.nz > 1 &&
      (input.time.enabled || std::abs(input.pde.ut) > 1e-12 || std::abs(input.pde.utt) > 1e-12)) {
    if (reason) *reason = "time-dependent 3D solves are not supported";
    return false;
  }
  if (HasPiecewiseBC(input) && !caps.supports_piecewise_bc) {
    if (reason) *reason = "piecewise boundary conditions are not supported";
    return false;
  }
  if (!input.nonlinear_derivatives.empty()) {
    if (reason) *reason = "nonlinear derivative terms are not supported";
    return false;
  }
  return true;
}

bool BackendSupportsMethodForInput(BackendKind kind, SolveMethod method,
                                   const SolveInput& input, std::string* reason) {
  if (!BackendSupportsMethod(kind, method)) {
    if (reason) *reason = "selected method is not supported on this backend";
    return false;
  }
  if (!BackendSupportsInput(kind, input, reason)) {
    return false;
  }

  const bool linear = IsLinear(input);
  const bool dirichlet_only = AllDirichlet(input) && !HasPeriodicBoundary(input.domain);
  const bool is_krylov = (method == SolveMethod::CG || method == SolveMethod::BiCGStab ||
                          method == SolveMethod::GMRES);
  const bool is_multigrid = (method == SolveMethod::MultigridVcycle);
  const bool is_relaxation = (method == SolveMethod::Jacobi || method == SolveMethod::GaussSeidel ||
                              method == SolveMethod::SOR);
  const bool has_var_coeff = HasVariableCoefficients(input.pde);
  const bool has_high_order = HasHigherOrderDerivatives(input.pde);

  if ((is_krylov || is_multigrid) && !linear) {
    if (reason) *reason = "linear PDE required for this method";
    return false;
  }
  if ((is_krylov || is_multigrid) && HasImplicitShape(input)) {
    if (reason) *reason = "selected method does not support implicit domain shapes";
    return false;
  }
  if ((is_krylov || is_multigrid) && input.domain.nz > 1) {
    if (reason) *reason = "selected method supports 2D domains only";
    return false;
  }
  if (is_krylov && !dirichlet_only) {
    if (reason) *reason = "Krylov methods require Dirichlet boundaries on all sides";
    return false;
  }
  if (is_multigrid && !dirichlet_only) {
    if (reason) *reason = "Multigrid requires Dirichlet boundaries on all sides";
    return false;
  }
  if (has_var_coeff && kind != BackendKind::CPU) {
    if (kind == BackendKind::CUDA || kind == BackendKind::Metal) {
      if (!is_relaxation) {
        if (reason) *reason = "variable coefficients are only supported for relaxation methods on GPU";
        return false;
      }
      if (!input.pde.az_latex.empty() || !input.pde.dz_latex.empty()) {
        if (reason) *reason = "variable z coefficients are not supported on this backend";
        return false;
      }
    } else {
      if (reason) *reason = "variable coefficients are not supported on this backend";
      return false;
    }
  }

  if (method == SolveMethod::CG) {
    if (has_var_coeff || std::abs(input.pde.c) > 1e-12 || std::abs(input.pde.d) > 1e-12 ||
        std::abs(input.pde.ab) > 1e-12 || std::abs(input.pde.ac) > 1e-12 ||
        std::abs(input.pde.bc) > 1e-12 || has_high_order) {
      if (reason) *reason = "CG requires symmetric operators with constant coefficients (no convection/mixed/higher-order terms)";
      return false;
    }
  }

  if (is_multigrid) {
    if (has_var_coeff) {
      if (reason) *reason = "multigrid requires constant coefficients";
      return false;
    }
    if (!input.integrals.empty()) {
      if (reason) *reason = "multigrid does not support integral terms";
      return false;
    }
    if (std::abs(input.pde.c) > 1e-12 || std::abs(input.pde.d) > 1e-12 ||
        std::abs(input.pde.ab) > 1e-12 || std::abs(input.pde.ac) > 1e-12 ||
        std::abs(input.pde.bc) > 1e-12 || has_high_order) {
      if (reason) *reason = "multigrid requires no convection, mixed, or higher-order derivatives";
      return false;
    }
    if (input.domain.nx < 5 || input.domain.ny < 5) {
      if (reason) *reason = "multigrid requires grid dimensions of at least 5";
      return false;
    }
    if ((input.domain.nx % 2) == 0 || (input.domain.ny % 2) == 0) {
      if (reason) *reason = "multigrid requires odd grid dimensions";
      return false;
    }
  }

  return true;
}

std::vector<BackendKind> BackendsSupportingMethod(SolveMethod method) {
  std::vector<BackendKind> backends;
  const BackendKind kinds[] = {BackendKind::CPU, BackendKind::CUDA, BackendKind::Metal, BackendKind::TPU};
  for (BackendKind kind : kinds) {
    if (BackendSupportsMethod(kind, method)) {
      backends.push_back(kind);
    }
  }
  return backends;
}

bool BackendSupportsCoupling(BackendKind kind, CouplingPattern pattern, std::string* reason) {
  const BackendCapabilities caps = GetBackendCapabilities(kind);

  switch (pattern) {
    case CouplingPattern::SingleField:
      // All backends support single-field (no coupling)
      return true;

    case CouplingPattern::ExplicitCoupling:
      if (!caps.supports_coupled_fields) {
        if (reason) *reason = "coupled fields are not supported on this backend";
        return false;
      }
      if (!caps.supports_explicit_coupling) {
        if (reason) *reason = "explicit coupling is not supported on this backend";
        return false;
      }
      return true;

    case CouplingPattern::SymmetricCoupling:
      if (!caps.supports_coupled_fields) {
        if (reason) *reason = "coupled fields are not supported on this backend";
        return false;
      }
      if (!caps.supports_implicit_coupling) {
        if (reason) *reason = "symmetric/implicit coupling requires iterative coupling which is not supported";
        return false;
      }
      return true;

    case CouplingPattern::NonlinearCoupling:
      if (!caps.supports_coupled_fields) {
        if (reason) *reason = "coupled fields are not supported on this backend";
        return false;
      }
      if (!caps.supports_nonlinear) {
        if (reason) *reason = "nonlinear terms are not supported on this backend";
        return false;
      }
      if (!caps.supports_implicit_coupling) {
        if (reason) *reason = "nonlinear coupling requires iterative coupling which is not supported";
        return false;
      }
      return true;

    default:
      if (reason) *reason = "unknown coupling pattern";
      return false;
  }
}

bool BackendSupportsMultiFieldInput(BackendKind kind, const MultiFieldEquation& multi_eq,
                                    const CouplingAnalysis& coupling, std::string* reason) {
  const BackendCapabilities caps = GetBackendCapabilities(kind);

  // Check if we have multiple fields at all
  if (multi_eq.equations.size() <= 1 && !multi_eq.HasCoupling()) {
    // Single field or no coupling - standard validation applies
    return true;
  }

  // Check basic multi-field support
  if (!caps.supports_coupled_fields) {
    if (reason) *reason = "multi-field coupled PDEs are not supported on this backend";
    return false;
  }

  // Check coupling pattern support
  if (!BackendSupportsCoupling(kind, coupling.pattern, reason)) {
    return false;
  }

  // Check each field's equation for backend compatibility
  for (const auto& eq : multi_eq.equations) {
    // Check for variable coefficients in self terms
    if (HasVariableCoefficients(eq.self_coeffs) && kind != BackendKind::CPU) {
      if (reason) *reason = "variable coefficients in field '" + eq.field_name +
                            "' are not supported on this backend";
      return false;
    }

    // Check for higher-order derivatives
    if (HasHigherOrderDerivatives(eq.self_coeffs) && kind != BackendKind::CPU) {
      if (reason) *reason = "higher-order derivatives in field '" + eq.field_name +
                            "' are not supported on this backend";
      return false;
    }

    // Check coupled terms
    for (const auto& cross : eq.coupled) {
      if (HasVariableCoefficients(cross.coeffs) && kind != BackendKind::CPU) {
        if (reason) *reason = "variable coefficients in coupling term from '" +
                              cross.source_field + "' to '" + eq.field_name +
                              "' are not supported on this backend";
        return false;
      }

      if (HasHigherOrderDerivatives(cross.coeffs) && kind != BackendKind::CPU) {
        if (reason) *reason = "higher-order derivatives in coupling term from '" +
                              cross.source_field + "' to '" + eq.field_name +
                              "' are not supported on this backend";
        return false;
      }
    }
  }

  // Warn about circular dependencies if present
  if (coupling.has_circular_dependency) {
    // Not an error, but coupling strategy matters
    if (!caps.supports_implicit_coupling) {
      if (reason) *reason = "circular coupling requires iterative solving which is not supported";
      return false;
    }
  }

  return true;
}
