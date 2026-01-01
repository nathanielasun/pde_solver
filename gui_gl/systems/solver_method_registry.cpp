#include "solver_method_registry.h"
#include "backend_capability_matrix.h"
#include <algorithm>

SolverMethodRegistry& SolverMethodRegistry::Instance() {
  static SolverMethodRegistry instance;
  if (!instance.initialized_) {
    instance.InitializeBuiltInMethods();
    instance.initialized_ = true;
  }
  return instance;
}

void SolverMethodRegistry::Register(const SolverMethodMetadata& metadata) {
  methods_[metadata.id] = metadata;
}

std::vector<SolveMethod> SolverMethodRegistry::GetMethods() const {
  std::vector<SolveMethod> result;
  result.reserve(methods_.size());
  for (const auto& pair : methods_) {
    result.push_back(pair.first);
  }
  return result;
}

const SolverMethodMetadata* SolverMethodRegistry::GetMetadata(SolveMethod method) const {
  auto it = methods_.find(method);
  if (it != methods_.end()) {
    return &it->second;
  }
  return nullptr;
}

std::vector<SolveMethod> SolverMethodRegistry::GetMethodsForBackend(BackendKind backend) const {
  std::vector<SolveMethod> result;
  for (const auto& pair : methods_) {
    const auto& metadata = pair.second;
    if (std::find(metadata.supported_backends.begin(), 
                  metadata.supported_backends.end(), 
                  backend) != metadata.supported_backends.end()) {
      result.push_back(pair.first);
    }
  }
  return result;
}

bool SolverMethodRegistry::IsApplicable(SolveMethod method, const SolveInput& input) const {
  const SolverMethodMetadata* metadata = GetMetadata(method);
  if (!metadata || !metadata->is_applicable) {
    return true;  // Default: assume applicable
  }
  return metadata->is_applicable(input);
}

SolveMethod SolverMethodRegistry::RecommendMethod(const SolveInput& input, BackendKind preferred_backend) const {
  // Simple heuristic: prefer faster methods for linear problems
  // For nonlinear problems, use basic iterative methods
  
  bool is_linear = input.nonlinear.empty() && 
                   input.nonlinear_derivatives.empty() &&
                   input.integrals.empty();
  
  bool has_time = input.time.enabled;
  
  // For time-dependent problems, prefer basic methods
  if (has_time) {
    return SolveMethod::Jacobi;  // Simple and stable
  }
  
  // For linear problems, prefer faster methods
  if (is_linear) {
    // Check if problem is symmetric (simplified check)
    bool likely_symmetric = (input.pde.a == input.pde.b) && 
                           (input.pde.ab == 0.0) &&
                           input.pde.c == 0.0;
    
    if (likely_symmetric) {
      // Try CG first (fastest for SPD)
      if (IsApplicable(SolveMethod::CG, input)) {
        return SolveMethod::CG;
      }
    }
    
    // Try multigrid (very fast for large problems)
    if (IsApplicable(SolveMethod::MultigridVcycle, input)) {
      return SolveMethod::MultigridVcycle;
    }
    
    // Try GMRES (general purpose)
    if (IsApplicable(SolveMethod::GMRES, input)) {
      return SolveMethod::GMRES;
    }
  }
  
  // Default to Jacobi (always available)
  return SolveMethod::Jacobi;
}

void SolverMethodRegistry::InitializeBuiltInMethods() {
  // Jacobi Method
  SolverMethodMetadata jacobi;
  jacobi.id = SolveMethod::Jacobi;
  jacobi.name = "Jacobi";
  jacobi.short_name = "jacobi";
  jacobi.description = "Basic point-wise iterative method";
  jacobi.detailed_description = 
      "The Jacobi method is a simple iterative solver that updates each grid point "
      "independently using values from the previous iteration. It is the most basic "
      "iterative method and is available on all backends. Convergence is typically slow "
      "but the method is very stable and easy to parallelize.";
  jacobi.supported_backends = BackendsSupportingMethod(SolveMethod::Jacobi);
  jacobi.default_max_iter = 10000;
  jacobi.default_tol = 1e-6;
  jacobi.is_applicable = [](const SolveInput&) { return true; };  // Always applicable
  jacobi.recommendation_reason = "Most stable, works for all problem types";
  jacobi.best_use_cases = {
    "Nonlinear problems",
    "Time-dependent problems",
    "Problems with complex boundary conditions",
    "When stability is more important than speed"
  };
  jacobi.limitations = {
    "Slow convergence",
    "May require many iterations for high accuracy"
  };
  Register(jacobi);
  
  // Gauss-Seidel Method
  SolverMethodMetadata gauss_seidel;
  gauss_seidel.id = SolveMethod::GaussSeidel;
  gauss_seidel.name = "Gauss-Seidel";
  gauss_seidel.short_name = "gs";
  gauss_seidel.description = "Sequential updates using latest values";
  gauss_seidel.detailed_description =
      "The Gauss-Seidel method updates each grid point using the most recently computed "
      "values from the current iteration. This typically converges faster than Jacobi "
      "but is harder to parallelize due to data dependencies.";
  gauss_seidel.supported_backends = BackendsSupportingMethod(SolveMethod::GaussSeidel);
  gauss_seidel.default_max_iter = 10000;
  gauss_seidel.default_tol = 1e-6;
  gauss_seidel.is_applicable = [](const SolveInput&) { return true; };
  gauss_seidel.recommendation_reason = "Faster than Jacobi, good for steady-state problems";
  gauss_seidel.best_use_cases = {
    "Steady-state linear problems",
    "2D problems",
    "When faster convergence than Jacobi is needed"
  };
  gauss_seidel.limitations = {
    "Slower on GPUs due to data dependencies",
    "May not converge for some nonlinear problems"
  };
  Register(gauss_seidel);
  
  // SOR Method
  SolverMethodMetadata sor;
  sor.id = SolveMethod::SOR;
  sor.name = "SOR (Successive Over-Relaxation)";
  sor.short_name = "sor";
  sor.description = "Gauss-Seidel with relaxation parameter";
  sor.detailed_description =
      "SOR extends Gauss-Seidel by introducing a relaxation parameter omega (ω). "
      "When ω = 1, SOR is equivalent to Gauss-Seidel. Optimal ω values (typically 1.5-1.9) "
      "can significantly accelerate convergence. Requires tuning the omega parameter.";
  sor.supported_backends = BackendsSupportingMethod(SolveMethod::SOR);
  sor.default_max_iter = 10000;
  sor.default_tol = 1e-6;
  sor.options.push_back({
    "omega", "double", "Relaxation parameter (1.0 = Gauss-Seidel, optimal ~1.5-1.9)",
    std::any(1.5), std::any(0.1), std::any(2.0)
  });
  sor.is_applicable = [](const SolveInput&) { return true; };
  sor.recommendation_reason = "Fast convergence with optimal omega";
  sor.best_use_cases = {
    "Steady-state linear problems",
    "When convergence speed is important",
    "2D elliptic problems"
  };
  sor.limitations = {
    "Requires tuning omega parameter",
    "Optimal omega depends on problem"
  };
  Register(sor);
  
  // Conjugate Gradient (CG)
  SolverMethodMetadata cg;
  cg.id = SolveMethod::CG;
  cg.name = "Conjugate Gradient (CG)";
  cg.short_name = "cg";
  cg.description = "Krylov method for symmetric positive-definite systems";
  cg.detailed_description =
      "The Conjugate Gradient method is a Krylov subspace method designed for symmetric "
      "positive-definite (SPD) linear systems. It typically converges much faster than "
      "iterative methods like Jacobi or Gauss-Seidel, often in O(√n) iterations. "
      "Requires linear PDEs with Dirichlet boundary conditions only.";
  cg.supported_backends = BackendsSupportingMethod(SolveMethod::CG);
  cg.default_max_iter = 1000;  // Usually converges much faster
  cg.default_tol = 1e-8;
  cg.is_applicable = [](const SolveInput& input) {
    // CG requires linear systems
    if (!input.nonlinear.empty() || !input.nonlinear_derivatives.empty()) {
      return false;
    }
    // CG works best with Dirichlet BCs (simplified check)
    // In practice, CG can work with other BCs but may be less stable
    return true;  // Let the solver decide
  };
  cg.recommendation_reason = "Very fast for symmetric positive-definite problems";
  cg.best_use_cases = {
    "Symmetric positive-definite linear systems",
    "Large sparse systems",
    "When fast convergence is critical"
  };
  cg.limitations = {
    "Requires linear PDEs",
    "Best for symmetric systems",
    "GPU implementation limited to 2D",
    "May not work well with Neumann/Robin BCs"
  };
  Register(cg);
  
  // BiCGStab Method
  SolverMethodMetadata bicgstab;
  bicgstab.id = SolveMethod::BiCGStab;
  bicgstab.name = "BiCGStab (Bi-Conjugate Gradient Stabilized)";
  bicgstab.short_name = "bicgstab";
  bicgstab.description = "Krylov method for general (non-symmetric) systems";
  bicgstab.detailed_description =
      "BiCGStab is a Krylov subspace method that can handle non-symmetric linear systems. "
      "It is more general than CG but typically requires more iterations. Useful for "
      "problems with mixed derivative terms or non-symmetric operators.";
  bicgstab.supported_backends = BackendsSupportingMethod(SolveMethod::BiCGStab);
  bicgstab.default_max_iter = 2000;
  bicgstab.default_tol = 1e-8;
  bicgstab.is_applicable = [](const SolveInput& input) {
    // BiCGStab requires linear systems
    return input.nonlinear.empty() && input.nonlinear_derivatives.empty();
  };
  bicgstab.recommendation_reason = "Handles non-symmetric linear systems";
  bicgstab.best_use_cases = {
    "Non-symmetric linear systems",
    "Problems with mixed derivatives",
    "General linear PDEs"
  };
  bicgstab.limitations = {
    "Requires linear PDEs",
    "GPU implementation limited to 2D",
    "May require more iterations than CG for symmetric problems"
  };
  Register(bicgstab);
  
  // GMRES Method
  SolverMethodMetadata gmres;
  gmres.id = SolveMethod::GMRES;
  gmres.name = "GMRES (Generalized Minimal Residual)";
  gmres.short_name = "gmres";
  gmres.description = "Krylov method with restart for general systems";
  gmres.detailed_description =
      "GMRES is a flexible Krylov subspace method that works for general linear systems. "
      "It uses a restart parameter to limit memory usage. Typically converges faster than "
      "iterative methods but requires more memory per iteration.";
  gmres.supported_backends = BackendsSupportingMethod(SolveMethod::GMRES);
  gmres.default_max_iter = 2000;
  gmres.default_tol = 1e-8;
  gmres.options.push_back({
    "restart", "int", "GMRES restart parameter (number of iterations before restart)",
    std::any(30), std::any(1), std::any(200)
  });
  gmres.is_applicable = [](const SolveInput& input) {
    // GMRES requires linear systems
    return input.nonlinear.empty() && input.nonlinear_derivatives.empty();
  };
  gmres.recommendation_reason = "Flexible and robust for general linear systems";
  gmres.best_use_cases = {
    "General linear systems",
    "Non-symmetric problems",
    "When other Krylov methods fail"
  };
  gmres.limitations = {
    "Requires linear PDEs",
    "GPU implementation limited to 2D",
    "Memory usage grows with restart parameter"
  };
  Register(gmres);
  
  // Multigrid V-cycle
  SolverMethodMetadata multigrid;
  multigrid.id = SolveMethod::MultigridVcycle;
  multigrid.name = "Multigrid V-cycle";
  multigrid.short_name = "mg";
  multigrid.description = "Multilevel method for fast convergence";
  multigrid.detailed_description =
      "Multigrid V-cycle uses a hierarchy of grid resolutions to accelerate convergence. "
      "It typically converges in O(n) operations, making it one of the fastest methods for "
      "large problems. Requires linear PDEs and works best with smooth solutions.";
  multigrid.supported_backends = BackendsSupportingMethod(SolveMethod::MultigridVcycle);
  multigrid.default_max_iter = 100;  // Usually converges very quickly
  multigrid.default_tol = 1e-8;
  multigrid.options.push_back({
    "pre_smooth", "int", "Pre-smoothing iterations per level",
    std::any(2), std::any(1), std::any(10)
  });
  multigrid.options.push_back({
    "post_smooth", "int", "Post-smoothing iterations per level",
    std::any(2), std::any(1), std::any(10)
  });
  multigrid.options.push_back({
    "coarse_iters", "int", "Iterations on coarsest level",
    std::any(50), std::any(10), std::any(200)
  });
  multigrid.options.push_back({
    "max_levels", "int", "Maximum number of multigrid levels",
    std::any(10), std::any(2), std::any(20)
  });
  multigrid.is_applicable = [](const SolveInput& input) {
    // Multigrid requires linear systems
    return input.nonlinear.empty() && input.nonlinear_derivatives.empty();
  };
  multigrid.recommendation_reason = "Fastest convergence for large linear problems";
  multigrid.best_use_cases = {
    "Large linear systems",
    "Smooth solutions",
    "When fastest convergence is needed"
  };
  multigrid.limitations = {
    "Requires linear PDEs",
    "GPU implementation limited to 2D",
    "May not work well for highly oscillatory solutions"
  };
  Register(multigrid);
}
