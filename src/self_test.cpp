#include "self_test.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include "backend.h"
#include "mms.h"
#include "pde_types.h"
#include "solver.h"

namespace {
struct ErrorNorms {
  double l2 = 0.0;
  double linf = 0.0;
};

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

ErrorNorms ComputeErrorNorms2D(const Domain& d,
                               const std::vector<double>& grid,
                               const std::function<double(double, double)>& exact) {
  ErrorNorms out;
  if (grid.empty()) {
    return out;
  }
  const int nx = d.nx;
  const int ny = d.ny;
  const double dx = (d.xmax - d.xmin) / static_cast<double>(std::max(1, nx - 1));
  const double dy = (d.ymax - d.ymin) / static_cast<double>(std::max(1, ny - 1));
  KahanSum sum_sq;
  double max_abs = 0.0;
  for (int j = 0; j < ny; ++j) {
    const double y = d.ymin + j * dy;
    for (int i = 0; i < nx; ++i) {
      const double x = d.xmin + i * dx;
      const size_t idx = static_cast<size_t>(j * nx + i);
      if (idx >= grid.size()) {
        continue;
      }
      const double err = grid[idx] - exact(x, y);
      sum_sq.Add(err * err);
      max_abs = std::max(max_abs, std::abs(err));
    }
  }
  out.l2 = std::sqrt(std::max(0.0, sum_sq.sum));
  out.linf = max_abs;
  return out;
}

BoundaryCondition::Expression MakeLinearExpr(double constant, double x, double y, double z) {
  BoundaryCondition::Expression expr;
  expr.constant = constant;
  expr.x = x;
  expr.y = y;
  expr.z = z;
  return expr;
}

BoundaryCondition::Expression MakeLatexExpr(const std::string& latex) {
  BoundaryCondition::Expression expr;
  expr.latex = latex;
  return expr;
}

BoundarySet MakeDirichletAll(const BoundaryCondition::Expression& expr) {
  BoundarySet bc;
  bc.left.kind = BCKind::Dirichlet;
  bc.right.kind = BCKind::Dirichlet;
  bc.bottom.kind = BCKind::Dirichlet;
  bc.top.kind = BCKind::Dirichlet;
  bc.left.value = expr;
  bc.right.value = expr;
  bc.bottom.value = expr;
  bc.top.value = expr;
  return bc;
}

Domain MakeUnitDomain(int n) {
  Domain d;
  d.xmin = 0.0;
  d.xmax = 1.0;
  d.ymin = 0.0;
  d.ymax = 1.0;
  d.zmin = 0.0;
  d.zmax = 1.0;
  d.nx = n;
  d.ny = n;
  d.nz = 1;
  d.coord_system = CoordinateSystem::Cartesian;
  return d;
}

double DiscreteIntegralConstantValue(const Domain& d, double value) {
  const double dx = (d.xmax - d.xmin) / static_cast<double>(std::max(1, d.nx - 1));
  const double dy = (d.ymax - d.ymin) / static_cast<double>(std::max(1, d.ny - 1));
  return value * static_cast<double>(d.nx) * static_cast<double>(d.ny) * dx * dy;
}

void ConfigureSolver(SolveInput* input, SolveMethod method, int max_iter, double tol) {
  if (!input) {
    return;
  }
  input->solver = {};
  input->solver.method = method;
  input->solver.max_iter = max_iter;
  input->solver.tol = tol;
  input->solver.sor_omega = 1.7;
  input->solver.gmres_restart = 30;
  input->solver.residual_interval = 0;
}

struct SelfTestCase {
  std::string name;
  SolveInput input;
  std::function<double(double, double)> exact;
  double max_error = 1e-6;
};

bool RunCase(const SelfTestCase& test) {
  BackendKind selected = BackendKind::CPU;
  std::string note;
  SolveOutput out = SolveWithBackend(test.input, BackendKind::CPU, &selected, &note);
  if (!out.error.empty()) {
    std::cout << "[FAIL] " << test.name << ": " << out.error << "\n";
    return false;
  }
  ErrorNorms err = ComputeErrorNorms2D(test.input.domain, out.grid, test.exact);
  const bool pass = err.linf <= test.max_error;
  if (pass) {
    std::cout << "[OK] " << test.name
              << " err2=" << err.l2
              << " errinf=" << err.linf
              << " resid2=" << out.residual_l2
              << " residinf=" << out.residual_linf;
    if (!note.empty()) {
      std::cout << " note=" << note;
    }
    std::cout << "\n";
  } else {
    std::cout << "[FAIL] " << test.name
              << " err2=" << err.l2
              << " errinf=" << err.linf
              << " tol=" << test.max_error
              << " resid2=" << out.residual_l2
              << " residinf=" << out.residual_linf;
    if (!note.empty()) {
      std::cout << " note=" << note;
    }
    std::cout << "\n";
  }
  return pass;
}

bool RunMmsConvergenceTest() {
  const ManufacturedSolution sol = BuildDefaultManufacturedSolution(2);
  const auto exact = [&](double x, double y) {
    return sol.eval(x, y, 0.0);
  };

  SolveInput input;
  input.pde.a = -1.0;
  input.pde.b = -1.0;
  input.bc = MakeDirichletAll(MakeLatexExpr(sol.u_latex));
  ConfigureSolver(&input, SolveMethod::CG, 8000, 1e-10);

  ManufacturedRhsResult rhs = BuildManufacturedRhs(input.pde, 2, sol);
  if (!rhs.ok) {
    std::cout << "[FAIL] MMS RHS build: " << rhs.error << "\n";
    return false;
  }
  input.pde.rhs_latex = rhs.rhs_latex;
  input.pde.f = 0.0;

  const Domain coarse = MakeUnitDomain(17);
  const Domain fine = MakeUnitDomain(33);

  input.domain = coarse;
  SolveOutput out_coarse = SolveWithBackend(input, BackendKind::CPU, nullptr, nullptr);
  if (!out_coarse.error.empty()) {
    std::cout << "[FAIL] MMS coarse solve: " << out_coarse.error << "\n";
    return false;
  }
  const ErrorNorms err_coarse = ComputeErrorNorms2D(coarse, out_coarse.grid, exact);

  input.domain = fine;
  SolveOutput out_fine = SolveWithBackend(input, BackendKind::CPU, nullptr, nullptr);
  if (!out_fine.error.empty()) {
    std::cout << "[FAIL] MMS fine solve: " << out_fine.error << "\n";
    return false;
  }
  const ErrorNorms err_fine = ComputeErrorNorms2D(fine, out_fine.grid, exact);

  const double ratio = err_coarse.linf / std::max(1e-14, err_fine.linf);
  const bool pass = ratio > 3.0;
  if (pass) {
    std::cout << "[OK] MMS convergence (CG) "
              << "errinf_coarse=" << err_coarse.linf
              << " errinf_fine=" << err_fine.linf
              << " ratio=" << ratio << "\n";
  } else {
    std::cout << "[FAIL] MMS convergence (CG) "
              << "errinf_coarse=" << err_coarse.linf
              << " errinf_fine=" << err_fine.linf
              << " ratio=" << ratio << "\n";
  }
  return pass;
}
}  // namespace

int RunSelfTest() {
  std::cout << "Running self-test (solver coverage + regression suite)...\n";

  int failures = 0;
  int total = 0;

  const Domain method_domain = MakeUnitDomain(33);
  SolveInput harmonic;
  harmonic.domain = method_domain;
  harmonic.pde.a = -1.0;
  harmonic.pde.b = -1.0;
  harmonic.pde.f = 0.0;
  harmonic.bc = MakeDirichletAll(MakeLinearExpr(0.0, 1.0, 2.0, 0.0));

  SolveInput zero = harmonic;
  zero.bc = MakeDirichletAll(MakeLinearExpr(0.0, 0.0, 0.0, 0.0));

  const auto exact_harmonic = [](double x, double y) {
    return x + 2.0 * y;
  };
  const auto exact_zero = [](double, double) {
    return 0.0;
  };

  std::cout << "Solver coverage:\n";
  struct MethodCase { SolveMethod method; const char* name; };
  const MethodCase methods[] = {
      {SolveMethod::SOR, "SOR"},
      {SolveMethod::CG, "CG"},
      {SolveMethod::BiCGStab, "BiCGStab"},
      {SolveMethod::GMRES, "GMRES"},
      {SolveMethod::MultigridVcycle, "MG"},
  };

  for (const auto& m : methods) {
    const bool is_krylov =
        (m.method == SolveMethod::CG || m.method == SolveMethod::BiCGStab ||
         m.method == SolveMethod::GMRES);
    SolveInput input = is_krylov ? zero : harmonic;
    ConfigureSolver(&input, m.method,
                    (m.method == SolveMethod::MultigridVcycle) ? 120 : 4000,
                    1e-8);
    const double threshold =
        is_krylov ? 1e-8 : (m.method == SolveMethod::MultigridVcycle ? 5e-6 : 1e-6);
    SelfTestCase test;
    test.name = std::string(is_krylov ? "Poisson zero (" : "Poisson harmonic (") + m.name + ")";
    test.input = std::move(input);
    test.exact = is_krylov ? exact_zero : exact_harmonic;
    test.max_error = threshold;
    total++;
    if (!RunCase(test)) {
      failures++;
    }
  }

  std::cout << "Regression PDE suite:\n";
  const Domain reg_domain = MakeUnitDomain(21);
  const auto exact_const = [](double, double) {
    return 1.0;
  };
  const auto exact_quadratic = [](double x, double y) {
    return x * x + y * y;
  };
  const auto exact_mixed = [](double x, double y) {
    return x * y;
  };

  const int reg_iters = 6000;
  const double reg_tol = 1e-10;

  {
    SolveInput input;
    input.domain = reg_domain;
    input.pde.a = -1.0;
    input.pde.b = -1.0;
    input.pde.e = 1.0;
    input.pde.f = -1.0;
    input.bc = MakeDirichletAll(MakeLinearExpr(1.0, 0.0, 0.0, 0.0));
    ConfigureSolver(&input, SolveMethod::SOR, reg_iters, reg_tol);

    SelfTestCase test;
    test.name = "Helmholtz constant (SOR)";
    test.input = std::move(input);
    test.exact = exact_const;
    test.max_error = 1e-7;
    total++;
    if (!RunCase(test)) {
      failures++;
    }
  }

  {
    SolveInput input;
    input.domain = reg_domain;
    input.pde.a = -2.0;
    input.pde.b = -3.0;
    input.pde.f = 10.0;
    input.bc = MakeDirichletAll(MakeLatexExpr("x^2 + y^2"));
    ConfigureSolver(&input, SolveMethod::SOR, reg_iters, reg_tol);

    SelfTestCase test;
    test.name = "Anisotropic diffusion (SOR)";
    test.input = std::move(input);
    test.exact = exact_quadratic;
    test.max_error = 1e-7;
    total++;
    if (!RunCase(test)) {
      failures++;
    }
  }

  {
    SolveInput input;
    input.domain = reg_domain;
    input.pde.a = -1.0;
    input.pde.b = -1.0;
    input.pde.ab = 1.0;
    input.pde.f = -1.0;
    input.bc = MakeDirichletAll(MakeLatexExpr("x*y"));
    ConfigureSolver(&input, SolveMethod::SOR, reg_iters, reg_tol);

    SelfTestCase test;
    test.name = "Mixed derivatives (SOR)";
    test.input = std::move(input);
    test.exact = exact_mixed;
    test.max_error = 1e-7;
    total++;
    if (!RunCase(test)) {
      failures++;
    }
  }

  {
    SolveInput input;
    input.domain = reg_domain;
    input.pde.e = 1.0;
    const double coeff = 0.5;
    const double integral_value = DiscreteIntegralConstantValue(reg_domain, 1.0);
    input.pde.f = -(1.0 + coeff * integral_value);
    input.integrals.push_back({coeff, ""});
    input.bc = MakeDirichletAll(MakeLinearExpr(1.0, 0.0, 0.0, 0.0));
    ConfigureSolver(&input, SolveMethod::SOR, reg_iters, reg_tol);

    SelfTestCase test;
    test.name = "Integral term (SOR)";
    test.input = std::move(input);
    test.exact = exact_const;
    test.max_error = 5e-6;
    total++;
    if (!RunCase(test)) {
      failures++;
    }
  }

  {
    SolveInput input;
    input.domain = reg_domain;
    input.pde.e = 1.0;
    input.pde.f = -2.0;
    input.nonlinear.push_back({NonlinearKind::Power, 1.0, 2});
    input.bc = MakeDirichletAll(MakeLinearExpr(1.0, 0.0, 0.0, 0.0));
    ConfigureSolver(&input, SolveMethod::SOR, reg_iters, reg_tol);

    SelfTestCase test;
    test.name = "Nonlinear term (SOR)";
    test.input = std::move(input);
    test.exact = exact_const;
    test.max_error = 5e-6;
    total++;
    if (!RunCase(test)) {
      failures++;
    }
  }

  std::cout << "MMS convergence:\n";
  total++;
  if (!RunMmsConvergenceTest()) {
    failures++;
  }

  if (failures == 0) {
    std::cout << "Self-test PASSED (" << total << " cases)\n";
    return 0;
  }
  std::cout << "Self-test FAILED (" << failures << " failing cases out of "
            << total << ")\n";
  return 1;
}
