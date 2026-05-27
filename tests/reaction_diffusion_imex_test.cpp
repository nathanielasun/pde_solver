#include "latex_parser.h"
#include "solver.h"

#include <cmath>
#include <iostream>
#include <vector>

int main() {
  LatexParser parser;
  const std::string pde = "u_t + u u_x = 0.01 u_{xx}";
  LatexParseResult parse = parser.Parse(pde);
  if (!parse.ok) {
    std::cerr << "parse failed: " << parse.error << "\n";
    return 1;
  }

  SolveInput input;
  input.pde = parse.coeffs;
  input.nonlinear_derivatives = parse.nonlinear_derivatives;
  input.domain = Domain{0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 32, 32, 1};
  input.bc.left.kind = BCKind::Dirichlet;
  input.bc.right.kind = BCKind::Dirichlet;
  input.bc.bottom.kind = BCKind::Dirichlet;
  input.bc.top.kind = BCKind::Dirichlet;
  input.time.enabled = true;
  input.time.t_start = 0.0;
  input.time.t_end = 0.1;
  input.time.dt = 0.002;
  input.time.frames = 10;
  input.time.integrator = TimeIntegrator::IMEX;
  input.solver.method = SolveMethod::Jacobi;

  SolveOutput out = SolvePDETimeSeries(input, nullptr, nullptr);
  if (!out.error.empty()) {
    std::cerr << "solve failed: " << out.error << "\n";
    return 1;
  }
  double max_abs = 0.0;
  for (double v : out.grid) {
    max_abs = std::max(max_abs, std::abs(v));
  }
  if (!std::isfinite(max_abs) || max_abs > 10.0) {
    std::cerr << "blow-up detected max_abs=" << max_abs << "\n";
    return 1;
  }
  std::cout << "reaction_diffusion_imex_test ok (max_abs=" << max_abs << ")\n";
  return 0;
}
