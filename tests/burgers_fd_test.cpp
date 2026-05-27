#include "latex_parser.h"
#include "solver.h"
#include "input_parse.h"

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
  input.domain = Domain{0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 64, 64, 1};
  input.bc.left.kind = BCKind::Dirichlet;
  input.bc.right.kind = BCKind::Dirichlet;
  input.bc.bottom.kind = BCKind::Dirichlet;
  input.bc.top.kind = BCKind::Dirichlet;
  input.time.enabled = true;
  input.time.t_start = 0.0;
  input.time.t_end = 0.05;
  input.time.dt = 0.001;
  input.time.frames = 5;
  input.solver.method = SolveMethod::Jacobi;

  SolveOutput out = SolvePDETimeSeries(input, nullptr, nullptr);
  if (!out.error.empty()) {
    std::cerr << "solve failed: " << out.error << "\n";
    return 1;
  }
  if (out.grid.empty()) {
    std::cerr << "empty grid\n";
    return 1;
  }
  std::cout << "burgers_fd_test ok (grid size=" << out.grid.size() << ")\n";
  return 0;
}
