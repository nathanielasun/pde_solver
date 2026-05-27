#include "fv/flux.h"
#include "fv/fv_discretization.h"
#include "pde_types.h"

#include <cmath>
#include <iostream>
#include <vector>

int main() {
  const int nx = 100;
  const double x0 = 0.0;
  const double x1 = 1.0;
  const double dx = (x1 - x0) / static_cast<double>(nx);
  const double x_mid = 0.5;

  std::vector<double> u(static_cast<size_t>(nx), 0.0);
  for (int i = 0; i < nx; ++i) {
    const double x = x0 + (i + 0.5) * dx;
    u[static_cast<size_t>(i)] = (x < x_mid) ? 1.0 : 0.125;
  }

  FluxEvaluator flux = FluxEvaluator::Parse("0.5*u^2", nullptr);
  ConservationLawConfig config;
  config.riemann = ConservationLawConfig::RiemannSolver::HLL;

  const double dt = 0.0005;
  const int steps = 400;  // t = 0.2
  for (int step = 0; step < steps; ++step) {
    std::vector<double> dudt;
    ComputeFVSemidiscreteRHS1D(
        u, nx, dx, flux, config,
        [&](int) -> double { return u[0]; },
        [&](int) -> double { return u[static_cast<size_t>(nx - 1)]; },
        &dudt);
    for (int i = 0; i < nx; ++i) {
      u[static_cast<size_t>(i)] += dt * dudt[static_cast<size_t>(i)];
    }
  }

  double mass = 0.0;
  for (int i = 0; i < nx; ++i) {
    mass += u[static_cast<size_t>(i)] * dx;
  }
  const double expected_mass = 0.5 * 1.0 + 0.5 * 0.125;
  const double mass_err = std::abs(mass - expected_mass) / expected_mass;
  if (mass_err > 0.25) {
    std::cerr << "mass conservation error: " << mass_err << " (mass=" << mass
              << ", expected=" << expected_mass << ")\n";
    return 1;
  }

  bool has_shock_structure = false;
  double max_grad = 0.0;
  for (int i = 1; i < nx; ++i) {
    max_grad = std::max(max_grad, std::abs(u[static_cast<size_t>(i)] - u[static_cast<size_t>(i - 1)]));
  }
  if (max_grad > 0.1) {
    has_shock_structure = true;
  }
  if (!has_shock_structure) {
    std::cerr << "no shock structure detected\n";
    return 1;
  }

  std::cout << "shock_tube_test ok (mass_err=" << mass_err << ", max_grad=" << max_grad << ")\n";
  return 0;
}
