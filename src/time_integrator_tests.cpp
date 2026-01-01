#include "time_integrator_tests.h"

#include <cmath>
#include <iostream>
#include <iomanip>
#include <algorithm>

#include "time_integrator.h"

// Test exponential decay: du/dt = -lambda * u
// Exact solution: u(t) = u0 * exp(-lambda * t)
TimeIntegratorTestResult RunExponentialDecayTest(
    TimeIntegrator method,
    double lambda,
    double t_final,
    int num_steps) {

  TimeIntegratorTestResult result;
  result.name = "ExponentialDecay";
  result.description = "du/dt = -lambda*u, u(0)=1";
  result.method = method;
  result.t_final = t_final;

  double dt = t_final / num_steps;
  result.dt = dt;

  // Initial condition
  double u0 = 1.0;
  std::vector<double> u = {u0};

  // RHS function: du/dt = -lambda * u
  auto rhs = [lambda](double /*t*/, const std::vector<double>& u_vec,
                       std::vector<double>* dudt) {
    (*dudt)[0] = -lambda * u_vec[0];
  };

  TimeIntegratorConfig config;
  config.method = method;

  int total_rhs_evals = 0;
  double t = 0.0;

  for (int step = 0; step < num_steps; ++step) {
    TimeStepResult step_result = TimeStep(u, t, dt, rhs, config);
    if (!step_result.success) {
      result.passed = false;
      result.note = "Time step failed: " + step_result.error;
      return result;
    }
    total_rhs_evals += step_result.rhs_evals;
    t += dt;
  }

  // Exact solution
  double u_exact = u0 * std::exp(-lambda * t_final);

  // Error
  double error = std::abs(u[0] - u_exact);
  result.error_l2 = error;
  result.error_linf = error;
  result.rhs_evals = total_rhs_evals;

  // Pass if error is reasonable for the method's order
  int order = GetOrder(method);
  double expected_error = std::pow(dt, order);
  result.passed = (error < 10.0 * expected_error);

  if (result.passed) {
    result.note = "Error within expected bounds for order " + std::to_string(order);
  } else {
    result.note = "Error " + std::to_string(error) + " exceeds expected " +
                  std::to_string(10.0 * expected_error);
  }

  return result;
}

// Test harmonic oscillator: d²u/dt² = -omega² * u
// Converted to first-order system: du/dt = v, dv/dt = -omega² * u
TimeIntegratorTestResult RunHarmonicOscillatorTest(
    TimeIntegrator method,
    double omega,
    int num_periods,
    int steps_per_period) {

  TimeIntegratorTestResult result;
  result.name = "HarmonicOscillator";
  result.description = "d²u/dt² = -omega²*u, system form";
  result.method = method;

  double period = 2.0 * M_PI / omega;
  double t_final = num_periods * period;
  int num_steps = num_periods * steps_per_period;
  double dt = t_final / num_steps;

  result.t_final = t_final;
  result.dt = dt;

  // Initial conditions: u(0) = 1, v(0) = 0
  double u0 = 1.0;
  double v0 = 0.0;
  std::vector<double> state = {u0, v0};  // [u, v]

  // RHS function: du/dt = v, dv/dt = -omega² * u
  auto rhs = [omega](double /*t*/, const std::vector<double>& s,
                      std::vector<double>* dsdt) {
    (*dsdt)[0] = s[1];                    // du/dt = v
    (*dsdt)[1] = -omega * omega * s[0];   // dv/dt = -omega² * u
  };

  TimeIntegratorConfig config;
  config.method = method;

  int total_rhs_evals = 0;
  double t = 0.0;

  // Track energy drift
  double initial_energy = 0.5 * (u0 * u0 + (v0 / omega) * (v0 / omega));
  double max_energy_error = 0.0;

  for (int step = 0; step < num_steps; ++step) {
    TimeStepResult step_result = TimeStep(state, t, dt, rhs, config);
    if (!step_result.success) {
      result.passed = false;
      result.note = "Time step failed: " + step_result.error;
      return result;
    }
    total_rhs_evals += step_result.rhs_evals;
    t += dt;

    // Check energy
    double energy = 0.5 * (state[0] * state[0] +
                          (state[1] / omega) * (state[1] / omega));
    double energy_error = std::abs(energy - initial_energy) / initial_energy;
    max_energy_error = std::max(max_energy_error, energy_error);
  }

  // Exact solution at t_final (should return to initial state)
  double u_exact = u0 * std::cos(omega * t_final);
  double v_exact = -u0 * omega * std::sin(omega * t_final);

  // Errors
  double error_u = std::abs(state[0] - u_exact);
  double error_v = std::abs(state[1] - v_exact);
  result.error_l2 = std::sqrt(error_u * error_u + error_v * error_v);
  result.error_linf = std::max(error_u, error_v);
  result.rhs_evals = total_rhs_evals;

  // Pass criteria: solution should return close to initial condition
  // after full periods
  int order = GetOrder(method);
  double tol = std::pow(dt, order) * num_periods * 10.0;
  result.passed = (result.error_l2 < std::max(tol, 0.1));

  result.note = "Energy drift: " + std::to_string(max_energy_error * 100.0) + "%";

  return result;
}

// Test advection equation: du/dt = -c * du/dx (MOL discretization)
TimeIntegratorTestResult RunAdvectionEquationTest(
    TimeIntegrator method,
    int nx,
    double velocity,
    double t_final,
    double cfl) {

  TimeIntegratorTestResult result;
  result.name = "AdvectionMOL";
  result.description = "du/dt = -c*du/dx with upwind spatial";
  result.method = method;
  result.t_final = t_final;

  // Domain [0, 1] with periodic BC
  double dx = 1.0 / nx;
  double dt = cfl * dx / std::abs(velocity);
  int num_steps = static_cast<int>(std::ceil(t_final / dt));
  dt = t_final / num_steps;  // Adjust to hit t_final exactly

  result.dt = dt;

  // Initial condition: Gaussian
  std::vector<double> u(nx);
  double x0 = 0.5;
  double sigma = 0.1;
  for (int i = 0; i < nx; ++i) {
    double x = (i + 0.5) * dx;
    u[i] = std::exp(-((x - x0) * (x - x0)) / (2.0 * sigma * sigma));
  }

  // Store initial for comparison
  std::vector<double> u_initial = u;

  // RHS function: du/dt = -c * du/dx with upwind
  auto rhs = [nx, dx, velocity](double /*t*/, const std::vector<double>& u_vec,
                                 std::vector<double>* dudt) {
    for (int i = 0; i < nx; ++i) {
      int im = (i - 1 + nx) % nx;  // Periodic
      int ip = (i + 1) % nx;

      double dudx;
      if (velocity > 0) {
        // Upwind: use left neighbor
        dudx = (u_vec[i] - u_vec[im]) / dx;
      } else {
        // Upwind: use right neighbor
        dudx = (u_vec[ip] - u_vec[i]) / dx;
      }
      (*dudt)[i] = -velocity * dudx;
    }
  };

  TimeIntegratorConfig config;
  config.method = method;

  int total_rhs_evals = 0;
  double t = 0.0;

  for (int step = 0; step < num_steps; ++step) {
    TimeStepResult step_result = TimeStep(u, t, dt, rhs, config);
    if (!step_result.success) {
      result.passed = false;
      result.note = "Time step failed: " + step_result.error;
      return result;
    }
    total_rhs_evals += step_result.rhs_evals;
    t += dt;
  }

  // Exact solution: shifted Gaussian (with periodic wrapping)
  double shift = velocity * t_final;
  shift = shift - std::floor(shift);  // Wrap to [0, 1]

  double error_l2 = 0.0;
  double error_linf = 0.0;

  for (int i = 0; i < nx; ++i) {
    double x = (i + 0.5) * dx;
    double x_orig = x - shift;
    if (x_orig < 0) x_orig += 1.0;
    if (x_orig >= 1.0) x_orig -= 1.0;

    double u_exact = std::exp(-((x_orig - x0) * (x_orig - x0)) /
                              (2.0 * sigma * sigma));
    double err = std::abs(u[i] - u_exact);
    error_l2 += err * err;
    error_linf = std::max(error_linf, err);
  }
  error_l2 = std::sqrt(error_l2 * dx);

  result.error_l2 = error_l2;
  result.error_linf = error_linf;
  result.rhs_evals = total_rhs_evals;

  // Pass if reasonable for method order
  int order = GetOrder(method);
  // Spatial error is O(dx) for upwind, so temporal error should dominate
  // for sufficiently small CFL
  result.passed = (error_l2 < 0.2);  // 20% L2 error tolerance

  result.note = "CFL=" + std::to_string(cfl) + ", L2=" +
                std::to_string(error_l2);

  return result;
}

// Test stiff decay equation: du/dt = -k * (u - u_eq)
TimeIntegratorTestResult RunStiffDecayTest(
    TimeIntegrator method,
    double k,
    double t_final,
    int num_steps) {

  TimeIntegratorTestResult result;
  result.name = "StiffDecay";
  result.description = "du/dt = -k*(u - u_eq), k=" + std::to_string(static_cast<int>(k));
  result.method = method;
  result.t_final = t_final;

  double dt = t_final / num_steps;
  result.dt = dt;

  // Initial condition
  double u0 = 1.0;
  double u_eq = 0.0;  // Equilibrium value
  std::vector<double> u = {u0};

  // RHS function: du/dt = -k * (u - u_eq)
  auto rhs = [k, u_eq](double /*t*/, const std::vector<double>& u_vec,
                        std::vector<double>* dudt) {
    (*dudt)[0] = -k * (u_vec[0] - u_eq);
  };

  TimeIntegratorConfig config;
  config.method = method;

  int total_rhs_evals = 0;
  double t = 0.0;
  bool stability_issue = false;

  for (int step = 0; step < num_steps; ++step) {
    TimeStepResult step_result = TimeStep(u, t, dt, rhs, config);
    if (!step_result.success) {
      result.passed = false;
      result.note = "Time step failed: " + step_result.error;
      return result;
    }
    total_rhs_evals += step_result.rhs_evals;
    t += dt;

    // Check for stability issues (oscillation or blowup)
    if (std::abs(u[0]) > 10.0 * std::abs(u0) || std::isnan(u[0])) {
      stability_issue = true;
      break;
    }
  }

  // Exact solution
  double u_exact = u_eq + (u0 - u_eq) * std::exp(-k * t_final);

  // Error
  double error = std::abs(u[0] - u_exact);
  result.error_l2 = error;
  result.error_linf = error;
  result.rhs_evals = total_rhs_evals;

  // Check stability criterion for explicit methods
  double stability_limit = 2.0 / k;  // For Forward Euler
  bool is_stable_regime = (dt < stability_limit);

  if (stability_issue) {
    result.passed = false;
    result.note = "Unstable (dt=" + std::to_string(dt) +
                  " > " + std::to_string(stability_limit) + " for explicit)";
  } else if (!IsExplicit(method)) {
    // Implicit methods should always be stable
    result.passed = (error < 0.1);
    result.note = "Implicit method: stable regardless of dt";
  } else if (is_stable_regime) {
    result.passed = (error < 0.1);
    result.note = "Stable regime (k*dt=" + std::to_string(k * dt) + ")";
  } else {
    // Explicit method in unstable regime - expected to fail
    result.passed = false;
    result.note = "Expected unstable (k*dt=" + std::to_string(k * dt) + " > 2)";
  }

  return result;
}

// Convergence study: run with multiple dt values and compute order
std::vector<TimeIntegratorTestResult> RunConvergenceStudy(
    TimeIntegrator method,
    const std::vector<int>& step_counts) {

  std::vector<TimeIntegratorTestResult> results;

  double lambda = 1.0;
  double t_final = 1.0;

  for (int num_steps : step_counts) {
    TimeIntegratorTestResult result =
        RunExponentialDecayTest(method, lambda, t_final, num_steps);
    results.push_back(result);
  }

  // Compute convergence rates
  for (size_t i = 1; i < results.size(); ++i) {
    double dt1 = results[i-1].dt;
    double dt2 = results[i].dt;
    double e1 = results[i-1].error_l2;
    double e2 = results[i].error_l2;

    if (e1 > 1e-15 && e2 > 1e-15) {
      double rate = std::log(e1 / e2) / std::log(dt1 / dt2);
      results[i].convergence_rate = rate;
    }
  }

  return results;
}

// Run all tests for a given integrator
std::vector<TimeIntegratorTestResult> RunTimeIntegratorTestSuite(
    TimeIntegrator method) {

  std::vector<TimeIntegratorTestResult> results;

  // Basic exponential decay
  results.push_back(RunExponentialDecayTest(method, 1.0, 1.0, 100));

  // Harmonic oscillator (1 period)
  results.push_back(RunHarmonicOscillatorTest(method, 1.0, 1, 100));

  // Advection equation
  results.push_back(RunAdvectionEquationTest(method, 100, 1.0, 1.0, 0.5));

  // Stiff decay (moderate stiffness)
  results.push_back(RunStiffDecayTest(method, 100.0, 0.1, 100));

  return results;
}

// Print test results
void PrintTimeIntegratorTestResults(
    const std::vector<TimeIntegratorTestResult>& results) {

  std::cout << "\n";
  std::cout << std::setw(20) << "Test"
            << std::setw(15) << "Method"
            << std::setw(12) << "dt"
            << std::setw(12) << "L2 Error"
            << std::setw(10) << "Rate"
            << std::setw(8) << "Pass"
            << "  Note\n";
  std::cout << std::string(90, '-') << "\n";

  for (const auto& r : results) {
    std::cout << std::setw(20) << r.name
              << std::setw(15) << TimeIntegratorToString(r.method)
              << std::setw(12) << std::scientific << std::setprecision(2) << r.dt
              << std::setw(12) << r.error_l2
              << std::setw(10) << std::fixed << std::setprecision(2);

    if (r.convergence_rate > 0) {
      std::cout << r.convergence_rate;
    } else {
      std::cout << "-";
    }

    std::cout << std::setw(8) << (r.passed ? "PASS" : "FAIL")
              << "  " << r.note << "\n";
  }
  std::cout << "\n";
}

// Compare all integrators on a given test
std::vector<TimeIntegratorTestResult> CompareTimeIntegrators(
    const std::vector<TimeIntegrator>& methods) {

  std::vector<TimeIntegratorTestResult> all_results;

  for (TimeIntegrator method : methods) {
    // Run convergence study for each method
    auto results = RunConvergenceStudy(method, {50, 100, 200, 400});

    // Take the finest resolution result
    if (!results.empty()) {
      all_results.push_back(results.back());
    }
  }

  return all_results;
}

// Test IMEX methods on stiff reaction-diffusion
// du/dt = D*d²u/dx² - k*(u - u_eq)
// where the reaction term -k*(u - u_eq) is stiff
TimeIntegratorTestResult RunIMEXTest(
    double stiffness,
    double t_final,
    int num_steps) {

  TimeIntegratorTestResult result;
  result.name = "IMEX-StiffReaction";
  result.description = "du/dt = diffusion - k*(u-u_eq)";
  result.method = TimeIntegrator::IMEX;
  result.t_final = t_final;

  double dt = t_final / num_steps;
  result.dt = dt;

  // Initial condition: Gaussian centered at 0.5
  int nx = 50;
  double dx = 1.0 / nx;
  double D = 0.01;  // Diffusion coefficient
  double u_eq = 0.0;  // Equilibrium

  std::vector<double> u(nx);
  for (int i = 0; i < nx; ++i) {
    double x = (i + 0.5) * dx;
    u[i] = std::exp(-50.0 * (x - 0.5) * (x - 0.5));
  }

  // Diffusion operator (explicit, non-stiff)
  auto rhs_diffusion = [nx, dx, D](double /*t*/, const std::vector<double>& u_vec,
                                    std::vector<double>* dudt) {
    for (int i = 0; i < nx; ++i) {
      int im = (i - 1 + nx) % nx;
      int ip = (i + 1) % nx;
      (*dudt)[i] = D * (u_vec[ip] - 2.0 * u_vec[im] + u_vec[im]) / (dx * dx);
    }
  };

  // Reaction operator (implicit, stiff)
  auto rhs_reaction = [nx, stiffness, u_eq](double /*t*/,
                                             const std::vector<double>& u_vec,
                                             std::vector<double>* dudt) {
    for (int i = 0; i < nx; ++i) {
      (*dudt)[i] = -stiffness * (u_vec[i] - u_eq);
    }
  };

  int total_rhs_evals = 0;
  double t = 0.0;

  for (int step = 0; step < num_steps; ++step) {
    TimeStepResult step_result = IMEXEulerStep(u, t, dt, rhs_diffusion,
                                                rhs_reaction, 100, 1e-10);
    if (!step_result.success) {
      result.passed = false;
      result.note = "IMEX step failed: " + step_result.error;
      return result;
    }
    total_rhs_evals += step_result.rhs_evals;
    t += dt;
  }

  // For stiff decay, solution should approach u_eq exponentially
  // The diffusion spreads while the reaction decays to u_eq
  double max_val = *std::max_element(u.begin(), u.end());
  double expected_decay = std::exp(-stiffness * t_final);

  result.error_l2 = max_val;  // Peak value
  result.error_linf = max_val;
  result.rhs_evals = total_rhs_evals;

  // Should have decayed significantly due to stiff reaction
  result.passed = (max_val < 0.1);
  result.note = "Peak=" + std::to_string(max_val) +
                " (expected decay " + std::to_string(expected_decay) + ")";

  return result;
}

// Test adaptive time stepping with error control
TimeIntegratorTestResult RunAdaptiveTimeStepTest(
    double t_final,
    double error_tol) {

  TimeIntegratorTestResult result;
  result.name = "AdaptiveTimeStepping";
  result.description = "Adaptive dt with error control";
  result.method = TimeIntegrator::RK2;
  result.t_final = t_final;

  // Initial condition
  double u0 = 1.0;
  double lambda = 5.0;  // Moderate decay rate
  std::vector<double> u = {u0};

  // RHS function: du/dt = -lambda * u
  auto rhs = [lambda](double /*t*/, const std::vector<double>& u_vec,
                       std::vector<double>* dudt) {
    (*dudt)[0] = -lambda * u_vec[0];
  };

  TimeIntegratorConfig config;
  config.method = TimeIntegrator::RK2;
  config.adaptive_dt = true;
  config.error_tol = error_tol;
  config.dt_min = 1e-8;
  config.dt_max = 0.5;
  config.dt_grow_factor = 1.5;
  config.dt_shrink_factor = 0.5;

  double t = 0.0;
  double dt = 0.1;  // Initial dt
  int total_steps = 0;
  int total_rhs_evals = 0;
  int rejected_steps = 0;

  while (t < t_final) {
    TimeStepResult step_result = AdaptiveTimeStep(u, t, dt, rhs, config);
    total_rhs_evals += step_result.rhs_evals;
    total_steps++;

    if (step_result.success) {
      t += step_result.dt_used;
      dt = step_result.dt_next;
    } else {
      rejected_steps++;
      dt = step_result.dt_next;
    }

    // Safety limit on iterations
    if (total_steps > 10000) {
      result.passed = false;
      result.note = "Too many steps";
      return result;
    }
  }

  // Exact solution
  double u_exact = u0 * std::exp(-lambda * t_final);
  double error = std::abs(u[0] - u_exact);

  result.dt = t_final / total_steps;  // Average dt
  result.error_l2 = error;
  result.error_linf = error;
  result.rhs_evals = total_rhs_evals;

  // Should meet error tolerance and be efficient
  result.passed = (error < 10.0 * error_tol);
  result.note = "Steps=" + std::to_string(total_steps) +
                ", rejected=" + std::to_string(rejected_steps) +
                ", error=" + std::to_string(error);

  return result;
}

// Test CFL-based time stepping for advection
TimeIntegratorTestResult RunCFLSteppingTest(
    double velocity,
    double cfl_target,
    int nx) {

  TimeIntegratorTestResult result;
  result.name = "CFLStepping";
  result.description = "CFL-based dt selection";
  result.method = TimeIntegrator::SSPRK2;
  result.t_final = 1.0;

  double dx = 1.0 / nx;
  double t_final = 1.0;

  // Use SuggestStableDt function
  double dt = SuggestStableDt(velocity, 0, 0,  // velocity
                               0, 0, 0,         // diffusion
                               dx, dx, dx,
                               TimeIntegrator::SSPRK2, cfl_target);

  result.dt = dt;

  // Compute actual CFL
  double actual_cfl = std::abs(velocity) * dt / dx;

  // Initialize solution
  std::vector<double> u(nx);
  for (int i = 0; i < nx; ++i) {
    double x = (i + 0.5) * dx;
    u[i] = std::exp(-50.0 * (x - 0.5) * (x - 0.5));
  }

  // Advection RHS with upwind
  auto rhs = [nx, dx, velocity](double /*t*/, const std::vector<double>& u_vec,
                                 std::vector<double>* dudt) {
    for (int i = 0; i < nx; ++i) {
      int im = (i - 1 + nx) % nx;
      int ip = (i + 1) % nx;

      double dudx;
      if (velocity > 0) {
        dudx = (u_vec[i] - u_vec[im]) / dx;
      } else {
        dudx = (u_vec[ip] - u_vec[i]) / dx;
      }
      (*dudt)[i] = -velocity * dudx;
    }
  };

  TimeIntegratorConfig config;
  config.method = TimeIntegrator::SSPRK2;

  int num_steps = static_cast<int>(std::ceil(t_final / dt));
  double t = 0.0;
  int total_rhs_evals = 0;

  for (int step = 0; step < num_steps; ++step) {
    double step_dt = std::min(dt, t_final - t);
    TimeStepResult step_result = TimeStep(u, t, step_dt, rhs, config);
    if (!step_result.success) {
      result.passed = false;
      result.note = "Time step failed";
      return result;
    }
    total_rhs_evals += step_result.rhs_evals;
    t += step_dt;
  }

  // Check mass conservation
  double mass = 0.0;
  for (double v : u) mass += v * dx;

  // Initial mass was approximately integral of Gaussian
  double initial_mass = std::sqrt(M_PI / 50.0);  // Approximate
  double mass_error = std::abs(mass - initial_mass) / initial_mass;

  result.error_l2 = mass_error;
  result.error_linf = mass_error;
  result.rhs_evals = total_rhs_evals;

  // CFL should be close to target and solution should be stable
  result.passed = (actual_cfl <= cfl_target * 1.1) && (mass_error < 0.05);
  result.note = "CFL=" + std::to_string(actual_cfl) +
                " (target " + std::to_string(cfl_target) + ")" +
                ", mass_err=" + std::to_string(mass_error * 100.0) + "%";

  return result;
}
