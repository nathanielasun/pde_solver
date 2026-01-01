#ifndef TIME_INTEGRATOR_TESTS_H
#define TIME_INTEGRATOR_TESTS_H

#include <string>
#include <vector>

#include "pde_types.h"

// Test result for time integrator validation
struct TimeIntegratorTestResult {
  bool passed = false;
  std::string name;
  std::string description;
  TimeIntegrator method;
  double dt;
  double t_final;
  double error_l2 = 0.0;
  double error_linf = 0.0;
  double convergence_rate = 0.0;  // Measured order of accuracy
  int rhs_evals = 0;
  std::string note;
};

// Test exponential decay: du/dt = -lambda * u
// Exact solution: u(t) = u0 * exp(-lambda * t)
TimeIntegratorTestResult RunExponentialDecayTest(
    TimeIntegrator method,
    double lambda = 1.0,
    double t_final = 1.0,
    int num_steps = 100);

// Test harmonic oscillator: d²u/dt² = -omega² * u
// Converted to first-order system: du/dt = v, dv/dt = -omega² * u
// Exact: u(t) = u0*cos(omega*t), v(t) = -u0*omega*sin(omega*t)
TimeIntegratorTestResult RunHarmonicOscillatorTest(
    TimeIntegrator method,
    double omega = 1.0,
    int num_periods = 1,
    int steps_per_period = 100);

// Test advection equation: du/dt = -c * du/dx
// with periodic boundary conditions and Gaussian initial condition
TimeIntegratorTestResult RunAdvectionEquationTest(
    TimeIntegrator method,
    int nx = 100,
    double velocity = 1.0,
    double t_final = 1.0,
    double cfl = 0.5);

// Test stiff equation: du/dt = -k * (u - u_eq)
// with large k (stiff)
TimeIntegratorTestResult RunStiffDecayTest(
    TimeIntegrator method,
    double k = 1000.0,    // Stiffness parameter
    double t_final = 0.1,
    int num_steps = 100);

// Convergence study: run with multiple dt values and compute order
std::vector<TimeIntegratorTestResult> RunConvergenceStudy(
    TimeIntegrator method,
    const std::vector<int>& step_counts = {50, 100, 200, 400});

// Run all tests for a given integrator
std::vector<TimeIntegratorTestResult> RunTimeIntegratorTestSuite(TimeIntegrator method);

// Print test results
void PrintTimeIntegratorTestResults(const std::vector<TimeIntegratorTestResult>& results);

// Compare all integrators on a given test
std::vector<TimeIntegratorTestResult> CompareTimeIntegrators(
    const std::vector<TimeIntegrator>& methods = {
        TimeIntegrator::ForwardEuler,
        TimeIntegrator::RK2,
        TimeIntegrator::RK4,
        TimeIntegrator::SSPRK2,
        TimeIntegrator::SSPRK3
    });

// Test IMEX methods on stiff reaction-diffusion
TimeIntegratorTestResult RunIMEXTest(
    double stiffness = 100.0,
    double t_final = 0.1,
    int num_steps = 100);

// Test adaptive time stepping
TimeIntegratorTestResult RunAdaptiveTimeStepTest(
    double t_final = 1.0,
    double error_tol = 1e-5);

// Test CFL-based time stepping for advection
TimeIntegratorTestResult RunCFLSteppingTest(
    double velocity = 1.0,
    double cfl_target = 0.5,
    int nx = 100);

#endif  // TIME_INTEGRATOR_TESTS_H
