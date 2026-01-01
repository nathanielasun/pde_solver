#ifndef PRESSURE_PROJECTION_TESTS_H
#define PRESSURE_PROJECTION_TESTS_H

#include <string>
#include <vector>

#include "pressure_projection.h"

// Test result for pressure projection validation
struct PressureProjectionTestResult {
  bool passed = false;
  std::string name;
  std::string description;
  double divergence_l2 = 0.0;
  double divergence_linf = 0.0;
  double error_u = 0.0;       // Error in u-velocity
  double error_v = 0.0;       // Error in v-velocity
  double kinetic_energy = 0.0;
  double enstrophy = 0.0;
  int poisson_iterations = 0;
  std::string note;
};

// Test simple projection of a non-divergence-free field
PressureProjectionTestResult RunSimpleProjectionTest(int nx = 32, int ny = 32);

// Test that divergence-free fields remain unchanged
PressureProjectionTestResult RunDivergenceFreeTest(int nx = 32, int ny = 32);

// Taylor-Green vortex: exact solution for 2D incompressible NS
// u = -cos(x)*sin(y)*exp(-2*nu*t)
// v =  sin(x)*cos(y)*exp(-2*nu*t)
// p = -1/4*(cos(2x) + cos(2y))*exp(-4*nu*t)
PressureProjectionTestResult RunTaylorGreenVortexTest(
    int nx = 64,
    int ny = 64,
    double nu = 0.01,
    double t_final = 1.0,
    int num_steps = 100);

// Lid-driven cavity benchmark
// Top wall moves with velocity u=1, all other walls no-slip
// Compare with reference data at Re=100, 400, 1000
struct LidDrivenCavityResult {
  bool passed = false;
  double reynolds;
  int nx, ny;
  int num_steps;
  double u_centerline_max;    // Max u along vertical centerline
  double v_centerline_max;    // Max v along horizontal centerline
  double primary_vortex_x;    // Primary vortex center x
  double primary_vortex_y;    // Primary vortex center y
  double stream_min;          // Min streamfunction (vortex strength)
  std::string note;
};

LidDrivenCavityResult RunLidDrivenCavityTest(
    int nx = 64,
    int ny = 64,
    double reynolds = 100.0,
    int num_steps = 10000);

// Convergence test: verify spatial order of accuracy
std::vector<PressureProjectionTestResult> RunProjectionConvergenceTest(
    const std::vector<int>& grid_sizes = {16, 32, 64, 128});

// Run all projection tests
std::vector<PressureProjectionTestResult> RunPressureProjectionTestSuite();

// Print test results
void PrintPressureProjectionTestResults(
    const std::vector<PressureProjectionTestResult>& results);

void PrintLidDrivenCavityResult(const LidDrivenCavityResult& result);

#endif  // PRESSURE_PROJECTION_TESTS_H
