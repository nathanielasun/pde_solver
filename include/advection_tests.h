#ifndef ADVECTION_TESTS_H
#define ADVECTION_TESTS_H

#include <functional>
#include <string>
#include <vector>

#include "pde_types.h"

// Result of an advection test
struct AdvectionTestResult {
  bool passed = false;
  std::string name;
  std::string description;
  AdvectionScheme scheme;
  int nx;
  int ny;
  double dt;
  double t_final;
  double error_l1 = 0.0;
  double error_l2 = 0.0;
  double error_linf = 0.0;
  double mass_initial = 0.0;
  double mass_final = 0.0;
  double mass_error = 0.0;  // Relative mass conservation error
  std::string note;
};

// Run a 1D advection test with a top-hat (square wave) initial condition
// Tests transport of discontinuous profile
AdvectionTestResult RunTopHatAdvectionTest1D(
    AdvectionScheme scheme,
    int nx = 100,
    double velocity = 1.0,
    double t_final = 1.0,
    double cfl = 0.5);

// Run a 2D advection test with a top-hat initial condition
// Tests diagonal transport
AdvectionTestResult RunTopHatAdvectionTest2D(
    AdvectionScheme scheme,
    int nx = 64,
    int ny = 64,
    double vx = 1.0,
    double vy = 1.0,
    double t_final = 1.0,
    double cfl = 0.5);

// Run a 1D advection test with a smooth (Gaussian) initial condition
// Tests accuracy for smooth profiles
AdvectionTestResult RunGaussianAdvectionTest1D(
    AdvectionScheme scheme,
    int nx = 100,
    double velocity = 1.0,
    double t_final = 1.0,
    double cfl = 0.5);

// Run a 2D rotation test (solid body rotation)
// Tests preservation of shape under rotation
AdvectionTestResult RunRotationTest2D(
    AdvectionScheme scheme,
    int nx = 64,
    int ny = 64,
    double omega = 1.0,  // Angular velocity
    int rotations = 1,   // Number of full rotations
    double cfl = 0.5);

// Run convergence study for smooth advection
// Returns results for multiple grid resolutions
std::vector<AdvectionTestResult> RunAdvectionConvergenceStudy(
    AdvectionScheme scheme,
    const std::vector<int>& resolutions = {32, 64, 128, 256},
    double velocity = 1.0,
    double t_final = 1.0,
    double cfl = 0.5);

// Run all advection tests for a given scheme
std::vector<AdvectionTestResult> RunAdvectionTestSuite(AdvectionScheme scheme);

// Print test results summary
void PrintAdvectionTestResults(const std::vector<AdvectionTestResult>& results);

// Compare schemes on a given test
std::vector<AdvectionTestResult> CompareAdvectionSchemes(
    const std::function<AdvectionTestResult(AdvectionScheme)>& test_runner,
    const std::vector<AdvectionScheme>& schemes = {
        AdvectionScheme::Upwind,
        AdvectionScheme::LaxWendroff,
        AdvectionScheme::MinMod,
        AdvectionScheme::Superbee,
        AdvectionScheme::VanLeer,
        AdvectionScheme::MC
    });

#endif  // ADVECTION_TESTS_H
