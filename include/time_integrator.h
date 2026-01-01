#ifndef TIME_INTEGRATOR_H
#define TIME_INTEGRATOR_H

#include <functional>
#include <string>
#include <vector>

#include "pde_types.h"  // TimeIntegrator enum defined here

// Parse time integrator from string
TimeIntegrator ParseTimeIntegrator(const std::string& name);

// Convert time integrator to string
std::string TimeIntegratorToString(TimeIntegrator method);

// Check if integrator is explicit
inline bool IsExplicit(TimeIntegrator method) {
  return method == TimeIntegrator::ForwardEuler ||
         method == TimeIntegrator::RK2 ||
         method == TimeIntegrator::RK4 ||
         method == TimeIntegrator::SSPRK2 ||
         method == TimeIntegrator::SSPRK3;
}

// Check if integrator is SSP (preserves TVD property)
inline bool IsSSP(TimeIntegrator method) {
  return method == TimeIntegrator::ForwardEuler ||
         method == TimeIntegrator::SSPRK2 ||
         method == TimeIntegrator::SSPRK3;
}

// Get theoretical order of accuracy
inline int GetOrder(TimeIntegrator method) {
  switch (method) {
    case TimeIntegrator::ForwardEuler:
    case TimeIntegrator::BackwardEuler:
      return 1;
    case TimeIntegrator::RK2:
    case TimeIntegrator::SSPRK2:
    case TimeIntegrator::CrankNicolson:
      return 2;
    case TimeIntegrator::SSPRK3:
      return 3;
    case TimeIntegrator::RK4:
      return 4;
    case TimeIntegrator::IMEX:
      return 2;  // Typically second-order
    default:
      return 1;
  }
}

// Time integration configuration
struct TimeIntegratorConfig {
  TimeIntegrator method = TimeIntegrator::ForwardEuler;

  // CFL-based time stepping
  bool use_cfl = false;
  double cfl_target = 0.5;        // Target CFL number
  double cfl_safety = 0.9;        // Safety factor for CFL

  // Adaptive time stepping
  bool adaptive_dt = false;
  double dt_min = 1e-12;          // Minimum allowed dt
  double dt_max = 1.0;            // Maximum allowed dt
  double error_tol = 1e-5;        // Error tolerance for adaptive stepping
  double dt_grow_factor = 1.5;    // Max dt growth per step
  double dt_shrink_factor = 0.5;  // Min dt shrink per step

  // Implicit solver settings (for backward Euler, Crank-Nicolson)
  int implicit_max_iter = 100;
  double implicit_tol = 1e-8;
};

// RHS function type: computes du/dt given current state
// Returns the right-hand side of du/dt = F(t, u)
using RHSFunction = std::function<void(double t, const std::vector<double>& u,
                                        std::vector<double>* dudt)>;

// Time step result
struct TimeStepResult {
  bool success = true;
  std::string error;
  double dt_used = 0.0;           // Actual dt used
  double dt_next = 0.0;           // Suggested next dt (for adaptive)
  double error_estimate = 0.0;    // Error estimate (for adaptive)
  int rhs_evals = 0;              // Number of RHS evaluations
};

// Perform a single time step using the specified integrator
// u: current solution (modified in place)
// t: current time
// dt: time step size
// rhs: function that computes du/dt
// config: integrator configuration
TimeStepResult TimeStep(std::vector<double>& u, double t, double dt,
                        const RHSFunction& rhs,
                        const TimeIntegratorConfig& config);

// Forward Euler: u_{n+1} = u_n + dt * F(t_n, u_n)
TimeStepResult ForwardEulerStep(std::vector<double>& u, double t, double dt,
                                 const RHSFunction& rhs);

// RK2 (Heun's method):
// k1 = F(t_n, u_n)
// k2 = F(t_n + dt, u_n + dt*k1)
// u_{n+1} = u_n + dt/2 * (k1 + k2)
TimeStepResult RK2Step(std::vector<double>& u, double t, double dt,
                        const RHSFunction& rhs);

// RK4 (Classical):
// k1 = F(t_n, u_n)
// k2 = F(t_n + dt/2, u_n + dt/2*k1)
// k3 = F(t_n + dt/2, u_n + dt/2*k2)
// k4 = F(t_n + dt, u_n + dt*k3)
// u_{n+1} = u_n + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
TimeStepResult RK4Step(std::vector<double>& u, double t, double dt,
                        const RHSFunction& rhs);

// SSPRK2 (TVD-preserving):
// u^(1) = u_n + dt * F(t_n, u_n)
// u_{n+1} = 1/2 * u_n + 1/2 * (u^(1) + dt * F(t_n + dt, u^(1)))
TimeStepResult SSPRK2Step(std::vector<double>& u, double t, double dt,
                           const RHSFunction& rhs);

// SSPRK3 (Shu-Osher, optimal 3rd-order SSP):
// u^(1) = u_n + dt * F(t_n, u_n)
// u^(2) = 3/4*u_n + 1/4*(u^(1) + dt * F(t_n + dt, u^(1)))
// u_{n+1} = 1/3*u_n + 2/3*(u^(2) + dt * F(t_n + dt/2, u^(2)))
TimeStepResult SSPRK3Step(std::vector<double>& u, double t, double dt,
                           const RHSFunction& rhs);

// Backward Euler (implicit):
// u_{n+1} = u_n + dt * F(t_{n+1}, u_{n+1})
// Requires Newton iteration or fixed-point iteration
TimeStepResult BackwardEulerStep(std::vector<double>& u, double t, double dt,
                                  const RHSFunction& rhs,
                                  int max_iter = 100, double tol = 1e-8);

// Crank-Nicolson (implicit trapezoidal):
// u_{n+1} = u_n + dt/2 * (F(t_n, u_n) + F(t_{n+1}, u_{n+1}))
TimeStepResult CrankNicolsonStep(std::vector<double>& u, double t, double dt,
                                  const RHSFunction& rhs,
                                  int max_iter = 100, double tol = 1e-8);

// Compute CFL number for advection-diffusion
// Returns max(|v|*dt/dx) for advection and max(D*dt/dx^2) for diffusion
double ComputeCFLAdvectionDiffusion(double vx, double vy, double vz,
                                     double Dx, double Dy, double Dz,
                                     double dx, double dy, double dz,
                                     double dt);

// Suggest stable dt based on CFL constraints
// For advection: CFL <= 1 for upwind, CFL <= 0.5 for higher-order
// For diffusion: Fourier number <= 0.5 in each direction
double SuggestStableDt(double vx, double vy, double vz,
                       double Dx, double Dy, double Dz,
                       double dx, double dy, double dz,
                       TimeIntegrator method, double cfl_target = 0.5);

// Adaptive time stepping with error control
// Uses embedded RK methods (RK2(3) or RK4(5)) for error estimation
TimeStepResult AdaptiveTimeStep(std::vector<double>& u, double t, double dt,
                                 const RHSFunction& rhs,
                                 const TimeIntegratorConfig& config);

// ===========================================================================
// IMEX (Implicit-Explicit) Methods for Stiff Systems
// ===========================================================================

// IMEX-Euler: Combines forward Euler (explicit) with backward Euler (implicit)
// For du/dt = F_explicit(u) + F_implicit(u)
// Uses: u^* = u^n + dt*F_explicit(u^n)
//       u^{n+1} = u^* + dt*F_implicit(u^{n+1})
TimeStepResult IMEXEulerStep(std::vector<double>& u, double t, double dt,
                              const RHSFunction& rhs_explicit,
                              const RHSFunction& rhs_implicit,
                              int max_iter = 100, double tol = 1e-8);

// IMEX-SSP2(2,2,2): Second-order SSP-preserving IMEX scheme
// Explicit part: SSPRK2
// Implicit part: SDIRK2 (L-stable)
TimeStepResult IMEXSSP2Step(std::vector<double>& u, double t, double dt,
                             const RHSFunction& rhs_explicit,
                             const RHSFunction& rhs_implicit,
                             int max_iter = 100, double tol = 1e-8);

// Operator splitting for reaction-diffusion systems
// Strang splitting: F_diff(dt/2) -> F_react(dt) -> F_diff(dt/2)
// Achieves second-order accuracy in time
struct OperatorSplitConfig {
  RHSFunction rhs_diffusion;        // Non-stiff diffusion operator
  RHSFunction rhs_reaction;         // Potentially stiff reaction term
  TimeIntegrator diffusion_method = TimeIntegrator::SSPRK2;  // Explicit for diffusion
  TimeIntegrator reaction_method = TimeIntegrator::BackwardEuler;  // Implicit for reaction
  bool use_strang = true;           // Strang splitting for 2nd order
  int implicit_max_iter = 100;
  double implicit_tol = 1e-8;
};

TimeStepResult OperatorSplitStep(std::vector<double>& u, double t, double dt,
                                  const OperatorSplitConfig& config);

#endif  // TIME_INTEGRATOR_H
