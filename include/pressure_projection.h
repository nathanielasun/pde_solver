#ifndef PRESSURE_PROJECTION_H
#define PRESSURE_PROJECTION_H

#include <functional>
#include <string>
#include <vector>

#include "pde_types.h"

// ===========================================================================
// Pressure Projection Method for Incompressible Flow
// ===========================================================================
//
// Implements the Chorin-Temam projection method (fractional step):
// 1. Compute intermediate velocity u* (ignoring pressure gradient)
// 2. Solve Poisson equation for pressure: nabla^2 p = (1/dt) * div(u*)
// 3. Project velocity to divergence-free: u = u* - dt * grad(p)
//
// This enforces the incompressibility constraint: div(u) = 0

// Velocity field in 2D (staggered or collocated)
struct VelocityField2D {
  std::vector<double> u;  // x-component
  std::vector<double> v;  // y-component
  int nx = 0;
  int ny = 0;
  double dx = 1.0;
  double dy = 1.0;
};

// Velocity field in 3D
struct VelocityField3D {
  std::vector<double> u;  // x-component
  std::vector<double> v;  // y-component
  std::vector<double> w;  // z-component
  int nx = 0;
  int ny = 0;
  int nz = 0;
  double dx = 1.0;
  double dy = 1.0;
  double dz = 1.0;
};

// Projection configuration
struct ProjectionConfig {
  // Poisson solver settings
  int max_iter = 1000;
  double tol = 1e-8;
  SolveMethod poisson_method = SolveMethod::CG;

  // Grid type
  bool staggered_grid = false;  // true for MAC grid, false for collocated

  // Boundary conditions for pressure (typically Neumann)
  // Velocity BCs are handled separately
  bool pressure_neumann = true;

  // Reference pressure point (for pure Neumann BCs)
  bool fix_pressure_corner = true;
  double reference_pressure = 0.0;
};

// Result of projection step
struct ProjectionResult {
  bool success = true;
  std::string error;
  int poisson_iterations = 0;
  double poisson_residual = 0.0;
  double max_divergence_before = 0.0;
  double max_divergence_after = 0.0;
  double l2_divergence_before = 0.0;
  double l2_divergence_after = 0.0;
};

// ===========================================================================
// 2D Pressure Projection
// ===========================================================================

// Compute divergence of velocity field (collocated grid)
void ComputeDivergence2D(const VelocityField2D& vel,
                          std::vector<double>* div);

// Compute divergence on staggered (MAC) grid
void ComputeDivergenceMAC2D(const VelocityField2D& vel,
                             std::vector<double>* div);

// Compute gradient of pressure field
void ComputeGradient2D(const std::vector<double>& p,
                        int nx, int ny, double dx, double dy,
                        std::vector<double>* grad_x,
                        std::vector<double>* grad_y);

// Solve Poisson equation: nabla^2 p = rhs
// With Neumann BCs (dp/dn = 0) or mixed BCs
bool SolvePressurePoisson2D(const std::vector<double>& rhs,
                             int nx, int ny, double dx, double dy,
                             std::vector<double>* p,
                             const ProjectionConfig& config,
                             int* iterations = nullptr,
                             double* residual = nullptr);

// Full projection step for 2D velocity field
// Modifies velocity in place to be divergence-free
ProjectionResult ProjectVelocity2D(VelocityField2D& vel,
                                    double dt,
                                    std::vector<double>* pressure,
                                    const ProjectionConfig& config);

// ===========================================================================
// 3D Pressure Projection
// ===========================================================================

// Compute divergence of velocity field (collocated grid)
void ComputeDivergence3D(const VelocityField3D& vel,
                          std::vector<double>* div);

// Compute gradient of pressure field
void ComputeGradient3D(const std::vector<double>& p,
                        int nx, int ny, int nz,
                        double dx, double dy, double dz,
                        std::vector<double>* grad_x,
                        std::vector<double>* grad_y,
                        std::vector<double>* grad_z);

// Solve Poisson equation in 3D
bool SolvePressurePoisson3D(const std::vector<double>& rhs,
                             int nx, int ny, int nz,
                             double dx, double dy, double dz,
                             std::vector<double>* p,
                             const ProjectionConfig& config,
                             int* iterations = nullptr,
                             double* residual = nullptr);

// Full projection step for 3D velocity field
ProjectionResult ProjectVelocity3D(VelocityField3D& vel,
                                    double dt,
                                    std::vector<double>* pressure,
                                    const ProjectionConfig& config);

// ===========================================================================
// Incompressible Navier-Stokes Time Stepper
// ===========================================================================

// Viscous term: nu * nabla^2 u
void ComputeViscousTerm2D(const VelocityField2D& vel, double nu,
                           std::vector<double>* visc_u,
                           std::vector<double>* visc_v);

// Advection term: -(u . nabla) u
void ComputeAdvectionTerm2D(const VelocityField2D& vel,
                             std::vector<double>* adv_u,
                             std::vector<double>* adv_v);

// Configuration for incompressible NS solver
struct IncompressibleNSConfig {
  double nu = 0.01;           // Kinematic viscosity (1/Re for unit velocity/length)
  double dt = 0.01;           // Time step
  TimeIntegrator time_method = TimeIntegrator::RK2;
  ProjectionConfig projection;
};

// Single time step of incompressible Navier-Stokes
// Uses projection method: advance momentum, then project
ProjectionResult NavierStokesStep2D(VelocityField2D& vel,
                                     std::vector<double>* pressure,
                                     const IncompressibleNSConfig& config);

// ===========================================================================
// Utility Functions
// ===========================================================================

// Compute L2 and L-infinity norms of divergence
void ComputeDivergenceNorms2D(const VelocityField2D& vel,
                               double* l2_norm,
                               double* linf_norm);

// Compute kinetic energy: 0.5 * integral(u^2 + v^2)
double ComputeKineticEnergy2D(const VelocityField2D& vel);

// Compute enstrophy: 0.5 * integral(omega^2) where omega = curl(u)
double ComputeEnstrophy2D(const VelocityField2D& vel);

// Compute vorticity field: omega = dv/dx - du/dy
void ComputeVorticity2D(const VelocityField2D& vel,
                         std::vector<double>* omega);

// Initialize velocity field with zeros
VelocityField2D CreateVelocityField2D(int nx, int ny, double dx, double dy);

// Apply velocity boundary conditions (lid-driven cavity)
void ApplyLidDrivenCavityBC(VelocityField2D& vel, double lid_velocity = 1.0);

// Apply no-slip boundary conditions on all walls
void ApplyNoSlipBC(VelocityField2D& vel);

#endif  // PRESSURE_PROJECTION_H
