#ifndef ADVECTION_H
#define ADVECTION_H

#include <cmath>
#include <string>
#include <vector>

#include "pde_types.h"  // For AdvectionScheme enum

// Convert string to scheme
AdvectionScheme ParseAdvectionScheme(const std::string& name);

// Convert scheme to string
std::string AdvectionSchemeToString(AdvectionScheme scheme);

// Advection configuration
struct AdvectionConfig {
  AdvectionScheme scheme = AdvectionScheme::Upwind;
  bool use_dimensional_splitting = true;  // Split 2D/3D into 1D sweeps
};

// Flux limiter functions for TVD schemes
// Returns limited slope ratio phi(r)
double MinModLimiter(double r);
double SuperbeeLimiter(double r);
double VanLeerLimiter(double r);
double MCLimiter(double r);

// Compute advection flux at cell interface (i+1/2) for 1D
// u: solution values
// v: velocity (positive = right)
// i: cell index
// dx: grid spacing
// scheme: discretization scheme
double ComputeAdvectionFlux1D(const double* u, int n, double v, int i,
                               double dx, AdvectionScheme scheme);

// Compute advection term contribution for a single cell in 2D
// Returns -d(F)/dx - d(G)/dy where F, G are x and y fluxes
// u: 2D solution array (row-major)
// vx, vy: velocity components (can be spatially varying)
// i, j: cell indices
// nx, ny: grid dimensions
// dx, dy: grid spacings
// scheme: discretization scheme
double ComputeAdvectionTerm2D(const std::vector<double>& u,
                               double vx, double vy,
                               int i, int j, int nx, int ny,
                               double dx, double dy,
                               AdvectionScheme scheme);

// Compute advection term contribution for a single cell in 3D
double ComputeAdvectionTerm3D(const std::vector<double>& u,
                               double vx, double vy, double vz,
                               int i, int j, int k, int nx, int ny, int nz,
                               double dx, double dy, double dz,
                               AdvectionScheme scheme);

// Check if scheme is TVD (requires flux limiting)
inline bool IsTVDScheme(AdvectionScheme scheme) {
  return scheme == AdvectionScheme::MinMod ||
         scheme == AdvectionScheme::Superbee ||
         scheme == AdvectionScheme::VanLeer ||
         scheme == AdvectionScheme::MC;
}

// Check if scheme is at least second-order accurate
inline bool IsSecondOrder(AdvectionScheme scheme) {
  return scheme != AdvectionScheme::Upwind;
}

// Compute CFL number for advection stability
// Returns max(|vx|*dt/dx, |vy|*dt/dy, |vz|*dt/dz)
double ComputeCFL(double vx, double vy, double vz,
                  double dx, double dy, double dz, double dt);

// Suggest stable dt based on CFL constraint
// cfl_target: desired CFL number (typically 0.5-0.9 for explicit methods)
double SuggestAdvectionDt(double vx, double vy, double vz,
                          double dx, double dy, double dz,
                          double cfl_target = 0.8);

#endif  // ADVECTION_H
