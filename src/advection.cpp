#include "advection.h"

#include <algorithm>
#include <cmath>

AdvectionScheme ParseAdvectionScheme(const std::string& name) {
  if (name == "centered" || name == "central") {
    return AdvectionScheme::Centered;
  }
  if (name == "upwind" || name == "first-order") {
    return AdvectionScheme::Upwind;
  }
  if (name == "lax-wendroff" || name == "lw") {
    return AdvectionScheme::LaxWendroff;
  }
  if (name == "beam-warming" || name == "bw") {
    return AdvectionScheme::BeamWarming;
  }
  if (name == "fromm") {
    return AdvectionScheme::Fromm;
  }
  if (name == "minmod") {
    return AdvectionScheme::MinMod;
  }
  if (name == "superbee") {
    return AdvectionScheme::Superbee;
  }
  if (name == "vanleer" || name == "van-leer") {
    return AdvectionScheme::VanLeer;
  }
  if (name == "mc" || name == "monotonized-central") {
    return AdvectionScheme::MC;
  }
  return AdvectionScheme::Upwind;  // Default
}

std::string AdvectionSchemeToString(AdvectionScheme scheme) {
  switch (scheme) {
    case AdvectionScheme::Centered: return "centered";
    case AdvectionScheme::Upwind: return "upwind";
    case AdvectionScheme::LaxWendroff: return "lax-wendroff";
    case AdvectionScheme::BeamWarming: return "beam-warming";
    case AdvectionScheme::Fromm: return "fromm";
    case AdvectionScheme::MinMod: return "minmod";
    case AdvectionScheme::Superbee: return "superbee";
    case AdvectionScheme::VanLeer: return "vanleer";
    case AdvectionScheme::MC: return "mc";
    default: return "unknown";
  }
}

// Flux limiters
// All limiters satisfy: phi(r) = 0 for r <= 0 (to preserve monotonicity)
// and phi(1) = 1 (second-order for smooth solutions)

double MinModLimiter(double r) {
  if (r <= 0.0) return 0.0;
  return std::min(r, 1.0);
}

double SuperbeeLimiter(double r) {
  if (r <= 0.0) return 0.0;
  return std::max(0.0, std::max(std::min(2.0 * r, 1.0), std::min(r, 2.0)));
}

double VanLeerLimiter(double r) {
  if (r <= 0.0) return 0.0;
  return (r + std::abs(r)) / (1.0 + std::abs(r));
}

double MCLimiter(double r) {
  if (r <= 0.0) return 0.0;
  double c = (1.0 + r) / 2.0;
  return std::max(0.0, std::min(std::min(2.0, 2.0 * r), c));
}

namespace {

// Get flux limiter function for a TVD scheme
double ApplyLimiter(AdvectionScheme scheme, double r) {
  switch (scheme) {
    case AdvectionScheme::MinMod: return MinModLimiter(r);
    case AdvectionScheme::Superbee: return SuperbeeLimiter(r);
    case AdvectionScheme::VanLeer: return VanLeerLimiter(r);
    case AdvectionScheme::MC: return MCLimiter(r);
    default: return 1.0;  // No limiting
  }
}

// Compute smoothness indicator r = (u_i - u_{i-1}) / (u_{i+1} - u_i)
// Handles division by zero
double ComputeSmoothnessRatio(double delta_minus, double delta_plus) {
  const double eps = 1e-12;
  if (std::abs(delta_plus) < eps) {
    return (delta_minus >= 0.0) ? 1e10 : -1e10;
  }
  return delta_minus / delta_plus;
}

// Safe array access with boundary handling (extrapolation)
inline double SafeAccess(const double* u, int n, int i) {
  if (i < 0) return u[0];
  if (i >= n) return u[n - 1];
  return u[i];
}

}  // namespace

double ComputeAdvectionFlux1D(const double* u, int n, double v, int i,
                               double dx, AdvectionScheme scheme) {
  // Compute flux at interface i+1/2
  // For advection equation u_t + v*u_x = 0, flux F = v*u
  // We need the interface value u_{i+1/2}

  if (n < 2) return 0.0;

  // Get neighboring values (with boundary handling)
  const double u_im1 = SafeAccess(u, n, i - 1);
  const double u_i = SafeAccess(u, n, i);
  const double u_ip1 = SafeAccess(u, n, i + 1);
  const double u_ip2 = SafeAccess(u, n, i + 2);

  double u_interface = 0.0;

  switch (scheme) {
    case AdvectionScheme::Centered:
      // Second-order central: u_{i+1/2} = (u_i + u_{i+1})/2
      u_interface = 0.5 * (u_i + u_ip1);
      break;

    case AdvectionScheme::Upwind:
      // First-order upwind: use upstream value
      if (v >= 0.0) {
        u_interface = u_i;
      } else {
        u_interface = u_ip1;
      }
      break;

    case AdvectionScheme::LaxWendroff:
      // Second-order Lax-Wendroff (equivalent to linear interpolation for interface)
      // For pure advection, LW is unstable for discontinuities - use linear reconstruction
      if (v >= 0.0) {
        // Right-biased linear interpolation
        u_interface = u_i + 0.5 * (u_ip1 - u_i);
      } else {
        u_interface = u_ip1 - 0.5 * (u_ip1 - u_i);
      }
      break;

    case AdvectionScheme::BeamWarming:
      // Second-order Beam-Warming (backward biased)
      if (v >= 0.0) {
        u_interface = u_i + 0.5 * (u_i - u_im1);
      } else {
        u_interface = u_ip1 + 0.5 * (u_ip1 - u_ip2);
      }
      break;

    case AdvectionScheme::Fromm:
      // Second-order Fromm (average of LW and BW)
      if (v >= 0.0) {
        u_interface = u_i + 0.25 * (u_ip1 - u_im1);
      } else {
        u_interface = u_ip1 - 0.25 * (u_ip2 - u_i);
      }
      break;

    case AdvectionScheme::MinMod:
    case AdvectionScheme::Superbee:
    case AdvectionScheme::VanLeer:
    case AdvectionScheme::MC: {
      // TVD schemes with flux limiting
      // Use MUSCL reconstruction with limiter
      if (v >= 0.0) {
        // Flow from left to right: reconstruct from left cell
        double delta_minus = u_i - u_im1;
        double delta_plus = u_ip1 - u_i;
        double r = ComputeSmoothnessRatio(delta_minus, delta_plus);
        double phi = ApplyLimiter(scheme, r);
        u_interface = u_i + 0.5 * phi * delta_plus;
      } else {
        // Flow from right to left: reconstruct from right cell
        double delta_minus = u_ip1 - u_i;
        double delta_plus = u_ip2 - u_ip1;
        double r = ComputeSmoothnessRatio(delta_plus, delta_minus);
        double phi = ApplyLimiter(scheme, r);
        u_interface = u_ip1 - 0.5 * phi * delta_minus;
      }
      break;
    }
  }

  return v * u_interface;
}

double ComputeAdvectionTerm2D(const std::vector<double>& u,
                               double vx, double vy,
                               int i, int j, int nx, int ny,
                               double dx, double dy,
                               AdvectionScheme scheme) {
  // Compute -div(v*u) = -(d(vx*u)/dx + d(vy*u)/dy)
  // Using finite volume approach: flux difference across cell

  if (i < 1 || i >= nx - 1 || j < 1 || j >= ny - 1) {
    return 0.0;  // Boundary cells handled separately
  }

  // X-direction fluxes
  // Extract row for x-sweep
  double u_row[5];  // Need 5 points for TVD stencil at interfaces i-1/2 and i+1/2
  for (int k = -2; k <= 2; ++k) {
    int ii = std::max(0, std::min(nx - 1, i + k));
    u_row[k + 2] = u[static_cast<size_t>(j * nx + ii)];
  }

  // Flux at i+1/2 (using indices 1,2,3,4 in local array -> global i-1,i,i+1,i+2)
  double flux_x_right = ComputeAdvectionFlux1D(u_row + 1, 4, vx, 1, dx, scheme);
  // Flux at i-1/2 (using indices 0,1,2,3 -> global i-2,i-1,i,i+1)
  double flux_x_left = ComputeAdvectionFlux1D(u_row, 4, vx, 1, dx, scheme);

  // Y-direction fluxes
  // Extract column for y-sweep
  double u_col[5];
  for (int k = -2; k <= 2; ++k) {
    int jj = std::max(0, std::min(ny - 1, j + k));
    u_col[k + 2] = u[static_cast<size_t>(jj * nx + i)];
  }

  // Flux at j+1/2
  double flux_y_top = ComputeAdvectionFlux1D(u_col + 1, 4, vy, 1, dy, scheme);
  // Flux at j-1/2
  double flux_y_bottom = ComputeAdvectionFlux1D(u_col, 4, vy, 1, dy, scheme);

  // Advection term: -div(v*u)
  double advection = -((flux_x_right - flux_x_left) / dx +
                       (flux_y_top - flux_y_bottom) / dy);

  return advection;
}

double ComputeAdvectionTerm3D(const std::vector<double>& u,
                               double vx, double vy, double vz,
                               int i, int j, int k, int nx, int ny, int nz,
                               double dx, double dy, double dz,
                               AdvectionScheme scheme) {
  if (i < 1 || i >= nx - 1 || j < 1 || j >= ny - 1 || k < 1 || k >= nz - 1) {
    return 0.0;
  }

  auto idx3d = [nx, ny](int ii, int jj, int kk) -> size_t {
    return static_cast<size_t>(kk * nx * ny + jj * nx + ii);
  };

  // X-direction
  double u_x[5];
  for (int m = -2; m <= 2; ++m) {
    int ii = std::max(0, std::min(nx - 1, i + m));
    u_x[m + 2] = u[idx3d(ii, j, k)];
  }
  double flux_x_right = ComputeAdvectionFlux1D(u_x + 1, 4, vx, 1, dx, scheme);
  double flux_x_left = ComputeAdvectionFlux1D(u_x, 4, vx, 1, dx, scheme);

  // Y-direction
  double u_y[5];
  for (int m = -2; m <= 2; ++m) {
    int jj = std::max(0, std::min(ny - 1, j + m));
    u_y[m + 2] = u[idx3d(i, jj, k)];
  }
  double flux_y_top = ComputeAdvectionFlux1D(u_y + 1, 4, vy, 1, dy, scheme);
  double flux_y_bottom = ComputeAdvectionFlux1D(u_y, 4, vy, 1, dy, scheme);

  // Z-direction
  double u_z[5];
  for (int m = -2; m <= 2; ++m) {
    int kk = std::max(0, std::min(nz - 1, k + m));
    u_z[m + 2] = u[idx3d(i, j, kk)];
  }
  double flux_z_front = ComputeAdvectionFlux1D(u_z + 1, 4, vz, 1, dz, scheme);
  double flux_z_back = ComputeAdvectionFlux1D(u_z, 4, vz, 1, dz, scheme);

  double advection = -((flux_x_right - flux_x_left) / dx +
                       (flux_y_top - flux_y_bottom) / dy +
                       (flux_z_front - flux_z_back) / dz);

  return advection;
}

double ComputeCFL(double vx, double vy, double vz,
                  double dx, double dy, double dz, double dt) {
  double cfl = 0.0;
  if (dx > 0.0) cfl = std::max(cfl, std::abs(vx) * dt / dx);
  if (dy > 0.0) cfl = std::max(cfl, std::abs(vy) * dt / dy);
  if (dz > 0.0) cfl = std::max(cfl, std::abs(vz) * dt / dz);
  return cfl;
}

double SuggestAdvectionDt(double vx, double vy, double vz,
                          double dx, double dy, double dz,
                          double cfl_target) {
  double max_speed_over_dx = 0.0;
  if (dx > 0.0) max_speed_over_dx = std::max(max_speed_over_dx, std::abs(vx) / dx);
  if (dy > 0.0) max_speed_over_dx = std::max(max_speed_over_dx, std::abs(vy) / dy);
  if (dz > 0.0) max_speed_over_dx = std::max(max_speed_over_dx, std::abs(vz) / dz);

  if (max_speed_over_dx < 1e-12) {
    return 1.0;  // No advection, return default
  }

  return cfl_target / max_speed_over_dx;
}
