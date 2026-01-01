#include "advection_tests.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <sstream>

#include "advection.h"

namespace {

// Top-hat (square wave) initial condition
double TopHat1D(double x, double x0, double width) {
  double half_width = width / 2.0;
  return (x >= x0 - half_width && x <= x0 + half_width) ? 1.0 : 0.0;
}

// Gaussian initial condition
double Gaussian1D(double x, double x0, double sigma) {
  double dx = x - x0;
  return std::exp(-dx * dx / (2.0 * sigma * sigma));
}

// 2D top-hat (cylinder)
double TopHat2D(double x, double y, double x0, double y0, double radius) {
  double dx = x - x0;
  double dy = y - y0;
  return (dx * dx + dy * dy <= radius * radius) ? 1.0 : 0.0;
}

// 2D Gaussian
double Gaussian2D(double x, double y, double x0, double y0, double sigma) {
  double dx = x - x0;
  double dy = y - y0;
  return std::exp(-(dx * dx + dy * dy) / (2.0 * sigma * sigma));
}

// Compute L1, L2, Linf errors between two arrays
void ComputeErrors(const std::vector<double>& numerical,
                   const std::vector<double>& exact,
                   double dx,
                   double& error_l1,
                   double& error_l2,
                   double& error_linf) {
  error_l1 = 0.0;
  error_l2 = 0.0;
  error_linf = 0.0;

  for (size_t i = 0; i < numerical.size(); ++i) {
    double diff = std::abs(numerical[i] - exact[i]);
    error_l1 += diff;
    error_l2 += diff * diff;
    error_linf = std::max(error_linf, diff);
  }

  error_l1 *= dx;
  error_l2 = std::sqrt(error_l2 * dx);
}

// Compute total mass (integral of u)
double ComputeMass(const std::vector<double>& u, double dx) {
  double mass = 0.0;
  for (double val : u) {
    mass += val;
  }
  return mass * dx;
}

double ComputeMass2D(const std::vector<double>& u, double dx, double dy) {
  double mass = 0.0;
  for (double val : u) {
    mass += val;
  }
  return mass * dx * dy;
}

// Explicit Euler time step for 1D advection
void AdvectionStep1D(std::vector<double>& u, double velocity,
                     double dx, double dt, AdvectionScheme scheme) {
  const int n = static_cast<int>(u.size());
  std::vector<double> u_new(n);

  // Compute fluxes at interfaces and update
  for (int i = 0; i < n; ++i) {
    // Periodic boundary conditions
    int im1 = (i - 1 + n) % n;
    int ip1 = (i + 1) % n;
    int ip2 = (i + 2) % n;

    // Create local array for flux computation (with periodic wrapping)
    double u_local[4];
    u_local[0] = u[static_cast<size_t>(im1)];
    u_local[1] = u[static_cast<size_t>(i)];
    u_local[2] = u[static_cast<size_t>(ip1)];
    u_local[3] = u[static_cast<size_t>(ip2)];

    // Flux at i+1/2
    double flux_right = ComputeAdvectionFlux1D(u_local, 4, velocity, 1, dx, scheme);

    // Flux at i-1/2: shift array left
    int im2 = (i - 2 + n) % n;
    u_local[0] = u[static_cast<size_t>(im2)];
    u_local[1] = u[static_cast<size_t>(im1)];
    u_local[2] = u[static_cast<size_t>(i)];
    u_local[3] = u[static_cast<size_t>(ip1)];
    double flux_left = ComputeAdvectionFlux1D(u_local, 4, velocity, 1, dx, scheme);

    // Update: u_new = u - dt/dx * (F_{i+1/2} - F_{i-1/2})
    u_new[static_cast<size_t>(i)] = u[static_cast<size_t>(i)] -
                                     dt / dx * (flux_right - flux_left);
  }

  u = std::move(u_new);
}

// Explicit Euler time step for 2D advection
void AdvectionStep2D(std::vector<double>& u, double vx, double vy,
                     int nx, int ny, double dx, double dy, double dt,
                     AdvectionScheme scheme) {
  std::vector<double> u_new(u.size());

  for (int j = 0; j < ny; ++j) {
    for (int i = 0; i < nx; ++i) {
      size_t idx = static_cast<size_t>(j * nx + i);

      // Compute advection term using periodic boundaries
      double advection = ComputeAdvectionTerm2D(u, vx, vy, i, j, nx, ny,
                                                 dx, dy, scheme);

      // For boundary cells, use simple upwind
      if (i < 1 || i >= nx - 1 || j < 1 || j >= ny - 1) {
        // Wrap indices for periodic BC
        int im1 = (i - 1 + nx) % nx;
        int ip1 = (i + 1) % nx;
        int jm1 = (j - 1 + ny) % ny;
        int jp1 = (j + 1) % ny;

        double u_i = u[idx];
        double u_im1 = u[static_cast<size_t>(j * nx + im1)];
        double u_ip1 = u[static_cast<size_t>(j * nx + ip1)];
        double u_jm1 = u[static_cast<size_t>(jm1 * nx + i)];
        double u_jp1 = u[static_cast<size_t>(jp1 * nx + i)];

        // Upwind for boundaries
        double du_dx = (vx >= 0) ? (u_i - u_im1) / dx : (u_ip1 - u_i) / dx;
        double du_dy = (vy >= 0) ? (u_i - u_jm1) / dy : (u_jp1 - u_i) / dy;
        advection = -(vx * du_dx + vy * du_dy);
      }

      u_new[idx] = u[idx] + dt * advection;
    }
  }

  u = std::move(u_new);
}

}  // namespace

AdvectionTestResult RunTopHatAdvectionTest1D(
    AdvectionScheme scheme, int nx, double velocity,
    double t_final, double cfl) {
  AdvectionTestResult result;
  result.name = "Top-Hat 1D";
  result.description = "Square wave advection in 1D with periodic BCs";
  result.scheme = scheme;
  result.nx = nx;
  result.ny = 1;
  result.t_final = t_final;

  // Domain [0, 1] with periodic BCs
  const double L = 1.0;
  const double dx = L / nx;
  const double dt = cfl * dx / std::abs(velocity);
  result.dt = dt;

  const int num_steps = static_cast<int>(std::ceil(t_final / dt));

  // Initial condition: top-hat centered at x=0.25 with width 0.25
  const double x0 = 0.25;
  const double width = 0.25;

  std::vector<double> u(static_cast<size_t>(nx));
  std::vector<double> u_exact(static_cast<size_t>(nx));

  for (int i = 0; i < nx; ++i) {
    double x = (static_cast<double>(i) + 0.5) * dx;  // Cell center
    u[static_cast<size_t>(i)] = TopHat1D(x, x0, width);
  }

  result.mass_initial = ComputeMass(u, dx);

  // Time stepping
  for (int step = 0; step < num_steps; ++step) {
    AdvectionStep1D(u, velocity, dx, dt, scheme);
  }

  // Exact solution (periodic domain)
  double actual_time = num_steps * dt;
  double x_shift = velocity * actual_time;
  x_shift = std::fmod(x_shift, L);
  if (x_shift < 0) x_shift += L;

  for (int i = 0; i < nx; ++i) {
    double x = (static_cast<double>(i) + 0.5) * dx;
    double x_orig = x - x_shift;
    if (x_orig < 0) x_orig += L;
    if (x_orig >= L) x_orig -= L;
    u_exact[static_cast<size_t>(i)] = TopHat1D(x_orig, x0, width);
  }

  result.mass_final = ComputeMass(u, dx);
  result.mass_error = std::abs(result.mass_final - result.mass_initial) /
                      std::max(1e-12, std::abs(result.mass_initial));

  ComputeErrors(u, u_exact, dx, result.error_l1, result.error_l2, result.error_linf);

  // Pass criterion: mass conservation within 1% and error reasonable
  result.passed = (result.mass_error < 0.01) && (result.error_l2 < 1.0);

  std::ostringstream note;
  note << "steps=" << num_steps << " CFL=" << std::fixed << std::setprecision(2) << cfl;
  result.note = note.str();

  return result;
}

AdvectionTestResult RunTopHatAdvectionTest2D(
    AdvectionScheme scheme, int nx, int ny, double vx, double vy,
    double t_final, double cfl) {
  AdvectionTestResult result;
  result.name = "Top-Hat 2D";
  result.description = "Cylinder advection in 2D with periodic BCs";
  result.scheme = scheme;
  result.nx = nx;
  result.ny = ny;
  result.t_final = t_final;

  const double Lx = 1.0;
  const double Ly = 1.0;
  const double dx = Lx / nx;
  const double dy = Ly / ny;

  double max_v = std::max(std::abs(vx) / dx, std::abs(vy) / dy);
  const double dt = cfl / max_v;
  result.dt = dt;

  const int num_steps = static_cast<int>(std::ceil(t_final / dt));

  // Initial condition: cylinder centered at (0.25, 0.5) with radius 0.15
  const double x0 = 0.25;
  const double y0 = 0.5;
  const double radius = 0.15;

  std::vector<double> u(static_cast<size_t>(nx * ny));

  for (int j = 0; j < ny; ++j) {
    double y = (static_cast<double>(j) + 0.5) * dy;
    for (int i = 0; i < nx; ++i) {
      double x = (static_cast<double>(i) + 0.5) * dx;
      u[static_cast<size_t>(j * nx + i)] = TopHat2D(x, y, x0, y0, radius);
    }
  }

  result.mass_initial = ComputeMass2D(u, dx, dy);

  // Time stepping
  for (int step = 0; step < num_steps; ++step) {
    AdvectionStep2D(u, vx, vy, nx, ny, dx, dy, dt, scheme);
  }

  // Exact solution
  double actual_time = num_steps * dt;
  double x_shift = vx * actual_time;
  double y_shift = vy * actual_time;
  x_shift = std::fmod(x_shift, Lx);
  y_shift = std::fmod(y_shift, Ly);
  if (x_shift < 0) x_shift += Lx;
  if (y_shift < 0) y_shift += Ly;

  std::vector<double> u_exact(static_cast<size_t>(nx * ny));
  for (int j = 0; j < ny; ++j) {
    double y = (static_cast<double>(j) + 0.5) * dy;
    for (int i = 0; i < nx; ++i) {
      double x = (static_cast<double>(i) + 0.5) * dx;
      double x_orig = x - x_shift;
      double y_orig = y - y_shift;
      if (x_orig < 0) x_orig += Lx;
      if (y_orig < 0) y_orig += Ly;
      if (x_orig >= Lx) x_orig -= Lx;
      if (y_orig >= Ly) y_orig -= Ly;
      u_exact[static_cast<size_t>(j * nx + i)] = TopHat2D(x_orig, y_orig, x0, y0, radius);
    }
  }

  result.mass_final = ComputeMass2D(u, dx, dy);
  result.mass_error = std::abs(result.mass_final - result.mass_initial) /
                      std::max(1e-12, std::abs(result.mass_initial));

  ComputeErrors(u, u_exact, dx * dy, result.error_l1, result.error_l2, result.error_linf);

  result.passed = (result.mass_error < 0.01) && (result.error_l2 < 1.0);

  std::ostringstream note;
  note << "steps=" << num_steps << " CFL=" << std::fixed << std::setprecision(2) << cfl;
  result.note = note.str();

  return result;
}

AdvectionTestResult RunGaussianAdvectionTest1D(
    AdvectionScheme scheme, int nx, double velocity,
    double t_final, double cfl) {
  AdvectionTestResult result;
  result.name = "Gaussian 1D";
  result.description = "Smooth Gaussian advection in 1D";
  result.scheme = scheme;
  result.nx = nx;
  result.ny = 1;
  result.t_final = t_final;

  const double L = 1.0;
  const double dx = L / nx;
  const double dt = cfl * dx / std::abs(velocity);
  result.dt = dt;

  const int num_steps = static_cast<int>(std::ceil(t_final / dt));

  // Initial condition: Gaussian centered at x=0.25 with sigma=0.05
  const double x0 = 0.25;
  const double sigma = 0.05;

  std::vector<double> u(static_cast<size_t>(nx));

  for (int i = 0; i < nx; ++i) {
    double x = (static_cast<double>(i) + 0.5) * dx;
    u[static_cast<size_t>(i)] = Gaussian1D(x, x0, sigma);
  }

  result.mass_initial = ComputeMass(u, dx);

  // Time stepping
  for (int step = 0; step < num_steps; ++step) {
    AdvectionStep1D(u, velocity, dx, dt, scheme);
  }

  // Exact solution
  double actual_time = num_steps * dt;
  double x_shift = velocity * actual_time;
  x_shift = std::fmod(x_shift, L);
  if (x_shift < 0) x_shift += L;

  std::vector<double> u_exact(static_cast<size_t>(nx));
  for (int i = 0; i < nx; ++i) {
    double x = (static_cast<double>(i) + 0.5) * dx;
    double x_orig = x - x_shift;
    if (x_orig < 0) x_orig += L;
    if (x_orig >= L) x_orig -= L;
    u_exact[static_cast<size_t>(i)] = Gaussian1D(x_orig, x0, sigma);
  }

  result.mass_final = ComputeMass(u, dx);
  result.mass_error = std::abs(result.mass_final - result.mass_initial) /
                      std::max(1e-12, std::abs(result.mass_initial));

  ComputeErrors(u, u_exact, dx, result.error_l1, result.error_l2, result.error_linf);

  result.passed = (result.mass_error < 0.01) && (result.error_l2 < 0.5);

  std::ostringstream note;
  note << "steps=" << num_steps << " L2=" << std::scientific << std::setprecision(2) << result.error_l2;
  result.note = note.str();

  return result;
}

AdvectionTestResult RunRotationTest2D(
    AdvectionScheme scheme, int nx, int ny, double omega,
    int rotations, double cfl) {
  AdvectionTestResult result;
  result.name = "Rotation 2D";
  result.description = "Solid body rotation test";
  result.scheme = scheme;
  result.nx = nx;
  result.ny = ny;
  result.t_final = 2.0 * M_PI * rotations / omega;

  const double Lx = 2.0;  // Domain [-1, 1] x [-1, 1]
  const double Ly = 2.0;
  const double dx = Lx / nx;
  const double dy = Ly / ny;

  // Maximum velocity at edge of domain
  double max_v = omega * 1.0;  // |v| = omega * r, max at r=1
  const double dt = cfl * std::min(dx, dy) / max_v;
  result.dt = dt;

  const int num_steps = static_cast<int>(std::ceil(result.t_final / dt));

  // Initial condition: Gaussian blob off-center
  const double x0 = 0.5;
  const double y0 = 0.0;
  const double sigma = 0.1;

  std::vector<double> u(static_cast<size_t>(nx * ny));
  std::vector<double> u_initial(static_cast<size_t>(nx * ny));

  for (int j = 0; j < ny; ++j) {
    double y = -1.0 + (static_cast<double>(j) + 0.5) * dy;
    for (int i = 0; i < nx; ++i) {
      double x = -1.0 + (static_cast<double>(i) + 0.5) * dx;
      double val = Gaussian2D(x, y, x0, y0, sigma);
      u[static_cast<size_t>(j * nx + i)] = val;
      u_initial[static_cast<size_t>(j * nx + i)] = val;
    }
  }

  result.mass_initial = ComputeMass2D(u, dx, dy);

  // Time stepping with rotation velocity field: v = omega * (-y, x)
  for (int step = 0; step < num_steps; ++step) {
    std::vector<double> u_new(u.size());

    for (int j = 0; j < ny; ++j) {
      double y = -1.0 + (static_cast<double>(j) + 0.5) * dy;
      for (int i = 0; i < nx; ++i) {
        double x = -1.0 + (static_cast<double>(i) + 0.5) * dx;
        double vx = -omega * y;
        double vy = omega * x;

        size_t idx = static_cast<size_t>(j * nx + i);
        double advection = ComputeAdvectionTerm2D(u, vx, vy, i, j, nx, ny,
                                                   dx, dy, scheme);

        // Boundary handling (simple extrapolation for rotation)
        if (i < 1 || i >= nx - 1 || j < 1 || j >= ny - 1) {
          advection = 0.0;  // Zero flux at boundaries
        }

        u_new[idx] = u[idx] + dt * advection;
      }
    }

    u = std::move(u_new);
  }

  // After full rotations, solution should match initial
  result.mass_final = ComputeMass2D(u, dx, dy);
  result.mass_error = std::abs(result.mass_final - result.mass_initial) /
                      std::max(1e-12, std::abs(result.mass_initial));

  ComputeErrors(u, u_initial, dx * dy, result.error_l1, result.error_l2, result.error_linf);

  result.passed = (result.mass_error < 0.05) && (result.error_l2 < 1.0);

  std::ostringstream note;
  note << rotations << " rotation(s), steps=" << num_steps;
  result.note = note.str();

  return result;
}

std::vector<AdvectionTestResult> RunAdvectionConvergenceStudy(
    AdvectionScheme scheme, const std::vector<int>& resolutions,
    double velocity, double t_final, double cfl) {
  std::vector<AdvectionTestResult> results;

  for (int nx : resolutions) {
    AdvectionTestResult r = RunGaussianAdvectionTest1D(scheme, nx, velocity, t_final, cfl);
    r.description = "Convergence study at nx=" + std::to_string(nx);
    results.push_back(r);
  }

  return results;
}

std::vector<AdvectionTestResult> RunAdvectionTestSuite(AdvectionScheme scheme) {
  std::vector<AdvectionTestResult> results;

  results.push_back(RunTopHatAdvectionTest1D(scheme, 100, 1.0, 1.0, 0.5));
  results.push_back(RunGaussianAdvectionTest1D(scheme, 100, 1.0, 1.0, 0.5));
  results.push_back(RunTopHatAdvectionTest2D(scheme, 64, 64, 1.0, 0.5, 0.5, 0.5));
  results.push_back(RunRotationTest2D(scheme, 64, 64, 1.0, 1, 0.5));

  return results;
}

void PrintAdvectionTestResults(const std::vector<AdvectionTestResult>& results) {
  std::cout << std::left << std::setw(20) << "Test"
            << std::setw(12) << "Scheme"
            << std::setw(10) << "Status"
            << std::setw(12) << "L2 Error"
            << std::setw(12) << "Mass Err"
            << "Note\n";
  std::cout << std::string(80, '-') << "\n";

  for (const auto& r : results) {
    std::cout << std::left << std::setw(20) << r.name
              << std::setw(12) << AdvectionSchemeToString(r.scheme)
              << std::setw(10) << (r.passed ? "[OK]" : "[FAIL]")
              << std::scientific << std::setprecision(2)
              << std::setw(12) << r.error_l2
              << std::setw(12) << r.mass_error
              << r.note << "\n";
  }
}

std::vector<AdvectionTestResult> CompareAdvectionSchemes(
    const std::function<AdvectionTestResult(AdvectionScheme)>& test_runner,
    const std::vector<AdvectionScheme>& schemes) {
  std::vector<AdvectionTestResult> results;
  for (AdvectionScheme scheme : schemes) {
    results.push_back(test_runner(scheme));
  }
  return results;
}
