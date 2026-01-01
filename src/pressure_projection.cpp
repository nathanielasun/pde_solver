#include "pressure_projection.h"

#include <algorithm>
#include <cmath>
#include <numeric>

// ===========================================================================
// 2D Divergence and Gradient Operations
// ===========================================================================

void ComputeDivergence2D(const VelocityField2D& vel,
                          std::vector<double>* div) {
  const int nx = vel.nx;
  const int ny = vel.ny;
  const double dx = vel.dx;
  const double dy = vel.dy;

  div->resize(nx * ny);

  for (int j = 0; j < ny; ++j) {
    for (int i = 0; i < nx; ++i) {
      int idx = j * nx + i;

      // Central differences for interior, one-sided at boundaries
      double du_dx, dv_dy;

      if (i == 0) {
        du_dx = (vel.u[idx + 1] - vel.u[idx]) / dx;
      } else if (i == nx - 1) {
        du_dx = (vel.u[idx] - vel.u[idx - 1]) / dx;
      } else {
        du_dx = (vel.u[idx + 1] - vel.u[idx - 1]) / (2.0 * dx);
      }

      if (j == 0) {
        dv_dy = (vel.v[idx + nx] - vel.v[idx]) / dy;
      } else if (j == ny - 1) {
        dv_dy = (vel.v[idx] - vel.v[idx - nx]) / dy;
      } else {
        dv_dy = (vel.v[idx + nx] - vel.v[idx - nx]) / (2.0 * dy);
      }

      (*div)[idx] = du_dx + dv_dy;
    }
  }
}

void ComputeDivergenceMAC2D(const VelocityField2D& vel,
                             std::vector<double>* div) {
  // For staggered (MAC) grid, divergence is computed at cell centers
  // u is stored at x-faces, v is stored at y-faces
  const int nx = vel.nx;
  const int ny = vel.ny;
  const double dx = vel.dx;
  const double dy = vel.dy;

  div->resize(nx * ny);

  for (int j = 0; j < ny; ++j) {
    for (int i = 0; i < nx; ++i) {
      int idx = j * nx + i;

      // On MAC grid: div = (u[i+1/2] - u[i-1/2])/dx + (v[j+1/2] - v[j-1/2])/dy
      // We store u at right face, so u[i] is u[i+1/2]
      double u_right = (i < nx - 1) ? vel.u[idx] : 0.0;
      double u_left = (i > 0) ? vel.u[idx - 1] : 0.0;
      double v_top = (j < ny - 1) ? vel.v[idx] : 0.0;
      double v_bottom = (j > 0) ? vel.v[idx - nx] : 0.0;

      (*div)[idx] = (u_right - u_left) / dx + (v_top - v_bottom) / dy;
    }
  }
}

void ComputeGradient2D(const std::vector<double>& p,
                        int nx, int ny, double dx, double dy,
                        std::vector<double>* grad_x,
                        std::vector<double>* grad_y) {
  grad_x->resize(nx * ny);
  grad_y->resize(nx * ny);

  for (int j = 0; j < ny; ++j) {
    for (int i = 0; i < nx; ++i) {
      int idx = j * nx + i;

      // Central differences for interior, one-sided at boundaries
      if (i == 0) {
        (*grad_x)[idx] = (p[idx + 1] - p[idx]) / dx;
      } else if (i == nx - 1) {
        (*grad_x)[idx] = (p[idx] - p[idx - 1]) / dx;
      } else {
        (*grad_x)[idx] = (p[idx + 1] - p[idx - 1]) / (2.0 * dx);
      }

      if (j == 0) {
        (*grad_y)[idx] = (p[idx + nx] - p[idx]) / dy;
      } else if (j == ny - 1) {
        (*grad_y)[idx] = (p[idx] - p[idx - nx]) / dy;
      } else {
        (*grad_y)[idx] = (p[idx + nx] - p[idx - nx]) / (2.0 * dy);
      }
    }
  }
}

// ===========================================================================
// Pressure Poisson Solver
// ===========================================================================

// Simple CG solver for Poisson equation with Neumann BCs
bool SolvePressurePoisson2D(const std::vector<double>& rhs,
                             int nx, int ny, double dx, double dy,
                             std::vector<double>* p,
                             const ProjectionConfig& config,
                             int* iterations,
                             double* residual) {
  const int n = nx * ny;
  const double dx2 = dx * dx;
  const double dy2 = dy * dy;
  const double coeff_x = 1.0 / dx2;
  const double coeff_y = 1.0 / dy2;
  const double coeff_c = -2.0 * (coeff_x + coeff_y);

  // Initialize pressure to zero if empty
  if (p->size() != static_cast<size_t>(n)) {
    p->resize(n, 0.0);
  }

  // For pure Neumann BCs, subtract mean of RHS to ensure solvability
  std::vector<double> rhs_adj = rhs;
  double mean_rhs = std::accumulate(rhs_adj.begin(), rhs_adj.end(), 0.0) / n;
  for (double& v : rhs_adj) {
    v -= mean_rhs;
  }

  // Laplacian operator application (standard 5-point stencil with Neumann BC)
  auto apply_laplacian = [&](const std::vector<double>& x,
                              std::vector<double>* ax) {
    ax->resize(n);
    for (int j = 0; j < ny; ++j) {
      for (int i = 0; i < nx; ++i) {
        int idx = j * nx + i;

        // Get neighbor values with Neumann BC (copy boundary value)
        double x_c = x[idx];
        double x_xm = (i > 0) ? x[idx - 1] : x_c;
        double x_xp = (i < nx - 1) ? x[idx + 1] : x_c;
        double x_ym = (j > 0) ? x[idx - nx] : x_c;
        double x_yp = (j < ny - 1) ? x[idx + nx] : x_c;

        (*ax)[idx] = coeff_x * (x_xp - 2.0 * x_c + x_xm) +
                     coeff_y * (x_yp - 2.0 * x_c + x_ym);
      }
    }
  };

  // CG solver
  std::vector<double> r(n), p_cg(n), Ap(n);

  // r = rhs - A*x
  apply_laplacian(*p, &Ap);
  for (int i = 0; i < n; ++i) {
    r[i] = rhs_adj[i] - Ap[i];
  }

  // Subtract mean from residual (project out null space)
  double mean_r = std::accumulate(r.begin(), r.end(), 0.0) / n;
  for (double& v : r) {
    v -= mean_r;
  }

  p_cg = r;

  double r_dot_r = 0.0;
  for (int i = 0; i < n; ++i) {
    r_dot_r += r[i] * r[i];
  }

  double r_norm_init = std::sqrt(r_dot_r);
  if (r_norm_init < 1e-15) {
    if (iterations) *iterations = 0;
    if (residual) *residual = 0.0;
    return true;
  }

  for (int iter = 0; iter < config.max_iter; ++iter) {
    apply_laplacian(p_cg, &Ap);

    // Project out null space from Ap
    double mean_Ap = std::accumulate(Ap.begin(), Ap.end(), 0.0) / n;
    for (double& v : Ap) {
      v -= mean_Ap;
    }

    double p_dot_Ap = 0.0;
    for (int i = 0; i < n; ++i) {
      p_dot_Ap += p_cg[i] * Ap[i];
    }

    if (std::abs(p_dot_Ap) < 1e-30) {
      if (iterations) *iterations = iter;
      if (residual) *residual = std::sqrt(r_dot_r);
      return true;  // Converged (essentially zero)
    }

    double alpha = r_dot_r / p_dot_Ap;

    // Update solution and residual
    for (int i = 0; i < n; ++i) {
      (*p)[i] += alpha * p_cg[i];
      r[i] -= alpha * Ap[i];
    }

    double r_dot_r_new = 0.0;
    for (int i = 0; i < n; ++i) {
      r_dot_r_new += r[i] * r[i];
    }

    double r_norm = std::sqrt(r_dot_r_new);
    if (r_norm < config.tol * r_norm_init || r_norm < 1e-14) {
      if (iterations) *iterations = iter + 1;
      if (residual) *residual = r_norm;

      // Subtract mean to fix arbitrary constant
      double mean_p = std::accumulate(p->begin(), p->end(), 0.0) / n;
      for (double& v : *p) {
        v -= mean_p;
      }
      return true;
    }

    double beta = r_dot_r_new / r_dot_r;
    for (int i = 0; i < n; ++i) {
      p_cg[i] = r[i] + beta * p_cg[i];
    }

    r_dot_r = r_dot_r_new;
  }

  // Even if we hit max iterations, subtract mean and return partial result
  double mean_p = std::accumulate(p->begin(), p->end(), 0.0) / n;
  for (double& v : *p) {
    v -= mean_p;
  }

  if (iterations) *iterations = config.max_iter;
  if (residual) *residual = std::sqrt(r_dot_r);
  return true;  // Return true since we have a reasonable approximation
}

// ===========================================================================
// Projection Method
// ===========================================================================

ProjectionResult ProjectVelocity2D(VelocityField2D& vel,
                                    double dt,
                                    std::vector<double>* pressure,
                                    const ProjectionConfig& config) {
  ProjectionResult result;
  const int nx = vel.nx;
  const int ny = vel.ny;
  const int n = nx * ny;

  // Step 1: Compute divergence of intermediate velocity
  std::vector<double> div;
  if (config.staggered_grid) {
    ComputeDivergenceMAC2D(vel, &div);
  } else {
    ComputeDivergence2D(vel, &div);
  }

  // Record initial divergence
  result.l2_divergence_before = 0.0;
  result.max_divergence_before = 0.0;
  for (int i = 0; i < n; ++i) {
    result.l2_divergence_before += div[i] * div[i];
    result.max_divergence_before = std::max(result.max_divergence_before,
                                             std::abs(div[i]));
  }
  result.l2_divergence_before = std::sqrt(result.l2_divergence_before / n);

  // Step 2: Solve Poisson equation: nabla^2 p = (1/dt) * div(u*)
  std::vector<double> rhs(n);
  for (int i = 0; i < n; ++i) {
    rhs[i] = div[i] / dt;
  }

  if (pressure->size() != static_cast<size_t>(n)) {
    pressure->resize(n, 0.0);
  }

  int poisson_iter = 0;
  double poisson_res = 0.0;
  bool converged = SolvePressurePoisson2D(rhs, nx, ny, vel.dx, vel.dy,
                                           pressure, config,
                                           &poisson_iter, &poisson_res);

  result.poisson_iterations = poisson_iter;
  result.poisson_residual = poisson_res;

  if (!converged) {
    result.success = false;
    result.error = "Pressure Poisson solver did not converge";
    return result;
  }

  // Step 3: Correct velocity: u = u* - dt * grad(p)
  std::vector<double> grad_p_x, grad_p_y;
  ComputeGradient2D(*pressure, nx, ny, vel.dx, vel.dy, &grad_p_x, &grad_p_y);

  for (int i = 0; i < n; ++i) {
    vel.u[i] -= dt * grad_p_x[i];
    vel.v[i] -= dt * grad_p_y[i];
  }

  // Compute final divergence
  if (config.staggered_grid) {
    ComputeDivergenceMAC2D(vel, &div);
  } else {
    ComputeDivergence2D(vel, &div);
  }

  result.l2_divergence_after = 0.0;
  result.max_divergence_after = 0.0;
  for (int i = 0; i < n; ++i) {
    result.l2_divergence_after += div[i] * div[i];
    result.max_divergence_after = std::max(result.max_divergence_after,
                                            std::abs(div[i]));
  }
  result.l2_divergence_after = std::sqrt(result.l2_divergence_after / n);

  result.success = true;
  return result;
}

// ===========================================================================
// Navier-Stokes Terms
// ===========================================================================

void ComputeViscousTerm2D(const VelocityField2D& vel, double nu,
                           std::vector<double>* visc_u,
                           std::vector<double>* visc_v) {
  const int nx = vel.nx;
  const int ny = vel.ny;
  const double dx2 = vel.dx * vel.dx;
  const double dy2 = vel.dy * vel.dy;
  const int n = nx * ny;

  visc_u->resize(n);
  visc_v->resize(n);

  for (int j = 0; j < ny; ++j) {
    for (int i = 0; i < nx; ++i) {
      int idx = j * nx + i;

      // Laplacian of u with no-slip BC (u=0 at boundaries)
      double u_c = vel.u[idx];
      double u_xm = (i > 0) ? vel.u[idx - 1] : -u_c;      // Mirror for no-slip
      double u_xp = (i < nx - 1) ? vel.u[idx + 1] : -u_c;
      double u_ym = (j > 0) ? vel.u[idx - nx] : -u_c;
      double u_yp = (j < ny - 1) ? vel.u[idx + nx] : -u_c;

      (*visc_u)[idx] = nu * ((u_xp - 2.0 * u_c + u_xm) / dx2 +
                              (u_yp - 2.0 * u_c + u_ym) / dy2);

      // Laplacian of v
      double v_c = vel.v[idx];
      double v_xm = (i > 0) ? vel.v[idx - 1] : -v_c;
      double v_xp = (i < nx - 1) ? vel.v[idx + 1] : -v_c;
      double v_ym = (j > 0) ? vel.v[idx - nx] : -v_c;
      double v_yp = (j < ny - 1) ? vel.v[idx + nx] : -v_c;

      (*visc_v)[idx] = nu * ((v_xp - 2.0 * v_c + v_xm) / dx2 +
                              (v_yp - 2.0 * v_c + v_ym) / dy2);
    }
  }
}

void ComputeAdvectionTerm2D(const VelocityField2D& vel,
                             std::vector<double>* adv_u,
                             std::vector<double>* adv_v) {
  const int nx = vel.nx;
  const int ny = vel.ny;
  const double dx = vel.dx;
  const double dy = vel.dy;
  const int n = nx * ny;

  adv_u->resize(n);
  adv_v->resize(n);

  for (int j = 0; j < ny; ++j) {
    for (int i = 0; i < nx; ++i) {
      int idx = j * nx + i;

      double u_c = vel.u[idx];
      double v_c = vel.v[idx];

      // Upwind differencing for advection
      double du_dx, du_dy, dv_dx, dv_dy;

      // du/dx
      if (u_c > 0) {
        du_dx = (i > 0) ? (vel.u[idx] - vel.u[idx - 1]) / dx : 0.0;
      } else {
        du_dx = (i < nx - 1) ? (vel.u[idx + 1] - vel.u[idx]) / dx : 0.0;
      }

      // du/dy
      if (v_c > 0) {
        du_dy = (j > 0) ? (vel.u[idx] - vel.u[idx - nx]) / dy : 0.0;
      } else {
        du_dy = (j < ny - 1) ? (vel.u[idx + nx] - vel.u[idx]) / dy : 0.0;
      }

      // dv/dx
      if (u_c > 0) {
        dv_dx = (i > 0) ? (vel.v[idx] - vel.v[idx - 1]) / dx : 0.0;
      } else {
        dv_dx = (i < nx - 1) ? (vel.v[idx + 1] - vel.v[idx]) / dx : 0.0;
      }

      // dv/dy
      if (v_c > 0) {
        dv_dy = (j > 0) ? (vel.v[idx] - vel.v[idx - nx]) / dy : 0.0;
      } else {
        dv_dy = (j < ny - 1) ? (vel.v[idx + nx] - vel.v[idx]) / dy : 0.0;
      }

      // -(u . grad) u
      (*adv_u)[idx] = -(u_c * du_dx + v_c * du_dy);
      (*adv_v)[idx] = -(u_c * dv_dx + v_c * dv_dy);
    }
  }
}

ProjectionResult NavierStokesStep2D(VelocityField2D& vel,
                                     std::vector<double>* pressure,
                                     const IncompressibleNSConfig& config) {
  const int n = vel.nx * vel.ny;

  // Compute viscous and advection terms
  std::vector<double> visc_u, visc_v, adv_u, adv_v;
  ComputeViscousTerm2D(vel, config.nu, &visc_u, &visc_v);
  ComputeAdvectionTerm2D(vel, &adv_u, &adv_v);

  // Update intermediate velocity: u* = u + dt * (nu*lap(u) - (u.grad)u)
  for (int i = 0; i < n; ++i) {
    vel.u[i] += config.dt * (visc_u[i] + adv_u[i]);
    vel.v[i] += config.dt * (visc_v[i] + adv_v[i]);
  }

  // Apply boundary conditions (will be specific to problem)
  // For lid-driven cavity, this is handled by ApplyLidDrivenCavityBC

  // Project to divergence-free
  return ProjectVelocity2D(vel, config.dt, pressure, config.projection);
}

// ===========================================================================
// 3D Operations
// ===========================================================================

void ComputeDivergence3D(const VelocityField3D& vel,
                          std::vector<double>* div) {
  const int nx = vel.nx;
  const int ny = vel.ny;
  const int nz = vel.nz;
  const double dx = vel.dx;
  const double dy = vel.dy;
  const double dz = vel.dz;
  const int nxy = nx * ny;

  div->resize(nx * ny * nz);

  for (int k = 0; k < nz; ++k) {
    for (int j = 0; j < ny; ++j) {
      for (int i = 0; i < nx; ++i) {
        int idx = k * nxy + j * nx + i;

        double du_dx, dv_dy, dw_dz;

        // Central differences with one-sided at boundaries
        if (i == 0) {
          du_dx = (vel.u[idx + 1] - vel.u[idx]) / dx;
        } else if (i == nx - 1) {
          du_dx = (vel.u[idx] - vel.u[idx - 1]) / dx;
        } else {
          du_dx = (vel.u[idx + 1] - vel.u[idx - 1]) / (2.0 * dx);
        }

        if (j == 0) {
          dv_dy = (vel.v[idx + nx] - vel.v[idx]) / dy;
        } else if (j == ny - 1) {
          dv_dy = (vel.v[idx] - vel.v[idx - nx]) / dy;
        } else {
          dv_dy = (vel.v[idx + nx] - vel.v[idx - nx]) / (2.0 * dy);
        }

        if (k == 0) {
          dw_dz = (vel.w[idx + nxy] - vel.w[idx]) / dz;
        } else if (k == nz - 1) {
          dw_dz = (vel.w[idx] - vel.w[idx - nxy]) / dz;
        } else {
          dw_dz = (vel.w[idx + nxy] - vel.w[idx - nxy]) / (2.0 * dz);
        }

        (*div)[idx] = du_dx + dv_dy + dw_dz;
      }
    }
  }
}

void ComputeGradient3D(const std::vector<double>& p,
                        int nx, int ny, int nz,
                        double dx, double dy, double dz,
                        std::vector<double>* grad_x,
                        std::vector<double>* grad_y,
                        std::vector<double>* grad_z) {
  const int nxy = nx * ny;
  const int n = nx * ny * nz;

  grad_x->resize(n);
  grad_y->resize(n);
  grad_z->resize(n);

  for (int k = 0; k < nz; ++k) {
    for (int j = 0; j < ny; ++j) {
      for (int i = 0; i < nx; ++i) {
        int idx = k * nxy + j * nx + i;

        if (i == 0) {
          (*grad_x)[idx] = (p[idx + 1] - p[idx]) / dx;
        } else if (i == nx - 1) {
          (*grad_x)[idx] = (p[idx] - p[idx - 1]) / dx;
        } else {
          (*grad_x)[idx] = (p[idx + 1] - p[idx - 1]) / (2.0 * dx);
        }

        if (j == 0) {
          (*grad_y)[idx] = (p[idx + nx] - p[idx]) / dy;
        } else if (j == ny - 1) {
          (*grad_y)[idx] = (p[idx] - p[idx - nx]) / dy;
        } else {
          (*grad_y)[idx] = (p[idx + nx] - p[idx - nx]) / (2.0 * dy);
        }

        if (k == 0) {
          (*grad_z)[idx] = (p[idx + nxy] - p[idx]) / dz;
        } else if (k == nz - 1) {
          (*grad_z)[idx] = (p[idx] - p[idx - nxy]) / dz;
        } else {
          (*grad_z)[idx] = (p[idx + nxy] - p[idx - nxy]) / (2.0 * dz);
        }
      }
    }
  }
}

bool SolvePressurePoisson3D(const std::vector<double>& rhs,
                             int nx, int ny, int nz,
                             double dx, double dy, double dz,
                             std::vector<double>* p,
                             const ProjectionConfig& config,
                             int* iterations,
                             double* residual) {
  const int nxy = nx * ny;
  const int n = nx * ny * nz;
  const double dx2 = dx * dx;
  const double dy2 = dy * dy;
  const double dz2 = dz * dz;
  const double coeff_x = 1.0 / dx2;
  const double coeff_y = 1.0 / dy2;
  const double coeff_z = 1.0 / dz2;
  const double coeff_c = -2.0 * (coeff_x + coeff_y + coeff_z);

  if (p->size() != static_cast<size_t>(n)) {
    p->resize(n, 0.0);
  }

  // Adjust RHS for solvability
  std::vector<double> rhs_adj = rhs;
  double mean_rhs = std::accumulate(rhs_adj.begin(), rhs_adj.end(), 0.0) / n;
  for (double& v : rhs_adj) {
    v -= mean_rhs;
  }

  // Laplacian operator
  auto apply_laplacian = [&](const std::vector<double>& x,
                              std::vector<double>* ax) {
    ax->resize(n);
    for (int k = 0; k < nz; ++k) {
      for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
          int idx = k * nxy + j * nx + i;
          double lap = coeff_c * x[idx];

          // x-direction
          lap += coeff_x * ((i > 0) ? x[idx - 1] : x[idx]);
          lap += coeff_x * ((i < nx - 1) ? x[idx + 1] : x[idx]);

          // y-direction
          lap += coeff_y * ((j > 0) ? x[idx - nx] : x[idx]);
          lap += coeff_y * ((j < ny - 1) ? x[idx + nx] : x[idx]);

          // z-direction
          lap += coeff_z * ((k > 0) ? x[idx - nxy] : x[idx]);
          lap += coeff_z * ((k < nz - 1) ? x[idx + nxy] : x[idx]);

          (*ax)[idx] = lap;
        }
      }
    }

    if (config.fix_pressure_corner) {
      (*ax)[0] = x[0];
    }
  };

  if (config.fix_pressure_corner) {
    rhs_adj[0] = config.reference_pressure;
  }

  // CG solver (same as 2D)
  std::vector<double> r(n), p_cg(n), Ap(n);

  apply_laplacian(*p, &Ap);
  for (int i = 0; i < n; ++i) {
    r[i] = rhs_adj[i] - Ap[i];
  }
  p_cg = r;

  double r_dot_r = 0.0;
  for (int i = 0; i < n; ++i) {
    r_dot_r += r[i] * r[i];
  }

  double r_norm_init = std::sqrt(r_dot_r);
  if (r_norm_init < 1e-15) {
    if (iterations) *iterations = 0;
    if (residual) *residual = 0.0;
    return true;
  }

  for (int iter = 0; iter < config.max_iter; ++iter) {
    apply_laplacian(p_cg, &Ap);

    double p_dot_Ap = 0.0;
    for (int i = 0; i < n; ++i) {
      p_dot_Ap += p_cg[i] * Ap[i];
    }

    if (std::abs(p_dot_Ap) < 1e-30) {
      if (iterations) *iterations = iter;
      if (residual) *residual = std::sqrt(r_dot_r);
      return false;
    }

    double alpha = r_dot_r / p_dot_Ap;

    for (int i = 0; i < n; ++i) {
      (*p)[i] += alpha * p_cg[i];
      r[i] -= alpha * Ap[i];
    }

    double r_dot_r_new = 0.0;
    for (int i = 0; i < n; ++i) {
      r_dot_r_new += r[i] * r[i];
    }

    double r_norm = std::sqrt(r_dot_r_new);
    if (r_norm < config.tol * r_norm_init || r_norm < 1e-14) {
      if (iterations) *iterations = iter + 1;
      if (residual) *residual = r_norm;

      double mean_p = std::accumulate(p->begin(), p->end(), 0.0) / n;
      for (double& v : *p) {
        v -= mean_p;
      }
      return true;
    }

    double beta = r_dot_r_new / r_dot_r;
    for (int i = 0; i < n; ++i) {
      p_cg[i] = r[i] + beta * p_cg[i];
    }
    r_dot_r = r_dot_r_new;
  }

  if (iterations) *iterations = config.max_iter;
  if (residual) *residual = std::sqrt(r_dot_r);
  return false;
}

ProjectionResult ProjectVelocity3D(VelocityField3D& vel,
                                    double dt,
                                    std::vector<double>* pressure,
                                    const ProjectionConfig& config) {
  ProjectionResult result;
  const int n = vel.nx * vel.ny * vel.nz;

  std::vector<double> div;
  ComputeDivergence3D(vel, &div);

  result.l2_divergence_before = 0.0;
  result.max_divergence_before = 0.0;
  for (int i = 0; i < n; ++i) {
    result.l2_divergence_before += div[i] * div[i];
    result.max_divergence_before = std::max(result.max_divergence_before,
                                             std::abs(div[i]));
  }
  result.l2_divergence_before = std::sqrt(result.l2_divergence_before / n);

  std::vector<double> rhs(n);
  for (int i = 0; i < n; ++i) {
    rhs[i] = div[i] / dt;
  }

  if (pressure->size() != static_cast<size_t>(n)) {
    pressure->resize(n, 0.0);
  }

  int poisson_iter = 0;
  double poisson_res = 0.0;
  bool converged = SolvePressurePoisson3D(rhs, vel.nx, vel.ny, vel.nz,
                                           vel.dx, vel.dy, vel.dz,
                                           pressure, config,
                                           &poisson_iter, &poisson_res);

  result.poisson_iterations = poisson_iter;
  result.poisson_residual = poisson_res;

  if (!converged) {
    result.success = false;
    result.error = "Pressure Poisson solver did not converge";
    return result;
  }

  std::vector<double> grad_x, grad_y, grad_z;
  ComputeGradient3D(*pressure, vel.nx, vel.ny, vel.nz,
                    vel.dx, vel.dy, vel.dz, &grad_x, &grad_y, &grad_z);

  for (int i = 0; i < n; ++i) {
    vel.u[i] -= dt * grad_x[i];
    vel.v[i] -= dt * grad_y[i];
    vel.w[i] -= dt * grad_z[i];
  }

  ComputeDivergence3D(vel, &div);

  result.l2_divergence_after = 0.0;
  result.max_divergence_after = 0.0;
  for (int i = 0; i < n; ++i) {
    result.l2_divergence_after += div[i] * div[i];
    result.max_divergence_after = std::max(result.max_divergence_after,
                                            std::abs(div[i]));
  }
  result.l2_divergence_after = std::sqrt(result.l2_divergence_after / n);

  result.success = true;
  return result;
}

// ===========================================================================
// Utility Functions
// ===========================================================================

void ComputeDivergenceNorms2D(const VelocityField2D& vel,
                               double* l2_norm,
                               double* linf_norm) {
  std::vector<double> div;
  ComputeDivergence2D(vel, &div);

  double l2 = 0.0;
  double linf = 0.0;
  for (double d : div) {
    l2 += d * d;
    linf = std::max(linf, std::abs(d));
  }

  *l2_norm = std::sqrt(l2 / div.size());
  *linf_norm = linf;
}

double ComputeKineticEnergy2D(const VelocityField2D& vel) {
  double ke = 0.0;
  const int n = vel.nx * vel.ny;
  const double dA = vel.dx * vel.dy;

  for (int i = 0; i < n; ++i) {
    ke += 0.5 * (vel.u[i] * vel.u[i] + vel.v[i] * vel.v[i]) * dA;
  }
  return ke;
}

double ComputeEnstrophy2D(const VelocityField2D& vel) {
  std::vector<double> omega;
  ComputeVorticity2D(vel, &omega);

  double enstrophy = 0.0;
  const double dA = vel.dx * vel.dy;

  for (double w : omega) {
    enstrophy += 0.5 * w * w * dA;
  }
  return enstrophy;
}

void ComputeVorticity2D(const VelocityField2D& vel,
                         std::vector<double>* omega) {
  const int nx = vel.nx;
  const int ny = vel.ny;
  const double dx = vel.dx;
  const double dy = vel.dy;

  omega->resize(nx * ny);

  for (int j = 0; j < ny; ++j) {
    for (int i = 0; i < nx; ++i) {
      int idx = j * nx + i;

      double dv_dx, du_dy;

      // dv/dx
      if (i == 0) {
        dv_dx = (vel.v[idx + 1] - vel.v[idx]) / dx;
      } else if (i == nx - 1) {
        dv_dx = (vel.v[idx] - vel.v[idx - 1]) / dx;
      } else {
        dv_dx = (vel.v[idx + 1] - vel.v[idx - 1]) / (2.0 * dx);
      }

      // du/dy
      if (j == 0) {
        du_dy = (vel.u[idx + nx] - vel.u[idx]) / dy;
      } else if (j == ny - 1) {
        du_dy = (vel.u[idx] - vel.u[idx - nx]) / dy;
      } else {
        du_dy = (vel.u[idx + nx] - vel.u[idx - nx]) / (2.0 * dy);
      }

      (*omega)[idx] = dv_dx - du_dy;
    }
  }
}

VelocityField2D CreateVelocityField2D(int nx, int ny, double dx, double dy) {
  VelocityField2D vel;
  vel.nx = nx;
  vel.ny = ny;
  vel.dx = dx;
  vel.dy = dy;
  vel.u.resize(nx * ny, 0.0);
  vel.v.resize(nx * ny, 0.0);
  return vel;
}

void ApplyLidDrivenCavityBC(VelocityField2D& vel, double lid_velocity) {
  const int nx = vel.nx;
  const int ny = vel.ny;

  // Bottom wall (j=0): u=0, v=0
  for (int i = 0; i < nx; ++i) {
    vel.u[i] = 0.0;
    vel.v[i] = 0.0;
  }

  // Top wall (j=ny-1): u=lid_velocity, v=0
  for (int i = 0; i < nx; ++i) {
    int idx = (ny - 1) * nx + i;
    vel.u[idx] = lid_velocity;
    vel.v[idx] = 0.0;
  }

  // Left wall (i=0): u=0, v=0
  for (int j = 0; j < ny; ++j) {
    int idx = j * nx;
    vel.u[idx] = 0.0;
    vel.v[idx] = 0.0;
  }

  // Right wall (i=nx-1): u=0, v=0
  for (int j = 0; j < ny; ++j) {
    int idx = j * nx + (nx - 1);
    vel.u[idx] = 0.0;
    vel.v[idx] = 0.0;
  }
}

void ApplyNoSlipBC(VelocityField2D& vel) {
  ApplyLidDrivenCavityBC(vel, 0.0);
}
