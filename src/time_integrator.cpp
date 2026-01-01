#include "time_integrator.h"

#include <algorithm>
#include <cmath>

TimeIntegrator ParseTimeIntegrator(const std::string& name) {
  if (name == "euler" || name == "forward-euler" || name == "fe") {
    return TimeIntegrator::ForwardEuler;
  }
  if (name == "rk2" || name == "heun") {
    return TimeIntegrator::RK2;
  }
  if (name == "rk4" || name == "classical-rk") {
    return TimeIntegrator::RK4;
  }
  if (name == "ssprk2" || name == "tvd2") {
    return TimeIntegrator::SSPRK2;
  }
  if (name == "ssprk3" || name == "tvd3" || name == "shu-osher") {
    return TimeIntegrator::SSPRK3;
  }
  if (name == "backward-euler" || name == "be" || name == "beuler" ||
      name == "implicit-euler") {
    return TimeIntegrator::BackwardEuler;
  }
  if (name == "crank-nicolson" || name == "cn" || name == "trapezoidal") {
    return TimeIntegrator::CrankNicolson;
  }
  if (name == "imex") {
    return TimeIntegrator::IMEX;
  }
  return TimeIntegrator::ForwardEuler;  // Default
}

std::string TimeIntegratorToString(TimeIntegrator method) {
  switch (method) {
    case TimeIntegrator::ForwardEuler: return "forward-euler";
    case TimeIntegrator::RK2: return "rk2";
    case TimeIntegrator::RK4: return "rk4";
    case TimeIntegrator::SSPRK2: return "ssprk2";
    case TimeIntegrator::SSPRK3: return "ssprk3";
    case TimeIntegrator::BackwardEuler: return "backward-euler";
    case TimeIntegrator::CrankNicolson: return "crank-nicolson";
    case TimeIntegrator::IMEX: return "imex";
    default: return "unknown";
  }
}

TimeStepResult ForwardEulerStep(std::vector<double>& u, double t, double dt,
                                 const RHSFunction& rhs) {
  TimeStepResult result;
  result.dt_used = dt;
  result.rhs_evals = 1;

  const size_t n = u.size();
  std::vector<double> dudt(n);

  // Compute RHS
  rhs(t, u, &dudt);

  // Update: u = u + dt * dudt
  for (size_t i = 0; i < n; ++i) {
    u[i] += dt * dudt[i];
  }

  result.success = true;
  result.dt_next = dt;
  return result;
}

TimeStepResult RK2Step(std::vector<double>& u, double t, double dt,
                        const RHSFunction& rhs) {
  TimeStepResult result;
  result.dt_used = dt;
  result.rhs_evals = 2;

  const size_t n = u.size();
  std::vector<double> k1(n), k2(n), u_temp(n);

  // k1 = F(t, u)
  rhs(t, u, &k1);

  // u_temp = u + dt * k1
  for (size_t i = 0; i < n; ++i) {
    u_temp[i] = u[i] + dt * k1[i];
  }

  // k2 = F(t + dt, u_temp)
  rhs(t + dt, u_temp, &k2);

  // u = u + dt/2 * (k1 + k2)
  for (size_t i = 0; i < n; ++i) {
    u[i] += 0.5 * dt * (k1[i] + k2[i]);
  }

  result.success = true;
  result.dt_next = dt;
  return result;
}

TimeStepResult RK4Step(std::vector<double>& u, double t, double dt,
                        const RHSFunction& rhs) {
  TimeStepResult result;
  result.dt_used = dt;
  result.rhs_evals = 4;

  const size_t n = u.size();
  std::vector<double> k1(n), k2(n), k3(n), k4(n), u_temp(n);

  // k1 = F(t, u)
  rhs(t, u, &k1);

  // u_temp = u + dt/2 * k1
  for (size_t i = 0; i < n; ++i) {
    u_temp[i] = u[i] + 0.5 * dt * k1[i];
  }

  // k2 = F(t + dt/2, u_temp)
  rhs(t + 0.5 * dt, u_temp, &k2);

  // u_temp = u + dt/2 * k2
  for (size_t i = 0; i < n; ++i) {
    u_temp[i] = u[i] + 0.5 * dt * k2[i];
  }

  // k3 = F(t + dt/2, u_temp)
  rhs(t + 0.5 * dt, u_temp, &k3);

  // u_temp = u + dt * k3
  for (size_t i = 0; i < n; ++i) {
    u_temp[i] = u[i] + dt * k3[i];
  }

  // k4 = F(t + dt, u_temp)
  rhs(t + dt, u_temp, &k4);

  // u = u + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
  for (size_t i = 0; i < n; ++i) {
    u[i] += dt / 6.0 * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]);
  }

  result.success = true;
  result.dt_next = dt;
  return result;
}

TimeStepResult SSPRK2Step(std::vector<double>& u, double t, double dt,
                           const RHSFunction& rhs) {
  // SSPRK2 (TVD-preserving second-order):
  // u^(1) = u_n + dt * F(t_n, u_n)
  // u_{n+1} = 1/2 * u_n + 1/2 * (u^(1) + dt * F(t_n + dt, u^(1)))

  TimeStepResult result;
  result.dt_used = dt;
  result.rhs_evals = 2;

  const size_t n = u.size();
  std::vector<double> u_n(u);  // Save initial state
  std::vector<double> k(n);

  // Stage 1: u^(1) = u_n + dt * F(t_n, u_n)
  rhs(t, u, &k);
  for (size_t i = 0; i < n; ++i) {
    u[i] = u_n[i] + dt * k[i];
  }

  // Stage 2: compute F(t + dt, u^(1))
  rhs(t + dt, u, &k);

  // u_{n+1} = 1/2 * u_n + 1/2 * (u^(1) + dt * F(t + dt, u^(1)))
  for (size_t i = 0; i < n; ++i) {
    u[i] = 0.5 * u_n[i] + 0.5 * (u[i] + dt * k[i]);
  }

  result.success = true;
  result.dt_next = dt;
  return result;
}

TimeStepResult SSPRK3Step(std::vector<double>& u, double t, double dt,
                           const RHSFunction& rhs) {
  // SSPRK3 (Shu-Osher, optimal 3rd-order SSP):
  // u^(1) = u_n + dt * F(t_n, u_n)
  // u^(2) = 3/4*u_n + 1/4*(u^(1) + dt * F(t_n + dt, u^(1)))
  // u_{n+1} = 1/3*u_n + 2/3*(u^(2) + dt * F(t_n + dt/2, u^(2)))

  TimeStepResult result;
  result.dt_used = dt;
  result.rhs_evals = 3;

  const size_t n = u.size();
  std::vector<double> u_n(u);  // Save initial state
  std::vector<double> u1(n), u2(n), k(n);

  // Stage 1: u^(1) = u_n + dt * F(t_n, u_n)
  rhs(t, u_n, &k);
  for (size_t i = 0; i < n; ++i) {
    u1[i] = u_n[i] + dt * k[i];
  }

  // Stage 2: u^(2) = 3/4*u_n + 1/4*(u^(1) + dt * F(t + dt, u^(1)))
  rhs(t + dt, u1, &k);
  for (size_t i = 0; i < n; ++i) {
    u2[i] = 0.75 * u_n[i] + 0.25 * (u1[i] + dt * k[i]);
  }

  // Stage 3: u_{n+1} = 1/3*u_n + 2/3*(u^(2) + dt * F(t + dt/2, u^(2)))
  rhs(t + 0.5 * dt, u2, &k);
  for (size_t i = 0; i < n; ++i) {
    u[i] = (1.0 / 3.0) * u_n[i] + (2.0 / 3.0) * (u2[i] + dt * k[i]);
  }

  result.success = true;
  result.dt_next = dt;
  return result;
}

TimeStepResult BackwardEulerStep(std::vector<double>& u, double t, double dt,
                                  const RHSFunction& rhs,
                                  int max_iter, double tol) {
  // Backward Euler: u_{n+1} = u_n + dt * F(t_{n+1}, u_{n+1})
  // Solve using fixed-point iteration

  TimeStepResult result;
  result.dt_used = dt;
  result.rhs_evals = 0;

  const size_t n = u.size();
  std::vector<double> u_n(u);  // Save initial state
  std::vector<double> k(n);

  // Fixed-point iteration
  for (int iter = 0; iter < max_iter; ++iter) {
    rhs(t + dt, u, &k);
    result.rhs_evals++;

    double max_change = 0.0;
    for (size_t i = 0; i < n; ++i) {
      double u_new = u_n[i] + dt * k[i];
      max_change = std::max(max_change, std::abs(u_new - u[i]));
      u[i] = u_new;
    }

    if (max_change < tol) {
      result.success = true;
      result.dt_next = dt;
      return result;
    }
  }

  result.success = false;
  result.error = "backward Euler did not converge";
  result.dt_next = dt * 0.5;  // Suggest smaller dt
  return result;
}

TimeStepResult CrankNicolsonStep(std::vector<double>& u, double t, double dt,
                                  const RHSFunction& rhs,
                                  int max_iter, double tol) {
  // Crank-Nicolson: u_{n+1} = u_n + dt/2 * (F(t_n, u_n) + F(t_{n+1}, u_{n+1}))
  // Solve using fixed-point iteration

  TimeStepResult result;
  result.dt_used = dt;
  result.rhs_evals = 1;

  const size_t n = u.size();
  std::vector<double> u_n(u);
  std::vector<double> k_n(n), k(n);

  // Compute F(t_n, u_n) once
  rhs(t, u_n, &k_n);

  // Fixed-point iteration for implicit part
  for (int iter = 0; iter < max_iter; ++iter) {
    rhs(t + dt, u, &k);
    result.rhs_evals++;

    double max_change = 0.0;
    for (size_t i = 0; i < n; ++i) {
      double u_new = u_n[i] + 0.5 * dt * (k_n[i] + k[i]);
      max_change = std::max(max_change, std::abs(u_new - u[i]));
      u[i] = u_new;
    }

    if (max_change < tol) {
      result.success = true;
      result.dt_next = dt;
      return result;
    }
  }

  result.success = false;
  result.error = "Crank-Nicolson did not converge";
  result.dt_next = dt * 0.5;
  return result;
}

TimeStepResult TimeStep(std::vector<double>& u, double t, double dt,
                        const RHSFunction& rhs,
                        const TimeIntegratorConfig& config) {
  switch (config.method) {
    case TimeIntegrator::ForwardEuler:
      return ForwardEulerStep(u, t, dt, rhs);
    case TimeIntegrator::RK2:
      return RK2Step(u, t, dt, rhs);
    case TimeIntegrator::RK4:
      return RK4Step(u, t, dt, rhs);
    case TimeIntegrator::SSPRK2:
      return SSPRK2Step(u, t, dt, rhs);
    case TimeIntegrator::SSPRK3:
      return SSPRK3Step(u, t, dt, rhs);
    case TimeIntegrator::BackwardEuler:
      return BackwardEulerStep(u, t, dt, rhs,
                                config.implicit_max_iter, config.implicit_tol);
    case TimeIntegrator::CrankNicolson:
      return CrankNicolsonStep(u, t, dt, rhs,
                                config.implicit_max_iter, config.implicit_tol);
    case TimeIntegrator::IMEX:
      // IMEX requires splitting - fall back to CN for now
      return CrankNicolsonStep(u, t, dt, rhs,
                                config.implicit_max_iter, config.implicit_tol);
    default:
      return ForwardEulerStep(u, t, dt, rhs);
  }
}

double ComputeCFLAdvectionDiffusion(double vx, double vy, double vz,
                                     double Dx, double Dy, double Dz,
                                     double dx, double dy, double dz,
                                     double dt) {
  double cfl = 0.0;

  // Advection CFL: |v| * dt / dx
  if (dx > 0.0) cfl = std::max(cfl, std::abs(vx) * dt / dx);
  if (dy > 0.0) cfl = std::max(cfl, std::abs(vy) * dt / dy);
  if (dz > 0.0) cfl = std::max(cfl, std::abs(vz) * dt / dz);

  // Diffusion Fourier number: D * dt / dx^2
  if (dx > 0.0) cfl = std::max(cfl, Dx * dt / (dx * dx));
  if (dy > 0.0) cfl = std::max(cfl, Dy * dt / (dy * dy));
  if (dz > 0.0) cfl = std::max(cfl, Dz * dt / (dz * dz));

  return cfl;
}

double SuggestStableDt(double vx, double vy, double vz,
                       double Dx, double Dy, double Dz,
                       double dx, double dy, double dz,
                       TimeIntegrator method, double cfl_target) {
  // For explicit methods, need CFL <= 1 (advection) and Fourier <= 0.5 (diffusion)
  // Use more restrictive constraint for higher-order methods

  double dt_advection = 1e30;
  double dt_diffusion = 1e30;

  // Advection constraint: CFL = |v| * dt / dx <= cfl_target
  if (dx > 0.0 && std::abs(vx) > 1e-12) {
    dt_advection = std::min(dt_advection, cfl_target * dx / std::abs(vx));
  }
  if (dy > 0.0 && std::abs(vy) > 1e-12) {
    dt_advection = std::min(dt_advection, cfl_target * dy / std::abs(vy));
  }
  if (dz > 0.0 && std::abs(vz) > 1e-12) {
    dt_advection = std::min(dt_advection, cfl_target * dz / std::abs(vz));
  }

  // Diffusion constraint: Fourier = D * dt / dx^2 <= 0.5
  double fourier_target = 0.5;
  if (!IsExplicit(method)) {
    // Implicit methods can use larger dt
    fourier_target = 10.0;
  }

  if (dx > 0.0 && Dx > 1e-12) {
    dt_diffusion = std::min(dt_diffusion, fourier_target * dx * dx / Dx);
  }
  if (dy > 0.0 && Dy > 1e-12) {
    dt_diffusion = std::min(dt_diffusion, fourier_target * dy * dy / Dy);
  }
  if (dz > 0.0 && Dz > 1e-12) {
    dt_diffusion = std::min(dt_diffusion, fourier_target * dz * dz / Dz);
  }

  // Combined constraint for advection-diffusion
  double dt = std::min(dt_advection, dt_diffusion);

  // Safety factor
  dt *= 0.9;

  return dt;
}

TimeStepResult AdaptiveTimeStep(std::vector<double>& u, double t, double dt,
                                 const RHSFunction& rhs,
                                 const TimeIntegratorConfig& config) {
  // Use embedded RK pair for error estimation
  // RK2(3): Use RK2 result and compare with Heun (RK2) with smaller step

  TimeStepResult result;
  const size_t n = u.size();

  // Compute both full step and two half steps
  std::vector<double> u_full(u);
  std::vector<double> u_half(u);

  // Full step with RK2
  result = RK2Step(u_full, t, dt, rhs);

  // Two half steps with RK2
  RK2Step(u_half, t, dt / 2.0, rhs);
  RK2Step(u_half, t + dt / 2.0, dt / 2.0, rhs);

  result.rhs_evals = 6;

  // Error estimate: ||u_full - u_half||
  double error = 0.0;
  double norm = 0.0;
  for (size_t i = 0; i < n; ++i) {
    error += (u_full[i] - u_half[i]) * (u_full[i] - u_half[i]);
    norm += u_half[i] * u_half[i];
  }
  error = std::sqrt(error);
  norm = std::sqrt(norm);

  // Relative error
  result.error_estimate = error / std::max(1e-12, norm);

  // Decide whether to accept step
  if (result.error_estimate < config.error_tol) {
    // Accept step - use more accurate half-step result
    u = std::move(u_half);
    result.success = true;
    result.dt_used = dt;

    // Suggest larger dt for next step
    double factor = std::pow(config.error_tol / std::max(1e-12, result.error_estimate), 0.5);
    factor = std::min(factor, config.dt_grow_factor);
    result.dt_next = std::min(dt * factor, config.dt_max);
  } else {
    // Reject step - restore original and suggest smaller dt
    result.success = false;
    result.error = "adaptive step rejected due to error";
    result.dt_used = 0.0;

    double factor = std::pow(config.error_tol / result.error_estimate, 0.5);
    factor = std::max(factor, config.dt_shrink_factor);
    result.dt_next = std::max(dt * factor, config.dt_min);
  }

  return result;
}

// ===========================================================================
// IMEX (Implicit-Explicit) Methods for Stiff Systems
// ===========================================================================

TimeStepResult IMEXEulerStep(std::vector<double>& u, double t, double dt,
                              const RHSFunction& rhs_explicit,
                              const RHSFunction& rhs_implicit,
                              int max_iter, double tol) {
  TimeStepResult result;
  result.dt_used = dt;
  result.rhs_evals = 1;

  const size_t n = u.size();
  std::vector<double> f_explicit(n), f_implicit(n);

  // Step 1: Explicit forward Euler for non-stiff part
  rhs_explicit(t, u, &f_explicit);
  std::vector<double> u_star(n);
  for (size_t i = 0; i < n; ++i) {
    u_star[i] = u[i] + dt * f_explicit[i];
  }

  // Step 2: Implicit backward Euler for stiff part
  // Solve: u_{n+1} = u_star + dt * F_implicit(u_{n+1})
  std::vector<double> u_new = u_star;
  std::vector<double> u_old(n);

  for (int iter = 0; iter < max_iter; ++iter) {
    u_old = u_new;
    rhs_implicit(t + dt, u_new, &f_implicit);
    result.rhs_evals++;

    double residual = 0.0;
    for (size_t i = 0; i < n; ++i) {
      u_new[i] = u_star[i] + dt * f_implicit[i];
      double diff = u_new[i] - u_old[i];
      residual += diff * diff;
    }
    residual = std::sqrt(residual);

    if (residual < tol) {
      u = std::move(u_new);
      result.success = true;
      result.dt_next = dt;
      return result;
    }
  }

  // Failed to converge
  result.success = false;
  result.error = "IMEX-Euler implicit solve failed to converge";
  return result;
}

TimeStepResult IMEXSSP2Step(std::vector<double>& u, double t, double dt,
                             const RHSFunction& rhs_explicit,
                             const RHSFunction& rhs_implicit,
                             int max_iter, double tol) {
  // IMEX-SSP2(2,2,2) scheme (Pareschi-Russo 2005)
  // Explicit tableau (SSPRK2):
  //   0 | 0   0
  //   1 | 1   0
  //  ---|------
  //     |1/2 1/2
  //
  // Implicit tableau (L-stable SDIRK):
  //   γ | γ   0
  // 1-γ | 1-2γ γ
  //  ---|--------
  //     | 1/2  1/2
  //
  // where γ = 1 - 1/sqrt(2) ≈ 0.2929

  TimeStepResult result;
  result.dt_used = dt;
  result.rhs_evals = 0;

  const size_t n = u.size();
  const double gamma = 1.0 - 1.0 / std::sqrt(2.0);

  std::vector<double> f_exp(n), f_imp(n);
  std::vector<double> u1(n), u2(n), u_new(n), u_old(n);

  // Stage 1: u1 = u + γ*dt*F_imp(u1)
  // Fixed-point iteration starting from u
  u1 = u;
  for (int iter = 0; iter < max_iter; ++iter) {
    u_old = u1;
    rhs_implicit(t + gamma * dt, u1, &f_imp);
    result.rhs_evals++;

    double residual = 0.0;
    for (size_t i = 0; i < n; ++i) {
      u1[i] = u[i] + gamma * dt * f_imp[i];
      double diff = u1[i] - u_old[i];
      residual += diff * diff;
    }

    if (std::sqrt(residual) < tol) break;
    if (iter == max_iter - 1) {
      result.success = false;
      result.error = "IMEX-SSP2 stage 1 failed to converge";
      return result;
    }
  }

  // Get explicit contribution at stage 1
  rhs_explicit(t, u, &f_exp);
  result.rhs_evals++;

  // Stage 2: u2 = u + dt*F_exp(u1) + (1-2γ)*dt*F_imp(u1) + γ*dt*F_imp(u2)
  rhs_explicit(t + dt, u1, &f_exp);
  result.rhs_evals++;
  rhs_implicit(t + gamma * dt, u1, &f_imp);
  result.rhs_evals++;

  std::vector<double> f_imp_u1 = f_imp;

  // Compute intermediate: u_star = u + dt*F_exp(u1) + (1-2γ)*dt*F_imp(u1)
  std::vector<double> u_star(n);
  for (size_t i = 0; i < n; ++i) {
    u_star[i] = u[i] + dt * f_exp[i] + (1.0 - 2.0 * gamma) * dt * f_imp_u1[i];
  }

  // Solve: u2 = u_star + γ*dt*F_imp(u2)
  u2 = u_star;
  for (int iter = 0; iter < max_iter; ++iter) {
    u_old = u2;
    rhs_implicit(t + (1.0 - gamma) * dt, u2, &f_imp);
    result.rhs_evals++;

    double residual = 0.0;
    for (size_t i = 0; i < n; ++i) {
      u2[i] = u_star[i] + gamma * dt * f_imp[i];
      double diff = u2[i] - u_old[i];
      residual += diff * diff;
    }

    if (std::sqrt(residual) < tol) break;
    if (iter == max_iter - 1) {
      result.success = false;
      result.error = "IMEX-SSP2 stage 2 failed to converge";
      return result;
    }
  }

  // Final combination: u_{n+1} = (u1 + u2) / 2
  for (size_t i = 0; i < n; ++i) {
    u[i] = 0.5 * (u1[i] + u2[i]);
  }

  result.success = true;
  result.dt_next = dt;
  return result;
}

TimeStepResult OperatorSplitStep(std::vector<double>& u, double t, double dt,
                                  const OperatorSplitConfig& config) {
  TimeStepResult result;
  result.dt_used = dt;
  result.rhs_evals = 0;

  TimeIntegratorConfig diff_config;
  diff_config.method = config.diffusion_method;

  // Create a wrapper config for implicit reaction solve
  TimeIntegratorConfig react_config;
  react_config.method = config.reaction_method;
  react_config.implicit_max_iter = config.implicit_max_iter;
  react_config.implicit_tol = config.implicit_tol;

  if (config.use_strang) {
    // Strang splitting: D(dt/2) -> R(dt) -> D(dt/2)

    // Step 1: Diffusion for dt/2
    auto step1 = TimeStep(u, t, dt / 2.0, config.rhs_diffusion, diff_config);
    if (!step1.success) {
      result.success = false;
      result.error = "Strang split: diffusion step 1 failed";
      return result;
    }
    result.rhs_evals += step1.rhs_evals;

    // Step 2: Reaction for dt (implicit)
    auto step2 = TimeStep(u, t + dt / 2.0, dt, config.rhs_reaction, react_config);
    if (!step2.success) {
      result.success = false;
      result.error = "Strang split: reaction step failed";
      return result;
    }
    result.rhs_evals += step2.rhs_evals;

    // Step 3: Diffusion for dt/2
    auto step3 = TimeStep(u, t + dt / 2.0, dt / 2.0, config.rhs_diffusion, diff_config);
    if (!step3.success) {
      result.success = false;
      result.error = "Strang split: diffusion step 2 failed";
      return result;
    }
    result.rhs_evals += step3.rhs_evals;

  } else {
    // Lie-Trotter splitting: D(dt) -> R(dt) (first-order)

    // Step 1: Diffusion for dt
    auto step1 = TimeStep(u, t, dt, config.rhs_diffusion, diff_config);
    if (!step1.success) {
      result.success = false;
      result.error = "Lie-Trotter split: diffusion step failed";
      return result;
    }
    result.rhs_evals += step1.rhs_evals;

    // Step 2: Reaction for dt (implicit)
    auto step2 = TimeStep(u, t, dt, config.rhs_reaction, react_config);
    if (!step2.success) {
      result.success = false;
      result.error = "Lie-Trotter split: reaction step failed";
      return result;
    }
    result.rhs_evals += step2.rhs_evals;
  }

  result.success = true;
  result.dt_next = dt;
  return result;
}
