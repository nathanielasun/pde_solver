#include "nonlinear_solve.h"

#include <algorithm>
#include <cmath>

#include "fv/flux.h"
#include "fv/fv_discretization.h"
#include "safe_math.h"

bool ShouldUseConservationSolver(const SolveInput& input) {
  if (input.discretization == Discretization::FiniteVolume) {
    return true;
  }
  if (input.problem_form == ProblemForm::ConservationLaw) {
    return true;
  }
  return false;
}

SolveOutput SolveConservationLawTimeSeries(const SolveInput& input,
                                           const FrameCallback& on_frame,
                                           const ProgressCallback& progress) {
  const Domain& d = input.domain;
  if (d.nz > 1) {
    return {"FV conservation solver supports 2D (nz=1) only", {}};
  }

  const int nx = d.nx;
  const int ny = d.ny;
  auto grid_check = pde::ValidateGridSize(nx, ny, d.nz);
  if (!grid_check.ok) {
    return {grid_check.error, {}};
  }
  auto grid_spacing = pde::ComputeGridSpacing(d.xmin, d.xmax, nx, d.ymin, d.ymax, ny);
  if (!grid_spacing.ok) {
    return {grid_spacing.error, {}};
  }
  const double dx = grid_spacing.dx;
  const double dy = grid_spacing.dy;

  std::string flux_error;
  std::string flux_latex = input.conservation.flux_latex;
  if (flux_latex.empty()) {
    flux_latex = "0.5*u^2";
  }
  FluxEvaluator flux = FluxEvaluator::Parse(flux_latex, &flux_error);
  if (!flux.ok()) {
    return {"invalid conservation flux: " + flux_error, {}};
  }

  const int frames = std::max(1, input.time.frames);
  const double t_start = input.time.t_start;
  double dt = input.time.dt;
  if (dt <= 0.0) {
    return {"time step dt must be positive", {}};
  }

  const double nu = std::max(std::abs(input.pde.a), std::abs(input.pde.b));

  std::vector<double> grid(static_cast<size_t>(nx * ny), 0.0);
  if (!input.initial_grid.empty()) {
    if (input.initial_grid.size() != grid.size()) {
      return {"initial grid size mismatch", {}};
    }
    grid = input.initial_grid;
  }

  std::vector<double> dudt(grid.size(), 0.0);
  std::vector<double> stage(grid.size(), 0.0);

  auto emit = [&](const std::string& phase, double v) {
    if (progress) {
      progress(phase, v);
    }
  };

  for (int frame = 0; frame < frames; ++frame) {
    if (input.cancel && input.cancel->load()) {
      return {"solve cancelled", {}};
    }
    const double t = t_start + frame * dt;
    emit("time", frames <= 1 ? 1.0 : static_cast<double>(frame) / static_cast<double>(frames - 1));

    if (on_frame && !on_frame(frame, t, grid, nullptr)) {
      break;
    }
    if (frame + 1 >= frames) {
      break;
    }

    ComputeFVSemidiscreteRHS2D(grid, nx, ny, dx, dy, flux, input.conservation, nu, input.bc, &dudt);

    // Explicit Euler step for semidiscrete conservation law
    for (size_t k = 0; k < grid.size(); ++k) {
      grid[k] += dt * dudt[k];
    }
  }

  SolveOutput out;
  out.grid = std::move(grid);
  return out;
}
