#include "fv/fv_discretization.h"

#include "fv/hyperbolic_bc.h"
#include "fv/riemann.h"
namespace {
inline int Idx(int i, int j, int nx) { return j * nx + i; }
}  // namespace

void ComputeFVSemidiscreteRHS1D(const std::vector<double>& u,
                                int nx,
                                double dx,
                                const FluxEvaluator& flux,
                                const ConservationLawConfig& config,
                                const std::function<double(int)>& left_bc,
                                const std::function<double(int)>& right_bc,
                                std::vector<double>* dudt) {
  if (!dudt) {
    return;
  }
  dudt->assign(static_cast<size_t>(nx), 0.0);
  for (int i = 0; i < nx; ++i) {
    const double u_left = (i == 0) ? left_bc(0) : u[static_cast<size_t>(i - 1)];
    const double u_c = u[static_cast<size_t>(i)];
    const double u_right = (i == nx - 1) ? right_bc(nx - 1) : u[static_cast<size_t>(i + 1)];
    const double flux_right = RiemannFlux(u_c, u_right, flux, config.riemann);
    const double flux_left = RiemannFlux(u_left, u_c, flux, config.riemann);
    (*dudt)[static_cast<size_t>(i)] = -(flux_right - flux_left) / dx;
  }
}

void ComputeFVSemidiscreteRHS1D(const std::vector<double>& u,
                                int nx,
                                double dx,
                                const FluxEvaluator& flux,
                                const ConservationLawConfig& config,
                                const BoundaryCondition& left_bc,
                                const BoundaryCondition& right_bc,
                                std::vector<double>* dudt) {
  ComputeFVSemidiscreteRHS1D(
      u, nx, dx, flux, config,
      [&](int) -> double {
        return FVGhostCellValue(left_bc, u[0], u[0]);
      },
      [&](int) -> double {
        return FVGhostCellValue(right_bc, u[static_cast<size_t>(nx - 1)],
                                u[static_cast<size_t>(nx - 1)]);
      },
      dudt);
}

void ComputeFVSemidiscreteRHS2D(const std::vector<double>& u,
                                int nx, int ny,
                                double dx, double dy,
                                const FluxEvaluator& flux,
                                const ConservationLawConfig& config,
                                double diffusion_coeff,
                                std::vector<double>* dudt) {
  if (!dudt) {
    return;
  }
  ComputeFVSemidiscreteRHS2D(u, nx, ny, dx, dy, flux, config, diffusion_coeff,
                             BoundarySet{}, dudt);
}

void ComputeFVSemidiscreteRHS2D(const std::vector<double>& u,
                                int nx, int ny,
                                double dx, double dy,
                                const FluxEvaluator& flux,
                                const ConservationLawConfig& config,
                                double diffusion_coeff,
                                const BoundarySet& bc,
                                std::vector<double>* dudt) {
  if (!dudt) {
    return;
  }
  dudt->assign(static_cast<size_t>(nx * ny), 0.0);

  for (int j = 0; j < ny; ++j) {
    for (int i = 0; i < nx; ++i) {
      const size_t idx = static_cast<size_t>(Idx(i, j, nx));
      const double u_c = u[idx];
      const double u_left =
          (i == 0) ? FVGhostCellValue(bc.left, u_c, u_c)
                   : u[static_cast<size_t>(Idx(i - 1, j, nx))];
      const double u_right = (i == nx - 1)
                                 ? FVGhostCellValue(bc.right, u_c, u_c)
                                 : u[static_cast<size_t>(Idx(i + 1, j, nx))];
      const double flux_x_right = RiemannFlux(u_c, u_right, flux, config.riemann);
      const double flux_x_left = RiemannFlux(u_left, u_c, flux, config.riemann);
      double rhs = -(flux_x_right - flux_x_left) / dx;

      if (std::abs(diffusion_coeff) > 1e-14 && i > 0 && i < nx - 1 && j > 0 && j < ny - 1) {
        const double u_xx =
            (u[static_cast<size_t>(Idx(i + 1, j, nx))] - 2.0 * u_c +
             u[static_cast<size_t>(Idx(i - 1, j, nx))]) /
            (dx * dx);
        const double u_yy =
            (u[static_cast<size_t>(Idx(i, j + 1, nx))] - 2.0 * u_c +
             u[static_cast<size_t>(Idx(i, j - 1, nx))]) /
            (dy * dy);
        rhs += diffusion_coeff * (u_xx + u_yy);
      }
      (*dudt)[idx] = rhs;
    }
  }
}
