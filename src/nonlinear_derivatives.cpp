#include "nonlinear_derivatives.h"

#include <cmath>

namespace {
inline int Idx2D(int i, int j, int nx) { return j * nx + i; }
inline int Idx3D(int i, int j, int k, int nx, int ny) { return (k * ny + j) * nx + i; }
}  // namespace

double CenteredDerivativeX(const std::vector<double>& grid, int i, int j, int nx, double dx) {
  const double inv2dx = 1.0 / (2.0 * dx);
  return (grid[static_cast<size_t>(Idx2D(i + 1, j, nx))] -
          grid[static_cast<size_t>(Idx2D(i - 1, j, nx))]) *
         inv2dx;
}

double CenteredDerivativeY(const std::vector<double>& grid, int i, int j, int nx, int ny, double dy) {
  const double inv2dy = 1.0 / (2.0 * dy);
  return (grid[static_cast<size_t>(Idx2D(i, j + 1, nx))] -
          grid[static_cast<size_t>(Idx2D(i, j - 1, nx))]) *
         inv2dy;
}

double CenteredDerivativeZ(const std::vector<double>& grid, int i, int j, int k, int nx, int ny, int nz,
                           double dz) {
  (void)nz;
  const double inv2dz = 1.0 / (2.0 * dz);
  return (grid[static_cast<size_t>(Idx3D(i, j, k + 1, nx, ny))] -
          grid[static_cast<size_t>(Idx3D(i, j, k - 1, nx, ny))]) *
         inv2dz;
}

double EvalNonlinearDerivative(const NonlinearDerivativeTerm& term,
                               const std::vector<double>& grid,
                               int i, int j, int nx, int ny,
                               double dx, double dy) {
  const double u = grid[static_cast<size_t>(Idx2D(i, j, nx))];
  const double ux = CenteredDerivativeX(grid, i, j, nx, dx);
  const double uy = CenteredDerivativeY(grid, i, j, nx, ny, dy);

  switch (term.kind) {
    case NonlinearDerivativeKind::UUx:
      return term.coeff * u * ux;
    case NonlinearDerivativeKind::UUy:
      return term.coeff * u * uy;
    case NonlinearDerivativeKind::UUz:
      return 0.0;
    case NonlinearDerivativeKind::UxUx:
      return term.coeff * ux * ux;
    case NonlinearDerivativeKind::UyUy:
      return term.coeff * uy * uy;
    case NonlinearDerivativeKind::UzUz:
      return 0.0;
    case NonlinearDerivativeKind::GradSquared:
      return term.coeff * (ux * ux + uy * uy);
    default:
      return 0.0;
  }
}

double EvalNonlinearDerivative3D(const NonlinearDerivativeTerm& term,
                                 const std::vector<double>& grid,
                                 int i, int j, int k, int nx, int ny, int nz,
                                 double dx, double dy, double dz) {
  const double u = grid[static_cast<size_t>(Idx3D(i, j, k, nx, ny))];
  const double ux = CenteredDerivativeX(grid, i, j, nx, dx);
  const double uy = CenteredDerivativeY(grid, i, j, nx, ny, dy);
  const double uz = CenteredDerivativeZ(grid, i, j, k, nx, ny, nz, dz);

  switch (term.kind) {
    case NonlinearDerivativeKind::UUx:
      return term.coeff * u * ux;
    case NonlinearDerivativeKind::UUy:
      return term.coeff * u * uy;
    case NonlinearDerivativeKind::UUz:
      return term.coeff * u * uz;
    case NonlinearDerivativeKind::UxUx:
      return term.coeff * ux * ux;
    case NonlinearDerivativeKind::UyUy:
      return term.coeff * uy * uy;
    case NonlinearDerivativeKind::UzUz:
      return term.coeff * uz * uz;
    case NonlinearDerivativeKind::GradSquared:
      return term.coeff * (ux * ux + uy * uy + uz * uz);
    default:
      return 0.0;
  }
}

double AccumulateNonlinearDerivatives(const std::vector<NonlinearDerivativeTerm>& terms,
                                      const std::vector<double>& grid,
                                      int i, int j, int nx, int ny,
                                      double dx, double dy) {
  double sum = 0.0;
  for (const auto& term : terms) {
    sum += EvalNonlinearDerivative(term, grid, i, j, nx, ny, dx, dy);
  }
  return sum;
}

double AccumulateNonlinearDerivatives3D(const std::vector<NonlinearDerivativeTerm>& terms,
                                        const std::vector<double>& grid,
                                        int i, int j, int k, int nx, int ny, int nz,
                                        double dx, double dy, double dz) {
  double sum = 0.0;
  for (const auto& term : terms) {
    sum += EvalNonlinearDerivative3D(term, grid, i, j, k, nx, ny, nz, dx, dy, dz);
  }
  return sum;
}

double EstimateNonlinearAdvectionSpeed(const std::vector<NonlinearDerivativeTerm>& terms,
                                       const std::vector<double>& grid,
                                       int nx, int ny,
                                       const std::vector<unsigned char>* active) {
  bool has_uux = false;
  for (const auto& term : terms) {
    if (term.kind == NonlinearDerivativeKind::UUx || term.kind == NonlinearDerivativeKind::UUy ||
        term.kind == NonlinearDerivativeKind::UUz) {
      has_uux = true;
      break;
    }
  }
  if (!has_uux) {
    return 0.0;
  }

  double max_speed = 0.0;
  for (int j = 1; j < ny - 1; ++j) {
    for (int i = 1; i < nx - 1; ++i) {
      const size_t idx = static_cast<size_t>(Idx2D(i, j, nx));
      if (active && !active->empty() && (*active)[idx] == 0) {
        continue;
      }
      max_speed = std::max(max_speed, std::abs(grid[idx]));
    }
  }
  return max_speed;
}
