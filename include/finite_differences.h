#ifndef FINITE_DIFFERENCES_H
#define FINITE_DIFFERENCES_H

// Finite difference stencils for mixed derivatives
// Provides functions to compute cross-derivatives using centered differences

// Compute u_xy (mixed derivative in x and y) using centered differences
// Requires a 3x3 stencil around point (i,j)
// For 2D grid with indices (i,j) where i is x-direction, j is y-direction
// Returns: (u(i+1,j+1) - u(i+1,j-1) - u(i-1,j+1) + u(i-1,j-1)) / (4*dx*dy)
double ComputeMixedDerivativeXY(double u_pp, double u_pm, double u_mp, double u_mm,
                                 double dx, double dy);

// Compute u_xz (mixed derivative in x and z) using centered differences
// For 3D grid with indices (i,j,k) where i is x, j is y, k is z
// Returns: (u(i+1,j,k+1) - u(i+1,j,k-1) - u(i-1,j,k+1) + u(i-1,j,k-1)) / (4*dx*dz)
double ComputeMixedDerivativeXZ(double u_pp, double u_pm, double u_mp, double u_mm,
                                 double dx, double dz);

// Compute u_yz (mixed derivative in y and z) using centered differences
// For 3D grid with indices (i,j,k) where i is x, j is y, k is z
// Returns: (u(i,j+1,k+1) - u(i,j+1,k-1) - u(i,j-1,k+1) + u(i,j-1,k-1)) / (4*dy*dz)
double ComputeMixedDerivativeYZ(double u_pp, double u_pm, double u_mp, double u_mm,
                                 double dy, double dz);

// Helper function to compute mixed derivative contribution to discretization
// For 2D: computes contribution from u_xy term
// For 3D: computes contribution from u_xy, u_xz, u_yz terms
struct MixedDerivativeContrib {
  double xy_contrib = 0.0;  // contribution from u_xy term
  double xz_contrib = 0.0;  // contribution from u_xz term (3D only)
  double yz_contrib = 0.0;  // contribution from u_yz term (3D only)
};

// Compute mixed derivative contributions for a 2D grid point
// grid: 1D array representing 2D grid (row-major: grid[i + j*nx])
// i, j: grid indices
// nx, ny: grid dimensions
// dx, dy: grid spacing
// ab: coefficient for u_xy term
MixedDerivativeContrib ComputeMixedDerivatives2D(const double* grid, int i, int j,
                                                   int nx, int ny, double dx, double dy,
                                                   double ab);

// Compute mixed derivative contributions for a 3D grid point
// grid: 1D array representing 3D grid (row-major: grid[i + j*nx + k*nx*ny])
// i, j, k: grid indices
// nx, ny, nz: grid dimensions
// dx, dy, dz: grid spacing
// ab: coefficient for u_xy term
// ac: coefficient for u_xz term
// bc: coefficient for u_yz term
MixedDerivativeContrib ComputeMixedDerivatives3D(const double* grid, int i, int j, int k,
                                                 int nx, int ny, int nz,
                                                 double dx, double dy, double dz,
                                                 double ab, double ac, double bc);

// Higher-order derivative stencils

// Compute third derivative u_xxx using 5-point centered difference
// u_xxx ≈ (-u(i+2) + 2*u(i+1) - 2*u(i-1) + u(i-2)) / (2*dx^3)
// Requires grid points at i-2, i-1, i, i+1, i+2
// For boundary points, uses one-sided differences
// grid: 1D array, i: x index, nx: x dimension
double ComputeThirdDerivativeX(const double* grid, int i, int nx, double dx);

// Compute third derivative u_yyy using 5-point centered difference
// grid: 1D array (row-major), i: x index, j: y index, nx, ny: dimensions
double ComputeThirdDerivativeY(const double* grid, int i, int j, int nx, int ny, double dy);

// Compute third derivative u_zzz using 5-point centered difference (3D only)
// grid: 1D array (row-major), i: x index, j: y index, k: z index, nx, ny, nz: dimensions
double ComputeThirdDerivativeZ(const double* grid, int i, int j, int k, int nx, int ny, int nz, double dz);

// Compute fourth derivative u_xxxx using 5-point centered difference
// u_xxxx ≈ (u(i+2) - 4*u(i+1) + 6*u(i) - 4*u(i-1) + u(i-2)) / dx^4
// Requires grid points at i-2, i-1, i, i+1, i+2
// For boundary points, uses one-sided differences
// grid: 1D array, i: x index, nx: x dimension
double ComputeFourthDerivativeX(const double* grid, int i, int nx, double dx);

// Compute fourth derivative u_yyyy using 5-point centered difference
// grid: 1D array (row-major), i: x index, j: y index, nx, ny: dimensions
double ComputeFourthDerivativeY(const double* grid, int i, int j, int nx, int ny, double dy);

// Compute fourth derivative u_zzzz using 5-point centered difference (3D only)
// grid: 1D array (row-major), i: x index, j: y index, k: z index, nx, ny, nz: dimensions
double ComputeFourthDerivativeZ(const double* grid, int i, int j, int k, int nx, int ny, int nz, double dz);

// Center coefficient used by fourth-derivative stencils for index-based updates.
// Returns 0 when grid is too small for the 5-point stencil.
inline double FourthDerivativeCenterCoeff(int index, int n, double h) {
  if (n < 5) {
    return 0.0;
  }
  const double inv_h4 = 1.0 / (h * h * h * h);
  if (index <= 0 || index >= n - 1) {
    return inv_h4;
  }
  if (index == n - 2) {
    return -4.0 * inv_h4;
  }
  return 6.0 * inv_h4;
}

#endif  // FINITE_DIFFERENCES_H
