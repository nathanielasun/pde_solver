#include "finite_differences.h"

#include <cmath>

double ComputeMixedDerivativeXY(double u_pp, double u_pm, double u_mp, double u_mm,
                                 double dx, double dy) {
  // Centered difference formula for u_xy:
  // u_xy ≈ (u(i+1,j+1) - u(i+1,j-1) - u(i-1,j+1) + u(i-1,j-1)) / (4*dx*dy)
  // where:
  //   u_pp = u(i+1,j+1)  (plus x, plus y)
  //   u_pm = u(i+1,j-1)  (plus x, minus y)
  //   u_mp = u(i-1,j+1)  (minus x, plus y)
  //   u_mm = u(i-1,j-1)  (minus x, minus y)
  const double inv_4dxdy = 1.0 / (4.0 * dx * dy);
  return (u_pp - u_pm - u_mp + u_mm) * inv_4dxdy;
}

double ComputeMixedDerivativeXZ(double u_pp, double u_pm, double u_mp, double u_mm,
                                 double dx, double dz) {
  // Centered difference formula for u_xz:
  // u_xz ≈ (u(i+1,j,k+1) - u(i+1,j,k-1) - u(i-1,j,k+1) + u(i-1,j,k-1)) / (4*dx*dz)
  // where:
  //   u_pp = u(i+1,j,k+1)  (plus x, plus z)
  //   u_pm = u(i+1,j,k-1)  (plus x, minus z)
  //   u_mp = u(i-1,j,k+1)  (minus x, plus z)
  //   u_mm = u(i-1,j,k-1)  (minus x, minus z)
  const double inv_4dxdz = 1.0 / (4.0 * dx * dz);
  return (u_pp - u_pm - u_mp + u_mm) * inv_4dxdz;
}

double ComputeMixedDerivativeYZ(double u_pp, double u_pm, double u_mp, double u_mm,
                                 double dy, double dz) {
  // Centered difference formula for u_yz:
  // u_yz ≈ (u(i,j+1,k+1) - u(i,j+1,k-1) - u(i,j-1,k+1) + u(i,j-1,k-1)) / (4*dy*dz)
  // where:
  //   u_pp = u(i,j+1,k+1)  (plus y, plus z)
  //   u_pm = u(i,j+1,k-1)  (plus y, minus z)
  //   u_mp = u(i,j-1,k+1)  (minus y, plus z)
  //   u_mm = u(i,j-1,k-1)  (minus y, minus z)
  const double inv_4dydz = 1.0 / (4.0 * dy * dz);
  return (u_pp - u_pm - u_mp + u_mm) * inv_4dydz;
}

// Helper function to compute 2D index
namespace {
inline int Index2D(int i, int j, int nx) {
  return i + j * nx;
}
}  // namespace

MixedDerivativeContrib ComputeMixedDerivatives2D(const double* grid, int i, int j,
                                                   int nx, int ny, double dx, double dy,
                                                   double ab) {
  MixedDerivativeContrib contrib;
  
  // Check bounds - need neighbors in both directions
  if (i <= 0 || i >= nx - 1 || j <= 0 || j >= ny - 1) {
    // At boundary, use one-sided differences or zero
    // For now, return zero contribution (could be improved with one-sided stencils)
    return contrib;
  }
  
  // Extract 2x2 stencil values
  const double u_pp = grid[Index2D(i + 1, j + 1, nx)];  // u(i+1,j+1)
  const double u_pm = grid[Index2D(i + 1, j - 1, nx)];  // u(i+1,j-1)
  const double u_mp = grid[Index2D(i - 1, j + 1, nx)];  // u(i-1,j+1)
  const double u_mm = grid[Index2D(i - 1, j - 1, nx)];  // u(i-1,j-1)
  
  // Compute u_xy
  const double u_xy = ComputeMixedDerivativeXY(u_pp, u_pm, u_mp, u_mm, dx, dy);
  
  // Multiply by coefficient
  contrib.xy_contrib = ab * u_xy;
  
  return contrib;
}

// Helper function to compute 3D index
namespace {
inline int Index3D(int i, int j, int k, int nx, int ny) {
  return i + j * nx + k * nx * ny;
}
}  // namespace

MixedDerivativeContrib ComputeMixedDerivatives3D(const double* grid, int i, int j, int k,
                                                 int nx, int ny, int nz,
                                                 double dx, double dy, double dz,
                                                 double ab, double ac, double bc) {
  MixedDerivativeContrib contrib;
  
  // Check bounds - need neighbors in all directions
  if (i <= 0 || i >= nx - 1 || j <= 0 || j >= ny - 1 || k <= 0 || k >= nz - 1) {
    // At boundary, return zero contribution (could be improved)
    return contrib;
  }
  
  // Compute u_xy: need neighbors in x and y directions
  const double u_xy_pp = grid[Index3D(i + 1, j + 1, k, nx, ny)];  // u(i+1,j+1,k)
  const double u_xy_pm = grid[Index3D(i + 1, j - 1, k, nx, ny)];  // u(i+1,j-1,k)
  const double u_xy_mp = grid[Index3D(i - 1, j + 1, k, nx, ny)];  // u(i-1,j+1,k)
  const double u_xy_mm = grid[Index3D(i - 1, j - 1, k, nx, ny)];  // u(i-1,j-1,k)
  const double u_xy = ComputeMixedDerivativeXY(u_xy_pp, u_xy_pm, u_xy_mp, u_xy_mm, dx, dy);
  contrib.xy_contrib = ab * u_xy;
  
  // Compute u_xz: need neighbors in x and z directions
  const double u_xz_pp = grid[Index3D(i + 1, j, k + 1, nx, ny)];  // u(i+1,j,k+1)
  const double u_xz_pm = grid[Index3D(i + 1, j, k - 1, nx, ny)];  // u(i+1,j,k-1)
  const double u_xz_mp = grid[Index3D(i - 1, j, k + 1, nx, ny)];  // u(i-1,j,k+1)
  const double u_xz_mm = grid[Index3D(i - 1, j, k - 1, nx, ny)];  // u(i-1,j,k-1)
  const double u_xz = ComputeMixedDerivativeXZ(u_xz_pp, u_xz_pm, u_xz_mp, u_xz_mm, dx, dz);
  contrib.xz_contrib = ac * u_xz;
  
  // Compute u_yz: need neighbors in y and z directions
  const double u_yz_pp = grid[Index3D(i, j + 1, k + 1, nx, ny)];  // u(i,j+1,k+1)
  const double u_yz_pm = grid[Index3D(i, j + 1, k - 1, nx, ny)];  // u(i,j+1,k-1)
  const double u_yz_mp = grid[Index3D(i, j - 1, k + 1, nx, ny)];  // u(i,j-1,k+1)
  const double u_yz_mm = grid[Index3D(i, j - 1, k - 1, nx, ny)];  // u(i,j-1,k-1)
  const double u_yz = ComputeMixedDerivativeYZ(u_yz_pp, u_yz_pm, u_yz_mp, u_yz_mm, dy, dz);
  contrib.yz_contrib = bc * u_yz;
  
  return contrib;
}

// Third derivative stencils

double ComputeThirdDerivativeX(const double* grid, int i, int nx, double dx) {
  const double dx3 = dx * dx * dx;
  const double inv_2dx3 = 1.0 / (2.0 * dx3);
  
  if (i >= 2 && i < nx - 2) {
    // Centered 5-point stencil: u_xxx ≈ (-u(i+2) + 2*u(i+1) - 2*u(i-1) + u(i-2)) / (2*dx^3)
    return (-grid[i + 2] + 2.0 * grid[i + 1] - 2.0 * grid[i - 1] + grid[i - 2]) * inv_2dx3;
  } else if (i == 0) {
    // Forward difference at left boundary
    // u_xxx ≈ (-u(4) + 4*u(3) - 5*u(2) + 2*u(1)) / dx^3
    if (nx >= 5) {
      return (-grid[4] + 4.0 * grid[3] - 5.0 * grid[2] + 2.0 * grid[1]) / dx3;
    } else {
      // Not enough points, use lower order
      return 0.0;
    }
  } else if (i == 1) {
    // Forward-biased stencil
    if (nx >= 5) {
      return (-grid[3] + 2.0 * grid[2] - 2.0 * grid[0] - grid[4]) * inv_2dx3;
    } else {
      return 0.0;
    }
  } else if (i == nx - 2) {
    // Backward-biased stencil
    if (nx >= 5) {
      return (grid[nx - 1] - 2.0 * grid[nx - 3] + 2.0 * grid[nx - 4] + grid[nx - 5]) * inv_2dx3;
    } else {
      return 0.0;
    }
  } else if (i == nx - 1) {
    // Backward difference at right boundary
    if (nx >= 5) {
      return (-grid[nx - 5] + 4.0 * grid[nx - 4] - 5.0 * grid[nx - 3] + 2.0 * grid[nx - 2]) / dx3;
    } else {
      return 0.0;
    }
  }
  return 0.0;
}

double ComputeThirdDerivativeY(const double* grid, int i, int j, int nx, int ny, double dy) {
  const double dy3 = dy * dy * dy;
  const double inv_2dy3 = 1.0 / (2.0 * dy3);
  
  if (j >= 2 && j < ny - 2) {
    // Centered 5-point stencil
    // For 2D: grid is row-major, so grid[i + j*nx] gives point (i, j)
    const int idx_m2 = Index2D(i, j - 2, nx);
    const int idx_m1 = Index2D(i, j - 1, nx);
    const int idx_p1 = Index2D(i, j + 1, nx);
    const int idx_p2 = Index2D(i, j + 2, nx);
    return (-grid[idx_p2] + 2.0 * grid[idx_p1] - 2.0 * grid[idx_m1] + grid[idx_m2]) * inv_2dy3;
  } else if (j == 0) {
    // Forward difference at bottom boundary
    if (ny >= 5) {
      const int idx0 = Index2D(i, 0, nx);
      const int idx1 = Index2D(i, 1, nx);
      const int idx2 = Index2D(i, 2, nx);
      const int idx3 = Index2D(i, 3, nx);
      const int idx4 = Index2D(i, 4, nx);
      return (-grid[idx4] + 4.0 * grid[idx3] - 5.0 * grid[idx2] + 2.0 * grid[idx1]) / dy3;
    } else {
      return 0.0;
    }
  } else if (j == 1) {
    // Forward-biased stencil
    if (ny >= 5) {
      const int idx0 = Index2D(i, 0, nx);
      const int idx2 = Index2D(i, 2, nx);
      const int idx3 = Index2D(i, 3, nx);
      const int idx4 = Index2D(i, 4, nx);
      return (-grid[idx3] + 2.0 * grid[idx2] - 2.0 * grid[idx0] - grid[idx4]) * inv_2dy3;
    } else {
      return 0.0;
    }
  } else if (j == ny - 2) {
    // Backward-biased stencil
    if (ny >= 5) {
      const int idx_last = Index2D(i, ny - 1, nx);
      const int idx_last2 = Index2D(i, ny - 2, nx);
      const int idx_last3 = Index2D(i, ny - 3, nx);
      const int idx_last4 = Index2D(i, ny - 4, nx);
      const int idx_last5 = Index2D(i, ny - 5, nx);
      return (grid[idx_last] - 2.0 * grid[idx_last3] + 2.0 * grid[idx_last4] + grid[idx_last5]) * inv_2dy3;
    } else {
      return 0.0;
    }
  } else if (j == ny - 1) {
    // Backward difference at top boundary
    if (ny >= 5) {
      const int idx_last = Index2D(i, ny - 1, nx);
      const int idx_last2 = Index2D(i, ny - 2, nx);
      const int idx_last3 = Index2D(i, ny - 3, nx);
      const int idx_last4 = Index2D(i, ny - 4, nx);
      const int idx_last5 = Index2D(i, ny - 5, nx);
      return (-grid[idx_last5] + 4.0 * grid[idx_last4] - 5.0 * grid[idx_last3] + 2.0 * grid[idx_last2]) / dy3;
    } else {
      return 0.0;
    }
  }
  return 0.0;
}

double ComputeThirdDerivativeZ(const double* grid, int i, int j, int k, int nx, int ny, int nz, double dz) {
  const double dz3 = dz * dz * dz;
  const double inv_2dz3 = 1.0 / (2.0 * dz3);
  
  if (k >= 2 && k < nz - 2) {
    // Centered 5-point stencil
    // For 3D: grid is row-major, so grid[i + j*nx + k*nx*ny] gives point (i, j, k)
    const int idx_m2 = Index3D(i, j, k - 2, nx, ny);
    const int idx_m1 = Index3D(i, j, k - 1, nx, ny);
    const int idx_p1 = Index3D(i, j, k + 1, nx, ny);
    const int idx_p2 = Index3D(i, j, k + 2, nx, ny);
    return (-grid[idx_p2] + 2.0 * grid[idx_p1] - 2.0 * grid[idx_m1] + grid[idx_m2]) * inv_2dz3;
  } else if (k == 0) {
    // Forward difference at front boundary
    if (nz >= 5) {
      const int idx0 = Index3D(i, j, 0, nx, ny);
      const int idx1 = Index3D(i, j, 1, nx, ny);
      const int idx2 = Index3D(i, j, 2, nx, ny);
      const int idx3 = Index3D(i, j, 3, nx, ny);
      const int idx4 = Index3D(i, j, 4, nx, ny);
      return (-grid[idx4] + 4.0 * grid[idx3] - 5.0 * grid[idx2] + 2.0 * grid[idx1]) / dz3;
    } else {
      return 0.0;
    }
  } else if (k == 1) {
    // Forward-biased stencil
    if (nz >= 5) {
      const int idx0 = Index3D(i, j, 0, nx, ny);
      const int idx2 = Index3D(i, j, 2, nx, ny);
      const int idx3 = Index3D(i, j, 3, nx, ny);
      const int idx4 = Index3D(i, j, 4, nx, ny);
      return (-grid[idx3] + 2.0 * grid[idx2] - 2.0 * grid[idx0] - grid[idx4]) * inv_2dz3;
    } else {
      return 0.0;
    }
  } else if (k == nz - 2) {
    // Backward-biased stencil
    if (nz >= 5) {
      const int idx_last = Index3D(i, j, nz - 1, nx, ny);
      const int idx_last3 = Index3D(i, j, nz - 3, nx, ny);
      const int idx_last4 = Index3D(i, j, nz - 4, nx, ny);
      const int idx_last5 = Index3D(i, j, nz - 5, nx, ny);
      return (grid[idx_last] - 2.0 * grid[idx_last3] + 2.0 * grid[idx_last4] + grid[idx_last5]) * inv_2dz3;
    } else {
      return 0.0;
    }
  } else if (k == nz - 1) {
    // Backward difference at back boundary
    if (nz >= 5) {
      const int idx_last = Index3D(i, j, nz - 1, nx, ny);
      const int idx_last2 = Index3D(i, j, nz - 2, nx, ny);
      const int idx_last3 = Index3D(i, j, nz - 3, nx, ny);
      const int idx_last4 = Index3D(i, j, nz - 4, nx, ny);
      const int idx_last5 = Index3D(i, j, nz - 5, nx, ny);
      return (-grid[idx_last5] + 4.0 * grid[idx_last4] - 5.0 * grid[idx_last3] + 2.0 * grid[idx_last2]) / dz3;
    } else {
      return 0.0;
    }
  }
  return 0.0;
}

// Fourth derivative stencils

double ComputeFourthDerivativeX(const double* grid, int i, int nx, double dx) {
  const double dx4 = dx * dx * dx * dx;
  const double inv_dx4 = 1.0 / dx4;
  
  if (i >= 2 && i < nx - 2) {
    // Centered 5-point stencil: u_xxxx ≈ (u(i+2) - 4*u(i+1) + 6*u(i) - 4*u(i-1) + u(i-2)) / dx^4
    return (grid[i + 2] - 4.0 * grid[i + 1] + 6.0 * grid[i] - 4.0 * grid[i - 1] + grid[i - 2]) * inv_dx4;
  } else if (i == 0) {
    // Forward difference at left boundary
    if (nx >= 5) {
      // u_xxxx ≈ (u(4) - 4*u(3) + 6*u(2) - 4*u(1) + u(0)) / dx^4
      return (grid[4] - 4.0 * grid[3] + 6.0 * grid[2] - 4.0 * grid[1] + grid[0]) * inv_dx4;
    } else {
      return 0.0;
    }
  } else if (i == 1) {
    // Forward-biased stencil
    if (nx >= 5) {
      return (grid[3] - 4.0 * grid[2] + 6.0 * grid[1] - 4.0 * grid[0] - grid[4]) * inv_dx4;
    } else {
      return 0.0;
    }
  } else if (i == nx - 2) {
    // Backward-biased stencil
    if (nx >= 5) {
      return (grid[nx - 1] - 4.0 * grid[nx - 2] + 6.0 * grid[nx - 3] - 4.0 * grid[nx - 4] - grid[nx - 5]) * inv_dx4;
    } else {
      return 0.0;
    }
  } else if (i == nx - 1) {
    // Backward difference at right boundary
    if (nx >= 5) {
      return (grid[nx - 5] - 4.0 * grid[nx - 4] + 6.0 * grid[nx - 3] - 4.0 * grid[nx - 2] + grid[nx - 1]) * inv_dx4;
    } else {
      return 0.0;
    }
  }
  return 0.0;
}

double ComputeFourthDerivativeY(const double* grid, int i, int j, int nx, int ny, double dy) {
  const double dy4 = dy * dy * dy * dy;
  const double inv_dy4 = 1.0 / dy4;
  
  if (j >= 2 && j < ny - 2) {
    // Centered 5-point stencil
    const int idx_m2 = Index2D(i, j - 2, nx);
    const int idx_m1 = Index2D(i, j - 1, nx);
    const int idx = Index2D(i, j, nx);
    const int idx_p1 = Index2D(i, j + 1, nx);
    const int idx_p2 = Index2D(i, j + 2, nx);
    return (grid[idx_p2] - 4.0 * grid[idx_p1] + 6.0 * grid[idx] - 4.0 * grid[idx_m1] + grid[idx_m2]) * inv_dy4;
  } else if (j == 0) {
    // Forward difference at bottom boundary
    if (ny >= 5) {
      const int idx0 = Index2D(i, 0, nx);
      const int idx1 = Index2D(i, 1, nx);
      const int idx2 = Index2D(i, 2, nx);
      const int idx3 = Index2D(i, 3, nx);
      const int idx4 = Index2D(i, 4, nx);
      return (grid[idx4] - 4.0 * grid[idx3] + 6.0 * grid[idx2] - 4.0 * grid[idx1] + grid[idx0]) * inv_dy4;
    } else {
      return 0.0;
    }
  } else if (j == 1) {
    // Forward-biased stencil
    if (ny >= 5) {
      const int idx0 = Index2D(i, 0, nx);
      const int idx1 = Index2D(i, 1, nx);
      const int idx2 = Index2D(i, 2, nx);
      const int idx3 = Index2D(i, 3, nx);
      const int idx4 = Index2D(i, 4, nx);
      return (grid[idx3] - 4.0 * grid[idx2] + 6.0 * grid[idx1] - 4.0 * grid[idx0] - grid[idx4]) * inv_dy4;
    } else {
      return 0.0;
    }
  } else if (j == ny - 2) {
    // Backward-biased stencil
    if (ny >= 5) {
      const int idx_last = Index2D(i, ny - 1, nx);
      const int idx_last2 = Index2D(i, ny - 2, nx);
      const int idx_last3 = Index2D(i, ny - 3, nx);
      const int idx_last4 = Index2D(i, ny - 4, nx);
      const int idx_last5 = Index2D(i, ny - 5, nx);
      return (grid[idx_last] - 4.0 * grid[idx_last2] + 6.0 * grid[idx_last3] - 4.0 * grid[idx_last4] - grid[idx_last5]) * inv_dy4;
    } else {
      return 0.0;
    }
  } else if (j == ny - 1) {
    // Backward difference at top boundary
    if (ny >= 5) {
      const int idx_last = Index2D(i, ny - 1, nx);
      const int idx_last2 = Index2D(i, ny - 2, nx);
      const int idx_last3 = Index2D(i, ny - 3, nx);
      const int idx_last4 = Index2D(i, ny - 4, nx);
      const int idx_last5 = Index2D(i, ny - 5, nx);
      return (grid[idx_last5] - 4.0 * grid[idx_last4] + 6.0 * grid[idx_last3] - 4.0 * grid[idx_last2] + grid[idx_last]) * inv_dy4;
    } else {
      return 0.0;
    }
  }
  return 0.0;
}

double ComputeFourthDerivativeZ(const double* grid, int i, int j, int k, int nx, int ny, int nz, double dz) {
  const double dz4 = dz * dz * dz * dz;
  const double inv_dz4 = 1.0 / dz4;
  
  if (k >= 2 && k < nz - 2) {
    // Centered 5-point stencil
    const int idx_m2 = Index3D(i, j, k - 2, nx, ny);
    const int idx_m1 = Index3D(i, j, k - 1, nx, ny);
    const int idx = Index3D(i, j, k, nx, ny);
    const int idx_p1 = Index3D(i, j, k + 1, nx, ny);
    const int idx_p2 = Index3D(i, j, k + 2, nx, ny);
    return (grid[idx_p2] - 4.0 * grid[idx_p1] + 6.0 * grid[idx] - 4.0 * grid[idx_m1] + grid[idx_m2]) * inv_dz4;
  } else if (k == 0) {
    // Forward difference at front boundary
    if (nz >= 5) {
      const int idx0 = Index3D(i, j, 0, nx, ny);
      const int idx1 = Index3D(i, j, 1, nx, ny);
      const int idx2 = Index3D(i, j, 2, nx, ny);
      const int idx3 = Index3D(i, j, 3, nx, ny);
      const int idx4 = Index3D(i, j, 4, nx, ny);
      return (grid[idx4] - 4.0 * grid[idx3] + 6.0 * grid[idx2] - 4.0 * grid[idx1] + grid[idx0]) * inv_dz4;
    } else {
      return 0.0;
    }
  } else if (k == 1) {
    // Forward-biased stencil
    if (nz >= 5) {
      const int idx0 = Index3D(i, j, 0, nx, ny);
      const int idx1 = Index3D(i, j, 1, nx, ny);
      const int idx2 = Index3D(i, j, 2, nx, ny);
      const int idx3 = Index3D(i, j, 3, nx, ny);
      const int idx4 = Index3D(i, j, 4, nx, ny);
      return (grid[idx3] - 4.0 * grid[idx2] + 6.0 * grid[idx1] - 4.0 * grid[idx0] - grid[idx4]) * inv_dz4;
    } else {
      return 0.0;
    }
  } else if (k == nz - 2) {
    // Backward-biased stencil
    if (nz >= 5) {
      const int idx_last = Index3D(i, j, nz - 1, nx, ny);
      const int idx_last2 = Index3D(i, j, nz - 2, nx, ny);
      const int idx_last3 = Index3D(i, j, nz - 3, nx, ny);
      const int idx_last4 = Index3D(i, j, nz - 4, nx, ny);
      const int idx_last5 = Index3D(i, j, nz - 5, nx, ny);
      return (grid[idx_last] - 4.0 * grid[idx_last2] + 6.0 * grid[idx_last3] - 4.0 * grid[idx_last4] - grid[idx_last5]) * inv_dz4;
    } else {
      return 0.0;
    }
  } else if (k == nz - 1) {
    // Backward difference at back boundary
    if (nz >= 5) {
      const int idx_last = Index3D(i, j, nz - 1, nx, ny);
      const int idx_last2 = Index3D(i, j, nz - 2, nx, ny);
      const int idx_last3 = Index3D(i, j, nz - 3, nx, ny);
      const int idx_last4 = Index3D(i, j, nz - 4, nx, ny);
      const int idx_last5 = Index3D(i, j, nz - 5, nx, ny);
      return (grid[idx_last5] - 4.0 * grid[idx_last4] + 6.0 * grid[idx_last3] - 4.0 * grid[idx_last2] + grid[idx_last]) * inv_dz4;
    } else {
      return 0.0;
    }
  }
  return 0.0;
}

