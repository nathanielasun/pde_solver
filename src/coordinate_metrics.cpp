#include "coordinate_metrics.h"

#include <cmath>
#include <limits>

#include "safe_math.h"

namespace {
const double kPi = 3.14159265358979323846;
// Use a slightly larger epsilon for singularity detection to provide a buffer zone
const double kSingularityBuffer = 1e-10;
}  // namespace

MetricDerivatives2D ComputeMetricDerivatives2D(
    CoordinateSystem coord_system,
    double r, double s,
    double u, double u_left, double u_right, double u_down, double u_up,
    double dr, double ds) {
  MetricDerivatives2D result;

  if (coord_system == CoordinateSystem::Cartesian) {
    // Standard Cartesian finite differences
    result.u_rr = (u_left - 2.0 * u + u_right) / (dr * dr);
    result.u_ss = (u_down - 2.0 * u + u_up) / (ds * ds);
    result.u_r = (u_right - u_left) / (2.0 * dr);
    result.u_s = (u_up - u_down) / (2.0 * ds);
    result.metric_factor = 1.0;
    return result;
  }

  if (coord_system == CoordinateSystem::Polar) {
    // Polar coordinates: (r, theta)
    // Laplacian: (1/r) ∂/∂r (r ∂u/∂r) + (1/r²) ∂²u/∂θ²
    // = u_rr + (1/r) u_r + (1/r²) u_θθ

    // Compute standard finite differences first
    result.u_rr = (u_left - 2.0 * u + u_right) / (dr * dr);
    result.u_ss = (u_down - 2.0 * u + u_up) / (ds * ds);
    result.u_r = (u_right - u_left) / (2.0 * dr);
    result.u_s = (u_up - u_down) / (2.0 * ds);

    if (std::abs(r) < kSingularityBuffer) {
      // At r=0 singularity: use L'Hôpital's rule
      // At r=0, the Laplacian simplifies: lim_{r->0} (1/r) ∂/∂r (r ∂u/∂r) = 2*u_rr
      // So we use u_rr + u_rr = 2*u_rr (no 1/r term) and no theta component
      result.u_ss = 0.0;  // No theta component at r=0
      result.u_s = 0.0;
      result.metric_factor = 0.0;  // No metric factor at r=0
      return result;
    }

    // Use safe division for metric factor to avoid potential overflow
    const double r_sq = r * r;
    result.metric_factor = pde::SafeMetricDiv(1.0, r_sq);  // 1/r² for theta component
    // Note: The (1/r) u_r term is handled separately in the PDE assembly
    return result;
  }

  if (coord_system == CoordinateSystem::Axisymmetric) {
    // Axisymmetric coordinates: (r, z)
    // Laplacian: (1/r) ∂/∂r (r ∂u/∂r) + ∂²u/∂z²
    // = u_rr + (1/r) u_r + u_zz

    // Compute standard finite differences
    result.u_rr = (u_left - 2.0 * u + u_right) / (dr * dr);
    result.u_ss = (u_down - 2.0 * u + u_up) / (ds * ds);
    result.u_r = (u_right - u_left) / (2.0 * dr);
    result.u_s = (u_up - u_down) / (2.0 * ds);
    result.metric_factor = 1.0;  // z component has no metric factor

    // Note: At r=0 singularity, the (1/r) u_r term is handled via L'Hôpital
    // in the PDE assembly code, not here
    return result;
  }

  if (coord_system == CoordinateSystem::SphericalSurface) {
    // Spherical surface: (theta, phi)
    // Surface Laplacian on unit sphere: (1/sin²θ) ∂²u/∂φ² + (1/sinθ) ∂/∂θ (sinθ ∂u/∂θ)
    // For now, treat as simple 2D (theta, phi) with metric factors
    // More accurate: u_θθ + (cosθ/sinθ) u_θ + (1/sin²θ) u_φφ

    // Compute standard finite differences
    result.u_rr = (u_left - 2.0 * u + u_right) / (dr * dr);
    result.u_ss = (u_down - 2.0 * u + u_up) / (ds * ds);
    result.u_r = (u_right - u_left) / (2.0 * dr);
    result.u_s = (u_up - u_down) / (2.0 * ds);

    const double sin_theta = std::sin(s);

    if (std::abs(sin_theta) < kSingularityBuffer) {
      // Near poles (theta=0 or pi), the phi direction becomes singular
      // Use simplified treatment: no phi contribution
      result.metric_factor = 0.0;  // Suppress phi component at poles
      return result;
    }

    // Use safe division for metric factor
    const double sin_sq = sin_theta * sin_theta;
    result.metric_factor = pde::SafeMetricDiv(1.0, sin_sq);  // 1/sin²θ for phi component
    // Note: The (cosθ/sinθ) u_θ term is handled separately
    return result;
  }

  if (coord_system == CoordinateSystem::ToroidalSurface) {
    // Toroidal surface: (theta, phi) - similar to spherical surface but on torus
    // For now, treat as simple 2D (no metric terms for toroidal surface)
    result.u_rr = (u_left - 2.0 * u + u_right) / (dr * dr);
    result.u_ss = (u_down - 2.0 * u + u_up) / (ds * ds);
    result.u_r = (u_right - u_left) / (2.0 * dr);
    result.u_s = (u_up - u_down) / (2.0 * ds);
    result.metric_factor = 1.0;
    return result;
  }

  // Default: Cartesian
  result.u_rr = (u_left - 2.0 * u + u_right) / (dr * dr);
  result.u_ss = (u_down - 2.0 * u + u_up) / (ds * ds);
  result.u_r = (u_right - u_left) / (2.0 * dr);
  result.u_s = (u_up - u_down) / (2.0 * ds);
  result.metric_factor = 1.0;
  return result;
}

MetricDerivatives3D ComputeMetricDerivatives3D(
    CoordinateSystem coord_system,
    double r, double s, double t,
    double u,
    double u_left, double u_right,
    double u_down, double u_up,
    double u_back, double u_front,
    double dr, double ds, double dt) {
  MetricDerivatives3D result;

  if (coord_system == CoordinateSystem::Cartesian ||
      coord_system == CoordinateSystem::Cylindrical ||
      coord_system == CoordinateSystem::SphericalVolume ||
      coord_system == CoordinateSystem::ToroidalVolume) {
    // For 3D systems, compute standard finite differences first
    result.u_rr = (u_left - 2.0 * u + u_right) / (dr * dr);
    result.u_ss = (u_down - 2.0 * u + u_up) / (ds * ds);
    result.u_tt = (u_back - 2.0 * u + u_front) / (dt * dt);
    result.u_r = (u_right - u_left) / (2.0 * dr);
    result.u_s = (u_up - u_down) / (2.0 * ds);
    result.u_t = (u_front - u_back) / (2.0 * dt);
    result.metric_factor_r = 1.0;
    result.metric_factor_s = 1.0;
    result.metric_factor_t = 1.0;

    if (coord_system == CoordinateSystem::Cylindrical) {
      // Cylindrical: (r, theta, z)
      // Laplacian: (1/r) ∂/∂r (r ∂u/∂r) + (1/r²) ∂²u/∂θ² + ∂²u/∂z²
      if (std::abs(r) < kSingularityBuffer) {
        // At r=0, theta component vanishes
        result.metric_factor_s = 0.0;
      } else {
        result.metric_factor_s = pde::SafeMetricDiv(1.0, r * r);  // 1/r² for theta
      }
      // Note: (1/r) u_r term handled separately
    } else if (coord_system == CoordinateSystem::SphericalVolume) {
      // Spherical: (r, theta, phi)
      // Laplacian: (1/r²) ∂/∂r (r² ∂u/∂r) + (1/(r² sin²θ)) ∂²u/∂φ² + (1/(r² sinθ)) ∂/∂θ (sinθ ∂u/∂θ)
      const double sin_theta = std::sin(s);
      const double r_sq = r * r;
      const double sin_sq = sin_theta * sin_theta;

      if (std::abs(r) < kSingularityBuffer) {
        // At r=0, all angular components vanish
        result.metric_factor_s = 0.0;
        result.metric_factor_t = 0.0;
      } else if (std::abs(sin_theta) < kSingularityBuffer) {
        // At poles, phi component vanishes
        result.metric_factor_s = pde::SafeMetricDiv(1.0, r_sq);  // 1/r² for theta
        result.metric_factor_t = 0.0;
      } else {
        result.metric_factor_s = pde::SafeMetricDiv(1.0, r_sq);  // 1/r² for theta
        result.metric_factor_t = pde::SafeMetricDiv(1.0, r_sq * sin_sq);  // 1/(r² sin²θ) for phi
      }
      // Note: Additional terms (2/r) u_r and (cosθ/(r² sinθ)) u_θ handled separately
    } else if (coord_system == CoordinateSystem::ToroidalVolume) {
      // Toroidal volume: (r, theta, phi) - complex metric, simplified for now
      if (std::abs(r) < kSingularityBuffer) {
        result.metric_factor_s = 0.0;
        result.metric_factor_t = 0.0;
      } else {
        const double r_sq = r * r;
        result.metric_factor_s = pde::SafeMetricDiv(1.0, r_sq);
        result.metric_factor_t = pde::SafeMetricDiv(1.0, r_sq);
      }
    }

    return result;
  }

  // Default: Cartesian
  result.u_rr = (u_left - 2.0 * u + u_right) / (dr * dr);
  result.u_ss = (u_down - 2.0 * u + u_up) / (ds * ds);
  result.u_tt = (u_back - 2.0 * u + u_front) / (dt * dt);
  result.u_r = (u_right - u_left) / (2.0 * dr);
  result.u_s = (u_up - u_down) / (2.0 * ds);
  result.u_t = (u_front - u_back) / (2.0 * dt);
  result.metric_factor_r = 1.0;
  result.metric_factor_s = 1.0;
  result.metric_factor_t = 1.0;
  return result;
}

bool IsCoordinateSingularity(CoordinateSystem coord_system, double r, double s, double t) {
  (void)t;  // Unused parameter
  if (coord_system == CoordinateSystem::Polar ||
      coord_system == CoordinateSystem::Axisymmetric ||
      coord_system == CoordinateSystem::Cylindrical ||
      coord_system == CoordinateSystem::SphericalVolume ||
      coord_system == CoordinateSystem::ToroidalVolume) {
    return std::abs(r) < kSingularityBuffer;
  }
  if (coord_system == CoordinateSystem::SphericalSurface ||
      coord_system == CoordinateSystem::SphericalVolume) {
    // Poles at theta=0 or pi
    return std::abs(std::sin(s)) < kSingularityBuffer;
  }
  return false;
}

bool IsPeriodicBoundary(CoordinateSystem coord_system, int boundary_index) {
  // boundary_index: 0=left, 1=right, 2=bottom, 3=top, 4=front, 5=back
  if (coord_system == CoordinateSystem::Polar) {
    // theta (y) boundaries are periodic
    return boundary_index == 2 || boundary_index == 3;
  }
  if (coord_system == CoordinateSystem::Cylindrical) {
    // theta (y) boundaries are periodic
    return boundary_index == 2 || boundary_index == 3;
  }
  if (coord_system == CoordinateSystem::SphericalSurface ||
      coord_system == CoordinateSystem::SphericalVolume) {
    // theta (y) boundaries may be periodic, phi (z) boundaries are not
    if (coord_system == CoordinateSystem::SphericalVolume) {
      return boundary_index == 2 || boundary_index == 3;  // theta periodic
    }
    // For surface: both theta and phi may be periodic
    return boundary_index == 0 || boundary_index == 1 || boundary_index == 2 || boundary_index == 3;
  }
  if (coord_system == CoordinateSystem::ToroidalSurface ||
      coord_system == CoordinateSystem::ToroidalVolume) {
    // Both theta and phi are periodic
    return boundary_index == 0 || boundary_index == 1 || boundary_index == 2 || boundary_index == 3;
  }
  return false;
}

