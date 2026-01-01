#include "coordinate_metrics.h"

#include <cmath>

namespace {
const double kPi = 3.14159265358979323846;
const double kEps = 1e-12;
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
    
    if (std::abs(r) < kEps) {
      // At r=0 singularity: use L'Hôpital's rule
      // At r=0, the Laplacian simplifies: lim_{r->0} (1/r) ∂/∂r (r ∂u/∂r) = 2*u_rr
      // So we use u_rr + u_rr = 2*u_rr (no 1/r term) and no theta component
      result.u_rr = (u_left - 2.0 * u + u_right) / (dr * dr);
      result.u_ss = 0.0;  // No theta component at r=0
      result.u_r = (u_right - u_left) / (2.0 * dr);
      result.u_s = 0.0;
      result.metric_factor = 0.0;  // No metric factor at r=0
      return result;
    }

    result.u_rr = (u_left - 2.0 * u + u_right) / (dr * dr);
    result.u_ss = (u_down - 2.0 * u + u_up) / (ds * ds);
    result.u_r = (u_right - u_left) / (2.0 * dr);
    result.u_s = (u_up - u_down) / (2.0 * ds);
    result.metric_factor = 1.0 / (r * r);  // 1/r² for theta component
    // Note: The (1/r) u_r term is handled separately in the PDE assembly
    return result;
  }

  if (coord_system == CoordinateSystem::Axisymmetric) {
    // Axisymmetric coordinates: (r, z)
    // Laplacian: (1/r) ∂/∂r (r ∂u/∂r) + ∂²u/∂z²
    // = u_rr + (1/r) u_r + u_zz
    
    if (std::abs(r) < kEps) {
      // At r=0 singularity: use L'Hôpital's rule
      // At r=0, (1/r) ∂/∂r (r ∂u/∂r) = 2*u_rr
      result.u_rr = (u_left - 2.0 * u + u_right) / (dr * dr);
      result.u_ss = (u_down - 2.0 * u + u_up) / (ds * ds);
      result.u_r = (u_right - u_left) / (2.0 * dr);
      result.u_s = (u_up - u_down) / (2.0 * ds);
      result.metric_factor = 1.0;  // z component has no metric factor
      return result;
    }

    result.u_rr = (u_left - 2.0 * u + u_right) / (dr * dr);
    result.u_ss = (u_down - 2.0 * u + u_up) / (ds * ds);
    result.u_r = (u_right - u_left) / (2.0 * dr);
    result.u_s = (u_up - u_down) / (2.0 * ds);
    result.metric_factor = 1.0;  // z component has no metric factor
    return result;
  }

  if (coord_system == CoordinateSystem::SphericalSurface) {
    // Spherical surface: (theta, phi)
    // Surface Laplacian on unit sphere: (1/sin²θ) ∂²u/∂φ² + (1/sinθ) ∂/∂θ (sinθ ∂u/∂θ)
    // For now, treat as simple 2D (theta, phi) with metric factors
    // More accurate: u_θθ + (cosθ/sinθ) u_θ + (1/sin²θ) u_φφ
    
    if (std::abs(std::sin(s)) < kEps) {
      // Near poles (theta=0 or pi), use special treatment
      result.u_rr = (u_left - 2.0 * u + u_right) / (dr * dr);
      result.u_ss = (u_down - 2.0 * u + u_up) / (ds * ds);
      result.u_r = (u_right - u_left) / (2.0 * dr);
      result.u_s = (u_up - u_down) / (2.0 * ds);
      result.metric_factor = 1.0;  // Simplified near poles
      return result;
    }

    const double sin_theta = std::sin(s);
    const double cos_theta = std::cos(s);
    
    result.u_rr = (u_left - 2.0 * u + u_right) / (dr * dr);
    result.u_ss = (u_down - 2.0 * u + u_up) / (ds * ds);
    result.u_r = (u_right - u_left) / (2.0 * dr);
    result.u_s = (u_up - u_down) / (2.0 * ds);
    result.metric_factor = 1.0 / (sin_theta * sin_theta);  // 1/sin²θ for phi component
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
      if (std::abs(r) < kEps) {
        r = std::max(r, kEps);
      }
      result.metric_factor_s = 1.0 / (r * r);  // 1/r² for theta
      // Note: (1/r) u_r term handled separately
    } else if (coord_system == CoordinateSystem::SphericalVolume) {
      // Spherical: (r, theta, phi)
      // Laplacian: (1/r²) ∂/∂r (r² ∂u/∂r) + (1/(r² sin²θ)) ∂²u/∂φ² + (1/(r² sinθ)) ∂/∂θ (sinθ ∂u/∂θ)
      if (std::abs(r) < kEps) {
        r = std::max(r, kEps);
      }
      const double sin_theta = std::abs(std::sin(s)) < kEps ? kEps : std::sin(s);
      result.metric_factor_s = 1.0 / (r * r);  // 1/r² for theta
      result.metric_factor_t = 1.0 / (r * r * sin_theta * sin_theta);  // 1/(r² sin²θ) for phi
      // Note: Additional terms (2/r) u_r and (cosθ/(r² sinθ)) u_θ handled separately
    } else if (coord_system == CoordinateSystem::ToroidalVolume) {
      // Toroidal volume: (r, theta, phi) - complex metric, simplified for now
      if (std::abs(r) < kEps) {
        r = std::max(r, kEps);
      }
      result.metric_factor_s = 1.0 / (r * r);
      result.metric_factor_t = 1.0 / (r * r);
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
  if (coord_system == CoordinateSystem::Polar ||
      coord_system == CoordinateSystem::Axisymmetric ||
      coord_system == CoordinateSystem::Cylindrical ||
      coord_system == CoordinateSystem::SphericalVolume ||
      coord_system == CoordinateSystem::ToroidalVolume) {
    return std::abs(r) < kEps;
  }
  if (coord_system == CoordinateSystem::SphericalSurface ||
      coord_system == CoordinateSystem::SphericalVolume) {
    // Poles at theta=0 or pi
    return std::abs(std::sin(s)) < kEps;
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

