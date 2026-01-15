#ifndef SAFE_MATH_H
#define SAFE_MATH_H

#include <cmath>
#include <limits>
#include <string>

namespace pde {

// Tolerance constants for numerical operations
constexpr double kSafeDivEps = 1e-12;
constexpr double kSingularityEps = 1e-10;
constexpr int kMaxParserRecursionDepth = 100;
constexpr int kMinGridSize = 2;

// Result type for safe division operations
struct SafeDivResult {
  double value = 0.0;
  bool ok = true;
  std::string error;
};

// Safe division that returns a result struct instead of potentially INF/NaN
inline SafeDivResult SafeDiv(double numerator, double denominator,
                             const char* context = nullptr) {
  SafeDivResult result;
  if (std::abs(denominator) < kSafeDivEps) {
    result.ok = false;
    result.value = 0.0;
    if (context) {
      result.error = std::string("division by near-zero denominator in ") + context;
    } else {
      result.error = "division by near-zero denominator";
    }
    return result;
  }
  result.value = numerator / denominator;
  // Check for overflow/underflow
  if (!std::isfinite(result.value)) {
    result.ok = false;
    result.error = context ? std::string("division overflow in ") + context
                           : "division overflow";
  }
  return result;
}

// Safe division that clamps to a fallback value on failure (for cases where we must continue)
inline double SafeDivClamped(double numerator, double denominator,
                             double fallback = 0.0) {
  if (std::abs(denominator) < kSafeDivEps) {
    return fallback;
  }
  double result = numerator / denominator;
  if (!std::isfinite(result)) {
    return fallback;
  }
  return result;
}

// Safe division specifically for coordinate metric calculations
// Returns 0.0 at singularities (which is correct per L'Hôpital for many metric terms)
inline double SafeMetricDiv(double numerator, double r_squared) {
  if (r_squared < kSingularityEps) {
    return 0.0;  // At singularity, metric contribution is handled specially
  }
  double result = numerator / r_squared;
  if (!std::isfinite(result)) {
    return 0.0;
  }
  return result;
}

// Validate grid dimensions
struct GridValidationResult {
  bool ok = true;
  std::string error;
};

inline GridValidationResult ValidateGridSize(int nx, int ny, int nz = 1) {
  GridValidationResult result;
  if (nx < kMinGridSize) {
    result.ok = false;
    result.error = "nx must be at least " + std::to_string(kMinGridSize) +
                   ", got " + std::to_string(nx);
    return result;
  }
  if (ny < kMinGridSize) {
    result.ok = false;
    result.error = "ny must be at least " + std::to_string(kMinGridSize) +
                   ", got " + std::to_string(ny);
    return result;
  }
  if (nz > 1 && nz < kMinGridSize) {
    result.ok = false;
    result.error = "nz must be at least " + std::to_string(kMinGridSize) +
                   " for 3D grids, got " + std::to_string(nz);
    return result;
  }
  return result;
}

// Validate domain bounds
inline GridValidationResult ValidateDomainBounds(double xmin, double xmax,
                                                  double ymin, double ymax,
                                                  double zmin = 0.0, double zmax = 0.0) {
  GridValidationResult result;
  if (xmax <= xmin) {
    result.ok = false;
    result.error = "xmax must be greater than xmin";
    return result;
  }
  if (ymax <= ymin) {
    result.ok = false;
    result.error = "ymax must be greater than ymin";
    return result;
  }
  if (zmax > zmin && zmax <= zmin) {
    result.ok = false;
    result.error = "zmax must be greater than zmin for 3D domains";
    return result;
  }
  return result;
}

// Compute grid spacing with validation
struct GridSpacingResult {
  double dx = 0.0;
  double dy = 0.0;
  double dz = 0.0;
  bool ok = true;
  std::string error;
};

inline GridSpacingResult ComputeGridSpacing(double xmin, double xmax, int nx,
                                            double ymin, double ymax, int ny,
                                            double zmin = 0.0, double zmax = 0.0, int nz = 1) {
  GridSpacingResult result;

  // Validate grid sizes first
  auto grid_check = ValidateGridSize(nx, ny, nz);
  if (!grid_check.ok) {
    result.ok = false;
    result.error = grid_check.error;
    return result;
  }

  // Validate domain bounds
  auto domain_check = ValidateDomainBounds(xmin, xmax, ymin, ymax, zmin, zmax);
  if (!domain_check.ok) {
    result.ok = false;
    result.error = domain_check.error;
    return result;
  }

  // Compute spacing (nx-1 because we have nx points creating nx-1 intervals)
  result.dx = (xmax - xmin) / static_cast<double>(nx - 1);
  result.dy = (ymax - ymin) / static_cast<double>(ny - 1);
  if (nz > 1) {
    result.dz = (zmax - zmin) / static_cast<double>(nz - 1);
  }

  // Final sanity check
  if (!std::isfinite(result.dx) || !std::isfinite(result.dy) ||
      (nz > 1 && !std::isfinite(result.dz))) {
    result.ok = false;
    result.error = "computed grid spacing is not finite";
  }

  return result;
}

// CFL condition checker for time stepping
struct CFLResult {
  bool stable = true;
  double max_dt = std::numeric_limits<double>::max();
  std::string warning;
};

inline CFLResult CheckCFL(double dt, double dx, double dy, double dz,
                          double diffusion_coeff, double advection_coeff = 0.0,
                          double safety_factor = 0.5) {
  CFLResult result;

  // Diffusion stability: dt < dx^2 / (2 * D * dim)
  double min_dx = std::min(dx, dy);
  if (dz > 0.0) {
    min_dx = std::min(min_dx, dz);
  }
  int dim = (dz > 0.0) ? 3 : 2;

  if (std::abs(diffusion_coeff) > kSafeDivEps) {
    double dt_diffusion = safety_factor * min_dx * min_dx / (2.0 * std::abs(diffusion_coeff) * dim);
    if (dt > dt_diffusion) {
      result.stable = false;
      result.warning = "dt exceeds diffusion stability limit (dt_max=" +
                       std::to_string(dt_diffusion) + ")";
    }
    result.max_dt = std::min(result.max_dt, dt_diffusion);
  }

  // Advection stability (CFL): dt < dx / |v|
  if (std::abs(advection_coeff) > kSafeDivEps) {
    double dt_advection = safety_factor * min_dx / std::abs(advection_coeff);
    if (dt > dt_advection) {
      result.stable = false;
      if (!result.warning.empty()) result.warning += "; ";
      result.warning += "dt exceeds advection CFL limit (dt_max=" +
                        std::to_string(dt_advection) + ")";
    }
    result.max_dt = std::min(result.max_dt, dt_advection);
  }

  return result;
}

// Check if a value is safe for use in numerical computations
inline bool IsNumericallyValid(double value) {
  return std::isfinite(value) && !std::isnan(value);
}

// Clamp a value to a safe range
inline double ClampToSafeRange(double value, double min_val, double max_val) {
  if (!std::isfinite(value)) {
    return (min_val + max_val) / 2.0;  // Return midpoint if invalid
  }
  return std::max(min_val, std::min(max_val, value));
}

}  // namespace pde

#endif  // SAFE_MATH_H
