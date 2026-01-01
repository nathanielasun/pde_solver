#ifndef COORDINATE_METRICS_H
#define COORDINATE_METRICS_H

#include "pde_types.h"

// Compute metric-corrected derivatives for non-Cartesian coordinate systems.
// For Cartesian, these return the standard finite differences.
// For other systems, they include metric terms (e.g., 1/r factors in polar).

struct MetricDerivatives2D {
  double u_rr = 0.0;  // second derivative in first coordinate
  double u_ss = 0.0;  // second derivative in second coordinate
  double u_r = 0.0;   // first derivative in first coordinate
  double u_s = 0.0;   // first derivative in second coordinate
  double metric_factor = 1.0;  // additional metric factor (e.g., 1/r² for polar theta)
};

// Compute metric-corrected derivatives for 2D coordinate systems.
// r, s are the coordinate values at the grid point (i, j).
// u, u_left, u_right, u_down, u_up are the function values.
// dr, ds are the grid spacings.
MetricDerivatives2D ComputeMetricDerivatives2D(
    CoordinateSystem coord_system,
    double r, double s,  // coordinate values at grid point
    double u, double u_left, double u_right, double u_down, double u_up,
    double dr, double ds);

struct MetricDerivatives3D {
  double u_rr = 0.0;
  double u_ss = 0.0;
  double u_tt = 0.0;
  double u_r = 0.0;
  double u_s = 0.0;
  double u_t = 0.0;
  double metric_factor_r = 1.0;
  double metric_factor_s = 1.0;
  double metric_factor_t = 1.0;
};

// Compute metric-corrected derivatives for 3D coordinate systems.
MetricDerivatives3D ComputeMetricDerivatives3D(
    CoordinateSystem coord_system,
    double r, double s, double t,  // coordinate values at grid point
    double u,
    double u_left, double u_right,
    double u_down, double u_up,
    double u_back, double u_front,
    double dr, double ds, double dt);

// Check if a coordinate value is at a singularity (e.g., r=0 in polar/cylindrical/spherical).
bool IsCoordinateSingularity(CoordinateSystem coord_system, double r, double s = 0.0, double t = 0.0);

// Get the appropriate boundary condition convention for a coordinate system.
// Returns true if the boundary should use a special convention (e.g., periodic for theta in polar).
bool IsPeriodicBoundary(CoordinateSystem coord_system, int boundary_index);

#endif  // COORDINATE_METRICS_H

