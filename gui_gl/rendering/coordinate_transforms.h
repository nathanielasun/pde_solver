#ifndef COORDINATE_TRANSFORMS_H
#define COORDINATE_TRANSFORMS_H

#include "GlViewer.h"
#include <cmath>

// Coordinate system transformation utilities
namespace CoordinateTransforms {

// Transform a point from coordinate system space to Cartesian 3D
// Parameters:
//   x, y, z: Input coordinates in the coordinate system
//   view_mode: The coordinate system mode
//   scale: Scaling factor
//   cx, cy, cz: Center offsets for Cartesian systems
//   torus_major, torus_minor: Torus parameters (for toroidal systems)
//   base_radius: Base radius for spherical surface
//   radial_span: Radial span for value-based height
//   t_z: Normalized z value (0-1) for value-based height
//   use_value_height: Whether to use value-based height
//   has_z: Whether the domain has a z dimension
// Output:
//   out_x, out_y, out_z: Transformed Cartesian coordinates
void TransformToCartesian(
    float x, float y, float z_val,
    GlViewer::ViewMode view_mode,
    float scale,
    float cx, float cy, float cz,
    float torus_major, float torus_minor,
    float base_radius, float radial_span, float t_z,
    bool use_value_height, bool has_z,
    float* out_x, float* out_y, float* out_z);

// Transform polar coordinates (r, theta) to Cartesian (x, y)
void PolarToCartesian(float r, float theta, float scale, 
                       float* out_x, float* out_y);

// Transform cylindrical coordinates (r, theta, z) to Cartesian
void CylindricalToCartesian(float r, float theta, float z, float scale,
                            float cz, float* out_x, float* out_y, float* out_z);

// Transform spherical surface coordinates (theta, phi) to Cartesian
void SphericalSurfaceToCartesian(float theta, float phi, float radius,
                                 float* out_x, float* out_y, float* out_z);

// Transform spherical volume coordinates (r, theta, phi) to Cartesian
void SphericalVolumeToCartesian(float r, float theta, float phi, float scale,
                                float* out_x, float* out_y, float* out_z);

// Transform toroidal surface coordinates (theta, phi) to Cartesian
void ToroidalSurfaceToCartesian(float theta, float phi, float r,
                                float R, float* out_x, float* out_y, float* out_z);

// Transform toroidal volume coordinates (r, theta, phi) to Cartesian
void ToroidalVolumeToCartesian(float r, float theta, float phi, float scale,
                               float R, float* out_x, float* out_y, float* out_z);

}  // namespace CoordinateTransforms

#endif  // COORDINATE_TRANSFORMS_H

