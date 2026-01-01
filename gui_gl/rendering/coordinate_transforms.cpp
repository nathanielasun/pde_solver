#include "coordinate_transforms.h"
#include <cmath>

namespace CoordinateTransforms {

void PolarToCartesian(float r, float theta, float scale, 
                       float* out_x, float* out_y) {
  *out_x = std::cos(theta) * r * scale;
  *out_y = std::sin(theta) * r * scale;
}

void CylindricalToCartesian(float r, float theta, float z, float scale,
                            float cz, float* out_x, float* out_y, float* out_z) {
  *out_x = std::cos(theta) * r * scale;
  *out_y = std::sin(theta) * r * scale;
  *out_z = (z - cz) * scale;
}

void SphericalSurfaceToCartesian(float theta, float phi, float radius,
                                 float* out_x, float* out_y, float* out_z) {
  const float sin_phi = std::sin(phi);
  *out_x = radius * sin_phi * std::cos(theta);
  *out_y = radius * sin_phi * std::sin(theta);
  *out_z = radius * std::cos(phi);
}

void SphericalVolumeToCartesian(float r, float theta, float phi, float scale,
                                float* out_x, float* out_y, float* out_z) {
  const float scaled_r = r * scale;
  const float sin_phi = std::sin(phi);
  *out_x = scaled_r * sin_phi * std::cos(theta);
  *out_y = scaled_r * sin_phi * std::sin(theta);
  *out_z = scaled_r * std::cos(phi);
}

void ToroidalSurfaceToCartesian(float theta, float phi, float r,
                                float R, float* out_x, float* out_y, float* out_z) {
  const float cos_t = std::cos(theta);
  const float sin_t = std::sin(theta);
  const float cos_p = std::cos(phi);
  const float sin_p = std::sin(phi);
  const float ring = R + r * cos_t;
  *out_x = ring * cos_p;
  *out_y = ring * sin_p;
  *out_z = r * sin_t;
}

void ToroidalVolumeToCartesian(float r, float theta, float phi, float scale,
                               float R, float* out_x, float* out_y, float* out_z) {
  const float scaled_r = r * scale;
  const float cos_t = std::cos(theta);
  const float sin_t = std::sin(theta);
  const float cos_p = std::cos(phi);
  const float sin_p = std::sin(phi);
  const float ring = R + scaled_r * cos_t;
  *out_x = ring * cos_p;
  *out_y = ring * sin_p;
  *out_z = scaled_r * sin_t;
}

void TransformToCartesian(
    float x, float y, float z_val,
    GlViewer::ViewMode view_mode,
    float scale,
    float cx, float cy, float cz,
    float torus_major, float torus_minor,
    float base_radius, float radial_span, float t_z,
    bool use_value_height, bool has_z,
    float* out_x, float* out_y, float* out_z) {
  
  switch (view_mode) {
    case GlViewer::ViewMode::Polar: {
      const float r = x;
      const float theta = y;
      PolarToCartesian(r, theta, scale, out_x, out_y);
      *out_z = use_value_height ? (base_radius + (t_z - 0.5f) * radial_span)
                                : (has_z ? (z_val - cz) * scale : 0.0f);
      break;
    }
    
    case GlViewer::ViewMode::CylindricalVolume: {
      const float r = x;
      const float theta = y;
      CylindricalToCartesian(r, theta, z_val, scale, cz, out_x, out_y, out_z);
      break;
    }
    
    case GlViewer::ViewMode::SphericalSurface: {
      const float theta = x;
      const float phi = y;
      const float radius = base_radius + (t_z - 0.5f) * radial_span;
      SphericalSurfaceToCartesian(theta, phi, radius, out_x, out_y, out_z);
      break;
    }
    
    case GlViewer::ViewMode::SphericalVolume: {
      const float r = x;
      const float theta = y;
      const float phi = z_val;
      SphericalVolumeToCartesian(r, theta, phi, scale, out_x, out_y, out_z);
      break;
    }
    
    case GlViewer::ViewMode::ToroidalSurface: {
      const float theta = x;
      const float phi = y;
      const float R = torus_major * scale;
      const float r = torus_minor * scale + (t_z - 0.5f) * radial_span;
      ToroidalSurfaceToCartesian(theta, phi, r, R, out_x, out_y, out_z);
      break;
    }
    
    case GlViewer::ViewMode::ToroidalVolume: {
      const float r = x * scale;
      const float theta = y;
      const float phi = z_val;
      const float R = torus_major * scale;
      ToroidalVolumeToCartesian(r, theta, phi, 1.0f, R, out_x, out_y, out_z);
      break;
    }
    
    case GlViewer::ViewMode::Axisymmetric:
    case GlViewer::ViewMode::Cartesian:
    default: {
      *out_x = (x - cx) * scale;
      *out_y = (y - cy) * scale;
      *out_z = has_z ? (z_val - cz) * scale : (base_radius + (t_z - 0.5f) * radial_span);
      break;
    }
  }
}

}  // namespace CoordinateTransforms

