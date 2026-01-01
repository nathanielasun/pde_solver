#include "projection.h"
#include "coordinate_transforms.h"
#include <cmath>
#include <iomanip>
#include <sstream>
#include <algorithm>

namespace Projection {

constexpr float kPi = 3.14159265358979323846f;

bool ProjectPoint(const float mvp[16], int tex_width, int tex_height,
                  float x, float y, float z,
                  float* out_x, float* out_y) {
  if (!mvp || tex_width <= 0 || tex_height <= 0) {
    return false;
  }
  
  // Transform point by MVP matrix
  const float clip_x = mvp[0] * x + mvp[4] * y + mvp[8] * z + mvp[12];
  const float clip_y = mvp[1] * x + mvp[5] * y + mvp[9] * z + mvp[13];
  const float clip_z = mvp[2] * x + mvp[6] * y + mvp[10] * z + mvp[14];
  const float clip_w = mvp[3] * x + mvp[7] * y + mvp[11] * z + mvp[15];
  
  if (std::abs(clip_w) < 1e-6f) {
    return false;
  }
  
  // Normalize device coordinates
  const float ndc_x = clip_x / clip_w;
  const float ndc_y = clip_y / clip_w;
  const float ndc_z = clip_z / clip_w;
  
  // Check if point is in view frustum
  if (ndc_z < -2.0f || ndc_z > 2.0f) {
    return false;
  }
  
  // Convert to screen coordinates
  if (out_x) {
    *out_x = (ndc_x * 0.5f + 0.5f) * static_cast<float>(tex_width);
  }
  if (out_y) {
    *out_y = (1.0f - (ndc_y * 0.5f + 0.5f)) * static_cast<float>(tex_height);
  }
  
  return true;
}

std::string FormatValue(double value) {
  std::ostringstream out;
  out << std::setprecision(4) << std::defaultfloat << value;
  return out.str();
}

std::vector<GlViewer::ScreenLabel> GenerateAxisLabels(const AxisLabelParams& params) {
  std::vector<GlViewer::ScreenLabel> labels;
  
  if (!params.domain || !params.mvp || params.tex_width <= 0 || params.tex_height <= 0) {
    return labels;
  }
  
  const auto& domain = *params.domain;
  const int divisions = std::max(2, params.grid_divisions);
  const bool has_z = domain.nz > 1;
  const bool is_polar = !params.has_point_cloud && params.view_mode == GlViewer::ViewMode::Polar;
  const bool is_axisymmetric = !params.has_point_cloud && 
                               params.view_mode == GlViewer::ViewMode::Axisymmetric;
  const bool is_cylindrical = !params.has_point_cloud && 
                              params.view_mode == GlViewer::ViewMode::CylindricalVolume;
  const bool is_spherical_surface = !params.has_point_cloud && 
                                    params.view_mode == GlViewer::ViewMode::SphericalSurface;
  const bool is_spherical_volume = !params.has_point_cloud && 
                                   params.view_mode == GlViewer::ViewMode::SphericalVolume;
  const bool is_toroidal_surface = !params.has_point_cloud && 
                                   params.view_mode == GlViewer::ViewMode::ToroidalSurface;
  const bool is_toroidal_volume = !params.has_point_cloud && 
                                  params.view_mode == GlViewer::ViewMode::ToroidalVolume;
  const bool is_cartesian = params.has_point_cloud || 
                           params.view_mode == GlViewer::ViewMode::Cartesian || 
                           is_axisymmetric;
  
  // Compute scale and center (similar to mesh building)
  const double span_x = domain.xmax - domain.xmin;
  const double span_y = domain.ymax - domain.ymin;
  const double span_z = domain.zmax - domain.zmin;
  double span = std::max(span_x, span_y);
  
  if (has_z && is_cartesian) {
    span = std::max(span, span_z);
  }
  if (is_polar || is_spherical_volume || is_cylindrical) {
    const double max_r = std::max(std::abs(domain.xmin), std::abs(domain.xmax));
    span = std::max(1e-12, 2.0 * max_r);
  } else if (is_spherical_surface) {
    span = 2.0;
  } else if (is_toroidal_surface || is_toroidal_volume) {
    const double r_max = is_toroidal_surface
                             ? params.torus_minor
                             : std::max(std::abs(domain.xmin), std::abs(domain.xmax));
    span = std::max(1e-12, 2.0 * (params.torus_major + r_max));
  }
  
  const float scale = span > 0.0 ? static_cast<float>((1.2 * params.data_scale) / span) 
                                  : params.data_scale;
  const float cx = is_cartesian
                       ? static_cast<float>((domain.xmin + domain.xmax) * 0.5)
                       : 0.0f;
  const float cy = is_cartesian
                       ? static_cast<float>((domain.ymin + domain.ymax) * 0.5)
                       : 0.0f;
  const float cz = is_cartesian
                       ? static_cast<float>((domain.zmin + domain.zmax) * 0.5)
                       : 0.0f;
  
  const float x_min = (static_cast<float>(domain.xmin) - cx) * scale;
  const float x_max = (static_cast<float>(domain.xmax) - cx) * scale;
  const float y_min = (static_cast<float>(domain.ymin) - cy) * scale;
  const float y_max = (static_cast<float>(domain.ymax) - cy) * scale;
  const float z_min = has_z ? (static_cast<float>(domain.zmin) - cz) * scale
                            : -0.3f * params.data_scale;
  const float z_max = has_z ? (static_cast<float>(domain.zmax) - cz) * scale
                            : 0.3f * params.data_scale;
  
  // Generate labels based on coordinate system
  if (is_polar) {
    labels.reserve(static_cast<size_t>(divisions + 1) * 2);
    double r_min = domain.xmin;
    double r_max = domain.xmax;
    if (r_max < r_min) {
      std::swap(r_min, r_max);
    }
    const double theta_min = domain.ymin;
    const double theta_max = domain.ymax;
    const double theta_anchor =
        (theta_min <= 0.0 && theta_max >= 0.0) ? 0.0 : theta_min;
    double r_label = r_max;
    if (std::abs(r_label) < 1e-12) {
      r_label = r_min;
    }
    auto to_xy = [&](double r, double theta, float* out_x, float* out_y) {
      *out_x = static_cast<float>(r * std::cos(theta) * scale);
      *out_y = static_cast<float>(r * std::sin(theta) * scale);
    };

    for (int i = 0; i <= divisions; ++i) {
      const double t = static_cast<double>(i) / static_cast<double>(divisions);
      const double r_val = r_min + t * (r_max - r_min);
      float x = 0.0f;
      float y = 0.0f;
      to_xy(r_val, theta_anchor, &x, &y);
      float sx = 0.0f;
      float sy = 0.0f;
      if (ProjectPoint(params.mvp, params.tex_width, params.tex_height, x, y, z_min, &sx, &sy)) {
        GlViewer::ScreenLabel label;
        label.x = sx;
        label.y = sy;
        label.text = "r=" + FormatValue(r_val);
        labels.push_back(std::move(label));
      }
    }

    for (int i = 0; i <= divisions; ++i) {
      const double t = static_cast<double>(i) / static_cast<double>(divisions);
      const double theta_val = theta_min + t * (theta_max - theta_min);
      float x = 0.0f;
      float y = 0.0f;
      to_xy(r_label, theta_val, &x, &y);
      float sx = 0.0f;
      float sy = 0.0f;
      if (ProjectPoint(params.mvp, params.tex_width, params.tex_height, x, y, z_min, &sx, &sy)) {
        GlViewer::ScreenLabel label;
        label.x = sx;
        label.y = sy;
        label.text = "theta=" + FormatValue(theta_val);
        labels.push_back(std::move(label));
      }
    }
  } else if (is_cylindrical) {
    labels.reserve(static_cast<size_t>(divisions + 1) * 3);
    double r_min = domain.xmin;
    double r_max = domain.xmax;
    if (r_max < r_min) {
      std::swap(r_min, r_max);
    }
    const double theta_min = domain.ymin;
    const double theta_max = domain.ymax;
    const double theta_anchor =
        (theta_min <= 0.0 && theta_max >= 0.0) ? 0.0 : theta_min;
    double r_label = r_max;
    if (std::abs(r_label) < 1e-12) {
      r_label = r_min;
    }
    auto to_xy = [&](double r, double theta, float* out_x, float* out_y) {
      *out_x = static_cast<float>(r * std::cos(theta) * scale);
      *out_y = static_cast<float>(r * std::sin(theta) * scale);
    };

    for (int i = 0; i <= divisions; ++i) {
      const double t = static_cast<double>(i) / static_cast<double>(divisions);
      const double r_val = r_min + t * (r_max - r_min);
      float x = 0.0f;
      float y = 0.0f;
      to_xy(r_val, theta_anchor, &x, &y);
      float sx = 0.0f;
      float sy = 0.0f;
      if (ProjectPoint(params.mvp, params.tex_width, params.tex_height, x, y, z_min, &sx, &sy)) {
        GlViewer::ScreenLabel label;
        label.x = sx;
        label.y = sy;
        label.text = "r=" + FormatValue(r_val);
        labels.push_back(std::move(label));
      }
    }

    for (int i = 0; i <= divisions; ++i) {
      const double t = static_cast<double>(i) / static_cast<double>(divisions);
      const double theta_val = theta_min + t * (theta_max - theta_min);
      float x = 0.0f;
      float y = 0.0f;
      to_xy(r_label, theta_val, &x, &y);
      float sx = 0.0f;
      float sy = 0.0f;
      if (ProjectPoint(params.mvp, params.tex_width, params.tex_height, x, y, z_min, &sx, &sy)) {
        GlViewer::ScreenLabel label;
        label.x = sx;
        label.y = sy;
        label.text = "theta=" + FormatValue(theta_val);
        labels.push_back(std::move(label));
      }
    }

    float z_anchor_x = 0.0f;
    float z_anchor_y = 0.0f;
    to_xy(r_label, theta_anchor, &z_anchor_x, &z_anchor_y);
    for (int i = 0; i <= divisions; ++i) {
      const float t = static_cast<float>(i) / static_cast<float>(divisions);
      const float z = z_min + t * (z_max - z_min);
      const double z_val = domain.zmin + static_cast<double>(t) * span_z;
      float sx = 0.0f;
      float sy = 0.0f;
      if (ProjectPoint(params.mvp, params.tex_width, params.tex_height, z_anchor_x,
                       z_anchor_y, z, &sx, &sy)) {
        GlViewer::ScreenLabel label;
        label.x = sx;
        label.y = sy;
        label.text = "z=" + FormatValue(z_val);
        labels.push_back(std::move(label));
      }
    }
  } else if (is_spherical_surface || is_spherical_volume) {
    labels.reserve(static_cast<size_t>(divisions + 1) * 3);
    const double theta_min = domain.xmin;
    const double theta_max = domain.xmax;
    const double phi_min = is_spherical_surface ? domain.ymin : domain.zmin;
    const double phi_max = is_spherical_surface ? domain.ymax : domain.zmax;
    const double theta_anchor =
        (theta_min <= 0.0 && theta_max >= 0.0) ? 0.0 : theta_min;
    const double phi_anchor =
        (phi_min <= (kPi * 0.5) && phi_max >= (kPi * 0.5)) ? (kPi * 0.5)
                                                           : (phi_min + phi_max) * 0.5;
    const float base_radius = is_spherical_surface ? scale : 0.0f;
    const double r_min = is_spherical_surface ? base_radius : domain.xmin * scale;
    const double r_max = is_spherical_surface ? base_radius : domain.xmax * scale;
    const double r_label = std::max(r_min, r_max);

    auto to_xyz = [&](double r, double theta, double phi, float* out_x, float* out_y,
                      float* out_z) {
      const double sin_phi = std::sin(phi);
      *out_x = static_cast<float>(r * sin_phi * std::cos(theta));
      *out_y = static_cast<float>(r * sin_phi * std::sin(theta));
      *out_z = static_cast<float>(r * std::cos(phi));
    };

    for (int i = 0; i <= divisions; ++i) {
      const double t = static_cast<double>(i) / static_cast<double>(divisions);
      const double theta_val = theta_min + t * (theta_max - theta_min);
      float sx = 0.0f;
      float sy = 0.0f;
      float x = 0.0f;
      float y = 0.0f;
      float z = 0.0f;
      to_xyz(r_label, theta_val, phi_anchor, &x, &y, &z);
      if (ProjectPoint(params.mvp, params.tex_width, params.tex_height, x, y, z, &sx, &sy)) {
        GlViewer::ScreenLabel label;
        label.x = sx;
        label.y = sy;
        label.text = "theta=" + FormatValue(theta_val);
        labels.push_back(std::move(label));
      }
    }

    for (int i = 0; i <= divisions; ++i) {
      const double t = static_cast<double>(i) / static_cast<double>(divisions);
      const double phi_val = phi_min + t * (phi_max - phi_min);
      float sx = 0.0f;
      float sy = 0.0f;
      float x = 0.0f;
      float y = 0.0f;
      float z = 0.0f;
      to_xyz(r_label, theta_anchor, phi_val, &x, &y, &z);
      if (ProjectPoint(params.mvp, params.tex_width, params.tex_height, x, y, z, &sx, &sy)) {
        GlViewer::ScreenLabel label;
        label.x = sx;
        label.y = sy;
        label.text = "phi=" + FormatValue(phi_val);
        labels.push_back(std::move(label));
      }
    }

    if (is_spherical_volume) {
      for (int i = 0; i <= divisions; ++i) {
        const double t = static_cast<double>(i) / static_cast<double>(divisions);
        const double r_val = domain.xmin + t * (domain.xmax - domain.xmin);
        float sx = 0.0f;
        float sy = 0.0f;
        float x = 0.0f;
        float y = 0.0f;
        float z = 0.0f;
        to_xyz(r_val * scale, theta_anchor, phi_anchor, &x, &y, &z);
        if (ProjectPoint(params.mvp, params.tex_width, params.tex_height, x, y, z, &sx, &sy)) {
          GlViewer::ScreenLabel label;
          label.x = sx;
          label.y = sy;
          label.text = "r=" + FormatValue(r_val);
          labels.push_back(std::move(label));
        }
      }
    }
  } else if (is_toroidal_surface || is_toroidal_volume) {
    labels.reserve(static_cast<size_t>(divisions + 1) * 3);
    const double theta_min = domain.xmin;
    const double theta_max = domain.xmax;
    const double phi_min = domain.ymin;
    const double phi_max = domain.ymax;
    const double theta_anchor =
        (theta_min <= 0.0 && theta_max >= 0.0) ? 0.0 : theta_min;
    const double phi_anchor =
        (phi_min <= 0.0 && phi_max >= 0.0) ? 0.0 : (phi_min + phi_max) * 0.5;
    const double r_label = is_toroidal_surface
                               ? params.torus_minor
                               : std::max(std::abs(domain.xmin), std::abs(domain.xmax));
    const float R = params.torus_major * scale;
    auto to_xyz = [&](double r, double theta, double phi, float* out_x, float* out_y,
                      float* out_z) {
      const float r_scaled = static_cast<float>(r * scale);
      const float cos_t = std::cos(static_cast<float>(theta));
      const float sin_t = std::sin(static_cast<float>(theta));
      const float cos_p = std::cos(static_cast<float>(phi));
      const float sin_p = std::sin(static_cast<float>(phi));
      const float ring = R + r_scaled * cos_t;
      *out_x = ring * cos_p;
      *out_y = ring * sin_p;
      *out_z = r_scaled * sin_t;
    };

    for (int i = 0; i <= divisions; ++i) {
      const double t = static_cast<double>(i) / static_cast<double>(divisions);
      const double theta_val = theta_min + t * (theta_max - theta_min);
      float sx = 0.0f;
      float sy = 0.0f;
      float x = 0.0f;
      float y = 0.0f;
      float z = 0.0f;
      to_xyz(r_label, theta_val, phi_anchor, &x, &y, &z);
      if (ProjectPoint(params.mvp, params.tex_width, params.tex_height, x, y, z, &sx, &sy)) {
        GlViewer::ScreenLabel label;
        label.x = sx;
        label.y = sy;
        label.text = "theta=" + FormatValue(theta_val);
        labels.push_back(std::move(label));
      }
    }

    for (int i = 0; i <= divisions; ++i) {
      const double t = static_cast<double>(i) / static_cast<double>(divisions);
      const double phi_val = phi_min + t * (phi_max - phi_min);
      float sx = 0.0f;
      float sy = 0.0f;
      float x = 0.0f;
      float y = 0.0f;
      float z = 0.0f;
      to_xyz(r_label, theta_anchor, phi_val, &x, &y, &z);
      if (ProjectPoint(params.mvp, params.tex_width, params.tex_height, x, y, z, &sx, &sy)) {
        GlViewer::ScreenLabel label;
        label.x = sx;
        label.y = sy;
        label.text = "phi=" + FormatValue(phi_val);
        labels.push_back(std::move(label));
      }
    }

    if (is_toroidal_volume) {
      for (int i = 0; i <= divisions; ++i) {
        const double t = static_cast<double>(i) / static_cast<double>(divisions);
        const double r_val = domain.xmin + t * (domain.xmax - domain.xmin);
        float sx = 0.0f;
        float sy = 0.0f;
        float x = 0.0f;
        float y = 0.0f;
        float z = 0.0f;
        to_xyz(r_val, theta_anchor, phi_anchor, &x, &y, &z);
        if (ProjectPoint(params.mvp, params.tex_width, params.tex_height, x, y, z, &sx, &sy)) {
          GlViewer::ScreenLabel label;
          label.x = sx;
          label.y = sy;
          label.text = "r=" + FormatValue(r_val);
          labels.push_back(std::move(label));
        }
      }
    }
  } else {
    // Cartesian and Axisymmetric
    labels.reserve(static_cast<size_t>(divisions + 1) * 2);
    for (int i = 0; i <= divisions; ++i) {
      const float t = static_cast<float>(i) / static_cast<float>(divisions);
      const float x = x_min + t * (x_max - x_min);
      const float y = y_min + t * (y_max - y_min);
      const double x_val = domain.xmin + t * (domain.xmax - domain.xmin);
      const double y_val = domain.ymin + t * (domain.ymax - domain.ymin);

      float sx = 0.0f;
      float sy = 0.0f;
      if (ProjectPoint(params.mvp, params.tex_width, params.tex_height, x, y_min, z_min, &sx, &sy)) {
        GlViewer::ScreenLabel label;
        label.x = sx;
        label.y = sy;
        if (is_axisymmetric) {
          label.text = "r=" + FormatValue(x_val);
        } else {
          label.text = FormatValue(x_val);
        }
        labels.push_back(std::move(label));
      }
      if (ProjectPoint(params.mvp, params.tex_width, params.tex_height, x_min, y, z_min, &sx, &sy)) {
        GlViewer::ScreenLabel label;
        label.x = sx;
        label.y = sy;
        if (is_axisymmetric) {
          label.text = "z=" + FormatValue(y_val);
        } else {
          label.text = FormatValue(y_val);
        }
        label.align_right = true;
        labels.push_back(std::move(label));
      }
    }
  }

  // Z-axis labels (Cartesian/polar/axisymmetric systems)
  if (!is_spherical_surface && !is_spherical_volume &&
      !is_toroidal_surface && !is_toroidal_volume &&
      !is_cylindrical) {
    for (int i = 0; i <= divisions; ++i) {
      const float t = static_cast<float>(i) / static_cast<float>(divisions);
      const float z = z_min + t * (z_max - z_min);
      const double z_val = params.use_value_height
                               ? params.value_min + static_cast<double>(t) * params.value_range
                               : (domain.zmin + static_cast<double>(t) * span_z);
      float sx = 0.0f;
      float sy = 0.0f;
      float z_anchor_x = x_max;
      float z_anchor_y = y_min;
      if (is_polar) {
        double r_label = domain.xmax;
        if (std::abs(r_label) < 1e-12) {
          r_label = domain.xmin;
        }
        const double theta_anchor =
            (domain.ymin <= 0.0 && domain.ymax >= 0.0) ? 0.0 : domain.ymin;
        z_anchor_x = static_cast<float>(r_label * std::cos(theta_anchor) * scale);
        z_anchor_y = static_cast<float>(r_label * std::sin(theta_anchor) * scale);
      }
      if (ProjectPoint(params.mvp, params.tex_width, params.tex_height, z_anchor_x, z_anchor_y, z, &sx, &sy)) {
        GlViewer::ScreenLabel label;
        label.x = sx;
        label.y = sy;
        label.text = FormatValue(z_val);
        labels.push_back(std::move(label));
      }
    }
  }
  
  return labels;
}

}  // namespace Projection
