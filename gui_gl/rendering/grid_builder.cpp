#include "grid_builder.h"
#include "coordinate_transforms.h"
#include <algorithm>
#include <cmath>
#include <limits>

namespace GridBuilder {

void AddLine(std::vector<Vertex>& vertices,
             float x0, float y0, float z0,
             float x1, float y1, float z1,
             float r, float g, float b) {
  vertices.push_back({x0, y0, z0, r, g, b});
  vertices.push_back({x1, y1, z1, r, g, b});
}

static double clamp_value(double value, double min_val, double max_val) {
  if (max_val < min_val) {
    std::swap(min_val, max_val);
  }
  if (value < min_val) {
    return min_val;
  }
  if (value > max_val) {
    return max_val;
  }
  return value;
}

GridBuildResult BuildGrid(const GridBuildParams& params) {
  GridBuildResult result;
  
  if (!params.grid_enabled || !params.domain) {
    return result;
  }
  
  const auto& domain = *params.domain;
  const int nx = domain.nx;
  const int ny = domain.ny;
  const int nz = std::max(1, domain.nz);
  if (nx <= 0 || ny <= 0 || nz <= 0) {
    return result;
  }
  
  // Compute span and scale (similar to mesh building)
  const double span_x = domain.xmax - domain.xmin;
  const double span_y = domain.ymax - domain.ymin;
  const double span_z = domain.zmax - domain.zmin;
  double span = std::max(span_x, span_y);
  const bool has_z = nz > 1;
  
  const bool is_polar = !params.has_point_cloud && 
                       params.view_mode == GlViewer::ViewMode::Polar;
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
  const float rx = static_cast<float>(span_x * scale * 0.5);
  const float ry = static_cast<float>(span_y * scale * 0.5);
  const float rz = has_z ? static_cast<float>(span_z * scale * 0.5) 
                         : 0.3f * params.data_scale;
  
  // Update scene bounds
  if (is_spherical_surface) {
    result.scene_half_x = std::max(0.01f, scale);
    result.scene_half_y = std::max(0.01f, scale);
    result.scene_half_z = std::max(0.01f, scale);
  } else if (is_polar || is_spherical_volume || is_cylindrical) {
    result.scene_half_x = std::max(0.01f, static_cast<float>(span * scale * 0.5f));
    result.scene_half_y = std::max(0.01f, static_cast<float>(span * scale * 0.5f));
    result.scene_half_z = std::max(0.01f, rz);
  } else if (is_toroidal_surface || is_toroidal_volume) {
    const float r_max = static_cast<float>(
        (is_toroidal_surface ? params.torus_minor
                             : std::max(std::abs(domain.xmin), std::abs(domain.xmax))));
    result.scene_half_x = std::max(0.01f, (params.torus_major + r_max) * scale);
    result.scene_half_y = std::max(0.01f, (params.torus_major + r_max) * scale);
    result.scene_half_z = std::max(0.01f, r_max * scale);
  } else {
    result.scene_half_x = std::max(0.01f, rx);
    result.scene_half_y = std::max(0.01f, ry);
    result.scene_half_z = std::max(0.01f, rz);
  }
  result.scene_radius = std::max(
      0.05f,
      std::sqrt(result.scene_half_x * result.scene_half_x +
                result.scene_half_y * result.scene_half_y +
                result.scene_half_z * result.scene_half_z));
  
  const float x_min = (static_cast<float>(domain.xmin) - cx) * scale;
  const float x_max = (static_cast<float>(domain.xmax) - cx) * scale;
  const float y_min = (static_cast<float>(domain.ymin) - cy) * scale;
  const float y_max = (static_cast<float>(domain.ymax) - cy) * scale;
  const float z_min = has_z ? (static_cast<float>(domain.zmin) - cz) * scale
                            : -0.3f * params.data_scale;
  const float z_max = has_z ? (static_cast<float>(domain.zmax) - cz) * scale
                            : 0.3f * params.data_scale;
  
  // Grid colors
  const float grid_r = 0.28f;
  const float grid_g = 0.3f;
  const float grid_b = 0.34f;
  const float edge_r = 0.45f;
  const float edge_g = 0.48f;
  const float edge_b = 0.52f;
  const float highlight_r = 0.95f;
  const float highlight_g = 0.78f;
  const float highlight_b = 0.22f;
  
  result.vertices.reserve(static_cast<size_t>((params.grid_divisions + 1) * 4 + 24));
  
  auto add_line = [&](float x0, float y0, float z0, float x1, float y1, float z1,
                      float r, float g, float b) {
    AddLine(result.vertices, x0, y0, z0, x1, y1, z1, r, g, b);
  };
  
  // Build grid based on coordinate system
  if (is_spherical_surface || is_spherical_volume) {
    const double theta_min = domain.xmin;
    const double theta_max = domain.xmax;
    const double phi_min = is_spherical_surface ? domain.ymin : domain.zmin;
    const double phi_max = is_spherical_surface ? domain.ymax : domain.zmax;
    const int segments = std::max(32, params.grid_divisions * 8);
    auto to_xyz = [&](double r, double theta, double phi, float* out_x, float* out_y,
                      float* out_z) {
      const double sin_phi = std::sin(phi);
      const double px = r * sin_phi * std::cos(theta);
      const double py = r * sin_phi * std::sin(theta);
      const double pz = r * std::cos(phi);
      *out_x = static_cast<float>(px);
      *out_y = static_cast<float>(py);
      *out_z = static_cast<float>(pz);
    };

    const double r_min = is_spherical_surface ? scale : domain.xmin * scale;
    const double r_max = is_spherical_surface ? scale : domain.xmax * scale;
    const double r_outer = std::max(r_min, r_max);
    const double r_inner = std::min(r_min, r_max);

    auto draw_lat_long = [&](double radius) {
      for (int i = 0; i <= params.grid_divisions; ++i) {
        const double t = static_cast<double>(i) / static_cast<double>(params.grid_divisions);
        const double theta = theta_min + t * (theta_max - theta_min);
        float prev_x = 0.0f;
        float prev_y = 0.0f;
        float prev_z = 0.0f;
        for (int s = 0; s <= segments; ++s) {
          const double tt = static_cast<double>(s) / static_cast<double>(segments);
          const double phi = phi_min + tt * (phi_max - phi_min);
          float x = 0.0f;
          float y = 0.0f;
          float z = 0.0f;
          to_xyz(radius, theta, phi, &x, &y, &z);
          if (s > 0) {
            add_line(prev_x, prev_y, prev_z, x, y, z, grid_r, grid_g, grid_b);
          }
          prev_x = x;
          prev_y = y;
          prev_z = z;
        }
      }

      for (int i = 0; i <= params.grid_divisions; ++i) {
        const double t = static_cast<double>(i) / static_cast<double>(params.grid_divisions);
        const double phi = phi_min + t * (phi_max - phi_min);
        float prev_x = 0.0f;
        float prev_y = 0.0f;
        float prev_z = 0.0f;
        for (int s = 0; s <= segments; ++s) {
          const double tt = static_cast<double>(s) / static_cast<double>(segments);
          const double theta = theta_min + tt * (theta_max - theta_min);
          float x = 0.0f;
          float y = 0.0f;
          float z = 0.0f;
          to_xyz(radius, theta, phi, &x, &y, &z);
          if (s > 0) {
            add_line(prev_x, prev_y, prev_z, x, y, z, grid_r, grid_g, grid_b);
          }
          prev_x = x;
          prev_y = y;
          prev_z = z;
        }
      }
    };

    draw_lat_long(r_outer);
    if (is_spherical_volume && std::abs(r_inner - r_outer) > 1e-6) {
      draw_lat_long(r_inner);
      for (int i = 0; i <= params.grid_divisions; ++i) {
        const double t = static_cast<double>(i) / static_cast<double>(params.grid_divisions);
        const double theta = theta_min + t * (theta_max - theta_min);
        for (int j = 0; j <= params.grid_divisions; ++j) {
          const double u = static_cast<double>(j) / static_cast<double>(params.grid_divisions);
          const double phi = phi_min + u * (phi_max - phi_min);
          float x0 = 0.0f;
          float y0 = 0.0f;
          float z0 = 0.0f;
          float x1 = 0.0f;
          float y1 = 0.0f;
          float z1 = 0.0f;
          to_xyz(r_inner, theta, phi, &x0, &y0, &z0);
          to_xyz(r_outer, theta, phi, &x1, &y1, &z1);
          add_line(x0, y0, z0, x1, y1, z1, grid_r, grid_g, grid_b);
        }
      }
    }
  } else if (is_toroidal_surface || is_toroidal_volume) {
    const double theta_min = domain.xmin;
    const double theta_max = domain.xmax;
    const double phi_min = domain.ymin;
    const double phi_max = domain.ymax;
    const int segments = std::max(32, params.grid_divisions * 8);
    const float R = params.torus_major * scale;
    const double r_min_raw = is_toroidal_surface ? params.torus_minor : domain.xmin;
    const double r_max_raw = is_toroidal_surface ? params.torus_minor : domain.xmax;
    const float r_inner = static_cast<float>(std::min(r_min_raw, r_max_raw) * scale);
    const float r_outer = static_cast<float>(std::max(r_min_raw, r_max_raw) * scale);

    auto to_xyz = [&](float r, double theta, double phi, float* out_x, float* out_y,
                      float* out_z) {
      const float cos_t = std::cos(static_cast<float>(theta));
      const float sin_t = std::sin(static_cast<float>(theta));
      const float cos_p = std::cos(static_cast<float>(phi));
      const float sin_p = std::sin(static_cast<float>(phi));
      const float ring = R + r * cos_t;
      *out_x = ring * cos_p;
      *out_y = ring * sin_p;
      *out_z = r * sin_t;
    };

    auto draw_surface = [&](float r) {
      for (int i = 0; i <= params.grid_divisions; ++i) {
        const double t = static_cast<double>(i) / static_cast<double>(params.grid_divisions);
        const double theta = theta_min + t * (theta_max - theta_min);
        float prev_x = 0.0f;
        float prev_y = 0.0f;
        float prev_z = 0.0f;
        for (int s = 0; s <= segments; ++s) {
          const double tt = static_cast<double>(s) / static_cast<double>(segments);
          const double phi = phi_min + tt * (phi_max - phi_min);
          float x = 0.0f;
          float y = 0.0f;
          float z = 0.0f;
          to_xyz(r, theta, phi, &x, &y, &z);
          if (s > 0) {
            add_line(prev_x, prev_y, prev_z, x, y, z, grid_r, grid_g, grid_b);
          }
          prev_x = x;
          prev_y = y;
          prev_z = z;
        }
      }

      for (int i = 0; i <= params.grid_divisions; ++i) {
        const double t = static_cast<double>(i) / static_cast<double>(params.grid_divisions);
        const double phi = phi_min + t * (phi_max - phi_min);
        float prev_x = 0.0f;
        float prev_y = 0.0f;
        float prev_z = 0.0f;
        for (int s = 0; s <= segments; ++s) {
          const double tt = static_cast<double>(s) / static_cast<double>(segments);
          const double theta = theta_min + tt * (theta_max - theta_min);
          float x = 0.0f;
          float y = 0.0f;
          float z = 0.0f;
          to_xyz(r, theta, phi, &x, &y, &z);
          if (s > 0) {
            add_line(prev_x, prev_y, prev_z, x, y, z, grid_r, grid_g, grid_b);
          }
          prev_x = x;
          prev_y = y;
          prev_z = z;
        }
      }
    };

    draw_surface(r_outer);
    if (is_toroidal_volume && std::abs(r_outer - r_inner) > 1e-6f) {
      draw_surface(r_inner);
      for (int i = 0; i <= params.grid_divisions; ++i) {
        const double t = static_cast<double>(i) / static_cast<double>(params.grid_divisions);
        const double theta = theta_min + t * (theta_max - theta_min);
        for (int j = 0; j <= params.grid_divisions; ++j) {
          const double u = static_cast<double>(j) / static_cast<double>(params.grid_divisions);
          const double phi = phi_min + u * (phi_max - phi_min);
          float x0 = 0.0f;
          float y0 = 0.0f;
          float z0 = 0.0f;
          float x1 = 0.0f;
          float y1 = 0.0f;
          float z1 = 0.0f;
          to_xyz(r_inner, theta, phi, &x0, &y0, &z0);
          to_xyz(r_outer, theta, phi, &x1, &y1, &z1);
          add_line(x0, y0, z0, x1, y1, z1, grid_r, grid_g, grid_b);
        }
      }
    }
  } else if (is_cylindrical) {
    double r_min = domain.xmin;
    double r_max = domain.xmax;
    if (r_max < r_min) {
      std::swap(r_min, r_max);
    }
    const double theta_min = domain.ymin;
    const double theta_max = domain.ymax;
    const int segments = std::max(32, params.grid_divisions * 8);
    auto to_xy = [&](double r, double theta, float* out_x, float* out_y) {
      *out_x = static_cast<float>(r * std::cos(theta) * scale);
      *out_y = static_cast<float>(r * std::sin(theta) * scale);
    };

    for (int i = 0; i <= params.grid_divisions; ++i) {
      const double t = static_cast<double>(i) / static_cast<double>(params.grid_divisions);
      const double r = r_min + t * (r_max - r_min);
      float prev_x = 0.0f;
      float prev_y = 0.0f;
      for (int s = 0; s <= segments; ++s) {
        const double tt = static_cast<double>(s) / static_cast<double>(segments);
        const double theta = theta_min + tt * (theta_max - theta_min);
        float x = 0.0f;
        float y = 0.0f;
        to_xy(r, theta, &x, &y);
        if (s > 0) {
          add_line(prev_x, prev_y, z_min, x, y, z_min, grid_r, grid_g, grid_b);
          add_line(prev_x, prev_y, z_max, x, y, z_max, grid_r, grid_g, grid_b);
        }
        prev_x = x;
        prev_y = y;
      }
    }

    for (int i = 0; i <= params.grid_divisions; ++i) {
      const double t = static_cast<double>(i) / static_cast<double>(params.grid_divisions);
      const double theta = theta_min + t * (theta_max - theta_min);
      float x0 = 0.0f;
      float y0 = 0.0f;
      float x1 = 0.0f;
      float y1 = 0.0f;
      to_xy(r_min, theta, &x0, &y0);
      to_xy(r_max, theta, &x1, &y1);
      add_line(x0, y0, z_min, x1, y1, z_min, grid_r, grid_g, grid_b);
      add_line(x0, y0, z_max, x1, y1, z_max, grid_r, grid_g, grid_b);
    }

    for (int i = 0; i <= params.grid_divisions; ++i) {
      const double t = static_cast<double>(i) / static_cast<double>(params.grid_divisions);
      const double r = r_min + t * (r_max - r_min);
      for (int j = 0; j <= params.grid_divisions; ++j) {
        const double u = static_cast<double>(j) / static_cast<double>(params.grid_divisions);
        const double theta = theta_min + u * (theta_max - theta_min);
        float x = 0.0f;
        float y = 0.0f;
        to_xy(r, theta, &x, &y);
        add_line(x, y, z_min, x, y, z_max, grid_r, grid_g, grid_b);
      }
    }
  } else if (is_polar) {
    double r_min = domain.xmin;
    double r_max = domain.xmax;
    if (r_max < r_min) {
      std::swap(r_min, r_max);
    }
    const double theta_min = domain.ymin;
    const double theta_max = domain.ymax;
    const int segments = std::max(32, params.grid_divisions * 8);
    auto to_xy = [&](double r, double theta, float* out_x, float* out_y) {
      *out_x = static_cast<float>(r * std::cos(theta) * scale);
      *out_y = static_cast<float>(r * std::sin(theta) * scale);
    };

    for (int i = 0; i <= params.grid_divisions; ++i) {
      const double t = static_cast<double>(i) / static_cast<double>(params.grid_divisions);
      const double r = r_min + t * (r_max - r_min);
      float prev_x = 0.0f;
      float prev_y = 0.0f;
      for (int s = 0; s <= segments; ++s) {
        const double tt = static_cast<double>(s) / static_cast<double>(segments);
        const double theta = theta_min + tt * (theta_max - theta_min);
        float x = 0.0f;
        float y = 0.0f;
        to_xy(r, theta, &x, &y);
        if (s > 0) {
          add_line(prev_x, prev_y, z_min, x, y, z_min, grid_r, grid_g, grid_b);
          add_line(prev_x, prev_y, z_max, x, y, z_max, grid_r, grid_g, grid_b);
        }
        prev_x = x;
        prev_y = y;
      }
    }

    for (int i = 0; i <= params.grid_divisions; ++i) {
      const double t = static_cast<double>(i) / static_cast<double>(params.grid_divisions);
      const double theta = theta_min + t * (theta_max - theta_min);
      float x0 = 0.0f;
      float y0 = 0.0f;
      float x1 = 0.0f;
      float y1 = 0.0f;
      to_xy(r_min, theta, &x0, &y0);
      to_xy(r_max, theta, &x1, &y1);
      add_line(x0, y0, z_min, x1, y1, z_min, grid_r, grid_g, grid_b);
      add_line(x0, y0, z_max, x1, y1, z_max, grid_r, grid_g, grid_b);
      add_line(x1, y1, z_min, x1, y1, z_max, grid_r, grid_g, grid_b);
      add_line(x0, y0, z_min, x0, y0, z_max, grid_r, grid_g, grid_b);
    }
  } else {
    // Cartesian and Axisymmetric
    for (int i = 0; i <= params.grid_divisions; ++i) {
      const float t = static_cast<float>(i) / static_cast<float>(params.grid_divisions);
      const float y = y_min + t * (y_max - y_min);
      add_line(x_min, y, z_min, x_max, y, z_min, grid_r, grid_g, grid_b);
      add_line(x_min, y, z_max, x_max, y, z_max, grid_r, grid_g, grid_b);
    }
    for (int i = 0; i <= params.grid_divisions; ++i) {
      const float t = static_cast<float>(i) / static_cast<float>(params.grid_divisions);
      const float x = x_min + t * (x_max - x_min);
      add_line(x, y_min, z_min, x, y_max, z_min, grid_r, grid_g, grid_b);
      add_line(x, y_min, z_max, x, y_max, z_max, grid_r, grid_g, grid_b);
    }

    if (has_z) {
      for (int i = 0; i <= params.grid_divisions; ++i) {
        const float t = static_cast<float>(i) / static_cast<float>(params.grid_divisions);
        const float z = z_min + t * (z_max - z_min);
        add_line(x_min, y_min, z, x_max, y_min, z, grid_r, grid_g, grid_b);
        add_line(x_min, y_max, z, x_max, y_max, z, grid_r, grid_g, grid_b);
      }
      for (int i = 0; i <= params.grid_divisions; ++i) {
        const float t = static_cast<float>(i) / static_cast<float>(params.grid_divisions);
        const float x = x_min + t * (x_max - x_min);
        add_line(x, y_min, z_min, x, y_min, z_max, grid_r, grid_g, grid_b);
        add_line(x, y_max, z_min, x, y_max, z_max, grid_r, grid_g, grid_b);
      }
      for (int i = 0; i <= params.grid_divisions; ++i) {
        const float t = static_cast<float>(i) / static_cast<float>(params.grid_divisions);
        const float z = z_min + t * (z_max - z_min);
        add_line(x_min, y_min, z, x_min, y_max, z, grid_r, grid_g, grid_b);
        add_line(x_max, y_min, z, x_max, y_max, z, grid_r, grid_g, grid_b);
      }
      for (int i = 0; i <= params.grid_divisions; ++i) {
        const float t = static_cast<float>(i) / static_cast<float>(params.grid_divisions);
        const float y = y_min + t * (y_max - y_min);
        add_line(x_min, y, z_min, x_min, y, z_max, grid_r, grid_g, grid_b);
        add_line(x_max, y, z_min, x_max, y, z_max, grid_r, grid_g, grid_b);
      }
    }

    // Edge lines (highlighted)
    add_line(x_min, y_min, z_min, x_max, y_min, z_min, edge_r, edge_g, edge_b);
    add_line(x_min, y_max, z_min, x_max, y_max, z_min, edge_r, edge_g, edge_b);
    add_line(x_min, y_min, z_max, x_max, y_min, z_max, edge_r, edge_g, edge_b);
    add_line(x_min, y_max, z_max, x_max, y_max, z_max, edge_r, edge_g, edge_b);
    add_line(x_min, y_min, z_min, x_min, y_max, z_min, edge_r, edge_g, edge_b);
    add_line(x_max, y_min, z_min, x_max, y_max, z_min, edge_r, edge_g, edge_b);
    add_line(x_min, y_min, z_max, x_min, y_max, z_max, edge_r, edge_g, edge_b);
    add_line(x_max, y_min, z_max, x_max, y_max, z_max, edge_r, edge_g, edge_b);
    add_line(x_min, y_min, z_min, x_min, y_min, z_max, edge_r, edge_g, edge_b);
    add_line(x_max, y_min, z_min, x_max, y_min, z_max, edge_r, edge_g, edge_b);
    add_line(x_min, y_max, z_min, x_min, y_max, z_max, edge_r, edge_g, edge_b);
    add_line(x_max, y_max, z_min, x_max, y_max, z_max, edge_r, edge_g, edge_b);
  }

  // Slice plane visualization
  if (params.slice_enabled) {
    const int segments = std::max(64, params.grid_divisions * 10);
    if (is_cartesian) {
      if (params.slice_axis == 0) {
        const double axis = clamp_value(params.slice_value, domain.xmin, domain.xmax);
        const float x = (static_cast<float>(axis) - cx) * scale;
        add_line(x, y_min, z_min, x, y_max, z_min, highlight_r, highlight_g, highlight_b);
        add_line(x, y_min, z_max, x, y_max, z_max, highlight_r, highlight_g, highlight_b);
        add_line(x, y_min, z_min, x, y_min, z_max, highlight_r, highlight_g, highlight_b);
        add_line(x, y_max, z_min, x, y_max, z_max, highlight_r, highlight_g, highlight_b);
      } else if (params.slice_axis == 1) {
        const double axis = clamp_value(params.slice_value, domain.ymin, domain.ymax);
        const float y = (static_cast<float>(axis) - cy) * scale;
        add_line(x_min, y, z_min, x_max, y, z_min, highlight_r, highlight_g, highlight_b);
        add_line(x_min, y, z_max, x_max, y, z_max, highlight_r, highlight_g, highlight_b);
        add_line(x_min, y, z_min, x_min, y, z_max, highlight_r, highlight_g, highlight_b);
        add_line(x_max, y, z_min, x_max, y, z_max, highlight_r, highlight_g, highlight_b);
      } else {
        const double axis = clamp_value(params.slice_value, domain.zmin, domain.zmax);
        const float z = (static_cast<float>(axis) - cz) * scale;
        add_line(x_min, y_min, z, x_max, y_min, z, highlight_r, highlight_g, highlight_b);
        add_line(x_min, y_max, z, x_max, y_max, z, highlight_r, highlight_g, highlight_b);
        add_line(x_min, y_min, z, x_min, y_max, z, highlight_r, highlight_g, highlight_b);
        add_line(x_max, y_min, z, x_max, y_max, z, highlight_r, highlight_g, highlight_b);
      }
    } else if (is_polar) {
      double r_min = domain.xmin;
      double r_max = domain.xmax;
      if (r_max < r_min) {
        std::swap(r_min, r_max);
      }
      const double theta_min = domain.ymin;
      const double theta_max = domain.ymax;
      auto to_xy = [&](double r, double theta, float* out_x, float* out_y) {
        *out_x = static_cast<float>(r * std::cos(theta) * scale);
        *out_y = static_cast<float>(r * std::sin(theta) * scale);
      };
      if (params.slice_axis == 0) {
        const double r_val = clamp_value(params.slice_value, r_min, r_max);
        float prev_x = 0.0f;
        float prev_y = 0.0f;
        for (int s = 0; s <= segments; ++s) {
          const double tt = static_cast<double>(s) / static_cast<double>(segments);
          const double theta = theta_min + tt * (theta_max - theta_min);
          float x = 0.0f;
          float y = 0.0f;
          to_xy(r_val, theta, &x, &y);
          if (s > 0) {
            add_line(prev_x, prev_y, z_min, x, y, z_min, highlight_r, highlight_g,
                     highlight_b);
          }
          prev_x = x;
          prev_y = y;
        }
      } else {
        const double theta_val = clamp_value(params.slice_value, theta_min, theta_max);
        float x0 = 0.0f;
        float y0 = 0.0f;
        float x1 = 0.0f;
        float y1 = 0.0f;
        to_xy(r_min, theta_val, &x0, &y0);
        to_xy(r_max, theta_val, &x1, &y1);
        add_line(x0, y0, z_min, x1, y1, z_min, highlight_r, highlight_g, highlight_b);
        add_line(x0, y0, z_max, x1, y1, z_max, highlight_r, highlight_g, highlight_b);
      }
    } else if (is_cylindrical) {
      double r_min = domain.xmin;
      double r_max = domain.xmax;
      if (r_max < r_min) {
        std::swap(r_min, r_max);
      }
      const double theta_min = domain.ymin;
      const double theta_max = domain.ymax;
      auto to_xy = [&](double r, double theta, float* out_x, float* out_y) {
        *out_x = static_cast<float>(r * std::cos(theta) * scale);
        *out_y = static_cast<float>(r * std::sin(theta) * scale);
      };
      if (params.slice_axis == 0) {
        const double r_val = clamp_value(params.slice_value, r_min, r_max);
        float prev_x = 0.0f;
        float prev_y = 0.0f;
        for (int s = 0; s <= segments; ++s) {
          const double tt = static_cast<double>(s) / static_cast<double>(segments);
          const double theta = theta_min + tt * (theta_max - theta_min);
          float x = 0.0f;
          float y = 0.0f;
          to_xy(r_val, theta, &x, &y);
          if (s > 0) {
            add_line(prev_x, prev_y, z_min, x, y, z_min, highlight_r, highlight_g,
                     highlight_b);
            add_line(prev_x, prev_y, z_max, x, y, z_max, highlight_r, highlight_g,
                     highlight_b);
          }
          prev_x = x;
          prev_y = y;
        }
      } else if (params.slice_axis == 1) {
        const double theta_val = clamp_value(params.slice_value, theta_min, theta_max);
        float x0 = 0.0f;
        float y0 = 0.0f;
        float x1 = 0.0f;
        float y1 = 0.0f;
        to_xy(r_min, theta_val, &x0, &y0);
        to_xy(r_max, theta_val, &x1, &y1);
        add_line(x0, y0, z_min, x1, y1, z_min, highlight_r, highlight_g, highlight_b);
        add_line(x0, y0, z_max, x1, y1, z_max, highlight_r, highlight_g, highlight_b);
      } else {
        const double z_val = clamp_value(params.slice_value, domain.zmin, domain.zmax);
        const float z = (static_cast<float>(z_val) - cz) * scale;
        float prev_x = 0.0f;
        float prev_y = 0.0f;
        for (int s = 0; s <= segments; ++s) {
          const double tt = static_cast<double>(s) / static_cast<double>(segments);
          const double theta = theta_min + tt * (theta_max - theta_min);
          float x = 0.0f;
          float y = 0.0f;
          to_xy(r_max, theta, &x, &y);
          if (s > 0) {
            add_line(prev_x, prev_y, z, x, y, z, highlight_r, highlight_g, highlight_b);
          }
          prev_x = x;
          prev_y = y;
        }
        if (r_min > 1e-6) {
          prev_x = 0.0f;
          prev_y = 0.0f;
          for (int s = 0; s <= segments; ++s) {
            const double tt = static_cast<double>(s) / static_cast<double>(segments);
            const double theta = theta_min + tt * (theta_max - theta_min);
            float x = 0.0f;
            float y = 0.0f;
            to_xy(r_min, theta, &x, &y);
            if (s > 0) {
              add_line(prev_x, prev_y, z, x, y, z, highlight_r, highlight_g, highlight_b);
            }
            prev_x = x;
            prev_y = y;
          }
        }
      }
    } else if (is_spherical_surface || is_spherical_volume) {
      const double theta_min = domain.xmin;
      const double theta_max = domain.xmax;
      const double phi_min = is_spherical_surface ? domain.ymin : domain.zmin;
      const double phi_max = is_spherical_surface ? domain.ymax : domain.zmax;
      auto to_xyz = [&](double r, double theta, double phi, float* out_x, float* out_y,
                        float* out_z) {
        const double sin_phi = std::sin(phi);
        const double px = r * sin_phi * std::cos(theta);
        const double py = r * sin_phi * std::sin(theta);
        const double pz = r * std::cos(phi);
        *out_x = static_cast<float>(px);
        *out_y = static_cast<float>(py);
        *out_z = static_cast<float>(pz);
      };
      if (params.slice_axis == 0) {
        const double theta_val = clamp_value(params.slice_value, theta_min, theta_max);
        const double r_val = is_spherical_surface
                                 ? scale
                                 : clamp_value(domain.xmax, domain.xmin, domain.xmax) * scale;
        float prev_x = 0.0f;
        float prev_y = 0.0f;
        float prev_z = 0.0f;
        for (int s = 0; s <= segments; ++s) {
          const double tt = static_cast<double>(s) / static_cast<double>(segments);
          const double phi = phi_min + tt * (phi_max - phi_min);
          float x = 0.0f;
          float y = 0.0f;
          float z = 0.0f;
          to_xyz(r_val, theta_val, phi, &x, &y, &z);
          if (s > 0) {
            add_line(prev_x, prev_y, prev_z, x, y, z, highlight_r, highlight_g,
                     highlight_b);
          }
          prev_x = x;
          prev_y = y;
          prev_z = z;
        }
      } else if (params.slice_axis == 1) {
        const double phi_val = clamp_value(params.slice_value, phi_min, phi_max);
        const double r_val = is_spherical_surface
                                 ? scale
                                 : clamp_value(domain.xmax, domain.xmin, domain.xmax) * scale;
        float prev_x = 0.0f;
        float prev_y = 0.0f;
        float prev_z = 0.0f;
        for (int s = 0; s <= segments; ++s) {
          const double tt = static_cast<double>(s) / static_cast<double>(segments);
          const double theta = theta_min + tt * (theta_max - theta_min);
          float x = 0.0f;
          float y = 0.0f;
          float z = 0.0f;
          to_xyz(r_val, theta, phi_val, &x, &y, &z);
          if (s > 0) {
            add_line(prev_x, prev_y, prev_z, x, y, z, highlight_r, highlight_g,
                     highlight_b);
          }
          prev_x = x;
          prev_y = y;
          prev_z = z;
        }
      } else if (is_spherical_volume) {
        const double r_min = domain.xmin;
        const double r_max = domain.xmax;
        const double r_val = clamp_value(params.slice_value, r_min, r_max) * scale;
        for (int i = 0; i <= params.grid_divisions; ++i) {
          const double t = static_cast<double>(i) / static_cast<double>(params.grid_divisions);
          const double theta = theta_min + t * (theta_max - theta_min);
          float prev_x = 0.0f;
          float prev_y = 0.0f;
          float prev_z = 0.0f;
          for (int s = 0; s <= segments; ++s) {
            const double tt = static_cast<double>(s) / static_cast<double>(segments);
            const double phi = phi_min + tt * (phi_max - phi_min);
            float x = 0.0f;
            float y = 0.0f;
            float z = 0.0f;
            to_xyz(r_val, theta, phi, &x, &y, &z);
            if (s > 0) {
              add_line(prev_x, prev_y, prev_z, x, y, z, highlight_r, highlight_g,
                       highlight_b);
            }
            prev_x = x;
            prev_y = y;
            prev_z = z;
          }
        }
      }
    } else if (is_toroidal_surface || is_toroidal_volume) {
      const double theta_min = domain.xmin;
      const double theta_max = domain.xmax;
      const double phi_min = domain.ymin;
      const double phi_max = domain.ymax;
      const float R = params.torus_major * scale;
      const double r_min = is_toroidal_surface ? params.torus_minor : domain.xmin;
      const double r_max = is_toroidal_surface ? params.torus_minor : domain.xmax;
      auto to_xyz = [&](float r, double theta, double phi, float* out_x, float* out_y,
                        float* out_z) {
        const float cos_t = std::cos(static_cast<float>(theta));
        const float sin_t = std::sin(static_cast<float>(theta));
        const float cos_p = std::cos(static_cast<float>(phi));
        const float sin_p = std::sin(static_cast<float>(phi));
        const float ring = R + r * cos_t;
        *out_x = ring * cos_p;
        *out_y = ring * sin_p;
        *out_z = r * sin_t;
      };
      if (params.slice_axis == 0) {
        const double theta_val = clamp_value(params.slice_value, theta_min, theta_max);
        const float r_val =
            static_cast<float>(clamp_value(params.slice_value, r_min, r_max) * scale);
        const float r_use = is_toroidal_surface ? params.torus_minor * scale : r_val;
        float prev_x = 0.0f;
        float prev_y = 0.0f;
        float prev_z = 0.0f;
        for (int s = 0; s <= segments; ++s) {
          const double tt = static_cast<double>(s) / static_cast<double>(segments);
          const double phi = phi_min + tt * (phi_max - phi_min);
          float x = 0.0f;
          float y = 0.0f;
          float z = 0.0f;
          to_xyz(r_use, theta_val, phi, &x, &y, &z);
          if (s > 0) {
            add_line(prev_x, prev_y, prev_z, x, y, z, highlight_r, highlight_g,
                     highlight_b);
          }
          prev_x = x;
          prev_y = y;
          prev_z = z;
        }
      } else if (params.slice_axis == 1) {
        const double phi_val = clamp_value(params.slice_value, phi_min, phi_max);
        const float r_use =
            static_cast<float>((is_toroidal_surface ? params.torus_minor : r_max) * scale);
        float prev_x = 0.0f;
        float prev_y = 0.0f;
        float prev_z = 0.0f;
        for (int s = 0; s <= segments; ++s) {
          const double tt = static_cast<double>(s) / static_cast<double>(segments);
          const double theta = theta_min + tt * (theta_max - theta_min);
          float x = 0.0f;
          float y = 0.0f;
          float z = 0.0f;
          to_xyz(r_use, theta, phi_val, &x, &y, &z);
          if (s > 0) {
            add_line(prev_x, prev_y, prev_z, x, y, z, highlight_r, highlight_g,
                     highlight_b);
          }
          prev_x = x;
          prev_y = y;
          prev_z = z;
        }
      } else if (is_toroidal_volume) {
        const float r_use =
            static_cast<float>(clamp_value(params.slice_value, r_min, r_max) * scale);
        const double theta_anchor =
            (theta_min <= 0.0 && theta_max >= 0.0) ? 0.0 : theta_min;
        const double phi_anchor =
            (phi_min <= 0.0 && phi_max >= 0.0) ? 0.0 : phi_min;
        float prev_x = 0.0f;
        float prev_y = 0.0f;
        float prev_z = 0.0f;
        for (int s = 0; s <= segments; ++s) {
          const double tt = static_cast<double>(s) / static_cast<double>(segments);
          const double theta = theta_min + tt * (theta_max - theta_min);
          float x = 0.0f;
          float y = 0.0f;
          float z = 0.0f;
          to_xyz(r_use, theta, phi_anchor, &x, &y, &z);
          if (s > 0) {
            add_line(prev_x, prev_y, prev_z, x, y, z, highlight_r, highlight_g,
                     highlight_b);
          }
          prev_x = x;
          prev_y = y;
          prev_z = z;
        }
        prev_x = 0.0f;
        prev_y = 0.0f;
        prev_z = 0.0f;
        for (int s = 0; s <= segments; ++s) {
          const double tt = static_cast<double>(s) / static_cast<double>(segments);
          const double phi = phi_min + tt * (phi_max - phi_min);
          float x = 0.0f;
          float y = 0.0f;
          float z = 0.0f;
          to_xyz(r_use, theta_anchor, phi, &x, &y, &z);
          if (s > 0) {
            add_line(prev_x, prev_y, prev_z, x, y, z, highlight_r, highlight_g,
                     highlight_b);
          }
          prev_x = x;
          prev_y = y;
          prev_z = z;
        }
      }
    }
  }
  
  // Axes
  const float axis_len = 1.2f * result.scene_radius;
  add_line(0.0f, 0.0f, 0.0f, axis_len, 0.0f, 0.0f, 1.0f, 0.2f, 0.2f);
  add_line(0.0f, 0.0f, 0.0f, 0.0f, axis_len, 0.0f, 0.2f, 1.0f, 0.2f);
  add_line(0.0f, 0.0f, 0.0f, 0.0f, 0.0f, axis_len, 0.2f, 0.2f, 1.0f);

  result.axes.push_back({axis_len, 0.0f, 0.0f, "X"});
  result.axes.push_back({0.0f, axis_len, 0.0f, "Y"});
  result.axes.push_back({0.0f, 0.0f, axis_len, "Z"});
  
  return result;
}

}  // namespace GridBuilder
