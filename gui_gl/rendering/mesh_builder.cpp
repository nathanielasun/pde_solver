#include "mesh_builder.h"
#include "coordinate_transforms.h"
#include "vtk_io.h"
#include <algorithm>
#include <cmath>
#include <limits>

namespace MeshBuilder {

double GetFieldValue(size_t idx, GlViewer::FieldType field_type,
                    const std::vector<double>& grid,
                    const DerivedFields* derived,
                    bool has_derived) {
  if (field_type == GlViewer::FieldType::Solution || !has_derived || !derived) {
    return idx < grid.size() ? grid[idx] : 0.0;
  }
  
  switch (field_type) {
    case GlViewer::FieldType::GradientX:
      return idx < derived->gradient_x.size() ? derived->gradient_x[idx] : 0.0;
    case GlViewer::FieldType::GradientY:
      return idx < derived->gradient_y.size() ? derived->gradient_y[idx] : 0.0;
    case GlViewer::FieldType::GradientZ:
      return idx < derived->gradient_z.size() ? derived->gradient_z[idx] : 0.0;
    case GlViewer::FieldType::Laplacian:
      return idx < derived->laplacian.size() ? derived->laplacian[idx] : 0.0;
    case GlViewer::FieldType::FluxX:
      return idx < derived->flux_x.size() ? derived->flux_x[idx] : 0.0;
    case GlViewer::FieldType::FluxY:
      return idx < derived->flux_y.size() ? derived->flux_y[idx] : 0.0;
    case GlViewer::FieldType::FluxZ:
      return idx < derived->flux_z.size() ? derived->flux_z[idx] : 0.0;
    case GlViewer::FieldType::EnergyNorm:
      return idx < derived->energy_norm.size() ? derived->energy_norm[idx] : 0.0;
    default:
      return idx < grid.size() ? grid[idx] : 0.0;
  }
}

void ValueToColor(float t, float* r, float* g, float* b) {
  *r = std::min(1.0f, 1.4f * t);
  *g = 0.25f + 0.65f * (1.0f - std::abs(2.0f * t - 1.0f));
  *b = std::min(1.0f, 1.4f * (1.0f - t));
}

MeshBuildResult BuildPointCloudMesh(const MeshBuildParams& params) {
  MeshBuildResult result;
  
  if (!params.point_cloud || params.point_cloud->empty() || !params.domain) {
    return result;
  }
  
  const auto& point_cloud = *params.point_cloud;
  const auto& domain = *params.domain;
  
  // Compute scale
  const double span_x = domain.xmax - domain.xmin;
  const double span_y = domain.ymax - domain.ymin;
  const double span_z = domain.zmax - domain.zmin;
  double span = std::max(span_x, span_y);
  if (domain.nz > 1) {
    span = std::max(span, span_z);
  }
  const float scale = span > 0.0 ? static_cast<float>((1.2 * params.data_scale) / span)
                                  : params.data_scale;
  const float cx = static_cast<float>((domain.xmin + domain.xmax) * 0.5);
  const float cy = static_cast<float>((domain.ymin + domain.ymax) * 0.5);
  const float cz = static_cast<float>((domain.zmin + domain.zmax) * 0.5);
  
  // Find value range
  double min_val = point_cloud.front().value;
  double max_val = point_cloud.front().value;
  for (const auto& pt : point_cloud) {
    min_val = std::min(min_val, pt.value);
    max_val = std::max(max_val, pt.value);
  }
  result.value_min = min_val;
  result.value_max = max_val;
  result.value_range = std::max(1e-12, result.value_max - result.value_min);
  
  // Compute slice/isosurface parameters
  double iso_band = params.iso_band;
  if (iso_band <= 0.0) {
    iso_band = result.value_range * 0.02;
  }
  const double iso_half = iso_band * 0.5;
  
  double axis_min = domain.xmin;
  double axis_max = domain.xmax;
  if (params.slice_axis == 1) {
    axis_min = domain.ymin;
    axis_max = domain.ymax;
  } else if (params.slice_axis == 2) {
    axis_min = domain.zmin;
    axis_max = domain.zmax;
  }
  double axis_span = axis_max - axis_min;
  if (axis_span <= 0.0) {
    axis_span = 1.0;
  }
  double slice_thickness = params.slice_thickness;
  if (slice_thickness <= 0.0) {
    slice_thickness = axis_span * 0.01;
  }
  const double slice_half = slice_thickness * 0.5;
  
  // Build vertices
  result.vertices.reserve(point_cloud.size() / static_cast<size_t>(std::max(1, params.stride)) + 1);
  
  float min_x = std::numeric_limits<float>::max();
  float min_y = std::numeric_limits<float>::max();
  float min_z = std::numeric_limits<float>::max();
  float max_x = std::numeric_limits<float>::lowest();
  float max_y = std::numeric_limits<float>::lowest();
  float max_z = std::numeric_limits<float>::lowest();
  
  const size_t step = static_cast<size_t>(std::max(1, params.stride));
  for (size_t idx = 0; idx < point_cloud.size(); idx += step) {
    const auto& pt = point_cloud[idx];
    
    // Apply slice filter
    if (params.slice_enabled) {
      const double coord = (params.slice_axis == 0) ? pt.x : 
                          (params.slice_axis == 1 ? pt.y : pt.z);
      if (std::abs(coord - params.slice_value) > slice_half) {
        continue;
      }
    }
    
    // Apply isosurface filter
    if (params.iso_enabled) {
      if (std::abs(pt.value - params.iso_value) > iso_half) {
        continue;
      }
    }
    
    // Transform to Cartesian (point clouds are always Cartesian)
    const float rxp = (static_cast<float>(pt.x) - cx) * scale;
    const float ryp = (static_cast<float>(pt.y) - cy) * scale;
    const float rzp = (static_cast<float>(pt.z) - cz) * scale;
    
    // Compute color
    const float t_color = static_cast<float>(
        (pt.value - min_val) / std::max(1e-12, max_val - min_val));
    float r, g, b;
    ValueToColor(t_color, &r, &g, &b);
    
    result.vertices.push_back({rxp, ryp, rzp, r, g, b});
    
    min_x = std::min(min_x, rxp);
    min_y = std::min(min_y, ryp);
    min_z = std::min(min_z, rzp);
    max_x = std::max(max_x, rxp);
    max_y = std::max(max_y, ryp);
    max_z = std::max(max_z, rzp);
  }
  
  // Compute scene bounds
  if (!result.vertices.empty()) {
    result.scene_center_x = (min_x + max_x) * 0.5f;
    result.scene_center_y = (min_y + max_y) * 0.5f;
    result.scene_center_z = (min_z + max_z) * 0.5f;
    result.scene_half_x = std::max(0.01f, (max_x - min_x) * 0.5f);
    result.scene_half_y = std::max(0.01f, (max_y - min_y) * 0.5f);
    result.scene_half_z = std::max(0.01f, (max_z - min_z) * 0.5f);
    result.scene_radius = std::max(
        0.05f,
        std::sqrt(result.scene_half_x * result.scene_half_x +
                  result.scene_half_y * result.scene_half_y +
                  result.scene_half_z * result.scene_half_z));
  }
  
  return result;
}

MeshBuildResult BuildGridMesh(const MeshBuildParams& params) {
  MeshBuildResult result;
  
  if (!params.grid || params.grid->empty() || !params.domain) {
    return result;
  }
  
  const auto& grid = *params.grid;
  const auto& domain = *params.domain;
  
  const int nx = domain.nx;
  const int ny = domain.ny;
  const int nz = std::max(1, domain.nz);
  if (nx <= 0 || ny <= 0 || nz <= 0) {
    return result;
  }
  
  // Compute grid spacing
  const double dx = (domain.xmax - domain.xmin) / std::max(1, nx - 1);
  const double dy = (domain.ymax - domain.ymin) / std::max(1, ny - 1);
  const double dz = (domain.zmax - domain.zmin) / std::max(1, nz - 1);
  
  // Compute span and scale
  const double span_x = domain.xmax - domain.xmin;
  const double span_y = domain.ymax - domain.ymin;
  const double span_z = domain.zmax - domain.zmin;
  double span = std::max(span_x, span_y);
  const bool has_z = nz > 1;
  
  const bool is_polar = params.view_mode == GlViewer::ViewMode::Polar;
  const bool is_axisymmetric = params.view_mode == GlViewer::ViewMode::Axisymmetric;
  const bool is_cylindrical = params.view_mode == GlViewer::ViewMode::CylindricalVolume;
  const bool is_spherical_surface = params.view_mode == GlViewer::ViewMode::SphericalSurface;
  const bool is_spherical_volume = params.view_mode == GlViewer::ViewMode::SphericalVolume;
  const bool is_toroidal_surface = params.view_mode == GlViewer::ViewMode::ToroidalSurface;
  const bool is_toroidal_volume = params.view_mode == GlViewer::ViewMode::ToroidalVolume;
  const bool use_value_height =
      !has_z && !is_spherical_surface && !is_spherical_volume &&
      !is_toroidal_surface && !is_toroidal_volume && !is_cylindrical;
  const bool is_cartesian = params.view_mode == GlViewer::ViewMode::Cartesian || is_axisymmetric;
  
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
  
  // Get field data
  const std::vector<double>* field_data = params.grid;
  if (params.field_type != GlViewer::FieldType::Solution && 
      params.has_derived_fields && params.derived_fields) {
    switch (params.field_type) {
      case GlViewer::FieldType::GradientX:
        field_data = &params.derived_fields->gradient_x;
        break;
      case GlViewer::FieldType::GradientY:
        field_data = &params.derived_fields->gradient_y;
        break;
      case GlViewer::FieldType::GradientZ:
        field_data = &params.derived_fields->gradient_z;
        break;
      case GlViewer::FieldType::Laplacian:
        field_data = &params.derived_fields->laplacian;
        break;
      case GlViewer::FieldType::FluxX:
        field_data = &params.derived_fields->flux_x;
        break;
      case GlViewer::FieldType::FluxY:
        field_data = &params.derived_fields->flux_y;
        break;
      case GlViewer::FieldType::FluxZ:
        field_data = &params.derived_fields->flux_z;
        break;
      case GlViewer::FieldType::EnergyNorm:
        field_data = &params.derived_fields->energy_norm;
        break;
      default:
        field_data = params.grid;
        break;
    }
  }
  
  if (field_data->empty()) {
    field_data = params.grid;
  }
  
  // Find value range
  double min_val = (*field_data)[0];
  double max_val = (*field_data)[0];
  for (double v : *field_data) {
    min_val = std::min(min_val, v);
    max_val = std::max(max_val, v);
  }
  
  double z_domain_min = min_val;
  double z_domain_max = max_val;
  if (use_value_height && params.z_domain_locked && 
      params.z_domain_max > params.z_domain_min + 1e-12) {
    z_domain_min = params.z_domain_min;
    z_domain_max = params.z_domain_max;
  }
  result.value_min = z_domain_min;
  result.value_max = z_domain_max;
  result.value_range = std::max(1e-12, result.value_max - result.value_min);
  const double denom = std::max(1e-12, max_val - min_val);
  const double denom_z = std::max(1e-12, z_domain_max - z_domain_min);
  
  // Compute filter parameters
  double iso_band = params.iso_band;
  if (iso_band <= 0.0) {
    iso_band = result.value_range * 0.02;
  }
  const double iso_half = iso_band * 0.5;
  
  double axis_min = domain.xmin;
  double axis_max = domain.xmax;
  if (params.slice_axis == 1) {
    axis_min = domain.ymin;
    axis_max = domain.ymax;
  } else if (params.slice_axis == 2) {
    if (use_value_height) {
      axis_min = z_domain_min;
      axis_max = z_domain_max;
    } else {
      axis_min = domain.zmin;
      axis_max = domain.zmax;
    }
  }
  double axis_span = axis_max - axis_min;
  if (axis_span <= 0.0) {
    axis_span = 1.0;
  }
  double slice_thickness = params.slice_thickness;
  if (slice_thickness <= 0.0) {
    slice_thickness = axis_span * 0.01;
  }
  const double slice_half = slice_thickness * 0.5;
  
  // Build vertices
  result.vertices.reserve(static_cast<size_t>((nx / params.stride + 1) * 
                                              (ny / params.stride + 1) *
                                              (nz / params.stride + 1)));
  
  float min_x = std::numeric_limits<float>::max();
  float min_y = std::numeric_limits<float>::max();
  float min_z = std::numeric_limits<float>::max();
  float max_x = std::numeric_limits<float>::lowest();
  float max_y = std::numeric_limits<float>::lowest();
  float max_z = std::numeric_limits<float>::lowest();
  
  const float base_radius = is_spherical_surface ? scale : 0.0f;
  const float radial_span = 0.4f * params.data_scale;
  const float z_min = -0.3f * params.data_scale;
  const float z_max = 0.3f * params.data_scale;
  
  for (int k = 0; k < nz; k += params.stride) {
    const float z_val = static_cast<float>(domain.zmin + k * dz);
    for (int j = 0; j < ny; j += params.stride) {
      const float y = static_cast<float>(domain.ymin + j * dy);
      for (int i = 0; i < nx; i += params.stride) {
        const float x = static_cast<float>(domain.xmin + i * dx);
        const size_t idx = static_cast<size_t>((k * ny + j) * nx + i);
        const double value = GetFieldValue(idx, params.field_type, grid,
                                          params.derived_fields, params.has_derived_fields);
        
        // Apply slice filter
        if (params.slice_enabled) {
          double coord = x;
          if (params.slice_axis == 1) {
            coord = y;
          } else if (params.slice_axis == 2) {
            coord = use_value_height ? value : z_val;
          }
          if (std::abs(coord - params.slice_value) > slice_half) {
            continue;
          }
        }
        
        // Apply isosurface filter
        if (params.iso_enabled) {
          if (std::abs(value - params.iso_value) > iso_half) {
            continue;
          }
        }
        
        // Compute color
        const float t_color = static_cast<float>((value - min_val) / denom);
        float t_z = static_cast<float>((value - z_domain_min) / denom_z);
        if (t_z < 0.0f) {
          t_z = 0.0f;
        } else if (t_z > 1.0f) {
          t_z = 1.0f;
        }
        const float z_value = z_min + t_z * (z_max - z_min);
        
        // Transform coordinates
        float rxp, ryp, rzp;
        CoordinateTransforms::TransformToCartesian(
            x, y, z_val, params.view_mode, scale,
            cx, cy, cz, params.torus_major, params.torus_minor,
            base_radius, radial_span, t_z, use_value_height, has_z,
            &rxp, &ryp, &rzp);
        
        // Override z for value height if needed
        if (use_value_height && !has_z && 
            params.view_mode == GlViewer::ViewMode::Cartesian) {
          rzp = z_value;
        }
        
        // Compute color
        float r, g, b;
        ValueToColor(t_color, &r, &g, &b);
        
        result.vertices.push_back({rxp, ryp, rzp, r, g, b});
        
        min_x = std::min(min_x, rxp);
        min_y = std::min(min_y, ryp);
        min_z = std::min(min_z, rzp);
        max_x = std::max(max_x, rxp);
        max_y = std::max(max_y, ryp);
        max_z = std::max(max_z, rzp);
      }
    }
  }
  
  // Compute scene bounds
  if (!result.vertices.empty()) {
    result.scene_center_x = (min_x + max_x) * 0.5f;
    result.scene_center_y = (min_y + max_y) * 0.5f;
    result.scene_center_z = (min_z + max_z) * 0.5f;
    result.scene_half_x = std::max(0.01f, (max_x - min_x) * 0.5f);
    result.scene_half_y = std::max(0.01f, (max_y - min_y) * 0.5f);
    result.scene_half_z = std::max(0.01f, (max_z - min_z) * 0.5f);
    result.scene_radius = std::max(
        0.05f,
        std::sqrt(result.scene_half_x * result.scene_half_x +
                  result.scene_half_y * result.scene_half_y +
                  result.scene_half_z * result.scene_half_z));
  }
  
  return result;
}

MeshBuildResult BuildMesh(const MeshBuildParams& params) {
  if (params.has_point_cloud && params.point_cloud && !params.point_cloud->empty()) {
    return BuildPointCloudMesh(params);
  } else {
    return BuildGridMesh(params);
  }
}

}  // namespace MeshBuilder

