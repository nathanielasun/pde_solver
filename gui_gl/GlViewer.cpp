#include "GlViewer.h"
#include "vtk_io.h"

#include "rendering/shader_manager.h"
#include "rendering/mesh_builder.h"
#include "rendering/grid_builder.h"
#include "rendering/projection.h"
#include "rendering/renderer.h"
#include "rendering/render_types.h"

#include <algorithm>
#include <cmath>
#include <vector>

#include <OpenGL/gl3.h>

using Vertex = ::Vertex;  // Use Vertex from render_types.h
using RenderMath::kPi;     // Use kPi from render_types.h

bool GlViewer::Init() {
  if (!CompileShaders()) {
    return false;
  }
  glGenVertexArrays(1, &vao_);
  glGenBuffers(1, &vbo_);
  glBindVertexArray(vao_);
  glBindBuffer(GL_ARRAY_BUFFER, vbo_);
  glBufferData(GL_ARRAY_BUFFER, 0, nullptr, GL_DYNAMIC_DRAW);

  glEnableVertexAttribArray(0);
  glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex),
                        reinterpret_cast<void*>(0));
  glEnableVertexAttribArray(1);
  glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex),
                        reinterpret_cast<void*>(3 * sizeof(float)));
  glBindBuffer(GL_ARRAY_BUFFER, 0);
  glBindVertexArray(0);

  glGenVertexArrays(1, &grid_vao_);
  glGenBuffers(1, &grid_vbo_);
  glBindVertexArray(grid_vao_);
  glBindBuffer(GL_ARRAY_BUFFER, grid_vbo_);
  glBufferData(GL_ARRAY_BUFFER, 0, nullptr, GL_DYNAMIC_DRAW);
  glEnableVertexAttribArray(0);
  glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex),
                        reinterpret_cast<void*>(0));
  glEnableVertexAttribArray(1);
  glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex),
                        reinterpret_cast<void*>(3 * sizeof(float)));
  glBindBuffer(GL_ARRAY_BUFFER, 0);
  glBindVertexArray(0);
  return true;
}

void GlViewer::Shutdown() {
  if (program_) {
    glDeleteProgram(program_);
    program_ = 0;
  }
  if (vbo_) {
    glDeleteBuffers(1, &vbo_);
    vbo_ = 0;
  }
  if (vao_) {
    glDeleteVertexArrays(1, &vao_);
    vao_ = 0;
  }
  if (grid_vbo_) {
    glDeleteBuffers(1, &grid_vbo_);
    grid_vbo_ = 0;
  }
  if (grid_vao_) {
    glDeleteVertexArrays(1, &grid_vao_);
    grid_vao_ = 0;
  }
  
  // Use Renderer::DestroyFbo for FBO cleanup
  Renderer::FboState fbo_state;
  fbo_state.fbo = fbo_;
  fbo_state.color_tex = color_tex_;
  fbo_state.depth_rb = depth_rb_;
  Renderer::DestroyFbo(&fbo_state);
  fbo_ = 0;
  color_tex_ = 0;
    depth_rb_ = 0;
}

bool GlViewer::CompileShaders() {
  program_ = ShaderManager::CreateDefaultProgram();
  return program_ != 0;
}

void GlViewer::SetClearColor(float r, float g, float b) {
  clear_color_[0] = r;
  clear_color_[1] = g;
  clear_color_[2] = b;
}

void GlViewer::SetStride(int stride) {
  const int next_stride = std::max(1, stride);
  if (next_stride != stride_) {
    stride_ = next_stride;
    data_dirty_ = true;
  }
}

void GlViewer::SetPointScale(float scale) {
  const float next_scale = std::max(0.1f, std::min(50.0f, scale));
  if (std::abs(next_scale - point_scale_) > 1e-4f) {
    point_scale_ = next_scale;
  }
}

void GlViewer::SetDataScale(float scale) {
  const float next_scale = std::max(0.01f, std::min(10.0f, scale));
  if (std::abs(next_scale - data_scale_) > 1e-4f) {
    data_scale_ = next_scale;
    data_dirty_ = true;
    grid_dirty_ = true;
  }
}

void GlViewer::SetGridEnabled(bool enabled) {
  if (grid_enabled_ != enabled) {
    grid_enabled_ = enabled;
    grid_dirty_ = true;
  }
}

void GlViewer::SetGridDivisions(int divisions) {
  const int next_div = std::max(2, std::min(32, divisions));
  if (grid_divisions_ != next_div) {
    grid_divisions_ = next_div;
    grid_dirty_ = true;
  }
}

void GlViewer::SetOrthographic(bool enabled) {
  use_ortho_ = enabled;
}

void GlViewer::SetZDomain(bool locked, double z_min, double z_max) {
  z_domain_locked_ = locked;
  if (z_min > z_max) {
    std::swap(z_min, z_max);
  }
  z_domain_min_ = z_min;
  z_domain_max_ = z_max;
  data_dirty_ = true;
}

void GlViewer::SetViewMode(ViewMode mode) {
  if (view_mode_ != mode) {
    view_mode_ = mode;
    data_dirty_ = true;
    grid_dirty_ = true;
  }
}

void GlViewer::SetTorusRadii(float major_radius, float minor_radius) {
  if (major_radius <= 0.0f) {
    major_radius = 1.0f;
  }
  if (minor_radius <= 0.0f) {
    minor_radius = 0.1f;
  }
  torus_major_ = major_radius;
  torus_minor_ = minor_radius;
  data_dirty_ = true;
  grid_dirty_ = true;
}

void GlViewer::SetData(const Domain& domain, const std::vector<double>& grid) {
  domain_ = domain;
  grid_ = grid;
  point_cloud_.clear();
  has_point_cloud_ = false;
  has_data_ = !grid_.empty();
  data_dirty_ = true;
  grid_dirty_ = true;
  // Clear multi-field state when using single-field SetData
  field_grids_.clear();
  field_names_.clear();
  active_field_index_ = 0;
}

void GlViewer::SetMultiFieldData(const Domain& domain,
                                  const std::vector<std::vector<double>>& field_grids,
                                  const std::vector<std::string>& field_names) {
  domain_ = domain;
  field_grids_ = field_grids;
  field_names_ = field_names;
  active_field_index_ = 0;
  point_cloud_.clear();
  has_point_cloud_ = false;

  // Set grid_ to the first field for visualization
  if (!field_grids_.empty()) {
    grid_ = field_grids_[0];
    has_data_ = !grid_.empty();
  } else {
    grid_.clear();
    has_data_ = false;
  }
  data_dirty_ = true;
  grid_dirty_ = true;
}

void GlViewer::SetActiveField(int field_index) {
  if (field_grids_.empty()) {
    return;
  }
  const int clamped_index = std::max(0, std::min(field_index,
                                                  static_cast<int>(field_grids_.size()) - 1));
  if (clamped_index != active_field_index_) {
    active_field_index_ = clamped_index;
    grid_ = field_grids_[static_cast<size_t>(active_field_index_)];
    data_dirty_ = true;
  }
}

void GlViewer::SetDerivedFields(const struct DerivedFields& fields) {
  if (!derived_fields_) {
    derived_fields_ = std::make_unique<struct DerivedFields>();
  }
  *derived_fields_ = fields;
  has_derived_fields_ = true;
  data_dirty_ = true;  // Rebuild mesh with new field
}

void GlViewer::ClearDerivedFields() {
  derived_fields_.reset();
  has_derived_fields_ = false;
  data_dirty_ = true;
}

void GlViewer::SetFieldType(FieldType field_type) {
  if (field_type_ != field_type) {
    field_type_ = field_type;
    data_dirty_ = true;  // Rebuild mesh with new field type
  }
}

namespace {
// Helper function to get field value based on field type
double GetFieldValue(size_t idx, GlViewer::FieldType field_type,
                    const std::vector<double>& grid,
                    const struct DerivedFields* derived,
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
}

void GlViewer::SetPointCloud(const Domain& domain, const std::vector<PointSample>& points) {
  domain_ = domain;
  point_cloud_ = points;
  grid_.clear();
  has_point_cloud_ = !point_cloud_.empty();
  has_data_ = has_point_cloud_;
  data_dirty_ = true;
  grid_dirty_ = true;
}

void GlViewer::SetSlice(bool enabled, int axis, double value, double thickness) {
  const int clamped_axis = std::max(0, std::min(2, axis));
  const bool changed =
      slice_enabled_ != enabled || slice_axis_ != clamped_axis ||
      std::abs(slice_value_ - value) > 1e-12 ||
      std::abs(slice_thickness_ - thickness) > 1e-12;
  if (!changed) {
    return;
  }
  slice_enabled_ = enabled;
  slice_axis_ = clamped_axis;
  slice_value_ = value;
  slice_thickness_ = thickness;
  data_dirty_ = true;
  grid_dirty_ = true;
}

void GlViewer::SetIsosurface(bool enabled, double value, double band) {
  const bool changed =
      iso_enabled_ != enabled || std::abs(iso_value_ - value) > 1e-12 ||
      std::abs(iso_band_ - band) > 1e-12;
  if (!changed) {
    return;
  }
  iso_enabled_ = enabled;
  iso_value_ = value;
  iso_band_ = band;
  data_dirty_ = true;
}

void GlViewer::FitToView() {
  if (!has_data_) {
    distance_ = 3.6f;
    yaw_ = 0.6f;
    pitch_ = -0.4f;
    return;
  }
  double span_x = domain_.xmax - domain_.xmin;
  double span_y = domain_.ymax - domain_.ymin;
  double span = std::max(span_x, span_y);
  const bool is_polar = view_mode_ == ViewMode::Polar;
  const bool is_axisymmetric = view_mode_ == ViewMode::Axisymmetric;
  const bool is_point_cloud = has_point_cloud_;
  const bool is_cylindrical = view_mode_ == ViewMode::CylindricalVolume;
  const bool is_spherical_surface = view_mode_ == ViewMode::SphericalSurface;
  const bool is_spherical_volume = view_mode_ == ViewMode::SphericalVolume;
  const bool is_toroidal_surface = view_mode_ == ViewMode::ToroidalSurface;
  const bool is_toroidal_volume = view_mode_ == ViewMode::ToroidalVolume;
  const bool is_cartesian = is_point_cloud || view_mode_ == ViewMode::Cartesian || is_axisymmetric;
  const bool has_z = domain_.nz > 1;
  if (has_z && is_cartesian) {
    span = std::max(span, domain_.zmax - domain_.zmin);
  }
  if (!is_point_cloud) {
    if (is_polar || is_spherical_volume) {
      const double max_r = std::max(std::abs(domain_.xmin), std::abs(domain_.xmax));
      span = std::max(1e-12, 2.0 * max_r);
      span_x = span;
      span_y = span;
    } else if (is_spherical_surface) {
      span = 2.0;
      span_x = span;
      span_y = span;
    } else if (is_cylindrical) {
      const double max_r = std::max(std::abs(domain_.xmin), std::abs(domain_.xmax));
      span = std::max(1e-12, 2.0 * max_r);
      span_x = span;
      span_y = span;
    } else if (is_toroidal_surface || is_toroidal_volume) {
      const double r_max = is_toroidal_surface
                               ? torus_minor_
                               : std::max(std::abs(domain_.xmin), std::abs(domain_.xmax));
      span = std::max(1e-12, 2.0 * (torus_major_ + r_max));
      span_x = span;
      span_y = span;
    }
  }
  const float scale = span > 0.0 ? static_cast<float>((1.2 * data_scale_) / span) : data_scale_;
  const float rx = static_cast<float>(span_x * scale * 0.5);
  const float ry = static_cast<float>(span_y * scale * 0.5);
  const float rz = 0.3f * data_scale_;
  if (vertex_count_ == 0) {
    scene_center_x_ = 0.0f;
    scene_center_y_ = 0.0f;
    scene_center_z_ = 0.0f;
    scene_half_x_ = std::max(0.01f, rx);
    scene_half_y_ = std::max(0.01f, ry);
    scene_half_z_ = std::max(0.01f, rz);
    scene_radius_ = std::max(0.05f, std::sqrt(rx * rx + ry * ry + rz * rz));
  }
  const float fov = kPi / 3.5f;
  const float tan_half = std::tan(fov * 0.5f);
  const float aspect = last_aspect_ > 0.1f ? last_aspect_ : 1.0f;
  const float req_y = ry / tan_half;
  const float req_x = rx / (tan_half * aspect);
  const float req = std::max(req_x, req_y);
  const float target = (req + rz) * 1.15f;
  fit_distance_ = std::max(0.8f, std::min(500.0f, target));
  if (use_ortho_) {
    distance_ = std::max(2.0f, scene_radius_ * 2.5f);
  } else {
    distance_ = fit_distance_;
  }
  point_scale_ = 1.0f;
  yaw_ = 0.6f;
  pitch_ = -0.4f;
}

void GlViewer::SetOrientation(float yaw, float pitch) {
  yaw_ = yaw;
  pitch_ = std::max(-1.4f, std::min(1.4f, pitch));
}

void GlViewer::GetOrientation(float* yaw, float* pitch) const {
  if (yaw) {
    *yaw = yaw_;
  }
  if (pitch) {
    *pitch = pitch_;
  }
}

void GlViewer::Rotate(float dx, float dy) {
  yaw_ += dx * 0.01f;
  pitch_ += dy * 0.01f;
  pitch_ = std::max(-1.4f, std::min(1.4f, pitch_));
}

void GlViewer::Zoom(float delta) {
  const float scaled = 1.0f - static_cast<float>(delta) * 0.05f;
  const float clamped = std::max(0.2f, std::min(2.0f, scaled));
  if (use_ortho_) {
    point_scale_ *= clamped;
    point_scale_ = std::max(0.1f, std::min(50.0f, point_scale_));
  } else {
    distance_ *= clamped;
    distance_ = std::max(0.5f, std::min(500.0f, distance_));
  }
}

std::vector<GlViewer::ScreenLabel> GlViewer::AxisLabels() const {
  std::vector<ScreenLabel> labels;
  if (!has_data_ || !grid_enabled_ || !has_mvp_ || tex_width_ <= 0 || tex_height_ <= 0) {
    return labels;
  }

  const bool has_z = domain_.nz > 1;
  const bool is_spherical_surface = view_mode_ == ViewMode::SphericalSurface;
  const bool is_spherical_volume = view_mode_ == ViewMode::SphericalVolume;
  const bool is_toroidal_surface = view_mode_ == ViewMode::ToroidalSurface;
  const bool is_toroidal_volume = view_mode_ == ViewMode::ToroidalVolume;
  const bool is_cylindrical = view_mode_ == ViewMode::CylindricalVolume;
  const bool use_value_height =
      !has_z && !is_spherical_surface && !is_spherical_volume &&
      !is_toroidal_surface && !is_toroidal_volume && !is_cylindrical;

  Projection::AxisLabelParams params;
  params.domain = &domain_;
  params.view_mode = view_mode_;
  params.grid_divisions = grid_divisions_;
  params.data_scale = data_scale_;
  params.torus_major = torus_major_;
  params.torus_minor = torus_minor_;
  params.has_point_cloud = has_point_cloud_;
  params.use_value_height = use_value_height;
  params.value_min = value_min_;
  params.value_max = value_max_;
  params.value_range = value_range_;
  params.mvp = last_mvp_;  // Current MVP matrix
  params.tex_width = tex_width_;
  params.tex_height = tex_height_;

  return Projection::GenerateAxisLabels(params);
}

void GlViewer::EnsureFbo(int width, int height) {
  Renderer::FboState fbo_state;
  fbo_state.fbo = fbo_;
  fbo_state.color_tex = color_tex_;
  fbo_state.depth_rb = depth_rb_;
  fbo_state.width = tex_width_;
  fbo_state.height = tex_height_;

  Renderer::EnsureFbo(&fbo_state, width, height);

  // Update member variables from FboState
  fbo_ = fbo_state.fbo;
  color_tex_ = fbo_state.color_tex;
  depth_rb_ = fbo_state.depth_rb;
  tex_width_ = fbo_state.width;
  tex_height_ = fbo_state.height;
}

void GlViewer::BuildMesh() {
  data_dirty_ = false;
  if (!has_data_) {
    vertex_count_ = 0;
    return;
  }

  MeshBuildParams params;
  params.domain = &domain_;
  params.grid = &grid_;
  params.point_cloud = &point_cloud_;
  params.derived_fields = derived_fields_.get();
  params.has_point_cloud = has_point_cloud_;
  params.has_derived_fields = has_derived_fields_;
  params.field_type = field_type_;
  params.view_mode = view_mode_;
  params.data_scale = data_scale_;
  params.torus_major = torus_major_;
  params.torus_minor = torus_minor_;
  params.stride = stride_;
  params.slice_enabled = slice_enabled_;
  params.slice_axis = slice_axis_;
  params.slice_value = slice_value_;
  params.slice_thickness = slice_thickness_;
  params.iso_enabled = iso_enabled_;
  params.iso_value = iso_value_;
  params.iso_band = iso_band_;
  params.z_domain_locked = z_domain_locked_;
  params.z_domain_min = z_domain_min_;
  params.z_domain_max = z_domain_max_;

  MeshBuildResult result = MeshBuilder::BuildMesh(params);

  value_min_ = result.value_min;
  value_max_ = result.value_max;
  value_range_ = result.value_range;
  scene_center_x_ = result.scene_center_x;
  scene_center_y_ = result.scene_center_y;
  scene_center_z_ = result.scene_center_z;
  scene_half_x_ = result.scene_half_x;
  scene_half_y_ = result.scene_half_y;
  scene_half_z_ = result.scene_half_z;
  scene_radius_ = result.scene_radius;

  vertex_count_ = static_cast<int>(result.vertices.size());
  if (vertex_count_ > 0) {
    glBindBuffer(GL_ARRAY_BUFFER, vbo_);
    glBufferData(GL_ARRAY_BUFFER, result.vertices.size() * sizeof(Vertex),
                 result.vertices.data(), GL_STATIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
  }
}

void GlViewer::BuildGrid() {
  grid_dirty_ = false;
  grid_vertex_count_ = 0;
  if (!grid_enabled_) {
    return;
  }

  GridBuildParams params;
  params.domain = &domain_;
  params.view_mode = view_mode_;
  params.has_point_cloud = has_point_cloud_;
  params.data_scale = data_scale_;
  params.torus_major = torus_major_;
  params.torus_minor = torus_minor_;
  params.grid_divisions = grid_divisions_;
  params.grid_enabled = grid_enabled_;
  params.slice_enabled = slice_enabled_;
  params.slice_axis = slice_axis_;
  params.slice_value = slice_value_;
  params.slice_thickness = slice_thickness_;

  grid_build_result_ = GridBuilder::BuildGrid(params);

  // Update scene bounds from grid result
  scene_half_x_ = grid_build_result_.scene_half_x;
  scene_half_y_ = grid_build_result_.scene_half_y;
  scene_half_z_ = grid_build_result_.scene_half_z;
  scene_radius_ = grid_build_result_.scene_radius;

  grid_vertex_count_ = static_cast<int>(grid_build_result_.vertices.size());
  if (grid_vertex_count_ > 0) {
    glBindBuffer(GL_ARRAY_BUFFER, grid_vbo_);
    glBufferData(GL_ARRAY_BUFFER, grid_build_result_.vertices.size() * sizeof(Vertex),
                 grid_build_result_.vertices.data(), GL_STATIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
  }
}

// Old BuildGrid implementation removed - now using GridBuilder::BuildGrid()

void GlViewer::RenderToTexture(int width, int height) {
  EnsureFbo(width, height);
  if (data_dirty_) {
    BuildMesh();
  }
  if (grid_dirty_) {
    BuildGrid();
  }

  // Create FboState from member variables
  Renderer::FboState fbo_state;
  fbo_state.fbo = fbo_;
  fbo_state.color_tex = color_tex_;
  fbo_state.depth_rb = depth_rb_;
  fbo_state.width = tex_width_;
  fbo_state.height = tex_height_;

  // Create RenderParams from member variables
  Renderer::RenderParams params;
  params.clear_color = clear_color_;
  params.program = program_;
  params.mesh_vao = vao_;
  params.grid_vao = grid_vao_;
  params.mesh_vertex_count = vertex_count_;
  params.grid_vertex_count = grid_vertex_count_;
  params.grid_enabled = grid_enabled_;
  params.grid_build_result = &grid_build_result_;
  params.yaw = yaw_;
  params.pitch = pitch_;
  params.distance = distance_;
  params.point_scale = point_scale_;
  params.use_ortho = use_ortho_;
  params.scene_center_x = scene_center_x_;
  params.scene_center_y = scene_center_y_;
  params.scene_center_z = scene_center_z_;
  params.scene_radius = scene_radius_;
  params.mvp_out = last_mvp_;
  params.last_aspect_out = &last_aspect_;

  Renderer::RenderToTexture(fbo_state, params, width, height);

  // Update member variables from FboState (in case they changed)
  fbo_ = fbo_state.fbo;
  color_tex_ = fbo_state.color_tex;
  depth_rb_ = fbo_state.depth_rb;
  tex_width_ = fbo_state.width;
  tex_height_ = fbo_state.height;
  
  has_mvp_ = true;
}

bool GlViewer::ProjectPoint(float x, float y, float z, float* out_x, float* out_y) const {
  if (!has_mvp_ || tex_width_ <= 0 || tex_height_ <= 0) {
    return false;
  }
  return Projection::ProjectPoint(last_mvp_, tex_width_, tex_height_, x, y, z, out_x, out_y);
}
