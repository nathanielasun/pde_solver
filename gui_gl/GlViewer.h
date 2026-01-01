#ifndef GL_VIEWER_H
#define GL_VIEWER_H

#include <string>
#include <vector>
#include <memory>

#include "pde_types.h"
#include "rendering/view_mode.h"
#include "rendering/grid_builder.h"

// Forward declarations
struct DerivedFields;

class GlViewer {
 public:
  // Re-export ViewMode for backward compatibility
  using ViewMode = ::ViewMode;
  
  enum class FieldType {
    Solution,      // The solution field u
    GradientX,     // ∂u/∂x
    GradientY,     // ∂u/∂y
    GradientZ,     // ∂u/∂z (3D only)
    Laplacian,     // ∇²u
    FluxX,         // -a*∂u/∂x
    FluxY,         // -b*∂u/∂y
    FluxZ,         // -c*∂u/∂z (3D only)
    EnergyNorm     // u²
  };

  struct ScreenLabel {
    float x = 0.0f;
    float y = 0.0f;
    std::string text;
    bool align_right = false;
  };

  bool Init();
  void Shutdown();

  void SetClearColor(float r, float g, float b);
  void SetStride(int stride);
  void SetPointScale(float scale);
  void SetDataScale(float scale);
  void SetGridEnabled(bool enabled);
  void SetGridDivisions(int divisions);
  void SetOrthographic(bool enabled);
  void SetZDomain(bool locked, double z_min, double z_max);
  void SetViewMode(ViewMode mode);
  void SetTorusRadii(float major_radius, float minor_radius);
  void SetData(const Domain& domain, const std::vector<double>& grid);
  void SetDerivedFields(const struct DerivedFields& fields);
  void ClearDerivedFields();
  void SetFieldType(FieldType field_type);
  FieldType GetFieldType() const { return field_type_; }
  void SetPointCloud(const Domain& domain, const std::vector<PointSample>& points);
  void SetSlice(bool enabled, int axis, double value, double thickness);
  void SetIsosurface(bool enabled, double value, double band);
  const Domain& domain() const { return domain_; }
  double value_min() const { return value_min_; }
  double value_max() const { return value_max_; }

  // Multi-field support
  void SetMultiFieldData(const Domain& domain,
                         const std::vector<std::vector<double>>& field_grids,
                         const std::vector<std::string>& field_names);
  void SetActiveField(int field_index);
  int GetActiveField() const { return active_field_index_; }
  const std::vector<std::string>& GetFieldNames() const { return field_names_; }
  bool IsMultiField() const { return field_grids_.size() > 1; }

  void FitToView();
  void SetOrientation(float yaw, float pitch);
  void GetOrientation(float* yaw, float* pitch) const;

  void RenderToTexture(int width, int height);
  unsigned int texture() const { return color_tex_; }
  int texture_width() const { return tex_width_; }
  int texture_height() const { return tex_height_; }
  bool has_data() const { return has_data_; }
  float point_scale() const { return point_scale_; }
  ViewMode view_mode() const { return view_mode_; }

  void Rotate(float dx, float dy);
  void Zoom(float delta);

  // Generate axis labels for the current view
  std::vector<ScreenLabel> AxisLabels() const;

 private:
  void EnsureFbo(int width, int height);
  void BuildMesh();
  void BuildGrid();
  bool CompileShaders();
  bool ProjectPoint(float x, float y, float z, float* out_x, float* out_y) const;

  Domain domain_{};
  std::vector<double> grid_;
  std::unique_ptr<struct DerivedFields> derived_fields_;  // Use pointer to avoid incomplete type
  bool has_derived_fields_ = false;
  FieldType field_type_ = FieldType::Solution;
  std::vector<PointSample> point_cloud_;
  int stride_ = 1;
  bool data_dirty_ = false;
  bool has_data_ = false;
  bool has_point_cloud_ = false;

  unsigned int fbo_ = 0;
  unsigned int color_tex_ = 0;
  unsigned int depth_rb_ = 0;
  int tex_width_ = 0;
  int tex_height_ = 0;

  unsigned int vao_ = 0;
  unsigned int vbo_ = 0;
  int vertex_count_ = 0;
  unsigned int grid_vao_ = 0;
  unsigned int grid_vbo_ = 0;
  int grid_vertex_count_ = 0;
  bool grid_dirty_ = true;
  bool has_mvp_ = false;
  float last_mvp_[16] = {};

  unsigned int program_ = 0;

  float clear_color_[3] = {0.08f, 0.09f, 0.11f};

  float yaw_ = 0.6f;
  float pitch_ = -0.4f;
  float distance_ = 3.6f;

  float fit_distance_ = 3.6f;
  float point_scale_ = 1.0f;
  float data_scale_ = 0.5f;
  float torus_major_ = 1.6f;
  float torus_minor_ = 0.45f;
  GridBuildResult grid_build_result_;
  float scene_radius_ = 1.0f;
  float scene_half_x_ = 1.0f;
  float scene_half_y_ = 1.0f;
  float scene_half_z_ = 1.0f;
  float scene_center_x_ = 0.0f;
  float scene_center_y_ = 0.0f;
  float scene_center_z_ = 0.0f;
  double value_min_ = 0.0;
  double value_max_ = 1.0;
  double value_range_ = 1.0;
  bool slice_enabled_ = false;
  int slice_axis_ = 0;
  double slice_value_ = 0.0;
  double slice_thickness_ = 0.0;
  bool iso_enabled_ = false;
  double iso_value_ = 0.0;
  double iso_band_ = 0.0;
  bool z_domain_locked_ = false;
  double z_domain_min_ = -1.0;
  double z_domain_max_ = 1.0;
  float last_aspect_ = 1.0f;
  int grid_divisions_ = 10;
  bool grid_enabled_ = true;
  bool use_ortho_ = true;
  ViewMode view_mode_ = ViewMode::Cartesian;

  // Multi-field support
  std::vector<std::vector<double>> field_grids_;
  std::vector<std::string> field_names_;
  int active_field_index_ = 0;
};

#endif
