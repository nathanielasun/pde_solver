#ifndef MESH_BUILDER_H
#define MESH_BUILDER_H

#include "GlViewer.h"
#include "render_types.h"
#include "pde_types.h"
#include <vector>

// Forward declaration
struct DerivedFields;

// Mesh building parameters
struct MeshBuildParams {
  const Domain* domain = nullptr;
  const std::vector<double>* grid = nullptr;
  const std::vector<PointSample>* point_cloud = nullptr;
  const DerivedFields* derived_fields = nullptr;
  
  bool has_point_cloud = false;
  bool has_derived_fields = false;
  GlViewer::FieldType field_type = GlViewer::FieldType::Solution;
  
  GlViewer::ViewMode view_mode = GlViewer::ViewMode::Cartesian;
  float data_scale = 0.5f;
  float torus_major = 1.6f;
  float torus_minor = 0.45f;
  
  int stride = 1;
  
  // Slice plane parameters
  bool slice_enabled = false;
  int slice_axis = 0;
  double slice_value = 0.0;
  double slice_thickness = 0.0;
  
  // Isosurface parameters
  bool iso_enabled = false;
  double iso_value = 0.0;
  double iso_band = 0.0;
  
  // Z domain locking
  bool z_domain_locked = false;
  double z_domain_min = -1.0;
  double z_domain_max = 1.0;
};

// Mesh building result
struct MeshBuildResult {
  std::vector<Vertex> vertices;
  double value_min = 0.0;
  double value_max = 1.0;
  double value_range = 1.0;
  
  // Scene bounds
  float scene_center_x = 0.0f;
  float scene_center_y = 0.0f;
  float scene_center_z = 0.0f;
  float scene_half_x = 1.0f;
  float scene_half_y = 1.0f;
  float scene_half_z = 1.0f;
  float scene_radius = 1.0f;
};

namespace MeshBuilder {

// Build mesh from point cloud data
MeshBuildResult BuildPointCloudMesh(const MeshBuildParams& params);

// Build mesh from grid data
MeshBuildResult BuildGridMesh(const MeshBuildParams& params);

// Build mesh (automatically chooses point cloud or grid)
MeshBuildResult BuildMesh(const MeshBuildParams& params);

// Helper: Get field value based on field type
double GetFieldValue(size_t idx, GlViewer::FieldType field_type,
                    const std::vector<double>& grid,
                    const DerivedFields* derived,
                    bool has_derived);

// Helper: Compute color from normalized value (0-1)
void ValueToColor(float t, float* r, float* g, float* b);

}  // namespace MeshBuilder

#endif  // MESH_BUILDER_H

