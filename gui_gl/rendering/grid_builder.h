#ifndef GRID_BUILDER_H
#define GRID_BUILDER_H

#include "view_mode.h"
#include "render_types.h"
#include "pde_types.h"
#include <string>
#include <vector>

// Grid building parameters
struct GridBuildParams {
  const Domain* domain = nullptr;
  ViewMode view_mode = ViewMode::Cartesian;
  bool has_point_cloud = false;
  
  float data_scale = 0.5f;
  float torus_major = 1.6f;
  float torus_minor = 0.45f;
  int grid_divisions = 10;
  bool grid_enabled = true;
  
  // Slice plane visualization
  bool slice_enabled = false;
  int slice_axis = 0;
  double slice_value = 0.0;
  double slice_thickness = 0.0;
};

// Information about a single axis, for labeling (3D grid axis endpoints)
struct GridAxisInfo {
  float end_x = 0.0f;
  float end_y = 0.0f;
  float end_z = 0.0f;
  std::string label;
};

// Grid building result
struct GridBuildResult {
  std::vector<Vertex> vertices;
  std::vector<GridAxisInfo> axes;
  
  // Scene bounds (updated during grid building)
  float scene_half_x = 1.0f;
  float scene_half_y = 1.0f;
  float scene_half_z = 1.0f;
  float scene_radius = 1.0f;
};

namespace GridBuilder {

// Build grid lines for visualization
GridBuildResult BuildGrid(const GridBuildParams& params);

// Helper: Add a line segment to the vertex list
void AddLine(std::vector<Vertex>& vertices,
             float x0, float y0, float z0,
             float x1, float y1, float z1,
             float r, float g, float b);

}  // namespace GridBuilder

#endif  // GRID_BUILDER_H

