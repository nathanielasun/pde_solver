#ifndef PROJECTION_H
#define PROJECTION_H

#include "GlViewer.h"
#include "pde_types.h"
#include <vector>
#include <string>

// 3D to 2D projection and axis labeling utilities
namespace Projection {

// Project a 3D point to 2D screen coordinates using MVP matrix
// Returns true if projection is valid (point is in view frustum)
bool ProjectPoint(const float mvp[16], int tex_width, int tex_height,
                  float x, float y, float z,
                  float* out_x, float* out_y);

// Generate axis labels for a coordinate system
struct AxisLabelParams {
  const Domain* domain = nullptr;
  GlViewer::ViewMode view_mode = GlViewer::ViewMode::Cartesian;
  int grid_divisions = 10;
  float data_scale = 0.5f;
  float torus_major = 1.6f;
  float torus_minor = 0.45f;
  bool has_point_cloud = false;
  bool use_value_height = false;
  double value_min = 0.0;
  double value_max = 1.0;
  double value_range = 1.0;
  const float* mvp = nullptr;  // 16-element MVP matrix
  int tex_width = 0;
  int tex_height = 0;
};

// Generate axis labels for display
std::vector<GlViewer::ScreenLabel> GenerateAxisLabels(const AxisLabelParams& params);

// Format a value for display
std::string FormatValue(double value);

}  // namespace Projection

#endif  // PROJECTION_H

