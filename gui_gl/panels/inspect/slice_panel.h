#ifndef SLICE_PANEL_H
#define SLICE_PANEL_H

#include "GlViewer.h"
#include <mutex>
#include <vector>
#include <string>

struct SlicePanelState {
  GlViewer& viewer;
  std::mutex& state_mutex;
  Domain& current_domain;
  
  bool& slice_enabled;
  int& slice_axis;
  double& slice_value;
  double& slice_thickness;
  
  // Coordinate flags needed for axis labels
  bool use_cartesian_3d;
  bool use_cylindrical_volume;
  bool use_volume;
  bool use_surface;
  
  float input_width;
};

void RenderSlicePanel(SlicePanelState& state, const std::vector<std::string>& components);

#endif // SLICE_PANEL_H

