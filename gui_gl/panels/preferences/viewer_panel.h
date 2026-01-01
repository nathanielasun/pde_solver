#ifndef VIEWER_PANEL_H
#define VIEWER_PANEL_H

#include "GlViewer.h"
#include "app_helpers.h"
#include <string>
#include <vector>

// State structure for viewer panel
struct ViewerPanelState {
  GlViewer& viewer;
  int coord_mode;
  
  bool& use_ortho;
  bool& lock_fit;
  int& stride;
  float& point_scale;
  float& data_scale;
  bool& z_domain_locked;
  double& z_domain_min;
  double& z_domain_max;
  bool& grid_enabled;
  int& grid_divisions;

  float input_width;
};

// Render the Viewer panel using configured components.
void RenderViewerPanel(ViewerPanelState& state, const std::vector<std::string>& components);

#endif  // VIEWER_PANEL_H
