#ifndef ISOSURFACE_PANEL_H
#define ISOSURFACE_PANEL_H

#include "GlViewer.h"
#include <vector>
#include <string>

struct IsosurfacePanelState {
  GlViewer& viewer;
  
  bool& iso_enabled;
  double& iso_value;
  double& iso_band;
  
  float input_width;
};

void RenderIsosurfacePanel(IsosurfacePanelState& state, const std::vector<std::string>& components);

#endif // ISOSURFACE_PANEL_H

