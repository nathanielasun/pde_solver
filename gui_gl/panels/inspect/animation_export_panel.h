#ifndef ANIMATION_EXPORT_PANEL_H
#define ANIMATION_EXPORT_PANEL_H

#include "GlViewer.h"
#include <string>
#include <vector>
#include <atomic>

struct AnimationExportPanelState {
  std::vector<std::string>& frame_paths;
  std::vector<double>& frame_times;
  GlViewer& viewer;
  float input_width;
};

void RenderAnimationExportPanel(AnimationExportPanelState& state,
                                 const std::vector<std::string>& components);

#endif  // ANIMATION_EXPORT_PANEL_H
