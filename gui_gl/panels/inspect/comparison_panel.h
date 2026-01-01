#ifndef COMPARISON_PANEL_H
#define COMPARISON_PANEL_H

#include "GlViewer.h"
#include <mutex>
#include <vector>
#include <string>

struct ComparisonPanelState {
  GlViewer& viewer;
  std::mutex& state_mutex;
  Domain& current_domain;
  std::vector<double>& current_grid;
  
  float input_width;
};

void RenderComparisonPanel(ComparisonPanelState& state, const std::vector<std::string>& components);

#endif // COMPARISON_PANEL_H

