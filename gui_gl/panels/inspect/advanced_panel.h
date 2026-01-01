#ifndef ADVANCED_PANEL_H
#define ADVANCED_PANEL_H

#include "GlViewer.h"
#include <mutex>
#include <vector>
#include <string>

struct AdvancedPanelState {
  GlViewer& viewer;
  std::mutex& state_mutex;
  Domain& current_domain;
  std::vector<double>& current_grid;
  DerivedFields& derived_fields;
  bool& has_derived_fields;
  
  float input_width;
};

void RenderAdvancedPanel(AdvancedPanelState& state, const std::vector<std::string>& components);

#endif // ADVANCED_PANEL_H

