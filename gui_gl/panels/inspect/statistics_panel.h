#ifndef STATISTICS_PANEL_H
#define STATISTICS_PANEL_H

#include "GlViewer.h"
#include "vtk_io.h"
#include <mutex>
#include <vector>
#include <string>

struct StatisticsPanelState {
  std::mutex& state_mutex;
  std::vector<double>& current_grid;
  DerivedFields& derived_fields;
  bool& has_derived_fields;
  Domain& current_domain;
  float input_width;
};

void RenderStatisticsPanel(StatisticsPanelState& state, const std::vector<std::string>& components);

#endif // STATISTICS_PANEL_H
