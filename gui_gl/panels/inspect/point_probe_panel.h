#ifndef POINT_PROBE_PANEL_H
#define POINT_PROBE_PANEL_H

#include "GlViewer.h"
#include "vtk_io.h"
#include <mutex>
#include <vector>
#include <string>

struct ProbeData {
  double x = 0.0, y = 0.0, z = 0.0;
  double value = 0.0;
  bool valid = false;
  std::string label;
};

struct PointProbePanelState {
  std::mutex& state_mutex;
  std::vector<double>& current_grid;
  DerivedFields& derived_fields;
  bool& has_derived_fields;
  Domain& current_domain;
  float input_width;
};

void RenderPointProbePanel(PointProbePanelState& state, const std::vector<std::string>& components);

#endif // POINT_PROBE_PANEL_H
