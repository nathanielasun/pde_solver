#ifndef SOURCE_TERM_PANEL_H
#define SOURCE_TERM_PANEL_H

#include <string>
#include <vector>

struct PointSource {
  double x = 0.5, y = 0.5, z = 0.0;
  double strength = 1.0;
};

struct SourceTermPanelState {
  std::string& source_expression;
  float input_width;
  std::vector<PointSource> point_sources;
};

void RenderSourceTermPanel(SourceTermPanelState& state,
                            const std::vector<std::string>& components);

#endif  // SOURCE_TERM_PANEL_H
