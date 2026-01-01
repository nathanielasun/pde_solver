#ifndef SOURCE_TERM_PANEL_H
#define SOURCE_TERM_PANEL_H

#include <string>
#include <vector>

struct SourceTermPanelState {
  std::string& source_expression;
  float input_width;
};

void RenderSourceTermPanel(SourceTermPanelState& state,
                            const std::vector<std::string>& components);

#endif  // SOURCE_TERM_PANEL_H
