#ifndef INITIAL_CONDITIONS_PANEL_H
#define INITIAL_CONDITIONS_PANEL_H

#include "app_state.h"
#include <string>
#include <vector>

struct InitialConditionsPanelState {
  std::string& ic_expression;    // LaTeX expression for initial condition
  bool time_dependent;           // Whether time-dependent mode is active
  float input_width;
};

void RenderInitialConditionsPanel(InitialConditionsPanelState& state,
                                   const std::vector<std::string>& components);

#endif  // INITIAL_CONDITIONS_PANEL_H
