#ifndef TESTING_PANEL_H
#define TESTING_PANEL_H

#include "../../testing/test_result_types.h"
#include "../../testing/test_runner.h"

#include <memory>
#include <mutex>
#include <vector>

// Panel state for the testing panel
struct TestingPanelState {
  float input_width = 200.0f;
};

// Render the testing panel
void RenderTestingPanel(const TestingPanelState& state);

#endif  // TESTING_PANEL_H
