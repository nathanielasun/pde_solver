#ifndef RUN_PANEL_H
#define RUN_PANEL_H

#include "app_state.h"
#include "progress_feedback.h"
#include <functional>
#include <string>
#include <vector>

// State structure for run panel
struct RunPanelState {
  SharedState& state;
  std::mutex& state_mutex;
  
  bool running;
  double progress;
  std::string phase;
  std::string status;
  bool has_duration;
  double last_duration;
  bool stability_warning;
  int stability_frame;
  double stability_ratio;
  double stability_max;
  std::vector<float> residual_l2;
  std::vector<float> residual_linf;
  int thread_active;
  int thread_total;
  
  // Callbacks
  std::function<void()> on_solve;
  std::function<void()> on_stop;
  std::function<void()> on_load_latest;
};

// Render the Run panel using configured components.
void RenderRunPanel(RunPanelState& state, const std::vector<std::string>& components);

#endif  // RUN_PANEL_H
