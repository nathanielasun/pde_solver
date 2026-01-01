#ifndef CONVERGENCE_PANEL_H
#define CONVERGENCE_PANEL_H

#include <mutex>
#include <vector>
#include <string>

struct ConvergencePanelState {
  std::mutex& state_mutex;
  std::vector<float>& residual_l2;
  std::vector<float>& residual_linf;
  float input_width;

  // Local state
  bool show_l2 = true;
  bool show_linf = true;
  bool log_scale = true;
  bool auto_scroll = true;
};

void RenderConvergencePanel(ConvergencePanelState& state, const std::vector<std::string>& components);

#endif // CONVERGENCE_PANEL_H
