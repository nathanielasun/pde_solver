#ifndef COMPUTE_PANEL_H
#define COMPUTE_PANEL_H

#include "app_state.h"
#include "systems/backend_capabilities.h"
#include "app_helpers.h"
#include <string>
#include <functional>
#include <vector>

// Forward declaration
class CommandHistory;

// State structure for compute panel
struct ComputePanelState {
  // Backend selection
  int& backend_index;
  
  // Solver method selection
  int& method_index;
  double& sor_omega;
  int& gmres_restart;
  
  // Preferences (for saving)
  int& pref_method_index;
  double& pref_sor_omega;
  int& pref_gmres_restart;
  bool& prefs_changed;
  
  // Threading
  int& thread_count;
  int& max_threads;
  
  // Metal tuning
  int& metal_reduce_interval;
  int& metal_tg_x;
  int& metal_tg_y;
  int& pref_metal_reduce_interval;
  int& pref_metal_tg_x;
  int& pref_metal_tg_y;
  
  // Backend registry (static, but passed for initialization)
  BackendUIRegistry* backend_registry;
  std::function<void()> initialize_backend_registry;
  
  // UI width
  float input_width;
  
  // Optional command history for undo/redo
  CommandHistory* cmd_history = nullptr;
};

// Render the Compute panel using configured components.
void RenderComputePanel(ComputePanelState& state, const std::vector<std::string>& components);

#endif // COMPUTE_PANEL_H
