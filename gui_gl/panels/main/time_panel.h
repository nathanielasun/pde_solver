#ifndef TIME_PANEL_H
#define TIME_PANEL_H

#include "app_state.h"
#include <string>
#include <vector>

// Forward declaration
class CommandHistory;

// Time integration methods
enum class TimeIntegrationMethod {
  ForwardEuler,
  BackwardEuler,
  CrankNicolson,
  RK2,
  RK4,
  BDF2,
};

// Get display name for time integration method
const char* GetTimeMethodName(TimeIntegrationMethod method);

// Time stepping mode
enum class TimeSteppingMode {
  Fixed,      // Fixed dt
  Adaptive,   // Adaptive based on CFL/error
};

// State structure for time panel
struct TimePanelState {
  // Core time parameters (references to application state)
  double& time_start;
  double& time_end;
  int& time_frames;

  // Time integration settings
  TimeIntegrationMethod& integration_method;
  TimeSteppingMode& stepping_mode;
  double& cfl_target;        // Target CFL number for adaptive stepping
  double& min_dt;            // Minimum allowed timestep
  double& max_dt;            // Maximum allowed timestep

  // Grid info for CFL calculation
  double dx;                 // Grid spacing in x
  double dy;                 // Grid spacing in y
  double dz;                 // Grid spacing in z (0 for 2D)
  double wave_speed;         // Characteristic wave speed (for CFL)

  // UI state
  bool time_dependent;       // Whether time-dependent features should be shown
  float input_width;

  // Command history for undo/redo
  CommandHistory* cmd_history = nullptr;
};

// Render the Time panel using configured components.
void RenderTimePanel(TimePanelState& state, const std::vector<std::string>& components);

// Utility: Calculate timestep from frames
inline double CalculateTimestep(double t0, double t1, int frames) {
  if (frames <= 1) return t1 - t0;
  return (t1 - t0) / static_cast<double>(frames - 1);
}

// Utility: Calculate CFL number
inline double CalculateCFL(double dt, double dx, double wave_speed) {
  if (dx <= 0.0) return 0.0;
  return wave_speed * dt / dx;
}

#endif  // TIME_PANEL_H
