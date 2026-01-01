#include "time_panel.h"

#include "imgui.h"

#include "styles/ui_style.h"
#include "ui_helpers.h"

#include <algorithm>
#include <cmath>
#include <vector>

namespace {

std::vector<std::string> DefaultComponents() {
  return {"time_controls"};
}

// Frame count presets
struct FramePreset {
  const char* label;
  int frames;
};

const FramePreset kFramePresets[] = {
    {"10", 10},
    {"50", 50},
    {"100", 100},
    {"250", 250},
    {"500", 500},
    {"1000", 1000},
};

// Time duration presets
struct DurationPreset {
  const char* label;
  double duration;
};

const DurationPreset kDurationPresets[] = {
    {"0.1", 0.1},
    {"0.5", 0.5},
    {"1.0", 1.0},
    {"5.0", 5.0},
    {"10.0", 10.0},
};

void RenderTimeRange(TimePanelState& state) {
  ImGui::TextUnformatted("Time Range");
  ImGui::Separator();

  // Start time
  ImGui::SetNextItemWidth(state.input_width * 0.6f);
  double t0 = state.time_start;
  if (UIInput::InputDouble("Start (t0)", &t0, 0.1, 1.0, "%.6g")) {
    state.time_start = t0;
  }

  // End time
  ImGui::SetNextItemWidth(state.input_width * 0.6f);
  double t1 = state.time_end;
  if (UIInput::InputDouble("End (t1)", &t1, 0.1, 1.0, "%.6g")) {
    state.time_end = t1;
  }

  // Duration display and quick presets
  double duration = state.time_end - state.time_start;
  ImGui::Text("Duration: %.6g", duration);
  ImGui::SameLine();
  ImGui::TextDisabled("Quick:");
  for (const auto& preset : kDurationPresets) {
    ImGui::SameLine();
    if (ImGui::SmallButton(preset.label)) {
      state.time_end = state.time_start + preset.duration;
    }
  }

  // Validation
  if (state.time_end <= state.time_start) {
    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 0.4f, 0.2f, 1.0f));
    ImGui::TextUnformatted("End time must be greater than start time");
    ImGui::PopStyleColor();
  }
}

void RenderFrameSettings(TimePanelState& state) {
  ImGui::Spacing();
  ImGui::TextUnformatted("Temporal Resolution");
  ImGui::Separator();

  // Frame count
  ImGui::SetNextItemWidth(state.input_width * 0.5f);
  int frames = state.time_frames;
  if (UIInput::InputInt("Frames", &frames, 1, 10)) {
    state.time_frames = std::max(2, std::min(frames, 1000000));
  }

  // Quick frame presets
  ImGui::SameLine();
  ImGui::TextDisabled("Quick:");
  for (const auto& preset : kFramePresets) {
    ImGui::SameLine();
    if (ImGui::SmallButton(preset.label)) {
      state.time_frames = preset.frames;
    }
  }

  // Calculate and display timestep
  double dt = CalculateTimestep(state.time_start, state.time_end, state.time_frames);
  ImGui::Text("Timestep (dt): %.6g", dt);

  // CFL number display if grid info available
  if (state.dx > 0.0 && state.wave_speed > 0.0) {
    double cfl = CalculateCFL(dt, state.dx, state.wave_speed);
    ImVec4 cfl_color;
    if (cfl <= 0.5) {
      cfl_color = ImVec4(0.2f, 0.8f, 0.2f, 1.0f);  // Green - stable
    } else if (cfl <= 1.0) {
      cfl_color = ImVec4(0.9f, 0.7f, 0.1f, 1.0f);  // Yellow - borderline
    } else {
      cfl_color = ImVec4(1.0f, 0.3f, 0.2f, 1.0f);  // Red - unstable
    }
    ImGui::TextColored(cfl_color, "CFL number: %.3f", cfl);
    if (cfl > 1.0) {
      ImGui::SameLine();
      ImGui::TextDisabled("(may be unstable)");
    }
  }
}

void RenderIntegrationMethod(TimePanelState& state) {
  ImGui::Spacing();
  ImGui::TextUnformatted("Integration Method");
  ImGui::Separator();

  // Method selector
  const char* method_names[] = {
      "Forward Euler",
      "Backward Euler",
      "Crank-Nicolson",
      "RK2 (Heun)",
      "RK4",
      "BDF2",
  };

  int method_idx = static_cast<int>(state.integration_method);
  ImGui::SetNextItemWidth(state.input_width * 0.7f);
  if (ImGui::Combo("Method", &method_idx, method_names, IM_ARRAYSIZE(method_names))) {
    state.integration_method = static_cast<TimeIntegrationMethod>(method_idx);
  }

  // Method info
  switch (state.integration_method) {
    case TimeIntegrationMethod::ForwardEuler:
      ImGui::TextDisabled("1st order, explicit, conditionally stable");
      break;
    case TimeIntegrationMethod::BackwardEuler:
      ImGui::TextDisabled("1st order, implicit, unconditionally stable");
      break;
    case TimeIntegrationMethod::CrankNicolson:
      ImGui::TextDisabled("2nd order, implicit, unconditionally stable");
      break;
    case TimeIntegrationMethod::RK2:
      ImGui::TextDisabled("2nd order, explicit, conditionally stable");
      break;
    case TimeIntegrationMethod::RK4:
      ImGui::TextDisabled("4th order, explicit, conditionally stable");
      break;
    case TimeIntegrationMethod::BDF2:
      ImGui::TextDisabled("2nd order, implicit, A-stable");
      break;
  }
}

void RenderAdaptiveSettings(TimePanelState& state) {
  ImGui::Spacing();
  ImGui::TextUnformatted("Adaptive Stepping");
  ImGui::Separator();

  // Stepping mode toggle
  bool is_adaptive = (state.stepping_mode == TimeSteppingMode::Adaptive);
  if (ImGui::Checkbox("Enable adaptive timestepping", &is_adaptive)) {
    state.stepping_mode = is_adaptive ? TimeSteppingMode::Adaptive : TimeSteppingMode::Fixed;
  }

  if (is_adaptive) {
    ImGui::Indent();

    // Target CFL
    ImGui::SetNextItemWidth(state.input_width * 0.4f);
    double cfl = state.cfl_target;
    if (UIInput::InputDouble("Target CFL", &cfl, 0.05, 0.1, "%.2f")) {
      state.cfl_target = std::max(0.01, std::min(cfl, 2.0));
    }

    // Min/Max dt
    ImGui::SetNextItemWidth(state.input_width * 0.4f);
    double min_dt = state.min_dt;
    if (UIInput::InputDouble("Min dt", &min_dt, 0.0, 0.0, "%.2e")) {
      state.min_dt = std::max(1e-15, min_dt);
    }

    ImGui::SetNextItemWidth(state.input_width * 0.4f);
    double max_dt = state.max_dt;
    if (UIInput::InputDouble("Max dt", &max_dt, 0.0, 0.0, "%.2e")) {
      state.max_dt = std::max(state.min_dt, max_dt);
    }

    ImGui::Unindent();
  }
}

void RenderSummary(TimePanelState& state) {
  ImGui::Spacing();
  ImGui::Separator();

  // Summary info
  double duration = state.time_end - state.time_start;
  double dt = CalculateTimestep(state.time_start, state.time_end, state.time_frames);

  ImGui::TextDisabled("Summary: %d frames over %.4g time units (dt = %.4g)",
                      state.time_frames, duration, dt);

  // Estimate memory/compute cost
  // This is a rough estimate - actual depends on grid size
  if (state.time_frames > 100) {
    ImGui::TextDisabled("Output: %d solution snapshots", state.time_frames);
  }
}

void RenderTimeControls(TimePanelState& state) {
  if (!state.time_dependent) {
    ImGui::TextDisabled("Time controls available for time-dependent PDEs.");
    ImGui::Spacing();
    ImGui::TextDisabled("Select a time-dependent PDE type (e.g., Heat equation,");
    ImGui::TextDisabled("Wave equation, Advection) to enable these settings.");
    return;
  }

  RenderTimeRange(state);
  RenderFrameSettings(state);
  RenderIntegrationMethod(state);
  RenderAdaptiveSettings(state);
  RenderSummary(state);
}

}  // namespace

const char* GetTimeMethodName(TimeIntegrationMethod method) {
  switch (method) {
    case TimeIntegrationMethod::ForwardEuler:
      return "Forward Euler";
    case TimeIntegrationMethod::BackwardEuler:
      return "Backward Euler";
    case TimeIntegrationMethod::CrankNicolson:
      return "Crank-Nicolson";
    case TimeIntegrationMethod::RK2:
      return "RK2 (Heun)";
    case TimeIntegrationMethod::RK4:
      return "RK4";
    case TimeIntegrationMethod::BDF2:
      return "BDF2";
  }
  return "Unknown";
}

void RenderTimePanel(TimePanelState& state, const std::vector<std::string>& components) {
  const std::vector<std::string> default_components = DefaultComponents();
  const std::vector<std::string>& component_list =
      components.empty() ? default_components : components;

  for (size_t i = 0; i < component_list.size(); ++i) {
    const std::string& id = component_list[i];
    if (id == "time_controls") {
      RenderTimeControls(state);
    } else {
      DrawUnknownComponentPlaceholder(id.c_str());
    }

    if (i + 1 < component_list.size()) {
      ImGui::Spacing();
    }
  }
}
