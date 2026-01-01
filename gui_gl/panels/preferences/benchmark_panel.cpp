#include "benchmark_panel.h"
#include "ui_helpers.h"
#include "styles/ui_style.h"
#include "imgui.h"

namespace {

std::vector<std::string> DefaultComponents() {
  return {"benchmark_controls"};
}

void RenderBenchmarkControls(BenchmarkPanelState& state) {
  ImGui::TextWrapped("Standard time-dependent benchmark PDE:");
  ImGui::TextDisabled("%s", state.config.pde.c_str());
  ImGui::TextDisabled("Domain: [%.2f, %.2f] x [%.2f, %.2f]",
                      state.config.xmin, state.config.xmax,
                      state.config.ymin, state.config.ymax);
  ImGui::TextDisabled("Grid: %d x %d", state.config.nx, state.config.ny);
  ImGui::TextDisabled("BC: %s on all sides", state.config.bc.c_str());
  ImGui::TextDisabled("Time: t=%.2f..%.2f (%d frames)",
                      state.config.t_start, state.config.t_end,
                      state.config.frames);
  ImGui::TextDisabled("Output dir: %s", state.config.output_dir.c_str());
  
  ImGui::BeginDisabled(state.running);
  if (UIButton::Button("Load benchmark settings", UIButton::Size::Medium, UIButton::Variant::Secondary)) {
    if (state.on_load_settings) {
      state.on_load_settings();
    }
  }
  ImGui::SameLine();
  if (UIButton::Button("Run benchmark", UIButton::Size::Medium, UIButton::Variant::Primary)) {
    if (state.on_load_settings) {
      state.on_load_settings();
    }
    if (state.on_run_benchmark) {
      state.on_run_benchmark();
    }
  }
  ImGui::EndDisabled();
  if (state.running) {
    ImGui::TextDisabled("Solver is running; benchmark controls are disabled.");
  }
}

}  // namespace

void RenderBenchmarkPanel(BenchmarkPanelState& state, const std::vector<std::string>& components) {
  const std::vector<std::string> default_components = DefaultComponents();
  const std::vector<std::string>& component_list =
      components.empty() ? default_components : components;
  
  for (size_t i = 0; i < component_list.size(); ++i) {
    const std::string& id = component_list[i];
    if (id == "benchmark_controls") {
      RenderBenchmarkControls(state);
    } else {
      DrawUnknownComponentPlaceholder(id.c_str());
    }
    if (i + 1 < component_list.size()) {
      ImGui::Spacing();
    }
  }
}
