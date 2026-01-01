#include "comparison_panel.h"
#include "ui_helpers.h"
#include "components/comparison_tools.h"
#include "imgui.h"
#include <algorithm>
#include <memory>

namespace {

std::vector<std::string> DefaultComponents() {
  return {"comparison_tools"};
}

void RenderComparisonToolsSection(ComparisonPanelState& state) {
  static std::unique_ptr<ComparisonToolsComponent> comparison_component;
  if (!comparison_component) {
    comparison_component = std::make_unique<ComparisonToolsComponent>();
    comparison_component->SetViewer(&state.viewer);
  }

  {
    std::lock_guard<std::mutex> lock(state.state_mutex);
    if (!state.current_grid.empty()) {
      comparison_component->SetCurrentSolution(state.current_domain, state.current_grid);
    }
  }

  comparison_component->Render();
}

} // namespace

void RenderComparisonPanel(ComparisonPanelState& state, const std::vector<std::string>& components) {
  const std::vector<std::string> default_components = DefaultComponents();
  const std::vector<std::string>& component_list =
      components.empty() ? default_components : components;

  for (size_t i = 0; i < component_list.size(); ++i) {
    const std::string& id = component_list[i];
    if (id == "comparison_tools") {
      RenderComparisonToolsSection(state);
    } else {
      DrawUnknownComponentPlaceholder(id.c_str());
    }

    if (i + 1 < component_list.size()) {
      ImGui::Spacing();
    }
  }
}

