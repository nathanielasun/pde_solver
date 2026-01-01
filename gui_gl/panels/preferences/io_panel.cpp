#include "io_panel.h"
#include "ui_helpers.h"
#include "systems/ui_config.h"
#include "styles/ui_style.h"
#include "imgui.h"
#include "imgui_stdlib.h"
#include <filesystem>

namespace {

std::vector<std::string> DefaultComponents() {
  return {"io_paths"};
}

void RenderIOPaths(IOPanelState& state) {
  ImGui::Text("Output path/dir");
  ImGui::BeginGroup();
  ImGui::SetNextItemWidth(state.input_width - 70.0f);
  UIInput::InputText("##output_path", &state.output_path);
  ImGui::SameLine();
  if (UIButton::Button("Browse##output", UIButton::Size::Small, UIButton::Variant::Secondary)) {
    std::filesystem::path default_path = state.output_path.empty() ?
        std::filesystem::current_path() : std::filesystem::path(state.output_path);
    auto selected = FileDialog::PickDirectory("Select Output Directory", default_path);
    if (selected) {
      state.output_path = selected->string();
    }
  }
  ImGui::EndGroup();
  
  ImGui::Text("Input directory");
  ImGui::BeginGroup();
  ImGui::SetNextItemWidth(state.input_width - 70.0f);
  UIInput::InputText("##input_dir", &state.input_dir);
  ImGui::SameLine();
  if (UIConfigManager::Instance().IsFeatureEnabled("file_dialogs")) {
    if (UIButton::Button("Browse##input", UIButton::Size::Small, UIButton::Variant::Secondary)) {
      std::filesystem::path default_path = state.input_dir.empty() ?
          std::filesystem::current_path() : std::filesystem::path(state.input_dir);
      auto selected = FileDialog::PickDirectory("Select Input Directory", default_path);
      if (selected) {
        state.input_dir = selected->string();
      }
    }
  } else {
    ImGui::BeginDisabled();
    UIButton::Button("Browse##input", UIButton::Size::Small, UIButton::Variant::Secondary);
    ImGui::EndDisabled();
  }
  ImGui::EndGroup();
}

}  // namespace

void RenderIOPanel(IOPanelState& state, const std::vector<std::string>& components) {
  const std::vector<std::string> default_components = DefaultComponents();
  const std::vector<std::string>& component_list =
      components.empty() ? default_components : components;
  
  for (size_t i = 0; i < component_list.size(); ++i) {
    const std::string& id = component_list[i];
    if (id == "io_paths") {
      RenderIOPaths(state);
    } else {
      DrawUnknownComponentPlaceholder(id.c_str());
    }
    if (i + 1 < component_list.size()) {
      ImGui::Spacing();
    }
  }
}
