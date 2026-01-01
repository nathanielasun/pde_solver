#include "preset_manager_panel.h"
#include "imgui.h"

void RenderPresetManagerPanel(PresetManagerPanelState& state,
                               const std::vector<std::string>& components) {
  ImGui::Text("Preset Manager");
  ImGui::Separator();
  ImGui::Spacing();

  ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f),
                     "Save and load complete problem setups.");
  ImGui::Spacing();

  // Placeholder UI
  if (ImGui::CollapsingHeader("Built-in Presets", ImGuiTreeNodeFlags_DefaultOpen)) {
    ImGui::BulletText("Heat Equation (2D)");
    ImGui::BulletText("Wave Equation (2D)");
    ImGui::BulletText("Laplace Equation");
    ImGui::BulletText("Poisson Equation");
    ImGui::BulletText("Diffusion-Reaction");
  }

  ImGui::Spacing();

  if (ImGui::CollapsingHeader("User Presets")) {
    ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "No user presets saved.");
  }

  ImGui::Spacing();
  ImGui::Separator();

  if (ImGui::Button("Save Current Setup", ImVec2(-1, 0))) {
    // TODO: Implement save functionality
  }

  ImGui::TextColored(ImVec4(1.0f, 0.7f, 0.3f, 1.0f),
                     "Full preset functionality coming soon.");
}
