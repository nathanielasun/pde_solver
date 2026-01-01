#include "material_properties_panel.h"
#include "imgui.h"

static int s_property_index = 0;
static float s_diffusivity_x = 1.0f;
static float s_diffusivity_y = 1.0f;
static float s_diffusivity_z = 1.0f;

void RenderMaterialPropertiesPanel(MaterialPropertiesPanelState& state,
                                    const std::vector<std::string>& components) {
  ImGui::Text("Material Properties");
  ImGui::Separator();
  ImGui::Spacing();

  ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f),
                     "Define spatially-varying coefficients.");
  ImGui::Spacing();

  ImGui::Text("Property:");
  ImGui::Combo("##mat_prop", &s_property_index,
               "Diffusivity X\0Diffusivity Y\0Diffusivity Z\0Conductivity\0");

  ImGui::Spacing();
  ImGui::Separator();
  ImGui::Spacing();

  ImGui::Text("Constant Values (uniform):");
  ImGui::SetNextItemWidth(state.input_width * 0.4f);
  ImGui::InputFloat("a (x-diffusivity)", &s_diffusivity_x, 0.01f, 0.1f, "%.3f");
  ImGui::SetNextItemWidth(state.input_width * 0.4f);
  ImGui::InputFloat("b (y-diffusivity)", &s_diffusivity_y, 0.01f, 0.1f, "%.3f");
  ImGui::SetNextItemWidth(state.input_width * 0.4f);
  ImGui::InputFloat("c (z-diffusivity)", &s_diffusivity_z, 0.01f, 0.1f, "%.3f");

  ImGui::Spacing();

  if (ImGui::CollapsingHeader("Preset Materials")) {
    ImGui::BulletText("Uniform (isotropic)");
    ImGui::BulletText("Layered (z-dependent)");
    ImGui::BulletText("Gradient (linear variation)");
    ImGui::BulletText("Anisotropic");
  }

  ImGui::Spacing();
  ImGui::TextColored(ImVec4(1.0f, 0.7f, 0.3f, 1.0f),
                     "Variable coefficients coming soon.");
}
