#include "isosurface_panel.h"
#include "ui_helpers.h"
#include "styles/ui_style.h"
#include "imgui.h"
#include <cfloat>

namespace {

std::vector<std::string> DefaultComponents() {
  return {"isosurface_controls"};
}

bool UseCompactLayout(float input_width) {
  const float avail = ImGui::GetContentRegionAvail().x;
  return avail < std::max(320.0f, input_width * 1.2f);
}

void RenderIsosurfaceControls(IsosurfacePanelState& state) {
  double iso_min = state.viewer.has_data() ? state.viewer.value_min() : 0.0;
  double iso_max = state.viewer.has_data() ? state.viewer.value_max() : 1.0;

  if (iso_max < iso_min) {
    std::swap(iso_min, iso_max);
  }
  if (iso_max - iso_min < 1e-12) {
    iso_max = iso_min + 1.0;
  }

  if (state.iso_value < iso_min || state.iso_value > iso_max) {
    state.iso_value = 0.5 * (iso_min + iso_max);
  }

  ImGui::Checkbox("Isosurface band", &state.iso_enabled);

  if (state.iso_enabled) {
    const bool compact = UseCompactLayout(state.input_width);

    ImGui::Text("Isovalue");
    ImGui::SetNextItemWidth(state.input_width);
    ImGui::SliderScalar("##iso_value", ImGuiDataType_Double, &state.iso_value,
                        &iso_min, &iso_max, "%.6g");

    if (!compact && ImGui::BeginTable("iso_layout", 2, ImGuiTableFlags_SizingStretchProp)) {
      ImGui::TableNextColumn();
      ImGui::Text("Band width");
      ImGui::SetNextItemWidth(-FLT_MIN);
      double band = state.iso_band;
      if (UIInput::InputDouble("##iso_band", &band, 0.0, 0.0, "%.6g")) {
        state.iso_band = std::max(0.0, band);
      }
      ImGui::EndTable();
    } else {
      ImGui::Text("Band width");
      ImGui::SetNextItemWidth(state.input_width);
      double band = state.iso_band;
      if (UIInput::InputDouble("##iso_band", &band, 0.0, 0.0, "%.6g")) {
        state.iso_band = std::max(0.0, band);
      }
    }
  }

  state.viewer.SetIsosurface(state.iso_enabled, state.iso_value, state.iso_band);
}

} // namespace

void RenderIsosurfacePanel(IsosurfacePanelState& state, const std::vector<std::string>& components) {
  const std::vector<std::string> default_components = DefaultComponents();
  const std::vector<std::string>& component_list =
      components.empty() ? default_components : components;

  for (size_t i = 0; i < component_list.size(); ++i) {
    const std::string& id = component_list[i];
    if (id == "isosurface_controls") {
      RenderIsosurfaceControls(state);
    } else {
      DrawUnknownComponentPlaceholder(id.c_str());
    }

    if (i + 1 < component_list.size()) {
      ImGui::Spacing();
    }
  }
}

