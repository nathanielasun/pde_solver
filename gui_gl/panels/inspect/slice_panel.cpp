#include "slice_panel.h"
#include "ui_helpers.h"
#include "utils/coordinate_utils.h"
#include "styles/ui_style.h"
#include "imgui.h"
#include <cfloat>

namespace {

std::vector<std::string> DefaultComponents() {
  return {"slice_controls"};
}

struct AxisInfo {
  const char* labels[3];
  int count;
  double min;
  double max;
};

AxisInfo ComputeAxisInfoForSlice(SlicePanelState& state, int axis_index) {
  AxisInfo info;

  const char* axis0 = "x";
  const char* axis1 = "y";
  const char* axis2 = "z";

  if (state.use_cartesian_3d) {
    axis0 = "x";
    axis1 = "y";
    axis2 = "z";
  } else if (state.use_cylindrical_volume) {
    axis0 = "r";
    axis1 = "theta";
    axis2 = "z";
  } else if (state.use_volume) {
    axis0 = "r";
    axis1 = "theta";
    axis2 = "phi";
  } else if (state.use_surface) {
    axis0 = "theta";
    axis1 = "phi";
    axis2 = "r";
  } else {
    axis0 = "r";
    axis1 = "theta";
    axis2 = "z";
  }

  info.labels[0] = axis0;
  info.labels[1] = axis1;
  info.labels[2] = axis2;
  info.count = state.use_volume ? 3 : 2;

  Domain axis_domain;
  if (state.viewer.has_data()) {
    axis_domain = state.viewer.domain();
  } else {
    std::lock_guard<std::mutex> lock(state.state_mutex);
    axis_domain = state.current_domain;
  }

  double axis_min = axis_domain.xmin;
  double axis_max = axis_domain.xmax;
  if (axis_index == 1) {
    axis_min = axis_domain.ymin;
    axis_max = axis_domain.ymax;
  } else if (axis_index == 2) {
    axis_min = axis_domain.zmin;
    axis_max = axis_domain.zmax;
  }

  if (axis_max < axis_min) {
    std::swap(axis_min, axis_max);
  }
  if (axis_max - axis_min < 1e-12) {
    axis_max = axis_min + 1.0;
  }

  info.min = axis_min;
  info.max = axis_max;
  return info;
}

bool UseCompactLayout(float input_width) {
  const float avail = ImGui::GetContentRegionAvail().x;
  return avail < std::max(320.0f, input_width * 1.2f);
}

void RenderSliceControls(SlicePanelState& state) {
  AxisInfo axis_info = ComputeAxisInfoForSlice(state, state.slice_axis);
  if (state.slice_axis >= axis_info.count) {
    state.slice_axis = axis_info.count - 1;
    axis_info = ComputeAxisInfoForSlice(state, state.slice_axis);
  }

  const char* axis_items[] = {axis_info.labels[0], axis_info.labels[1], axis_info.labels[2]};

  ImGui::Checkbox("Slice plane", &state.slice_enabled);

  if (state.slice_enabled) {
    const bool compact = UseCompactLayout(state.input_width);

    if (!compact && ImGui::BeginTable("slice_layout", 2, ImGuiTableFlags_SizingStretchProp)) {
      ImGui::TableNextColumn();
      ImGui::Text("Slice axis");
      ImGui::SetNextItemWidth(-FLT_MIN);
      UIInput::Combo("##slice_axis", &state.slice_axis, axis_items, axis_info.count);
      ImGui::TableNextColumn();
      ImGui::Text("Thickness");
      ImGui::SetNextItemWidth(-FLT_MIN);
      double thickness = state.slice_thickness;
      if (UIInput::InputDouble("##slice_thickness", &thickness, 0.0, 0.0, "%.6g")) {
        state.slice_thickness = std::max(0.0, thickness);
      }
      ImGui::EndTable();
    } else {
      ImGui::Text("Slice axis");
      ImGui::SetNextItemWidth(state.input_width);
      UIInput::Combo("##slice_axis", &state.slice_axis, axis_items, axis_info.count);

      ImGui::Text("Thickness");
      ImGui::SetNextItemWidth(state.input_width);
      double thickness = state.slice_thickness;
      if (UIInput::InputDouble("##slice_thickness", &thickness, 0.0, 0.0, "%.6g")) {
        state.slice_thickness = std::max(0.0, thickness);
      }
    }

    axis_info = ComputeAxisInfoForSlice(state, state.slice_axis);
    double axis_min = axis_info.min;
    double axis_max = axis_info.max;
    if (state.slice_value < axis_min || state.slice_value > axis_max) {
      state.slice_value = 0.5 * (axis_min + axis_max);
    }

    ImGui::Text("Slice position");
    ImGui::SetNextItemWidth(state.input_width);
    ImGui::SliderScalar("##slice_position", ImGuiDataType_Double, &state.slice_value,
                        &axis_min, &axis_max, "%.6g");

    ImGui::TextDisabled("Right-drag in the view to move the slice.");
  }

  state.viewer.SetSlice(state.slice_enabled, state.slice_axis,
                        state.slice_value, state.slice_thickness);
}

} // namespace

void RenderSlicePanel(SlicePanelState& state, const std::vector<std::string>& components) {
  const std::vector<std::string> default_components = DefaultComponents();
  const std::vector<std::string>& component_list =
      components.empty() ? default_components : components;

  for (size_t i = 0; i < component_list.size(); ++i) {
    const std::string& id = component_list[i];
    if (id == "slice_controls") {
      RenderSliceControls(state);
    } else {
      DrawUnknownComponentPlaceholder(id.c_str());
    }

    if (i + 1 < component_list.size()) {
      ImGui::Spacing();
    }
  }
}

