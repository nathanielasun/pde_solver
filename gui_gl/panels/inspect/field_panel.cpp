#include "field_panel.h"
#include "ui_helpers.h"
#include "styles/ui_style.h"
#include "vtk_io.h"
#include "imgui.h"

namespace {

std::vector<std::string> DefaultComponents() {
  return {"primary_field_selector", "field_selector"};
}

void RenderPrimaryFieldSelector(FieldPanelState& state) {
  // Only show selector if there are multiple primary fields
  if (state.field_names.size() <= 1) {
    return;
  }

  ImGui::Text("Primary Field");
  ImGui::SetNextItemWidth(state.input_width);

  // Build combo items from field names
  std::vector<const char*> items;
  for (const auto& name : state.field_names) {
    items.push_back(name.c_str());
  }

  if (UIInput::Combo("##primary_field", &state.active_field_index,
                     items.data(), static_cast<int>(items.size()))) {
    // Switch viewer to show selected field's data
    state.viewer.SetActiveField(state.active_field_index);

    // Update derived_fields reference to point to all_derived_fields[active_field_index]
    if (static_cast<size_t>(state.active_field_index) < state.all_derived_fields.size()) {
      std::lock_guard<std::mutex> lock(state.state_mutex);
      state.derived_fields = state.all_derived_fields[static_cast<size_t>(state.active_field_index)];
      state.viewer.SetDerivedFields(state.derived_fields);
    }
  }
  ImGui::Spacing();
}

void RenderFieldSelector(FieldPanelState& state) {
  bool has_derived = false;
  {
    std::lock_guard<std::mutex> lock(state.state_mutex);
    has_derived = state.has_derived_fields;
    if (has_derived) {
      state.viewer.SetDerivedFields(state.derived_fields);
    }
  }

  GlViewer::FieldType current_field = state.viewer.GetFieldType();
  if (current_field == GlViewer::FieldType::Solution) {
    state.field_type_index = 0;
  } else if (current_field == GlViewer::FieldType::GradientX) {
    state.field_type_index = 1;
  } else if (current_field == GlViewer::FieldType::GradientY) {
    state.field_type_index = 2;
  } else if (current_field == GlViewer::FieldType::GradientZ) {
    state.field_type_index = 3;
  } else if (current_field == GlViewer::FieldType::Laplacian) {
    state.field_type_index = state.use_volume ? 4 : 3;
  } else if (current_field == GlViewer::FieldType::FluxX) {
    state.field_type_index = state.use_volume ? 5 : 4;
  } else if (current_field == GlViewer::FieldType::FluxY) {
    state.field_type_index = state.use_volume ? 6 : 5;
  } else if (current_field == GlViewer::FieldType::FluxZ) {
    state.field_type_index = 7;
  } else if (current_field == GlViewer::FieldType::EnergyNorm) {
    state.field_type_index = state.use_volume ? 8 : 6;
  }

  const char* field_items[] = {
    "Solution (u)",
    "Gradient X (∂u/∂x)",
    "Gradient Y (∂u/∂y)",
    "Gradient Z (∂u/∂z)",
    "Laplacian (∇²u)",
    "Flux X",
    "Flux Y",
    "Flux Z",
    "Energy Norm (u²)"
  };

  if (!state.use_volume) {
    const char* field_items_2d[] = {
      "Solution (u)",
      "Gradient X (∂u/∂x)",
      "Gradient Y (∂u/∂y)",
      "Laplacian (∇²u)",
      "Flux X",
      "Flux Y",
      "Energy Norm (u²)"
    };
    ImGui::Text("Field to Visualize");
    ImGui::SetNextItemWidth(state.input_width);
    if (UIInput::Combo("##field_selector", &state.field_type_index, field_items_2d, 7)) {
      GlViewer::FieldType selected = GlViewer::FieldType::Solution;
      switch (state.field_type_index) {
        case 0: selected = GlViewer::FieldType::Solution; break;
        case 1: selected = GlViewer::FieldType::GradientX; break;
        case 2: selected = GlViewer::FieldType::GradientY; break;
        case 3: selected = GlViewer::FieldType::Laplacian; break;
        case 4: selected = GlViewer::FieldType::FluxX; break;
        case 5: selected = GlViewer::FieldType::FluxY; break;
        case 6: selected = GlViewer::FieldType::EnergyNorm; break;
      }
      state.viewer.SetFieldType(selected);
    }
  } else {
    ImGui::Text("Field to Visualize");
    ImGui::SetNextItemWidth(state.input_width);
    if (UIInput::Combo("##field_selector", &state.field_type_index, field_items, 9)) {
      GlViewer::FieldType selected = GlViewer::FieldType::Solution;
      switch (state.field_type_index) {
        case 0: selected = GlViewer::FieldType::Solution; break;
        case 1: selected = GlViewer::FieldType::GradientX; break;
        case 2: selected = GlViewer::FieldType::GradientY; break;
        case 3: selected = GlViewer::FieldType::GradientZ; break;
        case 4: selected = GlViewer::FieldType::Laplacian; break;
        case 5: selected = GlViewer::FieldType::FluxX; break;
        case 6: selected = GlViewer::FieldType::FluxY; break;
        case 7: selected = GlViewer::FieldType::FluxZ; break;
        case 8: selected = GlViewer::FieldType::EnergyNorm; break;
      }
      state.viewer.SetFieldType(selected);
    }
  }

  if (state.field_type_index > 0 && !has_derived) {
    ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.3f, 1.0f),
                       "Derived fields not available. Solve PDE to compute them.");
  }
}

} // namespace

void RenderFieldPanel(FieldPanelState& state, const std::vector<std::string>& components) {
  const std::vector<std::string> default_components = DefaultComponents();
  const std::vector<std::string>& component_list =
      components.empty() ? default_components : components;

  for (size_t i = 0; i < component_list.size(); ++i) {
    const std::string& id = component_list[i];
    if (id == "primary_field_selector") {
      RenderPrimaryFieldSelector(state);
    } else if (id == "field_selector") {
      RenderFieldSelector(state);
    } else {
      DrawUnknownComponentPlaceholder(id.c_str());
    }

    if (i + 1 < component_list.size()) {
      ImGui::Spacing();
    }
  }
}

