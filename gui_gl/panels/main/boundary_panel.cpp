#include "boundary_panel.h"
#include "imgui.h"
#include "imgui_stdlib.h"
#include "app_helpers.h"
#include "validation.h"
#include "systems/command_history.h"
#include "styles/ui_style.h"
#include "ui_helpers.h"
#include <string>

namespace {

std::vector<std::string> DefaultComponents() {
  return {"bc_inputs"};
}

void RenderBoundaryInputs(BoundaryPanelState& state) {
  auto draw_bc_cell = [&](const char* label, const char* id, BoundaryInput& input,
                          LatexTexture& preview) {
    const float cell_width = ImGui::GetContentRegionAvail().x;
    const float max_width = std::max(0.0f, cell_width - 8.0f);
    const float bc_combo_width = std::min(140.0f, max_width);
    const float bc_value_width = max_width;
    const float bc_coeff_width = std::max(24.0f, (bc_value_width - 10.0f) / 3.0f);
    const bool compact_coeffs = bc_value_width < 180.0f;
    ImGui::PushID(id);
    std::string label_text = std::string(label) + " value";
    ImGui::Text("%s", label_text.c_str());
    const char* kind_items[] = {"Dirichlet", "Neumann", "Robin/Mixed"};
    
    int old_kind = input.kind;
    std::string old_value = input.value;
    std::string old_alpha = input.alpha;
    std::string old_beta = input.beta;
    std::string old_gamma = input.gamma;
    
    ImGui::SetNextItemWidth(bc_combo_width);
    if (input.kind < 0 || input.kind > 2) {
      // Clamp invalid kind to avoid out-of-range UI indices.
      input.kind = 0;
    }
    if (UIInput::Combo("Type", &input.kind, kind_items, IM_ARRAYSIZE(kind_items))) {
      if (state.cmd_history && input.kind != old_kind) {
        auto cmd = std::make_unique<SetIntCommand>(&input.kind, input.kind,
                                                   std::string("Change ") + label + " type");
        state.cmd_history->Execute(std::move(cmd));
      }
    }
    if (input.kind == 0 || input.kind == 1) {
      if (UIInput::InputTextMultiline("##value", &input.value, ImVec2(bc_value_width, 30),
                                      ImGuiInputTextFlags_WordWrap)) {
        if (state.cmd_history && input.value != old_value) {
          auto cmd = std::make_unique<SetStringCommand>(&input.value, input.value,
                                                        std::string("Change ") + label + " value");
          state.cmd_history->Execute(std::move(cmd));
        }
      }
    } else {
      if (compact_coeffs) {
        ImGui::PushItemWidth(bc_value_width);
        if (UIInput::InputText("a", &input.alpha)) {
          if (state.cmd_history && input.alpha != old_alpha) {
            auto cmd = std::make_unique<SetStringCommand>(&input.alpha, input.alpha,
                                                          std::string("Change ") + label + " alpha");
            state.cmd_history->Execute(std::move(cmd));
          }
        }
        if (UIInput::InputText("b", &input.beta)) {
          if (state.cmd_history && input.beta != old_beta) {
            auto cmd = std::make_unique<SetStringCommand>(&input.beta, input.beta,
                                                          std::string("Change ") + label + " beta");
            state.cmd_history->Execute(std::move(cmd));
          }
        }
        if (UIInput::InputText("g", &input.gamma)) {
          if (state.cmd_history && input.gamma != old_gamma) {
            auto cmd = std::make_unique<SetStringCommand>(&input.gamma, input.gamma,
                                                          std::string("Change ") + label + " gamma");
            state.cmd_history->Execute(std::move(cmd));
          }
        }
        ImGui::PopItemWidth();
      } else {
        ImGui::PushItemWidth(bc_coeff_width);
        if (UIInput::InputText("a", &input.alpha)) {
          if (state.cmd_history && input.alpha != old_alpha) {
            auto cmd = std::make_unique<SetStringCommand>(&input.alpha, input.alpha,
                                                          std::string("Change ") + label + " alpha");
            state.cmd_history->Execute(std::move(cmd));
          }
        }
        ImGui::SameLine();
        if (UIInput::InputText("b", &input.beta)) {
          if (state.cmd_history && input.beta != old_beta) {
            auto cmd = std::make_unique<SetStringCommand>(&input.beta, input.beta,
                                                          std::string("Change ") + label + " beta");
            state.cmd_history->Execute(std::move(cmd));
          }
        }
        ImGui::SameLine();
        if (UIInput::InputText("g", &input.gamma)) {
          if (state.cmd_history && input.gamma != old_gamma) {
            auto cmd = std::make_unique<SetStringCommand>(&input.gamma, input.gamma,
                                                          std::string("Change ") + label + " gamma");
            state.cmd_history->Execute(std::move(cmd));
          }
        }
        ImGui::PopItemWidth();
      }
    }
    std::string latex;
    std::string error;
    if (BuildBoundaryLatex(input, &latex, &error)) {
      UpdateLatexTexture(preview, latex, state.python_path, state.script_path,
                         state.cache_dir, state.latex_color, state.latex_font_size);
      if (!preview.error.empty()) {
        ImGui::TextColored(ImVec4(1.0f, 0.5f, 0.5f, 1.0f), "%s", preview.error.c_str());
      } else if (preview.texture != 0) {
        DrawLatexPreview(preview, bc_value_width, 70.0f);
      }
    } else if (!error.empty()) {
      ImGui::TextColored(ImVec4(1.0f, 0.5f, 0.5f, 1.0f), "%s", error.c_str());
    }
    ImGui::PopID();
  };
  
  const char* left_label = "x min";
  const char* right_label = "x max";
  const char* bottom_label = "y min";
  const char* top_label = "y max";
  const char* front_label = "z min";
  const char* back_label = "z max";
  
  if (state.use_axisymmetric) {
    left_label = "r min";
    right_label = "r max";
    bottom_label = "z min";
    top_label = "z max";
  } else if (state.use_polar_coords) {
    left_label = "r min";
    right_label = "r max";
    bottom_label = "theta min";
    top_label = "theta max";
  } else if (state.use_cylindrical_volume) {
    left_label = "r min";
    right_label = "r max";
    bottom_label = "theta min";
    top_label = "theta max";
    front_label = "z min";
    back_label = "z max";
  } else if (state.use_cartesian_3d) {
    front_label = "z min";
    back_label = "z max";
  } else if (state.use_surface) {
    left_label = "theta min";
    right_label = "theta max";
    bottom_label = "phi min";
    top_label = "phi max";
  } else if (state.use_volume) {
    left_label = "r min";
    right_label = "r max";
    bottom_label = "theta min";
    top_label = "theta max";
    front_label = "phi min";
    back_label = "phi max";
  }
  
  ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(6.0f, 4.0f));
  const float bc_region = ImGui::GetContentRegionAvail().x;
  const int bc_columns = bc_region < 360.0f ? 1 : 2;
  if (ImGui::BeginTable("BCTable", bc_columns, ImGuiTableFlags_SizingStretchSame)) {
    ImGui::TableNextRow();
    ImGui::TableNextColumn();
    draw_bc_cell(left_label, "bc_left", state.bc_left, state.bc_left_preview);
    if (bc_columns > 1) {
      ImGui::TableNextColumn();
    } else {
      ImGui::TableNextRow();
      ImGui::TableNextColumn();
    }
    draw_bc_cell(right_label, "bc_right", state.bc_right, state.bc_right_preview);
    
    ImGui::TableNextRow();
    ImGui::TableNextColumn();
    draw_bc_cell(bottom_label, "bc_bottom", state.bc_bottom, state.bc_bottom_preview);
    if (bc_columns > 1) {
      ImGui::TableNextColumn();
    } else {
      ImGui::TableNextRow();
      ImGui::TableNextColumn();
    }
    draw_bc_cell(top_label, "bc_top", state.bc_top, state.bc_top_preview);
    
    if (state.use_volume) {
      ImGui::TableNextRow();
      ImGui::TableNextColumn();
      draw_bc_cell(front_label, "bc_front", state.bc_front, state.bc_front_preview);
      if (bc_columns > 1) {
        ImGui::TableNextColumn();
      } else {
        ImGui::TableNextRow();
        ImGui::TableNextColumn();
      }
      draw_bc_cell(back_label, "bc_back", state.bc_back, state.bc_back_preview);
    }
    ImGui::EndTable();
  }
  ImGui::PopStyleVar();
}

}  // namespace

void RenderBoundaryPanel(BoundaryPanelState& state, const std::vector<std::string>& components) {
  const std::vector<std::string> default_components = DefaultComponents();
  const std::vector<std::string>& component_list =
      components.empty() ? default_components : components;
  
  for (size_t i = 0; i < component_list.size(); ++i) {
    const std::string& id = component_list[i];
    if (id == "bc_inputs") {
      RenderBoundaryInputs(state);
    } else {
      DrawUnknownComponentPlaceholder(id.c_str());
    }
    if (i + 1 < component_list.size()) {
      ImGui::Spacing();
    }
  }
}
