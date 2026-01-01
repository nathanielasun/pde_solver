#include "equation_panel.h"

#include "imgui.h"
#include "imgui_stdlib.h"

#include "templates.h"
#include "app_helpers.h"
#include "styles/ui_style.h"
#include "ui_helpers.h"

#include <algorithm>
// #region agent log
#include <fstream>
#include <chrono>
// #endregion agent log

namespace {

std::vector<std::string> DefaultComponents() {
  return {"pde_input", "pde_preview", "pde_templates"};
}

void RenderPdeInput(EquationPanelState& state) {
  ImGui::Text("PDE (LaTeX-like)");
  ImGui::SetNextItemWidth(state.input_width);
  const ImVec2 box_size(state.input_width, 110.0f);
  const bool changed =
      UIInput::InputTextMultiline("##pde_input", &state.pde_text, box_size);
  if (changed) {
    state.pde_preview.dirty = true;
    state.refresh_coord_flags();
  }
}

void RenderPdePreview(EquationPanelState& state) {
  if (state.pde_preview.dirty || state.pde_preview.last_rendered != state.pde_text) {
    UpdateLatexTexture(state.pde_preview, state.pde_text,
                       state.python_path, state.script_path, state.cache_dir,
                       state.latex_color, state.latex_font_size);
  }

  if (!state.pde_preview.error.empty()) {
    ImGui::TextColored(ImVec4(1, 0.4f, 0.4f, 1), "Preview error: %s",
                       state.pde_preview.error.c_str());
  } else {
    DrawLatexPreview(state.pde_preview, state.input_width, 140.0f);
  }
}

void RenderTemplates(EquationPanelState& state) {
  if (!ImGui::TreeNode("Templates")) {
    return;
  }

  static std::vector<ProblemTemplate> templates = GetProblemTemplates();
  static int selected = -1;
  std::vector<const char*> template_names;
  template_names.reserve(templates.size() + 1);
  template_names.push_back("Select...");
  for (const auto& tmpl : templates) {
    template_names.push_back(tmpl.name.c_str());
  }

  ImGui::SetNextItemWidth(state.input_width);
  int combo_index = selected + 1;
  if (UIInput::Combo("Problem", &combo_index, template_names.data(),
                     static_cast<int>(template_names.size()))) {
    selected = combo_index - 1;
  }

  if (selected >= 0 && selected < static_cast<int>(templates.size())) {
    const auto& tmpl = templates[static_cast<size_t>(selected)];
    ImGui::TextWrapped("%s", tmpl.description.c_str());
    if (!tmpl.notes.empty()) {
      ImGui::TextWrapped("Notes: %s", tmpl.notes.c_str());
    }

    if (UIButton::Button("Apply template", UIButton::Size::Small, UIButton::Variant::Primary)) {
      ApplyTemplateToState(tmpl,
                           state.pde_text,
                           state.bound_xmin, state.bound_xmax,
                           state.bound_ymin, state.bound_ymax,
                           state.bound_zmin, state.bound_zmax,
                           state.grid_nx, state.grid_ny, state.grid_nz,
                           state.domain_mode, state.domain_shape,
                           state.coord_mode,
                           state.bc_left, state.bc_right,
                           state.bc_bottom, state.bc_top,
                           state.bc_front, state.bc_back,
                           state.method_index, state.sor_omega, state.gmres_restart,
                           state.time_start, state.time_end, state.time_frames);

      state.pde_preview.dirty = true;
      state.refresh_coord_flags();
      // #region agent log
      {
        std::ofstream f("/Users/nathaniel.sun/Desktop/programming/cursor/.cursor/debug.log",
                        std::ios::app);
        if (f) {
          const auto ts = std::chrono::duration_cast<std::chrono::milliseconds>(
                              std::chrono::system_clock::now().time_since_epoch())
                              .count();
          f << "{\"sessionId\":\"debug-session\",\"runId\":\"run9\",\"hypothesisId\":\"U\","
               "\"location\":\"gui_gl/panels/equation_panel.cpp:RenderTemplates\","
               "\"message\":\"Calling viewer.FitToView from Apply template\","
               "\"data\":{},\"timestamp\":" << ts << "}\n";
        }
      }
      // #endregion agent log
      state.viewer.FitToView();
    }
  }

  ImGui::TreePop();
}

}  // namespace

void RenderEquationPanel(EquationPanelState& state, const std::vector<std::string>& components) {
  const std::vector<std::string> default_components = DefaultComponents();
  const std::vector<std::string>& component_list =
      components.empty() ? default_components : components;

  for (size_t i = 0; i < component_list.size(); ++i) {
    const std::string& id = component_list[i];
    if (id == "pde_input") {
      RenderPdeInput(state);
    } else if (id == "pde_preview") {
      RenderPdePreview(state);
    } else if (id == "pde_templates") {
      RenderTemplates(state);
    } else {
      DrawUnknownComponentPlaceholder(id.c_str());
    }

    if (i + 1 < component_list.size()) {
      ImGui::Spacing();
    }
  }
}
