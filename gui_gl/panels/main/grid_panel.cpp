#include "grid_panel.h"
#include "imgui.h"
#include "validation.h"
#include "ui_helpers.h"
#include "pde_types.h"
#include "systems/command_history.h"
#include "styles/ui_style.h"
#include <vector>

namespace {

std::vector<std::string> DefaultComponents() {
  return {"grid_resolution"};
}

void RenderGridResolution(GridPanelState& state) {
  const float avail = ImGui::GetContentRegionAvail().x;
  const bool compact = avail < 260.0f;
  const float field_width = std::max(100.0f, std::min(160.0f, state.input_width * (compact ? 0.9f : 0.45f)));
  
  ImGui::PushItemWidth(field_width);
  
  int old_nx = state.grid_nx;
  int old_ny = state.grid_ny;
  int old_nz = state.grid_nz;
  
  if (state.use_cartesian_3d) {
    if (UIInput::InputInt("nx", &state.grid_nx)) {
      if (state.cmd_history && state.grid_nx != old_nx) {
        auto cmd = std::make_unique<SetIntCommand>(&state.grid_nx, state.grid_nx, "Change nx");
        state.cmd_history->Execute(std::move(cmd));
      }
    }
    if (!compact) {
      ImGui::SameLine();
    }
    if (UIInput::InputInt("ny", &state.grid_ny)) {
      if (state.cmd_history && state.grid_ny != old_ny) {
        auto cmd = std::make_unique<SetIntCommand>(&state.grid_ny, state.grid_ny, "Change ny");
        state.cmd_history->Execute(std::move(cmd));
      }
    }
    UIInput::InputInt("nz", &state.grid_nz);
    if (state.cmd_history && state.grid_nz != old_nz) {
      auto cmd = std::make_unique<SetIntCommand>(&state.grid_nz, state.grid_nz, "Change nz");
      state.cmd_history->Execute(std::move(cmd));
    }
  } else if (state.use_cylindrical_volume) {
    if (UIInput::InputInt("n_r", &state.grid_nx)) {
      if (state.cmd_history && state.grid_nx != old_nx) {
        auto cmd = std::make_unique<SetIntCommand>(&state.grid_nx, state.grid_nx, "Change n_r");
        state.cmd_history->Execute(std::move(cmd));
      }
    }
    if (!compact) {
      ImGui::SameLine();
    }
    if (UIInput::InputInt("n_theta", &state.grid_ny)) {
      if (state.cmd_history && state.grid_ny != old_ny) {
        auto cmd = std::make_unique<SetIntCommand>(&state.grid_ny, state.grid_ny, "Change n_theta");
        state.cmd_history->Execute(std::move(cmd));
      }
    }
    UIInput::InputInt("n_z", &state.grid_nz);
    if (state.cmd_history && state.grid_nz != old_nz) {
      auto cmd = std::make_unique<SetIntCommand>(&state.grid_nz, state.grid_nz, "Change n_z");
      state.cmd_history->Execute(std::move(cmd));
    }
  } else if (state.use_volume) {
    if (UIInput::InputInt("n_r", &state.grid_nx)) {
      if (state.cmd_history && state.grid_nx != old_nx) {
        auto cmd = std::make_unique<SetIntCommand>(&state.grid_nx, state.grid_nx, "Change n_r");
        state.cmd_history->Execute(std::move(cmd));
      }
    }
    if (!compact) {
      ImGui::SameLine();
    }
    if (UIInput::InputInt("n_theta", &state.grid_ny)) {
      if (state.cmd_history && state.grid_ny != old_ny) {
        auto cmd = std::make_unique<SetIntCommand>(&state.grid_ny, state.grid_ny, "Change n_theta");
        state.cmd_history->Execute(std::move(cmd));
      }
    }
    UIInput::InputInt("n_phi", &state.grid_nz);
    if (state.cmd_history && state.grid_nz != old_nz) {
      auto cmd = std::make_unique<SetIntCommand>(&state.grid_nz, state.grid_nz, "Change n_phi");
      state.cmd_history->Execute(std::move(cmd));
    }
  } else if (state.use_surface) {
    if (UIInput::InputInt("n_theta", &state.grid_nx)) {
      if (state.cmd_history && state.grid_nx != old_nx) {
        auto cmd = std::make_unique<SetIntCommand>(&state.grid_nx, state.grid_nx, "Change n_theta");
        state.cmd_history->Execute(std::move(cmd));
      }
    }
    if (!compact) {
      ImGui::SameLine();
    }
    if (UIInput::InputInt("n_phi", &state.grid_ny)) {
      if (state.cmd_history && state.grid_ny != old_ny) {
        auto cmd = std::make_unique<SetIntCommand>(&state.grid_ny, state.grid_ny, "Change n_phi");
        state.cmd_history->Execute(std::move(cmd));
      }
    }
  } else if (state.use_axisymmetric) {
    if (UIInput::InputInt("n_r", &state.grid_nx)) {
      if (state.cmd_history && state.grid_nx != old_nx) {
        auto cmd = std::make_unique<SetIntCommand>(&state.grid_nx, state.grid_nx, "Change n_r");
        state.cmd_history->Execute(std::move(cmd));
      }
    }
    if (!compact) {
      ImGui::SameLine();
    }
    if (UIInput::InputInt("n_z", &state.grid_ny)) {
      if (state.cmd_history && state.grid_ny != old_ny) {
        auto cmd = std::make_unique<SetIntCommand>(&state.grid_ny, state.grid_ny, "Change n_z");
        state.cmd_history->Execute(std::move(cmd));
      }
    }
  } else if (state.use_polar_coords) {
    if (UIInput::InputInt("n_r", &state.grid_nx)) {
      if (state.cmd_history && state.grid_nx != old_nx) {
        auto cmd = std::make_unique<SetIntCommand>(&state.grid_nx, state.grid_nx, "Change n_r");
        state.cmd_history->Execute(std::move(cmd));
      }
    }
    if (!compact) {
      ImGui::SameLine();
    }
    if (UIInput::InputInt("n_theta", &state.grid_ny)) {
      if (state.cmd_history && state.grid_ny != old_ny) {
        auto cmd = std::make_unique<SetIntCommand>(&state.grid_ny, state.grid_ny, "Change n_theta");
        state.cmd_history->Execute(std::move(cmd));
      }
    }
  } else {
    if (UIInput::InputInt("nx", &state.grid_nx)) {
      if (state.cmd_history && state.grid_nx != old_nx) {
        auto cmd = std::make_unique<SetIntCommand>(&state.grid_nx, state.grid_nx, "Change nx");
        state.cmd_history->Execute(std::move(cmd));
      }
    }
    if (!compact) {
      ImGui::SameLine();
    }
    if (UIInput::InputInt("ny", &state.grid_ny)) {
      if (state.cmd_history && state.grid_ny != old_ny) {
        auto cmd = std::make_unique<SetIntCommand>(&state.grid_ny, state.grid_ny, "Change ny");
        state.cmd_history->Execute(std::move(cmd));
      }
    }
  }
  
  ImGui::PopItemWidth();
  
  Domain temp_domain;
  temp_domain.nx = state.grid_nx;
  temp_domain.ny = state.grid_ny;
  temp_domain.nz = state.use_volume ? state.grid_nz : 1;
  ValidationState grid_validation = ValidateGrid(state.grid_nx, state.grid_ny,
                                                 state.use_volume ? state.grid_nz : 1,
                                                 state.use_volume, temp_domain);
  if (grid_validation.grid_status != ValidationStatus::Valid) {
    ImGui::SameLine();
    DrawValidationIndicator(grid_validation.grid_status, "", grid_validation.grid_warning);
    if (ImGui::IsItemHovered() && !grid_validation.grid_warning.empty()) {
      ImGui::BeginTooltip();
      ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.3f, 1.0f), "%s", grid_validation.grid_warning.c_str());
      ImGui::EndTooltip();
    }
  }
}

}  // namespace

void RenderGridPanel(GridPanelState& state, const std::vector<std::string>& components) {
  const std::vector<std::string> default_components = DefaultComponents();
  const std::vector<std::string>& component_list =
      components.empty() ? default_components : components;
  
  for (size_t i = 0; i < component_list.size(); ++i) {
    const std::string& id = component_list[i];
    if (id == "grid_resolution") {
      RenderGridResolution(state);
    } else {
      DrawUnknownComponentPlaceholder(id.c_str());
    }
    if (i + 1 < component_list.size()) {
      ImGui::Spacing();
    }
  }
}
