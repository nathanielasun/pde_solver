#include "viewer_panel.h"
#include "ui_helpers.h"
#include "styles/ui_style.h"
#include "imgui.h"
// #region agent log
#include <fstream>
#include <chrono>
// #endregion agent log

namespace {

std::vector<std::string> DefaultComponents() {
  return {"viewer_controls"};
}

void RenderViewerControls(ViewerPanelState& state) {
  if (ImGui::Checkbox("Orthographic projection", &state.use_ortho)) {
    state.viewer.SetOrthographic(state.use_ortho);
    // #region agent log
    {
      std::ofstream f("/Users/nathaniel.sun/Desktop/programming/cursor/.cursor/debug.log",
                      std::ios::app);
      if (f) {
        const auto ts = std::chrono::duration_cast<std::chrono::milliseconds>(
                            std::chrono::system_clock::now().time_since_epoch())
                            .count();
        f << "{\"sessionId\":\"debug-session\",\"runId\":\"run9\",\"hypothesisId\":\"T\","
             "\"location\":\"gui_gl/panels/viewer_panel.cpp:RenderViewerControls\","
             "\"message\":\"Calling viewer.FitToView from orthographic toggle\","
             "\"data\":{\"use_ortho\":"
          << (state.use_ortho ? "true" : "false") << "},\"timestamp\":" << ts << "}\n";
      }
    }
    // #endregion agent log
    state.viewer.FitToView();
    state.point_scale = 1.0f;
  }
  if (state.use_ortho) {
    ImGui::SameLine();
    if (ImGui::Checkbox("Keep object in view", &state.lock_fit)) {
      if (state.lock_fit && state.point_scale < 1.0f) {
        state.point_scale = 1.0f;
      }
    }
  }
  UIInput::InputInt("Stride", &state.stride);
  if (state.stride < 1) {
    state.stride = 1;
  }
  if (state.stride > 16) {
    state.stride = 16;
  }
  state.viewer.SetStride(state.stride);
  
  const char* zoom_label = state.use_ortho ? "Zoom (view scale)" : "View scale (zoom out)";
  UIInput::SliderFloat(zoom_label, &state.point_scale, 0.1f, 50.0f, "%.2f");
  if (state.use_ortho && state.lock_fit && state.point_scale < 1.0f) {
    state.point_scale = 1.0f;
  }
  state.viewer.SetPointScale(state.point_scale);
  UIInput::SliderFloat("Data scale", &state.data_scale, 0.01f, 10.0f, "%.3f");
  state.viewer.SetDataScale(state.data_scale);
  
  const bool view_is_surface =
      state.coord_mode == CoordMode::kSphericalSurface || state.coord_mode == CoordMode::kToroidalSurface;
  const bool view_is_volume =
      state.coord_mode == CoordMode::kSphericalVolume || state.coord_mode == CoordMode::kToroidalVolume ||
      state.coord_mode == CoordMode::kCartesian3D || state.coord_mode == CoordMode::kCylindricalVolume;
  const bool allow_value_lock = !(view_is_surface || view_is_volume);
  if (allow_value_lock) {
    bool z_domain_changed = false;
    if (ImGui::Checkbox("Lock Z domain (value)", &state.z_domain_locked)) {
      z_domain_changed = true;
    }
    if (state.z_domain_locked) {
      const float avail = ImGui::GetContentRegionAvail().x;
      const bool compact = avail < 260.0f;
      const float field_width = std::max(100.0f, std::min(160.0f, state.input_width * (compact ? 0.9f : 0.45f)));
      ImGui::PushItemWidth(field_width);
      if (UIInput::InputDouble("z min", &state.z_domain_min, 0.0, 0.0, "%.6g")) {
        z_domain_changed = true;
      }
      if (!compact) {
        ImGui::SameLine();
      }
      if (UIInput::InputDouble("z max", &state.z_domain_max, 0.0, 0.0, "%.6g")) {
        z_domain_changed = true;
      }
      ImGui::PopItemWidth();
      if (state.z_domain_max <= state.z_domain_min) {
        ImGui::TextColored(ImVec4(1.0f, 0.55f, 0.55f, 1.0f),
                           "z max must be > z min.");
      }
    }
    if (z_domain_changed) {
      state.viewer.SetZDomain(state.z_domain_locked, state.z_domain_min, state.z_domain_max);
    }
  } else if (state.z_domain_locked) {
    state.z_domain_locked = false;
    state.viewer.SetZDomain(false, state.z_domain_min, state.z_domain_max);
  }
  ImGui::Checkbox("Show domain grid", &state.grid_enabled);
  state.viewer.SetGridEnabled(state.grid_enabled);
  UIInput::SliderInt("Grid divisions", &state.grid_divisions, 2, 24);
  state.viewer.SetGridDivisions(state.grid_divisions);
}

}  // namespace

void RenderViewerPanel(ViewerPanelState& state, const std::vector<std::string>& components) {
  const std::vector<std::string> default_components = DefaultComponents();
  const std::vector<std::string>& component_list =
      components.empty() ? default_components : components;
  
  for (size_t i = 0; i < component_list.size(); ++i) {
    const std::string& id = component_list[i];
    if (id == "viewer_controls") {
      RenderViewerControls(state);
    } else {
      DrawUnknownComponentPlaceholder(id.c_str());
    }
    if (i + 1 < component_list.size()) {
      ImGui::Spacing();
    }
  }
}
