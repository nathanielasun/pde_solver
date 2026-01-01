#include "run_panel.h"
#include "ui_helpers.h"
#include "styles/ui_style.h"
#include "imgui.h"
#include <cfloat>
#include <algorithm>

namespace {

std::vector<std::string> DefaultComponents() {
  return {"run_controls"};
}

void RenderRunControls(RunPanelState& state) {
  if (state.running) {
    if (UIButton::Button("STOP SIMULATION", ImVec2(-1, 0), UIButton::Size::Large,
                         UIButton::Variant::Danger)) {
      if (state.on_stop) {
        state.on_stop();
      }
    }
    ImGui::TextDisabled("Press Esc to stop.");
    
    double clamped = std::max(0.0, std::min(1.0, state.progress));
    ImGui::ProgressBar(static_cast<float>(clamped), ImVec2(-1, 0));
    
    DetailedProgress detailed;
    {
      std::lock_guard<std::mutex> lock(state.state_mutex);
      detailed = state.state.detailed_progress;
    }
    
    ImGui::Text("Solver running...");
    if (state.phase == "solve") {
      ImGui::Text("Phase: Solving");
      if (detailed.total_iterations > 0) {
        ImGui::TextDisabled("Iteration: %d / %d (%.1f%%)",
                            detailed.current_iteration, detailed.total_iterations, clamped * 100.0);
      } else {
        ImGui::TextDisabled("Iteration: %d (%.1f%%)", detailed.current_iteration, clamped * 100.0);
      }
    } else if (state.phase == "time") {
      ImGui::Text("Phase: Time Stepping");
      ImGui::TextDisabled("Progress: %.1f%%", clamped * 100.0);
    } else if (state.phase == "write") {
      ImGui::Text("Phase: Writing Output");
      ImGui::TextDisabled("Progress: %.1f%%", clamped * 100.0);
    }
    
    if (detailed.iterations_per_second > 0) {
      ImGui::TextDisabled("Speed: %.1f iter/s", detailed.iterations_per_second);
    }
    if (detailed.elapsed_time > 0) {
      ImGui::TextDisabled("Elapsed: %s", detailed.FormatTime(detailed.elapsed_time).c_str());
    }
    if (detailed.estimated_remaining > 0 && detailed.estimated_remaining < 3600.0) {
      ImGui::TextDisabled("ETA: %s", detailed.FormatTime(detailed.estimated_remaining).c_str());
    }
    
    if (!detailed.backend_name.empty()) {
      ImGui::TextDisabled("Backend: %s", detailed.backend_name.c_str());
      if (!detailed.backend_note.empty()) {
        ImGui::TextDisabled("  %s", detailed.backend_note.c_str());
      }
    }
    
    if (detailed.is_converged) {
      ImGui::TextColored(ImVec4(0.3f, 1.0f, 0.3f, 1.0f), "✓ Converged");
    } else if (detailed.residual_l2 > 0) {
      ImGui::TextDisabled("Residual: L2=%.3e", detailed.residual_l2);
    }
    
    if (state.thread_total > 0) {
      ImGui::TextDisabled("Threads active: %d / %d", state.thread_active, state.thread_total);
    }
    if (!state.status.empty()) {
      ImGui::Text("Status: %s", state.status.c_str());
    }
    if (state.stability_warning) {
      ImGui::Spacing();
      ImGui::TextColored(ImVec4(1.0f, 0.7f, 0.3f, 1.0f),
                         "Stability warning: ||u||∞ grew too fast at frame %d.",
                         state.stability_frame);
      ImGui::TextDisabled("Last ratio r = %.3f, ||u||∞ = %.6g",
                          state.stability_ratio, state.stability_max);
      ImGui::TextDisabled("Heuristic: r = ||u||∞(n) / ||u||∞(n-1); warn if r > 1 + 0.35 for 3 steps.");
    }
  } else {
    if (UIButton::Button("Solve PDE", ImVec2(-1, 0), UIButton::Size::Large,
                         UIButton::Variant::Primary)) {
      if (state.on_solve) {
        state.on_solve();
      }
    }
    if (UIButton::Button("Load Latest VTK", ImVec2(-1, 0), UIButton::Size::Medium,
                         UIButton::Variant::Secondary)) {
      if (state.on_load_latest) {
        state.on_load_latest();
      }
    }
    
    if (state.has_duration) {
      ImGui::Spacing();
      ImGui::TextDisabled("Last run: %.2fs", state.last_duration);
    }
    if (state.phase == "solve") {
      ImGui::TextDisabled("Last: Solve completed");
    } else if (state.phase == "time") {
      ImGui::TextDisabled("Last: Time stepping completed");
    } else if (state.phase == "write") {
      ImGui::TextDisabled("Last: Write completed");
    }
    
    if (!state.residual_l2.empty()) {
      ImGui::Spacing();
      ImGui::Text("Convergence (residual norms)");
      ImGui::PlotLines("||Au-b||_2", state.residual_l2.data(),
                       static_cast<int>(state.residual_l2.size()),
                       0, nullptr, 0.0f, FLT_MAX, ImVec2(-1, 60));
      if (!state.residual_linf.empty()) {
        ImGui::PlotLines("||Au-b||_inf", state.residual_linf.data(),
                         static_cast<int>(state.residual_linf.size()),
                         0, nullptr, 0.0f, FLT_MAX, ImVec2(-1, 60));
      }
      ImGui::TextDisabled("Last: L2=%.3e, Linf=%.3e",
                          state.residual_l2.back(),
                          state.residual_linf.empty() ? 0.0f : state.residual_linf.back());
    }
  }
}

}  // namespace

void RenderRunPanel(RunPanelState& state, const std::vector<std::string>& components) {
  const std::vector<std::string> default_components = DefaultComponents();
  const std::vector<std::string>& component_list =
      components.empty() ? default_components : components;
  
  for (size_t i = 0; i < component_list.size(); ++i) {
    const std::string& id = component_list[i];
    if (id == "run_controls") {
      RenderRunControls(state);
    } else {
      DrawUnknownComponentPlaceholder(id.c_str());
    }
    if (i + 1 < component_list.size()) {
      ImGui::Spacing();
    }
  }
}
