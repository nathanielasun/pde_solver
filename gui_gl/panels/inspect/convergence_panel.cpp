#include "convergence_panel.h"
#include "components/plot_widget.h"
#include "imgui.h"
#include <fstream>
#include <sstream>
#include <iomanip>
#include <algorithm>

// Static state for persistent UI
static bool s_show_l2 = true;
static bool s_show_linf = true;
static bool s_log_scale = true;
static PlotWidget s_plot;
static bool s_plot_initialized = false;

void RenderConvergencePanel(ConvergencePanelState& state, const std::vector<std::string>& components) {
  // Initialize plot widget once
  if (!s_plot_initialized) {
    s_plot.AddSeries("L2 Norm", IM_COL32(100, 180, 255, 255));
    s_plot.AddSeries("L-inf Norm", IM_COL32(255, 150, 100, 255));
    s_plot.title = "Residual Convergence";
    s_plot.x_label = "Iteration";
    s_plot.y_label = "Residual";
    s_plot_initialized = true;
  }

  // Controls row
  ImGui::Text("Display:");
  ImGui::SameLine();
  ImGui::Checkbox("L2", &s_show_l2);
  ImGui::SameLine();
  ImGui::Checkbox("L-inf", &s_show_linf);
  ImGui::SameLine();
  ImGui::Checkbox("Log Scale", &s_log_scale);

  ImGui::Spacing();

  // Get data with lock
  std::vector<float> l2_data, linf_data;
  {
    std::lock_guard<std::mutex> lock(state.state_mutex);
    l2_data = state.residual_l2;
    linf_data = state.residual_linf;
  }

  // Build x-axis (iteration numbers)
  std::vector<float> x_l2(l2_data.size());
  std::vector<float> x_linf(linf_data.size());
  for (size_t i = 0; i < l2_data.size(); ++i) {
    x_l2[i] = static_cast<float>(i);
  }
  for (size_t i = 0; i < linf_data.size(); ++i) {
    x_linf[i] = static_cast<float>(i);
  }

  // Update plot data
  s_plot.log_scale_y = s_log_scale;

  if (s_show_l2 && s_plot.GetSeriesCount() > 0) {
    s_plot.SetSeriesData(0, x_l2, l2_data);
  } else if (s_plot.GetSeriesCount() > 0) {
    s_plot.ClearSeries(0);
  }

  if (s_show_linf && s_plot.GetSeriesCount() > 1) {
    s_plot.SetSeriesData(1, x_linf, linf_data);
  } else if (s_plot.GetSeriesCount() > 1) {
    s_plot.ClearSeries(1);
  }

  // Render plot
  float available_height = ImGui::GetContentRegionAvail().y - 80.0f;
  if (available_height < 100.0f) available_height = 100.0f;
  s_plot.Render(state.input_width, available_height);

  ImGui::Spacing();
  ImGui::Separator();

  // Statistics
  if (!l2_data.empty() || !linf_data.empty()) {
    ImGui::Text("Statistics:");

    if (!l2_data.empty()) {
      float min_l2 = *std::min_element(l2_data.begin(), l2_data.end());
      float max_l2 = *std::max_element(l2_data.begin(), l2_data.end());
      float final_l2 = l2_data.back();
      ImGui::Text("  L2:   Min=%.2e  Max=%.2e  Final=%.2e", min_l2, max_l2, final_l2);
    }

    if (!linf_data.empty()) {
      float min_linf = *std::min_element(linf_data.begin(), linf_data.end());
      float max_linf = *std::max_element(linf_data.begin(), linf_data.end());
      float final_linf = linf_data.back();
      ImGui::Text("  Linf: Min=%.2e  Max=%.2e  Final=%.2e", min_linf, max_linf, final_linf);
    }

    ImGui::Text("  Iterations: %zu", std::max(l2_data.size(), linf_data.size()));
  } else {
    ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "No convergence data available.");
    ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "Run a solve to see residual history.");
  }

  ImGui::Spacing();

  // Export button
  if (ImGui::Button("Export to CSV", ImVec2(-1, 0))) {
    // Simple CSV export
    std::ostringstream csv;
    csv << "Iteration,L2_Residual,Linf_Residual\n";
    size_t max_len = std::max(l2_data.size(), linf_data.size());
    for (size_t i = 0; i < max_len; ++i) {
      csv << i << ",";
      if (i < l2_data.size()) {
        csv << std::scientific << std::setprecision(6) << l2_data[i];
      }
      csv << ",";
      if (i < linf_data.size()) {
        csv << std::scientific << std::setprecision(6) << linf_data[i];
      }
      csv << "\n";
    }

    // Copy to clipboard (ImGui doesn't have native file dialog, so we use clipboard)
    ImGui::SetClipboardText(csv.str().c_str());
    ImGui::OpenPopup("CSV Exported");
  }

  // Popup notification
  if (ImGui::BeginPopup("CSV Exported")) {
    ImGui::Text("CSV data copied to clipboard!");
    ImGui::EndPopup();
  }

  // Clear button
  if (ImGui::Button("Clear History", ImVec2(-1, 0))) {
    std::lock_guard<std::mutex> lock(state.state_mutex);
    state.residual_l2.clear();
    state.residual_linf.clear();
  }
}
