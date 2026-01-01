#include "statistics_panel.h"
#include "tools/statistics_compute.h"
#include "imgui.h"
#include <algorithm>
#include <cmath>
#include <sstream>
#include <iomanip>

// Static state
static int s_field_index = 0;
static int s_histogram_bins = 20;
static FieldStatistics s_stats;
static Histogram s_histogram;
static bool s_stats_valid = false;
static std::vector<double> s_last_grid;

static const char* GetFieldName(int index) {
  static const char* field_names[] = {
    "Solution (u)",
    "Gradient X",
    "Gradient Y",
    "Gradient Z",
    "Laplacian",
    "Flux X",
    "Flux Y",
    "Flux Z",
    "Energy Norm"
  };
  if (index >= 0 && index < 9) return field_names[index];
  return "Unknown";
}

static const std::vector<double>* GetFieldData(int index,
                                                const std::vector<double>& grid,
                                                const DerivedFields& derived,
                                                bool has_derived) {
  if (index == 0) return &grid;
  if (!has_derived) return nullptr;

  switch (index) {
    case 1: return &derived.gradient_x;
    case 2: return &derived.gradient_y;
    case 3: return &derived.gradient_z;
    case 4: return &derived.laplacian;
    case 5: return &derived.flux_x;
    case 6: return &derived.flux_y;
    case 7: return &derived.flux_z;
    case 8: return &derived.energy_norm;
    default: return nullptr;
  }
}

void RenderStatisticsPanel(StatisticsPanelState& state, const std::vector<std::string>& components) {
  // Get data with lock
  std::vector<double> grid_copy;
  DerivedFields derived_copy;
  bool has_derived;
  Domain domain_copy;
  {
    std::lock_guard<std::mutex> lock(state.state_mutex);
    grid_copy = state.current_grid;
    derived_copy = state.derived_fields;
    has_derived = state.has_derived_fields;
    domain_copy = state.current_domain;
  }

  if (grid_copy.empty()) {
    ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "No solution data available.");
    ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "Solve a PDE to see statistics.");
    return;
  }

  // Field selector
  ImGui::Text("Field:");
  ImGui::SameLine();
  ImGui::SetNextItemWidth(state.input_width * 0.6f);
  if (ImGui::Combo("##field_select", &s_field_index,
                   "Solution (u)\0Gradient X\0Gradient Y\0Gradient Z\0"
                   "Laplacian\0Flux X\0Flux Y\0Flux Z\0Energy Norm\0")) {
    s_stats_valid = false;
  }

  // Check if data changed
  if (grid_copy != s_last_grid) {
    s_stats_valid = false;
    s_last_grid = grid_copy;
  }

  // Get selected field data
  const std::vector<double>* field_data = GetFieldData(s_field_index, grid_copy, derived_copy, has_derived);

  if (!field_data || field_data->empty()) {
    ImGui::TextColored(ImVec4(1.0f, 0.5f, 0.3f, 1.0f), "Field not available.");
    if (s_field_index > 0 && !has_derived) {
      ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "Derived fields not computed.");
    }
    return;
  }

  // Compute statistics if needed
  if (!s_stats_valid) {
    s_stats = ComputeStatistics(*field_data);
    s_histogram = ComputeHistogram(*field_data, s_histogram_bins);
    s_stats_valid = true;
  }

  ImGui::Spacing();
  ImGui::Separator();

  // Statistics table
  ImGui::Text("Field Statistics:");
  ImGui::Spacing();

  if (ImGui::BeginTable("stats_table", 2, ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg)) {
    ImGui::TableSetupColumn("Metric", ImGuiTableColumnFlags_WidthFixed, 100.0f);
    ImGui::TableSetupColumn("Value", ImGuiTableColumnFlags_WidthStretch);
    ImGui::TableHeadersRow();

    auto AddRow = [](const char* label, double value) {
      ImGui::TableNextRow();
      ImGui::TableNextColumn();
      ImGui::Text("%s", label);
      ImGui::TableNextColumn();
      if (std::abs(value) < 1e-3 || std::abs(value) >= 1e4) {
        ImGui::Text("%.6e", value);
      } else {
        ImGui::Text("%.6f", value);
      }
    };

    AddRow("Minimum", s_stats.min);
    AddRow("Maximum", s_stats.max);
    AddRow("Mean", s_stats.mean);
    AddRow("Median", s_stats.median);
    AddRow("Std Dev", s_stats.stddev);
    AddRow("RMS", s_stats.rms);
    AddRow("L2 Norm", s_stats.l2_norm);

    ImGui::TableNextRow();
    ImGui::TableNextColumn();
    ImGui::Text("Count");
    ImGui::TableNextColumn();
    ImGui::Text("%zu", s_stats.count);

    ImGui::EndTable();
  }

  ImGui::Spacing();
  ImGui::Separator();

  // Domain info
  ImGui::Text("Domain Info:");
  ImGui::Text("  Grid: %d x %d x %d", domain_copy.nx, domain_copy.ny, domain_copy.nz);
  ImGui::Text("  X: [%.3f, %.3f]", domain_copy.xmin, domain_copy.xmax);
  ImGui::Text("  Y: [%.3f, %.3f]", domain_copy.ymin, domain_copy.ymax);
  if (domain_copy.nz > 1) {
    ImGui::Text("  Z: [%.3f, %.3f]", domain_copy.zmin, domain_copy.zmax);
  }

  ImGui::Spacing();
  ImGui::Separator();

  // Histogram
  ImGui::Text("Histogram:");
  ImGui::SameLine();
  ImGui::SetNextItemWidth(80.0f);
  if (ImGui::InputInt("Bins##hist", &s_histogram_bins)) {
    s_histogram_bins = std::clamp(s_histogram_bins, 5, 100);
    s_stats_valid = false;
  }

  if (!s_histogram.counts.empty()) {
    // Find max count for scaling
    int max_count = *std::max_element(s_histogram.counts.begin(), s_histogram.counts.end());
    if (max_count == 0) max_count = 1;

    // Draw histogram bars
    float bar_width = (state.input_width - 20.0f) / static_cast<float>(s_histogram.counts.size());
    float max_height = 80.0f;

    ImVec2 cursor = ImGui::GetCursorScreenPos();
    ImDrawList* draw_list = ImGui::GetWindowDrawList();

    for (size_t i = 0; i < s_histogram.counts.size(); ++i) {
      float height = (static_cast<float>(s_histogram.counts[i]) / static_cast<float>(max_count)) * max_height;
      float x = cursor.x + static_cast<float>(i) * bar_width;
      float y = cursor.y + max_height - height;

      draw_list->AddRectFilled(
        ImVec2(x, y),
        ImVec2(x + bar_width - 1, cursor.y + max_height),
        IM_COL32(100, 150, 200, 255)
      );
    }

    // Reserve space
    ImGui::Dummy(ImVec2(state.input_width - 20.0f, max_height + 5.0f));

    // Min/max labels under histogram
    ImGui::Text("%.2e", s_histogram.min);
    ImGui::SameLine(state.input_width - 80.0f);
    ImGui::Text("%.2e", s_histogram.max);
  }

  ImGui::Spacing();

  // Export button
  if (ImGui::Button("Copy Statistics", ImVec2(-1, 0))) {
    std::ostringstream oss;
    oss << "Field: " << GetFieldName(s_field_index) << "\n";
    oss << "Min: " << std::scientific << std::setprecision(6) << s_stats.min << "\n";
    oss << "Max: " << s_stats.max << "\n";
    oss << "Mean: " << s_stats.mean << "\n";
    oss << "Median: " << s_stats.median << "\n";
    oss << "StdDev: " << s_stats.stddev << "\n";
    oss << "RMS: " << s_stats.rms << "\n";
    oss << "L2 Norm: " << s_stats.l2_norm << "\n";
    oss << "Count: " << s_stats.count << "\n";
    ImGui::SetClipboardText(oss.str().c_str());
  }
}
