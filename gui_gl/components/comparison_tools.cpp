#include "comparison_tools.h"
#include "io/file_utils.h"
#include "vtk_io.h"
#include "imgui.h"
#include "imgui_stdlib.h"
#include <sstream>
#include <iomanip>
#include <cmath>
#include <filesystem>

ComparisonToolsComponent::ComparisonToolsComponent() = default;

void ComparisonToolsComponent::Render() {
  if (!IsVisible()) {
    return;
  }
  
  // Note: This component is now rendered inside a tab, not as a standalone window
  // No ImGui::Begin/End needed - caller provides the context
  
  // Mode selection
  const char* modes[] = {"File Comparison", "Time Step Comparison"};
  ImGui::Combo("Mode", &time_step_mode_, modes, 2);
  ImGui::Separator();
  
  if (time_step_mode_ == 0) {
    RenderFileComparison();
  } else {
    RenderTimeStepComparison();
  }
  
  ImGui::Separator();
  
  // Statistics
  if (IsComparisonReady()) {
    RenderStatistics();
  }
}

void ComparisonToolsComponent::RenderFileComparison() {
  // Solution A
  ImGui::Text("Solution A (Reference):");
  ImGui::SetNextItemWidth(-60);  // Leave room for button
  ImGui::InputText("##path_a", &solution_a_path_);
  ImGui::SameLine();
  if (ImGui::Button("Load##a", ImVec2(50, 0))) {
    if (!solution_a_path_.empty() && std::filesystem::exists(solution_a_path_)) {
      if (LoadSolutionA(solution_a_path_)) {
        if (auto_update_) {
          UpdateComparison();
        }
      }
    }
  }

  ImGui::Spacing();

  // Solution B
  ImGui::Text("Solution B (Compare):");
  ImGui::SetNextItemWidth(-60);  // Leave room for button
  ImGui::InputText("##path_b", &solution_b_path_);
  ImGui::SameLine();
  if (ImGui::Button("Load##b", ImVec2(50, 0))) {
    if (!solution_b_path_.empty() && std::filesystem::exists(solution_b_path_)) {
      if (LoadSolutionB(solution_b_path_)) {
        if (auto_update_) {
          UpdateComparison();
        }
      }
    }
  }

  ImGui::Spacing();
  ImGui::Separator();
  ImGui::Spacing();

  // Controls on separate row
  ImGui::Checkbox("Auto-update on load", &auto_update_);
  if (ImGui::Button("Update Comparison", ImVec2(-1, 0))) {
    UpdateComparison();
  }

  ImGui::Spacing();

  if (comparator_.IsReady()) {
    if (!comparator_.AreDomainsCompatible()) {
      ImGui::TextColored(ImVec4(1.0f, 0.3f, 0.3f, 1.0f), "Error: Domains not compatible");
      ImGui::TextWrapped("Solutions must have matching grid dimensions and coordinate systems.");
    }
  }
}

void ComparisonToolsComponent::RenderTimeStepComparison() {
  if (!current_domain_.has_value() || !current_grid_.has_value()) {
    ImGui::Text("No current solution available.");
    ImGui::Text("Solve a time-dependent PDE to enable time step comparison.");
    return;
  }
  
  ImGui::Text("Time Step Comparison");
  ImGui::Text("Compare current frame with previous/next frame.");
  ImGui::Text("(Feature requires time series data)");
  
  // TODO: Implement time step comparison when time series support is added
  ImGui::TextColored(ImVec4(1.0f, 0.5f, 0.0f, 1.0f), "Time step comparison coming soon...");
}

void ComparisonToolsComponent::RenderStatistics() {
  ImGui::Text("Comparison Statistics:");
  
  ImGui::Text("Difference Field:");
  ImGui::Text("  Min: %s", FormatNumber(statistics_.min_diff).c_str());
  ImGui::Text("  Max: %s", FormatNumber(statistics_.max_diff).c_str());
  ImGui::Text("  Mean: %s", FormatNumber(statistics_.mean_diff).c_str());
  ImGui::Text("  L2 norm: %s", FormatNumber(statistics_.l2_diff).c_str());
  ImGui::Text("  L-infinity norm: %s", FormatNumber(statistics_.linf_diff).c_str());
  
  ImGui::Separator();
  
  ImGui::Text("Relative Error:");
  ImGui::Text("  L2 norm: %s", FormatNumber(statistics_.relative_error_l2).c_str());
  ImGui::Text("  L-infinity norm: %s", FormatNumber(statistics_.relative_error_linf).c_str());
  
  ImGui::Separator();
  
  ImGui::Text("Valid points: %zu / %zu", statistics_.valid_points, statistics_.total_points);
  
  // Visualization options
  ImGui::Separator();
  ImGui::Text("Visualization:");
  ImGui::Checkbox("Show Difference", &show_difference_);
  ImGui::Checkbox("Show Relative Error", &show_relative_error_);
  
  if (show_difference_ && !difference_field_.empty() && viewer_) {
    // TODO: Set difference field in viewer
    ImGui::Text("Difference field ready for visualization");
  }
  
  if (show_relative_error_ && !relative_error_field_.empty() && viewer_) {
    // TODO: Set relative error field in viewer
    ImGui::Text("Relative error field ready for visualization");
  }
}

void ComparisonToolsComponent::SetCurrentSolution(const Domain& domain, const std::vector<double>& grid) {
  current_domain_ = domain;
  current_grid_ = grid;
}

bool ComparisonToolsComponent::LoadSolutionA(const std::string& filepath) {
  VtkReadResult result = ReadVtkFile(filepath);
  if (!result.ok) {
    return false;
  }
  
  if (result.kind != VtkReadResult::Kind::StructuredPoints) {
    return false;  // Only support structured points for now
  }
  
  comparator_.SetSolutionA(result.domain, result.grid);
  solution_a_path_ = filepath;
  
  return true;
}

bool ComparisonToolsComponent::LoadSolutionB(const std::string& filepath) {
  VtkReadResult result = ReadVtkFile(filepath);
  if (!result.ok) {
    return false;
  }
  
  if (result.kind != VtkReadResult::Kind::StructuredPoints) {
    return false;  // Only support structured points for now
  }
  
  comparator_.SetSolutionB(result.domain, result.grid);
  solution_b_path_ = filepath;
  
  return true;
}

void ComparisonToolsComponent::UpdateComparison() {
  if (!IsComparisonReady()) {
    return;
  }
  
  difference_field_ = comparator_.ComputeDifference();
  relative_error_field_ = comparator_.ComputeRelativeError();
  statistics_ = comparator_.ComputeStatistics();
}

void ComparisonToolsComponent::ClearAll() {
  comparator_.ClearAll();
  difference_field_.clear();
  relative_error_field_.clear();
  statistics_ = ComparisonStatistics();
  solution_a_path_.clear();
  solution_b_path_.clear();
}

std::string ComparisonToolsComponent::FormatNumber(double value, int precision) const {
  std::ostringstream oss;
  oss << std::scientific << std::setprecision(precision) << value;
  return oss.str();
}

