#include "comparison_tools.h"
#include "io/file_utils.h"
#include "vtk_io.h"
#include "imgui.h"
#include "imgui_stdlib.h"
#include <algorithm>
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

  if (frame_paths_.empty()) {
    ImGui::Text("No time series data available.");
    ImGui::Text("Run a time-dependent solve to generate frames.");
    return;
  }

  const int frameCount = static_cast<int>(frame_paths_.size());

  ImGui::Text("Time Step Comparison");
  ImGui::Separator();
  ImGui::Text("Current frame: %d / %d", current_frame_index_, frameCount - 1);
  if (current_frame_index_ >= 0 && current_frame_index_ < frameCount) {
    ImGui::TextWrapped("File: %s", frame_paths_[current_frame_index_].c_str());
  }

  ImGui::Spacing();
  ImGui::Text("Compare with:");
  ImGui::RadioButton("Previous Frame", &comparison_mode_, 0);
  ImGui::SameLine();
  ImGui::RadioButton("Next Frame", &comparison_mode_, 1);
  ImGui::RadioButton("Specific Frame", &comparison_mode_, 2);

  int targetFrame = -1;
  if (comparison_mode_ == 0) {
    targetFrame = current_frame_index_ - 1;
    if (targetFrame < 0) {
      ImGui::TextColored(ImVec4(1.0f, 0.5f, 0.0f, 1.0f),
                         "No previous frame (already at first frame).");
    }
  } else if (comparison_mode_ == 1) {
    targetFrame = current_frame_index_ + 1;
    if (targetFrame >= frameCount) {
      ImGui::TextColored(ImVec4(1.0f, 0.5f, 0.0f, 1.0f),
                         "No next frame (already at last frame).");
    }
  } else {
    ImGui::SetNextItemWidth(120);
    ImGui::InputInt("Frame##specific", &specific_frame_index_);
    specific_frame_index_ = std::clamp(specific_frame_index_, 0, frameCount - 1);
    targetFrame = specific_frame_index_;
    if (targetFrame == current_frame_index_) {
      ImGui::TextColored(ImVec4(1.0f, 0.5f, 0.0f, 1.0f),
                         "Target frame is the same as current frame.");
    }
  }

  ImGui::Spacing();

  const bool canCompare = targetFrame >= 0 && targetFrame < frameCount &&
                          targetFrame != current_frame_index_;

  if (!canCompare) {
    ImGui::BeginDisabled();
  }
  if (ImGui::Button("Compare", ImVec2(-1, 0))) {
    // Load comparison frame from VTK file
    VtkReadResult compResult = ReadVtkFile(frame_paths_[targetFrame]);
    if (compResult.ok && compResult.kind == VtkReadResult::Kind::StructuredPoints) {
      comparator_.SetSolutionA(current_domain_.value(), current_grid_.value());
      comparator_.SetSolutionB(compResult.domain, compResult.grid);
      UpdateComparison();
    } else {
      ImGui::TextColored(ImVec4(1.0f, 0.3f, 0.3f, 1.0f),
                         "Failed to load comparison frame.");
    }
  }
  if (!canCompare) {
    ImGui::EndDisabled();
  }

  if (comparator_.IsReady() && !comparator_.AreDomainsCompatible()) {
    ImGui::TextColored(ImVec4(1.0f, 0.3f, 0.3f, 1.0f), "Error: Domains not compatible");
    ImGui::TextWrapped("Frames must have matching grid dimensions and coordinate systems.");
  }
}

void ComparisonToolsComponent::SetFramePaths(const std::vector<std::string>& paths) {
  frame_paths_ = paths;
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
    if (ImGui::Button("Display Difference Field", ImVec2(-1, 0))) {
      Domain compDomain = comparator_.GetDomain();
      viewer_->SetData(compDomain, difference_field_);
    }
  }

  if (show_relative_error_ && !relative_error_field_.empty() && viewer_) {
    if (ImGui::Button("Display Relative Error Field", ImVec2(-1, 0))) {
      Domain compDomain = comparator_.GetDomain();
      viewer_->SetData(compDomain, relative_error_field_);
    }
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

