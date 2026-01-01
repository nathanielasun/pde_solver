#include "inspection_tools.h"

#include "imgui.h"
#include "vtk_io.h"
#include "tools/statistics_compute.h"
#include "tools/selection_tool.h"
#include <algorithm>
#include <cmath>
#include <sstream>
#include <iomanip>
#include <cfloat>

namespace {

const char* kFieldNamesVolume[] = {
  "Solution (u)",
  "Gradient X",
  "Gradient Y",
  "Gradient Z",
  "Laplacian",
  "Flux X",
  "Flux Y",
  "Flux Z",
  "Energy Norm (u^2)"
};

const char* kFieldNamesSurface[] = {
  "Solution (u)",
  "Gradient X",
  "Gradient Y",
  "Laplacian",
  "Flux X",
  "Flux Y",
  "Energy Norm (u^2)"
};

}  // namespace

static std::unique_ptr<InspectionToolsComponent> g_inspection_component;

InspectionToolsComponent* GetInspectionComponentSingleton() {
  if (!g_inspection_component) {
    g_inspection_component = std::make_unique<InspectionToolsComponent>();
  }
  return g_inspection_component.get();
}

InspectionToolsComponent::InspectionToolsComponent() {
  visible_ = true;
}

void InspectionToolsComponent::SetData(const Domain* domain, 
                                       const std::vector<double>* grid,
                                       const struct DerivedFields* derived) {
  domain_ = domain;
  grid_ = grid;
  derived_fields_ = derived;
  has_derived_ = (derived != nullptr);
  stats_dirty_ = true;
  UpdateProbeValues();
  UpdateLinePlots();
}

void InspectionToolsComponent::AddSlice(const SlicePlane& slice) {
  slices_.push_back(slice);
}

void InspectionToolsComponent::RemoveSlice(size_t index) {
  if (index < slices_.size()) {
    slices_.erase(slices_.begin() + index);
  }
}

void InspectionToolsComponent::ClearSlices() {
  slices_.clear();
}

void InspectionToolsComponent::AddProbe(const ProbePoint& probe) {
  probes_.push_back(probe);
  UpdateProbeValues();
}

void InspectionToolsComponent::RemoveProbe(size_t index) {
  if (index < probes_.size()) {
    probes_.erase(probes_.begin() + index);
  }
}

void InspectionToolsComponent::ClearProbes() {
  probes_.clear();
}

void InspectionToolsComponent::AddLinePlot(const LinePlot& plot) {
  line_plots_.push_back(plot);
  UpdateLinePlots();
}

void InspectionToolsComponent::RemoveLinePlot(size_t index) {
  if (index < line_plots_.size()) {
    line_plots_.erase(line_plots_.begin() + index);
  }
}

void InspectionToolsComponent::ClearLinePlots() {
  line_plots_.clear();
}

double InspectionToolsComponent::GetValueAtPoint(double x, double y, double z) const {
  if (!domain_ || !grid_ || grid_->empty()) {
    return 0.0;
  }
  
  const int nx = domain_->nx;
  const int ny = domain_->ny;
  const int nz = std::max(1, domain_->nz);
  
  // Clamp coordinates to domain
  x = std::max(domain_->xmin, std::min(domain_->xmax, x));
  y = std::max(domain_->ymin, std::min(domain_->ymax, y));
  z = std::max(domain_->zmin, std::min(domain_->zmax, z));
  
  // Compute grid indices
  const double dx = (domain_->xmax - domain_->xmin) / std::max(1, nx - 1);
  const double dy = (domain_->ymax - domain_->ymin) / std::max(1, ny - 1);
  const double dz = (nz > 1) ? ((domain_->zmax - domain_->zmin) / std::max(1, nz - 1)) : 0.0;
  
  int i = static_cast<int>((x - domain_->xmin) / dx + 0.5);
  int j = static_cast<int>((y - domain_->ymin) / dy + 0.5);
  int k = (nz > 1) ? static_cast<int>((z - domain_->zmin) / dz + 0.5) : 0;
  
  i = std::max(0, std::min(nx - 1, i));
  j = std::max(0, std::min(ny - 1, j));
  k = std::max(0, std::min(nz - 1, k));
  
  const size_t idx = static_cast<size_t>((k * ny + j) * nx + i);
  if (idx < grid_->size()) {
    return (*grid_)[idx];
  }
  return 0.0;
}

void InspectionToolsComponent::UpdateProbeValues() {
  if (!domain_ || !grid_ || grid_->empty()) {
    return;
  }
  
  for (auto& probe : probes_) {
    if (probe.active) {
      probe.value = GetValueAtPoint(probe.x, probe.y, probe.z);
    }
  }
}

void InspectionToolsComponent::ComputeLinePlot(LinePlot& plot) const {
  if (!domain_ || !grid_ || grid_->empty() || !plot.enabled) {
    plot.positions.clear();
    plot.values.clear();
    return;
  }
  
  plot.positions.resize(plot.num_points);
  plot.values.resize(plot.num_points);
  
  const double dx = plot.x1 - plot.x0;
  const double dy = plot.y1 - plot.y0;
  const double dz = plot.z1 - plot.z0;
  const double length = std::sqrt(dx*dx + dy*dy + dz*dz);
  
  if (length < 1e-12) {
    // Degenerate line
    for (int i = 0; i < plot.num_points; ++i) {
      plot.positions[i] = 0.0;
      plot.values[i] = GetValueAtPoint(plot.x0, plot.y0, plot.z0);
    }
    return;
  }
  
  for (int i = 0; i < plot.num_points; ++i) {
    const double t = static_cast<double>(i) / std::max(1, plot.num_points - 1);
    plot.positions[i] = t * length;
    
    const double x = plot.x0 + t * dx;
    const double y = plot.y0 + t * dy;
    const double z = plot.z0 + t * dz;
    
    plot.values[i] = GetValueAtPoint(x, y, z);
  }
}

void InspectionToolsComponent::UpdateLinePlots() {
  for (auto& plot : line_plots_) {
    ComputeLinePlot(plot);
  }
}

std::string InspectionToolsComponent::FormatValue(double value) const {
  std::ostringstream oss;
  if (std::abs(value) < 1e-3 || std::abs(value) > 1e6) {
    oss << std::scientific << std::setprecision(6) << value;
  } else {
    oss << std::fixed << std::setprecision(6) << value;
  }
  return oss.str();
}

const std::vector<double>* InspectionToolsComponent::GetFieldDataByIndex(int idx) const {
  if (idx == 0) {
    return grid_;
  }
  if (!derived_fields_) return nullptr;
  switch (idx) {
    case 1: return &derived_fields_->gradient_x;
    case 2: return &derived_fields_->gradient_y;
    case 3: return &derived_fields_->gradient_z;
    case 4: return &derived_fields_->laplacian;
    case 5: return &derived_fields_->flux_x;
    case 6: return &derived_fields_->flux_y;
    case 7: return &derived_fields_->flux_z;
    case 8: return &derived_fields_->energy_norm;
    default: return nullptr;
  }
}

void InspectionToolsComponent::EnsureStatisticsComputed() {
  if (selected_field_index_ < 0) selected_field_index_ = 0;
  if (selected_field_index_ > 8) selected_field_index_ = 8;
  if (selected_field_index_ > 0 && !has_derived_) {
    stats_cache_.valid = false;
    return;
  }
  const std::vector<double>* data = GetFieldDataByIndex(selected_field_index_);
  if (!data || data->empty()) {
    stats_cache_.valid = false;
    return;
  }
  if (!stats_dirty_ && stats_cache_.valid) return;
  stats_cache_.stats = ComputeStatistics(*data, mask_);
  stats_cache_.hist = ComputeHistogram(*data, histogram_bins_, mask_);
  stats_cache_.valid = true;
  stats_dirty_ = false;
}

void InspectionToolsComponent::Render() {
  if (!IsVisible()) {
    return;
  }
  
  // Content rendered directly - parent "Advanced Inspection Tools" header is in inspect_panel.cpp
  
  if (!domain_ || !grid_ || grid_->empty()) {
    ImGui::TextDisabled("No data loaded. Solve PDE or load VTK file first.");
    return;
  }
  
  const double domain_min = std::min({domain_->xmin, domain_->ymin, domain_->zmin});
  const double domain_max = std::max({domain_->xmax, domain_->ymax, domain_->zmax});
  const double domain_span = domain_max - domain_min;
  
  // Multiple Slice Planes
  if (ImGui::TreeNode("Slice Planes")) {
    // Add new slice controls
    const char* axis_names[] = {"X", "Y", "Z"};
    ImGui::Combo("Axis##new_slice", &selected_slice_axis_, axis_names, 3);
    ImGui::SameLine();
    ImGui::SetNextItemWidth(120.0f);
    ImGui::SliderScalar("Value##new_slice", ImGuiDataType_Double, &selected_slice_value_,
                       &domain_min, &domain_max, "%.4f");
    ImGui::SameLine();
    ImGui::SetNextItemWidth(80.0f);
    ImGui::InputDouble("Thickness##new_slice", &selected_slice_thickness_, 0.0, 0.0, "%.4f");
    if (selected_slice_thickness_ < 0.0) {
      selected_slice_thickness_ = 0.0;
    }
    ImGui::SameLine();
    if (ImGui::Button("Add Slice")) {
      std::string name = "Slice " + std::string(axis_names[selected_slice_axis_]) + 
                        " = " + FormatValue(selected_slice_value_);
      AddSlice(SlicePlane(selected_slice_axis_, selected_slice_value_, 
                         selected_slice_thickness_, name));
    }
    
    // List existing slices
    ImGui::Separator();
    for (size_t i = 0; i < slices_.size(); ++i) {
      ImGui::PushID(static_cast<int>(i));
      ImGui::Checkbox("##enabled", &slices_[i].enabled);
      ImGui::SameLine();
      ImGui::Text("%s", slices_[i].name.c_str());
      ImGui::SameLine();
      ImGui::TextDisabled("(axis=%d, val=%.4f, thick=%.4f)", 
                         slices_[i].axis, slices_[i].value, slices_[i].thickness);
      ImGui::SameLine();
      if (ImGui::Button("Remove")) {
        RemoveSlice(i);
        ImGui::PopID();
        break;
      }
      ImGui::PopID();
    }
    
    if (slices_.empty()) {
      ImGui::TextDisabled("No slice planes defined");
    }
    
    ImGui::TreePop();
  }
  
  // Probe Tool
  if (ImGui::TreeNode("Probe Points")) {
    // Add new probe controls
    ImGui::SetNextItemWidth(100.0f);
    ImGui::InputDouble("X##probe", &probe_x_, 0.0, 0.0, "%.4f");
    ImGui::SameLine();
    ImGui::SetNextItemWidth(100.0f);
    ImGui::InputDouble("Y##probe", &probe_y_, 0.0, 0.0, "%.4f");
    ImGui::SameLine();
    ImGui::SetNextItemWidth(100.0f);
    ImGui::InputDouble("Z##probe", &probe_z_, 0.0, 0.0, "%.4f");
    ImGui::SameLine();
    if (ImGui::Button("Add Probe")) {
      std::string label = "Probe (" + FormatValue(probe_x_) + ", " + 
                         FormatValue(probe_y_) + ", " + FormatValue(probe_z_) + ")";
      AddProbe(ProbePoint(probe_x_, probe_y_, probe_z_, label));
    }
    
    // List existing probes
    ImGui::Separator();
    for (size_t i = 0; i < probes_.size(); ++i) {
      ImGui::PushID(static_cast<int>(i + 1000));
      ImGui::Checkbox("##active", &probes_[i].active);
      ImGui::SameLine();
      if (probes_[i].label.empty()) {
        ImGui::Text("Probe %zu", i);
      } else {
        ImGui::Text("%s", probes_[i].label.c_str());
      }
      ImGui::SameLine();
      ImGui::TextDisabled("(%.4f, %.4f, %.4f) = %s", 
                         probes_[i].x, probes_[i].y, probes_[i].z,
                         FormatValue(probes_[i].value).c_str());
      ImGui::SameLine();
      if (ImGui::Button("Remove")) {
        RemoveProbe(i);
        ImGui::PopID();
        break;
      }
      ImGui::PopID();
    }
    
    if (probes_.empty()) {
      ImGui::TextDisabled("No probe points defined");
    }
    
    ImGui::TreePop();
  }
  
  // Line Plots
  if (ImGui::TreeNode("Line Plots")) {
    // Add new line plot controls
    ImGui::Text("Start point:");
    ImGui::SetNextItemWidth(80.0f);
    ImGui::InputDouble("X0##line", &line_x0_, 0.0, 0.0, "%.4f");
    ImGui::SameLine();
    ImGui::SetNextItemWidth(80.0f);
    ImGui::InputDouble("Y0##line", &line_y0_, 0.0, 0.0, "%.4f");
    ImGui::SameLine();
    ImGui::SetNextItemWidth(80.0f);
    ImGui::InputDouble("Z0##line", &line_z0_, 0.0, 0.0, "%.4f");
    
    ImGui::Text("End point:");
    ImGui::SetNextItemWidth(80.0f);
    ImGui::InputDouble("X1##line", &line_x1_, 0.0, 0.0, "%.4f");
    ImGui::SameLine();
    ImGui::SetNextItemWidth(80.0f);
    ImGui::InputDouble("Y1##line", &line_y1_, 0.0, 0.0, "%.4f");
    ImGui::SameLine();
    ImGui::SetNextItemWidth(80.0f);
    ImGui::InputDouble("Z1##line", &line_z1_, 0.0, 0.0, "%.4f");
    
    ImGui::Text("Samples:");
    ImGui::SetNextItemWidth(120.0f);
    ImGui::InputInt("##line_samples", &line_num_points_);
    if (line_num_points_ < 2) {
      line_num_points_ = 2;
    }
    
    if (ImGui::Button("Add Line Plot")) {
      std::string name = "Line " + FormatValue(line_x0_) + "," + FormatValue(line_y0_) + " -> " +
                         FormatValue(line_x1_) + "," + FormatValue(line_y1_);
      AddLinePlot(LinePlot(line_x0_, line_y0_, line_z0_, 
                           line_x1_, line_y1_, line_z1_, 
                           line_num_points_, name));
    }
    
    ImGui::Separator();
    for (size_t i = 0; i < line_plots_.size(); ++i) {
      ImGui::PushID(static_cast<int>(i + 2000));
      ImGui::Checkbox("##enabled", &line_plots_[i].enabled);
      ImGui::SameLine();
      ImGui::Text("%s", line_plots_[i].name.c_str());
      ImGui::SameLine();
      if (ImGui::Button("Recompute")) {
        ComputeLinePlot(line_plots_[i]);
      }
      ImGui::SameLine();
      if (ImGui::Button("Remove")) {
        RemoveLinePlot(i);
        ImGui::PopID();
        break;
      }
      if (!line_plots_[i].values.empty()) {
        // ImGui expects float samples; convert from double.
        std::vector<float> samples;
        samples.reserve(line_plots_[i].values.size());
        for (double v : line_plots_[i].values) {
          samples.push_back(static_cast<float>(v));
        }
        ImGui::PlotLines("##lineplot", samples.data(),
                         static_cast<int>(samples.size()), 0, nullptr,
                         FLT_MAX, FLT_MAX, ImVec2(0, 80));
      }
      ImGui::PopID();
    }
    
    if (line_plots_.empty()) {
      ImGui::TextDisabled("No line plots defined");
    }
    
    ImGui::TreePop();
  }
  
  // Region selection
  if (ImGui::TreeNode("Region Selection (for masked stats)")) {
    const char* region_types[] = {"Box", "Sphere"};
    ImGui::Combo("Region type", &region_type_, region_types, 2);
    if (region_type_ == 0) {
      ImGui::Text("Box bounds");
      ImGui::SetNextItemWidth(120.0f);
      ImGui::InputDouble("xmin", &box_xmin_, 0.0, 0.0, "%.4f");
      ImGui::SameLine();
      ImGui::SetNextItemWidth(120.0f);
      ImGui::InputDouble("xmax", &box_xmax_, 0.0, 0.0, "%.4f");
      ImGui::SetNextItemWidth(120.0f);
      ImGui::InputDouble("ymin", &box_ymin_, 0.0, 0.0, "%.4f");
      ImGui::SameLine();
      ImGui::SetNextItemWidth(120.0f);
      ImGui::InputDouble("ymax", &box_ymax_, 0.0, 0.0, "%.4f");
      ImGui::SetNextItemWidth(120.0f);
      ImGui::InputDouble("zmin", &box_zmin_, 0.0, 0.0, "%.4f");
      ImGui::SameLine();
      ImGui::SetNextItemWidth(120.0f);
      ImGui::InputDouble("zmax", &box_zmax_, 0.0, 0.0, "%.4f");
      if (ImGui::Button("Apply Box Region")) {
        BoxRegion region(box_xmin_, box_xmax_, box_ymin_, box_ymax_, box_zmin_, box_zmax_);
        mask_ = CreateMaskFromRegion(*domain_, region);
        stats_dirty_ = true;
      }
    } else {
      ImGui::Text("Sphere (center + radius)");
      ImGui::SetNextItemWidth(120.0f);
      ImGui::InputDouble("cx", &sphere_cx_, 0.0, 0.0, "%.4f");
      ImGui::SameLine();
      ImGui::SetNextItemWidth(120.0f);
      ImGui::InputDouble("cy", &sphere_cy_, 0.0, 0.0, "%.4f");
      ImGui::SetNextItemWidth(120.0f);
      ImGui::InputDouble("cz", &sphere_cz_, 0.0, 0.0, "%.4f");
      ImGui::SetNextItemWidth(120.0f);
      ImGui::InputDouble("r", &sphere_r_, 0.0, 0.0, "%.4f");
      if (sphere_r_ < 0.0) sphere_r_ = 0.0;
      if (ImGui::Button("Apply Sphere Region")) {
        SphereRegion region(sphere_cx_, sphere_cy_, sphere_cz_, sphere_r_);
        mask_ = CreateMaskFromRegion(*domain_, region);
        stats_dirty_ = true;
      }
    }
    if (!mask_.empty()) {
      ImGui::SameLine();
      if (ImGui::Button("Clear Mask")) {
        mask_.clear();
        stats_dirty_ = true;
      }
    }
    ImGui::TreePop();
  }
  
  // Statistics & Histogram
  if (ImGui::TreeNode("Statistics & Histogram")) {
    // Field selector
    bool use_volume = (domain_->nz > 1);
    if (use_volume) {
      const int item_count = static_cast<int>(sizeof(kFieldNamesVolume) / sizeof(const char*));
      int choice = std::min(selected_field_index_, item_count - 1);
      ImGui::SetNextItemWidth(220.0f);
      if (ImGui::Combo("Field", &choice, kFieldNamesVolume, item_count)) {
        selected_field_index_ = choice;
        stats_dirty_ = true;
      }
    } else {
      const int item_count = static_cast<int>(sizeof(kFieldNamesSurface) / sizeof(const char*));
      int choice = 0;
      // Map full index to 2D list
      if (selected_field_index_ == 0) choice = 0;
      else if (selected_field_index_ == 1) choice = 1;
      else if (selected_field_index_ == 2) choice = 2;
      else if (selected_field_index_ == 4) choice = 3;
      else if (selected_field_index_ == 5) choice = 4;
      else if (selected_field_index_ == 6) choice = 5;
      else choice = 6;  // energy norm
      ImGui::SetNextItemWidth(220.0f);
      if (ImGui::Combo("Field", &choice, kFieldNamesSurface, item_count)) {
        switch (choice) {
          case 0: selected_field_index_ = 0; break;
          case 1: selected_field_index_ = 1; break;
          case 2: selected_field_index_ = 2; break;
          case 3: selected_field_index_ = 4; break;  // Laplacian
          case 4: selected_field_index_ = 5; break;  // Flux X
          case 5: selected_field_index_ = 6; break;  // Flux Y
          default: selected_field_index_ = 8; break; // Energy
        }
        stats_dirty_ = true;
      }
    }
    
    ImGui::SetNextItemWidth(120.0f);
    ImGui::SliderInt("Histogram bins", &histogram_bins_, 5, 200);
    histogram_bins_ = std::max(5, histogram_bins_);
    
    if (ImGui::Button("Compute statistics")) {
      stats_dirty_ = true;
    }
    ImGui::SameLine();
    ImGui::TextDisabled(mask_.empty() ? "Mask: none" : "Mask: %zu cells", mask_.size());
    
    EnsureStatisticsComputed();
    if (!stats_cache_.valid) {
      ImGui::TextDisabled("Statistics unavailable for selected field.");
    } else {
      const auto& st = stats_cache_.stats;
      ImGui::Text("Count: %zu", st.count);
      ImGui::Text("Min: %s", FormatValue(st.min).c_str());
      ImGui::Text("Max: %s", FormatValue(st.max).c_str());
      ImGui::Text("Mean: %s", FormatValue(st.mean).c_str());
      ImGui::Text("Median: %s", FormatValue(st.median).c_str());
      ImGui::Text("Stddev: %s", FormatValue(st.stddev).c_str());
      ImGui::Text("RMS: %s", FormatValue(st.rms).c_str());
      ImGui::Text("L2 norm: %s", FormatValue(st.l2_norm).c_str());
      
      if (!stats_cache_.hist.counts.empty()) {
        // Build float copy for plotting
        std::vector<float> hist_values;
        hist_values.reserve(stats_cache_.hist.counts.size());
        for (int c : stats_cache_.hist.counts) {
          hist_values.push_back(static_cast<float>(c));
        }
        ImGui::PlotHistogram("Histogram", hist_values.data(), 
                             static_cast<int>(hist_values.size()), 0, nullptr,
                             0.0f, FLT_MAX, ImVec2(0, 80));
        ImGui::TextDisabled("Range: [%s, %s]", 
                            FormatValue(stats_cache_.hist.min).c_str(),
                            FormatValue(stats_cache_.hist.max).c_str());
      }
    }
    
    ImGui::TreePop();
  }
}

