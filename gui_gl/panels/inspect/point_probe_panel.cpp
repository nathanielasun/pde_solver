#include "point_probe_panel.h"
#include "imgui.h"
#include <algorithm>
#include <cmath>
#include <sstream>
#include <iomanip>

// Static state
static std::vector<ProbeData> s_probes;
static double s_input_x = 0.0, s_input_y = 0.0, s_input_z = 0.0;
static int s_selected_probe = -1;
static int s_field_index = 0;

static double InterpolateValue(const std::vector<double>& grid, const Domain& domain,
                                double x, double y, double z) {
  if (grid.empty()) return 0.0;

  int nx = domain.nx;
  int ny = domain.ny;
  int nz = std::max(1, domain.nz);

  double dx = (domain.xmax - domain.xmin) / std::max(1, nx - 1);
  double dy = (domain.ymax - domain.ymin) / std::max(1, ny - 1);
  double dz = nz > 1 ? (domain.zmax - domain.zmin) / std::max(1, nz - 1) : 1.0;

  // Clamp to domain
  x = std::clamp(x, domain.xmin, domain.xmax);
  y = std::clamp(y, domain.ymin, domain.ymax);
  z = std::clamp(z, domain.zmin, domain.zmax);

  // Find cell indices
  double fx = (x - domain.xmin) / dx;
  double fy = (y - domain.ymin) / dy;
  double fz = nz > 1 ? (z - domain.zmin) / dz : 0.0;

  int ix = static_cast<int>(fx);
  int iy = static_cast<int>(fy);
  int iz = static_cast<int>(fz);

  ix = std::clamp(ix, 0, nx - 2);
  iy = std::clamp(iy, 0, ny - 2);
  iz = std::clamp(iz, 0, nz - 2);

  double tx = fx - ix;
  double ty = fy - iy;
  double tz = nz > 1 ? (fz - iz) : 0.0;

  // Trilinear interpolation (or bilinear for 2D)
  auto idx = [&](int i, int j, int k) -> size_t {
    return static_cast<size_t>(i + j * nx + k * nx * ny);
  };

  if (nz <= 1) {
    // 2D bilinear interpolation
    double v00 = grid[idx(ix, iy, 0)];
    double v10 = grid[idx(ix + 1, iy, 0)];
    double v01 = grid[idx(ix, iy + 1, 0)];
    double v11 = grid[idx(ix + 1, iy + 1, 0)];

    double v0 = v00 * (1 - tx) + v10 * tx;
    double v1 = v01 * (1 - tx) + v11 * tx;
    return v0 * (1 - ty) + v1 * ty;
  } else {
    // 3D trilinear interpolation
    double v000 = grid[idx(ix, iy, iz)];
    double v100 = grid[idx(ix + 1, iy, iz)];
    double v010 = grid[idx(ix, iy + 1, iz)];
    double v110 = grid[idx(ix + 1, iy + 1, iz)];
    double v001 = grid[idx(ix, iy, iz + 1)];
    double v101 = grid[idx(ix + 1, iy, iz + 1)];
    double v011 = grid[idx(ix, iy + 1, iz + 1)];
    double v111 = grid[idx(ix + 1, iy + 1, iz + 1)];

    double v00 = v000 * (1 - tx) + v100 * tx;
    double v01 = v001 * (1 - tx) + v101 * tx;
    double v10 = v010 * (1 - tx) + v110 * tx;
    double v11 = v011 * (1 - tx) + v111 * tx;

    double v0 = v00 * (1 - ty) + v10 * ty;
    double v1 = v01 * (1 - ty) + v11 * ty;

    return v0 * (1 - tz) + v1 * tz;
  }
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
    default: return nullptr;
  }
}

void RenderPointProbePanel(PointProbePanelState& state, const std::vector<std::string>& components) {
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
    ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "Solve a PDE to probe values.");
    return;
  }

  // Field selector
  ImGui::Text("Field:");
  ImGui::SameLine();
  ImGui::SetNextItemWidth(120.0f);
  ImGui::Combo("##probe_field", &s_field_index, "Solution\0Grad X\0Grad Y\0Grad Z\0Laplacian\0");

  ImGui::Spacing();
  ImGui::Separator();

  // Add probe section
  ImGui::Text("Add Probe Point:");

  float coord_width = (state.input_width - 80.0f) / 3.0f;

  ImGui::SetNextItemWidth(coord_width);
  ImGui::InputDouble("X##probe", &s_input_x, 0.0, 0.0, "%.4f");
  ImGui::SameLine();
  ImGui::SetNextItemWidth(coord_width);
  ImGui::InputDouble("Y##probe", &s_input_y, 0.0, 0.0, "%.4f");
  if (domain_copy.nz > 1) {
    ImGui::SameLine();
    ImGui::SetNextItemWidth(coord_width);
    ImGui::InputDouble("Z##probe", &s_input_z, 0.0, 0.0, "%.4f");
  }

  // Quick buttons to set center
  if (ImGui::Button("Center", ImVec2(60, 0))) {
    s_input_x = (domain_copy.xmin + domain_copy.xmax) * 0.5;
    s_input_y = (domain_copy.ymin + domain_copy.ymax) * 0.5;
    s_input_z = (domain_copy.zmin + domain_copy.zmax) * 0.5;
  }
  ImGui::SameLine();

  if (ImGui::Button("Add Probe", ImVec2(-1, 0))) {
    const std::vector<double>* field = GetFieldData(s_field_index, grid_copy, derived_copy, has_derived);
    if (field && !field->empty()) {
      ProbeData probe;
      probe.x = s_input_x;
      probe.y = s_input_y;
      probe.z = s_input_z;
      probe.value = InterpolateValue(*field, domain_copy, s_input_x, s_input_y, s_input_z);
      probe.valid = true;
      probe.label = "Probe " + std::to_string(s_probes.size() + 1);
      s_probes.push_back(probe);
    }
  }

  ImGui::Spacing();

  // Live value at cursor position
  const std::vector<double>* field = GetFieldData(s_field_index, grid_copy, derived_copy, has_derived);
  if (field && !field->empty()) {
    double live_value = InterpolateValue(*field, domain_copy, s_input_x, s_input_y, s_input_z);
    ImGui::Text("Value at cursor: ");
    ImGui::SameLine();
    if (std::abs(live_value) < 1e-3 || std::abs(live_value) >= 1e4) {
      ImGui::TextColored(ImVec4(0.4f, 0.8f, 1.0f, 1.0f), "%.6e", live_value);
    } else {
      ImGui::TextColored(ImVec4(0.4f, 0.8f, 1.0f, 1.0f), "%.6f", live_value);
    }
  }

  ImGui::Spacing();
  ImGui::Separator();

  // Probe list
  ImGui::Text("Probe Points (%zu):", s_probes.size());

  if (s_probes.empty()) {
    ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "No probes added yet.");
  } else {
    if (ImGui::BeginTable("probes_table", 5, ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg | ImGuiTableFlags_ScrollY,
                          ImVec2(-1, 150))) {
      ImGui::TableSetupColumn("", ImGuiTableColumnFlags_WidthFixed, 20.0f);
      ImGui::TableSetupColumn("X", ImGuiTableColumnFlags_WidthFixed, 60.0f);
      ImGui::TableSetupColumn("Y", ImGuiTableColumnFlags_WidthFixed, 60.0f);
      ImGui::TableSetupColumn("Z", ImGuiTableColumnFlags_WidthFixed, 60.0f);
      ImGui::TableSetupColumn("Value", ImGuiTableColumnFlags_WidthStretch);
      ImGui::TableHeadersRow();

      for (size_t i = 0; i < s_probes.size(); ++i) {
        ImGui::TableNextRow();

        ImGui::TableNextColumn();
        ImGui::Text("%zu", i + 1);

        ImGui::TableNextColumn();
        ImGui::Text("%.3f", s_probes[i].x);

        ImGui::TableNextColumn();
        ImGui::Text("%.3f", s_probes[i].y);

        ImGui::TableNextColumn();
        ImGui::Text("%.3f", s_probes[i].z);

        ImGui::TableNextColumn();
        if (std::abs(s_probes[i].value) < 1e-3 || std::abs(s_probes[i].value) >= 1e4) {
          ImGui::Text("%.4e", s_probes[i].value);
        } else {
          ImGui::Text("%.6f", s_probes[i].value);
        }

        // Row selection
        if (ImGui::IsItemClicked()) {
          s_selected_probe = static_cast<int>(i);
        }
      }

      ImGui::EndTable();
    }

    // Update all probe values (in case field changed)
    if (ImGui::Button("Refresh All", ImVec2(-1, 0))) {
      for (auto& probe : s_probes) {
        if (field && !field->empty()) {
          probe.value = InterpolateValue(*field, domain_copy, probe.x, probe.y, probe.z);
        }
      }
    }

    // Delete selected
    if (s_selected_probe >= 0 && s_selected_probe < static_cast<int>(s_probes.size())) {
      if (ImGui::Button("Delete Selected", ImVec2(-1, 0))) {
        s_probes.erase(s_probes.begin() + s_selected_probe);
        s_selected_probe = -1;
      }
    }

    // Clear all
    if (ImGui::Button("Clear All", ImVec2(-1, 0))) {
      s_probes.clear();
      s_selected_probe = -1;
    }

    // Export
    if (ImGui::Button("Copy to Clipboard", ImVec2(-1, 0))) {
      std::ostringstream oss;
      oss << "X\tY\tZ\tValue\n";
      for (const auto& p : s_probes) {
        oss << std::fixed << std::setprecision(6);
        oss << p.x << "\t" << p.y << "\t" << p.z << "\t";
        oss << std::scientific << p.value << "\n";
      }
      ImGui::SetClipboardText(oss.str().c_str());
    }
  }
}
