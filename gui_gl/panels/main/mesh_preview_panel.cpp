#include "mesh_preview_panel.h"
#include "imgui.h"

static int s_stride = 1;
static bool s_show_boundary = true;
static bool s_show_interior = true;

void RenderMeshPreviewPanel(MeshPreviewPanelState& state,
                             const std::vector<std::string>& components) {
  ImGui::Text("Mesh Preview");
  ImGui::Separator();
  ImGui::Spacing();

  // Grid info
  ImGui::Text("Grid Dimensions:");
  ImGui::BulletText("NX: %d", state.domain.nx);
  ImGui::BulletText("NY: %d", state.domain.ny);
  if (state.domain.nz > 1) {
    ImGui::BulletText("NZ: %d", state.domain.nz);
  }

  int total_cells = state.domain.nx * state.domain.ny * std::max(1, state.domain.nz);
  ImGui::BulletText("Total cells: %d", total_cells);

  ImGui::Spacing();

  // Grid spacing
  double dx = (state.domain.xmax - state.domain.xmin) / std::max(1, state.domain.nx - 1);
  double dy = (state.domain.ymax - state.domain.ymin) / std::max(1, state.domain.ny - 1);
  ImGui::Text("Grid Spacing:");
  ImGui::BulletText("dx: %.4f", dx);
  ImGui::BulletText("dy: %.4f", dy);
  if (state.domain.nz > 1) {
    double dz = (state.domain.zmax - state.domain.zmin) / std::max(1, state.domain.nz - 1);
    ImGui::BulletText("dz: %.4f", dz);
  }

  // Aspect ratio
  double aspect_xy = dx / dy;
  ImGui::BulletText("Aspect (dx/dy): %.3f", aspect_xy);

  ImGui::Spacing();
  ImGui::Separator();
  ImGui::Spacing();

  // Display options
  ImGui::Text("Display Options:");
  ImGui::SliderInt("Stride", &s_stride, 1, 10);
  ImGui::Checkbox("Show Boundary", &s_show_boundary);
  ImGui::Checkbox("Show Interior", &s_show_interior);

  ImGui::Spacing();

  // Placeholder for wireframe preview
  ImVec2 preview_size(state.input_width - 20.0f, 150.0f);
  ImGui::BeginChild("mesh_preview_area", preview_size, true);
  ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f),
                     "Wireframe preview");
  ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f),
                     "(OpenGL rendering coming soon)");
  ImGui::EndChild();

  ImGui::Spacing();
  ImGui::TextColored(ImVec4(1.0f, 0.7f, 0.3f, 1.0f),
                     "Interactive mesh preview coming soon.");
}
