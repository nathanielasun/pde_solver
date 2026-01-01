#include "source_term_panel.h"
#include "imgui.h"
#include "imgui_stdlib.h"

static int s_source_type = 0;

void RenderSourceTermPanel(SourceTermPanelState& state,
                            const std::vector<std::string>& components) {
  ImGui::Text("Source Term Editor");
  ImGui::Separator();
  ImGui::Spacing();

  ImGui::Text("Source Type:");
  ImGui::RadioButton("Expression f(x,y,z,t)", &s_source_type, 0);
  ImGui::RadioButton("Point Sources", &s_source_type, 1);
  ImGui::RadioButton("Region Source", &s_source_type, 2);

  ImGui::Spacing();
  ImGui::Separator();
  ImGui::Spacing();

  if (s_source_type == 0) {
    ImGui::Text("Source Expression:");
    ImGui::SetNextItemWidth(-1);
    ImGui::InputTextMultiline("##source_expr", &state.source_expression,
                               ImVec2(-1, 60));
    ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f),
                       "Variables: x, y, z, t");
  } else if (s_source_type == 1) {
    ImGui::Text("Point Sources:");
    ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f),
                       "Add point sources with position and strength.");
    if (ImGui::Button("Add Point Source", ImVec2(-1, 0))) {
      // TODO: Implement
    }
  } else {
    ImGui::Text("Region Source:");
    ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f),
                       "Define a region with constant source strength.");
  }

  ImGui::Spacing();
  ImGui::TextColored(ImVec4(1.0f, 0.7f, 0.3f, 1.0f),
                     "Advanced source editor coming soon.");
}
