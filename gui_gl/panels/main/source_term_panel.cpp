#include "source_term_panel.h"
#include "imgui.h"
#include "imgui_stdlib.h"
#include "styles/ui_style.h"
#include <sstream>
#include <iomanip>
#include <cmath>

static int s_source_type = 0;

// Sigma for Gaussian approximation of point sources.
static const double kPointSourceSigma = 0.02;

// Build a combined expression string from point sources using narrow Gaussians.
static std::string BuildPointSourceExpression(
    const std::vector<PointSource>& sources) {
  if (sources.empty()) return "0";
  std::ostringstream oss;
  oss << std::setprecision(8);
  double twoSigmaSq = 2.0 * kPointSourceSigma * kPointSourceSigma;
  for (size_t i = 0; i < sources.size(); ++i) {
    const auto& ps = sources[i];
    if (i > 0) oss << " + ";
    oss << ps.strength << " * exp(-(("
        << "x - " << ps.x << ")^2 + ("
        << "y - " << ps.y << ")^2 + ("
        << "z - " << ps.z << ")^2) / "
        << twoSigmaSq << ")";
  }
  return oss.str();
}

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
    ImGui::Spacing();

    if (ImGui::Button("Add Point Source", ImVec2(-1, 0))) {
      state.point_sources.push_back(PointSource{});
      state.source_expression = BuildPointSourceExpression(state.point_sources);
    }

    ImGui::Spacing();

    if (state.point_sources.empty()) {
      ImGui::TextColored(ImVec4(0.5f, 0.5f, 0.5f, 1.0f),
                         "No point sources defined.");
    } else {
      ImGui::Text("Sources: %d", static_cast<int>(state.point_sources.size()));
      ImGui::Spacing();

      bool changed = false;
      int removeIndex = -1;

      for (int i = 0; i < static_cast<int>(state.point_sources.size()); ++i) {
        ImGui::PushID(i);
        auto& ps = state.point_sources[i];

        // Header with remove button
        ImGui::Separator();
        ImGui::Text("Source %d", i + 1);
        ImGui::SameLine(ImGui::GetContentRegionAvail().x - 20.0f);
        if (ImGui::SmallButton("X")) {
          removeIndex = i;
        }

        // Position inputs
        ImGui::SetNextItemWidth(-1);
        if (UIInput::InputDouble("X", &ps.x, 0.01, 0.1, "%.4f")) changed = true;
        ImGui::SetNextItemWidth(-1);
        if (UIInput::InputDouble("Y", &ps.y, 0.01, 0.1, "%.4f")) changed = true;
        ImGui::SetNextItemWidth(-1);
        if (UIInput::InputDouble("Z", &ps.z, 0.01, 0.1, "%.4f")) changed = true;

        // Strength input
        ImGui::SetNextItemWidth(-1);
        if (UIInput::InputDouble("Strength", &ps.strength, 0.1, 1.0, "%.4f")) changed = true;

        ImGui::PopID();
      }

      // Handle removal after iteration
      if (removeIndex >= 0) {
        state.point_sources.erase(state.point_sources.begin() + removeIndex);
        changed = true;
      }

      // Rebuild expression when any value changes
      if (changed) {
        state.source_expression = BuildPointSourceExpression(state.point_sources);
      }
    }

    // Show generated expression (read-only preview)
    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();
    ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "Generated expression:");
    ImGui::TextWrapped("%s", state.source_expression.c_str());
  } else {
    ImGui::Text("Region Source:");
    ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f),
                       "Define a region with constant source strength.");
  }

  ImGui::Spacing();
  ImGui::TextColored(ImVec4(1.0f, 0.7f, 0.3f, 1.0f),
                     "Advanced source editor coming soon.");
}
