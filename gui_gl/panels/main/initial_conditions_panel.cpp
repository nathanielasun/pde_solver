#include "initial_conditions_panel.h"
#include "imgui.h"
#include "imgui_stdlib.h"
#include <cmath>

// Preset templates for common initial conditions
static const char* s_preset_names[] = {
  "Custom",
  "Zero",
  "Constant",
  "Gaussian",
  "Step Function",
  "Sine Wave",
  "Two Gaussians",
  "Ring/Shell"
};

static const char* s_preset_expressions[] = {
  "",  // Custom - keep current
  "0",
  "1",
  "e^{-((x-0.5)^2 + (y-0.5)^2) / 0.02}",
  "\\begin{cases} 1 & x < 0.5 \\\\ 0 & x \\geq 0.5 \\end{cases}",
  "\\sin(\\pi x) \\sin(\\pi y)",
  "e^{-((x-0.3)^2 + (y-0.5)^2)/0.01} + e^{-((x-0.7)^2 + (y-0.5)^2)/0.01}",
  "e^{-((\\sqrt{(x-0.5)^2 + (y-0.5)^2} - 0.3)^2)/0.01}"
};

static int s_selected_preset = 0;

void RenderInitialConditionsPanel(InitialConditionsPanelState& state,
                                   const std::vector<std::string>& components) {
  if (!state.time_dependent) {
    ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f),
                       "Initial conditions are only used for");
    ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f),
                       "time-dependent PDEs (ut or utt terms).");
    ImGui::Spacing();
    ImGui::TextColored(ImVec4(1.0f, 0.7f, 0.3f, 1.0f),
                       "Enable time-dependent mode in the");
    ImGui::TextColored(ImVec4(1.0f, 0.7f, 0.3f, 1.0f),
                       "Time Settings panel.");
    return;
  }

  ImGui::Text("Initial Condition u(x, y, z, 0):");
  ImGui::Spacing();

  // Preset selector
  ImGui::Text("Preset:");
  ImGui::SameLine();
  ImGui::SetNextItemWidth(state.input_width - 60.0f);
  if (ImGui::Combo("##ic_preset", &s_selected_preset, s_preset_names,
                   IM_ARRAYSIZE(s_preset_names))) {
    if (s_selected_preset > 0) {
      state.ic_expression = s_preset_expressions[s_selected_preset];
    }
  }

  ImGui::Spacing();

  // LaTeX expression input
  ImGui::Text("Expression:");
  ImGui::SetNextItemWidth(-1);
  if (ImGui::InputTextMultiline("##ic_expr", &state.ic_expression,
                                 ImVec2(-1, 80),
                                 ImGuiInputTextFlags_AllowTabInput)) {
    // If user modifies, switch to Custom
    s_selected_preset = 0;
  }

  ImGui::Spacing();

  // Help text
  ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "Variables: x, y, z");
  ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "Use LaTeX math notation");

  ImGui::Spacing();
  ImGui::Separator();
  ImGui::Spacing();

  // Common functions reference
  if (ImGui::CollapsingHeader("Function Reference")) {
    ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f), "Common functions:");
    ImGui::BulletText("sin(x), cos(x), tan(x)");
    ImGui::BulletText("exp(x) or e^x");
    ImGui::BulletText("log(x), log10(x)");
    ImGui::BulletText("sqrt(x) or \\sqrt{x}");
    ImGui::BulletText("abs(x) or |x|");
    ImGui::BulletText("pi for 3.14159...");

    ImGui::Spacing();
    ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f), "Operators:");
    ImGui::BulletText("+ - * / ^");
    ImGui::BulletText("Parentheses: ( )");
  }

  ImGui::Spacing();

  // Preset descriptions
  if (ImGui::CollapsingHeader("Preset Descriptions")) {
    ImGui::BulletText("Zero: u = 0 everywhere");
    ImGui::BulletText("Constant: u = 1 everywhere");
    ImGui::BulletText("Gaussian: Bell curve at center");
    ImGui::BulletText("Step: Jump discontinuity at x=0.5");
    ImGui::BulletText("Sine Wave: Product of sine waves");
    ImGui::BulletText("Two Gaussians: Two peaks");
    ImGui::BulletText("Ring: Circular ring pattern");
  }
}
