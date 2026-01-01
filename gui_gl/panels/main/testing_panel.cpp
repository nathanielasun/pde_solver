#include "testing_panel.h"

#include "imgui.h"

#include <algorithm>
#include <cstdio>

namespace {

// Panel state (persistent across frames)
struct TestingPanelInternalState {
  std::unique_ptr<TestRunner> runner;
  TestPanelConfig config;
  std::vector<TestSuiteResults> all_results;
  int selected_tab = 0;
  std::mutex results_mutex;

  // Latest results for each category (quick access)
  TestSuiteResults advection_results;
  TestSuiteResults time_integrator_results;
  TestSuiteResults projection_results;

  TestingPanelInternalState() : runner(std::make_unique<TestRunner>()) {}
};

static TestingPanelInternalState g_state;

// Scheme/method name lists for dropdowns
const char* kAdvectionSchemes[] = {"Upwind",       "Lax-Wendroff", "Beam-Warming",
                                   "Fromm",        "MinMod (TVD)", "Superbee (TVD)",
                                   "Van Leer (TVD)", "MC (TVD)"};

const char* kTimeIntegratorMethods[] = {"Forward Euler",   "RK2 (Heun)",
                                        "RK4 (Classical)", "SSPRK2",
                                        "SSPRK3",          "Backward Euler",
                                        "Crank-Nicolson"};

void RenderResultsTable(const TestSuiteResults& results) {
  if (results.results.empty()) {
    ImGui::TextDisabled("No results yet. Run tests to see results.");
    return;
  }

  // Summary header
  if (results.cancelled) {
    ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.2f, 1.0f), "Cancelled");
    ImGui::SameLine();
  }
  ImGui::Text("Passed: ");
  ImGui::SameLine();
  ImGui::TextColored(ImVec4(0.2f, 0.8f, 0.2f, 1.0f), "%d", results.passed_count);
  ImGui::SameLine();
  ImGui::Text("  Failed: ");
  ImGui::SameLine();
  ImGui::TextColored(ImVec4(0.9f, 0.2f, 0.2f, 1.0f), "%d", results.failed_count);
  ImGui::SameLine();
  ImGui::Text("  Runtime: %.1f ms", results.total_runtime_ms);
  ImGui::Separator();

  // Results table
  if (ImGui::BeginTable("TestResults", 5,
                        ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg |
                            ImGuiTableFlags_Resizable | ImGuiTableFlags_ScrollY,
                        ImVec2(0.0f, 200.0f))) {
    ImGui::TableSetupColumn("Status", ImGuiTableColumnFlags_WidthFixed, 50.0f);
    ImGui::TableSetupColumn("Test Name", ImGuiTableColumnFlags_WidthStretch);
    ImGui::TableSetupColumn("L2 Error", ImGuiTableColumnFlags_WidthFixed, 90.0f);
    ImGui::TableSetupColumn("Rate", ImGuiTableColumnFlags_WidthFixed, 50.0f);
    ImGui::TableSetupColumn("Note", ImGuiTableColumnFlags_WidthStretch);
    ImGui::TableSetupScrollFreeze(0, 1);
    ImGui::TableHeadersRow();

    for (const auto& r : results.results) {
      ImGui::TableNextRow();

      // Status column with color
      ImGui::TableNextColumn();
      if (r.passed) {
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.2f, 0.8f, 0.2f, 1.0f));
        ImGui::Text("PASS");
      } else {
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.9f, 0.2f, 0.2f, 1.0f));
        ImGui::Text("FAIL");
      }
      ImGui::PopStyleColor();

      // Test name
      ImGui::TableNextColumn();
      ImGui::Text("%s", r.name.c_str());
      if (!r.description.empty() && ImGui::IsItemHovered()) {
        ImGui::SetTooltip("%s", r.description.c_str());
      }

      // L2 Error
      ImGui::TableNextColumn();
      if (r.error_l2 > 0) {
        ImGui::Text("%.2e", r.error_l2);
      } else {
        ImGui::TextDisabled("-");
      }

      // Convergence rate
      ImGui::TableNextColumn();
      if (r.convergence_rate > 0) {
        ImGui::Text("%.2f", r.convergence_rate);
      } else {
        ImGui::TextDisabled("-");
      }

      // Note
      ImGui::TableNextColumn();
      if (!r.note.empty()) {
        ImGui::TextWrapped("%s", r.note.c_str());
      }
    }

    ImGui::EndTable();
  }
}

void RenderAdvectionTab() {
  auto& cfg = g_state.config.advection;

  ImGui::TextColored(ImVec4(0.4f, 0.7f, 1.0f, 1.0f), "Advection Discretization Tests");
  ImGui::TextDisabled("Test spatial advection schemes for accuracy and stability");
  ImGui::Separator();

  // Scheme selector
  ImGui::Combo("Scheme", &cfg.scheme_index, kAdvectionSchemes,
               IM_ARRAYSIZE(kAdvectionSchemes));

  ImGui::SliderInt("Grid Size", &cfg.grid_size, 32, 256);
  if (ImGui::IsItemHovered()) {
    ImGui::SetTooltip("Number of grid points in each dimension");
  }

  ImGui::SliderFloat("CFL", &cfg.cfl, 0.1f, 0.9f, "%.2f");
  if (ImGui::IsItemHovered()) {
    ImGui::SetTooltip("Courant-Friedrichs-Lewy number (stability limit)");
  }

  ImGui::Checkbox("Run All Schemes (Comparison)", &cfg.run_all_schemes);

  ImGui::Spacing();

  // Run button
  bool is_running = g_state.runner->IsRunning();
  if (is_running) {
    ImGui::BeginDisabled();
  }

  if (ImGui::Button("Run Advection Tests", ImVec2(180, 28))) {
    g_state.runner->RunAdvectionTests(
        cfg,
        [](float progress, const std::string& status) {
          // Progress callback (optional logging)
        },
        [](const TestSuiteResults& results) {
          std::lock_guard<std::mutex> lock(g_state.results_mutex);
          g_state.advection_results = results;
          g_state.all_results.push_back(results);
        });
  }

  if (is_running) {
    ImGui::EndDisabled();
    ImGui::SameLine();
    if (ImGui::Button("Cancel")) {
      g_state.runner->Cancel();
    }
    ImGui::SameLine();

    // Progress bar
    float progress = g_state.runner->GetProgress();
    char overlay[64];
    snprintf(overlay, sizeof(overlay), "%.0f%% - %s", progress * 100.0f,
             g_state.runner->GetCurrentTestName().c_str());
    ImGui::ProgressBar(progress, ImVec2(200, 0), overlay);
  }

  ImGui::Spacing();
  ImGui::Separator();
  ImGui::Spacing();

  // Display results
  std::lock_guard<std::mutex> lock(g_state.results_mutex);
  RenderResultsTable(g_state.advection_results);
}

void RenderTimeIntegratorTab() {
  auto& cfg = g_state.config.time_integrator;

  ImGui::TextColored(ImVec4(0.4f, 0.7f, 1.0f, 1.0f), "Time Integrator Tests");
  ImGui::TextDisabled("Test ODE integration methods for accuracy and stability");
  ImGui::Separator();

  // Method selector
  ImGui::Combo("Method", &cfg.method_index, kTimeIntegratorMethods,
               IM_ARRAYSIZE(kTimeIntegratorMethods));

  ImGui::Checkbox("Run Convergence Study", &cfg.run_convergence_study);
  if (ImGui::IsItemHovered()) {
    ImGui::SetTooltip("Measure order of accuracy with multiple step sizes");
  }

  ImGui::Checkbox("Run All Methods (Comparison)", &cfg.run_all_methods);

  ImGui::Spacing();

  // Run button
  bool is_running = g_state.runner->IsRunning();
  if (is_running) {
    ImGui::BeginDisabled();
  }

  if (ImGui::Button("Run Time Integrator Tests", ImVec2(200, 28))) {
    g_state.runner->RunTimeIntegratorTests(
        cfg,
        [](float progress, const std::string& status) {
          // Progress callback
        },
        [](const TestSuiteResults& results) {
          std::lock_guard<std::mutex> lock(g_state.results_mutex);
          g_state.time_integrator_results = results;
          g_state.all_results.push_back(results);
        });
  }

  if (is_running) {
    ImGui::EndDisabled();
    ImGui::SameLine();
    if (ImGui::Button("Cancel")) {
      g_state.runner->Cancel();
    }
    ImGui::SameLine();

    float progress = g_state.runner->GetProgress();
    char overlay[64];
    snprintf(overlay, sizeof(overlay), "%.0f%% - %s", progress * 100.0f,
             g_state.runner->GetCurrentTestName().c_str());
    ImGui::ProgressBar(progress, ImVec2(200, 0), overlay);
  }

  ImGui::Spacing();
  ImGui::Separator();
  ImGui::Spacing();

  // Display results
  std::lock_guard<std::mutex> lock(g_state.results_mutex);
  RenderResultsTable(g_state.time_integrator_results);
}

void RenderPressureProjectionTab() {
  auto& cfg = g_state.config.projection;

  ImGui::TextColored(ImVec4(0.4f, 0.7f, 1.0f, 1.0f), "Pressure Projection Tests");
  ImGui::TextDisabled("Test incompressible flow projection methods");
  ImGui::Separator();

  ImGui::SliderInt("Grid Size", &cfg.grid_size, 16, 128);

  ImGui::Checkbox("Basic Projection Tests", &cfg.run_basic_tests);
  if (ImGui::IsItemHovered()) {
    ImGui::SetTooltip("Simple projection, divergence-free preservation, Taylor-Green vortex");
  }

  ImGui::Checkbox("Convergence Study", &cfg.run_convergence_study);

  ImGui::Spacing();
  ImGui::Text("Lid-Driven Cavity Benchmark:");
  ImGui::Checkbox("Run Lid-Driven Cavity", &cfg.run_lid_cavity);

  if (cfg.run_lid_cavity) {
    ImGui::Indent();
    ImGui::SliderFloat("Reynolds Number", &cfg.reynolds_number, 10.0f, 1000.0f, "%.0f");
    if (ImGui::IsItemHovered()) {
      ImGui::SetTooltip("Higher Re = more turbulent flow (harder to solve)");
    }
    ImGui::SliderInt("Time Steps", &cfg.lid_cavity_steps, 1000, 20000);
    ImGui::Unindent();
  }

  ImGui::Spacing();

  // Run button
  bool is_running = g_state.runner->IsRunning();
  if (is_running) {
    ImGui::BeginDisabled();
  }

  if (ImGui::Button("Run Projection Tests", ImVec2(180, 28))) {
    g_state.runner->RunPressureProjectionTests(
        cfg,
        [](float progress, const std::string& status) {
          // Progress callback
        },
        [](const TestSuiteResults& results) {
          std::lock_guard<std::mutex> lock(g_state.results_mutex);
          g_state.projection_results = results;
          g_state.all_results.push_back(results);
        });
  }

  if (is_running) {
    ImGui::EndDisabled();
    ImGui::SameLine();
    if (ImGui::Button("Cancel")) {
      g_state.runner->Cancel();
    }
    ImGui::SameLine();

    float progress = g_state.runner->GetProgress();
    char overlay[64];
    snprintf(overlay, sizeof(overlay), "%.0f%% - %s", progress * 100.0f,
             g_state.runner->GetCurrentTestName().c_str());
    ImGui::ProgressBar(progress, ImVec2(200, 0), overlay);
  }

  ImGui::Spacing();
  ImGui::Separator();
  ImGui::Spacing();

  // Display results
  std::lock_guard<std::mutex> lock(g_state.results_mutex);
  RenderResultsTable(g_state.projection_results);
}

}  // anonymous namespace

void RenderTestingPanel(const TestingPanelState& state) {
  if (ImGui::BeginTabBar("TestingTabs")) {
    if (ImGui::BeginTabItem("Advection")) {
      RenderAdvectionTab();
      ImGui::EndTabItem();
    }

    if (ImGui::BeginTabItem("Time Integrators")) {
      RenderTimeIntegratorTab();
      ImGui::EndTabItem();
    }

    if (ImGui::BeginTabItem("Pressure Projection")) {
      RenderPressureProjectionTab();
      ImGui::EndTabItem();
    }

    ImGui::EndTabBar();
  }

  // Global controls
  ImGui::Separator();
  ImGui::Spacing();

  if (ImGui::Button("Clear All Results")) {
    std::lock_guard<std::mutex> lock(g_state.results_mutex);
    g_state.all_results.clear();
    g_state.advection_results = TestSuiteResults();
    g_state.time_integrator_results = TestSuiteResults();
    g_state.projection_results = TestSuiteResults();
  }

  // Show total test history count
  ImGui::SameLine();
  ImGui::TextDisabled("(%zu test runs in history)", g_state.all_results.size());
}
