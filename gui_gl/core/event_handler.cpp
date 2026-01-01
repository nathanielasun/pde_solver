#include "event_handler.h"

// #region agent log
#include <fstream>
#include <chrono>
#include <string>
static void AgentLog8(const char* location, const char* message, const char* hypothesisId,
                      const std::string& dataJson = "{}") {
  std::ofstream f("/Users/nathaniel.sun/Desktop/programming/cursor/.cursor/debug.log",
                  std::ios::app);
  if (!f) return;
  const auto ts = std::chrono::duration_cast<std::chrono::milliseconds>(
                      std::chrono::system_clock::now().time_since_epoch())
                      .count();
  f << "{\"sessionId\":\"debug-session\",\"runId\":\"run8\",\"hypothesisId\":\""
    << hypothesisId << "\",\"location\":\"" << location << "\",\"message\":\""
    << message << "\",\"data\":" << dataJson << ",\"timestamp\":" << ts << "}\n";
}
// #endregion agent log

bool EventHandler::IsModPressed(const ImGuiIO& io) {
#ifdef __APPLE__
  return io.KeySuper;
#else
  return io.KeyCtrl;
#endif
}

void EventHandler::ProcessShortcuts(const ImGuiIO& io) {
  const bool mod_pressed = IsModPressed(io);
  const bool shift_pressed = io.KeyShift;

  auto trigger = [&](AppAction action) {
    if (callbacks_.can_execute && !callbacks_.can_execute(action)) {
      return;
    }
    if (callbacks_.on_action) {
      callbacks_.on_action(action);
    }
  };

  // F1 - Help
  if (ImGui::IsKeyPressed(ImGuiKey_F1)) {
    trigger(AppAction::kHelpSearch);
  }

  // Escape - Stop solver
  if (ImGui::IsKeyPressed(ImGuiKey_Escape)) {
    trigger(AppAction::kStop);
  }

  // Mod+S - Solve
  if (mod_pressed && !shift_pressed && ImGui::IsKeyPressed(ImGuiKey_S)) {
    trigger(AppAction::kSolve);
  }

  // Mod+L - Load latest
  if (mod_pressed && !shift_pressed && ImGui::IsKeyPressed(ImGuiKey_L)) {
    trigger(AppAction::kLoadLatest);
  }

  // Mod+R - Reset view
  if (mod_pressed && !shift_pressed && ImGui::IsKeyPressed(ImGuiKey_R)) {
    // #region agent log
    {
      static int budget = 200;
      if (budget-- > 0) {
        AgentLog8("gui_gl/core/event_handler.cpp:ProcessShortcuts",
                  "Shortcut triggered: Mod+R (reset view)",
                  "P",
                  std::string("{\"keySuper\":") + (io.KeySuper ? "true" : "false") +
                      ",\"keyCtrl\":" + (io.KeyCtrl ? "true" : "false") +
                      ",\"keyShift\":" + (io.KeyShift ? "true" : "false") +
                      ",\"wantCaptureKeyboard\":" + (io.WantCaptureKeyboard ? "true" : "false") +
                      ",\"running\":null}");
      }
    }
    // #endregion agent log
    trigger(AppAction::kResetView);
  }

  // Mod+Z - Undo (without Shift)
  if (mod_pressed && !shift_pressed && ImGui::IsKeyPressed(ImGuiKey_Z)) {
    trigger(AppAction::kUndo);
  }

  // Mod+Shift+Z - Redo
  if (mod_pressed && shift_pressed && ImGui::IsKeyPressed(ImGuiKey_Z)) {
    trigger(AppAction::kRedo);
  }

  // Mod+1/2/3 - Switch tabs
  if (mod_pressed && !shift_pressed && ImGui::IsKeyPressed(ImGuiKey_1)) {
    trigger(AppAction::kGoMainTab);
  }
  if (mod_pressed && !shift_pressed && ImGui::IsKeyPressed(ImGuiKey_2)) {
    trigger(AppAction::kGoInspectTab);
  }
  if (mod_pressed && !shift_pressed && ImGui::IsKeyPressed(ImGuiKey_3)) {
    trigger(AppAction::kGoPreferencesTab);
  }
}
