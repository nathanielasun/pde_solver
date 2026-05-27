#include "event_handler.h"

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
