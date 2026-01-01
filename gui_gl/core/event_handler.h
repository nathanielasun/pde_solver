#ifndef EVENT_HANDLER_H
#define EVENT_HANDLER_H

#include "app_actions.h"
#include "imgui.h"
#include <functional>

/**
 * EventHandler - Handles keyboard shortcuts and input processing.
 * 
 * Responsibilities:
 * - Process keyboard shortcuts (Mod+S, Mod+L, Mod+R, Mod+Z, etc.)
 * - Route input events to appropriate callbacks
 * - Handle platform-specific modifier key differences (Cmd on Mac, Ctrl elsewhere)
 */
class EventHandler {
 public:
  // Callback types
  using ActionHandler = std::function<void(AppAction)>;
  using CanExecuteCallback = std::function<bool(AppAction)>;

  struct Callbacks {
    ActionHandler on_action;
    CanExecuteCallback can_execute;
  };

  EventHandler() = default;

  // Set callbacks
  void SetCallbacks(const Callbacks& callbacks) { callbacks_ = callbacks; }

  // Process keyboard shortcuts - call each frame
  void ProcessShortcuts(const ImGuiIO& io);

  // Check if a modifier key is pressed (Cmd on Mac, Ctrl elsewhere)
  static bool IsModPressed(const ImGuiIO& io);

 private:
  Callbacks callbacks_;
};

#endif  // EVENT_HANDLER_H
