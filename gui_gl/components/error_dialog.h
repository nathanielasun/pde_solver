#ifndef ERROR_DIALOG_H
#define ERROR_DIALOG_H

#include "../app_state.h"

// Renders a centralized modal error dialog for critical errors.
// The dialog is driven by SharedState::last_error and SharedState::error_dialog_open.
class ErrorDialogComponent {
 public:
  void Render(SharedState& state, std::mutex& state_mutex);

 private:
  // Copy of the most recent error to display (avoid holding mutex during ImGui calls).
  std::optional<ErrorInfo> active_error_;
};

#endif  // ERROR_DIALOG_H


