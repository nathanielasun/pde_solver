#include "solver_manager.h"
#include "handlers/solve_handler.h"

SolverManager::SolverManager(SharedState& state, std::mutex& state_mutex, GlViewer& viewer)
    : state_(state), state_mutex_(state_mutex), viewer_(viewer), cancel_requested_(false) {
}

SolverManager::~SolverManager() {
  Join();
}

void SolverManager::StartSolver() {
  std::lock_guard<std::mutex> lock(state_mutex_);
  if (state_.running) {
    return;
  }
  cancel_requested_.store(false);
  state_.running = true;
  state_.progress = 0.0;
  state_.phase = "solve";
  state_.status = "Solving...";
  state_.result.reset();
  state_.has_duration = false;
  state_.last_duration = 0.0;
  state_.stability_warning = false;
  state_.stability_frame = 0;
  state_.stability_ratio = 0.0;
  state_.stability_max = 0.0;
  state_.thread_total = 0;    // Reset thread count (GPU solves won't report this)
  state_.thread_active = 0;
  state_.residual_l2.clear();
  state_.residual_linf.clear();
  state_.detailed_progress = DetailedProgress();
  state_.detailed_progress.Start();
  state_.detailed_progress.phase = "solve";
}

void SolverManager::RequestStop() {
  if (cancel_requested_.exchange(true)) {
    return;  // Already requested
  }
  std::lock_guard<std::mutex> lock(state_mutex_);
  state_.status = "Stopping...";
}

void SolverManager::ReportStatus(const std::string& text) {
  std::lock_guard<std::mutex> lock(state_mutex_);
  state_.status = text;
  if (status_callback_) {
    status_callback_(text);
  }
}

void SolverManager::LaunchSolve(SolveHandlerState& solve_state) {
  // Join previous thread if still running
  if (solver_thread_.joinable()) {
    JoinIfFinished();
  }

  // Update solve_state to use our thread and callbacks.
  // Note: cancel_requested is a reference in SolveHandlerState and must be bound at construction.
  solve_state.solver_thread = &solver_thread_;
  solve_state.report_status = [this](const std::string& text) { ReportStatus(text); };
  solve_state.start_solver = [this]() { StartSolver(); };
  
  // Launch the solve operation (creates thread)
  ::LaunchSolve(solve_state);
}

bool SolverManager::IsRunning() const {
  std::lock_guard<std::mutex> lock(state_mutex_);
  return state_.running;
}

void SolverManager::JoinIfFinished() {
  bool running_now = false;
  {
    std::lock_guard<std::mutex> lock(state_mutex_);
    running_now = state_.running;
  }
  if (solver_thread_.joinable() && !running_now) {
    solver_thread_.join();
  }
}

void SolverManager::Join() {
  if (solver_thread_.joinable()) {
    solver_thread_.join();
  }
}

