#ifndef SOLVER_MANAGER_H
#define SOLVER_MANAGER_H

#include "app_state.h"
#include "GlViewer.h"
#include "handlers/solve_handler.h"
#include <atomic>
#include <thread>
#include <mutex>
#include <functional>

/**
 * SolverManager - Manages solver thread lifecycle and state synchronization.
 * 
 * Responsibilities:
 * - Manage solver thread creation and cleanup
 * - Handle cancel requests (atomic flag)
 * - Coordinate state updates (running, progress, status)
 * - Provide callbacks for starting/stopping solves
 */
class SolverManager {
 public:
  using StatusCallback = std::function<void(const std::string&)>;

  SolverManager(SharedState& state, std::mutex& state_mutex, GlViewer& viewer);
  ~SolverManager();

  // Non-copyable
  SolverManager(const SolverManager&) = delete;
  SolverManager& operator=(const SolverManager&) = delete;

  // Start solver (reset state, set running flag)
  void StartSolver();

  // Request solver to stop
  void RequestStop();

  // Launch solve operation (creates thread)
  // Takes SolveHandlerState constructed by caller (which has references to input parameters)
  void LaunchSolve(SolveHandlerState& solve_state);

  // Accessors for building SolveHandlerState without temporaries.
  std::thread* ThreadPtr() { return &solver_thread_; }
  std::atomic<bool>& CancelFlag() { return cancel_requested_; }
  std::function<void(const std::string&)> MakeReportStatusCallback() {
    return [this](const std::string& text) { ReportStatus(text); };
  }
  std::function<void()> MakeStartSolverCallback() {
    return [this]() { StartSolver(); };
  }

  // Check if solver is running
  bool IsRunning() const;

  // Get cancel requested flag
  bool IsCancelRequested() const { return cancel_requested_.load(); }

  // Join solver thread if joinable (call periodically or on shutdown)
  void JoinIfFinished();

  // Force join (for shutdown)
  void Join();

  // Set status callback
  void SetStatusCallback(StatusCallback callback) { status_callback_ = callback; }

 private:
  SharedState& state_;
  std::mutex& state_mutex_;
  GlViewer& viewer_;
  
  std::thread solver_thread_;
  std::atomic<bool> cancel_requested_;
  StatusCallback status_callback_;

  void ReportStatus(const std::string& text);
};

#endif  // SOLVER_MANAGER_H

