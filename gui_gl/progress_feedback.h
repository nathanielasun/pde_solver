#ifndef PROGRESS_FEEDBACK_H
#define PROGRESS_FEEDBACK_H

#include <string>
#include <chrono>

// Enhanced progress structure with detailed metrics
struct DetailedProgress {
  std::string phase = "idle";  // "solve", "time", "write", "idle"
  double phase_progress = 0.0;  // 0.0 to 1.0
  int current_iteration = 0;
  int total_iterations = 0;
  double elapsed_time = 0.0;  // seconds
  double estimated_remaining = 0.0;  // seconds
  std::string backend_name;
  std::string backend_note;
  double memory_mb = 0.0;
  double iterations_per_second = 0.0;
  bool is_converged = false;
  double residual_l2 = 0.0;
  double residual_linf = 0.0;
  
  // Time tracking
  std::chrono::steady_clock::time_point start_time;
  std::chrono::steady_clock::time_point last_update;
  int iteration_count_since_last_update = 0;
  
  // Initialize timing
  void Start() {
    start_time = std::chrono::steady_clock::now();
    last_update = start_time;
    iteration_count_since_last_update = 0;
  }
  
  // Update progress and compute metrics
  void Update(double progress, int iteration, int total_iters = 0) {
    auto now = std::chrono::steady_clock::now();
    phase_progress = progress;
    current_iteration = iteration;
    total_iterations = total_iters;
    
    elapsed_time = std::chrono::duration<double>(now - start_time).count();
    
    // Compute iterations per second
    double time_since_last = std::chrono::duration<double>(now - last_update).count();
    if (time_since_last > 0.1) {  // Update every 100ms
      iterations_per_second = iteration_count_since_last_update / time_since_last;
      iteration_count_since_last_update = 0;
      last_update = now;
    } else {
      iteration_count_since_last_update++;
    }
    
    // Estimate remaining time
    if (iterations_per_second > 0 && total_iterations > 0) {
      int remaining_iterations = total_iterations - current_iteration;
      estimated_remaining = remaining_iterations / iterations_per_second;
    } else if (iterations_per_second > 0 && phase_progress > 0.01) {
      // Estimate based on progress rate
      double progress_rate = phase_progress / elapsed_time;
      if (progress_rate > 0) {
        estimated_remaining = (1.0 - phase_progress) / progress_rate;
      }
    }
  }
  
  // Format time string
  std::string FormatTime(double seconds) const {
    if (seconds < 60.0) {
      return std::to_string(static_cast<int>(seconds)) + "s";
    } else if (seconds < 3600.0) {
      int mins = static_cast<int>(seconds / 60.0);
      int secs = static_cast<int>(seconds) % 60;
      return std::to_string(mins) + "m " + std::to_string(secs) + "s";
    } else {
      int hours = static_cast<int>(seconds / 3600.0);
      int mins = static_cast<int>((seconds - hours * 3600.0) / 60.0);
      return std::to_string(hours) + "h " + std::to_string(mins) + "m";
    }
  }
};

#endif  // PROGRESS_FEEDBACK_H

