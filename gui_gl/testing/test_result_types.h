#ifndef TEST_RESULT_TYPES_H
#define TEST_RESULT_TYPES_H

#include <atomic>
#include <chrono>
#include <mutex>
#include <string>
#include <vector>

// Generic test result for GUI display (unified format from CLI tests)
struct GUITestResult {
  bool passed = false;
  std::string name;
  std::string description;
  std::string category;  // "Advection", "TimeIntegrator", "Projection"
  double error_l1 = 0.0;
  double error_l2 = 0.0;
  double error_linf = 0.0;
  double convergence_rate = 0.0;
  int iterations = 0;
  double runtime_ms = 0.0;
  std::string note;
};

// Test suite results container
struct TestSuiteResults {
  std::string suite_name;
  std::vector<GUITestResult> results;
  int passed_count = 0;
  int failed_count = 0;
  double total_runtime_ms = 0.0;
  std::chrono::system_clock::time_point timestamp;
  bool cancelled = false;
};

// Advection test configuration
struct AdvectionTestConfig {
  int scheme_index = 0;  // Index into scheme list
  int grid_size = 100;
  float cfl = 0.5f;
  bool run_all_schemes = false;
};

// Time integrator test configuration
struct TimeIntegratorTestConfig {
  int method_index = 0;  // Index into method list
  bool run_convergence_study = true;
  bool run_all_methods = false;
  std::vector<int> step_counts = {50, 100, 200, 400};
};

// Pressure projection test configuration
struct PressureProjectionTestConfig {
  bool run_basic_tests = true;
  bool run_convergence_study = true;
  bool run_lid_cavity = false;
  float reynolds_number = 100.0f;
  int lid_cavity_steps = 5000;
  int grid_size = 32;
};

// Unified test configuration for the panel
struct TestPanelConfig {
  AdvectionTestConfig advection;
  TimeIntegratorTestConfig time_integrator;
  PressureProjectionTestConfig projection;
};

// Test execution state (for progress tracking across threads)
struct TestExecutionState {
  std::atomic<bool> running{false};
  std::atomic<bool> cancel_requested{false};
  std::atomic<int> current_test{0};
  std::atomic<int> total_tests{0};
  std::atomic<float> progress{0.0f};
  std::string current_test_name;
  mutable std::mutex status_mutex;

  void SetTestName(const std::string& name) {
    std::lock_guard<std::mutex> lock(status_mutex);
    current_test_name = name;
  }

  std::string GetTestName() const {
    std::lock_guard<std::mutex> lock(status_mutex);
    return current_test_name;
  }
};

#endif  // TEST_RESULT_TYPES_H
