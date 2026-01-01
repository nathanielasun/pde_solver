#ifndef TEST_RUNNER_H
#define TEST_RUNNER_H

#include "test_result_types.h"

#include <functional>
#include <memory>
#include <thread>

// Callback types for async test execution (prefixed to avoid collision with progress.h)
using TestProgressCallback = std::function<void(float progress, const std::string& status)>;
using TestCompletionCallback = std::function<void(const TestSuiteResults& results)>;

// Test runner manages background test execution
class TestRunner {
 public:
  TestRunner();
  ~TestRunner();

  // Launch tests in background thread
  void RunAdvectionTests(const AdvectionTestConfig& config,
                         TestProgressCallback on_progress,
                         TestCompletionCallback on_complete);

  void RunTimeIntegratorTests(const TimeIntegratorTestConfig& config,
                              TestProgressCallback on_progress,
                              TestCompletionCallback on_complete);

  void RunPressureProjectionTests(const PressureProjectionTestConfig& config,
                                  TestProgressCallback on_progress,
                                  TestCompletionCallback on_complete);

  // Cancel running tests
  void Cancel();

  // Query state
  bool IsRunning() const;
  float GetProgress() const;
  std::string GetCurrentTestName() const;

 private:
  TestExecutionState state_;
  std::unique_ptr<std::thread> worker_thread_;

  void CleanupThread();

  // Internal test execution
  TestSuiteResults ExecuteAdvectionTests(const AdvectionTestConfig& config,
                                         TestProgressCallback on_progress);
  TestSuiteResults ExecuteTimeIntegratorTests(const TimeIntegratorTestConfig& config,
                                              TestProgressCallback on_progress);
  TestSuiteResults ExecutePressureProjectionTests(const PressureProjectionTestConfig& config,
                                                  TestProgressCallback on_progress);
};

#endif  // TEST_RUNNER_H
