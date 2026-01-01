#include "test_runner.h"

#include "advection_tests.h"
#include "pde_types.h"
#include "pressure_projection_tests.h"
#include "time_integrator_tests.h"

#include <chrono>

TestRunner::TestRunner() = default;

TestRunner::~TestRunner() {
  Cancel();
  CleanupThread();
}

void TestRunner::CleanupThread() {
  if (worker_thread_ && worker_thread_->joinable()) {
    worker_thread_->join();
  }
  worker_thread_.reset();
}

bool TestRunner::IsRunning() const { return state_.running.load(); }

float TestRunner::GetProgress() const { return state_.progress.load(); }

void TestRunner::Cancel() { state_.cancel_requested.store(true); }

std::string TestRunner::GetCurrentTestName() const { return state_.GetTestName(); }

void TestRunner::RunAdvectionTests(const AdvectionTestConfig& config,
                                   TestProgressCallback on_progress,
                                   TestCompletionCallback on_complete) {
  CleanupThread();

  state_.running.store(true);
  state_.cancel_requested.store(false);
  state_.progress.store(0.0f);
  state_.current_test.store(0);

  worker_thread_ = std::make_unique<std::thread>(
      [this, config, on_progress, on_complete]() {
        auto results = ExecuteAdvectionTests(config, on_progress);
        state_.running.store(false);
        if (on_complete) {
          on_complete(results);
        }
      });
}

void TestRunner::RunTimeIntegratorTests(const TimeIntegratorTestConfig& config,
                                        TestProgressCallback on_progress,
                                        TestCompletionCallback on_complete) {
  CleanupThread();

  state_.running.store(true);
  state_.cancel_requested.store(false);
  state_.progress.store(0.0f);
  state_.current_test.store(0);

  worker_thread_ = std::make_unique<std::thread>(
      [this, config, on_progress, on_complete]() {
        auto results = ExecuteTimeIntegratorTests(config, on_progress);
        state_.running.store(false);
        if (on_complete) {
          on_complete(results);
        }
      });
}

void TestRunner::RunPressureProjectionTests(const PressureProjectionTestConfig& config,
                                            TestProgressCallback on_progress,
                                            TestCompletionCallback on_complete) {
  CleanupThread();

  state_.running.store(true);
  state_.cancel_requested.store(false);
  state_.progress.store(0.0f);
  state_.current_test.store(0);

  worker_thread_ = std::make_unique<std::thread>(
      [this, config, on_progress, on_complete]() {
        auto results = ExecutePressureProjectionTests(config, on_progress);
        state_.running.store(false);
        if (on_complete) {
          on_complete(results);
        }
      });
}

// Advection scheme list for indexing
static const std::vector<AdvectionScheme> kAdvectionSchemes = {
    AdvectionScheme::Upwind,     AdvectionScheme::LaxWendroff,
    AdvectionScheme::BeamWarming, AdvectionScheme::Fromm,
    AdvectionScheme::MinMod,     AdvectionScheme::Superbee,
    AdvectionScheme::VanLeer,    AdvectionScheme::MC};

static const char* kAdvectionSchemeNames[] = {
    "Upwind", "Lax-Wendroff", "Beam-Warming", "Fromm",
    "MinMod", "Superbee",     "Van Leer",     "MC"};

// Time integrator list for indexing
static const std::vector<TimeIntegrator> kTimeIntegrators = {
    TimeIntegrator::ForwardEuler, TimeIntegrator::RK2,
    TimeIntegrator::RK4,          TimeIntegrator::SSPRK2,
    TimeIntegrator::SSPRK3,       TimeIntegrator::BackwardEuler,
    TimeIntegrator::CrankNicolson};

static const char* kTimeIntegratorNames[] = {
    "Forward Euler", "RK2", "RK4", "SSPRK2", "SSPRK3", "Backward Euler", "Crank-Nicolson"};

TestSuiteResults TestRunner::ExecuteAdvectionTests(const AdvectionTestConfig& config,
                                                   TestProgressCallback on_progress) {
  TestSuiteResults suite;
  suite.suite_name = "Advection Discretization Tests";
  suite.timestamp = std::chrono::system_clock::now();

  auto start_time = std::chrono::high_resolution_clock::now();

  if (config.run_all_schemes) {
    // Run tests for all schemes
    int total = static_cast<int>(kAdvectionSchemes.size());
    state_.total_tests.store(total);

    for (int i = 0; i < total; ++i) {
      if (state_.cancel_requested.load()) {
        suite.cancelled = true;
        break;
      }

      AdvectionScheme scheme = kAdvectionSchemes[i];
      state_.SetTestName(std::string(kAdvectionSchemeNames[i]));
      state_.current_test.store(i);
      state_.progress.store(static_cast<float>(i) / total);

      if (on_progress) {
        on_progress(state_.progress.load(), state_.GetTestName());
      }

      // Run test suite for this scheme
      auto cli_results = RunAdvectionTestSuite(scheme);

      // Convert to GUI format
      for (const auto& r : cli_results) {
        GUITestResult gui_result;
        gui_result.passed = r.passed;
        gui_result.name = std::string(kAdvectionSchemeNames[i]) + ": " + r.name;
        gui_result.description = r.description;
        gui_result.category = "Advection";
        gui_result.error_l1 = r.error_l1;
        gui_result.error_l2 = r.error_l2;
        gui_result.error_linf = r.error_linf;
        gui_result.note = r.note;

        if (gui_result.passed) {
          suite.passed_count++;
        } else {
          suite.failed_count++;
        }

        suite.results.push_back(gui_result);
      }
    }
  } else {
    // Run tests for selected scheme only
    AdvectionScheme scheme =
        kAdvectionSchemes[std::min(config.scheme_index,
                                   static_cast<int>(kAdvectionSchemes.size()) - 1)];
    state_.SetTestName(std::string(kAdvectionSchemeNames[config.scheme_index]));
    state_.total_tests.store(1);

    if (on_progress) {
      on_progress(0.0f, state_.GetTestName());
    }

    auto cli_results = RunAdvectionTestSuite(scheme);

    for (const auto& r : cli_results) {
      if (state_.cancel_requested.load()) {
        suite.cancelled = true;
        break;
      }

      GUITestResult gui_result;
      gui_result.passed = r.passed;
      gui_result.name = r.name;
      gui_result.description = r.description;
      gui_result.category = "Advection";
      gui_result.error_l1 = r.error_l1;
      gui_result.error_l2 = r.error_l2;
      gui_result.error_linf = r.error_linf;
      gui_result.note = r.note;

      if (gui_result.passed) {
        suite.passed_count++;
      } else {
        suite.failed_count++;
      }

      suite.results.push_back(gui_result);
    }
  }

  state_.progress.store(1.0f);

  auto end_time = std::chrono::high_resolution_clock::now();
  suite.total_runtime_ms =
      std::chrono::duration<double, std::milli>(end_time - start_time).count();

  return suite;
}

TestSuiteResults TestRunner::ExecuteTimeIntegratorTests(
    const TimeIntegratorTestConfig& config, TestProgressCallback on_progress) {
  TestSuiteResults suite;
  suite.suite_name = "Time Integrator Tests";
  suite.timestamp = std::chrono::system_clock::now();

  auto start_time = std::chrono::high_resolution_clock::now();

  if (config.run_all_methods) {
    // Run tests for all methods
    int total = static_cast<int>(kTimeIntegrators.size());
    state_.total_tests.store(total);

    for (int i = 0; i < total; ++i) {
      if (state_.cancel_requested.load()) {
        suite.cancelled = true;
        break;
      }

      TimeIntegrator method = kTimeIntegrators[i];
      state_.SetTestName(std::string(kTimeIntegratorNames[i]));
      state_.current_test.store(i);
      state_.progress.store(static_cast<float>(i) / total);

      if (on_progress) {
        on_progress(state_.progress.load(), state_.GetTestName());
      }

      // Run test suite for this method
      auto cli_results = RunTimeIntegratorTestSuite(method);

      // Convert to GUI format
      for (const auto& r : cli_results) {
        GUITestResult gui_result;
        gui_result.passed = r.passed;
        gui_result.name = std::string(kTimeIntegratorNames[i]) + ": " + r.name;
        gui_result.description = r.description;
        gui_result.category = "TimeIntegrator";
        gui_result.error_l2 = r.error_l2;
        gui_result.error_linf = r.error_linf;
        gui_result.convergence_rate = r.convergence_rate;
        gui_result.note = r.note;

        if (gui_result.passed) {
          suite.passed_count++;
        } else {
          suite.failed_count++;
        }

        suite.results.push_back(gui_result);
      }
    }
  } else {
    // Run tests for selected method only
    TimeIntegrator method =
        kTimeIntegrators[std::min(config.method_index,
                                  static_cast<int>(kTimeIntegrators.size()) - 1)];
    state_.SetTestName(std::string(kTimeIntegratorNames[config.method_index]));
    state_.total_tests.store(1);

    if (on_progress) {
      on_progress(0.0f, state_.GetTestName());
    }

    auto cli_results = RunTimeIntegratorTestSuite(method);

    for (const auto& r : cli_results) {
      if (state_.cancel_requested.load()) {
        suite.cancelled = true;
        break;
      }

      GUITestResult gui_result;
      gui_result.passed = r.passed;
      gui_result.name = r.name;
      gui_result.description = r.description;
      gui_result.category = "TimeIntegrator";
      gui_result.error_l2 = r.error_l2;
      gui_result.error_linf = r.error_linf;
      gui_result.convergence_rate = r.convergence_rate;
      gui_result.note = r.note;

      if (gui_result.passed) {
        suite.passed_count++;
      } else {
        suite.failed_count++;
      }

      suite.results.push_back(gui_result);
    }

    // Run convergence study if requested
    if (config.run_convergence_study && !state_.cancel_requested.load()) {
      state_.SetTestName("Convergence Study");
      if (on_progress) {
        on_progress(0.5f, "Convergence Study");
      }

      auto conv_results = RunConvergenceStudy(method, config.step_counts);

      for (const auto& r : conv_results) {
        GUITestResult gui_result;
        gui_result.passed = r.passed;
        gui_result.name = "Convergence: " + r.name;
        gui_result.description = r.description;
        gui_result.category = "TimeIntegrator";
        gui_result.error_l2 = r.error_l2;
        gui_result.convergence_rate = r.convergence_rate;
        gui_result.note = r.note;

        if (gui_result.passed) {
          suite.passed_count++;
        } else {
          suite.failed_count++;
        }

        suite.results.push_back(gui_result);
      }
    }
  }

  state_.progress.store(1.0f);

  auto end_time = std::chrono::high_resolution_clock::now();
  suite.total_runtime_ms =
      std::chrono::duration<double, std::milli>(end_time - start_time).count();

  return suite;
}

TestSuiteResults TestRunner::ExecutePressureProjectionTests(
    const PressureProjectionTestConfig& config, TestProgressCallback on_progress) {
  TestSuiteResults suite;
  suite.suite_name = "Pressure Projection Tests";
  suite.timestamp = std::chrono::system_clock::now();

  auto start_time = std::chrono::high_resolution_clock::now();

  int total_tests = 0;
  if (config.run_basic_tests) total_tests += 3;  // Simple, DivFree, TaylorGreen
  if (config.run_convergence_study) total_tests += 1;
  if (config.run_lid_cavity) total_tests += 1;

  state_.total_tests.store(total_tests);
  int current = 0;

  if (config.run_basic_tests && !state_.cancel_requested.load()) {
    // Simple projection test
    state_.SetTestName("Simple Projection");
    state_.current_test.store(current);
    state_.progress.store(static_cast<float>(current) / total_tests);
    if (on_progress) {
      on_progress(state_.progress.load(), state_.GetTestName());
    }

    auto r = RunSimpleProjectionTest(config.grid_size, config.grid_size);
    GUITestResult gui_result;
    gui_result.passed = r.passed;
    gui_result.name = r.name;
    gui_result.description = r.description;
    gui_result.category = "Projection";
    gui_result.error_l2 = r.divergence_l2;
    gui_result.error_linf = r.divergence_linf;
    gui_result.iterations = r.poisson_iterations;
    gui_result.note = r.note;
    suite.results.push_back(gui_result);
    if (gui_result.passed) suite.passed_count++;
    else suite.failed_count++;
    current++;

    if (!state_.cancel_requested.load()) {
      // Divergence-free preservation test
      state_.SetTestName("Divergence-Free Preservation");
      state_.current_test.store(current);
      state_.progress.store(static_cast<float>(current) / total_tests);
      if (on_progress) {
        on_progress(state_.progress.load(), state_.GetTestName());
      }

      r = RunDivergenceFreeTest(config.grid_size, config.grid_size);
      gui_result.passed = r.passed;
      gui_result.name = r.name;
      gui_result.description = r.description;
      gui_result.error_l2 = r.divergence_l2;
      gui_result.error_linf = r.divergence_linf;
      gui_result.iterations = r.poisson_iterations;
      gui_result.note = r.note;
      suite.results.push_back(gui_result);
      if (gui_result.passed) suite.passed_count++;
      else suite.failed_count++;
      current++;
    }

    if (!state_.cancel_requested.load()) {
      // Taylor-Green vortex test
      state_.SetTestName("Taylor-Green Vortex");
      state_.current_test.store(current);
      state_.progress.store(static_cast<float>(current) / total_tests);
      if (on_progress) {
        on_progress(state_.progress.load(), state_.GetTestName());
      }

      r = RunTaylorGreenVortexTest(config.grid_size, config.grid_size, 0.01, 0.1, 10);
      gui_result.passed = r.passed;
      gui_result.name = r.name;
      gui_result.description = r.description;
      gui_result.error_l2 = r.divergence_l2;
      gui_result.error_linf = r.divergence_linf;
      gui_result.note = r.note;
      suite.results.push_back(gui_result);
      if (gui_result.passed) suite.passed_count++;
      else suite.failed_count++;
      current++;
    }
  }

  if (config.run_convergence_study && !state_.cancel_requested.load()) {
    state_.SetTestName("Convergence Study");
    state_.current_test.store(current);
    state_.progress.store(static_cast<float>(current) / total_tests);
    if (on_progress) {
      on_progress(state_.progress.load(), state_.GetTestName());
    }

    auto conv_results = RunProjectionConvergenceTest({16, 32, 64});
    for (const auto& r : conv_results) {
      GUITestResult gui_result;
      gui_result.passed = r.passed;
      gui_result.name = "Convergence: " + r.name;
      gui_result.description = r.description;
      gui_result.category = "Projection";
      gui_result.error_l2 = r.divergence_l2;
      gui_result.note = r.note;
      suite.results.push_back(gui_result);
      if (gui_result.passed) suite.passed_count++;
      else suite.failed_count++;
    }
    current++;
  }

  if (config.run_lid_cavity && !state_.cancel_requested.load()) {
    state_.SetTestName("Lid-Driven Cavity");
    state_.current_test.store(current);
    state_.progress.store(static_cast<float>(current) / total_tests);
    if (on_progress) {
      on_progress(state_.progress.load(), state_.GetTestName());
    }

    auto cavity_result = RunLidDrivenCavityTest(
        config.grid_size, config.grid_size,
        config.reynolds_number, config.lid_cavity_steps);

    GUITestResult gui_result;
    gui_result.passed = cavity_result.passed;
    gui_result.name = "Lid-Driven Cavity";
    gui_result.description =
        "Re=" + std::to_string(static_cast<int>(cavity_result.reynolds)) +
        ", " + std::to_string(cavity_result.nx) + "x" + std::to_string(cavity_result.ny);
    gui_result.category = "Projection";
    gui_result.note = cavity_result.note;
    suite.results.push_back(gui_result);
    if (gui_result.passed) suite.passed_count++;
    else suite.failed_count++;
    current++;
  }

  if (state_.cancel_requested.load()) {
    suite.cancelled = true;
  }

  state_.progress.store(1.0f);

  auto end_time = std::chrono::high_resolution_clock::now();
  suite.total_runtime_ms =
      std::chrono::duration<double, std::milli>(end_time - start_time).count();

  return suite;
}
