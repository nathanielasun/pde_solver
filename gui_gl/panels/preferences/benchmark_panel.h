#ifndef BENCHMARK_PANEL_H
#define BENCHMARK_PANEL_H

#include <string>
#include <functional>
#include <vector>

// Benchmark configuration
struct BenchmarkConfig {
  std::string pde;
  double xmin, xmax, ymin, ymax;
  int nx, ny;
  std::string bc;
  std::string output_dir;
  double t_start, t_end;
  int frames;
};

// State structure for benchmark panel
struct BenchmarkPanelState {
  const BenchmarkConfig& config;
  bool running;
  
  // Callbacks
  std::function<void()> on_load_settings;
  std::function<void()> on_run_benchmark;
};

// Render the Benchmark panel using configured components.
void RenderBenchmarkPanel(BenchmarkPanelState& state, const std::vector<std::string>& components);

#endif  // BENCHMARK_PANEL_H
