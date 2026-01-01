#ifndef RUN_SUMMARY_H
#define RUN_SUMMARY_H

#include <filesystem>
#include <string>
#include <vector>

#include "backend.h"
#include "run_config.h"

struct RunSummaryData {
  RunConfig run_config;
  BackendKind requested_backend = BackendKind::Auto;
  BackendKind selected_backend = BackendKind::CPU;
  std::string backend_note;
  std::string output_path;
  bool time_series = false;
  int frame_index = -1;
  double frame_time = 0.0;

  double solve_seconds = 0.0;
  double write_seconds = 0.0;
  double total_seconds = 0.0;

  double residual_l2 = 0.0;
  double residual_linf = 0.0;
  std::vector<int> residual_iters;
  std::vector<double> residual_l2_history;
  std::vector<double> residual_linf_history;

  bool monitors_enabled = false;
  double monitor_mass = 0.0;
  double monitor_energy = 0.0;
  double monitor_max_abs = 0.0;
  double monitor_mass_drift = 0.0;
  double monitor_energy_drift = 0.0;
  bool monitor_mass_warning = false;
  bool monitor_energy_warning = false;
  bool monitor_blowup_warning = false;
  int monitor_mass_warning_frame = -1;
  int monitor_energy_warning_frame = -1;
  int monitor_blowup_warning_frame = -1;
  double monitor_blowup_ratio = 0.0;
  double monitor_blowup_max = 0.0;
  std::vector<double> monitor_mass_history;
  std::vector<double> monitor_energy_history;
  std::vector<double> monitor_max_abs_history;
  std::vector<double> monitor_mass_drift_history;
  std::vector<double> monitor_energy_drift_history;

  std::vector<double> frame_times;
};

std::filesystem::path SummarySidecarPath(const std::filesystem::path& output_path);
std::string BuildRunSummaryJson(const RunSummaryData& data, int indent = 2);
bool WriteRunSummarySidecar(const std::filesystem::path& output_path,
                            const RunSummaryData& data,
                            std::string* error);
bool ReadRunSummarySidecar(const std::filesystem::path& output_path,
                           std::string* summary_json,
                           std::string* error);

#endif  // RUN_SUMMARY_H
