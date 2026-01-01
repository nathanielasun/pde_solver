#include "run_summary.h"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <fstream>
#include <iomanip>
#include <sstream>

#include <nlohmann/json.hpp>

#include "solver_tokens.h"

namespace {
using json = nlohmann::json;

bool EndsWith(const std::string& text, const std::string& suffix) {
  if (suffix.size() > text.size()) {
    return false;
  }
  return std::equal(suffix.rbegin(), suffix.rend(), text.rbegin());
}

std::string TimestampUtc() {
  const auto now = std::chrono::system_clock::now();
  const std::time_t t = std::chrono::system_clock::to_time_t(now);
  std::tm utc = {};
#if defined(_WIN32)
  gmtime_s(&utc, &t);
#else
  gmtime_r(&t, &utc);
#endif
  std::ostringstream oss;
  oss << std::put_time(&utc, "%Y-%m-%dT%H:%M:%SZ");
  return oss.str();
}

bool WriteFile(const std::filesystem::path& path, const std::string& data, std::string* error) {
  std::ofstream out(path);
  if (!out.is_open()) {
    if (error) {
      *error = "failed to write file: " + path.string();
    }
    return false;
  }
  out << data;
  if (!out.good()) {
    if (error) {
      *error = "failed to write file: " + path.string();
    }
    return false;
  }
  return true;
}

bool ReadFile(const std::filesystem::path& path, std::string* out, std::string* error) {
  std::ifstream in(path);
  if (!in.is_open()) {
    if (error) {
      *error = "failed to open file: " + path.string();
    }
    return false;
  }
  std::ostringstream buffer;
  buffer << in.rdbuf();
  *out = buffer.str();
  return true;
}
}  // namespace

std::filesystem::path SummarySidecarPath(const std::filesystem::path& output_path) {
  if (output_path.empty()) {
    return {};
  }
  const std::string name = output_path.filename().string();
  if (EndsWith(name, ".summary.json")) {
    return output_path;
  }
  return std::filesystem::path(output_path.string() + ".summary.json");
}

std::string BuildRunSummaryJson(const RunSummaryData& data, int indent) {
  json root;
  root["schema_version"] = 1;
  root["generated_at"] = TimestampUtc();

  std::string run_spec = SerializeRunConfig(data.run_config, 0);
  try {
    root["run"] = json::parse(run_spec);
  } catch (const json::exception&) {
    root["run"] = run_spec;
  }

  json solve;
  solve["requested_backend"] = BackendToken(data.requested_backend);
  solve["selected_backend"] = BackendToken(data.selected_backend);
  solve["time_series"] = data.time_series;
  if (!data.backend_note.empty()) {
    solve["backend_note"] = data.backend_note;
  }
  if (data.frame_index >= 0) {
    solve["frame_index"] = data.frame_index;
    solve["frame_time"] = data.frame_time;
  }
  root["solve"] = solve;

  json timing;
  if (data.solve_seconds > 0.0) {
    timing["solve_seconds"] = data.solve_seconds;
  }
  if (data.write_seconds > 0.0) {
    timing["write_seconds"] = data.write_seconds;
  }
  if (data.total_seconds > 0.0) {
    timing["total_seconds"] = data.total_seconds;
  } else if (data.solve_seconds > 0.0 || data.write_seconds > 0.0) {
    timing["total_seconds"] = data.solve_seconds + data.write_seconds;
  }
  if (!timing.empty()) {
    root["timing"] = timing;
  }

  if (!data.frame_times.empty()) {
    json frames;
    frames["count"] = static_cast<int>(data.frame_times.size());
    frames["times"] = data.frame_times;
    root["frames"] = frames;
  }

  json convergence;
  convergence["residual_l2"] = data.residual_l2;
  convergence["residual_linf"] = data.residual_linf;
  convergence["solver_tol"] = data.run_config.solver.tol;
  convergence["max_iter"] = data.run_config.solver.max_iter;
  const size_t samples = std::min(data.residual_l2_history.size(),
                                  data.residual_linf_history.size());
  convergence["residual_samples"] = static_cast<int>(samples);
  if (!data.residual_iters.empty()) {
    convergence["residual_iters"] = data.residual_iters;
  }
  if (!data.residual_l2_history.empty()) {
    convergence["residual_l2_history"] = data.residual_l2_history;
  }
  if (!data.residual_linf_history.empty()) {
    convergence["residual_linf_history"] = data.residual_linf_history;
  }
  root["convergence"] = convergence;

  const bool has_monitor_history =
      !data.monitor_mass_history.empty() ||
      !data.monitor_energy_history.empty() ||
      !data.monitor_max_abs_history.empty() ||
      !data.monitor_mass_drift_history.empty() ||
      !data.monitor_energy_drift_history.empty();
  const bool has_monitor_sample = data.monitors_enabled;
  const bool has_monitor_warnings =
      data.monitor_mass_warning || data.monitor_energy_warning || data.monitor_blowup_warning;
  if (has_monitor_sample || has_monitor_history || has_monitor_warnings) {
    json monitors;
    if (has_monitor_sample) {
      monitors["mass"] = data.monitor_mass;
      monitors["energy"] = data.monitor_energy;
      monitors["max_abs"] = data.monitor_max_abs;
      monitors["mass_drift"] = data.monitor_mass_drift;
      monitors["energy_drift"] = data.monitor_energy_drift;
    }
    if (!data.monitor_mass_history.empty()) {
      monitors["mass_history"] = data.monitor_mass_history;
    }
    if (!data.monitor_energy_history.empty()) {
      monitors["energy_history"] = data.monitor_energy_history;
    }
    if (!data.monitor_max_abs_history.empty()) {
      monitors["max_abs_history"] = data.monitor_max_abs_history;
    }
    if (!data.monitor_mass_drift_history.empty()) {
      monitors["mass_drift_history"] = data.monitor_mass_drift_history;
    }
    if (!data.monitor_energy_drift_history.empty()) {
      monitors["energy_drift_history"] = data.monitor_energy_drift_history;
    }
    json warnings;
    if (data.monitor_mass_warning) {
      warnings["mass_drift_frame"] = data.monitor_mass_warning_frame;
    }
    if (data.monitor_energy_warning) {
      warnings["energy_drift_frame"] = data.monitor_energy_warning_frame;
    }
    if (data.monitor_blowup_warning) {
      json blowup;
      blowup["frame"] = data.monitor_blowup_warning_frame;
      blowup["ratio"] = data.monitor_blowup_ratio;
      blowup["max_abs"] = data.monitor_blowup_max;
      warnings["blowup"] = blowup;
    }
    if (!warnings.empty()) {
      monitors["warnings"] = warnings;
    }
    root["monitors"] = monitors;
  }

  json output;
  output["path"] = data.output_path;
  if (!data.output_path.empty()) {
    std::filesystem::path out_path(data.output_path);
    if (out_path.has_extension()) {
      output["format"] = out_path.extension().string();
    }
  }
  root["output"] = output;

  return root.dump(indent);
}

bool WriteRunSummarySidecar(const std::filesystem::path& output_path,
                            const RunSummaryData& data,
                            std::string* error) {
  if (output_path.empty()) {
    if (error) {
      *error = "missing output path for summary";
    }
    return false;
  }
  const std::filesystem::path summary_path = SummarySidecarPath(output_path);
  if (summary_path.empty()) {
    if (error) {
      *error = "invalid summary path";
    }
    return false;
  }
  if (summary_path.has_parent_path()) {
    std::error_code ec;
    std::filesystem::create_directories(summary_path.parent_path(), ec);
    if (ec) {
      if (error) {
        *error = "failed to create summary directory: " + ec.message();
      }
      return false;
    }
  }
  const std::string payload = BuildRunSummaryJson(data, 2);
  return WriteFile(summary_path, payload, error);
}

bool ReadRunSummarySidecar(const std::filesystem::path& output_path,
                           std::string* summary_json,
                           std::string* error) {
  const std::filesystem::path summary_path = SummarySidecarPath(output_path);
  if (summary_path.empty()) {
    if (error) {
      *error = "invalid summary path";
    }
    return false;
  }
  if (!std::filesystem::exists(summary_path)) {
    if (error) {
      *error = "summary sidecar not found: " + summary_path.string();
    }
    return false;
  }
  return ReadFile(summary_path, summary_json, error);
}
