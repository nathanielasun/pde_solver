#include "dataset_tools.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <vector>

#include <nlohmann/json.hpp>

namespace {
using json = nlohmann::json;

constexpr const char* kSummarySuffix = ".summary.json";
constexpr const char* kMetaSuffix = ".meta.json";

bool EndsWith(const std::string& text, const std::string& suffix) {
  if (suffix.size() > text.size()) {
    return false;
  }
  return std::equal(suffix.rbegin(), suffix.rend(), text.rbegin());
}

std::filesystem::path StripSuffix(const std::filesystem::path& path,
                                  const std::string& suffix) {
  const std::string name = path.string();
  if (!EndsWith(name, suffix)) {
    return {};
  }
  return std::filesystem::path(name.substr(0, name.size() - suffix.size()));
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

bool WriteFile(const std::filesystem::path& path, const std::string& data, std::string* error) {
  if (path.has_parent_path()) {
    std::error_code ec;
    std::filesystem::create_directories(path.parent_path(), ec);
    if (ec) {
      if (error) {
        *error = "failed to create output directory: " + ec.message();
      }
      return false;
    }
  }
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

std::vector<std::filesystem::path> CollectFiles(const std::filesystem::path& root,
                                                const std::string& suffix) {
  std::vector<std::filesystem::path> files;
  if (!std::filesystem::exists(root) || !std::filesystem::is_directory(root)) {
    return files;
  }
  std::error_code ec;
  std::filesystem::recursive_directory_iterator it(
      root, std::filesystem::directory_options::skip_permission_denied, ec);
  std::filesystem::recursive_directory_iterator end;
  for (; it != end; it.increment(ec)) {
    if (ec) {
      ec.clear();
      continue;
    }
    if (!it->is_regular_file(ec)) {
      continue;
    }
    const std::string path_str = it->path().string();
    if (EndsWith(path_str, suffix)) {
      files.push_back(it->path());
    }
  }
  std::sort(files.begin(), files.end());
  return files;
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

bool ExtractOutputPath(const json& root, std::string* out) {
  if (!root.contains("output")) {
    return false;
  }
  const json& output = root.at("output");
  if (!output.is_object() || !output.contains("path") || !output.at("path").is_string()) {
    return false;
  }
  if (out) {
    *out = output.at("path").get<std::string>();
  }
  return true;
}

std::filesystem::path ResolveOutputPath(const std::filesystem::path& root,
                                        const std::filesystem::path& sidecar_path,
                                        const std::string& output_path) {
  std::filesystem::path candidate;
  if (!output_path.empty()) {
    candidate = std::filesystem::path(output_path);
  } else {
    candidate = StripSuffix(sidecar_path, kSummarySuffix);
    if (candidate.empty()) {
      candidate = StripSuffix(sidecar_path, kMetaSuffix);
    }
  }
  if (candidate.empty()) {
    return {};
  }
  if (candidate.is_absolute()) {
    return candidate;
  }
  const std::filesystem::path from_sidecar = sidecar_path.parent_path() / candidate;
  if (std::filesystem::exists(from_sidecar)) {
    return from_sidecar;
  }
  const std::filesystem::path from_root = root / candidate;
  if (std::filesystem::exists(from_root)) {
    return from_root;
  }
  return from_sidecar;
}

struct StatAccumulator {
  int count = 0;
  double sum = 0.0;
  double min = 0.0;
  double max = 0.0;

  void Add(double value) {
    if (!std::isfinite(value)) {
      return;
    }
    if (count == 0) {
      min = value;
      max = value;
    } else {
      min = std::min(min, value);
      max = std::max(max, value);
    }
    sum += value;
    count++;
  }

  json ToJson() const {
    if (count == 0) {
      return json();
    }
    json out;
    out["min"] = min;
    out["max"] = max;
    out["mean"] = sum / static_cast<double>(count);
    return out;
  }
};

bool ParseJsonFile(const std::filesystem::path& path, json* out, std::string* error) {
  std::string payload;
  if (!ReadFile(path, &payload, error)) {
    return false;
  }
  try {
    if (out) {
      *out = json::parse(payload);
    }
  } catch (const json::exception& exc) {
    if (error) {
      *error = exc.what();
    }
    return false;
  }
  return true;
}

bool ResolveOutputFromSidecar(const std::filesystem::path& root,
                              const std::filesystem::path& sidecar_path,
                              std::filesystem::path* output_path,
                              std::string* error) {
  json root_json;
  std::string json_error;
  std::string output_path_str;
  if (ParseJsonFile(sidecar_path, &root_json, &json_error)) {
    ExtractOutputPath(root_json, &output_path_str);
  }
  if (!json_error.empty() && error) {
    *error = json_error;
  }
  const std::filesystem::path resolved = ResolveOutputPath(root, sidecar_path, output_path_str);
  if (output_path) {
    *output_path = resolved;
  }
  return !resolved.empty();
}

bool IsFrameSummary(const json& root) {
  if (!root.contains("solve")) {
    return false;
  }
  const json& solve = root.at("solve");
  if (!solve.is_object() || !solve.contains("frame_index")) {
    return false;
  }
  const json& idx = solve.at("frame_index");
  return idx.is_number_integer() && idx.get<int>() >= 0;
}
}  // namespace

bool BuildDatasetIndex(const std::filesystem::path& root,
                       DatasetIndexResult* out,
                       std::string* error) {
  if (!out) {
    if (error) {
      *error = "missing dataset index output";
    }
    return false;
  }
  *out = {};
  if (!std::filesystem::exists(root) || !std::filesystem::is_directory(root)) {
    if (error) {
      *error = "dataset root not found: " + root.string();
    }
    return false;
  }

  const std::vector<std::filesystem::path> summaries = CollectFiles(root, kSummarySuffix);
  json runs = json::array();
  StatAccumulator residual_l2;
  StatAccumulator residual_linf;
  StatAccumulator solve_seconds;

  int skipped = 0;
  for (const auto& summary_path : summaries) {
    json summary;
    std::string parse_error;
    if (!ParseJsonFile(summary_path, &summary, &parse_error)) {
      skipped++;
      continue;
    }
    if (IsFrameSummary(summary)) {
      continue;
    }

    std::string output_path_str;
    ExtractOutputPath(summary, &output_path_str);
    const std::filesystem::path resolved_output =
        ResolveOutputPath(root, summary_path, output_path_str);
    const bool output_exists =
        !resolved_output.empty() && std::filesystem::exists(resolved_output);

    bool time_series = false;
    std::string selected_backend;
    if (summary.contains("solve") && summary.at("solve").is_object()) {
      const json& solve = summary.at("solve");
      if (solve.contains("time_series") && solve.at("time_series").is_boolean()) {
        time_series = solve.at("time_series").get<bool>();
      }
      if (solve.contains("selected_backend") && solve.at("selected_backend").is_string()) {
        selected_backend = solve.at("selected_backend").get<std::string>();
      }
    }

    int frame_count = time_series ? 0 : 1;
    if (time_series && summary.contains("frames") && summary.at("frames").is_object()) {
      const json& frames = summary.at("frames");
      if (frames.contains("count") && frames.at("count").is_number_integer()) {
        frame_count = frames.at("count").get<int>();
      } else if (frames.contains("times") && frames.at("times").is_array()) {
        frame_count = static_cast<int>(frames.at("times").size());
      }
    }

    double res_l2 = std::numeric_limits<double>::quiet_NaN();
    double res_linf = std::numeric_limits<double>::quiet_NaN();
    if (summary.contains("convergence") && summary.at("convergence").is_object()) {
      const json& conv = summary.at("convergence");
      if (conv.contains("residual_l2") && conv.at("residual_l2").is_number()) {
        res_l2 = conv.at("residual_l2").get<double>();
      }
      if (conv.contains("residual_linf") && conv.at("residual_linf").is_number()) {
        res_linf = conv.at("residual_linf").get<double>();
      }
    }

    double solve_time = std::numeric_limits<double>::quiet_NaN();
    if (summary.contains("timing") && summary.at("timing").is_object()) {
      const json& timing = summary.at("timing");
      if (timing.contains("solve_seconds") && timing.at("solve_seconds").is_number()) {
        solve_time = timing.at("solve_seconds").get<double>();
      }
    }

    std::string grid;
    std::string domain_bounds;
    if (summary.contains("run") && summary.at("run").is_object()) {
      const json& run = summary.at("run");
      if (run.contains("domain") && run.at("domain").is_object()) {
        const json& domain = run.at("domain");
        if (domain.contains("grid") && domain.at("grid").is_string()) {
          grid = domain.at("grid").get<std::string>();
        }
        if (domain.contains("bounds") && domain.at("bounds").is_string()) {
          domain_bounds = domain.at("bounds").get<std::string>();
        }
      }
    }

    bool monitor_warning = false;
    if (summary.contains("monitors") && summary.at("monitors").is_object()) {
      const json& monitors = summary.at("monitors");
      if (monitors.contains("warnings") && monitors.at("warnings").is_object()) {
        monitor_warning = !monitors.at("warnings").empty();
      }
    }

    json entry;
    entry["summary_path"] = summary_path.string();
    if (!output_path_str.empty()) {
      entry["output_path"] = output_path_str;
    }
    if (!resolved_output.empty()) {
      entry["output_path_resolved"] = resolved_output.string();
    }
    entry["output_exists"] = output_exists;
    entry["time_series"] = time_series;
    entry["frame_count"] = frame_count;
    if (!selected_backend.empty()) {
      entry["backend"] = selected_backend;
    }
    if (!grid.empty()) {
      entry["grid"] = grid;
    }
    if (!domain_bounds.empty()) {
      entry["domain_bounds"] = domain_bounds;
    }
    if (std::isfinite(res_l2)) {
      entry["residual_l2"] = res_l2;
    }
    if (std::isfinite(res_linf)) {
      entry["residual_linf"] = res_linf;
    }
    if (std::isfinite(solve_time)) {
      entry["solve_seconds"] = solve_time;
    }
    if (monitor_warning) {
      entry["monitor_warning"] = true;
    }
    runs.push_back(entry);

    out->runs_total++;
    if (output_exists) {
      out->runs_completed++;
      out->total_frames += frame_count;
      if (time_series) {
        out->runs_time_series++;
      } else {
        out->runs_steady++;
      }
      residual_l2.Add(res_l2);
      residual_linf.Add(res_linf);
      solve_seconds.Add(solve_time);
      if (monitor_warning) {
        out->monitor_warning_runs++;
      }
    } else {
      out->missing_outputs++;
    }
  }

  json stats;
  stats["runs_total"] = out->runs_total;
  stats["runs_completed"] = out->runs_completed;
  stats["runs_time_series"] = out->runs_time_series;
  stats["runs_steady"] = out->runs_steady;
  stats["total_frames"] = out->total_frames;
  stats["missing_outputs"] = out->missing_outputs;
  stats["monitor_warning_runs"] = out->monitor_warning_runs;
  if (residual_l2.count > 0) {
    stats["residual_l2"] = residual_l2.ToJson();
  }
  if (residual_linf.count > 0) {
    stats["residual_linf"] = residual_linf.ToJson();
  }
  if (solve_seconds.count > 0) {
    stats["solve_seconds"] = solve_seconds.ToJson();
  }
  if (skipped > 0) {
    stats["skipped_summaries"] = skipped;
  }

  json root_json;
  root_json["schema_version"] = 1;
  root_json["generated_at"] = TimestampUtc();
  root_json["root"] = root.string();
  root_json["stats"] = stats;
  root_json["runs"] = runs;
  out->json = root_json.dump(2);
  return true;
}

bool WriteDatasetIndex(const std::filesystem::path& path,
                       const DatasetIndexResult& result,
                       std::string* error) {
  if (result.json.empty()) {
    if (error) {
      *error = "dataset index is empty";
    }
    return false;
  }
  return WriteFile(path, result.json, error);
}

bool CleanupDataset(const std::filesystem::path& root,
                    bool dry_run,
                    bool remove_empty_dirs,
                    DatasetCleanupResult* out,
                    std::string* error) {
  if (!out) {
    if (error) {
      *error = "missing cleanup output";
    }
    return false;
  }
  *out = {};
  if (!std::filesystem::exists(root) || !std::filesystem::is_directory(root)) {
    if (error) {
      *error = "dataset root not found: " + root.string();
    }
    return false;
  }

  const std::vector<std::filesystem::path> summaries = CollectFiles(root, kSummarySuffix);
  for (const auto& summary_path : summaries) {
    std::filesystem::path output_path;
    ResolveOutputFromSidecar(root, summary_path, &output_path, nullptr);
    if (!output_path.empty() && std::filesystem::exists(output_path)) {
      continue;
    }
    if (!dry_run) {
      std::error_code ec;
      if (!std::filesystem::remove(summary_path, ec)) {
        out->skipped++;
        continue;
      }
    }
    out->removed_summaries++;
  }

  const std::vector<std::filesystem::path> metadata = CollectFiles(root, kMetaSuffix);
  for (const auto& meta_path : metadata) {
    std::filesystem::path output_path;
    ResolveOutputFromSidecar(root, meta_path, &output_path, nullptr);
    if (!output_path.empty() && std::filesystem::exists(output_path)) {
      continue;
    }
    if (!dry_run) {
      std::error_code ec;
      if (!std::filesystem::remove(meta_path, ec)) {
        out->skipped++;
        continue;
      }
    }
    out->removed_metadata++;
  }

  if (remove_empty_dirs) {
    std::vector<std::filesystem::path> dirs;
    std::error_code ec;
    std::filesystem::recursive_directory_iterator it(
        root, std::filesystem::directory_options::skip_permission_denied, ec);
    std::filesystem::recursive_directory_iterator end;
    for (; it != end; it.increment(ec)) {
      if (ec) {
        ec.clear();
        continue;
      }
      if (it->is_directory(ec)) {
        dirs.push_back(it->path());
      }
    }
    std::sort(dirs.begin(), dirs.end(),
              [](const std::filesystem::path& a, const std::filesystem::path& b) {
                return a.string().size() > b.string().size();
              });
    for (const auto& dir : dirs) {
      if (dir == root) {
        continue;
      }
      std::error_code empty_ec;
      if (std::filesystem::is_empty(dir, empty_ec) && !empty_ec) {
        if (!dry_run) {
          std::error_code rm_ec;
          if (!std::filesystem::remove(dir, rm_ec)) {
            out->skipped++;
            continue;
          }
        }
        out->removed_empty_dirs++;
      }
    }
  }

  return true;
}
