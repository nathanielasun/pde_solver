#include "run_metadata.h"

#include <chrono>
#include <cstddef>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <thread>

#include <nlohmann/json.hpp>

#include "solver_tokens.h"

#if defined(__APPLE__) || defined(__linux__) || defined(__unix__)
#include <sys/utsname.h>
#endif

#ifndef PDE_GIT_SHA
#define PDE_GIT_SHA "unknown"
#endif
#ifndef PDE_BUILD_TIMESTAMP
#define PDE_BUILD_TIMESTAMP __DATE__ " " __TIME__
#endif
#ifndef PDE_BUILD_TYPE
#define PDE_BUILD_TYPE "unknown"
#endif

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

std::string CompilerString() {
#if defined(__clang__)
  return std::string("clang ") + __clang_version__;
#elif defined(__GNUC__)
  std::ostringstream oss;
  oss << "gcc " << __GNUC__ << "." << __GNUC_MINOR__ << "." << __GNUC_PATCHLEVEL__;
  return oss.str();
#elif defined(_MSC_VER)
  std::ostringstream oss;
  oss << "msvc " << _MSC_VER;
  return oss.str();
#else
  return "unknown";
#endif
}

std::string ArchString() {
#if defined(__aarch64__) || defined(__arm64__)
  return "arm64";
#elif defined(__x86_64__) || defined(_M_X64)
  return "x86_64";
#elif defined(__i386__) || defined(_M_IX86)
  return "x86";
#elif defined(__arm__) || defined(_M_ARM)
  return "arm";
#else
  return "unknown";
#endif
}

json BuildBackendArray() {
  json arr = json::array();
  const std::vector<BackendStatus> statuses = DetectBackends();
  for (const auto& status : statuses) {
    json entry;
    entry["kind"] = BackendToken(status.kind);
    entry["name"] = status.name;
    entry["available"] = status.available;
    if (!status.note.empty()) {
      entry["note"] = status.note;
    }
    arr.push_back(entry);
  }
  return arr;
}

const json& CachedBackendArray() {
  static const json backends = BuildBackendArray();
  return backends;
}

json BuildPlatformJson() {
  json platform;
  platform["os"] = "unknown";
  platform["arch"] = ArchString();

#if defined(__APPLE__) || defined(__linux__) || defined(__unix__)
  struct utsname info;
  if (uname(&info) == 0) {
    platform["os"] = info.sysname;
    platform["os_release"] = info.release;
    platform["os_version"] = info.version;
    platform["arch"] = info.machine;
  }
#elif defined(_WIN32)
  platform["os"] = "Windows";
#endif

  const unsigned int threads = std::thread::hardware_concurrency();
  if (threads > 0) {
    platform["cpu_threads"] = threads;
  }
  platform["backends"] = CachedBackendArray();
  return platform;
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
}  // namespace

std::filesystem::path MetadataSidecarPath(const std::filesystem::path& output_path) {
  if (output_path.empty()) {
    return {};
  }
  const std::string name = output_path.filename().string();
  if (EndsWith(name, ".meta.json")) {
    return output_path;
  }
  return std::filesystem::path(output_path.string() + ".meta.json");
}

std::string BuildRunMetadataJson(const RunConfig& run_config,
                                 BackendKind requested_backend,
                                 BackendKind selected_backend,
                                 const std::string& backend_note,
                                 const std::string& output_path,
                                 bool time_series,
                                 int frame_index,
                                 double frame_time) {
  json root;
  root["schema_version"] = 1;
  root["generated_at"] = TimestampUtc();
  std::string git_sha = PDE_GIT_SHA;
  if (git_sha.empty()) {
    git_sha = "unknown";
  }
  root["git_sha"] = git_sha;

  json build;
  std::string build_type = PDE_BUILD_TYPE;
  if (build_type.empty()) {
    build_type = "unknown";
  }
  build["type"] = build_type;
  build["timestamp"] = PDE_BUILD_TIMESTAMP;
  build["compiler"] = CompilerString();
  build["cxx_standard"] = static_cast<int>(__cplusplus);
  root["build"] = build;

  root["platform"] = BuildPlatformJson();

  std::string run_spec = SerializeRunConfig(run_config, 0);
  try {
    root["run"] = json::parse(run_spec);
  } catch (const json::exception&) {
    root["run"] = run_spec;
  }

  json solve;
  solve["requested_backend"] = BackendToken(requested_backend);
  solve["selected_backend"] = BackendToken(selected_backend);
  solve["time_series"] = time_series;
  if (!backend_note.empty()) {
    solve["backend_note"] = backend_note;
  }
  if (frame_index >= 0) {
    solve["frame_index"] = frame_index;
    solve["frame_time"] = frame_time;
  }
  root["solve"] = solve;

  json output;
  output["path"] = output_path;
  root["output"] = output;

  return root.dump(2);
}

bool WriteRunMetadataSidecar(const std::filesystem::path& output_path,
                             const std::string& metadata_json,
                             std::string* error) {
  if (output_path.empty()) {
    if (error) {
      *error = "missing output path for metadata";
    }
    return false;
  }
  const std::filesystem::path meta_path = MetadataSidecarPath(output_path);
  if (meta_path.empty()) {
    if (error) {
      *error = "invalid metadata path";
    }
    return false;
  }
  if (meta_path.has_parent_path()) {
    std::error_code ec;
    std::filesystem::create_directories(meta_path.parent_path(), ec);
    if (ec) {
      if (error) {
        *error = "failed to create metadata directory: " + ec.message();
      }
      return false;
    }
  }
  return WriteFile(meta_path, metadata_json, error);
}

bool WriteRunMetadataSidecar(const std::filesystem::path& output_path,
                             const RunConfig& run_config,
                             BackendKind requested_backend,
                             BackendKind selected_backend,
                             const std::string& backend_note,
                             bool time_series,
                             int frame_index,
                             double frame_time,
                             std::string* error) {
  const std::string metadata_json =
      BuildRunMetadataJson(run_config, requested_backend, selected_backend, backend_note,
                           output_path.string(), time_series, frame_index, frame_time);
  return WriteRunMetadataSidecar(output_path, metadata_json, error);
}

bool ReadRunMetadataSidecar(const std::filesystem::path& output_path,
                            std::string* metadata_json,
                            std::string* error) {
  const std::filesystem::path meta_path = MetadataSidecarPath(output_path);
  if (meta_path.empty()) {
    if (error) {
      *error = "invalid metadata path";
    }
    return false;
  }
  if (!std::filesystem::exists(meta_path)) {
    if (error) {
      *error = "metadata sidecar not found: " + meta_path.string();
    }
    return false;
  }
  return ReadFile(meta_path, metadata_json, error);
}
