#include "file_utils.h"

#include "../../include/vtk_io.h"
#include <filesystem>
#include <iomanip>
#include <optional>
#include <sstream>
#include <string>

std::optional<std::filesystem::path> FindLatestVtk(const std::filesystem::path& dir) {
  if (!std::filesystem::exists(dir) || !std::filesystem::is_directory(dir)) {
    return std::nullopt;
  }
  std::optional<std::filesystem::path> latest;
  std::filesystem::file_time_type latest_time;
  for (const auto& entry : std::filesystem::directory_iterator(dir)) {
    if (!entry.is_regular_file()) {
      continue;
    }
    if (entry.path().extension() != ".vtk") {
      continue;
    }
    const auto time = entry.last_write_time();
    if (!latest || time > latest_time) {
      latest = entry.path();
      latest_time = time;
    }
  }
  return latest;
}

int FrameDigits(int frames) {
  int max_frame = std::max(0, frames - 1);
  int digits = 1;
  while (max_frame >= 10) {
    max_frame /= 10;
    ++digits;
  }
  return std::max(4, digits);
}

std::string PadFrameIndex(int frame, int digits) {
  std::ostringstream out;
  out << std::setw(digits) << std::setfill('0') << frame;
  return out.str();
}

std::filesystem::path BuildFramePath(const std::filesystem::path& base_path, int frame,
                                     int digits) {
  const std::string stem = base_path.stem().string();
  const std::string ext = base_path.extension().string();
  const std::string name = stem + "_t" + PadFrameIndex(frame, digits) + ext;
  return base_path.parent_path() / name;
}

std::filesystem::path ResolveOutputPath(const std::string& output_text, std::string* warning) {
  std::filesystem::path output_path;
  std::filesystem::path out_dir;
  if (output_text.empty()) {
    out_dir = "outputs";
  } else {
    std::filesystem::path candidate(output_text);
    if (std::filesystem::exists(candidate) && std::filesystem::is_directory(candidate)) {
      out_dir = candidate;
    } else if (!output_text.empty() && output_text.back() == '/') {
      out_dir = candidate;
    } else {
      if (!candidate.has_extension()) {
        candidate += ".vtk";
      } else if (candidate.extension() != ".vtk") {
        candidate.replace_extension(".vtk");
        if (warning) {
          *warning = "output extension adjusted to .vtk";
        }
      }
      output_path = candidate;
    }
  }

  std::error_code ec;
  if (!out_dir.empty()) {
    std::filesystem::create_directories(out_dir, ec);
    if (ec) {
      if (warning) {
        *warning = "failed to create output directory: " + ec.message();
      }
    } else {
      const std::string name = "solution_" + GenerateRandomTag(6) + ".vtk";
      output_path = out_dir / name;
    }
  } else if (output_path.has_parent_path()) {
    std::filesystem::create_directories(output_path.parent_path(), ec);
    if (ec && warning) {
      *warning = "failed to create output directory: " + ec.message();
    }
  }

  return output_path;
}

