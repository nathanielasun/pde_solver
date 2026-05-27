#include "latex/preview_font_paths.h"

#include <filesystem>
#include <mutex>
#include <vector>

namespace {

bool FileExists(const std::string& path) {
  return !path.empty() && std::filesystem::exists(path);
}

const std::vector<std::string>& SansBoldCandidates() {
  static const std::vector<std::string> kCandidates = {
      "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
      "/System/Library/Fonts/Supplemental/Comic Sans MS Bold.ttf",
      "/Library/Fonts/Arial Bold.ttf",
      "/Library/Fonts/Comic Sans MS Bold.ttf",
      "C:/Windows/Fonts/arialbd.ttf",
      "C:/Windows/Fonts/comicbd.ttf",
      "/usr/share/fonts/truetype/msttcorefonts/Arial_Bold.ttf",
      "/usr/share/fonts/truetype/msttcorefonts/arialbd.ttf",
      "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
      "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
  };
  return kCandidates;
}

}  // namespace

std::string FindPreviewSansBoldFont() {
  static std::mutex mutex;
  static std::string cached;
  static bool resolved = false;
  std::lock_guard<std::mutex> lock(mutex);
  if (resolved) {
    return cached;
  }
  resolved = true;
  for (const std::string& path : SansBoldCandidates()) {
    if (FileExists(path)) {
      cached = path;
      break;
    }
  }
  return cached;
}
