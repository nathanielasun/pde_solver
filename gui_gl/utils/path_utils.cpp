#include "path_utils.h"

#include <cstdlib>
#include <filesystem>
#include <vector>

std::filesystem::path FindScriptPath(const std::filesystem::path& exec_path) {
  std::vector<std::filesystem::path> roots;
  roots.push_back(std::filesystem::current_path());
  if (!exec_path.empty()) {
    roots.push_back(exec_path.parent_path());
  }

  for (const auto& base : roots) {
    std::filesystem::path cur = base;
    for (int i = 0; i < 6; ++i) {
      std::filesystem::path candidate = cur / "tools" / "render_latex.py";
      if (std::filesystem::exists(candidate)) {
        return candidate;
      }
      if (!cur.has_parent_path()) {
        break;
      }
      cur = cur.parent_path();
    }
  }
  return {};
}

std::filesystem::path FindUIFontDir(const std::filesystem::path& exec_path) {
  std::vector<std::filesystem::path> roots;
  roots.push_back(std::filesystem::current_path());
  if (!exec_path.empty()) {
    roots.push_back(exec_path.parent_path());
  }

  for (const auto& base : roots) {
    std::filesystem::path cur = base;
    for (int i = 0; i < 6; ++i) {
      std::filesystem::path candidate = cur / "gui_gl" / "styles" / "fonts";
      if (std::filesystem::exists(candidate)) {
        return candidate;
      }
      if (!cur.has_parent_path()) {
        break;
      }
      cur = cur.parent_path();
    }
  }
  return {};
}

std::string ResolvePythonPath(const std::filesystem::path& project_root) {
  const char* conda_prefix = std::getenv("CONDA_PREFIX");
  if (conda_prefix && *conda_prefix) {
    std::filesystem::path python_path =
        std::filesystem::path(conda_prefix) / "bin" / "python";
    return python_path.string();
  }
  if (!project_root.empty()) {
    std::filesystem::path candidate = project_root / "bin" / "python";
    if (std::filesystem::exists(candidate)) {
      return candidate.string();
    }
    candidate = project_root.parent_path() / "bin" / "python";
    if (std::filesystem::exists(candidate)) {
      return candidate.string();
    }
  }
  return "python3";
}

std::filesystem::path EnsureLatexCacheDir(const std::filesystem::path& base) {
  std::filesystem::path dir = base / "outputs" / "latex_cache";
  std::error_code ec;
  std::filesystem::create_directories(dir, ec);
  return dir;
}

std::filesystem::path ResolvePrefsPath(const std::filesystem::path& exec_path) {
  if (!exec_path.empty()) {
    return exec_path.parent_path() / "pde_gui_prefs.ini";
  }
  return std::filesystem::current_path() / "pde_gui_prefs.ini";
}
