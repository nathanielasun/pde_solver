#ifndef PATH_UTILS_H
#define PATH_UTILS_H

#include <filesystem>
#include <string>

// Path resolution functions
std::filesystem::path FindScriptPath(const std::filesystem::path& exec_path);
std::filesystem::path FindUIFontDir(const std::filesystem::path& exec_path);
std::string ResolvePythonPath(const std::filesystem::path& project_root);
std::filesystem::path EnsureLatexCacheDir(const std::filesystem::path& base);
std::filesystem::path ResolvePrefsPath(const std::filesystem::path& exec_path);

#endif // PATH_UTILS_H
