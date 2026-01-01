#ifndef FILE_UTILS_H
#define FILE_UTILS_H

#include <filesystem>
#include <optional>
#include <string>

// File utility functions
std::optional<std::filesystem::path> FindLatestVtk(const std::filesystem::path& dir);
int FrameDigits(int frames);
std::string PadFrameIndex(int frame, int digits);
std::filesystem::path BuildFramePath(const std::filesystem::path& base_path, int frame, int digits);
std::filesystem::path ResolveOutputPath(const std::string& output_text, std::string* warning);

#endif // FILE_UTILS_H

