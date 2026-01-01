#ifndef FILE_DIALOG_H
#define FILE_DIALOG_H

#include <string>
#include <optional>
#include <filesystem>
#include <vector>

namespace FileDialog {

// Open a directory picker dialog
// Returns the selected directory path, or empty optional if cancelled
std::optional<std::filesystem::path> PickDirectory(
    const std::string& title = "Select Directory",
    const std::filesystem::path& default_path = {});

// Open a file picker dialog for selecting a single file
// Returns the selected file path, or empty optional if cancelled
std::optional<std::filesystem::path> PickFile(
    const std::string& title = "Select File",
    const std::filesystem::path& default_path = {},
    const std::string& filter_description = "All Files",
    const std::vector<std::string>& filter_extensions = {});

// Open a file picker dialog for saving a file
// Returns the selected file path, or empty optional if cancelled
std::optional<std::filesystem::path> SaveFile(
    const std::string& title = "Save File",
    const std::filesystem::path& default_path = {},
    const std::string& default_filename = {},
    const std::string& filter_description = "All Files",
    const std::vector<std::string>& filter_extensions = {});

} // namespace FileDialog

#endif // FILE_DIALOG_H

