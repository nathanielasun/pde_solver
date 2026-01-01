#ifndef PREFERENCES_IO_H
#define PREFERENCES_IO_H

#include <filesystem>
#include <string>

// Forward declarations
struct Preferences;

// Preferences file I/O
bool LoadPreferences(const std::filesystem::path& path, Preferences* prefs, std::string* error);
bool SavePreferences(const std::filesystem::path& path, const Preferences& prefs, std::string* error);

#endif // PREFERENCES_IO_H

