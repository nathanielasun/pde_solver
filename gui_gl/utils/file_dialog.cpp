#include "file_dialog.h"
#include <filesystem>

#ifdef __APPLE__
#include <Cocoa/Cocoa.h>
#include <AppKit/AppKit.h>

namespace FileDialog {

std::optional<std::filesystem::path> PickDirectory(
    const std::string& title,
    const std::filesystem::path& default_path) {
  @autoreleasepool {
    NSOpenPanel* panel = [NSOpenPanel openPanel];
    
    [panel setTitle:[NSString stringWithUTF8String:title.c_str()]];
    [panel setCanChooseFiles:NO];
    [panel setCanChooseDirectories:YES];
    [panel setAllowsMultipleSelection:NO];
    [panel setResolvesAliases:YES];
    
    // Set default directory if provided
    if (!default_path.empty() && std::filesystem::exists(default_path)) {
      NSString* defaultDir = [NSString stringWithUTF8String:default_path.string().c_str()];
      [panel setDirectoryURL:[NSURL fileURLWithPath:defaultDir]];
    }
    
    NSInteger result = [panel runModal];
    
    if (result == NSModalResponseOK) {
      NSURL* selectedURL = [[panel URLs] firstObject];
      if (selectedURL) {
        NSString* path = [selectedURL path];
        return std::filesystem::path([path UTF8String]);
      }
    }
    
    return std::nullopt;
  }
}

std::optional<std::filesystem::path> PickFile(
    const std::string& title,
    const std::filesystem::path& default_path,
    const std::string& filter_description,
    const std::vector<std::string>& filter_extensions) {
  @autoreleasepool {
    NSOpenPanel* panel = [NSOpenPanel openPanel];
    
    [panel setTitle:[NSString stringWithUTF8String:title.c_str()]];
    [panel setCanChooseFiles:YES];
    [panel setCanChooseDirectories:NO];
    [panel setAllowsMultipleSelection:NO];
    [panel setResolvesAliases:YES];
    
    // Set file type filters if provided
    if (!filter_extensions.empty()) {
      NSMutableArray* allowedTypes = [NSMutableArray array];
      for (const auto& ext : filter_extensions) {
        // Remove leading dot if present
        std::string clean_ext = ext;
        if (!clean_ext.empty() && clean_ext[0] == '.') {
          clean_ext = clean_ext.substr(1);
        }
        [allowedTypes addObject:[NSString stringWithUTF8String:clean_ext.c_str()]];
      }
      [panel setAllowedFileTypes:allowedTypes];
    }
    
    // Set default directory if provided
    if (!default_path.empty()) {
      std::filesystem::path dir = default_path;
      if (std::filesystem::is_regular_file(dir)) {
        dir = dir.parent_path();
      }
      if (std::filesystem::exists(dir)) {
        NSString* defaultDir = [NSString stringWithUTF8String:dir.string().c_str()];
        [panel setDirectoryURL:[NSURL fileURLWithPath:defaultDir]];
      }
    }
    
    NSInteger result = [panel runModal];
    
    if (result == NSModalResponseOK) {
      NSURL* selectedURL = [[panel URLs] firstObject];
      if (selectedURL) {
        NSString* path = [selectedURL path];
        return std::filesystem::path([path UTF8String]);
      }
    }
    
    return std::nullopt;
  }
}

std::optional<std::filesystem::path> SaveFile(
    const std::string& title,
    const std::filesystem::path& default_path,
    const std::string& default_filename,
    const std::string& filter_description,
    const std::vector<std::string>& filter_extensions) {
  @autoreleasepool {
    NSSavePanel* panel = [NSSavePanel savePanel];
    
    [panel setTitle:[NSString stringWithUTF8String:title.c_str()]];
    [panel setCanCreateDirectories:YES];
    [panel setExtensionHidden:NO];
    
    // Set file type filters if provided
    if (!filter_extensions.empty()) {
      NSMutableArray* allowedTypes = [NSMutableArray array];
      for (const auto& ext : filter_extensions) {
        // Remove leading dot if present
        std::string clean_ext = ext;
        if (!clean_ext.empty() && clean_ext[0] == '.') {
          clean_ext = clean_ext.substr(1);
        }
        [allowedTypes addObject:[NSString stringWithUTF8String:clean_ext.c_str()]];
      }
      [panel setAllowedFileTypes:allowedTypes];
    }
    
    // Set default directory and filename
    std::filesystem::path dir = default_path;
    if (!default_path.empty() && std::filesystem::is_regular_file(default_path)) {
      dir = default_path.parent_path();
    } else if (default_path.empty() && !default_filename.empty()) {
      dir = std::filesystem::current_path();
    }
    
    if (std::filesystem::exists(dir)) {
      NSString* defaultDir = [NSString stringWithUTF8String:dir.string().c_str()];
      [panel setDirectoryURL:[NSURL fileURLWithPath:defaultDir]];
    }
    
    if (!default_filename.empty()) {
      [panel setNameFieldStringValue:[NSString stringWithUTF8String:default_filename.c_str()]];
    }
    
    NSInteger result = [panel runModal];
    
    if (result == NSModalResponseOK) {
      NSURL* selectedURL = [panel URL];
      if (selectedURL) {
        NSString* path = [selectedURL path];
        return std::filesystem::path([path UTF8String]);
      }
    }
    
    return std::nullopt;
  }
}

} // namespace FileDialog

#elif defined(_WIN32)

#include <windows.h>
#include <commdlg.h>
#include <shlobj.h>

namespace FileDialog {

static std::wstring toWide(const std::string& str) {
  if (str.empty()) return {};
  int sz = MultiByteToWideChar(CP_UTF8, 0, str.data(), (int)str.size(), nullptr, 0);
  std::wstring wide(sz, L'\0');
  MultiByteToWideChar(CP_UTF8, 0, str.data(), (int)str.size(), wide.data(), sz);
  return wide;
}

static std::wstring buildFilterString(const std::string& description,
                                      const std::vector<std::string>& extensions) {
  if (extensions.empty()) {
    std::wstring filter = toWide("All Files");
    filter.push_back(L'\0');
    filter += L"*.*";
    filter.push_back(L'\0');
    return filter;
  }

  std::wstring filter = toWide(description);
  filter.push_back(L'\0');

  std::wstring exts;
  for (size_t i = 0; i < extensions.size(); ++i) {
    if (i > 0) exts += L";";
    std::string ext = extensions[i];
    if (!ext.empty() && ext[0] != '.') ext = "." + ext;
    exts += L"*" + toWide(ext);
  }
  filter += exts;
  filter.push_back(L'\0');
  return filter;
}

std::optional<std::filesystem::path> PickDirectory(
    const std::string& title,
    const std::filesystem::path& default_path) {
  CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED);

  std::wstring wTitle = toWide(title);
  BROWSEINFOW bi = {};
  bi.lpszTitle = wTitle.c_str();
  bi.ulFlags = BIF_RETURNONLYFSDIRS | BIF_NEWDIALOGSTYLE;

  LPITEMIDLIST pidl = SHBrowseForFolderW(&bi);
  if (pidl) {
    wchar_t buf[MAX_PATH];
    if (SHGetPathFromIDListW(pidl, buf)) {
      CoTaskMemFree(pidl);
      CoUninitialize();
      return std::filesystem::path(buf);
    }
    CoTaskMemFree(pidl);
  }

  CoUninitialize();
  return std::nullopt;
}

std::optional<std::filesystem::path> PickFile(
    const std::string& title,
    const std::filesystem::path& default_path,
    const std::string& filter_description,
    const std::vector<std::string>& filter_extensions) {
  std::wstring filter = buildFilterString(filter_description, filter_extensions);
  wchar_t fileBuf[MAX_PATH] = {};

  std::wstring initialDir;
  if (!default_path.empty()) {
    std::filesystem::path dir = default_path;
    if (std::filesystem::is_regular_file(dir)) dir = dir.parent_path();
    if (std::filesystem::exists(dir)) initialDir = dir.wstring();
  }

  std::wstring wTitle = toWide(title);

  OPENFILENAMEW ofn = {};
  ofn.lStructSize = sizeof(ofn);
  ofn.lpstrFilter = filter.c_str();
  ofn.lpstrFile = fileBuf;
  ofn.nMaxFile = MAX_PATH;
  ofn.lpstrTitle = wTitle.c_str();
  ofn.Flags = OFN_FILEMUSTEXIST | OFN_PATHMUSTEXIST | OFN_NOCHANGEDIR;
  if (!initialDir.empty()) ofn.lpstrInitialDir = initialDir.c_str();

  if (GetOpenFileNameW(&ofn)) {
    return std::filesystem::path(fileBuf);
  }
  return std::nullopt;
}

std::optional<std::filesystem::path> SaveFile(
    const std::string& title,
    const std::filesystem::path& default_path,
    const std::string& default_filename,
    const std::string& filter_description,
    const std::vector<std::string>& filter_extensions) {
  std::wstring filter = buildFilterString(filter_description, filter_extensions);
  wchar_t fileBuf[MAX_PATH] = {};

  if (!default_filename.empty()) {
    std::wstring wName = toWide(default_filename);
    wcsncpy(fileBuf, wName.c_str(), MAX_PATH - 1);
  }

  std::wstring initialDir;
  if (!default_path.empty()) {
    std::filesystem::path dir = default_path;
    if (std::filesystem::is_regular_file(dir)) dir = dir.parent_path();
    if (std::filesystem::exists(dir)) initialDir = dir.wstring();
  }

  std::wstring wTitle = toWide(title);

  OPENFILENAMEW ofn = {};
  ofn.lStructSize = sizeof(ofn);
  ofn.lpstrFilter = filter.c_str();
  ofn.lpstrFile = fileBuf;
  ofn.nMaxFile = MAX_PATH;
  ofn.lpstrTitle = wTitle.c_str();
  ofn.Flags = OFN_OVERWRITEPROMPT | OFN_NOCHANGEDIR;
  if (!initialDir.empty()) ofn.lpstrInitialDir = initialDir.c_str();

  std::wstring defExt;
  if (!filter_extensions.empty()) {
    std::string ext = filter_extensions[0];
    if (!ext.empty() && ext[0] == '.') ext = ext.substr(1);
    defExt = toWide(ext);
    ofn.lpstrDefExt = defExt.c_str();
  }

  if (GetSaveFileNameW(&ofn)) {
    return std::filesystem::path(fileBuf);
  }
  return std::nullopt;
}

} // namespace FileDialog

#elif defined(__linux__)

#include <cstdio>
#include <cstdlib>
#include <array>
#include <algorithm>

namespace FileDialog {

static std::string shellEscape(const std::string& str) {
  std::string escaped = "'";
  for (char c : str) {
    if (c == '\'') {
      escaped += "'\\''";
    } else {
      escaped += c;
    }
  }
  escaped += "'";
  return escaped;
}

static bool commandExists(const std::string& cmd) {
  std::string check = "which " + cmd + " >/dev/null 2>&1";
  return system(check.c_str()) == 0;
}

static std::optional<std::string> runCommand(const std::string& cmd) {
  std::array<char, 4096> buffer;
  std::string result;
  FILE* pipe = popen(cmd.c_str(), "r");
  if (!pipe) return std::nullopt;

  while (fgets(buffer.data(), buffer.size(), pipe) != nullptr) {
    result += buffer.data();
  }

  int status = pclose(pipe);
  if (status != 0) return std::nullopt;

  while (!result.empty() && (result.back() == '\n' || result.back() == '\r')) {
    result.pop_back();
  }
  if (result.empty()) return std::nullopt;
  return result;
}

std::optional<std::filesystem::path> PickDirectory(
    const std::string& title,
    const std::filesystem::path& default_path) {
  if (commandExists("zenity")) {
    std::string cmd = "zenity --file-selection --directory --title=" + shellEscape(title);
    if (!default_path.empty() && std::filesystem::exists(default_path)) {
      cmd += " --filename=" + shellEscape(default_path.string() + "/");
    }
    auto result = runCommand(cmd);
    if (result) return std::filesystem::path(*result);
  }

  if (commandExists("kdialog")) {
    std::string cmd = "kdialog --getexistingdirectory";
    if (!default_path.empty() && std::filesystem::exists(default_path)) {
      cmd += " " + shellEscape(default_path.string());
    } else {
      cmd += " .";
    }
    cmd += " --title " + shellEscape(title);
    auto result = runCommand(cmd);
    if (result) return std::filesystem::path(*result);
  }

  return std::nullopt;
}

std::optional<std::filesystem::path> PickFile(
    const std::string& title,
    const std::filesystem::path& default_path,
    const std::string& filter_description,
    const std::vector<std::string>& filter_extensions) {
  if (commandExists("zenity")) {
    std::string cmd = "zenity --file-selection --title=" + shellEscape(title);
    if (!default_path.empty()) {
      std::filesystem::path dir = default_path;
      if (std::filesystem::is_regular_file(dir)) dir = dir.parent_path();
      if (std::filesystem::exists(dir)) {
        cmd += " --filename=" + shellEscape(dir.string() + "/");
      }
    }
    if (!filter_extensions.empty()) {
      std::string filterStr = filter_description + " |";
      for (const auto& ext : filter_extensions) {
        std::string e = ext;
        if (!e.empty() && e[0] != '.') e = "." + e;
        filterStr += " *" + e;
      }
      cmd += " --file-filter=" + shellEscape(filterStr);
    }
    auto result = runCommand(cmd);
    if (result) return std::filesystem::path(*result);
  }

  if (commandExists("kdialog")) {
    std::string cmd = "kdialog --getopenfilename";
    if (!default_path.empty()) {
      std::filesystem::path dir = default_path;
      if (std::filesystem::is_regular_file(dir)) dir = dir.parent_path();
      if (std::filesystem::exists(dir)) {
        cmd += " " + shellEscape(dir.string());
      } else {
        cmd += " .";
      }
    } else {
      cmd += " .";
    }
    if (!filter_extensions.empty()) {
      std::string filterStr;
      for (const auto& ext : filter_extensions) {
        std::string e = ext;
        if (!e.empty() && e[0] != '.') e = "." + e;
        if (!filterStr.empty()) filterStr += " ";
        filterStr += "*" + e;
      }
      filterStr += " | " + filter_description;
      cmd += " " + shellEscape(filterStr);
    }
    cmd += " --title " + shellEscape(title);
    auto result = runCommand(cmd);
    if (result) return std::filesystem::path(*result);
  }

  return std::nullopt;
}

std::optional<std::filesystem::path> SaveFile(
    const std::string& title,
    const std::filesystem::path& default_path,
    const std::string& default_filename,
    const std::string& filter_description,
    const std::vector<std::string>& filter_extensions) {
  if (commandExists("zenity")) {
    std::string cmd = "zenity --file-selection --save --confirm-overwrite --title=" + shellEscape(title);
    std::filesystem::path initialPath;
    if (!default_path.empty()) {
      std::filesystem::path dir = default_path;
      if (std::filesystem::is_regular_file(dir)) dir = dir.parent_path();
      initialPath = dir;
    }
    if (!default_filename.empty()) {
      if (initialPath.empty()) initialPath = std::filesystem::current_path();
      cmd += " --filename=" + shellEscape((initialPath / default_filename).string());
    } else if (!initialPath.empty()) {
      cmd += " --filename=" + shellEscape(initialPath.string() + "/");
    }
    if (!filter_extensions.empty()) {
      std::string filterStr = filter_description + " |";
      for (const auto& ext : filter_extensions) {
        std::string e = ext;
        if (!e.empty() && e[0] != '.') e = "." + e;
        filterStr += " *" + e;
      }
      cmd += " --file-filter=" + shellEscape(filterStr);
    }
    auto result = runCommand(cmd);
    if (result) return std::filesystem::path(*result);
  }

  if (commandExists("kdialog")) {
    std::string cmd = "kdialog --getsavefilename";
    std::filesystem::path initialPath;
    if (!default_path.empty()) {
      std::filesystem::path dir = default_path;
      if (std::filesystem::is_regular_file(dir)) dir = dir.parent_path();
      initialPath = dir;
    }
    if (!default_filename.empty()) {
      if (initialPath.empty()) initialPath = std::filesystem::current_path();
      cmd += " " + shellEscape((initialPath / default_filename).string());
    } else if (!initialPath.empty()) {
      cmd += " " + shellEscape(initialPath.string());
    } else {
      cmd += " .";
    }
    if (!filter_extensions.empty()) {
      std::string filterStr;
      for (const auto& ext : filter_extensions) {
        std::string e = ext;
        if (!e.empty() && e[0] != '.') e = "." + e;
        if (!filterStr.empty()) filterStr += " ";
        filterStr += "*" + e;
      }
      filterStr += " | " + filter_description;
      cmd += " " + shellEscape(filterStr);
    }
    cmd += " --title " + shellEscape(title);
    auto result = runCommand(cmd);
    if (result) return std::filesystem::path(*result);
  }

  return std::nullopt;
}

} // namespace FileDialog

#else

namespace FileDialog {

std::optional<std::filesystem::path> PickDirectory(
    const std::string& title,
    const std::filesystem::path& default_path) {
  return std::nullopt;
}

std::optional<std::filesystem::path> PickFile(
    const std::string& title,
    const std::filesystem::path& default_path,
    const std::string& filter_description,
    const std::vector<std::string>& filter_extensions) {
  return std::nullopt;
}

std::optional<std::filesystem::path> SaveFile(
    const std::string& title,
    const std::filesystem::path& default_path,
    const std::string& default_filename,
    const std::string& filter_description,
    const std::vector<std::string>& filter_extensions) {
  return std::nullopt;
}

} // namespace FileDialog

#endif // __APPLE__

