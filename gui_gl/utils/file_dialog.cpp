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

#else
// Placeholder implementations for non-macOS platforms
// TODO: Implement for Windows (GetOpenFileName) and Linux (GTK or similar)

namespace FileDialog {

std::optional<std::filesystem::path> PickDirectory(
    const std::string& title,
    const std::filesystem::path& default_path) {
  // TODO: Implement for Windows/Linux
  return std::nullopt;
}

std::optional<std::filesystem::path> PickFile(
    const std::string& title,
    const std::filesystem::path& default_path,
    const std::string& filter_description,
    const std::vector<std::string>& filter_extensions) {
  // TODO: Implement for Windows/Linux
  return std::nullopt;
}

std::optional<std::filesystem::path> SaveFile(
    const std::string& title,
    const std::filesystem::path& default_path,
    const std::string& default_filename,
    const std::string& filter_description,
    const std::vector<std::string>& filter_extensions) {
  // TODO: Implement for Windows/Linux
  return std::nullopt;
}

} // namespace FileDialog

#endif // __APPLE__

