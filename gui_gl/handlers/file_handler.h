#ifndef FILE_HANDLER_H
#define FILE_HANDLER_H

#include "app_state.h"
#include "GlViewer.h"
#include "vtk_io.h"
#include <string>
#include <filesystem>
#include <mutex>
#include <functional>
#include <vector>

// State structure for file handler
struct FileHandlerState {
  // Input/output paths
  std::string& input_dir;
  std::filesystem::path& prefs_path;
  
  // Shared state
  SharedState& state;
  std::mutex& state_mutex;
  
  // Viewer
  GlViewer& viewer;
  
  // Frame management (for time series)
  std::vector<std::string>& frame_paths;
  std::vector<double>& frame_times;
  int& frame_index;
  int& last_loaded_frame;
  bool& playing;
  int& coord_mode;
  float& point_scale;
  
  // Callbacks
  std::function<void(const std::string&)> report_status;
};

// Load a VTK file
void LoadVtkFile(FileHandlerState& handler_state, const std::filesystem::path& path, bool fit_view);

// Load the latest VTK file from input directory
void LoadLatestVtk(FileHandlerState& handler_state);

#endif // FILE_HANDLER_H

