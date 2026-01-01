#include "file_handler.h"
#include "app_helpers.h"
#include "app_state.h"
#include <filesystem>
// #region agent log
#include <fstream>
#include <chrono>
// #endregion agent log

void LoadVtkFile(FileHandlerState& handler_state, const std::filesystem::path& path, bool fit_view) {
  VtkReadResult read_result = ReadVtkFile(path.string());
  if (!read_result.ok) {
    handler_state.report_status("vtk load error: " + read_result.error);
    AddLog(handler_state.state, handler_state.state_mutex, "vtk: " + read_result.error);
    {
      std::lock_guard<std::mutex> lock(handler_state.state_mutex);
      handler_state.state.last_error = ErrorInfo{
        "VTK load error",
        "Failed to load the selected file.",
        read_result.error,
        {"Verify the file exists and is a supported VTK/VTI format.",
         "If this is a time series, try loading the .pvd manifest or a single frame file."},
        true
      };
      handler_state.state.error_dialog_open = true;
    }
    return;
  }
  if (read_result.kind == VtkReadResult::Kind::PointCloud) {
    handler_state.viewer.SetPointCloud(read_result.domain, read_result.points);
    handler_state.coord_mode = CoordMode::kCartesian3D;
    handler_state.viewer.SetViewMode(ViewModeForCoord(handler_state.coord_mode));
    {
      std::lock_guard<std::mutex> lock(handler_state.state_mutex);
      handler_state.state.has_derived_fields = false;  // Point clouds don't have derived fields
    }
  } else {
    handler_state.viewer.SetData(read_result.domain, read_result.grid);
    // Compute derived fields for loaded data (use default coefficients)
    DerivedFields derived = ComputeDerivedFields(read_result.domain, read_result.grid, 1.0, 1.0, 1.0);
    handler_state.viewer.SetDerivedFields(derived);
    {
      std::lock_guard<std::mutex> lock(handler_state.state_mutex);
      handler_state.state.derived_fields = derived;
      handler_state.state.has_derived_fields = true;
      handler_state.state.current_domain = read_result.domain;
      handler_state.state.current_grid = read_result.grid;  // Store grid for inspection tools
      // Use default PDE coefficients for loaded data
      handler_state.state.current_pde = PDECoefficients();
      handler_state.state.current_pde.a = 1.0;
      handler_state.state.current_pde.b = 1.0;
      handler_state.state.current_pde.az = 1.0;
    }
  }
  if (fit_view) {
    // #region agent log
    {
      std::ofstream f("/Users/nathaniel.sun/Desktop/programming/cursor/.cursor/debug.log",
                      std::ios::app);
      if (f) {
        const auto ts = std::chrono::duration_cast<std::chrono::milliseconds>(
                            std::chrono::system_clock::now().time_since_epoch())
                            .count();
        f << "{\"sessionId\":\"debug-session\",\"runId\":\"run9\",\"hypothesisId\":\"V\","
             "\"location\":\"gui_gl/handlers/file_handler.cpp:LoadVtkFile\","
             "\"message\":\"Calling viewer.FitToView from LoadVtkFile(fit_view=true)\","
             "\"data\":{},\"timestamp\":" << ts << "}\n";
      }
    }
    // #endregion agent log
    handler_state.viewer.FitToView();
    handler_state.point_scale = 1.0f;
  }
  handler_state.report_status("loaded " + path.filename().string());
  AddLog(handler_state.state, handler_state.state_mutex, "vtk: loaded " + path.string());
}

void LoadLatestVtk(FileHandlerState& handler_state) {
  std::filesystem::path dir(handler_state.input_dir);
  auto latest = FindLatestVtk(dir);
  if (!latest) {
    handler_state.report_status("no vtk files found in input dir");
    AddLog(handler_state.state, handler_state.state_mutex, "vtk: no files in " + dir.string());
    {
      std::lock_guard<std::mutex> lock(handler_state.state_mutex);
      handler_state.state.last_error = ErrorInfo{
        "No VTK files found",
        "There are no VTK/VTI files in the current input directory.",
        dir.string(),
        {"Set the Input directory to where solver outputs are written.",
         "Run a solve to generate outputs, then use Load Latest."},
        false
      };
      handler_state.state.error_dialog_open = true;
    }
    return;
  }
  handler_state.frame_paths.clear();
  handler_state.frame_times.clear();
  handler_state.frame_index = 0;
  handler_state.last_loaded_frame = -1;
  handler_state.playing = false;
  LoadVtkFile(handler_state, *latest, true);
}

