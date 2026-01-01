#ifndef APP_STATE_H
#define APP_STATE_H

#include "boundary_types.h"
#include "pde_types.h"
#include "backend.h"
#include "progress_feedback.h"
#include "vtk_io.h"
#include "imgui.h"
#include "log_service.h"
#include <string>
#include <vector>
#include <optional>
#include <mutex>
#include <map>

struct SolveResult {
  bool ok = false;
  std::string output_path;
  std::string error;
  BackendKind backend = BackendKind::CPU;
  std::string note;
  bool time_series = false;
  std::vector<std::string> frame_paths;
  std::vector<double> frame_times;
};

// Centralized error reporting for modal dialogs / UI toasts.
struct ErrorInfo {
  std::string title;
  std::string message;
  std::string details;
  std::vector<std::string> suggestions;
  bool is_critical = true;
};

// Color preferences structure
struct ColorPreferences {
  // Theme preset: 0=Dark, 1=Light, 2=HighContrast, 3=Custom
  int theme_preset = 0;
  
  // Core colors (ImGui style colors)
  ImVec4 window_bg = ImVec4(0.15f, 0.15f, 0.15f, 1.0f);
  ImVec4 panel_bg = ImVec4(0.20f, 0.20f, 0.20f, 1.0f);
  ImVec4 input_bg = ImVec4(0.25f, 0.25f, 0.25f, 1.0f);
  ImVec4 text_color = ImVec4(1.0f, 1.0f, 1.0f, 1.0f);
  ImVec4 text_disabled = ImVec4(0.50f, 0.50f, 0.50f, 1.0f);
  ImVec4 border_color = ImVec4(0.43f, 0.43f, 0.50f, 0.50f);
  
  // Accent colors
  ImVec4 accent_primary = ImVec4(0.26f, 0.59f, 0.98f, 1.0f);  // Material Blue
  ImVec4 accent_hover = ImVec4(0.35f, 0.66f, 1.0f, 1.0f);
  ImVec4 accent_active = ImVec4(0.20f, 0.50f, 0.90f, 1.0f);
  
  // Status colors
  ImVec4 color_success = ImVec4(0.2f, 0.8f, 0.2f, 1.0f);  // Green
  ImVec4 color_warning = ImVec4(1.0f, 0.8f, 0.2f, 1.0f);  // Yellow
  ImVec4 color_error = ImVec4(1.0f, 0.2f, 0.2f, 1.0f);     // Red
  ImVec4 color_info = ImVec4(0.2f, 0.6f, 1.0f, 1.0f);       // Blue
  
  // Visualization colors
  ImVec4 grid_color = ImVec4(0.86f, 0.88f, 0.92f, 0.9f);
  ImVec4 axis_label_color = ImVec4(0.86f, 0.88f, 0.92f, 0.9f);
  ImVec4 splitter_color = ImVec4(0.37f, 0.39f, 0.47f, 0.71f);
  ImVec4 clear_color = ImVec4(0.05f, 0.06f, 0.07f, 1.0f);
  
  // LaTeX colors
  ImVec4 latex_text = ImVec4(1.0f, 1.0f, 1.0f, 1.0f);
  ImVec4 latex_bg = ImVec4(0.2f, 0.24f, 0.28f, 1.0f);
};

struct Preferences {
  int metal_reduce_interval = 10;
  int metal_tg_x = 0;
  int metal_tg_y = 0;
  int latex_font_size = 18;
  int method_index = 0;  // 0 jacobi, 1 gs, 2 sor, 3 cg, 4 bicgstab, 5 gmres, 6 mg
  double sor_omega = 1.5;
  int gmres_restart = 30;
  ColorPreferences colors;  // Color preferences
  std::map<std::string, bool> ui_section_open;  // Expanded/collapsed state for key UI sections
};

struct SharedState {
  bool running = false;
  double progress = 0.0;
  std::string phase = "idle";
  std::string status;
  std::optional<SolveResult> result;
  std::optional<ErrorInfo> last_error;
  bool error_dialog_open = false;  // Request to open error dialog on next UI frame.
  std::vector<std::string> logs;
  LogService log_service;
  int thread_active = 0;
  int thread_total = 0;
  bool has_duration = false;
  double last_duration = 0.0;
  bool stability_warning = false;
  int stability_frame = 0;
  double stability_ratio = 0.0;
  double stability_max = 0.0;
  std::vector<float> residual_l2;
  std::vector<float> residual_linf;
  DetailedProgress detailed_progress;  // Enhanced progress tracking
  DerivedFields derived_fields;  // Computed derived fields
  bool has_derived_fields = false;  // Whether derived fields are available
  Domain current_domain;  // Current domain for derived field computation
  PDECoefficients current_pde;  // Current PDE coefficients for flux computation
  std::vector<double> current_grid;  // Current grid data for inspection tools

  // Multi-field support
  std::vector<std::string> field_names;           // Available primary field names
  int active_field_index = 0;                     // Currently selected field
  std::vector<std::vector<double>> field_grids;   // Per-field grid data
  std::vector<DerivedFields> all_derived_fields;  // Per-field derived quantities
};

// Helper function to add log entry
void AddLog(SharedState& state, std::mutex& state_mutex, const std::string& message);

#endif  // APP_STATE_H

