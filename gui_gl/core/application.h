#ifndef APPLICATION_H
#define APPLICATION_H

#include "window_manager.h"
#include "event_handler.h"
#include "app_actions.h"
#include "../solver/solver_manager.h"
#include "../models/application_state.h"
#include "../GlViewer.h"
#include "../io/preferences_io.h"
#include "../systems/ui_config.h"
#include "../components/error_dialog.h"
#include "../rendering/glyph_renderer.h"
#include "app_services.h"
#include "../components/inspection_tools.h"
#include "../app_helpers.h"
#include "../systems/command_history.h"
#include "../systems/backend_capabilities.h"
#include "../panels/preferences/benchmark_panel.h"
#include "../panels/preferences/ui_config_panel.h"
#include "../panels/main/time_panel.h"
#include "../docking/docking_context.h"
#include "../../include/run_config.h"
#include <string>
#include <filesystem>
#include <memory>
#include <optional>
#include <unordered_map>
#include <functional>

/**
 * Application - High-level coordinator for the GUI application.
 * 
 * Responsibilities:
 * - Initialize all subsystems (window, ImGui, viewer, etc.)
 * - Own application state and resources
 * - Run main event loop
 * - Coordinate panel rendering
 */
class Application {
 public:
  Application(int argc, char** argv);
  ~Application();

  // Non-copyable
  Application(const Application&) = delete;
  Application& operator=(const Application&) = delete;

  // Run the application (main loop)
  int Run();

  // View system integration - render a panel by ID for the docking system
  void RenderPanelById(const std::string& panel_id, float available_width);

  // Render the timeline controls for the docking system
  void RenderTimeline(float available_width, float available_height);
  const Preferences& prefs() const { return prefs_; }

 private:
  struct ImGuiContextGuard {
    ImGuiContext* ctx = nullptr;
    ~ImGuiContextGuard() {
      // Intentionally do not destroy the ImGui context at shutdown.
      // Some static/global teardown code may still touch ImGui APIs; destroying the context
      // can cause late-exit EXC_BAD_ACCESS inside ImGui (e.g., ImVector::back()).
      // The OS will reclaim process memory on exit.
      (void)ctx;
    }
  };

  // Must be destroyed last (declared first) to avoid member destructors calling ImGui after
  // the context is gone.
  ImGuiContextGuard imgui_ctx_;

  // Initialization
  bool Initialize();
  void LoadPreferences();
  void LoadUIConfig();
  void InitializeImGui();
  void InitializeViewer();
  void RegisterHelpEntries();
  void RegisterPanelRenderers();

  // Main loop
  void ProcessFrame();
  void RenderUI();
  void RenderMainTab(bool time_dependent);
  void RenderInspectTab();
  void RenderPreferencesTab();
  void RenderPanelsForTab(const std::string& tab_name, const char* tab_key, bool time_dependent);
  void RenderVisualizationPanel();
  void RenderVectorGlyphs(const ImVec2& image_min, const ImVec2& image_max);
  void RenderTimelineConfiguration(float available_width);
  void RenderTimelinePlayback(float available_width, float available_height);
  void RenderAnimationControls(float available_width);
  void SyncLayoutFromUIConfig();
  void SyncFontFromUIConfig();
  bool ApplyUIFont(const std::filesystem::path& font_path, float font_size, std::string* warning);
  void ForceApplyUISettings();
  void SavePreferencesIfNeeded();
  void RenderMenuBar();
  void RenderAboutModal();

  bool CanExecuteAction(AppAction action) const;
  bool IsActionChecked(AppAction action) const;
  void ExecuteAction(AppAction action);
  void OpenPanelInTab(const std::string& tab_name, const std::string& panel_id);
  void RequestTab(const std::string& tab_name);
  void ResetSessionState();
  void ResetLayoutState();
  bool IsTimeDependent() const { return time_frames_ > 1; }
  const AppServices& services() const { return services_; }
  RunConfig BuildRunConfig(std::string* error) const;
  bool ApplyRunConfig(const RunConfig& config, std::string* error);
  bool SaveRunConfigFile(const std::filesystem::path& path, std::string* error);
  bool LoadRunConfigFile(const std::filesystem::path& path, std::string* error);

  // Helper methods
  void UpdateApplicationState();
  void SyncStateFromApplicationState();
  void HandleFileLoading();
  void HandleTimeSeriesPlayback();

  // Window and core systems
  WindowManager window_manager_;
  EventHandler event_handler_;
  GlViewer viewer_;
  
  // Application state
  ApplicationState app_state_;
  SharedState& shared_state_;
  std::mutex state_mutex_;
  
  // Solver management
  SolverManager solver_manager_;
  
  // Preferences and config
  Preferences prefs_;
  std::filesystem::path prefs_path_;
  bool prefs_changed_ = false;
  UIConfigManager& ui_config_mgr_;
  std::filesystem::path ui_config_path_;
  std::filesystem::path run_config_path_;
  UIConfigPanelState ui_config_state_;
  
  // Error dialog
  ErrorDialogComponent error_dialog_;

  // UI state variables (will be gradually moved to ApplicationState)
  // For Phase 1, we keep these here to minimize changes
  std::string pde_text_;
  double bound_xmin_, bound_xmax_, bound_ymin_, bound_ymax_, bound_zmin_, bound_zmax_;
  int grid_nx_, grid_ny_, grid_nz_;
  int domain_mode_;
  std::string domain_shape_;
  std::string domain_shape_file_;
  std::string domain_shape_mask_path_;
  ShapeTransform shape_transform_;
  ShapeMask shape_mask_;
  double shape_mask_threshold_ = 0.0;
  bool shape_mask_invert_ = false;
  int coord_mode_;
  double torus_major_, torus_minor_;
  BoundaryInput bc_left_, bc_right_, bc_bottom_, bc_top_, bc_front_, bc_back_;
  std::string output_path_;
  std::string input_dir_;
  int backend_index_;
  int method_index_;
  double sor_omega_;
  int gmres_restart_;
  int solver_max_iter_;
  double solver_tol_;
  int solver_residual_interval_;
  int solver_mg_pre_smooth_;
  int solver_mg_post_smooth_;
  int solver_mg_coarse_iters_;
  int solver_mg_max_levels_;
  int thread_count_;
  int max_threads_;
  int metal_reduce_interval_, metal_tg_x_, metal_tg_y_;
  int pref_metal_reduce_interval_, pref_metal_tg_x_, pref_metal_tg_y_;
  double time_start_, time_end_;
  int time_frames_;
  TimeIntegrationMethod time_integration_method_ = TimeIntegrationMethod::ForwardEuler;
  TimeSteppingMode time_stepping_mode_ = TimeSteppingMode::Fixed;
  double time_cfl_target_ = 0.5;
  double time_min_dt_ = 1e-10;
  double time_max_dt_ = 1.0;
  int latex_font_size_;
  double pref_sor_omega_;
  int pref_gmres_restart_;
  int pref_method_index_;
  
  // Viewer state
  GlViewer::ViewMode view_mode_ = GlViewer::ViewMode::Cartesian;
  bool use_ortho_;
  bool lock_fit_;
  int stride_;
  float point_scale_;
  float data_scale_;
  float zoom_ = 1.0f;
  float yaw_ = 0.6f;
  float pitch_ = -0.4f;
  float distance_ = 3.6f;
  bool play_once_ = false;
  bool z_domain_locked_;
  double z_domain_min_, z_domain_max_;
  bool grid_enabled_;
  int grid_divisions_;
  bool should_save_prefs_ = false;
  
  // Visualization state
  bool slice_enabled_;
  int slice_axis_;
  double slice_value_, slice_thickness_;
  bool iso_enabled_;
  double iso_value_, iso_band_;
  int field_type_index_;
  
  // Time series state
  std::vector<std::string> frame_paths_;
  std::vector<double> frame_times_;
  int frame_index_;
  int last_loaded_frame_;
  bool playing_;
  double play_accum_;
  static constexpr double playback_fps_ = 24.0;

  // New panel state
  std::string initial_condition_expr_;
  std::string source_term_expr_;
  std::string preset_directory_;

  // UI layout state
  // UI layout constants (overridden by UI configuration)
  float input_width_ = 320.0f;
  float left_panel_min_width_ = 380.0f;
  float left_panel_max_width_ = 1200.0f;
  float splitter_width_ = 8.0f;
  std::filesystem::path ui_font_path_;
  std::filesystem::path ui_font_dir_;
  float ui_font_size_ = 0.0f;
  bool pending_font_rebuild_ = false;  // Defer font rebuild to next frame start

  float left_panel_width_;
  bool left_collapsed_;
  float inspect_left_panel_width_;
  bool inspect_left_collapsed_;
  std::string active_tab_ = "Main";
  std::optional<std::string> requested_tab_;
  bool show_about_modal_ = false;
  
  // LaTeX previews
  LatexTexture pde_preview_, shape_preview_;
  LatexTexture bc_left_preview_, bc_right_preview_, bc_bottom_preview_;
  LatexTexture bc_top_preview_, bc_front_preview_, bc_back_preview_;
  std::string python_path_;
  std::filesystem::path script_path_;
  std::filesystem::path cache_dir_;
  std::string latex_color_;
  
  // Benchmark config
  BenchmarkConfig benchmark_config_;
  
  // Command history
  CommandHistory cmd_history_;

  // Backend registry
  BackendUIRegistry backend_registry_;
  bool backend_registry_initialized_ = false;
  
  // File loading helpers
  std::vector<std::string> frame_paths_strings_;
  
  // Mouse interaction
  bool dragging_;
  ImVec2 last_mouse_;

  // Glyph rendering (2D overlay)
  GlyphRenderer2D glyph_renderer_;
  
  // Dependency bundle for injection into other modules
  AppServices services_;

  // Executable path (for finding resources)
  std::filesystem::path exec_path_;

  using PanelRenderer = std::function<void(const PanelConfig&, bool)>;
  std::unordered_map<std::string, PanelRenderer> panel_registry_;

  // Docking system
  std::unique_ptr<DockingContext> docking_ctx_;
  bool use_docking_ui_ = false;  // Feature flag to enable new docking UI
};

// Helper to get project directory for layout storage
std::filesystem::path GetProjectDirectory();

#endif  // APPLICATION_H
