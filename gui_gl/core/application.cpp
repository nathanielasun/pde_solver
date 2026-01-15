#define GLFW_INCLUDE_NONE

#include "application.h"
#include "native_menu.h"
#include "../app_helpers.h"
#include "../handlers/file_handler.h"
#include "../handlers/solve_handler.h"
#include "../panels/main/equation_panel.h"
#include "../panels/main/domain_panel.h"
#include "../panels/main/grid_panel.h"
#include "../panels/main/boundary_panel.h"
#include "../panels/main/compute_panel.h"
#include "../panels/inspect/field_panel.h"
#include "../panels/inspect/slice_panel.h"
#include "../panels/inspect/isosurface_panel.h"
#include "../panels/inspect/export_panel.h"
#include "../panels/inspect/advanced_panel.h"
#include "../panels/inspect/comparison_panel.h"
#include "../panels/preferences/color_preferences_panel.h"
#include "../panels/main/time_panel.h"
#include "../panels/main/run_panel.h"
#include "../panels/preferences/viewer_panel.h"
#include "../panels/preferences/io_panel.h"
#include "../panels/preferences/benchmark_panel.h"
#include "../panels/preferences/ui_config_panel.h"
#include "../panels/main/initial_conditions_panel.h"
#include "../panels/main/preset_manager_panel.h"
#include "../panels/main/source_term_panel.h"
#include "../panels/main/material_properties_panel.h"
#include "../panels/main/mesh_preview_panel.h"
#include "../panels/main/parameter_sweep_panel.h"
#include "../panels/main/testing_panel.h"
#include "../panels/inspect/convergence_panel.h"
#include "../panels/inspect/statistics_panel.h"
#include "../panels/inspect/point_probe_panel.h"
#include "../panels/inspect/animation_export_panel.h"
#include "../components/log_panel.h"
#include "../components/comparison_tools.h"
#include "../components/error_dialog.h"
#include "../components/inspection_tools.h"
#include "../systems/command_history.h"
#include "shape_io.h"
#include "../utils/math_utils.h"
#include "input_parse.h"
#include "solver_tokens.h"
#include "../systems/backend_providers.h"
#include "../systems/pde_type_registry.h"
#include "../systems/coordinate_system_registry.h"
#include "../systems/solver_method_registry.h"
#include "../systems/ui_config.h"
#include "../docking/docking_context.h"
#include "../docking/view_registration.h"
#include "../docking/preset_layouts.h"
#include "../styles/ui_style.h"
#include "../ui_helpers.h"
#include "../utils/file_dialog.h"
#include "../utils/coordinate_utils.h"
#include "../validation.h"
#include "../templates.h"
#include "latex_parser.h"
#include "../rendering/glyph_renderer.h"
#include "backend.h"
#include "../progress_feedback.h"
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include "imgui_stdlib.h"
#include <GLFW/glfw3.h>
#include <OpenGL/gl3.h>
#include <iostream>
#include <thread>
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <fstream>
#include <utility>
#include <sys/param.h>

namespace {
constexpr float kMinUIFontSize = 8.0f;
constexpr float kMaxUIFontSize = 72.0f;

bool IsVolumeCoordMode(int coord_mode) {
  return coord_mode == CoordMode::kSphericalVolume ||
         coord_mode == CoordMode::kToroidalVolume ||
         coord_mode == CoordMode::kCartesian3D ||
         coord_mode == CoordMode::kCylindricalVolume;
}

std::string CoordModeToken(int coord_mode) {
  switch (coord_mode) {
    case CoordMode::kCartesian2D:
      return "cartesian2d";
    case CoordMode::kCartesian3D:
      return "cartesian3d";
    case CoordMode::kPolar:
      return "polar";
    case CoordMode::kAxisymmetric:
      return "axisymmetric";
    case CoordMode::kCylindricalVolume:
      return "cylindrical_volume";
    case CoordMode::kSphericalSurface:
      return "spherical_surface";
    case CoordMode::kSphericalVolume:
      return "spherical_volume";
    case CoordMode::kToroidalSurface:
      return "toroidal_surface";
    case CoordMode::kToroidalVolume:
      return "toroidal_volume";
    default:
      return "cartesian2d";
  }
}

bool CoordModeFromToken(const std::string& token, int* coord_mode) {
  const std::string mode = ToLower(Trim(token));
  if (mode.empty()) {
    return false;
  }
  if (mode == "cartesian" || mode == "cartesian2d") {
    if (coord_mode) *coord_mode = CoordMode::kCartesian2D;
    return true;
  }
  if (mode == "cartesian3d") {
    if (coord_mode) *coord_mode = CoordMode::kCartesian3D;
    return true;
  }
  if (mode == "polar") {
    if (coord_mode) *coord_mode = CoordMode::kPolar;
    return true;
  }
  if (mode == "axisymmetric") {
    if (coord_mode) *coord_mode = CoordMode::kAxisymmetric;
    return true;
  }
  if (mode == "cylindrical" || mode == "cylindrical_volume") {
    if (coord_mode) *coord_mode = CoordMode::kCylindricalVolume;
    return true;
  }
  if (mode == "spherical_surface") {
    if (coord_mode) *coord_mode = CoordMode::kSphericalSurface;
    return true;
  }
  if (mode == "spherical_volume") {
    if (coord_mode) *coord_mode = CoordMode::kSphericalVolume;
    return true;
  }
  if (mode == "toroidal_surface") {
    if (coord_mode) *coord_mode = CoordMode::kToroidalSurface;
    return true;
  }
  if (mode == "toroidal_volume") {
    if (coord_mode) *coord_mode = CoordMode::kToroidalVolume;
    return true;
  }
  return false;
}

bool ApplyBoundarySpecToInputs(const std::string& spec,
                               BoundaryInput* left,
                               BoundaryInput* right,
                               BoundaryInput* bottom,
                               BoundaryInput* top,
                               BoundaryInput* front,
                               BoundaryInput* back,
                               std::string* error) {
  if (!left || !right || !bottom || !top || !front || !back) {
    if (error) *error = "missing boundary inputs";
    return false;
  }
  const std::string trimmed = Trim(spec);
  if (trimmed.empty()) {
    return true;
  }
  auto assign = [&](const std::string& side, const std::string& payload) -> bool {
    BoundaryInput* target = nullptr;
    if (side == "left") target = left;
    else if (side == "right") target = right;
    else if (side == "bottom") target = bottom;
    else if (side == "top") target = top;
    else if (side == "front") target = front;
    else if (side == "back") target = back;
    else {
      if (error) *error = "unknown boundary side: " + side;
      return false;
    }
    if (!SetBoundaryFromSpec(payload, target)) {
      if (error) *error = "invalid boundary spec for " + side;
      return false;
    }
    return true;
  };

  const std::vector<std::string> entries = Split(trimmed, ';');
  for (const std::string& entry : entries) {
    const std::string part = Trim(entry);
    if (part.empty()) {
      continue;
    }
    const size_t colon = part.find(':');
    if (colon == std::string::npos) {
      if (error) *error = "invalid boundary entry: " + part;
      return false;
    }
    const std::string side = ToLower(Trim(part.substr(0, colon)));
    const std::string payload = Trim(part.substr(colon + 1));
    if (!assign(side, payload)) {
      return false;
    }
  }
  return true;
}

void BuildUIFontRanges(ImGuiIO& io, ImVector<ImWchar>* ranges) {
  ImFontGlyphRangesBuilder builder;
  builder.AddRanges(io.Fonts->GetGlyphRangesDefault());
  builder.AddRanges(io.Fonts->GetGlyphRangesGreek());
  static const ImWchar kExtendedRanges[] = {
      0x2190, 0x21FF,  // Arrows (↻ ↺ → ← etc.)
      0x2200, 0x22FF,  // Mathematical Operators
      0x2300, 0x23FF,  // Miscellaneous Technical (⏮ ⏪ ⏸ ⏩ ⏭ etc.)
      0x25A0, 0x25FF,  // Geometric Shapes (◀ ▶ ■ □ etc.)
      0x2600, 0x26FF,  // Miscellaneous Symbols
      0
  };
  builder.AddRanges(kExtendedRanges);
  builder.BuildRanges(ranges);
}

std::filesystem::path ResolveUIFontPath(const std::string& font_path,
                                        const std::filesystem::path& font_dir) {
  if (font_path.empty()) {
    return {};
  }
  std::filesystem::path resolved(font_path);
  if (resolved.is_relative() && !font_dir.empty()) {
    resolved = font_dir / resolved;
  }
  return resolved;
}

bool IsDefaultOutputToken(const std::string& value) {
  return value.empty() || value == "outputs" || value == "outputs/";
}

std::string DefaultOutputPath(const std::filesystem::path& exec_path) {
  std::error_code ec;
  std::filesystem::path base = exec_path.empty()
      ? std::filesystem::current_path(ec)
      : exec_path.parent_path();
  if (ec) {
    base = ".";
  }
  base /= "outputs";
  std::string out = base.string();
  if (!out.empty() && out.back() != '/') {
    out.push_back('/');
  }
  return out;
}
}

Application::Application(int argc, char** argv)
    : shared_state_(app_state_.GetMutableState().shared_state),
      solver_manager_(shared_state_, state_mutex_, viewer_),
      ui_config_mgr_(UIConfigManager::Instance()),
      cmd_history_(50) {
  services_.shared_state = &shared_state_;
  services_.state_mutex = &state_mutex_;
  services_.viewer = &viewer_;
  services_.solver_manager = &solver_manager_;
  
  // Store executable path
  if (argc > 0 && argv && argv[0]) {
    std::error_code ec;
    exec_path_ = std::filesystem::weakly_canonical(argv[0], ec);
    if (ec) {
      exec_path_ = std::filesystem::path(argv[0]);
    }
  }
  
  // Initialize default values
  // Initialize UI state from ApplicationState defaults.
  SyncStateFromApplicationState();
  const std::string default_output = DefaultOutputPath(exec_path_);
  if (IsDefaultOutputToken(output_path_)) {
    output_path_ = default_output;
  }
  if (IsDefaultOutputToken(input_dir_)) {
    input_dir_ = default_output;
  }

  // Defaults not stored in ApplicationState yet.
  SolverConfig solver_defaults;
  pref_metal_reduce_interval_ = 10;
  pref_metal_tg_x_ = 0;
  pref_metal_tg_y_ = 0;
  latex_font_size_ = 18;
  pref_method_index_ = 0;
  pref_sor_omega_ = 1.5;
  pref_gmres_restart_ = 30;
  solver_max_iter_ = solver_defaults.max_iter;
  solver_tol_ = solver_defaults.tol;
  solver_residual_interval_ = solver_defaults.residual_interval;
  solver_mg_pre_smooth_ = solver_defaults.mg_pre_smooth;
  solver_mg_post_smooth_ = solver_defaults.mg_post_smooth;
  solver_mg_coarse_iters_ = solver_defaults.mg_coarse_iters;
  solver_mg_max_levels_ = solver_defaults.mg_max_levels;

  use_ortho_ = false;
  lock_fit_ = true;
  stride_ = 1;
  point_scale_ = 1.0f;
  data_scale_ = 0.5f;
  z_domain_locked_ = false;
  z_domain_min_ = -1.0;
  z_domain_max_ = 1.0;
  grid_enabled_ = true;
  grid_divisions_ = 10;

  play_accum_ = 0.0;
  
  left_panel_width_ = 400.0f;
  left_collapsed_ = false;
  inspect_left_panel_width_ = 400.0f;
  inspect_left_collapsed_ = false;
  
  dragging_ = false;
  
  // Benchmark config
  benchmark_config_.pde = "-\\frac{\\partial^2 u}{\\partial x^2} - \\frac{\\partial^2 u}{\\partial y^2} + "
                          "\\frac{\\partial^2 u}{\\partial t^2} = u";
  benchmark_config_.xmin = 0.0;
  benchmark_config_.xmax = 1.0;
  benchmark_config_.ymin = 0.0;
  benchmark_config_.ymax = 1.0;
  benchmark_config_.nx = 640;
  benchmark_config_.ny = 640;
  benchmark_config_.bc = "dirichlet:2";
  benchmark_config_.output_dir = default_output + "benchmarks/";
  benchmark_config_.t_start = 0.0;
  benchmark_config_.t_end = 1.0;
  benchmark_config_.frames = 1000;
}

Application::~Application() {
  solver_manager_.Join();
  viewer_.Shutdown();
  ImGui_ImplOpenGL3_Shutdown();
  ImGui_ImplGlfw_Shutdown();
  ImGui::DestroyContext();
}

bool Application::Initialize() {
  // Initialize window
  WindowManager::Config window_config;
  window_config.width = 1400;
  window_config.height = 900;
  window_config.title = "PDE Solver";
  if (!window_manager_.Init(window_config)) {
    return false;
  }
  
  // Initialize event handler
  event_handler_.SetCallbacks({});
  
  InitializeImGui();
  InitializeViewer();
  LoadPreferences();
  LoadUIConfig();
  RegisterHelpEntries();
  RegisterPanelRenderers();

  // Initialize docking system
  RegisterAllViews();
  docking_ctx_ = std::make_unique<DockingContext>();
  docking_ctx_->SetRoot(PresetLayouts::CreateDefault());

  // Set up services with app pointer for view rendering
  services_.app = this;

  // Initialize ApplicationState
  UpdateApplicationState();

  // Set up event handler callbacks (will be set again after window is created)
  // This is done in Run() after window is initialized

  AddLog(shared_state_, state_mutex_, "ui: OpenGL GUI ready");
  return true;
}

void Application::InitializeImGui() {
  IMGUI_CHECKVERSION();
  ImGui::CreateContext();
  ImGuiIO& io = ImGui::GetIO();
  io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
  
  ImGui::StyleColorsDark();
  UIStyle::Initialize();
  
  ImGui_ImplGlfw_InitForOpenGL(window_manager_.GetWindow(), true);
  ImGui_ImplOpenGL3_Init("#version 330");
}

void Application::InitializeViewer() {
  if (!viewer_.Init()) {
    std::cerr << "Failed to initialize viewer\n";
  }
  viewer_.SetOrthographic(use_ortho_);
  viewer_.SetTorusRadii(static_cast<float>(torus_major_), static_cast<float>(torus_minor_));
}

void Application::LoadPreferences() {
  prefs_path_ = ResolvePrefsPath(exec_path_);
  std::error_code prefs_ec;
  const bool prefs_found = std::filesystem::exists(prefs_path_, prefs_ec);
  bool prefs_loaded = true;
  std::string prefs_error;
  if (prefs_found) {
    prefs_loaded = ::LoadPreferences(prefs_path_, &prefs_, &prefs_error);
  }
  if (prefs_loaded && prefs_found) {
    pref_metal_reduce_interval_ = prefs_.metal_reduce_interval;
    pref_metal_tg_x_ = prefs_.metal_tg_x;
    pref_metal_tg_y_ = prefs_.metal_tg_y;
    metal_reduce_interval_ = pref_metal_reduce_interval_;
    metal_tg_x_ = pref_metal_tg_x_;
    metal_tg_y_ = pref_metal_tg_y_;
    latex_font_size_ = prefs_.latex_font_size;
    pref_method_index_ = prefs_.method_index;
    pref_sor_omega_ = prefs_.sor_omega;
    pref_gmres_restart_ = prefs_.gmres_restart;
    method_index_ = pref_method_index_;
    sor_omega_ = pref_sor_omega_;
    gmres_restart_ = pref_gmres_restart_;
    ApplyColorPreferences(prefs_.colors);
  } else {
    ApplyColorPreferences(prefs_.colors);
  }
  
  // Update LaTeX color from preferences
  std::ostringstream latex_color_oss;
  latex_color_oss << "#" << std::hex << std::setfill('0')
      << std::setw(2) << static_cast<int>(prefs_.colors.latex_text.x * 255)
      << std::setw(2) << static_cast<int>(prefs_.colors.latex_text.y * 255)
      << std::setw(2) << static_cast<int>(prefs_.colors.latex_text.z * 255);
  latex_color_ = latex_color_oss.str();
  
  if (!prefs_loaded && !prefs_error.empty()) {
    AddLog(shared_state_, state_mutex_, "prefs: " + prefs_error);
  } else if (prefs_found) {
    AddLog(shared_state_, state_mutex_, "prefs: loaded " + prefs_path_.string());
  }
}

void Application::LoadUIConfig() {
  // Save UI config local to the executable (same directory as preferences)
  std::filesystem::path config_dir = exec_path_.empty() ?
      std::filesystem::current_path() : exec_path_.parent_path();
  ui_config_path_ = config_dir / "ui_config.json";
  run_config_path_ = config_dir / "run_config.json";
  if (std::filesystem::exists(ui_config_path_)) {
    if (ui_config_mgr_.LoadFromFile(ui_config_path_.string())) {
      AddLog(shared_state_, state_mutex_, "ui: loaded config from " + ui_config_path_.string());
    } else {
      AddLog(shared_state_, state_mutex_, "ui: failed to load config, using defaults");
    }
  } else {
    std::filesystem::path default_config = exec_path_.parent_path() / "gui_gl" / "config" / "default_ui.json";
    if (std::filesystem::exists(default_config)) {
      ui_config_mgr_.LoadFromFile(default_config.string());
      AddLog(shared_state_, state_mutex_, "ui: loaded default config");
    } else {
      AddLog(shared_state_, state_mutex_, "ui: using built-in default config");
    }
  }

  SyncLayoutFromUIConfig();
}

void Application::RegisterHelpEntries() {
  // Register help entries (existing code from main.cpp)
  UIHelp::RegisterHelp(UIHelp::HelpEntry{
    "solve",
    "Solve PDE",
    "Click 'Solve PDE' to start solving. Use Mod+S (Cmd+S on Mac, Ctrl+S elsewhere) as a shortcut.",
    "Actions",
    {"solve", "run", "execute", "compute"},
    "Mod+S"
  });
  // ... (other help entries would go here)
}

void Application::RenderPanelById(const std::string& panel_id, float available_width) {
  auto it = panel_registry_.find(panel_id);
  if (it == panel_registry_.end()) {
    ImGui::TextDisabled("Unknown panel: %s", panel_id.c_str());
    return;
  }

  // Save and restore input width for proper panel rendering
  float saved_width = input_width_;
  input_width_ = available_width - 40.0f;  // Account for padding

  // Create a minimal panel config for rendering
  PanelConfig config;
  config.id = panel_id;
  // Use default components

  // Render the panel
  it->second(config, false);

  input_width_ = saved_width;
}

void Application::RenderTimeline(float available_width, float available_height) {
  // Pure animation playback view - renders transport controls, slider, and frame info
  ImGui::BeginChild("##timeline_view", ImVec2(available_width, available_height));
  RenderAnimationControls(available_width);
  ImGui::EndChild();
}

void Application::RenderAnimationControls(float available_width) {
  // Shared animation playback controls - used by both docking timeline and visualization panel
  const bool has_frames = !frame_paths_.empty();
  const int frame_count = has_frames ? static_cast<int>(frame_paths_.size()) : 0;

  if (has_frames) {
    frame_index_ = std::clamp(frame_index_, 0, std::max(0, frame_count - 1));
  }

  // Static state for playback direction
  static bool playing_reverse = false;
  static bool loop_playback = true;

  // Transport control labels - using ASCII for maximum compatibility
  // These are clean, professional labels that work with any font
  const char* icon_skip_start = "|<";
  const char* icon_step_back = "<";
  const char* icon_play_back = "<<";
  const char* icon_pause = "||";
  const char* icon_play_fwd = ">>";
  const char* icon_step_fwd = ">";
  const char* icon_skip_end = ">|";
  const char* icon_loop_on = "O";   // Loop enabled (circular)
  const char* icon_loop_off = "-";  // Loop disabled (linear)

  // Layout constants
  const float button_size = 32.0f;
  const float button_spacing = 4.0f;
  const float section_spacing = 12.0f;

  // Calculate total width of transport controls to center them
  const int num_transport_buttons = 7;  // skip, step, play_rev, play_fwd, step, skip, loop
  const float transport_width = num_transport_buttons * button_size +
                                 (num_transport_buttons - 2) * button_spacing +
                                 section_spacing;  // extra space before loop

  // Center the transport controls
  float start_x = (available_width - transport_width) * 0.5f;
  if (start_x > 0) {
    ImGui::SetCursorPosX(ImGui::GetCursorPosX() + start_x);
  }

  if (!has_frames) {
    ImGui::BeginDisabled();
  }

  // Transport button style
  ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 4.0f);
  ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(6, 6));

  // Skip to start
  if (ImGui::Button(icon_skip_start, ImVec2(button_size, button_size))) {
    frame_index_ = 0;
    playing_ = false;
    playing_reverse = false;
    play_accum_ = 0.0;
  }
  if (ImGui::IsItemHovered()) ImGui::SetTooltip("First frame (Home)");

  ImGui::SameLine(0, button_spacing);

  // Step backward
  if (ImGui::Button(icon_step_back, ImVec2(button_size, button_size))) {
    if (frame_index_ > 0) {
      frame_index_--;
    }
    playing_ = false;
    playing_reverse = false;
    play_accum_ = 0.0;
  }
  if (ImGui::IsItemHovered()) ImGui::SetTooltip("Previous frame");

  ImGui::SameLine(0, button_spacing);

  // Play backward - highlighted when active
  const bool reverse_active = playing_reverse;
  if (reverse_active) {
    ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.2f, 0.5f, 0.8f, 1.0f));
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.3f, 0.6f, 0.9f, 1.0f));
  }
  if (ImGui::Button(reverse_active ? icon_pause : icon_play_back, ImVec2(button_size, button_size))) {
    if (reverse_active) {
      playing_reverse = false;
    } else {
      playing_reverse = true;
      playing_ = false;
    }
    play_accum_ = 0.0;
  }
  if (reverse_active) {
    ImGui::PopStyleColor(2);
  }
  if (ImGui::IsItemHovered()) ImGui::SetTooltip(reverse_active ? "Pause" : "Play backward");

  ImGui::SameLine(0, button_spacing);

  // Play forward - highlighted when active
  const bool forward_active = playing_;
  if (forward_active) {
    ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.2f, 0.5f, 0.8f, 1.0f));
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.3f, 0.6f, 0.9f, 1.0f));
  }
  if (ImGui::Button(forward_active ? icon_pause : icon_play_fwd, ImVec2(button_size, button_size))) {
    if (forward_active) {
      playing_ = false;
    } else {
      playing_ = true;
      playing_reverse = false;
    }
    play_accum_ = 0.0;
  }
  if (forward_active) {
    ImGui::PopStyleColor(2);
  }
  if (ImGui::IsItemHovered()) ImGui::SetTooltip(forward_active ? "Pause" : "Play forward");

  ImGui::SameLine(0, button_spacing);

  // Step forward
  if (ImGui::Button(icon_step_fwd, ImVec2(button_size, button_size))) {
    if (frame_index_ < frame_count - 1) {
      frame_index_++;
    }
    playing_ = false;
    playing_reverse = false;
    play_accum_ = 0.0;
  }
  if (ImGui::IsItemHovered()) ImGui::SetTooltip("Next frame");

  ImGui::SameLine(0, button_spacing);

  // Skip to end
  if (ImGui::Button(icon_skip_end, ImVec2(button_size, button_size))) {
    frame_index_ = std::max(0, frame_count - 1);
    playing_ = false;
    playing_reverse = false;
    play_accum_ = 0.0;
  }
  if (ImGui::IsItemHovered()) ImGui::SetTooltip("Last frame (End)");

  ImGui::SameLine(0, section_spacing);

  // Loop toggle - highlighted when active
  const bool loop_active = loop_playback;
  if (loop_active) {
    ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.2f, 0.6f, 0.3f, 1.0f));
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.3f, 0.7f, 0.4f, 1.0f));
  }
  if (ImGui::Button(loop_active ? icon_loop_on : icon_loop_off, ImVec2(button_size, button_size))) {
    loop_playback = !loop_playback;
  }
  if (loop_active) {
    ImGui::PopStyleColor(2);
  }
  if (ImGui::IsItemHovered()) ImGui::SetTooltip(loop_active ? "Loop: ON" : "Loop: OFF");

  ImGui::PopStyleVar(2);

  if (!has_frames) {
    ImGui::EndDisabled();
  }

  // Row 2: Frame info centered
  ImGui::Spacing();
  {
    char frame_text[128];
    if (has_frames) {
      if (static_cast<size_t>(frame_index_) < frame_times_.size()) {
        snprintf(frame_text, sizeof(frame_text), "Frame %d / %d  |  t = %.4g",
                 frame_index_ + 1, frame_count, frame_times_[static_cast<size_t>(frame_index_)]);
      } else {
        snprintf(frame_text, sizeof(frame_text), "Frame %d / %d", frame_index_ + 1, frame_count);
      }
    } else {
      snprintf(frame_text, sizeof(frame_text), "No time series loaded");
    }
    float text_width = ImGui::CalcTextSize(frame_text).x;
    ImGui::SetCursorPosX((available_width - text_width) * 0.5f);
    if (has_frames) {
      ImGui::Text("%s", frame_text);
    } else {
      ImGui::TextDisabled("%s", frame_text);
    }
  }

  // Row 3: Timeline slider (full width with padding)
  ImGui::Spacing();
  const float slider_padding = 8.0f;
  if (!has_frames) {
    ImGui::BeginDisabled();
  }
  ImGui::SetCursorPosX(slider_padding);
  ImGui::SetNextItemWidth(available_width - slider_padding * 2.0f);
  ImGui::PushStyleVar(ImGuiStyleVar_GrabRounding, 4.0f);
  ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 2.0f);
  int slider_value = frame_index_;
  if (ImGui::SliderInt("##timeline_slider", &slider_value, 0, std::max(0, frame_count - 1), "")) {
    frame_index_ = slider_value;
    playing_ = false;
    playing_reverse = false;
    play_accum_ = 0.0;
  }
  ImGui::PopStyleVar(2);
  if (!has_frames) {
    ImGui::EndDisabled();
  }

  // Handle playback animation
  if (has_frames) {
    if (playing_) {
      play_accum_ += ImGui::GetIO().DeltaTime * playback_fps_;
      if (play_accum_ >= 1.0) {
        int steps = static_cast<int>(play_accum_);
        play_accum_ -= steps;
        frame_index_ += steps;
        if (frame_index_ >= frame_count) {
          if (loop_playback) {
            frame_index_ = 0;
          } else {
            frame_index_ = frame_count - 1;
            playing_ = false;
          }
        }
      }
    } else if (playing_reverse) {
      play_accum_ += ImGui::GetIO().DeltaTime * playback_fps_;
      if (play_accum_ >= 1.0) {
        int steps = static_cast<int>(play_accum_);
        play_accum_ -= steps;
        frame_index_ -= steps;
        if (frame_index_ < 0) {
          if (loop_playback) {
            frame_index_ = frame_count - 1;
          } else {
            frame_index_ = 0;
            playing_reverse = false;
          }
        }
      }
    }
  }
}

void Application::RegisterPanelRenderers() {
  panel_registry_.clear();

  panel_registry_["equation"] = [this](const PanelConfig& panel, bool) {
    EquationPanelState eq_state{
      pde_text_, bound_xmin_, bound_xmax_, bound_ymin_, bound_ymax_,
      bound_zmin_, bound_zmax_, grid_nx_, grid_ny_, grid_nz_,
      domain_mode_, domain_shape_, coord_mode_,
      bc_left_, bc_right_, bc_bottom_, bc_top_, bc_front_, bc_back_,
      method_index_, sor_omega_, gmres_restart_,
      time_start_, time_end_, time_frames_,
      viewer_, [this]() { app_state_.UpdateCoordinateFlags(); }, pde_preview_,
      python_path_, script_path_, cache_dir_, latex_color_, latex_font_size_,
      input_width_, &cmd_history_
    };
    RenderEquationPanel(eq_state, panel.components);
    UpdateApplicationState();
  };

  panel_registry_["domain"] = [this](const PanelConfig& panel, bool) {
    ApplicationState::State& s = app_state_.GetMutableState();
    DomainPanelState domain_state{
      coord_mode_,
      bound_xmin_, bound_xmax_, bound_ymin_, bound_ymax_, bound_zmin_, bound_zmax_,
      torus_major_, torus_minor_,
      domain_mode_,
      domain_shape_,
      domain_shape_file_,
      domain_shape_mask_path_,
      shape_mask_,
      shape_mask_threshold_,
      shape_mask_invert_,
      shape_transform_,
      viewer_,
      [this]() { app_state_.UpdateCoordinateFlags(); },
      s.use_polar_coords,
      s.use_cartesian_3d,
      s.use_axisymmetric,
      s.use_cylindrical_volume,
      s.use_spherical_surface,
      s.use_spherical_volume,
      s.use_toroidal_surface,
      s.use_toroidal_volume,
      s.use_surface,
      s.use_volume,
      shape_preview_,
      python_path_, script_path_, cache_dir_, latex_color_, latex_font_size_,
      input_width_,
      &cmd_history_
    };
    RenderDomainPanel(domain_state, panel.components);
    UpdateApplicationState();
  };

  panel_registry_["grid"] = [this](const PanelConfig& panel, bool) {
    ApplicationState::State& s = app_state_.GetMutableState();
    GridPanelState grid_state{
      grid_nx_, grid_ny_, grid_nz_,
      s.use_cartesian_3d,
      s.use_cylindrical_volume,
      s.use_volume,
      s.use_surface,
      s.use_axisymmetric,
      s.use_polar_coords,
      input_width_,
      &cmd_history_
    };
    RenderGridPanel(grid_state, panel.components);
    UpdateApplicationState();
  };

  panel_registry_["boundary"] = [this](const PanelConfig& panel, bool) {
    ApplicationState::State& s = app_state_.GetMutableState();
    BoundaryPanelState bc_state{
      bc_left_, bc_right_, bc_bottom_, bc_top_, bc_front_, bc_back_,
      bc_left_preview_, bc_right_preview_, bc_bottom_preview_,
      bc_top_preview_, bc_front_preview_, bc_back_preview_,
      s.use_cartesian_3d,
      s.use_axisymmetric,
      s.use_polar_coords,
      s.use_cylindrical_volume,
      s.use_surface,
      s.use_volume,
      python_path_, script_path_, cache_dir_, latex_color_, latex_font_size_,
      &cmd_history_
    };
    RenderBoundaryPanel(bc_state, panel.components);
    UpdateApplicationState();
  };

  panel_registry_["time"] = [this](const PanelConfig& panel, bool time_dependent) {
    // Calculate grid spacing from domain for CFL display
    double dx = 0.0, dy = 0.0, dz = 0.0;
    if (grid_nx_ > 1) {
      dx = (bound_xmax_ - bound_xmin_) / static_cast<double>(grid_nx_ - 1);
    }
    if (grid_ny_ > 1) {
      dy = (bound_ymax_ - bound_ymin_) / static_cast<double>(grid_ny_ - 1);
    }
    if (grid_nz_ > 1) {
      dz = (bound_zmax_ - bound_zmin_) / static_cast<double>(grid_nz_ - 1);
    }
    // Use minimum spacing for CFL
    double min_spacing = dx;
    if (dy > 0.0 && (min_spacing == 0.0 || dy < min_spacing)) min_spacing = dy;
    if (dz > 0.0 && (min_spacing == 0.0 || dz < min_spacing)) min_spacing = dz;

    // Estimate wave speed (for heat/diffusion, use 1.0 as placeholder)
    double wave_speed = 1.0;

    TimePanelState time_state{
      .time_start = time_start_,
      .time_end = time_end_,
      .time_frames = time_frames_,
      .integration_method = time_integration_method_,
      .stepping_mode = time_stepping_mode_,
      .cfl_target = time_cfl_target_,
      .min_dt = time_min_dt_,
      .max_dt = time_max_dt_,
      .dx = min_spacing,
      .dy = dy,
      .dz = dz,
      .wave_speed = wave_speed,
      .time_dependent = time_dependent,
      .input_width = input_width_,
      .cmd_history = &cmd_history_,
    };
    RenderTimePanel(time_state, panel.components);
    UpdateApplicationState();
  };

  panel_registry_["run"] = [this](const PanelConfig& panel, bool) {
    bool running_snap = false;
    double progress_snap = 0.0;
    std::string phase_snap;
    std::string status_snap;
    bool has_duration_snap = false;
    double last_duration_snap = 0.0;
    bool stability_warning_snap = false;
    int stability_frame_snap = 0;
    double stability_ratio_snap = 0.0;
    double stability_max_snap = 0.0;
    std::vector<float> residual_l2_snap;
    std::vector<float> residual_linf_snap;
    int thread_active_snap = 0;
    int thread_total_snap = 0;

    {
      std::lock_guard<std::mutex> lock(state_mutex_);
      running_snap = shared_state_.running;
      progress_snap = shared_state_.progress;
      phase_snap = shared_state_.phase;
      status_snap = shared_state_.status;
      has_duration_snap = shared_state_.has_duration;
      last_duration_snap = shared_state_.last_duration;
      stability_warning_snap = shared_state_.stability_warning;
      stability_frame_snap = shared_state_.stability_frame;
      stability_ratio_snap = shared_state_.stability_ratio;
      stability_max_snap = shared_state_.stability_max;
      residual_l2_snap = shared_state_.residual_l2;
      residual_linf_snap = shared_state_.residual_linf;
      thread_active_snap = shared_state_.thread_active;
      thread_total_snap = shared_state_.thread_total;
    }

    RunPanelState run_state{
      shared_state_, state_mutex_,
      running_snap, progress_snap, phase_snap,
      status_snap, has_duration_snap, last_duration_snap,
      stability_warning_snap, stability_frame_snap,
      stability_ratio_snap, stability_max_snap,
      residual_l2_snap, residual_linf_snap,
      thread_active_snap, thread_total_snap,
      [this]() {
        UpdateApplicationState();
        SolveHandlerState solve_state{
          pde_text_, coord_mode_,
          bound_xmin_, bound_xmax_, bound_ymin_, bound_ymax_, bound_zmin_, bound_zmax_,
          grid_nx_, grid_ny_, grid_nz_, domain_mode_, domain_shape_,
          domain_shape_file_, domain_shape_mask_path_, shape_mask_,
          shape_mask_threshold_, shape_mask_invert_, shape_transform_,
          bc_left_, bc_right_, bc_bottom_, bc_top_, bc_front_, bc_back_,
          backend_index_, method_index_, sor_omega_, gmres_restart_,
          solver_max_iter_, solver_tol_, solver_residual_interval_,
          solver_mg_pre_smooth_, solver_mg_post_smooth_, solver_mg_coarse_iters_, solver_mg_max_levels_,
          thread_count_,
          metal_reduce_interval_, metal_tg_x_, metal_tg_y_,
          time_start_, time_end_, time_frames_, output_path_,
          shared_state_, state_mutex_, viewer_,
          solver_manager_.ThreadPtr(),
          solver_manager_.CancelFlag(),
          solver_manager_.MakeReportStatusCallback(),
          solver_manager_.MakeStartSolverCallback()
        };
        solver_manager_.LaunchSolve(solve_state);
      },
      [this]() { solver_manager_.RequestStop(); },
      [this]() { HandleFileLoading(); }
    };
    RenderRunPanel(run_state, panel.components);
  };

  panel_registry_["log"] = [this](const PanelConfig& panel, bool) {
    const std::vector<std::string> default_components = {"log_view"};
    const std::vector<std::string>& component_list =
        panel.components.empty() ? default_components : panel.components;
    static std::unique_ptr<LogPanelComponent> log_panel;
    if (!log_panel) {
      log_panel = std::make_unique<LogPanelComponent>();
    }

    for (size_t i = 0; i < component_list.size(); ++i) {
      const std::string& id = component_list[i];
      if (id == "log_view") {
        log_panel->SetService(&shared_state_.log_service);
        log_panel->Render();
      } else {
        DrawUnknownComponentPlaceholder(id.c_str());
      }
      if (i + 1 < component_list.size()) {
        ImGui::Spacing();
      }
    }
  };

  panel_registry_["field_panel"] = [this](const PanelConfig& panel, bool) {
    ApplicationState::State& s = app_state_.GetMutableState();
    FieldPanelState field_state{
      viewer_,
      state_mutex_,
      shared_state_.derived_fields,
      shared_state_.has_derived_fields,
      field_type_index_,
      s.use_volume,
      input_width_,
      shared_state_.field_names,
      shared_state_.active_field_index,
      shared_state_.all_derived_fields
    };
    RenderFieldPanel(field_state, panel.components);
    UpdateApplicationState();
  };

  panel_registry_["slice_panel"] = [this](const PanelConfig& panel, bool) {
    ApplicationState::State& s = app_state_.GetMutableState();
    SlicePanelState slice_state{
      viewer_,
      state_mutex_,
      shared_state_.current_domain,
      slice_enabled_,
      slice_axis_,
      slice_value_,
      slice_thickness_,
      s.use_cartesian_3d,
      s.use_cylindrical_volume,
      s.use_volume,
      s.use_surface,
      input_width_
    };
    RenderSlicePanel(slice_state, panel.components);
    UpdateApplicationState();
  };

  panel_registry_["iso_panel"] = [this](const PanelConfig& panel, bool) {
    IsosurfacePanelState iso_state{
      viewer_,
      iso_enabled_,
      iso_value_,
      iso_band_,
      input_width_
    };
    RenderIsosurfacePanel(iso_state, panel.components);
    UpdateApplicationState();
  };

  panel_registry_["export_panel"] = [this](const PanelConfig& panel, bool) {
    ExportPanelState export_state{
      viewer_,
      input_width_
    };
    RenderExportPanel(export_state, panel.components);
  };

  panel_registry_["advanced_panel"] = [this](const PanelConfig& panel, bool) {
    AdvancedPanelState advanced_state{
      viewer_,
      state_mutex_,
      shared_state_.current_domain,
      shared_state_.current_grid,
      shared_state_.derived_fields,
      shared_state_.has_derived_fields,
      input_width_
    };
    RenderAdvancedPanel(advanced_state, panel.components);
  };

  panel_registry_["comparison_panel"] = [this](const PanelConfig& panel, bool) {
    ComparisonPanelState comparison_state{
      viewer_,
      state_mutex_,
      shared_state_.current_domain,
      shared_state_.current_grid,
      input_width_
    };
    RenderComparisonPanel(comparison_state, panel.components);
  };

  // Combined "inspect" panel for backward compatibility with configs that use a single panel
  panel_registry_["inspect"] = [this](const PanelConfig& panel, bool) {
    ApplicationState::State& s = app_state_.GetMutableState();
    const std::vector<std::string>& component_list = panel.components;

    for (size_t i = 0; i < component_list.size(); ++i) {
      const std::string& id = component_list[i];

      if (id == "field_selector") {
        FieldPanelState field_state{
          viewer_,
          state_mutex_,
          shared_state_.derived_fields,
          shared_state_.has_derived_fields,
          field_type_index_,
          s.use_volume,
          input_width_,
          shared_state_.field_names,
          shared_state_.active_field_index,
          shared_state_.all_derived_fields
        };
        RenderFieldPanel(field_state, {id});
        UpdateApplicationState();
      } else if (id == "slice_controls") {
        SlicePanelState slice_state{
          viewer_,
          state_mutex_,
          shared_state_.current_domain,
          slice_enabled_,
          slice_axis_,
          slice_value_,
          slice_thickness_,
          s.use_cartesian_3d,
          s.use_cylindrical_volume,
          s.use_volume,
          s.use_surface,
          input_width_
        };
        RenderSlicePanel(slice_state, {id});
        UpdateApplicationState();
      } else if (id == "isosurface_controls") {
        IsosurfacePanelState iso_state{
          viewer_,
          iso_enabled_,
          iso_value_,
          iso_band_,
          input_width_
        };
        RenderIsosurfacePanel(iso_state, {id});
        UpdateApplicationState();
      } else if (id == "image_export") {
        ExportPanelState export_state{
          viewer_,
          input_width_
        };
        RenderExportPanel(export_state, {id});
      } else if (id == "advanced_inspection") {
        AdvancedPanelState advanced_state{
          viewer_,
          state_mutex_,
          shared_state_.current_domain,
          shared_state_.current_grid,
          shared_state_.derived_fields,
          shared_state_.has_derived_fields,
          input_width_
        };
        RenderAdvancedPanel(advanced_state, {id});
      } else if (id == "comparison_tools") {
        ComparisonPanelState comparison_state{
          viewer_,
          state_mutex_,
          shared_state_.current_domain,
          shared_state_.current_grid,
          input_width_
        };
        RenderComparisonPanel(comparison_state, {id});
      } else {
        DrawUnknownComponentPlaceholder(id.c_str());
      }

      if (i + 1 < component_list.size()) {
        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();
      }
    }
  };

  panel_registry_["compute"] = [this](const PanelConfig& panel, bool) {
    auto ensure_backend_registry = [this]() {
      if (!backend_registry_initialized_) {
        InitializeBackendProviders(backend_registry_);
        backend_registry_initialized_ = true;
      }
    };
    ComputePanelState compute_state{
      backend_index_, method_index_, sor_omega_, gmres_restart_,
      pref_method_index_, pref_sor_omega_, pref_gmres_restart_, prefs_changed_,
      thread_count_, max_threads_,
      metal_reduce_interval_, metal_tg_x_, metal_tg_y_,
      pref_metal_reduce_interval_, pref_metal_tg_x_, pref_metal_tg_y_,
      &backend_registry_,
      ensure_backend_registry,
      input_width_,
      &cmd_history_
    };
    RenderComputePanel(compute_state, panel.components);
  };

  panel_registry_["colors"] = [this](const PanelConfig& panel, bool) {
    const std::vector<std::string> default_components = {"color_preferences"};
    const std::vector<std::string>& component_list =
        panel.components.empty() ? default_components : panel.components;

    for (size_t i = 0; i < component_list.size(); ++i) {
      const std::string& id = component_list[i];
      if (id == "color_preferences") {
        ColorPreferencesPanelState color_state{
          prefs_.colors, prefs_changed_, input_width_
        };
        RenderColorPreferencesPanel(color_state);
        if (prefs_changed_) {
          ApplyColorPreferences(prefs_.colors);
          std::ostringstream latex_color_oss;
          latex_color_oss << "#" << std::hex << std::setfill('0')
              << std::setw(2) << static_cast<int>(prefs_.colors.latex_text.x * 255)
              << std::setw(2) << static_cast<int>(prefs_.colors.latex_text.y * 255)
              << std::setw(2) << static_cast<int>(prefs_.colors.latex_text.z * 255);
          latex_color_ = latex_color_oss.str();
        }
      } else {
        DrawUnknownComponentPlaceholder(id.c_str());
      }

      if (i + 1 < component_list.size()) {
        ImGui::Spacing();
      }
    }
  };

  panel_registry_["viewer"] = [this](const PanelConfig& panel, bool) {
    ViewerPanelState viewer_state{
      viewer_, coord_mode_,
      use_ortho_, lock_fit_, stride_, point_scale_, data_scale_,
      z_domain_locked_, z_domain_min_, z_domain_max_,
      grid_enabled_, grid_divisions_,
      input_width_
    };
    RenderViewerPanel(viewer_state, panel.components);
  };

  panel_registry_["io"] = [this](const PanelConfig& panel, bool) {
    IOPanelState io_state{
      output_path_, input_dir_,
      input_width_
    };
    RenderIOPanel(io_state, panel.components);
  };

  panel_registry_["latex"] = [this](const PanelConfig& panel, bool) {
    const std::vector<std::string> default_components = {"latex_settings"};
    const std::vector<std::string>& component_list =
        panel.components.empty() ? default_components : panel.components;

    for (size_t i = 0; i < component_list.size(); ++i) {
      const std::string& id = component_list[i];
      if (id == "latex_settings") {
        ImGui::Text("LaTeX Preview");
        ImGui::SetNextItemWidth(input_width_);
        if (UIInput::SliderInt("Font size", &latex_font_size_, 10, 36)) {
          latex_font_size_ = std::max(8, latex_font_size_);
          prefs_changed_ = true;
          pde_preview_.dirty = true;
          shape_preview_.dirty = true;
          bc_left_preview_.dirty = true;
          bc_right_preview_.dirty = true;
          bc_bottom_preview_.dirty = true;
          bc_top_preview_.dirty = true;
          bc_front_preview_.dirty = true;
          bc_back_preview_.dirty = true;
        }
      } else {
        DrawUnknownComponentPlaceholder(id.c_str());
      }
      if (i + 1 < component_list.size()) {
        ImGui::Spacing();
      }
    }
  };

  panel_registry_["benchmark"] = [this](const PanelConfig& panel, bool) {
    BenchmarkPanelState benchmark_state{
      benchmark_config_,
      solver_manager_.IsRunning(),
      [this]() {
        pde_text_ = benchmark_config_.pde;
        bound_xmin_ = benchmark_config_.xmin;
        bound_xmax_ = benchmark_config_.xmax;
        bound_ymin_ = benchmark_config_.ymin;
        bound_ymax_ = benchmark_config_.ymax;
        bound_zmin_ = 0.0;
        bound_zmax_ = 1.0;
        grid_nx_ = benchmark_config_.nx;
        grid_ny_ = benchmark_config_.ny;
        grid_nz_ = 64;
        domain_mode_ = 0;
        domain_shape_.clear();
        domain_shape_file_.clear();
        domain_shape_mask_path_.clear();
        shape_mask_ = ShapeMask();
        shape_mask_threshold_ = 0.0;
        shape_mask_invert_ = false;
        shape_transform_ = ShapeTransform();
        coord_mode_ = CoordMode::kCartesian2D;
        viewer_.SetViewMode(ViewModeForCoord(coord_mode_));
        SetBoundaryFromSpec(benchmark_config_.bc, &bc_left_);
        SetBoundaryFromSpec(benchmark_config_.bc, &bc_right_);
        SetBoundaryFromSpec(benchmark_config_.bc, &bc_bottom_);
        SetBoundaryFromSpec(benchmark_config_.bc, &bc_top_);
        SetBoundaryFromSpec("dirichlet:0", &bc_front_);
        SetBoundaryFromSpec("dirichlet:0", &bc_back_);
        output_path_ = benchmark_config_.output_dir;
        input_dir_ = benchmark_config_.output_dir;
        backend_index_ = BackendToIndex(BackendKind::Auto);
        time_start_ = benchmark_config_.t_start;
        time_end_ = benchmark_config_.t_end;
        time_frames_ = benchmark_config_.frames;
        AddLog(shared_state_, state_mutex_, "benchmark: settings loaded");
        UpdateApplicationState();
      },
      [this]() {
        SyncStateFromApplicationState();
        SolveHandlerState solve_state{
          pde_text_, coord_mode_,
          bound_xmin_, bound_xmax_, bound_ymin_, bound_ymax_, bound_zmin_, bound_zmax_,
          grid_nx_, grid_ny_, grid_nz_, domain_mode_, domain_shape_,
          domain_shape_file_, domain_shape_mask_path_, shape_mask_,
          shape_mask_threshold_, shape_mask_invert_, shape_transform_,
          bc_left_, bc_right_, bc_bottom_, bc_top_, bc_front_, bc_back_,
          backend_index_, method_index_, sor_omega_, gmres_restart_,
          solver_max_iter_, solver_tol_, solver_residual_interval_,
          solver_mg_pre_smooth_, solver_mg_post_smooth_, solver_mg_coarse_iters_, solver_mg_max_levels_,
          thread_count_,
          metal_reduce_interval_, metal_tg_x_, metal_tg_y_,
          time_start_, time_end_, time_frames_, output_path_,
          shared_state_, state_mutex_, viewer_,
          solver_manager_.ThreadPtr(),
          solver_manager_.CancelFlag(),
          solver_manager_.MakeReportStatusCallback(),
          solver_manager_.MakeStartSolverCallback()
        };
        solver_manager_.LaunchSolve(solve_state);
      }
    };
    RenderBenchmarkPanel(benchmark_state, panel.components);
  };

  panel_registry_["ui_config"] = [this](const PanelConfig& panel, bool) {
    const std::vector<std::string> default_components = {"ui_config_editor"};
    const std::vector<std::string>& component_list =
        panel.components.empty() ? default_components : panel.components;

    for (size_t i = 0; i < component_list.size(); ++i) {
      const std::string& id = component_list[i];
      if (id == "ui_config_editor") {
        ui_config_state_.config_manager = &ui_config_mgr_;
        ui_config_state_.config_file_path = ui_config_path_;
        ui_config_state_.font_dir = ui_font_dir_;
        ui_config_state_.input_width = input_width_;
        ui_config_state_.on_apply_settings = [this]() {
          ForceApplyUISettings();
        };
        RenderUIConfigPanel(ui_config_state_);
        ui_config_path_ = ui_config_state_.config_file_path;
      } else {
        DrawUnknownComponentPlaceholder(id.c_str());
      }

      if (i + 1 < component_list.size()) {
        ImGui::Spacing();
      }
    }
  };

  // === New Panels ===

  panel_registry_["initial_conditions"] = [this](const PanelConfig& panel, bool) {
    InitialConditionsPanelState state{
      initial_condition_expr_,
      IsTimeDependent(),
      input_width_
    };
    RenderInitialConditionsPanel(state, panel.components);
  };

  panel_registry_["preset_manager"] = [this](const PanelConfig& panel, bool) {
    PresetManagerPanelState state{
      input_width_,
      preset_directory_
    };
    RenderPresetManagerPanel(state, panel.components);
  };

  panel_registry_["source_term"] = [this](const PanelConfig& panel, bool) {
    SourceTermPanelState state{
      source_term_expr_,
      input_width_
    };
    RenderSourceTermPanel(state, panel.components);
  };

  panel_registry_["material_properties"] = [this](const PanelConfig& panel, bool) {
    MaterialPropertiesPanelState state{
      input_width_
    };
    RenderMaterialPropertiesPanel(state, panel.components);
  };

  panel_registry_["mesh_preview"] = [this](const PanelConfig& panel, bool) {
    std::lock_guard<std::mutex> lock(state_mutex_);
    MeshPreviewPanelState state{
      shared_state_.current_domain,
      input_width_
    };
    RenderMeshPreviewPanel(state, panel.components);
  };

  panel_registry_["parameter_sweep"] = [this](const PanelConfig& panel, bool) {
    ParameterSweepPanelState state{
      input_width_,
      pde_text_,
      coord_mode_,
      bound_xmin_,
      bound_xmax_,
      bound_ymin_,
      bound_ymax_,
      bound_zmin_,
      bound_zmax_,
      grid_nx_,
      grid_ny_,
      grid_nz_,
      domain_mode_,
      domain_shape_,
      domain_shape_file_,
      domain_shape_mask_path_,
      shape_mask_,
      shape_mask_threshold_,
      shape_mask_invert_,
      shape_transform_,
      bc_left_,
      bc_right_,
      bc_bottom_,
      bc_top_,
      bc_front_,
      bc_back_,
      backend_index_,
      method_index_,
      sor_omega_,
      gmres_restart_,
      solver_max_iter_,
      solver_tol_,
      solver_residual_interval_,
      solver_mg_pre_smooth_,
      solver_mg_post_smooth_,
      solver_mg_coarse_iters_,
      solver_mg_max_levels_,
      thread_count_,
      metal_reduce_interval_,
      metal_tg_x_,
      metal_tg_y_,
      time_start_,
      time_end_,
      time_frames_,
      output_path_,
      shared_state_,
      state_mutex_
    };
    RenderParameterSweepPanel(state, panel.components);
  };

  panel_registry_["convergence_panel"] = [this](const PanelConfig& panel, bool) {
    ConvergencePanelState state{
      state_mutex_,
      shared_state_.residual_l2,
      shared_state_.residual_linf,
      input_width_
    };
    RenderConvergencePanel(state, panel.components);
  };

  panel_registry_["point_probe"] = [this](const PanelConfig& panel, bool) {
    PointProbePanelState state{
      state_mutex_,
      shared_state_.current_grid,
      shared_state_.derived_fields,
      shared_state_.has_derived_fields,
      shared_state_.current_domain,
      input_width_
    };
    RenderPointProbePanel(state, panel.components);
  };

  panel_registry_["statistics_panel"] = [this](const PanelConfig& panel, bool) {
    StatisticsPanelState state{
      state_mutex_,
      shared_state_.current_grid,
      shared_state_.derived_fields,
      shared_state_.has_derived_fields,
      shared_state_.current_domain,
      input_width_
    };
    RenderStatisticsPanel(state, panel.components);
  };

  panel_registry_["animation_export"] = [this](const PanelConfig& panel, bool) {
    AnimationExportPanelState state{
      frame_paths_,
      frame_times_,
      input_width_
    };
    RenderAnimationExportPanel(state, panel.components);
  };

  panel_registry_["testing"] = [this](const PanelConfig& panel, bool) {
    TestingPanelState state{
      input_width_
    };
    RenderTestingPanel(state);
  };
}

void Application::SyncLayoutFromUIConfig() {
  const UIConfig& ui_cfg = ui_config_mgr_.GetConfig();
  input_width_ = ui_cfg.theme.input_width;
  left_panel_min_width_ = ui_cfg.left_panel_min_width;
  left_panel_max_width_ = ui_cfg.left_panel_max_width;
  splitter_width_ = ui_cfg.splitter_width;
}

bool Application::ApplyUIFont(const std::filesystem::path& font_path, float font_size, std::string* warning) {
  ImGuiIO& io = ImGui::GetIO();

  io.Fonts->Clear();

  ImFont* font = nullptr;
  ImFontConfig config;
  config.OversampleH = 2;
  config.OversampleV = 2;
  config.PixelSnapH = true;
  config.SizePixels = font_size;

  ImVector<ImWchar> ranges;
  BuildUIFontRanges(io, &ranges);

  if (!font_path.empty()) {
    if (!std::filesystem::exists(font_path)) {
      if (warning) {
        *warning = "UI font file not found; using default font.";
      }
    } else {
      font = io.Fonts->AddFontFromFileTTF(font_path.string().c_str(), font_size, &config, ranges.Data);
      if (!font && warning) {
        *warning = "Failed to load UI font; using default font.";
      }
    }
  }

  if (!font) {
    font = io.Fonts->AddFontDefault(&config);
  }

  // Try to merge a symbol font for media control icons
  // Look for common symbol fonts on macOS
  static const char* symbol_font_paths[] = {
    "/System/Library/Fonts/Apple Symbols.ttf",
    "/System/Library/Fonts/Supplemental/Apple Symbols.ttf",
    "/Library/Fonts/NotoSansSymbols2-Regular.ttf",
    "/usr/share/fonts/truetype/noto/NotoSansSymbols2-Regular.ttf",
    nullptr
  };

  // Symbol ranges for media controls
  static const ImWchar symbol_ranges[] = {
    0x2300, 0x23FF,  // Miscellaneous Technical (⏮ ⏪ ⏸ ⏩ ⏭)
    0x25A0, 0x25FF,  // Geometric Shapes (◀ ▶)
    0
  };

  ImFontConfig merge_config;
  merge_config.MergeMode = true;
  merge_config.OversampleH = 2;
  merge_config.OversampleV = 2;
  merge_config.PixelSnapH = true;

  for (const char** path = symbol_font_paths; *path != nullptr; ++path) {
    if (std::filesystem::exists(*path)) {
      io.Fonts->AddFontFromFileTTF(*path, font_size, &merge_config, symbol_ranges);
      break;
    }
  }

  io.FontDefault = font;
  ImGuiStyle& style = ImGui::GetStyle();
  style.FontSizeBase = font_size;
  style._NextFrameFontSizeBase = font_size;
  return font != nullptr;
}

void Application::SyncFontFromUIConfig() {
  const UIConfig& ui_cfg = ui_config_mgr_.GetConfig();
  const float desired_size = std::max(kMinUIFontSize, std::min(kMaxUIFontSize, ui_cfg.theme.font_size));
  const std::filesystem::path resolved_path = ResolveUIFontPath(ui_cfg.theme.font_path, ui_font_dir_);

  if (resolved_path == ui_font_path_ && std::abs(desired_size - ui_font_size_) < 0.01f) {
    return;
  }

  std::string warning;
  ApplyUIFont(resolved_path, desired_size, &warning);
  ui_font_path_ = resolved_path;
  ui_font_size_ = desired_size;

  // Rebuild font atlas
  ImGui_ImplOpenGL3_DestroyDeviceObjects();
  ImGui_ImplOpenGL3_CreateDeviceObjects();

  if (!warning.empty()) {
    UIToast::Show(UIToast::Type::Warning, warning);
  }
}

void Application::ForceApplyUISettings() {
  // Force sync of layout settings (safe to do mid-frame)
  SyncLayoutFromUIConfig();

  // Defer font rebuild to the start of the next frame.
  // Rebuilding fonts mid-frame causes a crash because the current frame's
  // font pointer becomes invalid while ImGui is still rendering.
  pending_font_rebuild_ = true;
}

void Application::SavePreferencesIfNeeded() {
  if (!prefs_changed_) {
    return;
  }

  Preferences to_save;
  to_save.metal_reduce_interval = pref_metal_reduce_interval_;
  to_save.metal_tg_x = pref_metal_tg_x_;
  to_save.metal_tg_y = pref_metal_tg_y_;
  to_save.latex_font_size = latex_font_size_;
  to_save.method_index = pref_method_index_;
  to_save.sor_omega = pref_sor_omega_;
  to_save.gmres_restart = pref_gmres_restart_;
  to_save.colors = prefs_.colors;
  to_save.ui_section_open = prefs_.ui_section_open;

  std::string error;
  if (!SavePreferences(prefs_path_, to_save, &error)) {
    AddLog(shared_state_, state_mutex_, "prefs: " + error);
    UIToast::Show(UIToast::Type::Error, "Failed to save preferences");
  } else {
    AddLog(shared_state_, state_mutex_, "prefs: saved");
  }
  prefs_changed_ = false;
}

void Application::RenderMenuBar() {
  if (NativeMenu::Instance().IsInstalled()) {
    return;
  }
  if (!ImGui::BeginMenuBar()) {
    return;
  }

  auto menu_item = [&](const char* label, AppAction action, const char* shortcut = nullptr) {
    const bool enabled = CanExecuteAction(action);
    const bool checked = IsActionChecked(action);
    if (ImGui::MenuItem(label, shortcut, checked, enabled)) {
      ExecuteAction(action);
    }
  };

#ifdef __APPLE__
  const char* undo_shortcut = "Cmd+Z";
  const char* redo_shortcut = "Cmd+Shift+Z";
  const char* solve_shortcut = "Cmd+S";
  const char* load_shortcut = "Cmd+L";
  const char* reset_view_shortcut = "Cmd+R";
  const char* main_tab_shortcut = "Cmd+1";
  const char* inspect_tab_shortcut = "Cmd+2";
  const char* prefs_tab_shortcut = "Cmd+3";
  const char* help_shortcut = "F1";
#else
  const char* undo_shortcut = "Ctrl+Z";
  const char* redo_shortcut = "Ctrl+Shift+Z";
  const char* solve_shortcut = "Ctrl+S";
  const char* load_shortcut = "Ctrl+L";
  const char* reset_view_shortcut = "Ctrl+R";
  const char* main_tab_shortcut = "Ctrl+1";
  const char* inspect_tab_shortcut = "Ctrl+2";
  const char* prefs_tab_shortcut = "Ctrl+3";
  const char* help_shortcut = "F1";
#endif

  if (ImGui::BeginMenu("File")) {
    menu_item("New Session", AppAction::kNewSession);
    menu_item("Load Latest Result", AppAction::kLoadLatest, load_shortcut);
    menu_item("Export Image", AppAction::kExportImage);
    ImGui::Separator();
    menu_item("Load Run Config", AppAction::kLoadRunConfig);
    menu_item("Save Run Config", AppAction::kSaveRunConfig);
    ImGui::Separator();
    menu_item("Load UI Config", AppAction::kLoadUIConfig);
    menu_item("Save UI Config", AppAction::kSaveUIConfig);
    ImGui::Separator();
    menu_item("Quit", AppAction::kQuit);
    ImGui::EndMenu();
  }

  if (ImGui::BeginMenu("Edit")) {
    menu_item("Undo", AppAction::kUndo, undo_shortcut);
    menu_item("Redo", AppAction::kRedo, redo_shortcut);
    ImGui::Separator();
    menu_item("Copy PDE", AppAction::kCopyPDE);
    menu_item("Paste PDE", AppAction::kPastePDE);
    ImGui::EndMenu();
  }

  if (ImGui::BeginMenu("View")) {
    menu_item("Reset View", AppAction::kResetView, reset_view_shortcut);
    menu_item("Orthographic Projection", AppAction::kToggleOrtho);
    menu_item("Show Domain Grid", AppAction::kToggleGrid);
    menu_item("Lock Z Domain", AppAction::kToggleZLock);
    ImGui::EndMenu();
  }

  if (ImGui::BeginMenu("Go")) {
    menu_item("Main Tab", AppAction::kGoMainTab, main_tab_shortcut);
    menu_item("Inspect Tab", AppAction::kGoInspectTab, inspect_tab_shortcut);
    menu_item("Preferences Tab", AppAction::kGoPreferencesTab, prefs_tab_shortcut);
    ImGui::EndMenu();
  }

  if (ImGui::BeginMenu("Tools")) {
    menu_item("Solve PDE", AppAction::kSolve, solve_shortcut);
    menu_item("Stop Solver", AppAction::kStop, "Esc");
    ImGui::Separator();
    menu_item("Benchmarks", AppAction::kOpenBenchmarks);
    menu_item("Comparison Tools", AppAction::kOpenComparisonTools);
    menu_item("Advanced Inspection", AppAction::kOpenAdvancedInspection);
    menu_item("UI Configuration", AppAction::kOpenUIConfigPanel);
    menu_item("Validate UI Config", AppAction::kValidateUIConfig);
    ImGui::EndMenu();
  }

  if (ImGui::BeginMenu("Window")) {
    menu_item("Show Left Panel", AppAction::kToggleLeftPanel);
    menu_item("Reset Layout", AppAction::kResetLayout);
    ImGui::Separator();
    menu_item("Enable Docking UI", AppAction::kToggleDockingUI);
    const bool layout_enabled = CanExecuteAction(AppAction::kLayoutDefault);
    if (ImGui::BeginMenu("Layout Presets", layout_enabled)) {
      menu_item("Default Layout", AppAction::kLayoutDefault);
      menu_item("Inspection Layout", AppAction::kLayoutInspection);
      menu_item("Dual Viewer", AppAction::kLayoutDualViewer);
      menu_item("Minimal", AppAction::kLayoutMinimal);
      menu_item("Full Configuration", AppAction::kLayoutFull);
      ImGui::EndMenu();
    }
    if (ImGui::BeginMenu("Load Layout", layout_enabled)) {
      auto names = docking_ctx_->GetSavedLayoutNames(exec_path_.parent_path());
      if (names.empty()) {
        ImGui::TextDisabled("No saved layouts");
      } else {
        for (const auto& name : names) {
          if (ImGui::MenuItem(name.c_str())) {
            docking_ctx_->LoadLayout(name, exec_path_.parent_path());
          }
        }
      }
      ImGui::EndMenu();
    }
    if (ImGui::MenuItem("Save Layout...", nullptr, false, layout_enabled)) {
      // TODO: Open save dialog.
    }
    ImGui::EndMenu();
  }

  if (ImGui::BeginMenu("Help")) {
    menu_item("Help Search", AppAction::kHelpSearch, help_shortcut);
    menu_item("About PDE Solver", AppAction::kAbout);
    ImGui::EndMenu();
  }

  ImGui::EndMenuBar();
}

void Application::RenderAboutModal() {
  if (show_about_modal_) {
    ImGui::OpenPopup("About PDE Solver");
    show_about_modal_ = false;
  }
  if (ImGui::BeginPopupModal("About PDE Solver", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
    ImGui::Text("PDE Solver");
    ImGui::Separator();
    ImGui::Text("Interactive PDE solver with inspection tools.");
    ImGui::Text("UI: ImGui + OpenGL");
    ImGui::Spacing();
    if (ImGui::Button("OK", ImVec2(120.0f, 0.0f))) {
      ImGui::CloseCurrentPopup();
    }
    ImGui::EndPopup();
  }
}

bool Application::CanExecuteAction(AppAction action) const {
  const bool running = solver_manager_.IsRunning();
  switch (action) {
    case AppAction::kSolve:
      return !running;
    case AppAction::kStop:
      return running;
    case AppAction::kLoadLatest:
    case AppAction::kNewSession:
    case AppAction::kLoadRunConfig:
    case AppAction::kSaveRunConfig:
      return !running;
    case AppAction::kUndo:
      return cmd_history_.CanUndo() && !running;
    case AppAction::kRedo:
      return cmd_history_.CanRedo() && !running;
    case AppAction::kGoMainTab:
    case AppAction::kGoInspectTab:
    case AppAction::kGoPreferencesTab:
      return !use_docking_ui_;
    case AppAction::kToggleLeftPanel:
      return !use_docking_ui_ && (active_tab_ == "Main" || active_tab_ == "Inspect");
    case AppAction::kLoadUIConfig:
    case AppAction::kSaveUIConfig:
      return !ui_config_path_.empty();
    case AppAction::kCopyPDE:
      return !pde_text_.empty();
    case AppAction::kToggleDockingUI:
      return docking_ctx_ != nullptr;
    case AppAction::kLayoutDefault:
    case AppAction::kLayoutInspection:
    case AppAction::kLayoutDualViewer:
    case AppAction::kLayoutMinimal:
    case AppAction::kLayoutFull:
      return docking_ctx_ != nullptr && use_docking_ui_;
    case AppAction::kToggleZLock: {
      const bool view_is_surface =
          coord_mode_ == CoordMode::kSphericalSurface || coord_mode_ == CoordMode::kToroidalSurface;
      const bool view_is_volume =
          coord_mode_ == CoordMode::kSphericalVolume || coord_mode_ == CoordMode::kToroidalVolume ||
          coord_mode_ == CoordMode::kCartesian3D || coord_mode_ == CoordMode::kCylindricalVolume;
      return !(view_is_surface || view_is_volume);
    }
    default:
      return true;
  }
}

bool Application::IsActionChecked(AppAction action) const {
  switch (action) {
    case AppAction::kToggleOrtho:
      return use_ortho_;
    case AppAction::kToggleGrid:
      return grid_enabled_;
    case AppAction::kToggleZLock:
      return z_domain_locked_;
    case AppAction::kToggleLeftPanel:
      if (active_tab_ == "Inspect") {
        return !inspect_left_collapsed_;
      }
      if (active_tab_ == "Main") {
        return !left_collapsed_;
      }
      return false;
    case AppAction::kToggleDockingUI:
      return use_docking_ui_;
    default:
      return false;
  }
}

void Application::ExecuteAction(AppAction action) {
  if (!CanExecuteAction(action)) {
    return;
  }

  switch (action) {
    case AppAction::kSolve: {
      UpdateApplicationState();
      SolveHandlerState solve_state{
        pde_text_, coord_mode_,
        bound_xmin_, bound_xmax_, bound_ymin_, bound_ymax_, bound_zmin_, bound_zmax_,
        grid_nx_, grid_ny_, grid_nz_, domain_mode_, domain_shape_,
        domain_shape_file_, domain_shape_mask_path_, shape_mask_,
        shape_mask_threshold_, shape_mask_invert_, shape_transform_,
        bc_left_, bc_right_, bc_bottom_, bc_top_, bc_front_, bc_back_,
        backend_index_, method_index_, sor_omega_, gmres_restart_,
        solver_max_iter_, solver_tol_, solver_residual_interval_,
        solver_mg_pre_smooth_, solver_mg_post_smooth_, solver_mg_coarse_iters_, solver_mg_max_levels_,
        thread_count_,
        metal_reduce_interval_, metal_tg_x_, metal_tg_y_,
        time_start_, time_end_, time_frames_, output_path_,
        shared_state_, state_mutex_, viewer_,
        solver_manager_.ThreadPtr(),
        solver_manager_.CancelFlag(),
        solver_manager_.MakeReportStatusCallback(),
        solver_manager_.MakeStartSolverCallback()
      };
      solver_manager_.LaunchSolve(solve_state);
      break;
    }
    case AppAction::kStop:
      solver_manager_.RequestStop();
      break;
    case AppAction::kLoadLatest:
      HandleFileLoading();
      break;
    case AppAction::kResetView:
      viewer_.FitToView();
      point_scale_ = 1.0f;
      viewer_.SetPointScale(point_scale_);
      break;
    case AppAction::kUndo:
      cmd_history_.Undo();
      break;
    case AppAction::kRedo:
      cmd_history_.Redo();
      break;
    case AppAction::kHelpSearch:
      UIHelp::OpenSearch();
      break;
    case AppAction::kNewSession:
      ResetSessionState();
      break;
    case AppAction::kLoadUIConfig: {
      bool loaded = false;
      std::filesystem::path config_path = ui_config_path_;
      if (ui_config_mgr_.IsFeatureEnabled("file_dialogs")) {
        auto selected = FileDialog::PickFile(
            "Open UI Config",
            config_path,
            "JSON Files",
            {"json"});
        if (!selected) {
          break;
        }
        config_path = *selected;
      }
      if (!config_path.empty() && std::filesystem::exists(config_path)) {
        loaded = ui_config_mgr_.LoadFromFile(config_path.string());
      }
      if (loaded) {
        ui_config_path_ = config_path;
        ForceApplyUISettings();
        UIToast::Show(UIToast::Type::Success, "UI config loaded");
      } else {
        UIToast::Show(UIToast::Type::Error, "Failed to load UI config");
      }
      break;
    }
    case AppAction::kLoadRunConfig: {
      bool loaded = false;
      std::filesystem::path config_path = run_config_path_;
      if (ui_config_mgr_.IsFeatureEnabled("file_dialogs")) {
        auto selected = FileDialog::PickFile(
            "Open Run Config",
            config_path,
            "JSON Files",
            {"json"});
        if (!selected) {
          break;
        }
        config_path = *selected;
      }
      std::string error;
      if (!config_path.empty() && std::filesystem::exists(config_path)) {
        loaded = LoadRunConfigFile(config_path, &error);
      }
      if (loaded) {
        run_config_path_ = config_path;
        UIToast::Show(UIToast::Type::Success, "Run config loaded");
      } else {
        UIToast::Show(UIToast::Type::Error, error.empty() ? "Failed to load run config" : error);
      }
      break;
    }
    case AppAction::kSaveUIConfig: {
      std::filesystem::path config_path = ui_config_path_;
      if (ui_config_mgr_.IsFeatureEnabled("file_dialogs")) {
        auto selected = FileDialog::SaveFile(
            "Save UI Config",
            config_path,
            "ui_config.json",
            "JSON Files",
            {"json"});
        if (!selected) {
          break;
        }
        config_path = *selected;
      }
      if (!config_path.empty() && ui_config_mgr_.SaveToFile(config_path.string())) {
        ui_config_path_ = config_path;
        UIToast::Show(UIToast::Type::Success, "UI config saved");
      } else {
        UIToast::Show(UIToast::Type::Error, "Failed to save UI config");
      }
      break;
    }
    case AppAction::kSaveRunConfig: {
      std::filesystem::path config_path = run_config_path_;
      if (ui_config_mgr_.IsFeatureEnabled("file_dialogs")) {
        auto selected = FileDialog::SaveFile(
            "Save Run Config",
            config_path,
            "run_config.json",
            "JSON Files",
            {"json"});
        if (!selected) {
          break;
        }
        config_path = *selected;
      }
      std::string error;
      if (!config_path.empty() && SaveRunConfigFile(config_path, &error)) {
        run_config_path_ = config_path;
        UIToast::Show(UIToast::Type::Success, "Run config saved");
      } else {
        UIToast::Show(UIToast::Type::Error, error.empty() ? "Failed to save run config" : error);
      }
      break;
    }
    case AppAction::kExportImage:
      OpenPanelInTab("Inspect", "export_panel");
      break;
    case AppAction::kQuit:
      window_manager_.SetShouldClose(true);
      break;
    case AppAction::kCopyPDE:
      ImGui::SetClipboardText(pde_text_.c_str());
      UIToast::Show(UIToast::Type::Success, "PDE copied to clipboard");
      break;
    case AppAction::kPastePDE: {
      const char* clip = ImGui::GetClipboardText();
      if (clip && *clip) {
        pde_text_ = clip;
        UpdateApplicationState();
        UIToast::Show(UIToast::Type::Success, "PDE pasted from clipboard");
      } else {
        UIToast::Show(UIToast::Type::Warning, "Clipboard is empty");
      }
      break;
    }
    case AppAction::kToggleOrtho:
      use_ortho_ = !use_ortho_;
      viewer_.SetOrthographic(use_ortho_);
      viewer_.FitToView();
      point_scale_ = 1.0f;
      viewer_.SetPointScale(point_scale_);
      break;
    case AppAction::kToggleGrid:
      grid_enabled_ = !grid_enabled_;
      viewer_.SetGridEnabled(grid_enabled_);
      break;
    case AppAction::kToggleZLock: {
      z_domain_locked_ = !z_domain_locked_;
      viewer_.SetZDomain(z_domain_locked_, z_domain_min_, z_domain_max_);
      break;
    }
    case AppAction::kToggleLeftPanel:
      if (active_tab_ == "Main") {
        left_collapsed_ = !left_collapsed_;
        if (!left_collapsed_) {
          left_panel_width_ = std::max(left_panel_width_, left_panel_min_width_);
        }
      } else if (active_tab_ == "Inspect") {
        inspect_left_collapsed_ = !inspect_left_collapsed_;
        if (!inspect_left_collapsed_) {
          inspect_left_panel_width_ = std::max(inspect_left_panel_width_, left_panel_min_width_);
        }
      }
      break;
    case AppAction::kGoMainTab:
      RequestTab("Main");
      break;
    case AppAction::kGoInspectTab:
      RequestTab("Inspect");
      break;
    case AppAction::kGoPreferencesTab:
      RequestTab("Preferences");
      break;
    case AppAction::kOpenBenchmarks:
      OpenPanelInTab("Preferences", "benchmark");
      break;
    case AppAction::kOpenComparisonTools:
      OpenPanelInTab("Inspect", "comparison_panel");
      break;
    case AppAction::kOpenAdvancedInspection:
      OpenPanelInTab("Inspect", "advanced_panel");
      break;
    case AppAction::kOpenUIConfigPanel:
      OpenPanelInTab("Preferences", "ui_config");
      break;
    case AppAction::kValidateUIConfig: {
      OpenPanelInTab("Preferences", "ui_config");
      ui_config_state_.show_validation = true;
      ValidationResult validation = ui_config_mgr_.Validate();
      if (validation.valid) {
        UIToast::Show(UIToast::Type::Success, "UI config is valid");
      } else {
        UIToast::Show(UIToast::Type::Error, "UI config has errors");
      }
      break;
    }
    case AppAction::kToggleDockingUI:
      if (docking_ctx_) {
        use_docking_ui_ = !use_docking_ui_;
      }
      break;
    case AppAction::kLayoutDefault:
      if (docking_ctx_) {
        docking_ctx_->SetRoot(PresetLayouts::CreateDefault());
      }
      break;
    case AppAction::kLayoutInspection:
      if (docking_ctx_) {
        docking_ctx_->SetRoot(PresetLayouts::CreateInspection());
      }
      break;
    case AppAction::kLayoutDualViewer:
      if (docking_ctx_) {
        docking_ctx_->SetRoot(PresetLayouts::CreateDualViewer());
      }
      break;
    case AppAction::kLayoutMinimal:
      if (docking_ctx_) {
        docking_ctx_->SetRoot(PresetLayouts::CreateMinimal());
      }
      break;
    case AppAction::kLayoutFull:
      if (docking_ctx_) {
        docking_ctx_->SetRoot(PresetLayouts::CreateFullConfiguration());
      }
      break;
    case AppAction::kResetLayout:
      ResetLayoutState();
      break;
    case AppAction::kAbout:
      show_about_modal_ = true;
      break;
  }
}

void Application::OpenPanelInTab(const std::string& tab_name, const std::string& panel_id) {
  RequestTab(tab_name);
  const char* tab_key = nullptr;
  if (tab_name == "Main") {
    tab_key = "main";
    left_collapsed_ = false;
    left_panel_width_ = std::max(left_panel_width_, left_panel_min_width_);
  } else if (tab_name == "Inspect") {
    tab_key = "inspect";
    inspect_left_collapsed_ = false;
    inspect_left_panel_width_ = std::max(inspect_left_panel_width_, left_panel_min_width_);
  } else if (tab_name == "Preferences") {
    tab_key = "prefs";
  }
  if (tab_key && !panel_id.empty()) {
    prefs_.ui_section_open[std::string(tab_key) + "." + panel_id] = true;
    prefs_changed_ = true;
  }
}

void Application::RequestTab(const std::string& tab_name) {
  requested_tab_ = tab_name;
}

void Application::ResetSessionState() {
  ApplicationState defaults;
  const ApplicationState::State state = defaults.GetState();
  SolverConfig solver_defaults;

  pde_text_ = state.pde_text;
  bound_xmin_ = state.bound_xmin;
  bound_xmax_ = state.bound_xmax;
  bound_ymin_ = state.bound_ymin;
  bound_ymax_ = state.bound_ymax;
  bound_zmin_ = state.bound_zmin;
  bound_zmax_ = state.bound_zmax;
  grid_nx_ = state.grid_nx;
  grid_ny_ = state.grid_ny;
  grid_nz_ = state.grid_nz;
  domain_mode_ = state.domain_mode;
  domain_shape_ = state.domain_shape;
  domain_shape_file_ = state.domain_shape_file;
  domain_shape_mask_path_ = state.domain_shape_mask;
  shape_mask_threshold_ = state.domain_shape_mask_threshold;
  shape_mask_invert_ = state.domain_shape_mask_invert;
  shape_transform_ = state.shape_transform;
  shape_mask_ = ShapeMask();
  coord_mode_ = state.coord_mode;
  torus_major_ = state.torus_major;
  torus_minor_ = state.torus_minor;
  bc_left_ = state.bc_left;
  bc_right_ = state.bc_right;
  bc_bottom_ = state.bc_bottom;
  bc_top_ = state.bc_top;
  bc_front_ = state.bc_front;
  bc_back_ = state.bc_back;
  backend_index_ = state.backend_index;
  method_index_ = state.method_index;
  sor_omega_ = state.sor_omega;
  gmres_restart_ = state.gmres_restart;
  solver_max_iter_ = solver_defaults.max_iter;
  solver_tol_ = solver_defaults.tol;
  solver_residual_interval_ = solver_defaults.residual_interval;
  solver_mg_pre_smooth_ = solver_defaults.mg_pre_smooth;
  solver_mg_post_smooth_ = solver_defaults.mg_post_smooth;
  solver_mg_coarse_iters_ = solver_defaults.mg_coarse_iters;
  solver_mg_max_levels_ = solver_defaults.mg_max_levels;
  thread_count_ = state.thread_count;
  max_threads_ = state.max_threads;
  metal_reduce_interval_ = state.metal_reduce_interval;
  metal_tg_x_ = state.metal_tg_x;
  metal_tg_y_ = state.metal_tg_y;
  time_start_ = state.time_start;
  time_end_ = state.time_end;
  time_frames_ = state.time_frames;
  output_path_ = state.output_path;
  input_dir_ = state.input_dir;
  slice_enabled_ = state.slice_enabled;
  slice_axis_ = state.slice_axis;
  slice_value_ = state.slice_value;
  slice_thickness_ = state.slice_thickness;
  iso_enabled_ = state.iso_enabled;
  iso_value_ = state.iso_value;
  iso_band_ = state.iso_band;
  field_type_index_ = state.field_type_index;

  frame_paths_.clear();
  frame_times_.clear();
  frame_index_ = 0;
  last_loaded_frame_ = -1;
  playing_ = false;
  play_accum_ = 0.0;

  viewer_.SetViewMode(ViewModeForCoord(coord_mode_));
  viewer_.SetTorusRadii(static_cast<float>(torus_major_), static_cast<float>(torus_minor_));
  viewer_.SetData(state.domain, {});
  viewer_.ClearDerivedFields();
  viewer_.SetFieldType(GlViewer::FieldType::Solution);
  viewer_.FitToView();
  point_scale_ = 1.0f;
  viewer_.SetPointScale(point_scale_);

  {
    std::lock_guard<std::mutex> lock(state_mutex_);
    shared_state_.current_grid.clear();
    shared_state_.has_derived_fields = false;
    shared_state_.derived_fields = {};
    shared_state_.current_domain = state.domain;
    shared_state_.current_pde = {};
  }

  UpdateApplicationState();
  UIToast::Show(UIToast::Type::Info, "Session reset");
}

void Application::ResetLayoutState() {
  left_collapsed_ = false;
  inspect_left_collapsed_ = false;
  left_panel_width_ = std::max(left_panel_min_width_, std::min(400.0f, left_panel_max_width_));
  inspect_left_panel_width_ = std::max(left_panel_min_width_, std::min(400.0f, left_panel_max_width_));
}

void Application::UpdateApplicationState() {
  ApplicationState::State& s = app_state_.GetMutableState();
  s.pde_text = pde_text_;
  s.bound_xmin = bound_xmin_;
  s.bound_xmax = bound_xmax_;
  s.bound_ymin = bound_ymin_;
  s.bound_ymax = bound_ymax_;
  s.bound_zmin = bound_zmin_;
  s.bound_zmax = bound_zmax_;
  s.grid_nx = grid_nx_;
  s.grid_ny = grid_ny_;
  s.grid_nz = grid_nz_;
  s.domain_mode = domain_mode_;
  s.domain_shape = domain_shape_;
  s.domain_shape_file = domain_shape_file_;
  s.domain_shape_mask = domain_shape_mask_path_;
  s.domain_shape_mask_threshold = shape_mask_threshold_;
  s.domain_shape_mask_invert = shape_mask_invert_;
  s.shape_transform = shape_transform_;
  s.coord_mode = coord_mode_;
  s.torus_major = torus_major_;
  s.torus_minor = torus_minor_;
  s.bc_left = bc_left_;
  s.bc_right = bc_right_;
  s.bc_bottom = bc_bottom_;
  s.bc_top = bc_top_;
  s.bc_front = bc_front_;
  s.bc_back = bc_back_;
  s.backend_index = backend_index_;
  s.method_index = method_index_;
  s.sor_omega = sor_omega_;
  s.gmres_restart = gmres_restart_;
  s.thread_count = thread_count_;
  s.max_threads = max_threads_;
  s.metal_reduce_interval = metal_reduce_interval_;
  s.metal_tg_x = metal_tg_x_;
  s.metal_tg_y = metal_tg_y_;
  s.time_start = time_start_;
  s.time_end = time_end_;
  s.time_frames = time_frames_;
  s.output_path = output_path_;
  s.input_dir = input_dir_;
  s.slice_enabled = slice_enabled_;
  s.slice_axis = slice_axis_;
  s.slice_value = slice_value_;
  s.slice_thickness = slice_thickness_;
  s.iso_enabled = iso_enabled_;
  s.iso_value = iso_value_;
  s.iso_band = iso_band_;
  s.field_type_index = field_type_index_;
  for (const auto& path : frame_paths_) {
    s.frame_paths.push_back(std::filesystem::path(path));
  }
  s.frame_times = frame_times_;
  s.frame_index = frame_index_;
  s.last_loaded_frame = last_loaded_frame_;
  s.playing = playing_;
  s.pref_method_index = pref_method_index_;
  s.pref_sor_omega = pref_sor_omega_;
  s.pref_gmres_restart = pref_gmres_restart_;
  s.pref_metal_reduce_interval = pref_metal_reduce_interval_;
  s.pref_metal_tg_x = pref_metal_tg_x_;
  s.pref_metal_tg_y = pref_metal_tg_y_;
  app_state_.UpdateCoordinateFlags();
}

void Application::SyncStateFromApplicationState() {
  const ApplicationState::State& s = app_state_.GetState();
  pde_text_ = s.pde_text;
  bound_xmin_ = s.bound_xmin;
  bound_xmax_ = s.bound_xmax;
  bound_ymin_ = s.bound_ymin;
  bound_ymax_ = s.bound_ymax;
  bound_zmin_ = s.bound_zmin;
  bound_zmax_ = s.bound_zmax;
  grid_nx_ = s.grid_nx;
  grid_ny_ = s.grid_ny;
  grid_nz_ = s.grid_nz;
  domain_mode_ = s.domain_mode;
  domain_shape_ = s.domain_shape;
  domain_shape_file_ = s.domain_shape_file;
  domain_shape_mask_path_ = s.domain_shape_mask;
  shape_mask_threshold_ = s.domain_shape_mask_threshold;
  shape_mask_invert_ = s.domain_shape_mask_invert;
  shape_transform_ = s.shape_transform;
  coord_mode_ = s.coord_mode;

  if (domain_shape_.empty() && !domain_shape_file_.empty()) {
    std::string shape_error;
    if (!LoadShapeExpressionFromFile(domain_shape_file_, &domain_shape_, &shape_error)) {
      domain_shape_.clear();
    }
  }
  if (!domain_shape_mask_path_.empty()) {
    std::string mask_error;
    if (!LoadShapeMaskFromVtk(domain_shape_mask_path_, &shape_mask_, &mask_error)) {
      shape_mask_ = ShapeMask();
    }
  } else {
    shape_mask_ = ShapeMask();
  }
  torus_major_ = s.torus_major;
  torus_minor_ = s.torus_minor;
  bc_left_ = s.bc_left;
  bc_right_ = s.bc_right;
  bc_bottom_ = s.bc_bottom;
  bc_top_ = s.bc_top;
  bc_front_ = s.bc_front;
  bc_back_ = s.bc_back;
  backend_index_ = s.backend_index;
  method_index_ = s.method_index;
  sor_omega_ = s.sor_omega;
  gmres_restart_ = s.gmres_restart;
  thread_count_ = s.thread_count;
  max_threads_ = s.max_threads;
  metal_reduce_interval_ = s.metal_reduce_interval;
  metal_tg_x_ = s.metal_tg_x;
  metal_tg_y_ = s.metal_tg_y;
  time_start_ = s.time_start;
  time_end_ = s.time_end;
  time_frames_ = s.time_frames;
  output_path_ = s.output_path;
  input_dir_ = s.input_dir;
  slice_enabled_ = s.slice_enabled;
  slice_axis_ = s.slice_axis;
  slice_value_ = s.slice_value;
  slice_thickness_ = s.slice_thickness;
  iso_enabled_ = s.iso_enabled;
  iso_value_ = s.iso_value;
  iso_band_ = s.iso_band;
  field_type_index_ = s.field_type_index;
  frame_paths_.clear();
  for (const auto& p : s.frame_paths) {
    frame_paths_.push_back(p.string());
  }
  frame_times_ = s.frame_times;
  frame_index_ = s.frame_index;
  last_loaded_frame_ = s.last_loaded_frame;
  playing_ = s.playing;
}

RunConfig Application::BuildRunConfig(std::string* error) const {
  RunConfig config;
  config.pde_latex = pde_text_;
  config.coord_mode = CoordModeToken(coord_mode_);
  config.domain_mode = (domain_mode_ == 1) ? "implicit" : "box";

  const bool use_volume = IsVolumeCoordMode(coord_mode_);
  if (use_volume) {
    config.domain_bounds = FormatBounds3D(bound_xmin_, bound_xmax_, bound_ymin_, bound_ymax_,
                                          bound_zmin_, bound_zmax_);
    config.grid = FormatGrid3D(grid_nx_, grid_ny_, grid_nz_);
  } else {
    config.domain_bounds = FormatBounds(bound_xmin_, bound_xmax_, bound_ymin_, bound_ymax_);
    config.grid = FormatGrid(grid_nx_, grid_ny_);
  }
  config.domain_shape = domain_shape_;
  config.domain_shape_file = domain_shape_file_;
  config.domain_shape_mask = domain_shape_mask_path_;
  config.domain_shape_mask_threshold = shape_mask_threshold_;
  config.domain_shape_mask_invert = shape_mask_invert_;
  config.shape_transform = shape_transform_;

  std::string bc_error;
  if (!BuildBoundarySpec(bc_left_, bc_right_, bc_bottom_, bc_top_, bc_front_, bc_back_,
                         &config.boundary_spec, &bc_error)) {
    if (error) {
      *error = bc_error.empty() ? "invalid boundary specification" : bc_error;
    }
  }

  config.backend = BackendToken(BackendFromIndex(backend_index_));
  config.method = MethodToken(MethodFromIndex(method_index_));

  config.solver.max_iter = solver_max_iter_;
  config.solver.tol = solver_tol_;
  config.solver.residual_interval = solver_residual_interval_;
  config.solver.thread_count = thread_count_;
  config.solver.metal_reduce_interval = metal_reduce_interval_;
  config.solver.metal_tg_x = metal_tg_x_;
  config.solver.metal_tg_y = metal_tg_y_;
  config.solver.sor_omega = sor_omega_;
  config.solver.gmres_restart = gmres_restart_;
  config.solver.mg_pre_smooth = solver_mg_pre_smooth_;
  config.solver.mg_post_smooth = solver_mg_post_smooth_;
  config.solver.mg_coarse_iters = solver_mg_coarse_iters_;
  config.solver.mg_max_levels = solver_mg_max_levels_;

  config.time.enabled = IsTimeDependent();
  config.time.t_start = time_start_;
  config.time.t_end = time_end_;
  config.time.frames = std::max(1, time_frames_);
  if (config.time.frames > 1) {
    config.time.dt =
        (time_end_ - time_start_) / static_cast<double>(std::max(1, config.time.frames - 1));
  }

  config.output_format = "vtk";
  config.output_dir.clear();
  config.output_path.clear();
  if (output_path_.empty()) {
    config.output_dir = "outputs";
  } else {
    std::filesystem::path output_path(output_path_);
    const bool ends_with_slash = !output_path_.empty() && output_path_.back() == '/';
    const bool is_dir = std::filesystem::exists(output_path) &&
                        std::filesystem::is_directory(output_path);
    if (ends_with_slash || is_dir) {
      config.output_dir = output_path_.empty() ? output_path_ : output_path.string();
    } else {
      config.output_path = output_path.string();
    }
    if (output_path.extension() == ".vti") {
      config.output_format = "vti";
    }
  }

  return config;
}

bool Application::ApplyRunConfig(const RunConfig& config, std::string* error) {
  if (!ValidateRunConfig(config, error)) {
    return false;
  }
  Domain domain;
  ParseResult domain_result = ParseDomain(config.domain_bounds, &domain);
  if (!domain_result.ok) {
    if (error) *error = domain_result.error;
    return false;
  }
  ParseResult grid_result = ParseGrid(config.grid, &domain);
  if (!grid_result.ok) {
    if (error) *error = grid_result.error;
    return false;
  }

  pde_text_ = config.pde_latex;
  bound_xmin_ = domain.xmin;
  bound_xmax_ = domain.xmax;
  bound_ymin_ = domain.ymin;
  bound_ymax_ = domain.ymax;
  bound_zmin_ = domain.zmin;
  bound_zmax_ = domain.zmax;
  grid_nx_ = domain.nx;
  grid_ny_ = domain.ny;
  grid_nz_ = domain.nz;

  if (!config.coord_mode.empty()) {
    int mapped = coord_mode_;
    if (!CoordModeFromToken(config.coord_mode, &mapped)) {
      if (error) *error = "unknown coord_mode: " + config.coord_mode;
      return false;
    }
    coord_mode_ = mapped;
  } else {
    coord_mode_ = (domain.nz > 1) ? CoordMode::kCartesian3D : CoordMode::kCartesian2D;
  }

  if (!config.domain_mode.empty()) {
    const std::string mode = ToLower(Trim(config.domain_mode));
    domain_mode_ = (mode == "implicit") ? 1 : 0;
  } else {
    domain_mode_ = (config.domain_shape.empty() && config.domain_shape_mask.empty()) ? 0 : 1;
  }
  domain_shape_ = config.domain_shape;
  domain_shape_file_ = config.domain_shape_file;
  domain_shape_mask_path_ = config.domain_shape_mask;
  shape_mask_threshold_ = config.domain_shape_mask_threshold;
  shape_mask_invert_ = config.domain_shape_mask_invert;
  shape_transform_ = config.shape_transform;
  if (domain_shape_.empty() && !domain_shape_file_.empty()) {
    std::string shape_error;
    if (!LoadShapeExpressionFromFile(domain_shape_file_, &domain_shape_, &shape_error)) {
      if (error) *error = "shape file error: " + shape_error;
      return false;
    }
  }
  if (!domain_shape_mask_path_.empty()) {
    std::string mask_error;
    if (!LoadShapeMaskFromVtk(domain_shape_mask_path_, &shape_mask_, &mask_error)) {
      if (error) *error = "shape mask error: " + mask_error;
      return false;
    }
  } else {
    shape_mask_ = ShapeMask();
  }

  BoundaryInput left = {};
  BoundaryInput right = {};
  BoundaryInput bottom = {};
  BoundaryInput top = {};
  BoundaryInput front = {};
  BoundaryInput back = {};
  std::string bc_error;
  if (!ApplyBoundarySpecToInputs(config.boundary_spec, &left, &right, &bottom, &top, &front, &back,
                                 &bc_error)) {
    if (error) *error = bc_error;
    return false;
  }
  bc_left_ = left;
  bc_right_ = right;
  bc_bottom_ = bottom;
  bc_top_ = top;
  bc_front_ = front;
  bc_back_ = back;

  if (!config.backend.empty()) {
    backend_index_ = BackendToIndex(ParseBackendKind(config.backend));
  }
  if (!config.method.empty()) {
    method_index_ = MethodToIndex(ParseSolveMethodToken(config.method));
  }

  sor_omega_ = config.solver.sor_omega;
  gmres_restart_ = config.solver.gmres_restart;
  solver_max_iter_ = config.solver.max_iter;
  solver_tol_ = config.solver.tol;
  solver_residual_interval_ = config.solver.residual_interval;
  solver_mg_pre_smooth_ = config.solver.mg_pre_smooth;
  solver_mg_post_smooth_ = config.solver.mg_post_smooth;
  solver_mg_coarse_iters_ = config.solver.mg_coarse_iters;
  solver_mg_max_levels_ = config.solver.mg_max_levels;
  thread_count_ = config.solver.thread_count;
  metal_reduce_interval_ = config.solver.metal_reduce_interval;
  metal_tg_x_ = config.solver.metal_tg_x;
  metal_tg_y_ = config.solver.metal_tg_y;

  time_start_ = config.time.t_start;
  time_end_ = config.time.t_end;
  time_frames_ = config.time.enabled ? std::max(1, config.time.frames) : 1;

  if (!config.output_path.empty()) {
    output_path_ = config.output_path;
  } else if (!config.output_dir.empty()) {
    output_path_ = config.output_dir;
  } else {
    output_path_.clear();
  }
  if (IsDefaultOutputToken(output_path_)) {
    output_path_ = DefaultOutputPath(exec_path_);
  }

  viewer_.SetViewMode(ViewModeForCoord(coord_mode_));
  viewer_.SetTorusRadii(static_cast<float>(torus_major_), static_cast<float>(torus_minor_));
  pde_preview_.dirty = true;
  shape_preview_.dirty = true;
  bc_left_preview_.dirty = true;
  bc_right_preview_.dirty = true;
  bc_bottom_preview_.dirty = true;
  bc_top_preview_.dirty = true;
  bc_front_preview_.dirty = true;
  bc_back_preview_.dirty = true;

  UpdateApplicationState();
  return true;
}

bool Application::SaveRunConfigFile(const std::filesystem::path& path, std::string* error) {
  std::string build_error;
  RunConfig config = BuildRunConfig(&build_error);
  if (!build_error.empty()) {
    if (error) *error = build_error;
    return false;
  }
  return SaveRunConfigToFile(path, config, error);
}

bool Application::LoadRunConfigFile(const std::filesystem::path& path, std::string* error) {
  RunConfig config;
  if (!LoadRunConfigFromFile(path, &config, error)) {
    return false;
  }
  return ApplyRunConfig(config, error);
}

void Application::HandleFileLoading() {
  ApplicationState::State current_state = app_state_.GetState();
  frame_paths_strings_.clear();
  for (const auto& p : current_state.frame_paths) {
    frame_paths_strings_.push_back(p.string());
  }
  FileHandlerState file_state{
    current_state.input_dir, prefs_path_, shared_state_, state_mutex_, viewer_,
    frame_paths_strings_, current_state.frame_times, 
    current_state.frame_index, current_state.last_loaded_frame,
    current_state.playing, current_state.coord_mode, point_scale_, 
    [](const std::string&) {}
  };
  LoadLatestVtk(file_state);
  UpdateApplicationState();
}

void Application::HandleTimeSeriesPlayback() {
  if (playing_ && !frame_paths_.empty()) {
    ImGuiIO& io = ImGui::GetIO();
    play_accum_ += io.DeltaTime;
    const double frame_step = 1.0 / playback_fps_;
    while (play_accum_ >= frame_step) {
      play_accum_ -= frame_step;
      frame_index_++;
      if (frame_index_ >= static_cast<int>(frame_paths_.size())) {
        frame_index_ = static_cast<int>(frame_paths_.size()) - 1;
        playing_ = false;
        break;
      }
    }
  }
}

int Application::Run() {
  if (!Initialize()) {
    return 1;
  }
  
  // Resolve paths
  script_path_ = FindScriptPath(exec_path_);
  cache_dir_ = EnsureLatexCacheDir(script_path_.empty() ? 
      std::filesystem::current_path() : script_path_.parent_path().parent_path());
  python_path_ = ResolvePythonPath(script_path_.empty() ? 
      std::filesystem::current_path() : script_path_.parent_path().parent_path());
  ui_font_dir_ = FindUIFontDir(exec_path_);
  
  if (script_path_.empty()) {
    AddLog(shared_state_, state_mutex_, "latex: render_latex.py not found");
  }
  
  // Initialize registries
  PDETypeRegistry::Instance().InitializeBuiltInTypes();
  CoordinateSystemRegistry::Instance().InitializeBuiltInSystems();
  SolverMethodRegistry::Instance().InitializeBuiltInMethods();
  
  // Set up event handler callbacks now that everything is initialized
  EventHandler::Callbacks event_callbacks;
  event_callbacks.on_action = [this](AppAction action) { ExecuteAction(action); };
  event_callbacks.can_execute = [this](AppAction action) { return CanExecuteAction(action); };
  event_handler_.SetCallbacks(event_callbacks);

  NativeMenuCallbacks menu_callbacks;
  menu_callbacks.on_action = [this](AppAction action) { ExecuteAction(action); };
  menu_callbacks.can_execute = [this](AppAction action) { return CanExecuteAction(action); };
  menu_callbacks.is_checked = [this](AppAction action) { return IsActionChecked(action); };
  NativeMenu::Instance().Install(window_manager_.GetWindow(), menu_callbacks);
  
  // Main loop
  while (!window_manager_.ShouldClose()) {
    ProcessFrame();
  }
  
  return 0;
}

void Application::ProcessFrame() {
  window_manager_.PollEvents();
  solver_manager_.JoinIfFinished();

  SyncLayoutFromUIConfig();

  // Handle deferred font rebuild (requested by ForceApplyUISettings during previous frame)
  // Must happen before ImGui::NewFrame() to avoid invalid font pointer crashes
  if (pending_font_rebuild_) {
    pending_font_rebuild_ = false;
    const UIConfig& ui_cfg = ui_config_mgr_.GetConfig();
    const float desired_size = std::max(kMinUIFontSize, std::min(kMaxUIFontSize, ui_cfg.theme.font_size));
    const std::filesystem::path resolved_path = ResolveUIFontPath(ui_cfg.theme.font_path, ui_font_dir_);

    std::string warning;
    ApplyUIFont(resolved_path, desired_size, &warning);
    ui_font_path_ = resolved_path;
    ui_font_size_ = desired_size;

    // Rebuild font atlas
    ImGui_ImplOpenGL3_DestroyDeviceObjects();
    ImGui_ImplOpenGL3_CreateDeviceObjects();

    if (!warning.empty()) {
      UIToast::Show(UIToast::Type::Warning, warning);
    }
  }

  SyncFontFromUIConfig();

  ImGui_ImplOpenGL3_NewFrame();
  ImGui_ImplGlfw_NewFrame();
  ImGui::NewFrame();
  
  HandleTimeSeriesPlayback();
  
  RenderUI();
  
  // Update animations
  ImGuiIO& io = ImGui::GetIO();
  UIAnimation::Update(io.DeltaTime);
  
  // Render toast notifications
  UIToast::Render();

  // Help system modal (must be before ImGui::Render())
  UIHelp::ShowHelpSearch();

  // Error dialog modal (must be before ImGui::Render())
  error_dialog_.Render(shared_state_, state_mutex_);

  ImGui::Render();
  
  // Handle solve results
  std::optional<SolveResult> result;
  {
    std::lock_guard<std::mutex> lock(state_mutex_);
    if (shared_state_.result.has_value()) {
      result = std::move(shared_state_.result);
      shared_state_.result.reset();
    }
  }
  
  if (result.has_value()) {
    if (result->ok) {
      // Handle successful solve
      if (result->time_series && !result->frame_paths.empty()) {
        frame_paths_ = result->frame_paths;
        frame_times_ = result->frame_times;
        frame_index_ = 0;
        last_loaded_frame_ = -1;
        playing_ = false;
        UpdateApplicationState();
      } else {
        frame_paths_.clear();
        frame_times_.clear();
        frame_index_ = 0;
        last_loaded_frame_ = -1;
        playing_ = false;

        Domain domain;
        std::vector<double> grid;
        DerivedFields derived;
        bool has_derived = false;
        {
          std::lock_guard<std::mutex> lock(state_mutex_);
          domain = shared_state_.current_domain;
          grid = shared_state_.current_grid;
          has_derived = shared_state_.has_derived_fields;
          if (has_derived) {
            derived = shared_state_.derived_fields;
          }
        }

        if (!grid.empty()) {
          viewer_.SetData(domain, grid);
          if (has_derived) {
            viewer_.SetDerivedFields(derived);
          }
          viewer_.FitToView();
          point_scale_ = 1.0f;
          viewer_.SetPointScale(point_scale_);
        } else {
          // Fallback to file-based loading if the grid is missing.
          HandleFileLoading();
        }
        UpdateApplicationState();
      }
    }
  }
  
  // Render OpenGL
  int display_w, display_h;
  window_manager_.GetFramebufferSize(&display_w, &display_h);
  glViewport(0, 0, display_w, display_h);
  glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
  glClear(GL_COLOR_BUFFER_BIT);
  ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
  window_manager_.SwapBuffers();
}

void Application::RenderUI() {
  ImGuiIO& io = ImGui::GetIO();

  ImGui::SetNextWindowPos(ImVec2(0, 0), ImGuiCond_Always);
  ImGui::SetNextWindowSize(io.DisplaySize, ImGuiCond_Always);
  ImGuiWindowFlags window_flags = ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove |
      ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoBringToFrontOnFocus |
      ImGuiWindowFlags_NoTitleBar;
  if (!NativeMenu::Instance().IsInstalled()) {
    window_flags |= ImGuiWindowFlags_MenuBar;
  }
  ImGui::Begin("PDE Solver", nullptr, window_flags);
  
  // Process keyboard shortcuts
  event_handler_.ProcessShortcuts(io);
  
  RenderMenuBar();
  
  // Check time-dependent status
  LatexParser time_parser;
  LatexParseResult time_check = time_parser.Parse(pde_text_);
  const bool time_dependent =
      time_check.ok &&
      (std::abs(time_check.coeffs.ut) > 1e-12 || std::abs(time_check.coeffs.utt) > 1e-12);
  
  if (use_docking_ui_ && docking_ctx_) {
    // Render the docking system
    ImVec2 content_pos = ImGui::GetCursorScreenPos();
    ImVec2 content_size = ImGui::GetContentRegionAvail();
    docking_ctx_->Render(content_pos, content_size, services_);
  } else {
    // Traditional tab-based UI
    std::vector<std::string> tab_order = ui_config_mgr_.GetConfig().tab_order;
    if (tab_order.empty()) {
      tab_order = {"Main", "Inspect", "Preferences"};
    }

    // Tab bar
    if (ImGui::BeginTabBar("MainTabs")) {
      for (const auto& tab : tab_order) {
        ImGuiTabItemFlags tab_flags = ImGuiTabItemFlags_None;
        if (requested_tab_ && *requested_tab_ == tab) {
          tab_flags |= ImGuiTabItemFlags_SetSelected;
        }
        if (ImGui::BeginTabItem(tab.c_str(), nullptr, tab_flags)) {
          active_tab_ = tab;
          if (requested_tab_ && *requested_tab_ == tab) {
            requested_tab_.reset();
          }
          if (tab == "Main") {
            RenderMainTab(time_dependent);
          } else if (tab == "Inspect") {
            RenderInspectTab();
          } else if (tab == "Preferences") {
            RenderPreferencesTab();
          } else {
            const std::string child_id = "CustomTab##" + tab;
            ImGui::BeginChild(child_id.c_str(), ImVec2(-1, -1), false);
            RenderPanelsForTab(tab, tab.c_str(), false);
            ImGui::EndChild();
          }
          ImGui::EndTabItem();
        }
      }
      ImGui::EndTabBar();
    }
  }

  ImGui::End();

  RenderAboutModal();

  SavePreferencesIfNeeded();
}

// Visualization helpers (kept here for now; cleaned logs)
void Application::RenderVectorGlyphs(const ImVec2& image_min, const ImVec2& image_max) {
  const auto t0 = std::chrono::steady_clock::now();
  const Domain& dom = viewer_.domain();
  if (dom.nz > 1) return;

  DerivedFields derived_copy;
  bool has_derived = false;
  const auto lock_t0 = std::chrono::steady_clock::now();
  {
    std::lock_guard<std::mutex> lock(state_mutex_);
    has_derived = shared_state_.has_derived_fields;
    if (has_derived) {
      derived_copy = shared_state_.derived_fields;
    }
  }
  const auto lock_t1 = std::chrono::steady_clock::now();
  if (!has_derived) return;

  const size_t expected = static_cast<size_t>(dom.nx * dom.ny * std::max(1, dom.nz));
  if (derived_copy.gradient_x.size() != expected ||
      derived_copy.gradient_y.size() != expected) {
    return;
  }

  const std::vector<double>* vx = &derived_copy.gradient_x;
  const std::vector<double>* vy = &derived_copy.gradient_y;
  GlViewer::FieldType field = viewer_.GetFieldType();
  if (field == GlViewer::FieldType::FluxX || field == GlViewer::FieldType::FluxY ||
      field == GlViewer::FieldType::FluxZ) {
    if (derived_copy.flux_x.size() == expected && derived_copy.flux_y.size() == expected) {
      vx = &derived_copy.flux_x;
      vy = &derived_copy.flux_y;
    }
  }

  const int stride_x = std::max(1, dom.nx / 24);
  const int stride_y = std::max(1, dom.ny / 24);

  double max_mag = 0.0;
  for (int j = 0; j < dom.ny; j += stride_y) {
    for (int i = 0; i < dom.nx; i += stride_x) {
      size_t idx = static_cast<size_t>(j * dom.nx + i);
      double mag = std::sqrt((*vx)[idx] * (*vx)[idx] + (*vy)[idx] * (*vy)[idx]);
      if (mag > max_mag) max_mag = mag;
    }
  }
  if (max_mag <= 0.0) return;

  const float width = image_max.x - image_min.x;
  const float height = image_max.y - image_min.y;
  const float arrow_scale = std::min(width, height) / 25.0f;

  std::vector<ImVec2> starts;
  std::vector<ImVec2> ends;
  for (int j = 0; j < dom.ny; j += stride_y) {
    const double y = dom.ymin + (dom.ymax - dom.ymin) * (static_cast<double>(j) / std::max(1, dom.ny - 1));
    for (int i = 0; i < dom.nx; i += stride_x) {
      const double x = dom.xmin + (dom.xmax - dom.xmin) * (static_cast<double>(i) / std::max(1, dom.nx - 1));
      size_t idx = static_cast<size_t>(j * dom.nx + i);
      double vx_val = (*vx)[idx];
      double vy_val = (*vy)[idx];
      double mag = std::sqrt(vx_val * vx_val + vy_val * vy_val);
      if (mag <= 0.0) continue;
      float u = (x - dom.xmin) / (dom.xmax - dom.xmin);
      float v = (y - dom.ymin) / (dom.ymax - dom.ymin);
      float sx = image_min.x + u * width;
      float sy = image_max.y - v * height;  // flip vertical
      float scale = arrow_scale * static_cast<float>(mag / max_mag);
      ImVec2 start = ImVec2(sx, sy);
      ImVec2 end = ImVec2(sx + static_cast<float>(vx_val / mag) * scale,
                          sy - static_cast<float>(vy_val / mag) * scale);
      starts.push_back(start);
      ends.push_back(end);
    }
  }

  glyph_renderer_.SetArrows(starts, ends);
  glyph_renderer_.Render(ImGui::GetWindowDrawList(), IM_COL32(200, 220, 255, 200), 1.4f);

  const auto t1 = std::chrono::steady_clock::now();
  const auto total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
  const auto lock_ms = std::chrono::duration_cast<std::chrono::milliseconds>(lock_t1 - lock_t0).count();
  (void)total_ms;
  (void)lock_ms;
}

static const char* ViewModeLabel(GlViewer::ViewMode mode) {
  switch (mode) {
    case GlViewer::ViewMode::Cartesian: return "Cartesian";
    case GlViewer::ViewMode::Polar: return "Polar";
    case GlViewer::ViewMode::Axisymmetric: return "Axisymmetric";
    case GlViewer::ViewMode::CylindricalVolume: return "Cylindrical Volume";
    case GlViewer::ViewMode::SphericalSurface: return "Spherical Surface";
    case GlViewer::ViewMode::SphericalVolume: return "Spherical Volume";
    case GlViewer::ViewMode::ToroidalSurface: return "Toroidal Surface";
    case GlViewer::ViewMode::ToroidalVolume: return "Toroidal Volume";
  }
  return "Unknown";
}

static const char* FieldTypeLabel(GlViewer::FieldType type) {
  switch (type) {
    case GlViewer::FieldType::Solution: return "Solution";
    case GlViewer::FieldType::GradientX: return "Gradient X";
    case GlViewer::FieldType::GradientY: return "Gradient Y";
    case GlViewer::FieldType::GradientZ: return "Gradient Z";
    case GlViewer::FieldType::Laplacian: return "Laplacian";
    case GlViewer::FieldType::FluxX: return "Flux X";
    case GlViewer::FieldType::FluxY: return "Flux Y";
    case GlViewer::FieldType::FluxZ: return "Flux Z";
    case GlViewer::FieldType::EnergyNorm: return "Energy Norm";
  }
  return "Unknown";
}
void Application::RenderVisualizationPanel() {
  ImGui::BeginChild("RightPanel", ImVec2(0, 0), false, ImGuiWindowFlags_NoScrollbar);

  ImVec2 right_avail = ImGui::GetContentRegionAvail();
  float timebar_height = std::max(72.0f, ImGui::GetFrameHeightWithSpacing() * 3.0f);
  if (right_avail.y > 0.0f) {
    timebar_height = std::min(timebar_height, right_avail.y * 0.35f);
  }

  ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(UISpacing::MD, UISpacing::SM));
  ImGui::BeginChild("TimebarPanel", ImVec2(0, timebar_height), false,
                    ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse);
  ImGui::PopStyleVar();

  const bool has_frames = !frame_paths_.empty();

  // Animation controls - shared between docking UI and traditional layout
  RenderAnimationControls(ImGui::GetContentRegionAvail().x);

  ImGui::EndChild();

  if (has_frames && frame_index_ >= 0 &&
      frame_index_ < static_cast<int>(frame_paths_.size()) &&
      frame_index_ != last_loaded_frame_) {
    FileHandlerState file_state{
      input_dir_, prefs_path_, shared_state_, state_mutex_, viewer_,
      frame_paths_, frame_times_,
      frame_index_, last_loaded_frame_, playing_, coord_mode_, point_scale_,
      [](const std::string&) {}
    };
    const bool fit_view = (last_loaded_frame_ < 0);
    LoadVtkFile(file_state, frame_paths_[static_cast<size_t>(frame_index_)], fit_view);
    last_loaded_frame_ = frame_index_;
    if (fit_view) {
      viewer_.SetPointScale(point_scale_);
    }
  }

  ImGui::Spacing();

  ImGui::BeginChild("ViewerPanel", ImVec2(0, 0), false, ImGuiWindowFlags_NoScrollbar);

  // Render-only: fill the right pane with the texture and interactions.
  ImVec2 image_pos = ImGui::GetCursorScreenPos();
  ImVec2 avail = ImGui::GetContentRegionAvail();
  viewer_.RenderToTexture(static_cast<int>(avail.x), static_cast<int>(avail.y));
  ImTextureID texture_id = (ImTextureID)(uint64_t)viewer_.texture();
  ImGui::Image(texture_id, avail, ImVec2(0, 1), ImVec2(1, 0));
  
  ImVec2 image_min = image_pos;
  ImVec2 image_max(image_pos.x + avail.x, image_pos.y + avail.y);
  if (field_type_index_ == static_cast<int>(GlViewer::FieldType::GradientX) ||
      field_type_index_ == static_cast<int>(GlViewer::FieldType::GradientY) ||
      field_type_index_ == static_cast<int>(GlViewer::FieldType::FluxX) ||
      field_type_index_ == static_cast<int>(GlViewer::FieldType::FluxY)) {
    RenderVectorGlyphs(image_min, image_max);
  }
  DrawAxisLabels(viewer_, image_min, image_max, prefs_.colors.axis_label_color);
  
  const bool image_hovered = ImGui::IsItemHovered();
  if (image_hovered) {
    ImGuiIO& io = ImGui::GetIO();
    if (ImGui::IsMouseDown(ImGuiMouseButton_Left)) {
      ImGui::SetTooltip("LMB drag: rotate\nMouse wheel: zoom\nFit to view for reset");
    }
    
    if (ImGui::IsMouseDragging(ImGuiMouseButton_Left)) {
      const ImVec2 delta = ImGui::GetIO().MouseDelta;
      viewer_.Rotate(delta.x, delta.y);
      yaw_ = std::clamp(yaw_ + delta.x * 0.01f, -3.14f, 3.14f);
      pitch_ = std::clamp(pitch_ + delta.y * 0.01f, -1.4f, 1.4f);
      should_save_prefs_ = true;
    }

    if (io.MouseWheel != 0.0f) {
        viewer_.Zoom(io.MouseWheel);
        if (use_ortho_) {
          point_scale_ = viewer_.point_scale();
          if (lock_fit_ && point_scale_ < 1.0f) {
            point_scale_ = 1.0f;
            viewer_.SetPointScale(point_scale_);
          }
        }
      }

      if (ImGui::IsMouseClicked(ImGuiMouseButton_Left) && !ImGui::IsMouseDragging(ImGuiMouseButton_Left)) {
        const ImVec2 mp = ImGui::GetMousePos();
        const float width_px = image_max.x - image_min.x;
        const float height_px = image_max.y - image_min.y;
        if (width_px > 1.0f && height_px > 1.0f) {
          float u = (mp.x - image_min.x) / width_px;
        float v = (image_max.y - mp.y) / height_px;
          u = std::clamp(u, 0.0f, 1.0f);
          v = std::clamp(v, 0.0f, 1.0f);
          const Domain& dom = viewer_.domain();
          double px = dom.xmin + (dom.xmax - dom.xmin) * u;
          double py = dom.ymin + (dom.ymax - dom.ymin) * v;
          double pz = (dom.zmin + dom.zmax) * 0.5;
          if (dom.nz <= 1) {
          pz = dom.zmin;
          }
          if (InspectionToolsComponent* insp = GetInspectionComponentSingleton()) {
            std::string label = "Probe (" + std::to_string(px) + ", " + std::to_string(py) + ")";
            insp->AddProbe(ProbePoint(px, py, pz, label));
          }
        }
      }
    } else {
      dragging_ = false;
  }
  
  ImVec2 right_pos = ImGui::GetWindowPos();
  ImVec2 right_size = ImGui::GetWindowSize();
  const float gimbal_size = 92.0f;
  ImVec2 gimbal_pos(right_pos.x + right_size.x - 12.0f,
                    right_pos.y + right_size.y - gimbal_size - 12.0f);
  ImGuiIO& io = ImGui::GetIO();
  DrawGimbal(viewer_, gimbal_pos, gimbal_size, io);
  
  ImGui::EndChild();

  ImGui::EndChild();
}
