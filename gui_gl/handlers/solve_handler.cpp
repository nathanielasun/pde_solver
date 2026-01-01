#include "solve_handler.h"
#include "app_helpers.h"
#include "app_state.h"
#include "conserved_monitor.h"
#include "latex_parser.h"
#include "input_parse.h"
#include "solver.h"
#include "vtk_io.h"
#include "solve_service.h"
#include "run_metadata.h"
#include "run_summary.h"
#include "solver_tokens.h"
#include "shape_io.h"
#ifdef USE_METAL
#include "MetalSolve.h"
#endif
#include <fstream>
#include <chrono>
#include <filesystem>
#include <algorithm>
#include <cmath>
#include <utility>

namespace {
std::string CoordModeTokenForConfig(int coord_mode) {
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

RunConfig BuildRunConfigForHandler(const SolveHandlerState& state,
                                   const std::string& domain_bounds,
                                   const std::string& grid_text,
                                   const std::string& bc_spec,
                                   const std::filesystem::path& resolved_output,
                                   bool time_enabled,
                                   double time_dt) {
  RunConfig config;
  config.pde_latex = state.pde_text;
  config.domain_bounds = domain_bounds;
  config.grid = grid_text;
  config.boundary_spec = bc_spec;
  config.domain_shape = state.domain_shape;
  config.domain_shape_file = state.domain_shape_file;
  config.domain_shape_mask = state.domain_shape_mask_path;
  config.domain_shape_mask_threshold = state.shape_mask_threshold;
  config.domain_shape_mask_invert = state.shape_mask_invert;
  config.shape_transform = state.shape_transform;
  config.domain_mode = (state.domain_mode == 1) ? "implicit" : "box";
  config.coord_mode = CoordModeTokenForConfig(state.coord_mode);
  config.backend = BackendToken(BackendFromIndex(state.backend_index));
  config.method = MethodToken(MethodFromIndex(state.method_index));
  config.solver.max_iter = state.solver_max_iter;
  config.solver.tol = state.solver_tol;
  config.solver.residual_interval = state.solver_residual_interval;
  config.solver.thread_count = state.thread_count;
  config.solver.metal_reduce_interval = state.metal_reduce_interval;
  config.solver.metal_tg_x = state.metal_tg_x;
  config.solver.metal_tg_y = state.metal_tg_y;
  config.solver.sor_omega = state.sor_omega;
  config.solver.gmres_restart = state.gmres_restart;
  config.solver.mg_pre_smooth = state.solver_mg_pre_smooth;
  config.solver.mg_post_smooth = state.solver_mg_post_smooth;
  config.solver.mg_coarse_iters = state.solver_mg_coarse_iters;
  config.solver.mg_max_levels = state.solver_mg_max_levels;
  config.time.enabled = time_enabled;
  config.time.t_start = state.time_start;
  config.time.t_end = state.time_end;
  config.time.frames = time_enabled ? std::max(1, state.time_frames) : 1;
  config.time.dt = time_dt;

  config.output_format = "vtk";
  if (resolved_output.extension() == ".vti") {
    config.output_format = "vti";
  }
  config.output_path.clear();
  config.output_dir.clear();
  if (state.output_path.empty()) {
    config.output_dir = "outputs";
  } else {
    std::filesystem::path output_path(state.output_path);
    const bool ends_with_slash =
        !state.output_path.empty() && state.output_path.back() == '/';
    const bool is_dir =
        std::filesystem::exists(output_path) && std::filesystem::is_directory(output_path);
    if (ends_with_slash || is_dir) {
      config.output_dir = state.output_path;
    } else {
      config.output_path = state.output_path;
    }
    if (output_path.extension() == ".vti") {
      config.output_format = "vti";
    }
  }
  return config;
}
}  // namespace

void LaunchSolve(SolveHandlerState& handler_state) {
  // #region agent log
  {
    std::ofstream f("/Users/nathaniel.sun/Desktop/programming/cursor/.cursor/debug.log",
                    std::ios::app);
    if (f) {
      const auto ts = std::chrono::duration_cast<std::chrono::milliseconds>(
                          std::chrono::system_clock::now().time_since_epoch())
                          .count();
      f << "{\"sessionId\":\"debug-session\",\"runId\":\"run1\",\"hypothesisId\":\"B\","
           "\"location\":\"gui_gl/handlers/solve_handler.cpp:LaunchSolve\",\"message\":\"Enter\","
           "\"data\":{\"solver_thread_ptr_null\":" << (handler_state.solver_thread ? "false" : "true")
        << ",\"solver_thread_joinable\":"
        << ((handler_state.solver_thread && handler_state.solver_thread->joinable()) ? "true" : "false")
        << "},\"timestamp\":" << ts << "}\n";
    }
  }
  // #endregion agent log
  if (handler_state.solver_thread && handler_state.solver_thread->joinable()) {
    handler_state.solver_thread->join();
  }
  handler_state.start_solver();

  SolveInput input;
  LatexParser parser;
  const bool use_cartesian_3d = (handler_state.coord_mode == CoordMode::kCartesian3D);
  const bool use_axisymmetric = (handler_state.coord_mode == CoordMode::kAxisymmetric);
  const bool use_cylindrical_volume = (handler_state.coord_mode == CoordMode::kCylindricalVolume);
  const bool use_spherical_surface = (handler_state.coord_mode == CoordMode::kSphericalSurface);
  const bool use_spherical_volume = (handler_state.coord_mode == CoordMode::kSphericalVolume);
  const bool use_toroidal_surface = (handler_state.coord_mode == CoordMode::kToroidalSurface);
  const bool use_toroidal_volume = (handler_state.coord_mode == CoordMode::kToroidalVolume);
  const bool use_surface = use_spherical_surface || use_toroidal_surface;
  const bool use_volume =
      use_spherical_volume || use_toroidal_volume || use_cartesian_3d || use_cylindrical_volume;
  std::string pde_for_parse = handler_state.pde_text;
  if (use_surface) {
    pde_for_parse = RemapSphericalSurfaceExpr(pde_for_parse);
  }
  if (use_axisymmetric) {
    pde_for_parse = RemapAxisymmetricExpr(pde_for_parse);
  }
  LatexParseResult parse_result = parser.Parse(pde_for_parse);
  if (!parse_result.ok) {
    handler_state.report_status("LaTeX parse error: " + parse_result.error);
    AddLog(handler_state.state, handler_state.state_mutex, "latex: " + parse_result.error);
    std::lock_guard<std::mutex> lock(handler_state.state_mutex);
    handler_state.state.last_error = ErrorInfo{
      "LaTeX parse error",
      "Failed to parse PDE input.",
      parse_result.error,
      {"Check for unmatched braces or missing \\frac{...}{...} braces.",
       "Try simplifying the PDE and reintroducing terms incrementally.",
       "If using surface/axisymmetric modes, verify variable names map correctly (x,y)."},
      true
    };
    handler_state.state.error_dialog_open = true;
    handler_state.state.running = false;
    return;
  }
  const bool time_dependent =
      std::abs(parse_result.coeffs.ut) > 1e-12 || std::abs(parse_result.coeffs.utt) > 1e-12;
  double time_dt = 0.0;
  bool time_enabled = false;

  Domain domain;
  const int min_grid = 3;
  if (handler_state.grid_nx < min_grid) {
    handler_state.grid_nx = min_grid;
  }
  if (handler_state.grid_ny < min_grid) {
    handler_state.grid_ny = min_grid;
  }
  if (use_volume && handler_state.grid_nz < min_grid) {
    handler_state.grid_nz = min_grid;
  }
  std::string domain_bounds;
  std::string grid_text;
  if (use_volume) {
    domain_bounds =
        FormatBounds3D(handler_state.bound_xmin, handler_state.bound_xmax, 
                      handler_state.bound_ymin, handler_state.bound_ymax,
                      handler_state.bound_zmin, handler_state.bound_zmax);
    grid_text = FormatGrid3D(handler_state.grid_nx, handler_state.grid_ny, handler_state.grid_nz);
  } else {
    domain_bounds = FormatBounds(handler_state.bound_xmin, handler_state.bound_xmax, 
                                 handler_state.bound_ymin, handler_state.bound_ymax);
    grid_text = FormatGrid(handler_state.grid_nx, handler_state.grid_ny);
  }
  ParseResult domain_result = ParseDomain(domain_bounds, &domain);
  if (!domain_result.ok) {
    handler_state.report_status(domain_result.error);
    AddLog(handler_state.state, handler_state.state_mutex, "input: " + domain_result.error);
    std::lock_guard<std::mutex> lock(handler_state.state_mutex);
    handler_state.state.last_error = ErrorInfo{
      "Domain input error",
      "Invalid domain bounds.",
      domain_result.error,
      {"Ensure min < max for each axis.",
       "For radial coordinates, ensure r_min >= 0."},
      true
    };
    handler_state.state.error_dialog_open = true;
    handler_state.state.running = false;
    return;
  }
  ParseResult grid_result = ParseGrid(grid_text, &domain);
  if (!grid_result.ok) {
    handler_state.report_status(grid_result.error);
    AddLog(handler_state.state, handler_state.state_mutex, "input: " + grid_result.error);
    std::lock_guard<std::mutex> lock(handler_state.state_mutex);
    handler_state.state.last_error = ErrorInfo{
      "Grid input error",
      "Invalid grid resolution.",
      grid_result.error,
      {"Use at least 3 points per axis.",
       "Avoid extremely large grids that exceed memory."},
      true
    };
    handler_state.state.error_dialog_open = true;
    handler_state.state.running = false;
    return;
  }

  // Set coordinate system from GUI selection
  switch (handler_state.coord_mode) {
    case CoordMode::kCartesian2D:
      domain.coord_system = CoordinateSystem::Cartesian;
      break;
    case CoordMode::kCartesian3D:
      domain.coord_system = CoordinateSystem::Cartesian;
      break;
    case CoordMode::kPolar:
      domain.coord_system = CoordinateSystem::Polar;
      break;
    case CoordMode::kAxisymmetric:
      domain.coord_system = CoordinateSystem::Axisymmetric;
      break;
    case CoordMode::kCylindricalVolume:
      domain.coord_system = CoordinateSystem::Cylindrical;
      break;
    case CoordMode::kSphericalSurface:
      domain.coord_system = CoordinateSystem::SphericalSurface;
      break;
    case CoordMode::kSphericalVolume:
      domain.coord_system = CoordinateSystem::SphericalVolume;
      break;
    case CoordMode::kToroidalSurface:
      domain.coord_system = CoordinateSystem::ToroidalSurface;
      break;
    case CoordMode::kToroidalVolume:
      domain.coord_system = CoordinateSystem::ToroidalVolume;
      break;
    default:
      domain.coord_system = CoordinateSystem::Cartesian;
      break;
  }

  std::string domain_shape_eval = handler_state.domain_shape;
  if (domain_shape_eval.empty() && !handler_state.domain_shape_file.empty()) {
    std::string shape_error;
    if (!LoadShapeExpressionFromFile(handler_state.domain_shape_file, &domain_shape_eval,
                                     &shape_error)) {
      handler_state.report_status("shape file error: " + shape_error);
      AddLog(handler_state.state, handler_state.state_mutex,
             "input: shape file error " + shape_error);
      std::lock_guard<std::mutex> lock(handler_state.state_mutex);
      handler_state.state.last_error = ErrorInfo{
        "Shape file error",
        "Failed to load implicit shape from file.",
        shape_error,
        {"Verify the file path and format.",
         "Ensure the file contains a valid expression."},
        true
      };
      handler_state.state.error_dialog_open = true;
      handler_state.state.running = false;
      return;
    }
  }
  if (!domain_shape_eval.empty()) {
    if (use_surface) {
      domain_shape_eval = RemapSphericalSurfaceExpr(domain_shape_eval);
    }
    if (use_axisymmetric) {
      domain_shape_eval = RemapAxisymmetricExpr(domain_shape_eval);
    }
  }

  ShapeMask shape_mask = handler_state.shape_mask;
  if (handler_state.domain_mode == 1 &&
      !handler_state.domain_shape_mask_path.empty() &&
      !HasShapeMask(shape_mask)) {
    std::string mask_error;
    if (!LoadShapeMaskFromVtk(handler_state.domain_shape_mask_path, &shape_mask, &mask_error)) {
      handler_state.report_status("shape mask error: " + mask_error);
      AddLog(handler_state.state, handler_state.state_mutex,
             "input: shape mask error " + mask_error);
      std::lock_guard<std::mutex> lock(handler_state.state_mutex);
      handler_state.state.last_error = ErrorInfo{
        "Shape mask error",
        "Failed to load implicit mask from file.",
        mask_error,
        {"Verify the mask file path and VTK format.",
         "Ensure the mask contains scalar values."},
        true
      };
      handler_state.state.error_dialog_open = true;
      handler_state.state.running = false;
      return;
    }
  }

  if (handler_state.domain_mode == 1) {
    if (Trim(domain_shape_eval).empty() && !HasShapeMask(shape_mask)) {
      handler_state.report_status("Implicit domain requires a shape function or mask");
      AddLog(handler_state.state, handler_state.state_mutex, "input: missing implicit shape");
      std::lock_guard<std::mutex> lock(handler_state.state_mutex);
      handler_state.state.last_error = ErrorInfo{
        "Implicit domain missing shape",
        "Implicit domain mode requires a shape function or mask.",
        "Provide a scalar function f(x,y[,z]) where f<=0 indicates inside the domain, "
        "or load a sampled mask.",
        {"Switch back to bounding-box domain mode, or enter a shape function.",
         "For surfaces, use the remapped variables x,y."},
        true
      };
      handler_state.state.error_dialog_open = true;
      handler_state.state.running = false;
      return;
    }
    input.domain_shape = domain_shape_eval;
    input.shape_transform = handler_state.shape_transform;
    input.shape_mask = std::move(shape_mask);
    input.shape_mask_threshold = handler_state.shape_mask_threshold;
    input.shape_mask_invert = handler_state.shape_mask_invert;
  }

  BoundarySet bc;
  BoundaryInput bc_left_eval = handler_state.bc_left;
  BoundaryInput bc_right_eval = handler_state.bc_right;
  BoundaryInput bc_bottom_eval = handler_state.bc_bottom;
  BoundaryInput bc_top_eval = handler_state.bc_top;
  BoundaryInput bc_front_eval = handler_state.bc_front;
  BoundaryInput bc_back_eval = handler_state.bc_back;
  if (use_surface || use_axisymmetric) {
    auto remap_bc = [&](BoundaryInput* input) {
      if (use_surface) {
        input->value = RemapSphericalSurfaceExpr(input->value);
        input->alpha = RemapSphericalSurfaceExpr(input->alpha);
        input->beta = RemapSphericalSurfaceExpr(input->beta);
        input->gamma = RemapSphericalSurfaceExpr(input->gamma);
      }
      if (use_axisymmetric) {
        input->value = RemapAxisymmetricExpr(input->value);
        input->alpha = RemapAxisymmetricExpr(input->alpha);
        input->beta = RemapAxisymmetricExpr(input->beta);
        input->gamma = RemapAxisymmetricExpr(input->gamma);
      }
    };
    remap_bc(&bc_left_eval);
    remap_bc(&bc_right_eval);
    remap_bc(&bc_bottom_eval);
    remap_bc(&bc_top_eval);
    if (use_volume) {
      remap_bc(&bc_front_eval);
    }
    remap_bc(&bc_back_eval);
  }
  std::string bc_spec;
  std::string bc_error;
  if (!BuildBoundarySpec(bc_left_eval, bc_right_eval, bc_bottom_eval, bc_top_eval,
                         bc_front_eval, bc_back_eval, &bc_spec, &bc_error)) {
    handler_state.report_status("boundary error: " + bc_error);
    AddLog(handler_state.state, handler_state.state_mutex, "boundary: " + bc_error);
    std::lock_guard<std::mutex> lock(handler_state.state_mutex);
    handler_state.state.last_error = ErrorInfo{
      "Boundary condition error",
      "Failed to build boundary specification.",
      bc_error,
      {"Check that each boundary has valid Dirichlet/Neumann/Robin parameters.",
       "For Robin: alpha, beta, and gamma must be non-empty."},
      true
    };
    handler_state.state.error_dialog_open = true;
    handler_state.state.running = false;
    return;
  }
  ParseResult bc_result = ApplyBoundarySpec(bc_spec, &bc);
  if (!bc_result.ok) {
    handler_state.report_status("boundary error: " + bc_result.error);
    AddLog(handler_state.state, handler_state.state_mutex, "boundary: " + bc_result.error);
    std::lock_guard<std::mutex> lock(handler_state.state_mutex);
    handler_state.state.last_error = ErrorInfo{
      "Boundary condition error",
      "Failed to parse boundary specification.",
      bc_result.error,
      {"Verify boundary spec formatting and required parameters.",
       "Try setting all boundaries to simple Dirichlet (u=0) to isolate the issue."},
      true
    };
    handler_state.state.error_dialog_open = true;
    handler_state.state.running = false;
    return;
  }

  input.pde = parse_result.coeffs;
  input.integrals = parse_result.integrals;
  input.nonlinear = parse_result.nonlinear;
  input.nonlinear_derivatives = parse_result.nonlinear_derivatives;
  input.domain = domain;
  input.bc = bc;
  input.solver.max_iter = handler_state.solver_max_iter;
  input.solver.tol = handler_state.solver_tol;
  input.solver.residual_interval = handler_state.solver_residual_interval;
  input.solver.thread_count = handler_state.thread_count;
  input.cancel = &handler_state.cancel_requested;
  input.solver.metal_reduce_interval = handler_state.metal_reduce_interval;
  input.solver.metal_tg_x = handler_state.metal_tg_x;
  input.solver.metal_tg_y = handler_state.metal_tg_y;
  input.solver.method = MethodFromIndex(handler_state.method_index);
  input.solver.sor_omega = handler_state.sor_omega;
  input.solver.gmres_restart = handler_state.gmres_restart;
  input.solver.mg_pre_smooth = handler_state.solver_mg_pre_smooth;
  input.solver.mg_post_smooth = handler_state.solver_mg_post_smooth;
  input.solver.mg_coarse_iters = handler_state.solver_mg_coarse_iters;
  input.solver.mg_max_levels = handler_state.solver_mg_max_levels;
  if (time_dependent) {
    if (handler_state.time_end < handler_state.time_start) {
      handler_state.report_status("time range must satisfy t end >= t start");
      AddLog(handler_state.state, handler_state.state_mutex, "time: invalid range");
      std::lock_guard<std::mutex> lock(handler_state.state_mutex);
      handler_state.state.last_error = ErrorInfo{
        "Time range error",
        "Invalid time interval.",
        "t end must be >= t start.",
        {"Swap the time endpoints or set t end >= t start.",
         "If you intended steady-state, remove time derivatives from the PDE."},
        true
      };
      handler_state.state.error_dialog_open = true;
      handler_state.state.running = false;
      return;
    }
    if (handler_state.time_frames < 1) {
      handler_state.time_frames = 1;
    }
    const double denom = static_cast<double>(std::max(1, handler_state.time_frames - 1));
    const double dt = (handler_state.time_end - handler_state.time_start) / denom;
    time_dt = dt;
    time_enabled = true;
    input.time.enabled = true;
    input.time.t_start = handler_state.time_start;
    input.time.t_end = handler_state.time_end;
    input.time.frames = handler_state.time_frames;
    input.time.dt = dt;
  }

  BackendKind requested = BackendFromIndex(handler_state.backend_index);

  std::string warning;
  std::filesystem::path resolved_output = ResolveOutputPath(handler_state.output_path, &warning);
  if (!warning.empty()) {
    AddLog(handler_state.state, handler_state.state_mutex, "io: " + warning);
  }

  std::string bc_spec_config = bc_spec;
  std::string bc_config_error;
  if (!BuildBoundarySpec(handler_state.bc_left, handler_state.bc_right, handler_state.bc_bottom,
                         handler_state.bc_top, handler_state.bc_front, handler_state.bc_back,
                         &bc_spec_config, &bc_config_error)) {
    bc_spec_config = bc_spec;
  }
  const RunConfig run_config =
      BuildRunConfigForHandler(handler_state, domain_bounds, grid_text, bc_spec_config,
                               resolved_output, time_enabled, time_dt);
  const bool run_config_ready = !bc_spec_config.empty();

  if (handler_state.solver_thread) {
    *handler_state.solver_thread = std::thread([=, &state = handler_state.state, &state_mutex = handler_state.state_mutex]() {
      // #region agent log
      {
        std::ofstream f("/Users/nathaniel.sun/Desktop/programming/cursor/.cursor/debug.log",
                        std::ios::app);
        if (f) {
          const auto ts = std::chrono::duration_cast<std::chrono::milliseconds>(
                              std::chrono::system_clock::now().time_since_epoch())
                              .count();
          f << "{\"sessionId\":\"debug-session\",\"runId\":\"run1\",\"hypothesisId\":\"C\","
               "\"location\":\"gui_gl/handlers/solve_handler.cpp:solver_thread\",\"message\":\"ThreadStarted\","
               "\"data\":{},\"timestamp\":" << ts << "}\n";
        }
      }
      // #endregion agent log
    const auto start_time = std::chrono::steady_clock::now();
    auto finalize = [&](SolveResult result) {
      const auto end_time = std::chrono::steady_clock::now();
      const double elapsed =
          std::chrono::duration_cast<std::chrono::duration<double>>(end_time - start_time)
              .count();
      std::lock_guard<std::mutex> lock(state_mutex);
      state.running = false;
      state.result = std::move(result);
      state.last_duration = elapsed;
      state.has_duration = true;
    };
    ProgressCallback callback = [&](const std::string& phase, double value) {
      std::unique_lock<std::mutex> lock(state_mutex, std::try_to_lock);
      if (!lock.owns_lock()) {
        return;
      }
      if (phase == "threads_total") {
        state.thread_total = static_cast<int>(value);
        return;
      }
      if (phase == "threads_active") {
        state.thread_active = static_cast<int>(value);
        return;
      }
      if (phase == "residual_l2") {
        state.residual_l2.push_back(static_cast<float>(std::max(0.0, value)));
        state.detailed_progress.residual_l2 = value;
        return;
      }
      if (phase == "residual_linf") {
        state.residual_linf.push_back(static_cast<float>(std::max(0.0, value)));
        state.detailed_progress.residual_linf = value;
        return;
      }
      if (phase == "solve_total") {
        state.detailed_progress.total_iterations = static_cast<int>(value);
        return;
      }
      if (phase == "iteration") {
        state.detailed_progress.Update(state.progress, static_cast<int>(value), 
                                      state.detailed_progress.total_iterations);
        return;
      }
      state.phase = phase;
      state.progress = value;
      state.detailed_progress.phase = phase;
      state.detailed_progress.Update(value, state.detailed_progress.current_iteration,
                                    state.detailed_progress.total_iterations);
    };

    BackendKind selected = BackendKind::CPU;
    std::string note;
    {
      std::lock_guard<std::mutex> lock(state_mutex);
      state.detailed_progress.backend_name = "CPU";  // Default, will be updated
    }
    if (input.time.enabled) {
      // Time-dependent PDEs: Metal is supported for 2D, otherwise CPU
      bool use_metal_time = false;
#ifdef USE_METAL
      AddLog(state, state_mutex, "time-series: checking Metal availability (requested=" +
             std::to_string(static_cast<int>(requested)) + ", nz=" + std::to_string(input.domain.nz) + ")");
      if ((requested == BackendKind::Metal || requested == BackendKind::Auto) && input.domain.nz <= 1) {
        std::string metal_note;
        if (MetalIsAvailable(&metal_note)) {
          use_metal_time = true;
          selected = BackendKind::Metal;
          note = metal_note;
          AddLog(state, state_mutex, "time-series: Metal selected (" + metal_note + ")");
          std::lock_guard<std::mutex> lock(state_mutex);
          state.detailed_progress.backend_name = "Metal";
          state.detailed_progress.backend_note = note;
        } else {
          AddLog(state, state_mutex, "time-series: Metal unavailable (" + metal_note + ")");
        }
      }
#else
      AddLog(state, state_mutex, "time-series: Metal not compiled in");
#endif
      if (!use_metal_time) {
        selected = BackendKind::CPU;
        if (requested != BackendKind::CPU && requested != BackendKind::Auto) {
          note = "time stepping uses CPU (Metal unavailable or 3D)";
          std::lock_guard<std::mutex> lock(state_mutex);
          state.detailed_progress.backend_note = note;
        }
      }
    }
    if (input.time.enabled) {
      std::error_code series_ec;
      const std::filesystem::path series_dir =
          resolved_output.parent_path() /
          (resolved_output.stem().string() + "_" + GenerateRandomTag(6));
      std::filesystem::create_directories(series_dir, series_ec);
      if (series_ec) {
        finalize(SolveResult{false, "", "failed to create series directory: " + series_ec.message(),
                             selected, note});
        return;
      }
      const std::filesystem::path series_base = series_dir / resolved_output.filename();
      std::vector<std::string> series_paths;
      std::vector<double> series_times;
      std::string frame_error;
      const int digits = FrameDigits(input.time.frames);
      ConservedMonitor monitor;
      auto frame_cb = [&](int frame, double t, const std::vector<double>& grid,
                          const std::vector<double>*) -> bool {
        const ConservedSample sample = UpdateConservedMonitor(domain, frame, grid, &monitor);
        if (sample.ok) {
          if (monitor.mass_warning && monitor.mass_warning_frame == frame) {
            AddLog(state, state_mutex,
                   "monitor: mass drift exceeds 1% at frame " + std::to_string(frame));
          }
          if (monitor.energy_warning && monitor.energy_warning_frame == frame) {
            AddLog(state, state_mutex,
                   "monitor: energy drift exceeds 1% at frame " + std::to_string(frame));
          }
          if (monitor.blowup_warning && monitor.blowup_warning_frame == frame) {
            AddLog(state, state_mutex,
                   "stability: rapid growth detected (ratio " + std::to_string(monitor.blowup_ratio) +
                       " at frame " + std::to_string(frame) + ")");
          }
          if (monitor.blowup_warning) {
            std::lock_guard<std::mutex> lock(state_mutex);
            state.stability_warning = true;
            state.stability_frame = monitor.blowup_warning_frame;
            state.stability_ratio = monitor.blowup_ratio;
            state.stability_max = monitor.blowup_max;
          }
        }
        const std::filesystem::path frame_path = BuildFramePath(series_base, frame, digits);
        
        // Compute derived fields for this frame
        DerivedFields derived = ComputeDerivedFields(domain, grid, input.pde.a, input.pde.b, input.pde.az);
        std::vector<std::vector<double>> derived_field_data;
        std::vector<std::string> derived_field_names;
        
        derived_field_data.push_back(derived.gradient_x);
        derived_field_names.push_back("gradient_x");
        derived_field_data.push_back(derived.gradient_y);
        derived_field_names.push_back("gradient_y");
        if (domain.nz > 1 && !derived.gradient_z.empty()) {
          derived_field_data.push_back(derived.gradient_z);
          derived_field_names.push_back("gradient_z");
        }
        derived_field_data.push_back(derived.laplacian);
        derived_field_names.push_back("laplacian");
        derived_field_data.push_back(derived.flux_x);
        derived_field_names.push_back("flux_x");
        derived_field_data.push_back(derived.flux_y);
        derived_field_names.push_back("flux_y");
        if (domain.nz > 1 && !derived.flux_z.empty()) {
          derived_field_data.push_back(derived.flux_z);
          derived_field_names.push_back("flux_z");
        }
        derived_field_data.push_back(derived.energy_norm);
        derived_field_names.push_back("energy_norm");

        // Write VTK XML if .vti extension, otherwise legacy VTK
        VtkWriteResult write_result;
        if (frame_path.extension() == ".vti") {
          write_result = WriteVtkXmlImageData(frame_path.string(), domain, grid,
                                              &derived_field_data, &derived_field_names);
        } else {
          write_result = WriteVtkStructuredPoints(frame_path.string(), domain, grid);
        }
        if (!write_result.ok) {
          frame_error = write_result.error;
          return false;
        }
        if (run_config_ready) {
          std::string meta_error;
          if (!WriteRunMetadataSidecar(frame_path, run_config, requested, selected, note, true,
                                       frame, t, &meta_error)) {
            AddLog(state, state_mutex, "metadata: " + meta_error);
          }
          RunSummaryData summary;
          summary.run_config = run_config;
          summary.requested_backend = requested;
          summary.selected_backend = selected;
          summary.backend_note = note;
          summary.output_path = frame_path.string();
          summary.time_series = true;
          summary.frame_index = frame;
          summary.frame_time = t;
          if (sample.ok) {
            summary.monitors_enabled = true;
            summary.monitor_mass = sample.mass;
            summary.monitor_energy = sample.energy;
            summary.monitor_max_abs = sample.max_abs;
            summary.monitor_mass_drift = sample.mass_drift;
            summary.monitor_energy_drift = sample.energy_drift;
            summary.monitor_mass_warning = monitor.mass_warning;
            summary.monitor_energy_warning = monitor.energy_warning;
            summary.monitor_blowup_warning = monitor.blowup_warning;
            summary.monitor_mass_warning_frame = monitor.mass_warning_frame;
            summary.monitor_energy_warning_frame = monitor.energy_warning_frame;
            summary.monitor_blowup_warning_frame = monitor.blowup_warning_frame;
            summary.monitor_blowup_ratio = monitor.blowup_ratio;
            summary.monitor_blowup_max = monitor.blowup_max;
          }
          std::string summary_error;
          if (!WriteRunSummarySidecar(frame_path, summary, &summary_error)) {
            AddLog(state, state_mutex, "summary: " + summary_error);
          }
        }
        series_paths.push_back(frame_path.string());
        series_times.push_back(t);
        {
          std::lock_guard<std::mutex> lock(state_mutex);
          state.status = "Solved frame " + std::to_string(frame + 1) + "/" +
                         std::to_string(input.time.frames);
        }
        return true;
      };

      SolveOutput output;
#ifdef USE_METAL
      if (selected == BackendKind::Metal) {
        AddLog(state, state_mutex, "time-series: invoking Metal GPU backend");
        output = SolvePDETimeSeriesMetal(input, frame_cb, callback);
        if (!output.error.empty()) {
          AddLog(state, state_mutex, "Metal time-series error: " + output.error);
        } else {
          AddLog(state, state_mutex, "time-series: Metal solve completed successfully");
        }
      } else {
        AddLog(state, state_mutex, "time-series: using CPU backend");
        output = SolvePDETimeSeries(input, frame_cb, callback);
      }
#else
      AddLog(state, state_mutex, "time-series: using CPU backend (Metal not compiled)");
      output = SolvePDETimeSeries(input, frame_cb, callback);
#endif
      if (!output.error.empty()) {
        finalize(SolveResult{false, "", output.error, selected, note});
        return;
      }
      if (!frame_error.empty()) {
        finalize(SolveResult{false, "", frame_error, selected, note});
        return;
      }

      if (!series_paths.empty()) {
        const std::filesystem::path manifest =
            series_base.parent_path() /
            (series_base.stem().string() + "_series.pvd");
        WriteVtkSeriesPvd(manifest.string(), series_paths, series_times);
        if (run_config_ready) {
          std::string meta_error;
          if (!WriteRunMetadataSidecar(manifest, run_config, requested, selected, note, true, -1,
                                       0.0, &meta_error)) {
            AddLog(state, state_mutex, "metadata: " + meta_error);
          }
          RunSummaryData summary;
          summary.run_config = run_config;
          summary.requested_backend = requested;
          summary.selected_backend = selected;
          summary.backend_note = note;
          summary.output_path = manifest.string();
          summary.time_series = true;
          summary.frame_times = series_times;
          const double elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(
                                     std::chrono::steady_clock::now() - start_time)
                                     .count();
          summary.solve_seconds = elapsed;
          summary.total_seconds = elapsed;
          if (!monitor.mass_history.empty()) {
            summary.monitors_enabled = true;
            summary.monitor_mass_history = monitor.mass_history;
            summary.monitor_energy_history = monitor.energy_history;
            summary.monitor_max_abs_history = monitor.max_abs_history;
            summary.monitor_mass_drift_history = monitor.mass_drift_history;
            summary.monitor_energy_drift_history = monitor.energy_drift_history;
            summary.monitor_mass = monitor.mass_history.back();
            summary.monitor_energy = monitor.energy_history.back();
            summary.monitor_max_abs = monitor.max_abs_history.back();
            summary.monitor_mass_drift = monitor.mass_drift_history.back();
            summary.monitor_energy_drift = monitor.energy_drift_history.back();
            summary.monitor_mass_warning = monitor.mass_warning;
            summary.monitor_energy_warning = monitor.energy_warning;
            summary.monitor_blowup_warning = monitor.blowup_warning;
            summary.monitor_mass_warning_frame = monitor.mass_warning_frame;
            summary.monitor_energy_warning_frame = monitor.energy_warning_frame;
            summary.monitor_blowup_warning_frame = monitor.blowup_warning_frame;
            summary.monitor_blowup_ratio = monitor.blowup_ratio;
            summary.monitor_blowup_max = monitor.blowup_max;
          }
          std::string summary_error;
          if (!WriteRunSummarySidecar(manifest, summary, &summary_error)) {
            AddLog(state, state_mutex, "summary: " + summary_error);
          }
        }
      }

      SolveResult result;
      result.ok = true;
      result.output_path = series_base.string();
      result.backend = selected;
      result.note = note;
      result.time_series = true;
      result.frame_paths = std::move(series_paths);
      result.frame_times = std::move(series_times);
      finalize(std::move(result));
      return;
    }

    SolveRequest request;
    request.input = input;
    request.requested_backend = requested;
    request.output_path = resolved_output.string();
    request.progress = callback;

    SolveResponse response = ExecuteSolve(request);

    // Update backend info in detailed progress
    {
      std::lock_guard<std::mutex> lock(state_mutex);
      switch (response.backend_used) {
        case BackendKind::CPU:
          state.detailed_progress.backend_name = "CPU";
          break;
        case BackendKind::CUDA:
          state.detailed_progress.backend_name = "CUDA";
          break;
        case BackendKind::Metal:
          state.detailed_progress.backend_name = "Metal";
          break;
        case BackendKind::TPU:
          state.detailed_progress.backend_name = "TPU";
          break;
        default:
          state.detailed_progress.backend_name = "Auto";
          break;
      }
      if (!response.note.empty()) {
        state.detailed_progress.backend_note = response.note;
      }
      if (response.residual_l2 > 0 && response.residual_l2 < 1e-10) {
        state.detailed_progress.is_converged = true;
      }
    }
    if (!response.ok) {
      finalize(SolveResult{false, "", response.error, response.backend_used, response.note});
      return;
    }

    {
      std::lock_guard<std::mutex> lock(state_mutex);
      state.derived_fields = response.derived;
      state.has_derived_fields = true;
      state.current_domain = domain;
      state.current_pde = input.pde;
      state.current_grid = response.grid;  // Store grid for inspection tools
    }

    if (run_config_ready) {
      std::string meta_error;
      if (!WriteRunMetadataSidecar(resolved_output, run_config, requested, response.backend_used,
                                   response.note, false, -1, 0.0, &meta_error)) {
        AddLog(state, state_mutex, "metadata: " + meta_error);
      }
      RunSummaryData summary;
      summary.run_config = run_config;
      summary.requested_backend = requested;
      summary.selected_backend = response.backend_used;
      summary.backend_note = response.note;
      summary.output_path = resolved_output.string();
      summary.time_series = false;
      const double elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(
                                 std::chrono::steady_clock::now() - start_time)
                                 .count();
      summary.solve_seconds = elapsed;
      summary.total_seconds = elapsed;
      summary.residual_l2 = response.residual_l2;
      summary.residual_linf = response.residual_linf;
      summary.residual_iters = response.residual_iters;
      summary.residual_l2_history = response.residual_l2_history;
      summary.residual_linf_history = response.residual_linf_history;
      std::string summary_error;
      if (!WriteRunSummarySidecar(resolved_output, summary, &summary_error)) {
        AddLog(state, state_mutex, "summary: " + summary_error);
      }
    }
    
    finalize(SolveResult{true, resolved_output.string(), "", response.backend_used, response.note});
  });
  }
}
