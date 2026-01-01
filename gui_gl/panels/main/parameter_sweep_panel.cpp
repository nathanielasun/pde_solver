#include "parameter_sweep_panel.h"

#include "app_helpers.h"
#include "backend.h"
#include "input_parse.h"
#include "latex_parser.h"
#include "solver.h"
#include "vtk_io.h"
#include "io/file_utils.h"
#include "utils/coordinate_utils.h"
#include "utils/math_utils.h"
#include "utils/string_utils.h"
#include "shape_io.h"

#include "imgui.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <thread>
#include <utility>

namespace {

enum class SweepParam {
  GridResolution = 0,
  DomainSize = 1,
  Diffusivity = 2,
  TimeStep = 3,
  BoundaryValue = 4
};

enum class SweepStatus {
  Pending,
  Running,
  Done,
  Failed,
  Skipped
};

struct SweepRun {
  int index = 0;
  double value = 0.0;
  std::string label;
  SweepStatus status = SweepStatus::Pending;
  std::string output_path;
  std::string error;
  double elapsed_seconds = 0.0;
};

struct SweepSnapshot {
  std::string pde_text;
  int coord_mode = 0;
  double bound_xmin = 0.0;
  double bound_xmax = 1.0;
  double bound_ymin = 0.0;
  double bound_ymax = 1.0;
  double bound_zmin = 0.0;
  double bound_zmax = 1.0;
  int grid_nx = 64;
  int grid_ny = 64;
  int grid_nz = 1;
  int domain_mode = 0;
  std::string domain_shape;
  std::string domain_shape_file;
  std::string domain_shape_mask_path;
  ShapeMask shape_mask;
  double shape_mask_threshold = 0.0;
  bool shape_mask_invert = false;
  ShapeTransform shape_transform;
  BoundaryInput bc_left;
  BoundaryInput bc_right;
  BoundaryInput bc_bottom;
  BoundaryInput bc_top;
  BoundaryInput bc_front;
  BoundaryInput bc_back;
  int backend_index = 0;
  int method_index = 0;
  double sor_omega = 1.5;
  int gmres_restart = 30;
  int solver_max_iter = 2000;
  double solver_tol = 1e-6;
  int solver_residual_interval = 25;
  int solver_mg_pre_smooth = 2;
  int solver_mg_post_smooth = 2;
  int solver_mg_coarse_iters = 10;
  int solver_mg_max_levels = 5;
  int thread_count = 0;
  int metal_reduce_interval = 10;
  int metal_tg_x = 0;
  int metal_tg_y = 0;
  double time_start = 0.0;
  double time_end = 1.0;
  int time_frames = 1;
  std::string output_path;
};

struct OutputBase {
  std::filesystem::path base_dir;
  std::string base_stem;
  std::string extension;
};

struct SweepState {
  std::mutex mutex;
  std::thread worker;
  std::atomic<bool> cancel{false};
  std::atomic<bool> pause{false};
  bool running = false;
  SweepParam param = SweepParam::GridResolution;
  int total_runs = 0;
  int completed_runs = 0;
  int current_run = -1;
  double current_progress = 0.0;
  std::string current_phase;
  std::string status_line;
  std::string last_error;
  std::filesystem::path output_dir;
  bool save_all = false;
  bool gen_report = true;
  std::vector<SweepRun> runs;
};

static int s_sweep_param = 0;
static float s_param_min = 0.0f;
static float s_param_max = 1.0f;
static int s_num_steps = 5;
static SweepState s_sweep;

const char* SweepParamLabel(SweepParam param) {
  switch (param) {
    case SweepParam::GridResolution:
      return "Grid Resolution";
    case SweepParam::DomainSize:
      return "Domain Size";
    case SweepParam::Diffusivity:
      return "Diffusivity";
    case SweepParam::TimeStep:
      return "Time Step";
    case SweepParam::BoundaryValue:
      return "BC Value";
    default:
      return "Unknown";
  }
}

const char* SweepStatusLabel(SweepStatus status) {
  switch (status) {
    case SweepStatus::Pending:
      return "Pending";
    case SweepStatus::Running:
      return "Running";
    case SweepStatus::Done:
      return "Done";
    case SweepStatus::Failed:
      return "Failed";
    case SweepStatus::Skipped:
      return "Skipped";
    default:
      return "Unknown";
  }
}

ImVec4 SweepStatusColor(SweepStatus status) {
  switch (status) {
    case SweepStatus::Done:
      return ImVec4(0.3f, 1.0f, 0.3f, 1.0f);
    case SweepStatus::Failed:
      return ImVec4(1.0f, 0.3f, 0.3f, 1.0f);
    case SweepStatus::Skipped:
      return ImVec4(1.0f, 0.7f, 0.3f, 1.0f);
    case SweepStatus::Running:
      return ImVec4(0.4f, 0.7f, 1.0f, 1.0f);
    case SweepStatus::Pending:
    default:
      return ImVec4(0.7f, 0.7f, 0.7f, 1.0f);
  }
}

std::string FormatValue(double value, int precision) {
  std::ostringstream out;
  out << std::setprecision(precision) << std::fixed << value;
  return out.str();
}

std::string FormatTagValue(double value) {
  std::string out = FormatValue(value, 4);
  ReplaceAll(&out, "-", "m");
  ReplaceAll(&out, ".", "p");
  ReplaceAll(&out, "+", "");
  return out;
}

std::string PadIndex(int index, int digits) {
  std::ostringstream out;
  out << std::setw(digits) << std::setfill('0') << index;
  return out.str();
}

std::string BuildRunLabel(SweepParam param, double value, int index, int digits) {
  std::ostringstream out;
  out << "run_" << PadIndex(index, digits) << "_";
  switch (param) {
    case SweepParam::GridResolution:
      out << "grid_" << static_cast<int>(std::round(value));
      break;
    case SweepParam::DomainSize:
      out << "size_" << FormatTagValue(value);
      break;
    case SweepParam::Diffusivity:
      out << "diff_" << FormatTagValue(value);
      break;
    case SweepParam::TimeStep:
      out << "dt_" << FormatTagValue(value);
      break;
    case SweepParam::BoundaryValue:
      out << "bc_" << FormatTagValue(value);
      break;
    default:
      out << "param_" << FormatTagValue(value);
      break;
  }
  return out.str();
}

std::vector<double> BuildSweepValues(double min_value, double max_value, int steps,
                                     bool integer_values) {
  std::vector<double> values;
  steps = std::max(1, steps);
  if (steps == 1) {
    values.push_back(min_value);
    return values;
  }
  const double step = (max_value - min_value) / static_cast<double>(steps - 1);
  values.reserve(static_cast<size_t>(steps));
  for (int i = 0; i < steps; ++i) {
    double value = min_value + step * static_cast<double>(i);
    if (integer_values) {
      value = std::round(value);
    }
    values.push_back(value);
  }
  if (!integer_values) {
    return values;
  }
  std::vector<double> unique_values;
  int last = std::numeric_limits<int>::min();
  for (double value : values) {
    int rounded = static_cast<int>(std::round(value));
    if (rounded != last) {
      unique_values.push_back(static_cast<double>(rounded));
      last = rounded;
    }
  }
  return unique_values;
}

SweepSnapshot CaptureSnapshot(const ParameterSweepPanelState& state) {
  SweepSnapshot snapshot;
  snapshot.pde_text = state.pde_text;
  snapshot.coord_mode = state.coord_mode;
  snapshot.bound_xmin = state.bound_xmin;
  snapshot.bound_xmax = state.bound_xmax;
  snapshot.bound_ymin = state.bound_ymin;
  snapshot.bound_ymax = state.bound_ymax;
  snapshot.bound_zmin = state.bound_zmin;
  snapshot.bound_zmax = state.bound_zmax;
  snapshot.grid_nx = state.grid_nx;
  snapshot.grid_ny = state.grid_ny;
  snapshot.grid_nz = state.grid_nz;
  snapshot.domain_mode = state.domain_mode;
  snapshot.domain_shape = state.domain_shape;
  snapshot.domain_shape_file = state.domain_shape_file;
  snapshot.domain_shape_mask_path = state.domain_shape_mask_path;
  snapshot.shape_mask = state.shape_mask;
  snapshot.shape_mask_threshold = state.shape_mask_threshold;
  snapshot.shape_mask_invert = state.shape_mask_invert;
  snapshot.shape_transform = state.shape_transform;
  snapshot.bc_left = state.bc_left;
  snapshot.bc_right = state.bc_right;
  snapshot.bc_bottom = state.bc_bottom;
  snapshot.bc_top = state.bc_top;
  snapshot.bc_front = state.bc_front;
  snapshot.bc_back = state.bc_back;
  snapshot.backend_index = state.backend_index;
  snapshot.method_index = state.method_index;
  snapshot.sor_omega = state.sor_omega;
  snapshot.gmres_restart = state.gmres_restart;
  snapshot.solver_max_iter = state.solver_max_iter;
  snapshot.solver_tol = state.solver_tol;
  snapshot.solver_residual_interval = state.solver_residual_interval;
  snapshot.solver_mg_pre_smooth = state.solver_mg_pre_smooth;
  snapshot.solver_mg_post_smooth = state.solver_mg_post_smooth;
  snapshot.solver_mg_coarse_iters = state.solver_mg_coarse_iters;
  snapshot.solver_mg_max_levels = state.solver_mg_max_levels;
  snapshot.thread_count = state.thread_count;
  snapshot.metal_reduce_interval = state.metal_reduce_interval;
  snapshot.metal_tg_x = state.metal_tg_x;
  snapshot.metal_tg_y = state.metal_tg_y;
  snapshot.time_start = state.time_start;
  snapshot.time_end = state.time_end;
  snapshot.time_frames = state.time_frames;
  snapshot.output_path = state.output_path;
  return snapshot;
}

bool IsSolverRunning(const ParameterSweepPanelState& state) {
  std::lock_guard<std::mutex> lock(state.shared_state_mutex);
  return state.shared_state.running;
}

OutputBase ResolveOutputBase(const std::string& output_path) {
  OutputBase base;
  base.base_dir = "outputs";
  base.base_stem = "solution";
  base.extension = ".vtk";

  if (output_path.empty()) {
    return base;
  }

  std::filesystem::path path(output_path);
  const bool ends_with_slash = !output_path.empty() && output_path.back() == '/';
  const bool is_dir = std::filesystem::exists(path) && std::filesystem::is_directory(path);
  if (ends_with_slash || is_dir) {
    base.base_dir = path;
    return base;
  }

  if (!path.has_extension()) {
    path += ".vtk";
  }
  std::string ext = path.extension().string();
  if (ext != ".vtk" && ext != ".vti") {
    ext = ".vtk";
  }
  base.extension = ext;
  base.base_stem = path.stem().string();
  base.base_dir = path.parent_path().empty() ? std::filesystem::path("outputs") : path.parent_path();
  return base;
}

void ScaleBounds(double* min_val, double* max_val, double scale) {
  if (!min_val || !max_val) {
    return;
  }
  double center = 0.5 * (*min_val + *max_val);
  double half = 0.5 * (*max_val - *min_val);
  if (half <= 1e-12) {
    half = 0.5;
  }
  double new_half = half * scale;
  *min_val = center - new_half;
  *max_val = center + new_half;
}

void ApplyBoundaryValue(BoundaryInput* input, double value) {
  if (!input) {
    return;
  }
  std::string formatted = FormatValue(value, 6);
  if (input->kind == 2) {
    input->gamma = formatted;
  } else {
    input->value = formatted;
  }
}

void ScaleLatexCoeff(std::string* expr, double scale) {
  if (!expr || expr->empty()) {
    return;
  }
  if (std::abs(scale - 1.0) < 1e-12) {
    return;
  }
  std::ostringstream out;
  out << std::setprecision(8) << scale;
  *expr = "(" + out.str() + ")*(" + *expr + ")";
}

void ScaleDiffusivity(PDECoefficients* coeffs, double scale) {
  if (!coeffs) {
    return;
  }
  coeffs->a *= scale;
  coeffs->b *= scale;
  coeffs->az *= scale;
  ScaleLatexCoeff(&coeffs->a_latex, scale);
  ScaleLatexCoeff(&coeffs->b_latex, scale);
  ScaleLatexCoeff(&coeffs->az_latex, scale);
}

enum class BuildResult {
  Ok,
  Failed,
  Skipped
};

BuildResult BuildSolveInput(const SweepSnapshot& snapshot, const LatexParseResult& parsed,
                            bool use_surface, bool use_axisymmetric, bool use_volume,
                            bool time_dependent, SweepParam param, double param_value,
                            SolveInput* input, std::string* message) {
  if (!input) {
    if (message) {
      *message = "internal error: missing input";
    }
    return BuildResult::Failed;
  }

  double xmin = snapshot.bound_xmin;
  double xmax = snapshot.bound_xmax;
  double ymin = snapshot.bound_ymin;
  double ymax = snapshot.bound_ymax;
  double zmin = snapshot.bound_zmin;
  double zmax = snapshot.bound_zmax;
  int nx = snapshot.grid_nx;
  int ny = snapshot.grid_ny;
  int nz = snapshot.grid_nz;

  if (param == SweepParam::GridResolution) {
    const int grid_value = std::max(3, static_cast<int>(std::round(param_value)));
    nx = grid_value;
    ny = grid_value;
    if (use_volume) {
      nz = grid_value;
    } else {
      nz = 1;
    }
  } else if (param == SweepParam::DomainSize) {
    if (param_value <= 0.0) {
      if (message) {
        *message = "domain size must be > 0";
      }
      return BuildResult::Failed;
    }
    ScaleBounds(&xmin, &xmax, param_value);
    ScaleBounds(&ymin, &ymax, param_value);
    if (use_volume) {
      ScaleBounds(&zmin, &zmax, param_value);
    }
  }

  const int min_grid = 3;
  nx = std::max(nx, min_grid);
  ny = std::max(ny, min_grid);
  if (use_volume) {
    nz = std::max(nz, min_grid);
  } else {
    nz = 1;
  }

  Domain domain;
  std::string domain_bounds;
  std::string grid_text;
  if (use_volume) {
    domain_bounds = FormatBounds3D(xmin, xmax, ymin, ymax, zmin, zmax);
    grid_text = FormatGrid3D(nx, ny, nz);
  } else {
    domain_bounds = FormatBounds(xmin, xmax, ymin, ymax);
    grid_text = FormatGrid(nx, ny);
  }

  ParseResult domain_result = ParseDomain(domain_bounds, &domain);
  if (!domain_result.ok) {
    if (message) {
      *message = domain_result.error;
    }
    return BuildResult::Failed;
  }
  ParseResult grid_result = ParseGrid(grid_text, &domain);
  if (!grid_result.ok) {
    if (message) {
      *message = grid_result.error;
    }
    return BuildResult::Failed;
  }

  domain.coord_system = CoordModeToSystem(snapshot.coord_mode);

  std::string domain_shape_eval = snapshot.domain_shape;
  if (domain_shape_eval.empty() && !snapshot.domain_shape_file.empty()) {
    std::string shape_error;
    if (!LoadShapeExpressionFromFile(snapshot.domain_shape_file, &domain_shape_eval, &shape_error)) {
      if (message) {
        *message = "shape file error: " + shape_error;
      }
      return BuildResult::Failed;
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
  ShapeMask shape_mask = snapshot.shape_mask;
  if (snapshot.domain_mode == 1 &&
      !snapshot.domain_shape_mask_path.empty() &&
      !HasShapeMask(shape_mask)) {
    std::string mask_error;
    if (!LoadShapeMaskFromVtk(snapshot.domain_shape_mask_path, &shape_mask, &mask_error)) {
      if (message) {
        *message = "shape mask error: " + mask_error;
      }
      return BuildResult::Failed;
    }
  }
  if (snapshot.domain_mode == 1 && Trim(domain_shape_eval).empty() && !HasShapeMask(shape_mask)) {
    if (message) {
      *message = "implicit domain requires a shape function or mask";
    }
    return BuildResult::Failed;
  }

  BoundaryInput bc_left = snapshot.bc_left;
  BoundaryInput bc_right = snapshot.bc_right;
  BoundaryInput bc_bottom = snapshot.bc_bottom;
  BoundaryInput bc_top = snapshot.bc_top;
  BoundaryInput bc_front = snapshot.bc_front;
  BoundaryInput bc_back = snapshot.bc_back;

  if (param == SweepParam::BoundaryValue) {
    ApplyBoundaryValue(&bc_left, param_value);
    ApplyBoundaryValue(&bc_right, param_value);
    ApplyBoundaryValue(&bc_bottom, param_value);
    ApplyBoundaryValue(&bc_top, param_value);
    ApplyBoundaryValue(&bc_front, param_value);
    ApplyBoundaryValue(&bc_back, param_value);
  }

  if (use_surface || use_axisymmetric) {
    auto remap_bc = [&](BoundaryInput* bc) {
      if (!bc) {
        return;
      }
      if (use_surface) {
        bc->value = RemapSphericalSurfaceExpr(bc->value);
        bc->alpha = RemapSphericalSurfaceExpr(bc->alpha);
        bc->beta = RemapSphericalSurfaceExpr(bc->beta);
        bc->gamma = RemapSphericalSurfaceExpr(bc->gamma);
      }
      if (use_axisymmetric) {
        bc->value = RemapAxisymmetricExpr(bc->value);
        bc->alpha = RemapAxisymmetricExpr(bc->alpha);
        bc->beta = RemapAxisymmetricExpr(bc->beta);
        bc->gamma = RemapAxisymmetricExpr(bc->gamma);
      }
    };
    remap_bc(&bc_left);
    remap_bc(&bc_right);
    remap_bc(&bc_bottom);
    remap_bc(&bc_top);
    if (use_volume) {
      remap_bc(&bc_front);
    }
    remap_bc(&bc_back);
  }

  std::string bc_spec;
  std::string bc_error;
  if (!BuildBoundarySpec(bc_left, bc_right, bc_bottom, bc_top, bc_front, bc_back, &bc_spec,
                         &bc_error)) {
    if (message) {
      *message = "boundary error: " + bc_error;
    }
    return BuildResult::Failed;
  }
  BoundarySet bc;
  ParseResult bc_result = ApplyBoundarySpec(bc_spec, &bc);
  if (!bc_result.ok) {
    if (message) {
      *message = "boundary error: " + bc_result.error;
    }
    return BuildResult::Failed;
  }

  input->pde = parsed.coeffs;
  if (param == SweepParam::Diffusivity) {
    ScaleDiffusivity(&input->pde, param_value);
  }
  input->integrals = parsed.integrals;
  input->nonlinear = parsed.nonlinear;
  input->nonlinear_derivatives = parsed.nonlinear_derivatives;
  input->domain = domain;
  input->bc = bc;
  input->solver.max_iter = snapshot.solver_max_iter;
  input->solver.tol = snapshot.solver_tol;
  input->solver.residual_interval = snapshot.solver_residual_interval;
  input->solver.thread_count = snapshot.thread_count;
  input->solver.metal_reduce_interval = snapshot.metal_reduce_interval;
  input->solver.metal_tg_x = snapshot.metal_tg_x;
  input->solver.metal_tg_y = snapshot.metal_tg_y;
  input->solver.method = MethodFromIndex(snapshot.method_index);
  input->solver.sor_omega = snapshot.sor_omega;
  input->solver.gmres_restart = snapshot.gmres_restart;
  input->solver.mg_pre_smooth = snapshot.solver_mg_pre_smooth;
  input->solver.mg_post_smooth = snapshot.solver_mg_post_smooth;
  input->solver.mg_coarse_iters = snapshot.solver_mg_coarse_iters;
  input->solver.mg_max_levels = snapshot.solver_mg_max_levels;

  if (snapshot.domain_mode == 1) {
    input->domain_shape = domain_shape_eval;
    input->shape_mask = std::move(shape_mask);
    input->shape_transform = snapshot.shape_transform;
    input->shape_mask_threshold = snapshot.shape_mask_threshold;
    input->shape_mask_invert = snapshot.shape_mask_invert;
  }

  if (param == SweepParam::TimeStep && !time_dependent) {
    if (message) {
      *message = "time step sweep requires u_t or u_tt";
    }
    return BuildResult::Skipped;
  }

  if (time_dependent) {
    input->time.enabled = true;
    input->time.t_start = snapshot.time_start;
    input->time.frames = std::max(1, snapshot.time_frames);
    if (param == SweepParam::TimeStep) {
      if (param_value <= 0.0) {
        if (message) {
          *message = "time step must be > 0";
        }
        return BuildResult::Failed;
      }
      input->time.dt = param_value;
      input->time.t_end = snapshot.time_start +
                          param_value * static_cast<double>(std::max(0, input->time.frames - 1));
    } else {
      if (snapshot.time_end < snapshot.time_start) {
        if (message) {
          *message = "time range must satisfy t end >= t start";
        }
        return BuildResult::Failed;
      }
      input->time.t_end = snapshot.time_end;
      if (input->time.frames > 1) {
        input->time.dt =
            (snapshot.time_end - snapshot.time_start) /
            static_cast<double>(std::max(1, input->time.frames - 1));
      } else {
        input->time.dt = 0.0;
      }
    }
  }

  return BuildResult::Ok;
}

std::string CsvEscape(const std::string& text) {
  if (text.find_first_of(",\"\n") == std::string::npos) {
    return text;
  }
  std::string out = "\"";
  for (char c : text) {
    if (c == '"') {
      out += "\"\"";
    } else {
      out += c;
    }
  }
  out += "\"";
  return out;
}

bool WriteSweepReport(const std::filesystem::path& path, const std::vector<SweepRun>& runs,
                      const std::string& param_label, std::string* error) {
  std::ofstream file(path);
  if (!file) {
    if (error) {
      *error = "failed to write report: " + path.string();
    }
    return false;
  }
  file << "index,param,value,status,elapsed_s,output_path,error\n";
  for (const SweepRun& run : runs) {
    file << run.index << "," << CsvEscape(param_label) << "," << run.value << ","
         << CsvEscape(SweepStatusLabel(run.status)) << ","
         << run.elapsed_seconds << "," << CsvEscape(run.output_path) << ","
         << CsvEscape(run.error) << "\n";
  }
  return true;
}

void JoinSweepIfFinished() {
  if (!s_sweep.worker.joinable()) {
    return;
  }
  bool running = false;
  {
    std::lock_guard<std::mutex> lock(s_sweep.mutex);
    running = s_sweep.running;
  }
  if (!running) {
    s_sweep.worker.join();
  }
}

void MarkRemainingSkipped(int start_index, const std::string& reason) {
  std::lock_guard<std::mutex> lock(s_sweep.mutex);
  for (size_t i = static_cast<size_t>(start_index); i < s_sweep.runs.size(); ++i) {
    if (s_sweep.runs[i].status == SweepStatus::Pending ||
        s_sweep.runs[i].status == SweepStatus::Running) {
      s_sweep.runs[i].status = SweepStatus::Skipped;
      s_sweep.runs[i].error = reason;
    }
  }
  s_sweep.completed_runs = s_sweep.total_runs;
}

void RunSweep(const SweepSnapshot snapshot, const std::vector<double> values,
              const SweepParam param, const bool save_all, const bool gen_report) {
  const bool use_cartesian_3d = (snapshot.coord_mode == CoordMode::kCartesian3D);
  const bool use_axisymmetric = (snapshot.coord_mode == CoordMode::kAxisymmetric);
  const bool use_cylindrical_volume = (snapshot.coord_mode == CoordMode::kCylindricalVolume);
  const bool use_spherical_surface = (snapshot.coord_mode == CoordMode::kSphericalSurface);
  const bool use_spherical_volume = (snapshot.coord_mode == CoordMode::kSphericalVolume);
  const bool use_toroidal_surface = (snapshot.coord_mode == CoordMode::kToroidalSurface);
  const bool use_toroidal_volume = (snapshot.coord_mode == CoordMode::kToroidalVolume);
  const bool use_surface = use_spherical_surface || use_toroidal_surface;
  const bool use_volume =
      use_spherical_volume || use_toroidal_volume || use_cartesian_3d || use_cylindrical_volume;

  OutputBase output_base = ResolveOutputBase(snapshot.output_path);
  std::filesystem::path sweep_dir;
  if (save_all || gen_report) {
    std::error_code ec;
    std::filesystem::create_directories(output_base.base_dir, ec);
    if (ec) {
      std::lock_guard<std::mutex> lock(s_sweep.mutex);
      s_sweep.last_error = "failed to create output directory: " + ec.message();
      for (auto& run : s_sweep.runs) {
        run.status = SweepStatus::Failed;
        run.error = s_sweep.last_error;
      }
      s_sweep.completed_runs = s_sweep.total_runs;
      s_sweep.running = false;
      return;
    }
    const std::string sweep_name = output_base.base_stem + "_sweep_" + GenerateRandomTag(6);
    sweep_dir = output_base.base_dir / sweep_name;
    std::filesystem::create_directories(sweep_dir, ec);
    if (ec) {
      std::lock_guard<std::mutex> lock(s_sweep.mutex);
      s_sweep.last_error = "failed to create sweep directory: " + ec.message();
      for (auto& run : s_sweep.runs) {
        run.status = SweepStatus::Failed;
        run.error = s_sweep.last_error;
      }
      s_sweep.completed_runs = s_sweep.total_runs;
      s_sweep.running = false;
      return;
    }
  }

  {
    std::lock_guard<std::mutex> lock(s_sweep.mutex);
    s_sweep.output_dir = sweep_dir;
    s_sweep.status_line = "Parsing PDE...";
  }

  LatexParser parser;
  std::string pde_for_parse = snapshot.pde_text;
  if (use_surface) {
    pde_for_parse = RemapSphericalSurfaceExpr(pde_for_parse);
  }
  if (use_axisymmetric) {
    pde_for_parse = RemapAxisymmetricExpr(pde_for_parse);
  }
  LatexParseResult parse_result = parser.Parse(pde_for_parse);
  if (!parse_result.ok) {
    std::lock_guard<std::mutex> lock(s_sweep.mutex);
    s_sweep.last_error = "LaTeX parse error: " + parse_result.error;
    for (auto& run : s_sweep.runs) {
      run.status = SweepStatus::Failed;
      run.error = s_sweep.last_error;
    }
    s_sweep.completed_runs = s_sweep.total_runs;
    s_sweep.running = false;
    return;
  }

  const bool time_dependent =
      std::abs(parse_result.coeffs.ut) > 1e-12 || std::abs(parse_result.coeffs.utt) > 1e-12;

  const int total_runs = static_cast<int>(values.size());
  const int digits = std::max(2, static_cast<int>(std::to_string(total_runs).size()));

  for (int run_index = 0; run_index < total_runs; ++run_index) {
    if (s_sweep.cancel.load()) {
      MarkRemainingSkipped(run_index, "cancelled");
      break;
    }
    while (s_sweep.pause.load()) {
      {
        std::lock_guard<std::mutex> lock(s_sweep.mutex);
        s_sweep.status_line = "Paused";
      }
      if (s_sweep.cancel.load()) {
        break;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(120));
    }

    const double param_value = values[static_cast<size_t>(run_index)];
    const std::string run_label = BuildRunLabel(param, param_value, run_index + 1, digits);

    {
      std::lock_guard<std::mutex> lock(s_sweep.mutex);
      s_sweep.current_run = run_index;
      s_sweep.current_phase = "setup";
      s_sweep.current_progress = 0.0;
      s_sweep.status_line = "Running " + std::to_string(run_index + 1) + "/" +
                            std::to_string(total_runs);
      if (static_cast<size_t>(run_index) < s_sweep.runs.size()) {
        s_sweep.runs[static_cast<size_t>(run_index)].status = SweepStatus::Running;
        s_sweep.runs[static_cast<size_t>(run_index)].label = run_label;
      }
    }

    const auto start_time = std::chrono::steady_clock::now();
    SolveInput input;
    input.cancel = &s_sweep.cancel;
    std::string build_error;
    const BuildResult build_result =
        BuildSolveInput(snapshot, parse_result, use_surface, use_axisymmetric, use_volume,
                        time_dependent, param, param_value, &input, &build_error);

    if (build_result == BuildResult::Skipped) {
      std::lock_guard<std::mutex> lock(s_sweep.mutex);
      s_sweep.runs[static_cast<size_t>(run_index)].status = SweepStatus::Skipped;
      s_sweep.runs[static_cast<size_t>(run_index)].error = build_error;
      s_sweep.completed_runs++;
      s_sweep.current_progress = 1.0;
      continue;
    }
    if (build_result == BuildResult::Failed) {
      std::lock_guard<std::mutex> lock(s_sweep.mutex);
      s_sweep.runs[static_cast<size_t>(run_index)].status = SweepStatus::Failed;
      s_sweep.runs[static_cast<size_t>(run_index)].error = build_error;
      s_sweep.completed_runs++;
      s_sweep.current_progress = 1.0;
      continue;
    }

    const BackendKind requested = BackendFromIndex(snapshot.backend_index);
    std::string note;
    BackendKind selected = BackendKind::CPU;

    ProgressCallback progress = [&](const std::string& phase, double value) {
      std::unique_lock<std::mutex> lock(s_sweep.mutex, std::try_to_lock);
      if (!lock.owns_lock()) {
        return;
      }
      s_sweep.current_phase = phase;
      if (phase == "solve" || phase == "time") {
        s_sweep.current_progress = value;
      }
    };

    bool ok = true;
    std::string error;
    std::string output_path;

    if (input.time.enabled) {
      if (requested != BackendKind::CPU && requested != BackendKind::Auto) {
        note = "time stepping uses CPU";
      }

      std::filesystem::path series_base;
      std::vector<std::string> frame_paths;
      std::vector<double> frame_times;
      std::string frame_error;
      if (save_all) {
        std::error_code ec;
        const std::filesystem::path run_dir = sweep_dir / run_label;
        std::filesystem::create_directories(run_dir, ec);
        if (ec) {
          ok = false;
          error = "failed to create run directory: " + ec.message();
        } else {
          series_base = run_dir / (output_base.base_stem + output_base.extension);
        }
      }

      if (ok) {
        const int frame_digits = FrameDigits(input.time.frames);
        auto frame_cb = [&](int frame, double t, const std::vector<double>& grid,
                            const std::vector<double>*) -> bool {
          if (s_sweep.cancel.load()) {
            return false;
          }
          if (save_all) {
            const std::filesystem::path frame_path =
                BuildFramePath(series_base, frame, frame_digits);
            VtkWriteResult write_result;
            if (frame_path.extension() == ".vti") {
              write_result = WriteVtkXmlImageData(frame_path.string(), input.domain, grid);
            } else {
              write_result = WriteVtkStructuredPoints(frame_path.string(), input.domain, grid);
            }
            if (!write_result.ok) {
              frame_error = write_result.error;
              return false;
            }
            frame_paths.push_back(frame_path.string());
            frame_times.push_back(t);
          }
          {
            std::unique_lock<std::mutex> lock(s_sweep.mutex, std::try_to_lock);
            if (lock.owns_lock()) {
              s_sweep.status_line = "Solved frame " + std::to_string(frame + 1) + "/" +
                                    std::to_string(std::max(1, input.time.frames));
            }
          }
          return true;
        };

        SolveOutput output = SolvePDETimeSeries(input, frame_cb, progress);
        if (!output.error.empty()) {
          ok = false;
          error = output.error;
        } else if (!frame_error.empty()) {
          ok = false;
          error = frame_error;
        } else if (save_all) {
          if (!frame_paths.empty()) {
            const std::filesystem::path manifest =
                series_base.parent_path() / (output_base.base_stem + "_series.pvd");
            WriteVtkSeriesPvd(manifest.string(), frame_paths, frame_times);
            output_path = manifest.string();
          } else {
            output_path = series_base.parent_path().string();
          }
        }
      }
    } else {
      SolveOutput output = SolveWithBackend(input, requested, &selected, &note, progress);
      if (!output.error.empty()) {
        ok = false;
        error = output.error;
      } else if (save_all) {
        std::filesystem::path out_path = sweep_dir / (run_label + output_base.extension);
        VtkWriteResult write_result;
        if (out_path.extension() == ".vti") {
          write_result = WriteVtkXmlImageData(out_path.string(), input.domain, output.grid);
        } else {
          write_result = WriteVtkStructuredPoints(out_path.string(), input.domain, output.grid);
        }
        if (!write_result.ok) {
          ok = false;
          error = write_result.error;
        } else {
          output_path = out_path.string();
        }
      }
    }

    const auto end_time = std::chrono::steady_clock::now();
    const double elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(
                               end_time - start_time)
                               .count();

    {
      std::lock_guard<std::mutex> lock(s_sweep.mutex);
      SweepRun& run = s_sweep.runs[static_cast<size_t>(run_index)];
      run.elapsed_seconds = elapsed;
      run.output_path = output_path;
      run.error = error;
      run.status = ok ? SweepStatus::Done : SweepStatus::Failed;
      s_sweep.completed_runs++;
      s_sweep.current_progress = 1.0;
    }
  }

  if (gen_report && !sweep_dir.empty()) {
    std::vector<SweepRun> runs_copy;
    {
      std::lock_guard<std::mutex> lock(s_sweep.mutex);
      runs_copy = s_sweep.runs;
    }
    const std::filesystem::path report_path = sweep_dir / "sweep_report.csv";
    std::string report_error;
    WriteSweepReport(report_path, runs_copy, SweepParamLabel(param), &report_error);
  }

  {
    std::lock_guard<std::mutex> lock(s_sweep.mutex);
    s_sweep.running = false;
    s_sweep.current_run = -1;
    s_sweep.current_phase = "idle";
    if (s_sweep.cancel.load()) {
      s_sweep.status_line = "Sweep cancelled";
    } else {
      s_sweep.status_line = "Sweep complete";
    }
  }
}

void StartSweep(const ParameterSweepPanelState& state, SweepParam param, double min_val,
                double max_val, int steps, bool save_all, bool gen_report) {
  {
    std::lock_guard<std::mutex> lock(s_sweep.mutex);
    if (s_sweep.running) {
      s_sweep.last_error = "Sweep already running.";
      return;
    }
  }
  if (IsSolverRunning(state)) {
    std::lock_guard<std::mutex> lock(s_sweep.mutex);
    s_sweep.last_error = "Solver is running; stop it before starting a sweep.";
    return;
  }
  if (Trim(state.pde_text).empty()) {
    std::lock_guard<std::mutex> lock(s_sweep.mutex);
    s_sweep.last_error = "PDE input is empty.";
    return;
  }

  JoinSweepIfFinished();

  std::vector<double> values =
      BuildSweepValues(min_val, max_val, steps, param == SweepParam::GridResolution);
  if (values.empty()) {
    std::lock_guard<std::mutex> lock(s_sweep.mutex);
    s_sweep.last_error = "no sweep values specified";
    return;
  }

  {
    std::lock_guard<std::mutex> lock(s_sweep.mutex);
    s_sweep.cancel.store(false);
    s_sweep.pause.store(false);
    s_sweep.running = true;
    s_sweep.param = param;
    s_sweep.total_runs = static_cast<int>(values.size());
    s_sweep.completed_runs = 0;
    s_sweep.current_run = -1;
    s_sweep.current_phase = "setup";
    s_sweep.current_progress = 0.0;
    s_sweep.status_line = "Initializing sweep...";
    s_sweep.last_error.clear();
    s_sweep.save_all = save_all;
    s_sweep.gen_report = gen_report;
    s_sweep.runs.clear();
    s_sweep.runs.reserve(values.size());
    for (size_t i = 0; i < values.size(); ++i) {
      SweepRun run;
      run.index = static_cast<int>(i + 1);
      run.value = values[i];
      run.status = SweepStatus::Pending;
      s_sweep.runs.push_back(run);
    }
  }

  const SweepSnapshot snapshot = CaptureSnapshot(state);

  s_sweep.worker = std::thread([snapshot, values, param, save_all, gen_report]() {
    RunSweep(snapshot, values, param, save_all, gen_report);
  });
}

}  // namespace

void RenderParameterSweepPanel(ParameterSweepPanelState& state,
                                const std::vector<std::string>& components) {
  (void)components;
  JoinSweepIfFinished();

  ImGui::Text("Parameter Sweep");
  ImGui::Separator();
  ImGui::Spacing();

  ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f),
                     "Run batch solves with varying parameters.");
  ImGui::Spacing();

  ImGui::Text("Sweep Parameter:");
  ImGui::Combo("##sweep_param", &s_sweep_param,
               "Grid Resolution\0Domain Size\0Diffusivity\0Time Step\0BC Value\0");

  ImGui::Spacing();
  ImGui::Text("Range:");
  ImGui::SetNextItemWidth(state.input_width * 0.35f);
  ImGui::InputFloat("Min", &s_param_min);
  ImGui::SameLine();
  ImGui::SetNextItemWidth(state.input_width * 0.35f);
  ImGui::InputFloat("Max", &s_param_max);
  ImGui::SetNextItemWidth(state.input_width * 0.35f);
  ImGui::InputInt("Steps", &s_num_steps);
  s_num_steps = std::clamp(s_num_steps, 1, 200);

  if (s_sweep_param == static_cast<int>(SweepParam::GridResolution)) {
    ImGui::TextDisabled("Grid sweep sets nx = ny (= nz for 3D).");
  } else if (s_sweep_param == static_cast<int>(SweepParam::DomainSize)) {
    ImGui::TextDisabled("Domain size acts as a scale factor (1.0 = current size).");
  } else if (s_sweep_param == static_cast<int>(SweepParam::TimeStep)) {
    ImGui::TextDisabled("Time step sweep requires u_t or u_tt in the PDE.");
  }

  ImGui::Spacing();
  ImGui::Separator();
  ImGui::Spacing();

  ImGui::Text("Output Options:");
  static bool save_all = false;
  static bool gen_report = true;
  ImGui::Checkbox("Save all solutions", &save_all);
  ImGui::Checkbox("Generate report", &gen_report);

  ImGui::Spacing();

  bool running = false;
  {
    std::lock_guard<std::mutex> lock(s_sweep.mutex);
    running = s_sweep.running;
  }

  if (!running) {
    if (ImGui::Button("Run Sweep", ImVec2(-1, 30))) {
      StartSweep(state, static_cast<SweepParam>(s_sweep_param), s_param_min, s_param_max,
                 s_num_steps, save_all, gen_report);
    }
    if (ImGui::Button("Clear Results", ImVec2(-1, 0))) {
      std::lock_guard<std::mutex> lock(s_sweep.mutex);
      s_sweep.runs.clear();
      s_sweep.last_error.clear();
      s_sweep.status_line.clear();
      s_sweep.output_dir.clear();
    }
  } else {
    const bool paused = s_sweep.pause.load();
    if (ImGui::Button(paused ? "Resume" : "Pause", ImVec2(-1, 0))) {
      s_sweep.pause.store(!paused);
    }
    if (ImGui::Button("Cancel Sweep", ImVec2(-1, 0))) {
      s_sweep.cancel.store(true);
    }
  }

  {
    std::lock_guard<std::mutex> lock(s_sweep.mutex);
    if (!s_sweep.last_error.empty()) {
      ImGui::Spacing();
      ImGui::TextColored(ImVec4(1.0f, 0.5f, 0.3f, 1.0f), "%s", s_sweep.last_error.c_str());
    }
    if (!s_sweep.status_line.empty()) {
      ImGui::Spacing();
      ImGui::Text("Status: %s", s_sweep.status_line.c_str());
    }
    if (!s_sweep.output_dir.empty()) {
      ImGui::TextDisabled("Sweep output: %s", s_sweep.output_dir.string().c_str());
    }
  }

  ImGui::Spacing();
  ImGui::Separator();
  ImGui::Spacing();

  int total_runs = 0;
  int completed_runs = 0;
  double current_progress = 0.0;
  std::string current_phase;
  {
    std::lock_guard<std::mutex> lock(s_sweep.mutex);
    total_runs = s_sweep.total_runs;
    completed_runs = s_sweep.completed_runs;
    current_progress = s_sweep.current_progress;
    current_phase = s_sweep.current_phase;
  }

  if (total_runs > 0) {
    const double overall = total_runs == 0 ? 0.0 :
        static_cast<double>(completed_runs) / static_cast<double>(total_runs);
    ImGui::Text("Overall Progress: %d / %d", completed_runs, total_runs);
    ImGui::ProgressBar(static_cast<float>(overall), ImVec2(-1, 0));
    if (!current_phase.empty()) {
      ImGui::TextDisabled("Current phase: %s (%.1f%%)", current_phase.c_str(),
                          std::max(0.0, std::min(1.0, current_progress)) * 100.0);
    }
  }

  ImGui::Spacing();
  ImGui::Separator();
  ImGui::Spacing();

  static int s_filter_status = 0;
  static char s_filter_text[64] = "";
  ImGui::Text("Filter Runs:");
  ImGui::Combo("##sweep_filter_status", &s_filter_status,
               "All\0Pending\0Running\0Done\0Failed\0Skipped\0");
  ImGui::SameLine();
  ImGui::SetNextItemWidth(state.input_width * 0.5f);
  ImGui::InputText("Search", s_filter_text, sizeof(s_filter_text));

  ImGui::Spacing();

  std::vector<SweepRun> runs_copy;
  {
    std::lock_guard<std::mutex> lock(s_sweep.mutex);
    runs_copy = s_sweep.runs;
  }

  if (runs_copy.empty()) {
    ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "No sweep runs yet.");
    return;
  }

  const std::string filter_text = ToLower(std::string(s_filter_text));
  if (ImGui::BeginTable("sweep_runs", 5,
                        ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg |
                        ImGuiTableFlags_SizingStretchSame)) {
    ImGui::TableSetupColumn("Run");
    ImGui::TableSetupColumn("Value");
    ImGui::TableSetupColumn("Status");
    ImGui::TableSetupColumn("Output");
    ImGui::TableSetupColumn("Notes");
    ImGui::TableHeadersRow();

    for (const SweepRun& run : runs_copy) {
      if (s_filter_status != 0) {
        const int status_index = static_cast<int>(run.status) + 1;
        if (status_index != s_filter_status) {
          continue;
        }
      }
      if (!filter_text.empty()) {
        std::string haystack = ToLower(run.label + " " + run.output_path + " " + run.error);
        if (haystack.find(filter_text) == std::string::npos) {
          continue;
        }
      }
      ImGui::TableNextRow();
      ImGui::TableNextColumn();
      ImGui::Text("%d", run.index);
      ImGui::TableNextColumn();
      if (s_sweep.param == SweepParam::GridResolution) {
        ImGui::Text("%d", static_cast<int>(std::round(run.value)));
      } else {
        ImGui::Text("%.4g", run.value);
      }
      ImGui::TableNextColumn();
      ImGui::TextColored(SweepStatusColor(run.status), "%s", SweepStatusLabel(run.status));
      ImGui::TableNextColumn();
      ImGui::TextUnformatted(run.output_path.empty() ? "-" : run.output_path.c_str());
      ImGui::TableNextColumn();
      if (!run.error.empty()) {
        ImGui::TextColored(ImVec4(1.0f, 0.6f, 0.4f, 1.0f), "%s", run.error.c_str());
      } else {
        ImGui::TextUnformatted("-");
      }
    }
    ImGui::EndTable();
  }
}
