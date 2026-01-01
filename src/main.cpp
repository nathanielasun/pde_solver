#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cctype>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <string>
#include <system_error>
#include <vector>

#include <nlohmann/json.hpp>

#include "advection.h"
#include "advection_tests.h"
#include "time_integrator.h"
#include "time_integrator_tests.h"
#include "pressure_projection.h"
#include "pressure_projection_tests.h"
#include "backend.h"
#include "coefficient_evaluator.h"
#include "conserved_monitor.h"
#include "coupled_examples.h"
#include "coupled_solver.h"
#include "dataset_tools.h"
#include "latex_parser.h"
#include "mms.h"
#include "residual.h"
#include "run_config.h"
#include "run_metadata.h"
#include "run_summary.h"
#include "self_test.h"
#include "solver.h"
#include "solver_tokens.h"
#include "input_parse.h"
#include "vtk_io.h"
#include "shape_io.h"
#include "mesh_io.h"
#include "unstructured_solver.h"
#include "string_utils.h"
#ifdef USE_METAL
#include "backends/metal/MetalSolve.h"
#endif
#include <iomanip>
#include <sstream>

namespace {
void PrintUsage() {
  std::cout
      << "Usage: pde_sim --pde <latex> --domain xmin,xmax,ymin,ymax[,zmin,zmax] "
         "--grid nx,ny[,nz] --out output.vtk\n"
      << "Optional: --bc \"left:dirichlet:0;right:neumann:0;bottom:robin:alpha=1,beta=1,gamma=0;"
         "top:dirichlet:0;front:dirichlet:0;back:dirichlet:0\"\n"
      << "Note: boundary values accept linear expressions in x/y/z (e.g., 1+0.5*x-2*y+z)\n"
      << "Optional: --shape <latex> (domain shape function, uses f(x,y) <= 0)\n"
      << "Optional: --shape-file <file> (load domain shape expression)\n"
      << "Optional: --shape-mask <file> (load VTK mask for implicit domain)\n"
      << "Optional: --shape-mask-threshold T (inside if value <= T, default 0)\n"
      << "Optional: --shape-mask-invert (invert mask inside/outside)\n"
      << "Optional: --shape-offset x,y[,z] --shape-scale x,y[,z] (shape transform)\n"
      << "Optional: --mesh <file> (load unstructured mesh: .vtk or .msh)\n"
      << "Optional: --mesh-format vtk|msh (override mesh format)\n"
      << "Optional: --mesh-discretization fe|fv (unstructured solver stub)\n"
      << "Optional: --mesh-solve (attempt unstructured solve stub)\n"
      << "Optional: --out-dir <dir> (supports naming tokens; see README)\n"
      << "Optional: --in-dir <dir> (load .vtk files from directory)\n"
      << "Optional: --config <file> (load JSON run configuration)\n"
      << "Optional: --export-config <file> (write JSON run configuration)\n"
      << "Optional: --batch <file> (run batch JSON spec)\n"
      << "Optional: --backend auto|cpu|cuda|metal|tpu\n"
      << "Optional: --method jacobi|gs|sor|cg|bicgstab|gmres|mg\n"
      << "Optional: --omega W (SOR relaxation, default 1.5)\n"
      << "Optional: --gmres-restart N (default 30)\n"
      << "Optional: --residual-interval N (0 = off, prints residual samples)\n"
      << "Optional: --self-test (run built-in regression suite)\n"
      << "Optional: --mms (override RHS/BCs with manufactured solution)\n"
      << "Optional: --convergence <list> (grid sweep, CSV + plot output)\n"
      << "Optional: --convergence-out <file> (CSV output path)\n"
      << "Optional: --convergence-plot [file] (gnuplot script path)\n"
      << "Optional: --convergence-json [file] (JSON output path)\n"
      << "Optional: --max-iter N --tol T --threads N (0 = auto)\n"
      << "Optional: --metal-reduce-interval N --metal-threadgroup X,Y (0 = auto)\n"
      << "Optional: --time t_start,t_end,dt,frames (enable time-dependent solve)\n"
      << "Optional: --buffer-mb N (frame buffer size in MB, default 256)\n"
      << "Optional: --checkpoint <file> (write checkpoint after each frame)\n"
      << "Optional: --restart <file> (resume from checkpoint)\n"
      << "Optional: --format vtk|vti (output format, default: vtk)\n"
      << "Optional: --validate (parse inputs and exit without solving)\n"
      << "Optional: --dump-operator[=json] (print parsed operator)\n"
      << "Optional: --dump-metadata[=<file>] (print or export run metadata)\n"
      << "Optional: --dump-summary[=<file>] (print or export run summary)\n"
      << "Optional: --dataset-index <dir> (write dataset index + stats)\n"
      << "Optional: --dataset-index-out <file> (override dataset index output path)\n"
      << "Optional: --dataset-cleanup <dir> (remove orphaned sidecars)\n"
      << "Optional: --dataset-cleanup-dry-run (preview cleanup)\n"
      << "Optional: --dataset-cleanup-empty-dirs (remove empty directories)\n"
      << "Optional: --list-examples (list available coupled PDE examples)\n"
      << "Optional: --run-example <name> (run a coupled PDE example)\n"
      << "Optional: --advection-test [scheme] (run advection discretization tests)\n"
      << "Optional: --advection-scheme <name> (upwind|minmod|superbee|vanleer|mc)\n"
      << "Optional: --time-integrator-test [method] (run time integrator tests)\n"
      << "Optional: --time-integrator <name> (euler|rk2|rk4|ssprk2|ssprk3|beuler|cn)\n"
      << "Optional: --projection-test (run pressure projection tests)\n"
      << "Optional: --lid-cavity [Re] (run lid-driven cavity benchmark, default Re=100)\n";
}

using json = nlohmann::json;

struct OutputPatternContext {
  int index = 1;
  int index0 = 0;
  std::string name;
  std::string backend;
  std::string method;
  std::string format;
  Domain domain;
  std::string tag;
  std::string timestamp;
};

std::string SanitizeTokenValue(const std::string& value) {
  std::string out;
  out.reserve(value.size());
  for (unsigned char ch : value) {
    if (std::isalnum(ch) || ch == '-' || ch == '_' || ch == '.') {
      out.push_back(static_cast<char>(ch));
    } else {
      out.push_back('_');
    }
  }
  return out;
}

std::string FormatTimestamp() {
  std::time_t now = std::time(nullptr);
  std::tm tm = {};
  if (auto* tm_ptr = std::localtime(&now)) {
    tm = *tm_ptr;
  }
  std::ostringstream out;
  out << std::put_time(&tm, "%Y%m%d_%H%M%S");
  return out.str();
}

std::string BuildGridTag(const Domain& domain) {
  std::ostringstream out;
  out << domain.nx << "x" << domain.ny;
  if (domain.nz > 1) {
    out << "x" << domain.nz;
  }
  return out.str();
}

std::string ReplaceAll(std::string value, const std::string& from, const std::string& to) {
  if (from.empty()) {
    return value;
  }
  size_t start = 0;
  while ((start = value.find(from, start)) != std::string::npos) {
    value.replace(start, from.size(), to);
    start += to.size();
  }
  return value;
}

bool ContainsPatternToken(const std::string& text) {
  return text.find('{') != std::string::npos;
}

std::string ApplyOutputPattern(const std::string& pattern, const OutputPatternContext& ctx) {
  if (pattern.empty()) {
    return pattern;
  }
  const std::string safe_name = SanitizeTokenValue(ctx.name);
  const std::string safe_backend = SanitizeTokenValue(ctx.backend);
  const std::string safe_method = SanitizeTokenValue(ctx.method);
  const std::string safe_format = SanitizeTokenValue(ctx.format);
  const std::string safe_grid = SanitizeTokenValue(BuildGridTag(ctx.domain));

  std::string out = pattern;
  out = ReplaceAll(out, "{index}", std::to_string(ctx.index));
  out = ReplaceAll(out, "{index0}", std::to_string(ctx.index0));
  out = ReplaceAll(out, "{name}", safe_name);
  out = ReplaceAll(out, "{grid}", safe_grid);
  out = ReplaceAll(out, "{nx}", std::to_string(ctx.domain.nx));
  out = ReplaceAll(out, "{ny}", std::to_string(ctx.domain.ny));
  out = ReplaceAll(out, "{nz}", std::to_string(ctx.domain.nz));
  out = ReplaceAll(out, "{backend}", safe_backend);
  out = ReplaceAll(out, "{method}", safe_method);
  out = ReplaceAll(out, "{format}", safe_format);
  out = ReplaceAll(out, "{tag}", ctx.tag);
  out = ReplaceAll(out, "{timestamp}", ctx.timestamp);
  return out;
}

std::filesystem::path EnsureUniquePath(const std::filesystem::path& path) {
  std::error_code ec;
  if (!std::filesystem::exists(path, ec)) {
    return path;
  }
  const std::filesystem::path parent = path.parent_path();
  const std::string stem = path.stem().string();
  const std::string ext = path.extension().string();
  for (int suffix = 1; suffix < 10000; ++suffix) {
    std::filesystem::path candidate = parent / (stem + "_" + std::to_string(suffix) + ext);
    if (!std::filesystem::exists(candidate, ec)) {
      return candidate;
    }
  }
  return path;
}

std::filesystem::path BuildSeriesFramePath(const std::filesystem::path& base_path, int frame,
                                           int digits, const std::string& output_format) {
  std::ostringstream out;
  out << base_path.string() << "_" << std::setfill('0') << std::setw(digits) << frame;
  if (output_format == "vti") {
    out << ".vti";
  } else {
    out << ".vtk";
  }
  return std::filesystem::path(out.str());
}

std::filesystem::path EnsureUniqueSeriesBase(const std::filesystem::path& base_path, int digits,
                                             const std::string& output_format) {
  std::error_code ec;
  const std::filesystem::path probe = BuildSeriesFramePath(base_path, 0, digits, output_format);
  if (!std::filesystem::exists(base_path, ec) && !std::filesystem::exists(probe, ec)) {
    return base_path;
  }
  const std::filesystem::path parent = base_path.parent_path();
  const std::string stem = base_path.stem().string();
  for (int suffix = 1; suffix < 10000; ++suffix) {
    std::filesystem::path candidate = parent / (stem + "_" + std::to_string(suffix));
    const std::filesystem::path candidate_probe =
        BuildSeriesFramePath(candidate, 0, digits, output_format);
    if (!std::filesystem::exists(candidate, ec) &&
        !std::filesystem::exists(candidate_probe, ec)) {
      return candidate;
    }
  }
  return base_path;
}

void BuildDerivedFieldVectors(const Domain& domain,
                              const std::vector<double>& grid,
                              double coef_a,
                              double coef_b,
                              double coef_az,
                              std::vector<std::vector<double>>* derived_field_data,
                              std::vector<std::string>* derived_field_names) {
  if (!derived_field_data || !derived_field_names) {
    return;
  }
  derived_field_data->clear();
  derived_field_names->clear();
  DerivedFields derived = ComputeDerivedFields(domain, grid, coef_a, coef_b, coef_az);
  derived_field_data->push_back(derived.gradient_x);
  derived_field_names->push_back("gradient_x");
  derived_field_data->push_back(derived.gradient_y);
  derived_field_names->push_back("gradient_y");
  if (domain.nz > 1 && !derived.gradient_z.empty()) {
    derived_field_data->push_back(derived.gradient_z);
    derived_field_names->push_back("gradient_z");
  }
  derived_field_data->push_back(derived.laplacian);
  derived_field_names->push_back("laplacian");
  derived_field_data->push_back(derived.flux_x);
  derived_field_names->push_back("flux_x");
  derived_field_data->push_back(derived.flux_y);
  derived_field_names->push_back("flux_y");
  if (domain.nz > 1 && !derived.flux_z.empty()) {
    derived_field_data->push_back(derived.flux_z);
    derived_field_names->push_back("flux_z");
  }
  derived_field_data->push_back(derived.energy_norm);
  derived_field_names->push_back("energy_norm");
}

struct ErrorNorms {
  double l1 = 0.0;
  double l2 = 0.0;
  double linf = 0.0;
};

ErrorNorms ComputeErrorNorms2D(const Domain& d,
                               const std::vector<double>& grid,
                               const std::function<double(double, double)>& exact) {
  ErrorNorms out;
  struct KahanSum {
    double sum = 0.0;
    double c = 0.0;
    void Add(double value) {
      const double y = value - c;
      const double t = sum + y;
      c = (t - sum) - y;
      sum = t;
    }
  };
  const int nx = d.nx;
  const int ny = d.ny;
  const double dx = (d.xmax - d.xmin) / static_cast<double>(std::max(1, nx - 1));
  const double dy = (d.ymax - d.ymin) / static_cast<double>(std::max(1, ny - 1));
  KahanSum sum_abs;
  KahanSum sum_sq;
  double max_abs = 0.0;
  for (int j = 0; j < ny; ++j) {
    const double y = d.ymin + j * dy;
    for (int i = 0; i < nx; ++i) {
      const double x = d.xmin + i * dx;
      const size_t idx = static_cast<size_t>(j * nx + i);
      if (idx >= grid.size()) {
        continue;
      }
      const double err = grid[idx] - exact(x, y);
      const double abs_err = std::abs(err);
      sum_abs.Add(abs_err);
      sum_sq.Add(err * err);
      max_abs = std::max(max_abs, abs_err);
    }
  }
  const double weight = dx * dy;
  out.l1 = sum_abs.sum * weight;
  out.l2 = std::sqrt(std::max(0.0, sum_sq.sum * weight));
  out.linf = max_abs;
  return out;
}

ErrorNorms ComputeErrorNorms3D(const Domain& d,
                               const std::vector<double>& grid,
                               const std::function<double(double, double, double)>& exact) {
  ErrorNorms out;
  struct KahanSum {
    double sum = 0.0;
    double c = 0.0;
    void Add(double value) {
      const double y = value - c;
      const double t = sum + y;
      c = (t - sum) - y;
      sum = t;
    }
  };
  const int nx = d.nx;
  const int ny = d.ny;
  const int nz = d.nz;
  const double dx = (d.xmax - d.xmin) / static_cast<double>(std::max(1, nx - 1));
  const double dy = (d.ymax - d.ymin) / static_cast<double>(std::max(1, ny - 1));
  const double dz = (d.zmax - d.zmin) / static_cast<double>(std::max(1, nz - 1));
  KahanSum sum_abs;
  KahanSum sum_sq;
  double max_abs = 0.0;
  for (int k = 0; k < nz; ++k) {
    const double z = d.zmin + k * dz;
    for (int j = 0; j < ny; ++j) {
      const double y = d.ymin + j * dy;
      for (int i = 0; i < nx; ++i) {
        const double x = d.xmin + i * dx;
        const size_t idx = static_cast<size_t>((k * ny + j) * nx + i);
        if (idx >= grid.size()) {
          continue;
        }
        const double err = grid[idx] - exact(x, y, z);
        const double abs_err = std::abs(err);
        sum_abs.Add(abs_err);
        sum_sq.Add(err * err);
        max_abs = std::max(max_abs, abs_err);
      }
    }
  }
  const double weight = dx * dy * dz;
  out.l1 = sum_abs.sum * weight;
  out.l2 = std::sqrt(std::max(0.0, sum_sq.sum * weight));
  out.linf = max_abs;
  return out;
}

BoundaryCondition MakeDirichletFromLatex(const std::string& latex) {
  BoundaryCondition bc;
  bc.kind = BCKind::Dirichlet;
  bc.value = {};
  bc.value.latex = latex;
  return bc;
}

void ApplyMmsDirichlet(BoundarySet* bc, const std::string& latex) {
  if (!bc) {
    return;
  }
  const BoundaryCondition dirichlet = MakeDirichletFromLatex(latex);
  bc->left = dirichlet;
  bc->right = dirichlet;
  bc->bottom = dirichlet;
  bc->top = dirichlet;
  bc->front = dirichlet;
  bc->back = dirichlet;
  bc->left_piecewise.clear();
  bc->right_piecewise.clear();
  bc->bottom_piecewise.clear();
  bc->top_piecewise.clear();
  bc->front_piecewise.clear();
  bc->back_piecewise.clear();
}

double ComputeGridSpacing(const Domain& d) {
  const double dx = (d.xmax - d.xmin) / static_cast<double>(std::max(1, d.nx - 1));
  const double dy = (d.ymax - d.ymin) / static_cast<double>(std::max(1, d.ny - 1));
  double h = std::max(dx, dy);
  if (d.nz > 1) {
    const double dz = (d.zmax - d.zmin) / static_cast<double>(std::max(1, d.nz - 1));
    h = std::max(h, dz);
  }
  return h;
}

struct ConvergenceRow {
  Domain domain;
  double h = 0.0;
  ErrorNorms norms;
  double order_l1 = std::numeric_limits<double>::quiet_NaN();
  double order_l2 = std::numeric_limits<double>::quiet_NaN();
  double order_linf = std::numeric_limits<double>::quiet_NaN();
};

bool ParseConvergenceSpec(const std::string& spec,
                          const Domain& base,
                          std::vector<Domain>* grids,
                          std::string* error) {
  if (!grids) {
    return false;
  }
  grids->clear();
  if (spec.empty()) {
    if (error) {
      *error = "empty convergence spec";
    }
    return false;
  }
  const bool is_3d = base.nz > 1;
  const bool has_explicit = spec.find(';') != std::string::npos;
  if (has_explicit) {
    std::stringstream ss(spec);
    std::string item;
    while (std::getline(ss, item, ';')) {
      if (item.empty()) {
        continue;
      }
      Domain d = base;
      ParseResult res = ParseGrid(item, &d);
      if (!res.ok) {
        if (error) {
          *error = "invalid grid in convergence spec: " + item + " (" + res.error + ")";
        }
        return false;
      }
      if (is_3d && d.nz <= 1) {
        if (error) {
          *error = "3D convergence requires grid specs with nz: " + item;
        }
        return false;
      }
      if (!is_3d && d.nz > 1) {
        if (error) {
          *error = "2D convergence does not accept 3D grid specs: " + item;
        }
        return false;
      }
      if (d.nx < 3 || d.ny < 3 || (is_3d && d.nz < 3)) {
        if (error) {
          *error = "grid sizes must be >= 3: " + item;
        }
        return false;
      }
      grids->push_back(d);
    }
    if (grids->empty()) {
      if (error) {
        *error = "no grids parsed from convergence spec";
      }
      return false;
    }
    return true;
  }

  std::stringstream ss(spec);
  std::string token;
  while (std::getline(ss, token, ',')) {
    if (token.empty()) {
      if (error) {
        *error = "invalid convergence grid list";
      }
      return false;
    }
    char* end = nullptr;
    long value = std::strtol(token.c_str(), &end, 10);
    if (end == token.c_str() || value <= 0) {
      if (error) {
        *error = "invalid convergence grid size: " + token;
      }
      return false;
    }
    const int n = static_cast<int>(value);
    if (n < 3) {
      if (error) {
        *error = "grid sizes must be >= 3: " + token;
      }
      return false;
    }
    Domain d = base;
    d.nx = n;
    d.ny = n;
    d.nz = is_3d ? n : 1;
    grids->push_back(d);
  }
  if (grids->empty()) {
    if (error) {
      *error = "no grids parsed from convergence spec";
    }
    return false;
  }
  return true;
}

void PrintBackendStatus() {
  const auto statuses = DetectBackends();
  std::cout << "Available backends:\n";
  for (const auto& status : statuses) {
    std::cout << "  - " << status.name << ": "
              << (status.available ? "yes" : "no");
    if (!status.note.empty()) {
      std::cout << " (" << status.note << ")";
    }
    std::cout << "\n";
  }
}

void PrintParseSummary(const LatexParseResult& parse_result) {
  const PDECoefficients& c = parse_result.coeffs;
  std::cout << "PDE parse: ok\n";
  std::cout << "  constants:\n";
  if (c.a != 0.0) std::cout << "    u_xx: " << c.a << "\n";
  if (c.b != 0.0) std::cout << "    u_yy: " << c.b << "\n";
  if (c.az != 0.0) std::cout << "    u_zz: " << c.az << "\n";
  if (c.c != 0.0) std::cout << "    u_x: " << c.c << "\n";
  if (c.d != 0.0) std::cout << "    u_y: " << c.d << "\n";
  if (c.dz != 0.0) std::cout << "    u_z: " << c.dz << "\n";
  if (c.e != 0.0) std::cout << "    u: " << c.e << "\n";
  if (c.ut != 0.0) std::cout << "    u_t: " << c.ut << "\n";
  if (c.utt != 0.0) std::cout << "    u_tt: " << c.utt << "\n";
  if (c.ab != 0.0) std::cout << "    u_xy: " << c.ab << "\n";
  if (c.ac != 0.0) std::cout << "    u_xz: " << c.ac << "\n";
  if (c.bc != 0.0) std::cout << "    u_yz: " << c.bc << "\n";
  if (c.a3 != 0.0) std::cout << "    u_xxx: " << c.a3 << "\n";
  if (c.b3 != 0.0) std::cout << "    u_yyy: " << c.b3 << "\n";
  if (c.az3 != 0.0) std::cout << "    u_zzz: " << c.az3 << "\n";
  if (c.a4 != 0.0) std::cout << "    u_xxxx: " << c.a4 << "\n";
  if (c.b4 != 0.0) std::cout << "    u_yyyy: " << c.b4 << "\n";
  if (c.az4 != 0.0) std::cout << "    u_zzzz: " << c.az4 << "\n";
  if (c.f != 0.0) std::cout << "    rhs_const: " << c.f << "\n";

  if (!c.rhs_latex.empty()) {
    std::cout << "  rhs_latex: " << c.rhs_latex << "\n";
  }

  std::cout << "  variable coefficients:\n";
  if (!c.a_latex.empty()) std::cout << "    u_xx: " << c.a_latex << "\n";
  if (!c.b_latex.empty()) std::cout << "    u_yy: " << c.b_latex << "\n";
  if (!c.az_latex.empty()) std::cout << "    u_zz: " << c.az_latex << "\n";
  if (!c.c_latex.empty()) std::cout << "    u_x: " << c.c_latex << "\n";
  if (!c.d_latex.empty()) std::cout << "    u_y: " << c.d_latex << "\n";
  if (!c.dz_latex.empty()) std::cout << "    u_z: " << c.dz_latex << "\n";
  if (!c.e_latex.empty()) std::cout << "    u: " << c.e_latex << "\n";
  if (!c.ab_latex.empty()) std::cout << "    u_xy: " << c.ab_latex << "\n";
  if (!c.ac_latex.empty()) std::cout << "    u_xz: " << c.ac_latex << "\n";
  if (!c.bc_latex.empty()) std::cout << "    u_yz: " << c.bc_latex << "\n";
  if (!c.a3_latex.empty()) std::cout << "    u_xxx: " << c.a3_latex << "\n";
  if (!c.b3_latex.empty()) std::cout << "    u_yyy: " << c.b3_latex << "\n";
  if (!c.az3_latex.empty()) std::cout << "    u_zzz: " << c.az3_latex << "\n";
  if (!c.a4_latex.empty()) std::cout << "    u_xxxx: " << c.a4_latex << "\n";
  if (!c.b4_latex.empty()) std::cout << "    u_yyyy: " << c.b4_latex << "\n";
  if (!c.az4_latex.empty()) std::cout << "    u_zzzz: " << c.az4_latex << "\n";

  if (!parse_result.integrals.empty()) {
    std::cout << "  integrals: " << parse_result.integrals.size() << "\n";
  }
  if (!parse_result.nonlinear.empty()) {
    std::cout << "  nonlinear_terms: " << parse_result.nonlinear.size() << "\n";
  }
  if (!parse_result.nonlinear_derivatives.empty()) {
    std::cout << "  nonlinear_derivative_terms: " << parse_result.nonlinear_derivatives.size() << "\n";
  }
}

std::string TermLabel(const PDETerm& term) {
  std::string suffix;
  if (term.dt > 0) {
    suffix.append(static_cast<size_t>(term.dt), 't');
  }
  suffix.append(static_cast<size_t>(term.dx), 'x');
  suffix.append(static_cast<size_t>(term.dy), 'y');
  suffix.append(static_cast<size_t>(term.dz), 'z');
  if (suffix.empty()) {
    return "u";
  }
  return "u_" + suffix;
}

std::string JsonEscape(const std::string& text) {
  std::string out;
  out.reserve(text.size() + 8);
  for (char ch : text) {
    switch (ch) {
      case '\\': out += "\\\\"; break;
      case '"': out += "\\\""; break;
      case '\n': out += "\\n"; break;
      case '\r': out += "\\r"; break;
      case '\t': out += "\\t"; break;
      default: out += ch; break;
    }
  }
  return out;
}

const char* NonlinearKindLabel(NonlinearKind kind) {
  switch (kind) {
    case NonlinearKind::Power: return "power";
    case NonlinearKind::Sin: return "sin";
    case NonlinearKind::Cos: return "cos";
    case NonlinearKind::Exp: return "exp";
    case NonlinearKind::Abs: return "abs";
  }
  return "unknown";
}

const char* NonlinearDerivativeKindLabel(NonlinearDerivativeKind kind) {
  switch (kind) {
    case NonlinearDerivativeKind::UUx: return "u_ux";
    case NonlinearDerivativeKind::UUy: return "u_uy";
    case NonlinearDerivativeKind::UUz: return "u_uz";
    case NonlinearDerivativeKind::UxUx: return "ux_ux";
    case NonlinearDerivativeKind::UyUy: return "uy_uy";
    case NonlinearDerivativeKind::UzUz: return "uz_uz";
    case NonlinearDerivativeKind::GradSquared: return "grad_squared";
  }
  return "unknown";
}

void PrintOperatorDumpText(const PDEOperator& op) {
  std::cout << "Operator dump:\n";
  if (op.lhs_terms.empty()) {
    std::cout << "  LHS terms: none\n";
  } else {
    std::cout << "  LHS terms:\n";
    for (const auto& term : op.lhs_terms) {
      std::cout << "    " << TermLabel(term) << " ";
      if (!term.coeff_latex.empty()) {
        std::cout << "coeff_latex=" << term.coeff_latex;
        if (term.coeff != 1.0) {
          std::cout << " scale=" << term.coeff;
        }
      } else {
        std::cout << "coeff=" << term.coeff;
      }
      std::cout << "\n";
    }
  }

  if (!op.rhs_latex.empty()) {
    std::cout << "  RHS latex: " << op.rhs_latex << "\n";
  } else {
    std::cout << "  RHS constant: " << op.rhs_constant << "\n";
  }

  if (!op.integrals.empty()) {
    std::cout << "  integrals:\n";
    for (const auto& term : op.integrals) {
      std::cout << "    coeff=" << term.coeff;
      if (!term.kernel_latex.empty()) {
        std::cout << " kernel=" << term.kernel_latex;
      }
      std::cout << "\n";
    }
  }

  if (!op.nonlinear.empty()) {
    std::cout << "  nonlinear_terms: " << op.nonlinear.size() << "\n";
  }
  if (!op.nonlinear_derivatives.empty()) {
    std::cout << "  nonlinear_derivative_terms: " << op.nonlinear_derivatives.size() << "\n";
  }
}

void PrintOperatorDumpJson(const PDEOperator& op) {
  std::cout << "{";
  std::cout << "\"schema_version\":1,";
  std::cout << "\"lhs_terms\":[";
  for (size_t i = 0; i < op.lhs_terms.size(); ++i) {
    const auto& term = op.lhs_terms[i];
    if (i > 0) {
      std::cout << ",";
    }
    std::cout << "{"
              << "\"dx\":" << term.dx
              << ",\"dy\":" << term.dy
              << ",\"dz\":" << term.dz
              << ",\"dt\":" << term.dt
              << ",\"coeff\":" << term.coeff
              << ",\"coeff_latex\":\"" << JsonEscape(term.coeff_latex) << "\""
              << ",\"label\":\"" << JsonEscape(TermLabel(term)) << "\""
              << "}";
  }
  std::cout << "],";
  std::cout << "\"rhs\":{"
            << "\"constant\":" << op.rhs_constant
            << ",\"latex\":\"" << JsonEscape(op.rhs_latex) << "\""
            << "},";
  std::cout << "\"integrals\":[";
  for (size_t i = 0; i < op.integrals.size(); ++i) {
    const auto& term = op.integrals[i];
    if (i > 0) {
      std::cout << ",";
    }
    std::cout << "{"
              << "\"coeff\":" << term.coeff
              << ",\"kernel_latex\":\"" << JsonEscape(term.kernel_latex) << "\""
              << "}";
  }
  std::cout << "],";
  std::cout << "\"nonlinear_terms\":[";
  for (size_t i = 0; i < op.nonlinear.size(); ++i) {
    const auto& term = op.nonlinear[i];
    if (i > 0) {
      std::cout << ",";
    }
    std::cout << "{"
              << "\"kind\":\"" << NonlinearKindLabel(term.kind) << "\""
              << ",\"coeff\":" << term.coeff;
    if (term.kind == NonlinearKind::Power) {
      std::cout << ",\"power\":" << term.power;
    }
    std::cout << "}";
  }
  std::cout << "],";
  std::cout << "\"nonlinear_derivative_terms\":[";
  for (size_t i = 0; i < op.nonlinear_derivatives.size(); ++i) {
    const auto& term = op.nonlinear_derivatives[i];
    if (i > 0) {
      std::cout << ",";
    }
    std::cout << "{"
              << "\"kind\":\"" << NonlinearDerivativeKindLabel(term.kind) << "\""
              << ",\"coeff\":" << term.coeff
              << "}";
  }
  std::cout << "]";
  std::cout << "}\n";
}

std::vector<std::filesystem::path> CollectVtkFiles(const std::filesystem::path& dir) {
  std::vector<std::filesystem::path> files;
  try {
    for (const auto& entry : std::filesystem::directory_iterator(dir)) {
      if (!entry.is_regular_file()) {
        continue;
      }
      if (entry.path().extension() == ".vtk") {
        files.push_back(entry.path());
      }
    }
  } catch (const std::exception& exc) {
    std::cerr << "failed to read directory: " << exc.what() << "\n";
  }
  std::sort(files.begin(), files.end());
  return files;
}

std::vector<std::filesystem::path> CollectMetadataFiles(const std::filesystem::path& dir) {
  std::vector<std::filesystem::path> files;
  try {
    for (const auto& entry : std::filesystem::directory_iterator(dir)) {
      if (!entry.is_regular_file()) {
        continue;
      }
      const std::string ext = entry.path().extension().string();
      if (ext == ".vtk" || ext == ".vti" || ext == ".pvd") {
        files.push_back(entry.path());
      }
    }
  } catch (const std::exception& exc) {
    std::cerr << "failed to read directory: " << exc.what() << "\n";
  }
  std::sort(files.begin(), files.end());
  return files;
}

bool DumpMetadataFile(const std::filesystem::path& path) {
  std::string metadata_json;
  std::string error;
  if (!ReadRunMetadataSidecar(path, &metadata_json, &error)) {
    std::cerr << "metadata read error (" << path.filename() << "): " << error << "\n";
    return false;
  }
  std::cout << "metadata: " << path.filename() << "\n";
  std::cout << metadata_json << "\n";
  return true;
}

bool DumpMetadataDirectory(const std::filesystem::path& dir) {
  if (!std::filesystem::exists(dir) || !std::filesystem::is_directory(dir)) {
    std::cerr << "input directory not found: " << dir << "\n";
    return false;
  }
  std::vector<std::filesystem::path> files = CollectMetadataFiles(dir);
  if (files.empty()) {
    std::cerr << "no output files found in directory: " << dir << "\n";
    return false;
  }
  bool dumped_any = false;
  for (const auto& path : files) {
    dumped_any = DumpMetadataFile(path) || dumped_any;
  }
  if (!dumped_any) {
    std::cerr << "no metadata sidecars found in directory: " << dir << "\n";
  }
  return dumped_any;
}

bool DumpSummaryFile(const std::filesystem::path& path) {
  std::string summary_json;
  std::string error;
  if (!ReadRunSummarySidecar(path, &summary_json, &error)) {
    std::cerr << "summary read error (" << path.filename() << "): " << error << "\n";
    return false;
  }
  std::cout << "summary: " << path.filename() << "\n";
  std::cout << summary_json << "\n";
  return true;
}

bool DumpSummaryDirectory(const std::filesystem::path& dir) {
  if (!std::filesystem::exists(dir) || !std::filesystem::is_directory(dir)) {
    std::cerr << "input directory not found: " << dir << "\n";
    return false;
  }
  std::vector<std::filesystem::path> files = CollectMetadataFiles(dir);
  if (files.empty()) {
    std::cerr << "no output files found in directory: " << dir << "\n";
    return false;
  }
  bool dumped_any = false;
  for (const auto& path : files) {
    dumped_any = DumpSummaryFile(path) || dumped_any;
  }
  if (!dumped_any) {
    std::cerr << "no summary sidecars found in directory: " << dir << "\n";
  }
  return dumped_any;
}

bool LoadVtkDirectory(const std::filesystem::path& dir) {
  if (!std::filesystem::exists(dir) || !std::filesystem::is_directory(dir)) {
    std::cerr << "input directory not found: " << dir << "\n";
    return false;
  }
  std::vector<std::filesystem::path> files = CollectVtkFiles(dir);
  if (files.empty()) {
    std::cerr << "no .vtk files found in directory: " << dir << "\n";
    return false;
  }

  bool loaded_any = false;
  for (const auto& path : files) {
    VtkReadResult read_result = ReadVtkFile(path.string());
    if (!read_result.ok) {
      std::cerr << "vtk read error (" << path.filename() << "): " << read_result.error << "\n";
      continue;
    }
    double min_val = 0.0;
    double max_val = 0.0;
    bool has_values = false;
    if (read_result.kind == VtkReadResult::Kind::StructuredPoints) {
      if (!read_result.grid.empty()) {
        min_val = read_result.grid.front();
        max_val = read_result.grid.front();
        for (double value : read_result.grid) {
          min_val = std::min(min_val, value);
          max_val = std::max(max_val, value);
        }
        has_values = true;
      }
      std::cout << "loaded " << path.filename() << " (" << read_result.domain.nx << "x"
                << read_result.domain.ny << "x" << read_result.domain.nz << ")";
    } else {
      if (!read_result.points.empty()) {
        min_val = read_result.points.front().value;
        max_val = read_result.points.front().value;
        for (const auto& pt : read_result.points) {
          min_val = std::min(min_val, pt.value);
          max_val = std::max(max_val, pt.value);
        }
        has_values = true;
      }
      std::cout << "loaded " << path.filename() << " (points: " << read_result.points.size()
                << ")";
    }
    if (has_values) {
      std::cout << " range [" << min_val << ", " << max_val << "]";
    }
    std::cout << "\n";
    loaded_any = true;
  }
  if (!loaded_any) {
    std::cerr << "no readable .vtk files in directory: " << dir << "\n";
  }
  return loaded_any;
}

bool WriteTextFile(const std::filesystem::path& path, const std::string& text, std::string* error) {
  if (path.has_parent_path()) {
    std::error_code ec;
    std::filesystem::create_directories(path.parent_path(), ec);
    if (ec) {
      if (error) {
        *error = "failed to create output directory: " + ec.message();
      }
      return false;
    }
  }
  std::ofstream out(path);
  if (!out.is_open()) {
    if (error) {
      *error = "failed to write file: " + path.string();
    }
    return false;
  }
  out << text;
  if (!out.good()) {
    if (error) {
      *error = "failed to write file: " + path.string();
    }
    return false;
  }
  return true;
}

bool WriteConvergencePlotScript(const std::filesystem::path& plot_path,
                                const std::filesystem::path& csv_path,
                                std::string* error) {
  std::filesystem::path image_path = plot_path;
  image_path.replace_extension(".png");
  const std::string csv_abs = std::filesystem::absolute(csv_path).string();
  const std::string img_abs = std::filesystem::absolute(image_path).string();

  std::ostringstream script;
  script << "set datafile separator ','\n";
  script << "set logscale xy\n";
  script << "set grid\n";
  script << "set key left top\n";
  script << "set xlabel 'h'\n";
  script << "set ylabel 'Error norm'\n";
  script << "set term pngcairo size 900,700\n";
  script << "set output '" << img_abs << "'\n";
  script << "plot '" << csv_abs << "' using 4:5 with linespoints title 'L1', \\\n";
  script << "     '" << csv_abs << "' using 4:6 with linespoints title 'L2', \\\n";
  script << "     '" << csv_abs << "' using 4:7 with linespoints title 'Linf'\n";

  return WriteTextFile(plot_path, script.str(), error);
}

bool WriteConvergenceJson(const std::filesystem::path& json_path,
                          const std::vector<ConvergenceRow>& rows,
                          int dimension,
                          BackendKind requested_backend,
                          BackendKind selected_backend,
                          const std::string& backend_note,
                          const ManufacturedSolution& solution,
                          std::string* error) {
  using json = nlohmann::json;
  json root;
  root["schema_version"] = 1;
  root["dimension"] = dimension;
  json backend;
  backend["requested"] = BackendToken(requested_backend);
  backend["selected"] = BackendToken(selected_backend);
  if (!backend_note.empty()) {
    backend["note"] = backend_note;
  }
  root["backend"] = backend;
  root["solution"] = {{"u_latex", solution.u_latex}};

  auto order_value = [](double v) -> json {
    if (!std::isfinite(v)) {
      return nullptr;
    }
    return v;
  };

  json rows_json = json::array();
  for (const auto& row : rows) {
    json entry;
    entry["nx"] = row.domain.nx;
    entry["ny"] = row.domain.ny;
    entry["nz"] = row.domain.nz;
    entry["h"] = row.h;
    entry["l1"] = row.norms.l1;
    entry["l2"] = row.norms.l2;
    entry["linf"] = row.norms.linf;
    entry["order_l1"] = order_value(row.order_l1);
    entry["order_l2"] = order_value(row.order_l2);
    entry["order_linf"] = order_value(row.order_linf);
    rows_json.push_back(entry);
  }
  root["rows"] = rows_json;

  std::string payload;
  try {
    payload = root.dump(2);
  } catch (const std::exception& exc) {
    if (error) {
      *error = std::string("failed to serialize convergence JSON: ") + exc.what();
    }
    return false;
  }
  return WriteTextFile(json_path, payload, error);
}

bool RunConvergenceStudy(const SolveInput& base_input,
                         BackendKind backend,
                         const std::vector<Domain>& grids,
                         const ManufacturedSolution& solution,
                         const std::filesystem::path& csv_path,
                         const std::filesystem::path& plot_path,
                         const std::filesystem::path& json_path,
                         std::string* error) {
  if (grids.empty()) {
    if (error) {
      *error = "no convergence grids provided";
    }
    return false;
  }

  std::vector<ConvergenceRow> rows;
  rows.reserve(grids.size());

  BackendKind first_selected = BackendKind::CPU;
  std::string first_note;
  bool selected_set = false;

  for (const auto& grid : grids) {
    SolveInput input = base_input;
    input.domain = grid;
    BackendKind selected = BackendKind::CPU;
    std::string note;
    SolveOutput out = SolveWithBackend(input, backend, &selected, &note);
    if (!out.error.empty()) {
      if (error) {
        *error = out.error;
      }
      return false;
    }

    ErrorNorms norms;
    if (solution.dimension > 2) {
      norms = ComputeErrorNorms3D(grid, out.grid,
                                  [&](double x, double y, double z) {
                                    return solution.eval(x, y, z);
                                  });
    } else {
      norms = ComputeErrorNorms2D(grid, out.grid,
                                  [&](double x, double y) {
                                    return solution.eval(x, y, 0.0);
                                  });
    }

    ConvergenceRow row;
    row.domain = grid;
    row.h = ComputeGridSpacing(grid);
    row.norms = norms;
    rows.push_back(row);

    if (!selected_set) {
      first_selected = selected;
      first_note = note;
      selected_set = true;
    }
  }

  for (size_t i = 1; i < rows.size(); ++i) {
    const double h_ratio = rows[i - 1].h / std::max(1e-12, rows[i].h);
    if (rows[i].norms.l1 > 0.0 && rows[i - 1].norms.l1 > 0.0 && h_ratio > 0.0) {
      rows[i].order_l1 = std::log(rows[i - 1].norms.l1 / rows[i].norms.l1) / std::log(h_ratio);
    }
    if (rows[i].norms.l2 > 0.0 && rows[i - 1].norms.l2 > 0.0 && h_ratio > 0.0) {
      rows[i].order_l2 = std::log(rows[i - 1].norms.l2 / rows[i].norms.l2) / std::log(h_ratio);
    }
    if (rows[i].norms.linf > 0.0 && rows[i - 1].norms.linf > 0.0 && h_ratio > 0.0) {
      rows[i].order_linf = std::log(rows[i - 1].norms.linf / rows[i].norms.linf) / std::log(h_ratio);
    }
  }

  std::cout << "Convergence study (" << rows.size() << " grids), backend="
            << BackendKindName(first_selected);
  if (!first_note.empty()) {
    std::cout << " (" << first_note << ")";
  }
  std::cout << "\n";
  std::cout << "grid,h,L1,L2,Linf,order(L1),order(L2),order(Linf)\n";
  for (const auto& row : rows) {
    std::cout << row.domain.nx << "x" << row.domain.ny;
    if (row.domain.nz > 1) {
      std::cout << "x" << row.domain.nz;
    }
    std::cout << "," << row.h
              << "," << row.norms.l1
              << "," << row.norms.l2
              << "," << row.norms.linf
              << "," << row.order_l1
              << "," << row.order_l2
              << "," << row.order_linf
              << "\n";
  }

  std::ostringstream csv;
  csv << "nx,ny,nz,h,l1,l2,linf,order_l1,order_l2,order_linf\n";
  csv.setf(std::ios::scientific);
  csv << std::setprecision(12);
  for (const auto& row : rows) {
    csv << row.domain.nx << "," << row.domain.ny << "," << row.domain.nz << ","
        << row.h << "," << row.norms.l1 << "," << row.norms.l2 << "," << row.norms.linf << ","
        << row.order_l1 << "," << row.order_l2 << "," << row.order_linf << "\n";
  }

  std::string write_error;
  if (!WriteTextFile(csv_path, csv.str(), &write_error)) {
    if (error) {
      *error = write_error;
    }
    return false;
  }

  if (!json_path.empty()) {
    std::string json_error;
    const int dimension = (solution.dimension > 2) ? 3 : 2;
    if (!WriteConvergenceJson(json_path, rows, dimension,
                              backend, first_selected, first_note,
                              solution, &json_error)) {
      if (error) {
        *error = json_error;
      }
      return false;
    }
  }

  if (!plot_path.empty()) {
    std::string plot_error;
    if (!WriteConvergencePlotScript(plot_path, csv_path, &plot_error)) {
      if (error) {
        *error = plot_error;
      }
      return false;
    }
  }

  return true;
}

struct BatchRunEntry {
  int index = 0;
  std::string name;
  std::filesystem::path source_path;
  RunConfig config;
};

struct BatchSpec {
  std::filesystem::path batch_path;
  std::filesystem::path base_dir;
  std::string output_dir;
  std::string output_path;
  std::string output_format;
  std::filesystem::path manifest_json;
  std::filesystem::path manifest_csv;
  std::vector<BatchRunEntry> runs;
};

struct BatchRunResult {
  int index = 0;
  std::string name;
  std::filesystem::path source_path;
  RunConfig config;
  bool ok = false;
  std::string error;
  std::string output_path;
  std::string output_format;
  bool time_series = false;
  int frame_count = 0;
  BackendKind requested_backend = BackendKind::Auto;
  BackendKind selected_backend = BackendKind::CPU;
  std::string backend_note;
  double solve_seconds = 0.0;
  double write_seconds = 0.0;
  double total_seconds = 0.0;
  double residual_l2 = 0.0;
  double residual_linf = 0.0;
};

bool ReadTextFile(const std::filesystem::path& path, std::string* out, std::string* error) {
  std::ifstream in(path);
  if (!in.is_open()) {
    if (error) {
      *error = "failed to open file: " + path.string();
    }
    return false;
  }
  std::ostringstream buffer;
  buffer << in.rdbuf();
  *out = buffer.str();
  return true;
}

bool ReadJsonStringField(const json& j, const char* key, std::string* out, std::string* error) {
  if (!j.contains(key)) {
    return true;
  }
  const json& value = j.at(key);
  if (!value.is_string()) {
    if (error) {
      *error = std::string("expected string for '") + key + "'";
    }
    return false;
  }
  if (out) {
    *out = value.get<std::string>();
  }
  return true;
}

bool ReadJsonIntField(const json& j, const char* key, int* out, std::string* error) {
  if (!j.contains(key)) {
    return true;
  }
  const json& value = j.at(key);
  if (!value.is_number_integer()) {
    if (error) {
      *error = std::string("expected integer for '") + key + "'";
    }
    return false;
  }
  if (out) {
    *out = value.get<int>();
  }
  return true;
}

std::filesystem::path ResolveBatchPath(const std::filesystem::path& base_dir,
                                       const std::filesystem::path& path) {
  if (path.empty() || path.is_absolute() || base_dir.empty()) {
    return path;
  }
  return base_dir / path;
}

std::string DefaultRunName(const std::string& name, int index) {
  if (!name.empty()) {
    return name;
  }
  return "run_" + std::to_string(index);
}

std::string CsvEscape(const std::string& field) {
  if (field.find_first_of(",\"\n") == std::string::npos) {
    return field;
  }
  std::string out = "\"";
  for (char ch : field) {
    if (ch == '"') {
      out += "\"\"";
    } else {
      out += ch;
    }
  }
  out += "\"";
  return out;
}

bool LoadBatchSpec(const std::filesystem::path& path,
                   const std::string& out_dir_override,
                   const std::string& format_override,
                   BatchSpec* spec,
                   std::string* error) {
  if (!spec) {
    if (error) {
      *error = "missing batch spec output";
    }
    return false;
  }
  std::string content;
  if (!ReadTextFile(path, &content, error)) {
    return false;
  }
  json root;
  try {
    root = json::parse(content);
  } catch (const json::exception& exc) {
    if (error) {
      *error = std::string("invalid JSON: ") + exc.what();
    }
    return false;
  }
  if (!root.is_object()) {
    if (error) {
      *error = "batch spec must be a JSON object";
    }
    return false;
  }
  int schema_version = 0;
  if (!ReadJsonIntField(root, "schema_version", &schema_version, error)) {
    return false;
  }
  if (schema_version != 1) {
    if (error) {
      *error = "unsupported batch schema_version";
    }
    return false;
  }

  BatchSpec parsed;
  parsed.batch_path = path;
  parsed.base_dir = path.parent_path();
  if (parsed.base_dir.empty()) {
    parsed.base_dir = ".";
  }

  if (root.contains("output")) {
    const json& output = root.at("output");
    if (!output.is_object()) {
      if (error) {
        *error = "output must be an object";
      }
      return false;
    }
    if (!ReadJsonStringField(output, "dir", &parsed.output_dir, error)) return false;
    if (!ReadJsonStringField(output, "path", &parsed.output_path, error)) return false;
    if (!ReadJsonStringField(output, "format", &parsed.output_format, error)) return false;
  }

  if (!format_override.empty()) {
    parsed.output_format = format_override;
  }
  if (!parsed.output_format.empty()) {
    parsed.output_format = pde::ToLower(parsed.output_format);
    if (parsed.output_format != "vtk" && parsed.output_format != "vti") {
      if (error) {
        *error = "output.format must be vtk or vti";
      }
      return false;
    }
  }
  if (!out_dir_override.empty()) {
    parsed.output_dir = out_dir_override;
  }

  if (root.contains("manifest")) {
    const json& manifest = root.at("manifest");
    if (!manifest.is_object()) {
      if (error) {
        *error = "manifest must be an object";
      }
      return false;
    }
    std::string json_path;
    std::string csv_path;
    if (!ReadJsonStringField(manifest, "json", &json_path, error)) return false;
    if (!ReadJsonStringField(manifest, "csv", &csv_path, error)) return false;
    if (!json_path.empty()) {
      parsed.manifest_json = ResolveBatchPath(parsed.base_dir, json_path);
    }
    if (!csv_path.empty()) {
      parsed.manifest_csv = ResolveBatchPath(parsed.base_dir, csv_path);
    }
  }

  if (!root.contains("runs") || !root.at("runs").is_array()) {
    if (error) {
      *error = "runs must be a JSON array";
    }
    return false;
  }

  int index = 1;
  for (const auto& run_entry : root.at("runs")) {
    BatchRunEntry entry;
    entry.index = index++;
    if (run_entry.is_string()) {
      std::filesystem::path config_path = ResolveBatchPath(parsed.base_dir,
                                                           run_entry.get<std::string>());
      entry.source_path = config_path;
      std::string run_error;
      if (!LoadRunConfigFromFile(config_path, &entry.config, &run_error)) {
        if (error) {
          *error = "failed to load run config: " + run_error;
        }
        return false;
      }
      entry.name = config_path.stem().string();
    } else if (run_entry.is_object()) {
      std::string name;
      if (!ReadJsonStringField(run_entry, "name", &name, error)) return false;
      if (run_entry.contains("config")) {
        const json& config_value = run_entry.at("config");
        if (!config_value.is_string()) {
          if (error) {
            *error = "run entry config must be a string path";
          }
          return false;
        }
        std::filesystem::path config_path =
            ResolveBatchPath(parsed.base_dir, config_value.get<std::string>());
        entry.source_path = config_path;
        std::string run_error;
        if (!LoadRunConfigFromFile(config_path, &entry.config, &run_error)) {
          if (error) {
            *error = "failed to load run config: " + run_error;
          }
          return false;
        }
        entry.name = name.empty() ? config_path.stem().string() : name;
      } else {
        json run_spec = run_entry;
        if (run_entry.contains("run")) {
          const json& run_value = run_entry.at("run");
          if (!run_value.is_object()) {
            if (error) {
              *error = "run entry 'run' must be an object";
            }
            return false;
          }
          run_spec = run_value;
        }
        std::string run_error;
        if (!LoadRunConfigFromString(run_spec.dump(), &entry.config, &run_error)) {
          if (error) {
            *error = "failed to parse run config: " + run_error;
          }
          return false;
        }
        entry.name = name;
      }
    } else {
      if (error) {
        *error = "run entries must be objects or config file paths";
      }
      return false;
    }

    entry.name = DefaultRunName(entry.name, entry.index);

    if (entry.config.output_path.empty() && !parsed.output_path.empty()) {
      entry.config.output_path = parsed.output_path;
    }
    if (!parsed.output_dir.empty() &&
        (entry.config.output_dir.empty() || entry.config.output_dir == "outputs")) {
      entry.config.output_dir = parsed.output_dir;
    }
    if (!parsed.output_format.empty() &&
        (entry.config.output_format.empty() ||
         pde::ToLower(entry.config.output_format) == "vtk")) {
      entry.config.output_format = parsed.output_format;
    }

    parsed.runs.push_back(std::move(entry));
  }

  if (parsed.runs.empty()) {
    if (error) {
      *error = "batch spec has no runs";
    }
    return false;
  }

  if (parsed.manifest_json.empty()) {
    const std::string stem = path.stem().string();
    parsed.manifest_json = parsed.base_dir / (stem + "_manifest.json");
  }
  if (parsed.manifest_csv.empty()) {
    const std::string stem = path.stem().string();
    parsed.manifest_csv = parsed.base_dir / (stem + "_manifest.csv");
  }

  *spec = std::move(parsed);
  return true;
}

std::filesystem::path ResolvePatternPath(const std::filesystem::path& base_dir,
                                         const std::string& pattern,
                                         const OutputPatternContext& ctx) {
  if (pattern.empty()) {
    return {};
  }
  std::string expanded = ApplyOutputPattern(pattern, ctx);
  std::filesystem::path resolved(expanded);
  if (!resolved.is_absolute() && !base_dir.empty()) {
    resolved = base_dir / resolved;
  }
  return resolved;
}

BatchRunResult RunBatchEntry(const BatchRunEntry& entry, const BatchSpec& spec) {
  BatchRunResult result;
  result.index = entry.index;
  result.name = entry.name;
  result.source_path = entry.source_path;
  result.config = entry.config;

  RunConfig config = entry.config;
  if (!config.coord_mode.empty()) {
    const std::string token = pde::ToLower(config.coord_mode);
    if (token != "cartesian" && token != "cartesian2d" && token != "cartesian3d") {
      result.error = "config coord_mode unsupported in CLI: " + config.coord_mode;
      return result;
    }
  }
  if (!config.backend.empty() && !IsBackendToken(config.backend)) {
    result.error = "unknown backend in config: " + config.backend;
    return result;
  }
  if (!config.method.empty() && !IsMethodToken(config.method)) {
    result.error = "unknown method in config: " + config.method;
    return result;
  }
  if (config.output_format.empty()) {
    config.output_format = "vtk";
  }
  config.output_format = pde::ToLower(config.output_format);
  if (config.output_format != "vtk" && config.output_format != "vti") {
    result.error = "invalid output format: " + config.output_format;
    return result;
  }

  LatexParser parser;
  LatexParseResult parse_result = parser.Parse(config.pde_latex);
  if (!parse_result.ok) {
    result.error = "latex parse error: " + parse_result.error;
    return result;
  }
  CoefficientEvaluator coeff_eval = BuildCoefficientEvaluator(parse_result.coeffs);
  if (!coeff_eval.ok) {
    result.error = "invalid coefficient expression: " + coeff_eval.error;
    return result;
  }

  Domain domain;
  ParseResult domain_result = ParseDomain(config.domain_bounds, &domain);
  if (!domain_result.ok) {
    result.error = domain_result.error;
    return result;
  }
  ParseResult grid_result = ParseGrid(config.grid, &domain);
  if (!grid_result.ok) {
    result.error = grid_result.error;
    return result;
  }

  BoundarySet bc;
  if (!config.boundary_spec.empty()) {
    ParseResult bc_result = ApplyBoundarySpec(config.boundary_spec, &bc);
    if (!bc_result.ok) {
      result.error = "invalid boundary spec: " + bc_result.error;
      return result;
    }
    ParseResult bc_validate = ValidateBoundaryConditions(bc, domain);
    if (!bc_validate.ok) {
      result.error = "boundary validation failed: " + bc_validate.error;
      return result;
    }
  }

  TimeConfig time_config = config.time;
  if (time_config.frames > 1 && time_config.dt <= 0.0) {
    time_config.dt = (time_config.t_end - time_config.t_start) /
                     static_cast<double>(std::max(1, time_config.frames - 1));
  }

  SolverConfig solver = config.solver;
  const std::string method_token = config.method.empty() ? "jacobi" : config.method;
  solver.method = ParseSolveMethodToken(method_token);
  const std::string backend_token = config.backend.empty() ? "auto" : config.backend;
  BackendKind backend = ParseBackendKind(backend_token);

  SolveInput input;
  input.pde = parse_result.coeffs;
  input.integrals = parse_result.integrals;
  input.nonlinear = parse_result.nonlinear;
  input.nonlinear_derivatives = parse_result.nonlinear_derivatives;
  input.domain = domain;
  input.bc = bc;
  std::string shape_expr = NormalizeShapeExpression(config.domain_shape);
  if (shape_expr.empty() && !config.domain_shape_file.empty()) {
    std::string shape_error;
    if (!LoadShapeExpressionFromFile(config.domain_shape_file, &shape_expr, &shape_error)) {
      result.error = "shape file error: " + shape_error;
      return result;
    }
  }
  input.domain_shape = shape_expr;
  input.shape_transform = config.shape_transform;
  input.shape_mask_threshold = config.domain_shape_mask_threshold;
  input.shape_mask_invert = config.domain_shape_mask_invert;
  if (!config.domain_shape_mask.empty()) {
    std::string mask_error;
    if (!LoadShapeMaskFromVtk(config.domain_shape_mask, &input.shape_mask, &mask_error)) {
      result.error = "shape mask error: " + mask_error;
      return result;
    }
  }
  input.time = time_config;
  input.solver = solver;

  OutputPatternContext ctx;
  ctx.index = entry.index;
  ctx.index0 = entry.index - 1;
  ctx.name = entry.name;
  ctx.backend = pde::ToLower(backend_token);
  ctx.method = pde::ToLower(method_token);
  ctx.format = config.output_format;
  ctx.domain = domain;
  ctx.tag = GenerateRandomTag(6);
  ctx.timestamp = FormatTimestamp();

  const bool out_path_pattern = ContainsPatternToken(config.output_path);
  std::filesystem::path resolved_out_path = ResolvePatternPath(spec.base_dir, config.output_path, ctx);
  std::filesystem::path resolved_out_dir = ResolvePatternPath(spec.base_dir, config.output_dir, ctx);
  bool output_path_is_dir = false;
  if (!config.output_path.empty()) {
    const char last = config.output_path.back();
    output_path_is_dir = (last == '/' || last == '\\');
  }
  if (output_path_is_dir) {
    resolved_out_dir = resolved_out_path;
    resolved_out_path.clear();
  } else if (!resolved_out_path.empty()) {
    std::error_code ec;
    if (std::filesystem::exists(resolved_out_path, ec) &&
        std::filesystem::is_directory(resolved_out_path, ec)) {
      resolved_out_dir = resolved_out_path;
      resolved_out_path.clear();
    }
  }

  if (resolved_out_dir.empty()) {
    resolved_out_dir = ResolveBatchPath(spec.base_dir, "outputs");
  }

  const std::string safe_name = SanitizeTokenValue(entry.name);
  const std::string base_name =
      safe_name.empty() ? ("run_" + std::to_string(entry.index)) : safe_name;

  ProgressCallback cb;
  auto solve_start = std::chrono::steady_clock::now();
  double last_l2 = -1.0;
  int sample = 0;
  if (input.solver.residual_interval > 0) {
    cb = [&](const std::string& phase, double value) {
      if (phase == "residual_l2") {
        last_l2 = value;
        return;
      }
      if (phase == "residual_linf") {
        const double t = std::chrono::duration_cast<std::chrono::duration<double>>(
                             std::chrono::steady_clock::now() - solve_start)
                             .count();
        std::cout << "[" << entry.index << ":" << entry.name << "] residual[" << sample++
                  << "] t=" << t << " L2=" << last_l2 << " Linf=" << value << "\n";
        return;
      }
    };
  }

  result.requested_backend = backend;
  result.output_format = config.output_format;

  if (time_config.enabled) {
    BackendKind selected_backend = BackendKind::CPU;
    std::string selection_note_ts;
    if (backend != BackendKind::Auto && backend != BackendKind::CPU) {
      selection_note_ts = "time series uses CPU backend";
    }

    std::error_code ec;
    std::filesystem::path series_base;
    if (!resolved_out_path.empty()) {
      series_base = resolved_out_path;
      if (series_base.has_extension()) {
        series_base.replace_extension("");
      }
    } else {
      std::filesystem::create_directories(resolved_out_dir, ec);
      if (ec) {
        result.error = "failed to create output directory: " + ec.message();
        return result;
      }
      series_base = resolved_out_dir / base_name;
    }

    const int digits = std::max(4, static_cast<int>(std::log10(time_config.frames) + 1));
    if (out_path_pattern || config.output_path.empty()) {
      series_base = EnsureUniqueSeriesBase(series_base, digits, config.output_format);
    }

    std::vector<std::string> series_paths;
    std::vector<double> series_times;
    std::string frame_error;
    ConservedMonitor monitor;

    auto frame_cb = [&](int frame, double t, const std::vector<double>& grid,
                        const std::vector<double>* velocity) -> bool {
      const ConservedSample sample = UpdateConservedMonitor(input.domain, frame, grid, &monitor);
      if (sample.ok) {
        if (monitor.mass_warning && monitor.mass_warning_frame == frame) {
          std::cerr << "warning: mass drift exceeds 1% at frame " << frame << "\n";
        }
        if (monitor.energy_warning && monitor.energy_warning_frame == frame) {
          std::cerr << "warning: energy drift exceeds 1% at frame " << frame << "\n";
        }
        if (monitor.blowup_warning && monitor.blowup_warning_frame == frame) {
          std::cerr << "warning: rapid growth detected at frame " << frame
                    << " (ratio " << monitor.blowup_ratio
                    << ", ||u||inf " << monitor.blowup_max << ")\n";
        }
      }
      const std::filesystem::path frame_path =
          BuildSeriesFramePath(series_base, frame, digits, config.output_format);

      std::vector<std::vector<double>> derived_field_data;
      std::vector<std::string> derived_field_names;
      BuildDerivedFieldVectors(input.domain, grid, input.pde.a, input.pde.b, input.pde.az,
                               &derived_field_data, &derived_field_names);

      VtkWriteResult write_result;
      if (config.output_format == "vti") {
        write_result = WriteVtkXmlImageData(frame_path.string(), input.domain, grid,
                                            &derived_field_data, &derived_field_names);
      } else {
        write_result = WriteVtkStructuredPoints(frame_path.string(), input.domain, grid);
      }
      if (!write_result.ok) {
        frame_error = write_result.error;
        return false;
      }
      std::string meta_error;
      if (!WriteRunMetadataSidecar(frame_path, config, backend, selected_backend,
                                   selection_note_ts, true, frame, t, &meta_error)) {
        std::cerr << "metadata warning: " << meta_error << "\n";
      }
      RunSummaryData summary;
      summary.run_config = config;
      summary.requested_backend = backend;
      summary.selected_backend = selected_backend;
      summary.backend_note = selection_note_ts;
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
        std::cerr << "summary warning: " << summary_error << "\n";
      }
      series_paths.push_back(frame_path.string());
      series_times.push_back(t);
      return true;
    };

    SolveOutput output = SolvePDETimeSeries(input, frame_cb, cb);
    result.solve_seconds = std::chrono::duration_cast<std::chrono::duration<double>>(
                               std::chrono::steady_clock::now() - solve_start)
                               .count();
    result.total_seconds = result.solve_seconds;
    if (!output.error.empty()) {
      result.error = "solve error: " + output.error;
      return result;
    }
    if (!frame_error.empty()) {
      result.error = "frame write error: " + frame_error;
      return result;
    }

    if (!series_paths.empty()) {
      const std::filesystem::path manifest = series_base.parent_path() /
                                             (series_base.stem().string() + "_series.pvd");
      WriteVtkSeriesPvd(manifest.string(), series_paths, series_times);
      std::cout << "wrote series manifest: " << manifest << "\n";
      std::string meta_error;
      if (!WriteRunMetadataSidecar(manifest, config, backend, selected_backend,
                                   selection_note_ts, true, -1, 0.0, &meta_error)) {
        std::cerr << "metadata warning: " << meta_error << "\n";
      }
      RunSummaryData summary;
      summary.run_config = config;
      summary.requested_backend = backend;
      summary.selected_backend = selected_backend;
      summary.backend_note = selection_note_ts;
      summary.output_path = manifest.string();
      summary.time_series = true;
      summary.solve_seconds = result.solve_seconds;
      summary.total_seconds = result.solve_seconds;
      summary.frame_times = series_times;
      std::string summary_error;
      if (!WriteRunSummarySidecar(manifest, summary, &summary_error)) {
        std::cerr << "summary warning: " << summary_error << "\n";
      }
      result.output_path = manifest.string();
      result.time_series = true;
      result.frame_count = static_cast<int>(series_paths.size());
    } else {
      result.output_path = series_base.string();
    }

    result.selected_backend = selected_backend;
    result.backend_note = selection_note_ts;
    result.ok = true;
    return result;
  }

  SolveOutput output;
  BackendKind selected = BackendKind::CPU;
  std::string selection_note;
  output = SolveWithBackend(input, backend, &selected, &selection_note, cb);
  result.solve_seconds = std::chrono::duration_cast<std::chrono::duration<double>>(
                             std::chrono::steady_clock::now() - solve_start)
                             .count();
  if (!output.error.empty()) {
    result.error = "solve error: " + output.error;
    return result;
  }

  std::filesystem::path final_path;
  std::error_code ec;
  if (!resolved_out_path.empty()) {
    std::filesystem::path candidate(resolved_out_path);
    if (std::filesystem::exists(candidate, ec) && std::filesystem::is_directory(candidate, ec)) {
      resolved_out_dir = candidate;
    } else {
      if (!candidate.has_extension()) {
        candidate += (config.output_format == "vti") ? ".vti" : ".vtk";
      } else {
        const std::string ext = candidate.extension().string();
        if (ext != ".vtk" && ext != ".vti") {
          candidate.replace_extension(config.output_format == "vti" ? ".vti" : ".vtk");
          std::cerr << "warning: output extension adjusted to ." << config.output_format << "\n";
        }
      }
      if (candidate.has_parent_path()) {
        std::filesystem::create_directories(candidate.parent_path(), ec);
        if (ec) {
          result.error = "failed to create output directory: " + ec.message();
          return result;
        }
      }
      final_path = candidate;
    }
  }

  if (final_path.empty()) {
    std::filesystem::create_directories(resolved_out_dir, ec);
    if (ec) {
      result.error = "failed to create output directory: " + ec.message();
      return result;
    }
    final_path = resolved_out_dir / (base_name +
                                     (config.output_format == "vti" ? ".vti" : ".vtk"));
  }

  if (out_path_pattern || config.output_path.empty()) {
    final_path = EnsureUniquePath(final_path);
  }

  auto write_start = std::chrono::steady_clock::now();
  std::vector<std::vector<double>> derived_field_data;
  std::vector<std::string> derived_field_names;
  BuildDerivedFieldVectors(domain, output.grid, input.pde.a, input.pde.b, input.pde.az,
                           &derived_field_data, &derived_field_names);

  std::filesystem::path vtk_path = final_path;
  if (vtk_path.extension() == ".vti" || config.output_format == "vti") {
    VtkWriteResult write_result = WriteVtkXmlImageData(
        vtk_path.string(), domain, output.grid, &derived_field_data, &derived_field_names);
    if (!write_result.ok) {
      result.error = "vtk xml write error: " + write_result.error;
      return result;
    }
  } else {
    VtkWriteResult write_result =
        WriteVtkStructuredPoints(final_path.string(), domain, output.grid);
    if (!write_result.ok) {
      result.error = "vtk write error: " + write_result.error;
      return result;
    }
  }
  result.write_seconds = std::chrono::duration_cast<std::chrono::duration<double>>(
                             std::chrono::steady_clock::now() - write_start)
                             .count();
  result.total_seconds = result.solve_seconds + result.write_seconds;

  std::string meta_error;
  if (!WriteRunMetadataSidecar(final_path, config, backend, selected, selection_note,
                               false, -1, 0.0, &meta_error)) {
    std::cerr << "metadata warning: " << meta_error << "\n";
  }
  RunSummaryData summary;
  summary.run_config = config;
  summary.requested_backend = backend;
  summary.selected_backend = selected;
  summary.backend_note = selection_note;
  summary.output_path = final_path.string();
  summary.time_series = false;
  summary.solve_seconds = result.solve_seconds;
  summary.write_seconds = result.write_seconds;
  summary.total_seconds = result.total_seconds;
  summary.residual_l2 = output.residual_l2;
  summary.residual_linf = output.residual_linf;
  summary.residual_iters = output.residual_iters;
  summary.residual_l2_history = output.residual_l2_history;
  summary.residual_linf_history = output.residual_linf_history;
  std::string summary_error;
  if (!WriteRunSummarySidecar(final_path, summary, &summary_error)) {
    std::cerr << "summary warning: " << summary_error << "\n";
  }

  result.output_path = final_path.string();
  result.residual_l2 = output.residual_l2;
  result.residual_linf = output.residual_linf;
  result.selected_backend = selected;
  result.backend_note = selection_note;
  result.ok = true;
  return result;
}

bool WriteBatchManifest(const BatchSpec& spec,
                        const std::vector<BatchRunResult>& results,
                        std::string* error) {
  json root;
  root["schema_version"] = 1;
  root["batch_path"] = spec.batch_path.string();
  root["generated_at"] = FormatTimestamp();

  int success_count = 0;
  int failure_count = 0;
  json runs = json::array();
  for (const auto& result : results) {
    if (result.ok) {
      ++success_count;
    } else {
      ++failure_count;
    }
    json entry;
    entry["index"] = result.index;
    entry["name"] = result.name;
    if (!result.source_path.empty()) {
      entry["config_path"] = result.source_path.string();
    }
    entry["status"] = result.ok ? "ok" : "error";
    if (!result.error.empty()) {
      entry["error"] = result.error;
    }
    if (!result.output_path.empty()) {
      entry["output_path"] = result.output_path;
      entry["metadata_path"] = MetadataSidecarPath(result.output_path).string();
      entry["summary_path"] = SummarySidecarPath(result.output_path).string();
    }
    entry["output_format"] = result.output_format;
    entry["time_series"] = result.time_series;
    entry["frames"] = result.frame_count;
    entry["requested_backend"] = BackendToken(result.requested_backend);
    entry["selected_backend"] = BackendToken(result.selected_backend);
    if (!result.backend_note.empty()) {
      entry["backend_note"] = result.backend_note;
    }
    entry["solve_seconds"] = result.solve_seconds;
    entry["write_seconds"] = result.write_seconds;
    entry["total_seconds"] = result.total_seconds;
    if (!result.time_series) {
      entry["residual_l2"] = result.residual_l2;
      entry["residual_linf"] = result.residual_linf;
    }
    try {
      entry["run_config"] = json::parse(SerializeRunConfig(result.config, 2));
    } catch (const json::exception&) {
      entry["run_config"] = SerializeRunConfig(result.config, 2);
    }
    runs.push_back(entry);
  }
  root["run_count"] = static_cast<int>(results.size());
  root["success_count"] = success_count;
  root["failure_count"] = failure_count;
  root["runs"] = runs;

  std::string json_payload;
  try {
    json_payload = root.dump(2);
  } catch (const std::exception& exc) {
    if (error) {
      *error = std::string("failed to serialize batch manifest: ") + exc.what();
    }
    return false;
  }
  if (!WriteTextFile(spec.manifest_json, json_payload, error)) {
    return false;
  }

  std::ostringstream csv;
  csv << "index,name,status,output_path,output_format,time_series,frames,requested_backend,"
         "selected_backend,solve_seconds,write_seconds,total_seconds,residual_l2,residual_linf,"
         "error,grid,domain,pde\n";
  csv.setf(std::ios::scientific);
  csv << std::setprecision(12);
  for (const auto& result : results) {
    const std::string status = result.ok ? "ok" : "error";
    const std::string grid = result.config.grid;
    const std::string domain = result.config.domain_bounds;
    const std::string pde = result.config.pde_latex;
    csv << result.index << ","
        << CsvEscape(result.name) << ","
        << status << ","
        << CsvEscape(result.output_path) << ","
        << result.output_format << ","
        << (result.time_series ? "true" : "false") << ","
        << result.frame_count << ","
        << BackendToken(result.requested_backend) << ","
        << BackendToken(result.selected_backend) << ","
        << result.solve_seconds << ","
        << result.write_seconds << ","
        << result.total_seconds << ","
        << result.residual_l2 << ","
        << result.residual_linf << ","
        << CsvEscape(result.error) << ","
        << CsvEscape(grid) << ","
        << CsvEscape(domain) << ","
        << CsvEscape(pde) << "\n";
  }
  return WriteTextFile(spec.manifest_csv, csv.str(), error);
}

int RunBatch(const BatchSpec& spec) {
  PrintBackendStatus();
  std::vector<BatchRunResult> results;
  results.reserve(spec.runs.size());
  bool all_ok = true;
  int run_index = 0;
  for (const auto& entry : spec.runs) {
    ++run_index;
    std::cout << "batch run " << run_index << "/" << spec.runs.size()
              << ": " << entry.name << "\n";
    BatchRunResult result = RunBatchEntry(entry, spec);
    if (!result.ok) {
      all_ok = false;
      std::cerr << "batch run failed (" << entry.name << "): " << result.error << "\n";
    }
    results.push_back(std::move(result));
  }

  std::string manifest_error;
  if (!WriteBatchManifest(spec, results, &manifest_error)) {
    std::cerr << "failed to write batch manifest: " << manifest_error << "\n";
    return 1;
  }
  std::cout << "wrote batch manifest: " << spec.manifest_json << "\n";
  std::cout << "wrote batch manifest: " << spec.manifest_csv << "\n";
  return all_ok ? 0 : 1;
}

int RunBatchFromArgs(int argc, char** argv) {
  std::string batch_path;
  std::string out_dir_override;
  std::string format_override;
  std::string backend_override;

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--help" || arg == "-h") {
      PrintUsage();
      return 0;
    }
    if (arg == "--batch") {
      if (i + 1 >= argc) {
        std::cerr << "--batch requires a file path\n";
        return 1;
      }
      batch_path = argv[++i];
      continue;
    }
    if (arg.rfind("--batch=", 0) == 0) {
      batch_path = arg.substr(std::string("--batch=").size());
      continue;
    }
    if (arg == "--out-dir") {
      if (i + 1 >= argc) {
        std::cerr << "--out-dir requires a value\n";
        return 1;
      }
      out_dir_override = argv[++i];
      continue;
    }
    if (arg.rfind("--out-dir=", 0) == 0) {
      out_dir_override = arg.substr(std::string("--out-dir=").size());
      continue;
    }
    if (arg == "--format") {
      if (i + 1 >= argc) {
        std::cerr << "--format requires a value\n";
        return 1;
      }
      format_override = argv[++i];
      continue;
    }
    if (arg.rfind("--format=", 0) == 0) {
      format_override = arg.substr(std::string("--format=").size());
      continue;
    }
    if (arg == "--backend") {
      if (i + 1 >= argc) {
        std::cerr << "--backend requires a value\n";
        return 1;
      }
      const std::string token = argv[++i];
      if (!IsBackendToken(token)) {
        std::cerr << "unknown backend: " << token << "\n";
        return 1;
      }
      backend_override = token;
      continue;
    }
    if (arg.rfind("--backend=", 0) == 0) {
      const std::string token = arg.substr(std::string("--backend=").size());
      if (!IsBackendToken(token)) {
        std::cerr << "unknown backend: " << token << "\n";
        return 1;
      }
      backend_override = token;
      continue;
    }

    if (arg.rfind("--", 0) == 0) {
      std::cerr << "unsupported option with --batch: " << arg << "\n";
      return 1;
    }
    if (!arg.empty() && arg[0] != '-') {
      std::cerr << "unexpected argument with --batch: " << arg << "\n";
      return 1;
    }
  }

  if (batch_path.empty()) {
    std::cerr << "--batch requires a file path\n";
    return 1;
  }
  BatchSpec spec;
  std::string batch_error;
  if (!LoadBatchSpec(batch_path, out_dir_override, format_override, &spec, &batch_error)) {
    std::cerr << "batch load error: " << batch_error << "\n";
    return 1;
  }
  if (!backend_override.empty()) {
    for (auto& entry : spec.runs) {
      entry.config.backend = backend_override;
    }
  }
  return RunBatch(spec);
}

bool HasBatchArg(int argc, char** argv) {
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--batch" || arg.rfind("--batch=", 0) == 0) {
      return true;
    }
  }
  return false;
}

std::filesystem::path MakeRandomOutputPath(const std::filesystem::path& dir) {
  return dir / ("pde_" + GenerateRandomTag(6) + ".vtk");
}

std::filesystem::path ResolveExecutableDir(const char* argv0) {
  std::error_code ec;
  if (!argv0 || std::string(argv0).empty()) {
    return std::filesystem::current_path(ec);
  }
  std::filesystem::path exe_path(argv0);
  if (exe_path.is_relative()) {
    exe_path = std::filesystem::current_path(ec) / exe_path;
  }
  std::filesystem::path resolved = std::filesystem::weakly_canonical(exe_path, ec);
  if (ec) {
    resolved = std::filesystem::absolute(exe_path, ec);
  }
  if (ec) {
    return std::filesystem::current_path(ec);
  }
  return resolved.has_parent_path() ? resolved.parent_path() : resolved;
}

std::string DefaultOutputDir(const char* argv0) {
  std::filesystem::path dir = ResolveExecutableDir(argv0);
  dir /= "outputs";
  return dir.string();
}

bool ParseShapeVector(const std::string& text, double* x, double* y, double* z, std::string* error) {
  std::vector<double> values;
  std::istringstream iss(text);
  std::string token;
  while (std::getline(iss, token, ',')) {
    values.push_back(std::strtod(token.c_str(), nullptr));
  }
  if (values.size() < 2 || values.size() > 3) {
    if (error) {
      *error = "expected x,y or x,y,z";
    }
    return false;
  }
  if (x) {
    *x = values[0];
  }
  if (y) {
    *y = values[1];
  }
  if (values.size() == 3 && z) {
    *z = values[2];
  }
  return true;
}

std::string MeshCellTypeName(int vtk_type) {
  switch (vtk_type) {
    case 1: return "vertex";
    case 3: return "line";
    case 5: return "triangle";
    case 9: return "quad";
    case 10: return "tetra";
    case 12: return "hexahedron";
    case 13: return "wedge";
    case 14: return "pyramid";
    default: return "unknown";
  }
}

void PrintMeshSummary(const MeshSummary& summary, const UnstructuredMesh& mesh) {
  std::cout << "Mesh summary:\n";
  std::cout << "  points: " << summary.point_count << "\n";
  std::cout << "  cells: " << summary.cell_count << "\n";
  std::cout << "  dimension: " << summary.dimension << "D\n";
  std::cout << "  bounds: [" << summary.xmin << ", " << summary.xmax << "] x ["
            << summary.ymin << ", " << summary.ymax << "] x ["
            << summary.zmin << ", " << summary.zmax << "]\n";

  std::map<int, int> type_histogram;
  for (int type : mesh.cell_types) {
    ++type_histogram[type];
  }
  if (!type_histogram.empty()) {
    std::cout << "  cell types:\n";
    for (const auto& entry : type_histogram) {
      std::cout << "    " << MeshCellTypeName(entry.first)
                << " (" << entry.first << "): " << entry.second << "\n";
    }
  }
  if (mesh.point_scalars.size() == static_cast<size_t>(summary.point_count)) {
    const std::string name =
        mesh.point_scalar_name.empty() ? "scalar" : mesh.point_scalar_name;
    std::cout << "  point scalars: " << name << "\n";
  }
  if (mesh.cell_scalars.size() == static_cast<size_t>(summary.cell_count)) {
    const std::string name =
        mesh.cell_scalar_name.empty() ? "cell_scalar" : mesh.cell_scalar_name;
    std::cout << "  cell scalars: " << name << "\n";
  }
}

void EnsureMeshPreviewScalars(UnstructuredMesh* mesh) {
  if (!mesh) {
    return;
  }
  const size_t point_count = mesh->points.size() / 3;
  if (point_count == 0) {
    return;
  }
  if (!mesh->point_scalars.empty()) {
    return;
  }
  mesh->point_scalars.assign(point_count, 0.0);
  if (mesh->point_scalar_name.empty()) {
    mesh->point_scalar_name = "preview";
  }
}

int RunUnstructuredMeshWorkflow(const std::string& mesh_path,
                                const std::string& mesh_format,
                                const std::string& mesh_discretization,
                                bool mesh_solve,
                                bool validate_only,
                                const std::string& latex,
                                const std::string& out_path,
                                const std::string& out_dir,
                                const std::string& output_format) {
  MeshReadResult mesh_result = ReadUnstructuredMesh(mesh_path, mesh_format);
  if (!mesh_result.ok) {
    std::cerr << "mesh load error: " << mesh_result.error << "\n";
    return 1;
  }
  if (!mesh_result.warning.empty()) {
    std::cerr << "mesh load warning: " << mesh_result.warning << "\n";
  }
  const MeshSummary summary = SummarizeUnstructuredMesh(mesh_result.mesh);
  PrintMeshSummary(summary, mesh_result.mesh);

  if (validate_only) {
    return 0;
  }

  if (ContainsPatternToken(out_path) || ContainsPatternToken(out_dir)) {
    std::cerr << "warning: mesh output does not expand naming tokens\n";
  }

  UnstructuredDiscretization discretization = UnstructuredDiscretization::FiniteElement;
  if (!mesh_discretization.empty() &&
      !ParseDiscretizationToken(mesh_discretization, &discretization)) {
    std::cerr << "unknown mesh discretization: " << mesh_discretization << "\n";
    return 1;
  }

  UnstructuredMesh output_mesh = mesh_result.mesh;
  if (mesh_solve) {
    if (latex.empty()) {
      std::cerr << "--mesh-solve requires --pde\n";
      return 1;
    }
    LatexParser parser;
    LatexParseResult parse_result = parser.Parse(latex);
    if (!parse_result.ok) {
      std::cerr << "latex parse error: " << parse_result.error << "\n";
      return 1;
    }
    UnstructuredSolveInput input;
    input.pde = parse_result.coeffs;
    input.mesh = mesh_result.mesh;
    input.discretization = discretization;
    UnstructuredSolveOutput solve_output = SolveUnstructuredPDE(input);
    if (!solve_output.ok) {
      std::cerr << "unstructured solve error: " << solve_output.error << "\n";
      return 1;
    }
    if (!solve_output.point_values.empty()) {
      output_mesh.point_scalars = std::move(solve_output.point_values);
      output_mesh.point_scalar_name = "solution";
    }
  } else {
    EnsureMeshPreviewScalars(&output_mesh);
  }

  const std::string format = pde::ToLower(output_format);
  if (!format.empty() && format != "vtk") {
    std::cerr << "warning: mesh output only supports vtk; writing vtk\n";
  }

  std::string out_dir_resolved = out_dir.empty() ? "." : out_dir;
  std::filesystem::path final_path;
  std::error_code ec;
  if (!out_path.empty()) {
    std::filesystem::path candidate(out_path);
    if (std::filesystem::exists(candidate, ec) && std::filesystem::is_directory(candidate, ec)) {
      out_dir_resolved = candidate.string();
    } else {
      if (!candidate.has_extension()) {
        candidate += ".vtk";
      } else if (candidate.extension() != ".vtk") {
        candidate.replace_extension(".vtk");
        std::cerr << "warning: mesh output extension adjusted to .vtk\n";
      }
      if (candidate.has_parent_path()) {
        std::filesystem::create_directories(candidate.parent_path(), ec);
        if (ec) {
          std::cerr << "failed to create output directory: " << ec.message() << "\n";
          return 1;
        }
      }
      final_path = candidate;
    }
  }

  if (final_path.empty()) {
    std::filesystem::path out_dir_path(out_dir_resolved);
    std::filesystem::create_directories(out_dir_path, ec);
    if (ec) {
      std::cerr << "failed to create output directory: " << ec.message() << "\n";
      return 1;
    }
    final_path = out_dir_path / ("mesh_" + GenerateRandomTag(6) + ".vtk");
  }

  if (out_path.empty()) {
    final_path = EnsureUniquePath(final_path);
  }

  MeshWriteResult write_result = WriteVtkUnstructuredGrid(final_path.string(), output_mesh);
  if (!write_result.ok) {
    std::cerr << "mesh write error: " << write_result.error << "\n";
    return 1;
  }
  std::cout << "wrote \"" << final_path.string() << "\"\n";
  return 0;
}

RunConfig BuildRunConfigForCli(const std::string& latex,
                               const std::string& domain_str,
                               const std::string& grid_str,
                               const std::string& bc_str,
                               const std::string& shape_str,
                               const std::string& shape_file,
                               const std::string& shape_mask,
                               double shape_mask_threshold,
                               bool shape_mask_invert,
                               const ShapeTransform& shape_transform,
                               const Domain& domain,
                               BackendKind backend,
                               SolveMethod method,
                               const SolverConfig& solver,
                               const TimeConfig& time_config,
                               const std::string& out_path,
                               const std::string& out_dir,
                               const std::string& output_format) {
  RunConfig config;
  config.pde_latex = latex;
  config.domain_bounds = domain_str;
  config.grid = grid_str;
  config.boundary_spec = bc_str.empty() ? DefaultBoundarySpec() : bc_str;
  config.domain_shape = shape_str;
  config.domain_shape_file = shape_file;
  config.domain_shape_mask = shape_mask;
  config.domain_shape_mask_threshold = shape_mask_threshold;
  config.domain_shape_mask_invert = shape_mask_invert;
  config.shape_transform = shape_transform;
  const bool has_shape = !shape_str.empty() || !shape_mask.empty();
  config.domain_mode = has_shape ? "implicit" : "box";
  config.coord_mode = (domain.nz > 1) ? "cartesian3d" : "cartesian2d";
  config.backend = BackendToken(backend);
  config.method = MethodToken(method);
  config.solver = solver;
  config.time = time_config;
  config.output_path = out_path;
  config.output_dir = out_dir;
  config.output_format = output_format;
  return config;
}

RunConfig BuildRunConfigForMesh(const std::string& mesh_path,
                                const std::string& mesh_format,
                                const std::string& mesh_discretization,
                                const std::string& latex,
                                const std::string& out_path,
                                const std::string& out_dir,
                                const std::string& output_format) {
  RunConfig config;
  config.pde_latex = latex;
  config.domain_mesh = mesh_path;
  config.domain_mesh_format = mesh_format;
  config.domain_mesh_discretization = mesh_discretization;
  config.domain_mode = "mesh";
  config.output_path = out_path;
  config.output_dir = out_dir;
  config.output_format = output_format;
  return config;
}

}

int main(int argc, char** argv) {
  if (argc == 1) {
    PrintUsage();
    return 1;
  }
  if (HasBatchArg(argc, argv)) {
    return RunBatchFromArgs(argc, argv);
  }

  std::string latex;
  std::string domain_str;
  std::string grid_str;
  std::string bc_str;
  std::string shape_str;
  std::string shape_file;
  std::string shape_mask_path;
  ShapeTransform shape_transform;
  ShapeMask shape_mask;
  double shape_mask_threshold = 0.0;
  bool shape_mask_invert = false;
  std::string mesh_path;
  std::string mesh_format;
  std::string mesh_discretization = "fe";
  bool mesh_solve = false;
  const std::string default_out_dir = DefaultOutputDir((argc > 0) ? argv[0] : nullptr);
  std::string out_path;
  std::string out_dir = default_out_dir;
  bool out_dir_arg = false;
  std::string in_dir;
  std::string config_path;
  std::string export_config_path;
  std::string checkpoint_path;
  std::string restart_path;
  std::string output_format = "vtk";  // "vtk" or "vti"
  SolverConfig solver;
  BackendKind backend = BackendKind::Auto;
  SolveMethod method = SolveMethod::Jacobi;
  TimeConfig time_config;
  bool validate_only = false;
  bool dump_operator = false;
  bool dump_operator_json = false;
  bool dump_metadata = false;
  std::string dump_metadata_path;
  bool dump_summary = false;
  std::string dump_summary_path;
  bool use_mms = false;
  bool mms_forced = false;
  std::string convergence_spec;
  std::string convergence_out_path;
  std::string convergence_plot_path;
  std::string convergence_json_path;
  std::string dataset_index_dir;
  std::string dataset_index_out;
  std::string dataset_cleanup_dir;
  bool dataset_cleanup_dry_run = false;
  bool dataset_cleanup_empty_dirs = false;

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--config") {
      if (i + 1 >= argc) {
        std::cerr << "--config requires a file path\n";
        return 1;
      }
      config_path = argv[++i];
      continue;
    }
    if (arg == "--export-config") {
      if (i + 1 >= argc) {
        std::cerr << "--export-config requires a file path\n";
        return 1;
      }
      export_config_path = argv[++i];
      continue;
    }
  }

  if (!config_path.empty()) {
    RunConfig config;
    std::string config_error;
    if (!LoadRunConfigFromFile(config_path, &config, &config_error)) {
      std::cerr << "config load error: " << config_error << "\n";
      return 1;
    }
    const std::string coord_mode = config.coord_mode;
    if (!coord_mode.empty()) {
      const std::string token = pde::ToLower(coord_mode);
      if (token != "cartesian" && token != "cartesian2d" && token != "cartesian3d") {
        std::cerr << "config coord_mode unsupported in CLI: " << coord_mode << "\n";
        return 1;
      }
    }
    if (!config.pde_latex.empty()) {
      latex = config.pde_latex;
    }
    if (!config.domain_bounds.empty()) {
      domain_str = config.domain_bounds;
    }
    if (!config.grid.empty()) {
      grid_str = config.grid;
    }
    if (!config.boundary_spec.empty()) {
      bc_str = config.boundary_spec;
    }
    if (!config.domain_shape.empty()) {
      shape_str = NormalizeShapeExpression(config.domain_shape);
    }
    if (!config.domain_shape_file.empty()) {
      shape_file = config.domain_shape_file;
    }
    if (!config.domain_shape_mask.empty()) {
      shape_mask_path = config.domain_shape_mask;
    }
    shape_mask_threshold = config.domain_shape_mask_threshold;
    shape_mask_invert = config.domain_shape_mask_invert;
    shape_transform = config.shape_transform;
    if (!config.domain_mesh.empty()) {
      mesh_path = config.domain_mesh;
    }
    if (!config.domain_mesh_format.empty()) {
      mesh_format = config.domain_mesh_format;
    }
    if (!config.domain_mesh_discretization.empty()) {
      mesh_discretization = config.domain_mesh_discretization;
    }
    if (!config.backend.empty()) {
      if (!IsBackendToken(config.backend)) {
        std::cerr << "unknown backend in config: " << config.backend << "\n";
        return 1;
      }
      backend = ParseBackendKind(config.backend);
    }
    if (!config.method.empty()) {
      if (!IsMethodToken(config.method)) {
        std::cerr << "unknown method in config: " << config.method << "\n";
        return 1;
      }
      method = ParseSolveMethodToken(config.method);
    }
    solver = config.solver;
    time_config = config.time;
    if (time_config.frames > 1 && time_config.dt <= 0.0) {
      time_config.dt = (time_config.t_end - time_config.t_start) /
                       static_cast<double>(std::max(1, time_config.frames - 1));
    }
    if (!config.output_path.empty()) {
      out_path = config.output_path;
    }
    if (!config.output_dir.empty()) {
      out_dir = config.output_dir;
    }
    if (!config.output_format.empty()) {
      output_format = pde::ToLower(config.output_format);
    }
  }

  if (!shape_file.empty() && shape_str.empty()) {
    std::string shape_error;
    if (!LoadShapeExpressionFromFile(shape_file, &shape_str, &shape_error)) {
      std::cerr << "shape file error: " << shape_error << "\n";
      return 1;
    }
  }
  if (!shape_mask_path.empty()) {
    std::string mask_error;
    if (!LoadShapeMaskFromVtk(shape_mask_path, &shape_mask, &mask_error)) {
      std::cerr << "shape mask error: " << mask_error << "\n";
      return 1;
    }
  }

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--help" || arg == "-h") {
      PrintUsage();
      return 0;
    }
    if (arg == "--self-test") {
      return RunSelfTest();
    }
    if (arg == "--list-examples") {
      auto examples = GetCoupledPDEExamples();
      std::cout << "Available coupled PDE examples:\n";
      for (const auto& ex : examples) {
        std::cout << "  " << ex.name << "\n";
        std::cout << "    " << ex.description << "\n";
        std::cout << "    Fields: ";
        for (size_t i = 0; i < ex.fields.size(); ++i) {
          if (i > 0) std::cout << ", ";
          std::cout << ex.fields[i].name;
        }
        std::cout << "\n    " << ex.notes << "\n\n";
      }
      return 0;
    }
    if (arg == "--run-example") {
      if (i + 1 >= argc) {
        std::cerr << "--run-example requires an example name\n";
        return 1;
      }
      std::string example_name = argv[++i];
      auto examples = GetCoupledPDEExamples();
      CoupledPDEExample found;
      bool found_match = false;
      for (const auto& ex : examples) {
        if (ex.name == example_name) {
          found = ex;
          found_match = true;
          break;
        }
      }
      if (!found_match) {
        std::cerr << "Unknown example: " << example_name << "\n";
        std::cerr << "Use --list-examples to see available examples\n";
        return 1;
      }
      std::cout << "Running example: " << found.name << "\n";
      std::cout << found.description << "\n";

      SolveInput input = BuildSolveInputFromExample(found);
      auto progress_cb = [](const std::string& phase, double pct) {
        std::cout << "\r" << phase << " " << static_cast<int>(pct * 100) << "%" << std::flush;
      };

      SolveOutput result;
      if (found.time.enabled) {
        CoupledFrameCallback on_frame = [&](int frame, double time,
            const std::map<std::string, std::vector<double>>& field_grids) {
          std::cout << "\nFrame " << frame << " t=" << time << "\n";
          for (const auto& kv : field_grids) {
            double min_val = *std::min_element(kv.second.begin(), kv.second.end());
            double max_val = *std::max_element(kv.second.begin(), kv.second.end());
            std::cout << "  " << kv.first << ": [" << min_val << ", " << max_val << "]\n";
          }
          return true;
        };
        result = SolveCoupledPDETimeSeries(input, on_frame, progress_cb);
      } else {
        result = SolveCoupledPDE(input, progress_cb);
      }

      std::cout << "\n";
      if (!result.error.empty()) {
        std::cerr << "Error: " << result.error << "\n";
        return 1;
      }
      std::cout << "Solve completed.\n";
      std::cout << "Coupling iterations: " << result.coupling_diagnostics.coupling_iters << "\n";
      std::cout << "Converged: " << (result.coupling_diagnostics.converged ? "yes" : "no") << "\n";
      if (!result.coupling_diagnostics.warning.empty()) {
        std::cout << "Warning: " << result.coupling_diagnostics.warning << "\n";
      }
      for (const auto& fo : result.field_outputs) {
        std::cout << "Field '" << fo.name << "': residual_l2=" << fo.residual_l2
                  << " residual_linf=" << fo.residual_linf << "\n";
      }
      return 0;
    }
    if (arg == "--advection-test") {
      // Get optional scheme name
      AdvectionScheme scheme = AdvectionScheme::Upwind;
      if (i + 1 < argc) {
        std::string next = argv[i + 1];
        if (!next.empty() && next[0] != '-') {
          scheme = ParseAdvectionScheme(next);
          ++i;
        }
      }
      std::cout << "Running advection tests with scheme: "
                << AdvectionSchemeToString(scheme) << "\n\n";

      // Run test suite for the specified scheme
      auto results = RunAdvectionTestSuite(scheme);
      PrintAdvectionTestResults(results);

      // Also run scheme comparison for top-hat test
      std::cout << "\nScheme comparison (Top-Hat 1D):\n";
      auto compare_results = CompareAdvectionSchemes(
          [](AdvectionScheme s) {
            return RunTopHatAdvectionTest1D(s, 100, 1.0, 1.0, 0.5);
          });
      PrintAdvectionTestResults(compare_results);

      // Count failures
      int failures = 0;
      for (const auto& r : results) {
        if (!r.passed) ++failures;
      }
      return (failures > 0) ? 1 : 0;
    }
    if (arg == "--time-integrator-test") {
      // Get optional method name
      TimeIntegrator method = TimeIntegrator::RK4;
      if (i + 1 < argc) {
        std::string next = argv[i + 1];
        if (!next.empty() && next[0] != '-') {
          method = ParseTimeIntegrator(next);
          ++i;
        }
      }
      std::cout << "Running time integrator tests with method: "
                << TimeIntegratorToString(method) << "\n\n";

      // Run test suite for the specified method
      auto results = RunTimeIntegratorTestSuite(method);
      PrintTimeIntegratorTestResults(results);

      // Run convergence study
      std::cout << "\nConvergence study (Exponential Decay):\n";
      auto conv_results = RunConvergenceStudy(method, {50, 100, 200, 400});
      PrintTimeIntegratorTestResults(conv_results);

      // Compare all methods
      std::cout << "\nMethod comparison:\n";
      auto compare_results = CompareTimeIntegrators();
      PrintTimeIntegratorTestResults(compare_results);

      // Run additional feature tests
      std::cout << "\nAdditional tests:\n";
      std::vector<TimeIntegratorTestResult> feature_results;
      feature_results.push_back(RunIMEXTest(100.0, 0.1, 100));
      feature_results.push_back(RunAdaptiveTimeStepTest(1.0, 1e-5));
      feature_results.push_back(RunCFLSteppingTest(1.0, 0.5, 100));
      PrintTimeIntegratorTestResults(feature_results);

      // Count all failures
      int failures = 0;
      for (const auto& r : results) {
        if (!r.passed) ++failures;
      }
      for (const auto& r : feature_results) {
        if (!r.passed) ++failures;
      }
      return (failures > 0) ? 1 : 0;
    }
    if (arg == "--projection-test") {
      std::cout << "Running pressure projection tests...\n\n";

      // Run basic tests
      auto results = RunPressureProjectionTestSuite();
      PrintPressureProjectionTestResults(results);

      // Run convergence study
      std::cout << "\nConvergence study:\n";
      auto conv_results = RunProjectionConvergenceTest({16, 32, 64});
      PrintPressureProjectionTestResults(conv_results);

      // Count failures
      int failures = 0;
      for (const auto& r : results) {
        if (!r.passed) ++failures;
      }
      return (failures > 0) ? 1 : 0;
    }
    if (arg == "--lid-cavity") {
      // Get optional Reynolds number
      double reynolds = 100.0;
      if (i + 1 < argc) {
        std::string next = argv[i + 1];
        if (!next.empty() && next[0] != '-') {
          reynolds = std::stod(next);
          ++i;
        }
      }
      std::cout << "Running lid-driven cavity benchmark at Re=" << reynolds << "\n\n";

      // Choose grid and steps based on Reynolds number
      int nx = 64, ny = 64, num_steps = 5000;
      if (reynolds > 500) {
        nx = 128; ny = 128; num_steps = 20000;
      }
      if (reynolds > 2000) {
        nx = 256; ny = 256; num_steps = 50000;
      }

      auto result = RunLidDrivenCavityTest(nx, ny, reynolds, num_steps);
      PrintLidDrivenCavityResult(result);

      return result.passed ? 0 : 1;
    }
    if (arg == "--mms") {
      use_mms = true;
      continue;
    }
    if (arg == "--convergence") {
      if (i + 1 >= argc) {
        std::cerr << "--convergence requires a grid list\n";
        return 1;
      }
      convergence_spec = argv[++i];
      if (!use_mms) {
        use_mms = true;
        mms_forced = true;
      }
      continue;
    }
    if (arg == "--convergence-out") {
      if (i + 1 >= argc) {
        std::cerr << "--convergence-out requires a file path\n";
        return 1;
      }
      convergence_out_path = argv[++i];
      continue;
    }
    if (arg == "--convergence-plot") {
      if (i + 1 < argc) {
        std::string next = argv[i + 1];
        if (!next.empty() && next[0] != '-') {
          convergence_plot_path = next;
          ++i;
        }
      }
      continue;
    }
    if (arg == "--convergence-json") {
      if (i + 1 < argc) {
        std::string next = argv[i + 1];
        if (!next.empty() && next[0] != '-') {
          convergence_json_path = next;
          ++i;
        }
      }
      continue;
    }
    if (arg == "--validate") {
      validate_only = true;
      continue;
    }
    if (arg == "--dump-operator") {
      dump_operator = true;
      if (i + 1 < argc) {
        std::string next = argv[i + 1];
        if (!next.empty() && next[0] != '-') {
          if (next == "json") {
            dump_operator_json = true;
            ++i;
          } else if (next == "text") {
            ++i;
          }
        }
      }
      continue;
    }
    if (arg == "--dump-metadata") {
      dump_metadata = true;
      if (i + 1 < argc) {
        std::string next = argv[i + 1];
        if (!next.empty() && next[0] != '-') {
          dump_metadata_path = next;
          ++i;
        }
      }
      continue;
    }
    if (arg == "--dump-summary") {
      dump_summary = true;
      if (i + 1 < argc) {
        std::string next = argv[i + 1];
        if (!next.empty() && next[0] != '-') {
          dump_summary_path = next;
          ++i;
        }
      }
      continue;
    }
    if (arg == "--dataset-index") {
      if (i + 1 >= argc) {
        std::cerr << "--dataset-index requires a directory\n";
        return 1;
      }
      dataset_index_dir = argv[++i];
      continue;
    }
    if (arg == "--dataset-index-out") {
      if (i + 1 >= argc) {
        std::cerr << "--dataset-index-out requires a file path\n";
        return 1;
      }
      dataset_index_out = argv[++i];
      continue;
    }
    if (arg == "--dataset-cleanup") {
      if (i + 1 >= argc) {
        std::cerr << "--dataset-cleanup requires a directory\n";
        return 1;
      }
      dataset_cleanup_dir = argv[++i];
      continue;
    }
    if (arg == "--dataset-cleanup-dry-run") {
      dataset_cleanup_dry_run = true;
      continue;
    }
    if (arg == "--dataset-cleanup-empty-dirs") {
      dataset_cleanup_empty_dirs = true;
      continue;
    }
    if (arg.rfind("--dump-operator=", 0) == 0) {
      dump_operator = true;
      std::string format = arg.substr(std::string("--dump-operator=").size());
      if (format.empty() || format == "text") {
        dump_operator_json = false;
      } else if (format == "json") {
        dump_operator_json = true;
      } else {
        std::cerr << "unknown dump-operator format: " << format << "\n";
        return 1;
      }
      continue;
    }
    if (arg.rfind("--dump-metadata=", 0) == 0) {
      dump_metadata = true;
      dump_metadata_path = arg.substr(std::string("--dump-metadata=").size());
      continue;
    }
    if (arg.rfind("--dump-summary=", 0) == 0) {
      dump_summary = true;
      dump_summary_path = arg.substr(std::string("--dump-summary=").size());
      continue;
    }
    if (arg.rfind("--dataset-index=", 0) == 0) {
      dataset_index_dir = arg.substr(std::string("--dataset-index=").size());
      continue;
    }
    if (arg.rfind("--dataset-index-out=", 0) == 0) {
      dataset_index_out = arg.substr(std::string("--dataset-index-out=").size());
      continue;
    }
    if (arg.rfind("--dataset-cleanup=", 0) == 0) {
      dataset_cleanup_dir = arg.substr(std::string("--dataset-cleanup=").size());
      continue;
    }
    if (arg.rfind("--convergence=", 0) == 0) {
      convergence_spec = arg.substr(std::string("--convergence=").size());
      if (!use_mms) {
        use_mms = true;
        mms_forced = true;
      }
      continue;
    }
    if (arg.rfind("--convergence-out=", 0) == 0) {
      convergence_out_path = arg.substr(std::string("--convergence-out=").size());
      continue;
    }
    if (arg.rfind("--convergence-plot=", 0) == 0) {
      convergence_plot_path = arg.substr(std::string("--convergence-plot=").size());
      continue;
    }
    if (arg.rfind("--convergence-json=", 0) == 0) {
      convergence_json_path = arg.substr(std::string("--convergence-json=").size());
      continue;
    }
    if (arg == "--config" || arg == "--export-config") {
      ++i;
      continue;
    }
    if (arg == "--pde" && i + 1 < argc) {
      latex = argv[++i];
    } else if (arg == "--domain" && i + 1 < argc) {
      domain_str = argv[++i];
    } else if (arg == "--grid" && i + 1 < argc) {
      grid_str = argv[++i];
    } else if (arg == "--bc" && i + 1 < argc) {
      bc_str = argv[++i];
    } else if ((arg == "--shape" || arg == "--domain-shape") && i + 1 < argc) {
      shape_str = NormalizeShapeExpression(argv[++i]);
      shape_file.clear();
    } else if (arg == "--shape-file" && i + 1 < argc) {
      shape_file = argv[++i];
      std::string shape_error;
      if (!LoadShapeExpressionFromFile(shape_file, &shape_str, &shape_error)) {
        std::cerr << "shape file error: " << shape_error << "\n";
        return 1;
      }
    } else if (arg == "--shape-mask" && i + 1 < argc) {
      shape_mask_path = argv[++i];
      std::string mask_error;
      if (!LoadShapeMaskFromVtk(shape_mask_path, &shape_mask, &mask_error)) {
        std::cerr << "shape mask error: " << mask_error << "\n";
        return 1;
      }
    } else if (arg == "--shape-mask-threshold" && i + 1 < argc) {
      shape_mask_threshold = std::strtod(argv[++i], nullptr);
    } else if (arg == "--shape-mask-invert") {
      shape_mask_invert = true;
    } else if (arg == "--shape-offset" && i + 1 < argc) {
      std::string parse_error;
      if (!ParseShapeVector(argv[++i], &shape_transform.offset_x,
                            &shape_transform.offset_y, &shape_transform.offset_z,
                            &parse_error)) {
        std::cerr << "invalid --shape-offset: " << parse_error << "\n";
        return 1;
      }
    } else if (arg == "--shape-scale" && i + 1 < argc) {
      std::string parse_error;
      if (!ParseShapeVector(argv[++i], &shape_transform.scale_x,
                            &shape_transform.scale_y, &shape_transform.scale_z,
                            &parse_error)) {
        std::cerr << "invalid --shape-scale: " << parse_error << "\n";
        return 1;
      }
    } else if (arg == "--mesh" && i + 1 < argc) {
      mesh_path = argv[++i];
    } else if (arg == "--mesh-format" && i + 1 < argc) {
      mesh_format = argv[++i];
    } else if (arg == "--mesh-discretization" && i + 1 < argc) {
      mesh_discretization = argv[++i];
    } else if (arg == "--mesh-solve") {
      mesh_solve = true;
    } else if (arg == "--out" && i + 1 < argc) {
      out_path = argv[++i];
    } else if (arg == "--out-dir" && i + 1 < argc) {
      out_dir = argv[++i];
      out_dir_arg = true;
    } else if (arg == "--in-dir" && i + 1 < argc) {
      in_dir = argv[++i];
    } else if (arg == "--method" && i + 1 < argc) {
      std::string token = argv[++i];
      if (!IsMethodToken(token)) {
        std::cerr << "unknown method: " << token << "\n";
        return 1;
      }
      method = ParseSolveMethodToken(token);
    } else if (arg == "--omega" && i + 1 < argc) {
      solver.sor_omega = std::strtod(argv[++i], nullptr);
    } else if (arg == "--gmres-restart" && i + 1 < argc) {
      solver.gmres_restart = std::atoi(argv[++i]);
    } else if (arg == "--residual-interval" && i + 1 < argc) {
      solver.residual_interval = std::atoi(argv[++i]);
    } else if (arg == "--max-iter" && i + 1 < argc) {
      solver.max_iter = std::atoi(argv[++i]);
    } else if (arg == "--tol" && i + 1 < argc) {
      solver.tol = std::strtod(argv[++i], nullptr);
    } else if (arg == "--threads" && i + 1 < argc) {
      solver.thread_count = std::atoi(argv[++i]);
    } else if (arg == "--metal-reduce-interval" && i + 1 < argc) {
      solver.metal_reduce_interval = std::atoi(argv[++i]);
    } else if (arg == "--metal-threadgroup" && i + 1 < argc) {
      Domain tg_domain;
      ParseResult tg_result = ParseGrid(argv[++i], &tg_domain);
      if (!tg_result.ok) {
        std::cerr << "invalid metal threadgroup: " << tg_result.error << "\n";
        return 1;
      }
      solver.metal_tg_x = tg_domain.nx;
      solver.metal_tg_y = tg_domain.ny;
    } else if (arg == "--backend" && i + 1 < argc) {
      std::string token = argv[++i];
      if (!IsBackendToken(token)) {
        std::cerr << "unknown backend: " << token << "\n";
        return 1;
      }
      backend = ParseBackendKind(token);
    } else if (arg == "--time" && i + 1 < argc) {
      std::string time_str = argv[++i];
      std::vector<double> values;
      std::istringstream iss(time_str);
      std::string token;
      while (std::getline(iss, token, ',')) {
        values.push_back(std::strtod(token.c_str(), nullptr));
      }
      if (values.size() >= 4) {
        time_config.enabled = true;
        time_config.t_start = values[0];
        time_config.t_end = values[1];
        time_config.dt = values[2];
        time_config.frames = static_cast<int>(values[3]);
      } else {
        std::cerr << "invalid time spec: need t_start,t_end,dt,frames\n";
        return 1;
      }
    } else if (arg == "--buffer-mb" && i + 1 < argc) {
      time_config.buffer_mb = std::atoi(argv[++i]);
      if (time_config.buffer_mb < 1) {
        std::cerr << "buffer-mb must be at least 1\n";
        return 1;
      }
    } else if (arg == "--checkpoint" && i + 1 < argc) {
      checkpoint_path = argv[++i];
    } else if (arg == "--restart" && i + 1 < argc) {
      restart_path = argv[++i];
    } else if (arg == "--format" && i + 1 < argc) {
      output_format = argv[++i];
      output_format = pde::ToLower(output_format);
      if (output_format != "vtk" && output_format != "vti") {
        std::cerr << "invalid format: must be 'vtk' or 'vti'\n";
        return 1;
      }
    } else {
      std::cerr << "unknown or incomplete argument: " << arg << "\n";
      return 1;
    }
  }

  if (!dataset_cleanup_dir.empty() || !dataset_index_dir.empty()) {
    bool ok = true;
    if (!dataset_cleanup_dir.empty()) {
      DatasetCleanupResult cleanup;
      std::string cleanup_error;
      if (!CleanupDataset(std::filesystem::path(dataset_cleanup_dir),
                          dataset_cleanup_dry_run, dataset_cleanup_empty_dirs,
                          &cleanup, &cleanup_error)) {
        std::cerr << "dataset cleanup error: " << cleanup_error << "\n";
        ok = false;
      } else {
        const char* mode = dataset_cleanup_dry_run ? "dry-run" : "cleanup";
        std::cout << "dataset " << mode << ": removed "
                  << cleanup.removed_summaries << " summaries, "
                  << cleanup.removed_metadata << " metadata";
        if (dataset_cleanup_empty_dirs) {
          std::cout << ", " << cleanup.removed_empty_dirs << " empty dirs";
        }
        if (cleanup.skipped > 0) {
          std::cout << " (" << cleanup.skipped << " skipped)";
        }
        std::cout << "\n";
      }
    }

    if (!dataset_index_dir.empty()) {
      DatasetIndexResult index;
      std::string index_error;
      if (!BuildDatasetIndex(std::filesystem::path(dataset_index_dir), &index, &index_error)) {
        std::cerr << "dataset index error: " << index_error << "\n";
        ok = false;
      } else {
        const bool index_to_stdout = (dataset_index_out == "-");
        if (index_to_stdout) {
          std::cout << index.json << "\n";
        } else {
          std::filesystem::path out_path = dataset_index_out.empty()
                                               ? (std::filesystem::path(dataset_index_dir) /
                                                  "dataset_index.json")
                                               : std::filesystem::path(dataset_index_out);
          std::string write_error;
          if (!WriteDatasetIndex(out_path, index, &write_error)) {
            std::cerr << "dataset index write error: " << write_error << "\n";
            ok = false;
          } else {
            std::cout << "wrote dataset index: " << out_path << "\n";
          }
        }
        std::ostream& stats_out = index_to_stdout ? std::cerr : std::cout;
        stats_out << "dataset stats: completed " << index.runs_completed
                  << " / " << index.runs_total << " runs, "
                  << index.runs_time_series << " time-series, "
                  << index.runs_steady << " steady, "
                  << index.total_frames << " frames";
        if (index.missing_outputs > 0) {
          stats_out << ", " << index.missing_outputs << " missing outputs";
        }
        if (index.monitor_warning_runs > 0) {
          stats_out << ", " << index.monitor_warning_runs << " monitor warnings";
        }
        stats_out << "\n";
      }
    }
    return ok ? 0 : 1;
  }

  if (latex.empty() && in_dir.empty() && !validate_only && (dump_metadata || dump_summary)) {
    bool handled = false;
    bool ok = true;
    if (dump_metadata) {
      handled = true;
      if (dump_metadata_path.empty()) {
        std::cerr << "--dump-metadata requires a path or an active solve\n";
        ok = false;
      } else {
        const std::filesystem::path meta_path = dump_metadata_path;
        if (std::filesystem::exists(meta_path) && std::filesystem::is_directory(meta_path)) {
          ok = DumpMetadataDirectory(meta_path) && ok;
        } else {
          ok = DumpMetadataFile(meta_path) && ok;
        }
      }
    }
    if (dump_summary) {
      handled = true;
      if (dump_summary_path.empty()) {
        std::cerr << "--dump-summary requires a path or an active solve\n";
        ok = false;
      } else {
        const std::filesystem::path summary_path = dump_summary_path;
        if (std::filesystem::exists(summary_path) && std::filesystem::is_directory(summary_path)) {
          ok = DumpSummaryDirectory(summary_path) && ok;
        } else {
          ok = DumpSummaryFile(summary_path) && ok;
        }
      }
    }
    if (handled) {
      return ok ? 0 : 1;
    }
  }

  if (!in_dir.empty() && latex.empty() && !validate_only) {
    if (dump_metadata || dump_summary) {
      bool ok = true;
      if (dump_metadata) {
        ok = DumpMetadataDirectory(std::filesystem::path(in_dir)) && ok;
      }
      if (dump_summary) {
        ok = DumpSummaryDirectory(std::filesystem::path(in_dir)) && ok;
      }
      return ok ? 0 : 1;
    }
    return LoadVtkDirectory(std::filesystem::path(in_dir)) ? 0 : 1;
  }

  const bool mesh_mode = !mesh_path.empty();
  if (mesh_mode) {
    if (!convergence_spec.empty() || use_mms) {
      std::cerr << "mesh mode does not support MMS or convergence sweeps\n";
      return 1;
    }
    if (dump_metadata || dump_summary) {
      std::cerr << "warning: --dump-metadata/--dump-summary ignored in mesh mode\n";
    }
    if (dump_operator) {
      std::cerr << "warning: --dump-operator ignored in mesh mode\n";
    }
    if (!domain_str.empty() || !grid_str.empty()) {
      std::cerr << "warning: --domain/--grid ignored in mesh mode\n";
    }
    if (!shape_str.empty() || !shape_mask_path.empty() || !shape_file.empty()) {
      std::cerr << "warning: --shape options ignored in mesh mode\n";
    }
    return RunUnstructuredMeshWorkflow(mesh_path, mesh_format, mesh_discretization,
                                       mesh_solve, validate_only, latex,
                                       out_path, out_dir, output_format);
  }

  if (!validate_only) {
    if (latex.empty() || domain_str.empty() || grid_str.empty()) {
      std::cerr << "missing required arguments\n";
      PrintUsage();
      return 1;
    }
    if (!in_dir.empty()) {
      std::cerr << "warning: --in-dir ignored while solving\n";
    }
  } else if (latex.empty()) {
    std::cerr << "missing required arguments: --pde\n";
    return 1;
  }

  LatexParser parser;
  LatexParseResult parse_result = parser.Parse(latex);
  if (!parse_result.ok) {
    std::cerr << "latex parse error: " << parse_result.error << "\n";
    return 1;
  }
  CoefficientEvaluator coeff_eval = BuildCoefficientEvaluator(parse_result.coeffs);
  if (!coeff_eval.ok) {
    std::cerr << "invalid coefficient expression: " << coeff_eval.error << "\n";
    return 1;
  }

  Domain domain;
  if (!domain_str.empty() || !grid_str.empty()) {
    if (domain_str.empty() || grid_str.empty()) {
      std::cerr << "validation requires both --domain and --grid if either is provided\n";
      return 1;
    }
    ParseResult domain_result = ParseDomain(domain_str, &domain);
    if (!domain_result.ok) {
      std::cerr << domain_result.error << "\n";
      return 1;
    }
    ParseResult grid_result = ParseGrid(grid_str, &domain);
    if (!grid_result.ok) {
      std::cerr << grid_result.error << "\n";
      return 1;
    }
  }

  BoundarySet bc;
  if (!bc_str.empty()) {
    ParseResult bc_result = ApplyBoundarySpec(bc_str, &bc);
    if (!bc_result.ok) {
      std::cerr << "invalid boundary spec: " << bc_result.error << "\n";
      return 1;
    }
    if (!domain_str.empty()) {
      ParseResult bc_validate = ValidateBoundaryConditions(bc, domain);
      if (!bc_validate.ok) {
        std::cerr << "boundary validation failed: " << bc_validate.error << "\n";
        return 1;
      }
    }
  }

  std::vector<Domain> convergence_domains;
  if (!convergence_spec.empty()) {
    std::string conv_error;
    if (!ParseConvergenceSpec(convergence_spec, domain, &convergence_domains, &conv_error)) {
      std::cerr << "invalid convergence spec: " << conv_error << "\n";
      return 1;
    }
    if (convergence_domains.size() < 2) {
      std::cerr << "warning: convergence spec includes fewer than 2 grids\n";
    }
  }

  RunConfig run_config;
  bool run_config_ready = false;
  if (!latex.empty() && !domain_str.empty() && !grid_str.empty()) {
    run_config =
        BuildRunConfigForCli(latex, domain_str, grid_str, bc_str, shape_str,
                             shape_file, shape_mask_path,
                             shape_mask_threshold, shape_mask_invert, shape_transform,
                             domain, backend, method, solver, time_config,
                             out_path, out_dir, output_format);
    run_config_ready = true;
  }

  if (!export_config_path.empty()) {
    RunConfig export_config;
    if (!mesh_path.empty()) {
      export_config = BuildRunConfigForMesh(mesh_path, mesh_format, mesh_discretization,
                                            latex, out_path, out_dir, output_format);
    } else {
      if (latex.empty() || domain_str.empty() || grid_str.empty()) {
        std::cerr << "export config requires --pde, --domain, and --grid\n";
        return 1;
      }
      export_config = run_config_ready
                          ? run_config
                          : BuildRunConfigForCli(latex, domain_str, grid_str,
                                                 bc_str, shape_str, shape_file,
                                                 shape_mask_path, shape_mask_threshold,
                                                 shape_mask_invert, shape_transform,
                                                 domain, backend, method, solver,
                                                 time_config, out_path, out_dir,
                                                 output_format);
    }

    std::string export_error;
    if (!SaveRunConfigToFile(export_config_path, export_config, &export_error)) {
      std::cerr << "failed to export config: " << export_error << "\n";
      return 1;
    }
  }

  std::string out_path_resolved = out_path;
  std::string out_dir_resolved = out_dir;
  const bool out_path_pattern = ContainsPatternToken(out_path);
  const bool out_dir_pattern = ContainsPatternToken(out_dir);
  if (out_path_pattern || out_dir_pattern) {
    OutputPatternContext ctx;
    ctx.index = 1;
    ctx.index0 = 0;
    ctx.name = "run_1";
    ctx.backend = pde::ToLower(BackendToken(backend));
    ctx.method = pde::ToLower(MethodToken(method));
    ctx.format = output_format;
    ctx.domain = domain;
    ctx.tag = GenerateRandomTag(6);
    ctx.timestamp = FormatTimestamp();
    out_path_resolved = ApplyOutputPattern(out_path, ctx);
    out_dir_resolved = ApplyOutputPattern(out_dir, ctx);
  }

  if (dump_operator) {
    if (dump_operator_json) {
      PrintOperatorDumpJson(parse_result.op);
    } else {
      PrintOperatorDumpText(parse_result.op);
    }
  }

  if (validate_only) {
    PrintParseSummary(parse_result);
    if (!domain_str.empty()) {
      std::cout << "Domain: [" << domain.xmin << "," << domain.xmax << "]"
                << " x [" << domain.ymin << "," << domain.ymax << "]";
      if (domain.nz > 1) {
        std::cout << " x [" << domain.zmin << "," << domain.zmax << "]";
      }
      std::cout << " grid=" << domain.nx << "x" << domain.ny;
      if (domain.nz > 1) {
        std::cout << "x" << domain.nz;
      }
      std::cout << "\n";
    }
    if (!shape_str.empty()) {
      std::cout << "Domain shape: " << shape_str << "\n";
    }
    if (!shape_file.empty()) {
      std::cout << "Shape file: " << shape_file << "\n";
    }
    if (!shape_mask_path.empty()) {
      std::cout << "Shape mask: " << shape_mask_path << " (threshold="
                << shape_mask_threshold << ", invert="
                << (shape_mask_invert ? "true" : "false") << ")\n";
    }
    if (!shape_str.empty() || !shape_mask_path.empty()) {
      const bool has_offset =
          std::abs(shape_transform.offset_x) > 1e-12 ||
          std::abs(shape_transform.offset_y) > 1e-12 ||
          std::abs(shape_transform.offset_z) > 1e-12;
      const bool has_scale =
          std::abs(shape_transform.scale_x - 1.0) > 1e-12 ||
          std::abs(shape_transform.scale_y - 1.0) > 1e-12 ||
          std::abs(shape_transform.scale_z - 1.0) > 1e-12;
      if (has_offset || has_scale) {
        std::cout << "Shape transform: offset=(" << shape_transform.offset_x << ","
                  << shape_transform.offset_y << "," << shape_transform.offset_z
                  << ") scale=(" << shape_transform.scale_x << ","
                  << shape_transform.scale_y << "," << shape_transform.scale_z << ")\n";
      }
    }
    if (!bc_str.empty() && !domain_str.empty()) {
      std::cout << "Boundary conditions: ok\n";
    }
    return 0;
  }

  SolveInput input;
  input.pde = parse_result.coeffs;
  input.integrals = parse_result.integrals;
  input.nonlinear = parse_result.nonlinear;
  input.nonlinear_derivatives = parse_result.nonlinear_derivatives;
  input.domain = domain;
  input.bc = bc;
  input.domain_shape = shape_str;
  input.shape_transform = shape_transform;
  input.shape_mask = shape_mask;
  input.shape_mask_threshold = shape_mask_threshold;
  input.shape_mask_invert = shape_mask_invert;
  input.time = time_config;
  solver.method = method;
  input.solver = solver;

  if (!convergence_spec.empty() && !restart_path.empty()) {
    std::cerr << "--convergence cannot be combined with --restart\n";
    return 1;
  }
  if (!convergence_spec.empty() && time_config.enabled) {
    std::cerr << "--convergence only supports steady-state solves\n";
    return 1;
  }
  if (use_mms && !restart_path.empty()) {
    std::cerr << "--mms cannot be combined with --restart\n";
    return 1;
  }
  if (use_mms && time_config.enabled) {
    std::cerr << "--mms only supports steady-state solves\n";
    return 1;
  }

  ManufacturedSolution mms_solution;
  bool mms_active = false;
  if (use_mms) {
    std::string mms_error;
    if (!ValidateMmsInput(input, &mms_error)) {
      std::cerr << "MMS validation failed: " << mms_error << "\n";
      return 1;
    }
    const int dim = (input.domain.nz > 1) ? 3 : 2;
    mms_solution = BuildDefaultManufacturedSolution(dim);
    ManufacturedRhsResult rhs = BuildManufacturedRhs(input.pde, dim, mms_solution);
    if (!rhs.ok) {
      std::cerr << "MMS RHS build failed: " << rhs.error << "\n";
      return 1;
    }
    input.pde.rhs_latex = rhs.rhs_latex;
    input.pde.f = 0.0;
    ApplyMmsDirichlet(&input.bc, mms_solution.u_latex);
    mms_active = true;
    std::cout << "MMS: exact u = " << mms_solution.u_latex << "\n";
    std::cout << "MMS: rhs_latex = " << input.pde.rhs_latex << "\n";
    std::cout << "MMS: Dirichlet boundaries overridden to exact solution\n";
    if (mms_forced) {
      std::cout << "MMS: enabled automatically for convergence study\n";
    }
  }

  if (!convergence_spec.empty()) {
    std::filesystem::path csv_path;
    if (!convergence_out_path.empty()) {
      csv_path = convergence_out_path;
    } else {
      std::filesystem::path dir = out_dir_resolved.empty() ? "." : out_dir_resolved;
      csv_path = dir / "convergence.csv";
    }
    std::filesystem::path plot_path;
    if (!convergence_plot_path.empty()) {
      plot_path = convergence_plot_path;
    } else {
      plot_path = csv_path;
      plot_path.replace_extension(".gp");
    }
    std::filesystem::path json_path;
    if (!convergence_json_path.empty()) {
      json_path = convergence_json_path;
    } else {
      json_path = csv_path;
      json_path.replace_extension(".json");
    }

    PrintBackendStatus();
    std::string conv_error;
    if (!RunConvergenceStudy(input, backend, convergence_domains, mms_solution,
                             csv_path, plot_path, json_path, &conv_error)) {
      std::cerr << "convergence error: " << conv_error << "\n";
      return 1;
    }
    std::cout << "wrote convergence CSV: " << csv_path << "\n";
    if (!json_path.empty()) {
      std::cout << "wrote convergence JSON: " << json_path << "\n";
    }
    if (!plot_path.empty()) {
      std::cout << "wrote convergence plot script: " << plot_path << "\n";
    }
    return 0;
  }

  // Handle restart from checkpoint
  std::vector<double> initial_grid;
  std::vector<double> initial_velocity;
  int restart_frame = 0;
  if (!restart_path.empty()) {
    CheckpointData checkpoint;
    VtkReadResult read_result = ReadCheckpoint(restart_path, &checkpoint);
    if (!read_result.ok) {
      std::cerr << "failed to read checkpoint: " << read_result.error << "\n";
      return 1;
    }
    // Restore state from checkpoint
    input.domain = checkpoint.domain;
    input.pde = checkpoint.pde;
    input.bc = checkpoint.bc;
    initial_grid = checkpoint.grid;
    initial_velocity = checkpoint.velocity;
    restart_frame = checkpoint.frame_index;
    if (time_config.enabled) {
      // Adjust time range to continue from checkpoint
      time_config.t_start = checkpoint.t_current;
      time_config.frames = std::max(1, time_config.frames - checkpoint.frame_index);
      input.time = time_config;
    }
    if (std::abs(checkpoint.pde.utt) > 1e-12 && checkpoint.velocity.empty()) {
      std::cerr << "warning: checkpoint missing velocity for u_tt; restart may not be reproducible\n";
    }
    std::cout << "restarting from checkpoint at t=" << checkpoint.t_current
              << " frame=" << checkpoint.frame_index << "\n";
  }

  PrintBackendStatus();
  BackendKind selected = BackendKind::CPU;
  std::string selection_note;
  ProgressCallback cb;
  auto solve_start = std::chrono::steady_clock::now();
  double last_l2 = -1.0;
  int sample = 0;
  if (input.solver.residual_interval > 0) {
    cb = [&](const std::string& phase, double value) {
      if (phase == "residual_l2") {
        last_l2 = value;
        return;
      }
      if (phase == "residual_linf") {
        const double t = std::chrono::duration_cast<std::chrono::duration<double>>(
                             std::chrono::steady_clock::now() - solve_start)
                             .count();
        std::cout << "residual[" << sample++ << "] t=" << t
                  << " L2=" << last_l2
                  << " Linf=" << value << "\n";
        return;
      }
    };
  }
  SolveOutput output;
  std::vector<std::string> series_paths;
  std::vector<double> series_times;
  double solve_seconds = 0.0;
  double write_seconds = 0.0;

  if (time_config.enabled) {
    // Time-dependent solve
    BackendKind selected_backend = BackendKind::CPU;
    std::string selection_note_ts;
    bool use_metal_time = false;
#ifdef USE_METAL
    if ((backend == BackendKind::Metal || backend == BackendKind::Auto) && input.domain.nz <= 1) {
      std::string metal_note;
      if (MetalIsAvailable(&metal_note)) {
        use_metal_time = true;
        selected_backend = BackendKind::Metal;
        selection_note_ts = metal_note;
      }
    }
#endif
    if (!use_metal_time && backend != BackendKind::Auto && backend != BackendKind::CPU) {
      selection_note_ts = "time series uses CPU backend (Metal unavailable or 3D)";
    }
    std::error_code ec;
    std::filesystem::path series_base;
    if (!out_path_resolved.empty()) {
      series_base = std::filesystem::path(out_path_resolved);
      if (series_base.has_extension()) {
        series_base.replace_extension("");
      }
    } else {
      std::filesystem::path out_dir_path(out_dir_resolved.empty() ? "." : out_dir_resolved);
      std::filesystem::create_directories(out_dir_path, ec);
      if (ec) {
        std::cerr << "failed to create output directory: " << ec.message() << "\n";
        return 1;
      }
      series_base = out_dir_path / ("pde_" + GenerateRandomTag(6));
    }

    const int digits = std::max(4, static_cast<int>(std::log10(time_config.frames) + 1));
    if (out_path_pattern || out_path.empty()) {
      series_base = EnsureUniqueSeriesBase(series_base, digits, output_format);
    }
    auto BuildFramePath = [&](int frame) {
      return BuildSeriesFramePath(series_base, frame, digits, output_format);
    };

    int current_frame = restart_frame;
    std::string frame_error;
    ConservedMonitor monitor;
    auto frame_cb = [&](int frame, double t, const std::vector<double>& grid,
                        const std::vector<double>* velocity) -> bool {
      const ConservedSample sample = UpdateConservedMonitor(input.domain, frame, grid, &monitor);
      if (sample.ok) {
        if (monitor.mass_warning && monitor.mass_warning_frame == frame) {
          std::cerr << "warning: mass drift exceeds 1% at frame " << frame << "\n";
        }
        if (monitor.energy_warning && monitor.energy_warning_frame == frame) {
          std::cerr << "warning: energy drift exceeds 1% at frame " << frame << "\n";
        }
        if (monitor.blowup_warning && monitor.blowup_warning_frame == frame) {
          std::cerr << "warning: rapid growth detected at frame " << frame
                    << " (ratio " << monitor.blowup_ratio
                    << ", ||u||inf " << monitor.blowup_max << ")\n";
        }
      }
      const std::filesystem::path frame_path = BuildFramePath(frame);

      std::vector<std::vector<double>> derived_field_data;
      std::vector<std::string> derived_field_names;
      BuildDerivedFieldVectors(input.domain, grid, input.pde.a, input.pde.b, input.pde.az,
                               &derived_field_data, &derived_field_names);

      // Write frame
      VtkWriteResult write_result;
      if (output_format == "vti") {
        write_result = WriteVtkXmlImageData(frame_path.string(), input.domain, grid,
                                            &derived_field_data, &derived_field_names);
      } else {
        write_result = WriteVtkStructuredPoints(frame_path.string(), input.domain, grid);
      }
      if (!write_result.ok) {
        frame_error = write_result.error;
        return false;
      }
      if (run_config_ready) {
        std::string meta_error;
        if (!WriteRunMetadataSidecar(frame_path, run_config, backend, selected_backend,
                                     selection_note_ts, true, frame, t, &meta_error)) {
          std::cerr << "metadata warning: " << meta_error << "\n";
        }
        RunSummaryData summary;
        summary.run_config = run_config;
        summary.requested_backend = backend;
        summary.selected_backend = selected_backend;
        summary.backend_note = selection_note_ts;
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
          std::cerr << "summary warning: " << summary_error << "\n";
        }
      }
      series_paths.push_back(frame_path.string());
      series_times.push_back(t);

      // Write checkpoint if requested
      if (!checkpoint_path.empty()) {
        CheckpointData checkpoint;
        checkpoint.domain = input.domain;
        checkpoint.grid = grid;
        if (velocity && !velocity->empty()) {
          checkpoint.velocity = *velocity;
        }
        checkpoint.t_current = t;
        checkpoint.frame_index = frame;
        checkpoint.pde = input.pde;
        checkpoint.bc = input.bc;
        VtkWriteResult cp_result = WriteCheckpoint(checkpoint_path, checkpoint);
        if (!cp_result.ok) {
          std::cerr << "warning: checkpoint write failed: " << cp_result.error << "\n";
        }
      }

      std::cout << "wrote frame " << frame << " (t=" << t << ") to " << frame_path << "\n";
      return true;
    };

    // Initialize grid from checkpoint if restarting
    if (!initial_grid.empty()) {
      input.initial_grid = std::move(initial_grid);
      input.initial_velocity = std::move(initial_velocity);
    }

#ifdef USE_METAL
    if (use_metal_time) {
      output = SolvePDETimeSeriesMetal(input, frame_cb, cb);
    } else {
      output = SolvePDETimeSeries(input, frame_cb, cb);
    }
#else
    output = SolvePDETimeSeries(input, frame_cb, cb);
#endif
    solve_seconds = std::chrono::duration_cast<std::chrono::duration<double>>(
                        std::chrono::steady_clock::now() - solve_start)
                        .count();
    if (!output.error.empty()) {
      std::cerr << "solve error: " << output.error << "\n";
      return 1;
    }
    if (!frame_error.empty()) {
      std::cerr << "frame write error: " << frame_error << "\n";
      return 1;
    }

    // Write series manifest
    if (!series_paths.empty()) {
      const std::filesystem::path manifest = series_base.parent_path() /
                                              (series_base.stem().string() + "_series.pvd");
      WriteVtkSeriesPvd(manifest.string(), series_paths, series_times);
      std::cout << "wrote series manifest: " << manifest << "\n";
      if (run_config_ready) {
        std::string meta_error;
        if (!WriteRunMetadataSidecar(manifest, run_config, backend, selected_backend,
                                     selection_note_ts, true, -1, 0.0, &meta_error)) {
          std::cerr << "metadata warning: " << meta_error << "\n";
        }
        RunSummaryData summary;
        summary.run_config = run_config;
        summary.requested_backend = backend;
        summary.selected_backend = selected_backend;
        summary.backend_note = selection_note_ts;
        summary.output_path = manifest.string();
        summary.time_series = true;
        summary.solve_seconds = solve_seconds;
        summary.total_seconds = solve_seconds;
        summary.frame_times = series_times;
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
          std::cerr << "summary warning: " << summary_error << "\n";
        }
        if (dump_metadata) {
          const std::string metadata_json =
              BuildRunMetadataJson(run_config, backend, selected_backend, selection_note_ts,
                                   manifest.string(), true, -1, 0.0);
          if (!dump_metadata_path.empty()) {
            std::string dump_error;
            if (!WriteTextFile(std::filesystem::path(dump_metadata_path), metadata_json,
                               &dump_error)) {
              std::cerr << "failed to write metadata: " << dump_error << "\n";
              return 1;
            }
          } else {
            std::cout << metadata_json << "\n";
          }
        }
        if (dump_summary) {
          const std::string summary_json = BuildRunSummaryJson(summary, 2);
          if (!dump_summary_path.empty()) {
            std::string dump_error;
            if (!WriteTextFile(std::filesystem::path(dump_summary_path), summary_json,
                               &dump_error)) {
              std::cerr << "failed to write summary: " << dump_error << "\n";
              return 1;
            }
          } else {
            std::cout << summary_json << "\n";
          }
        }
      }
    }

    std::cout << "wrote " << series_paths.size() << " frames\n";
    return 0;
  }

  // Steady-state solve
  output = SolveWithBackend(input, backend, &selected, &selection_note, cb);
  solve_seconds = std::chrono::duration_cast<std::chrono::duration<double>>(
                      std::chrono::steady_clock::now() - solve_start)
                      .count();
  std::cout << "Selected backend: " << BackendKindName(selected);
  if (!selection_note.empty()) {
    std::cout << " (" << selection_note << ")";
  }
  std::cout << "\n";
  if (!output.error.empty()) {
    std::cerr << "solve error: " << output.error << "\n";
    return 1;
  }
  std::cout << "Residual ||Au-b||_2 = " << output.residual_l2
            << ", ||Au-b||_inf = " << output.residual_linf << "\n";
  if (mms_active) {
    ErrorNorms err;
    if (mms_solution.dimension > 2) {
      err = ComputeErrorNorms3D(input.domain, output.grid,
                                [&](double x, double y, double z) {
                                  return mms_solution.eval(x, y, z);
                                });
    } else {
      err = ComputeErrorNorms2D(input.domain, output.grid,
                                [&](double x, double y) {
                                  return mms_solution.eval(x, y, 0.0);
                                });
    }
    std::cout << "MMS error: L1=" << err.l1
              << " L2=" << err.l2
              << " Linf=" << err.linf << "\n";
  }

  std::filesystem::path final_path;
  std::error_code ec;
  if (!out_path_resolved.empty()) {
    std::filesystem::path candidate(out_path_resolved);
    if (std::filesystem::exists(candidate) && std::filesystem::is_directory(candidate)) {
      out_dir_resolved = candidate.string();
    } else {
      if (!candidate.has_extension()) {
        candidate += (output_format == "vti") ? ".vti" : ".vtk";
      } else {
        const std::string ext = candidate.extension().string();
        if (ext != ".vtk" && ext != ".vti") {
          candidate.replace_extension(output_format == "vti" ? ".vti" : ".vtk");
          std::cerr << "warning: output extension adjusted to ." << output_format << "\n";
        }
      }
      if (candidate.has_parent_path()) {
        std::filesystem::create_directories(candidate.parent_path(), ec);
        if (ec) {
          std::cerr << "failed to create output directory: " << ec.message() << "\n";
          return 1;
        }
      }
      final_path = candidate;
    }
  }

  if (final_path.empty()) {
    std::filesystem::path out_dir_path(out_dir_resolved.empty() ? "." : out_dir_resolved);
    std::filesystem::create_directories(out_dir_path, ec);
    if (ec) {
      std::cerr << "failed to create output directory: " << ec.message() << "\n";
      return 1;
    }
    final_path = out_dir_path / ("pde_" + GenerateRandomTag(6) + 
                                 (output_format == "vti" ? ".vti" : ".vtk"));
  }

  if (out_path_pattern || out_path.empty()) {
    final_path = EnsureUniquePath(final_path);
  }

  auto write_start = std::chrono::steady_clock::now();
  std::vector<std::vector<double>> derived_field_data;
  std::vector<std::string> derived_field_names;
  BuildDerivedFieldVectors(domain, output.grid, input.pde.a, input.pde.b, input.pde.az,
                           &derived_field_data, &derived_field_names);

  // Write VTK XML format if .vti extension, otherwise legacy VTK
  std::filesystem::path vtk_path = final_path;
  if (vtk_path.extension() == ".vti" || output_format == "vti") {
    VtkWriteResult write_result = WriteVtkXmlImageData(
        vtk_path.string(), domain, output.grid, &derived_field_data, &derived_field_names);
    if (!write_result.ok) {
      std::cerr << "vtk xml write error: " << write_result.error << "\n";
      return 1;
    }
  } else {
    VtkWriteResult write_result =
        WriteVtkStructuredPoints(final_path.string(), domain, output.grid);
    if (!write_result.ok) {
      std::cerr << "vtk write error: " << write_result.error << "\n";
      return 1;
    }
  }
  write_seconds = std::chrono::duration_cast<std::chrono::duration<double>>(
                      std::chrono::steady_clock::now() - write_start)
                      .count();

  if (run_config_ready) {
    std::string meta_error;
    if (!WriteRunMetadataSidecar(final_path, run_config, backend, selected, selection_note,
                                 false, -1, 0.0, &meta_error)) {
      std::cerr << "metadata warning: " << meta_error << "\n";
    }
    RunSummaryData summary;
    summary.run_config = run_config;
    summary.requested_backend = backend;
    summary.selected_backend = selected;
    summary.backend_note = selection_note;
    summary.output_path = final_path.string();
    summary.time_series = false;
    summary.solve_seconds = solve_seconds;
    summary.write_seconds = write_seconds;
    summary.total_seconds = solve_seconds + write_seconds;
    summary.residual_l2 = output.residual_l2;
    summary.residual_linf = output.residual_linf;
    summary.residual_iters = output.residual_iters;
    summary.residual_l2_history = output.residual_l2_history;
    summary.residual_linf_history = output.residual_linf_history;
    std::string summary_error;
    if (!WriteRunSummarySidecar(final_path, summary, &summary_error)) {
      std::cerr << "summary warning: " << summary_error << "\n";
    }
    if (dump_metadata) {
      const std::string metadata_json =
          BuildRunMetadataJson(run_config, backend, selected, selection_note,
                               final_path.string(), false, -1, 0.0);
      if (!dump_metadata_path.empty()) {
        std::string dump_error;
        if (!WriteTextFile(std::filesystem::path(dump_metadata_path), metadata_json, &dump_error)) {
          std::cerr << "failed to write metadata: " << dump_error << "\n";
          return 1;
        }
      } else {
        std::cout << metadata_json << "\n";
      }
    }
    if (dump_summary) {
      const std::string summary_json = BuildRunSummaryJson(summary, 2);
      if (!dump_summary_path.empty()) {
        std::string dump_error;
        if (!WriteTextFile(std::filesystem::path(dump_summary_path), summary_json, &dump_error)) {
          std::cerr << "failed to write summary: " << dump_error << "\n";
          return 1;
        }
      } else {
        std::cout << summary_json << "\n";
      }
    }
  }

  std::cout << "wrote " << final_path << "\n";
  return 0;
}
