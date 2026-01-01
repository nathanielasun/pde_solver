#include "run_config.h"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <sstream>

#include <nlohmann/json.hpp>

#include "solver_tokens.h"

namespace {
using json = nlohmann::json;

std::string ToLowerCopy(const std::string& text) {
  std::string out = text;
  std::transform(out.begin(), out.end(), out.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  return out;
}

bool IsMeshFormatToken(const std::string& token) {
  return token == "vtk" || token == "msh";
}

bool IsMeshDiscretizationToken(const std::string& token) {
  return token == "fe" || token == "fv";
}

bool ReadFile(const std::filesystem::path& path, std::string* out, std::string* error) {
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

bool WriteFile(const std::filesystem::path& path, const std::string& data, std::string* error) {
  std::ofstream out(path);
  if (!out.is_open()) {
    if (error) {
      *error = "failed to write file: " + path.string();
    }
    return false;
  }
  out << data;
  if (!out.good()) {
    if (error) {
      *error = "failed to write file: " + path.string();
    }
    return false;
  }
  return true;
}

bool ReadStringField(const json& j, const char* key, std::string* out, std::string* error) {
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

bool ReadIntField(const json& j, const char* key, int* out, std::string* error) {
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

bool ReadDoubleField(const json& j, const char* key, double* out, std::string* error) {
  if (!j.contains(key)) {
    return true;
  }
  const json& value = j.at(key);
  if (!value.is_number()) {
    if (error) {
      *error = std::string("expected number for '") + key + "'";
    }
    return false;
  }
  if (out) {
    *out = value.get<double>();
  }
  return true;
}

bool ReadBoolField(const json& j, const char* key, bool* out, std::string* error) {
  if (!j.contains(key)) {
    return true;
  }
  const json& value = j.at(key);
  if (!value.is_boolean()) {
    if (error) {
      *error = std::string("expected boolean for '") + key + "'";
    }
    return false;
  }
  if (out) {
    *out = value.get<bool>();
  }
  return true;
}

bool ReadThreadgroup(const json& j, int* tg_x, int* tg_y, std::string* error) {
  if (!j.contains("metal_threadgroup")) {
    return true;
  }
  const json& value = j.at("metal_threadgroup");
  if (!value.is_array() || value.size() != 2 ||
      !value[0].is_number_integer() || !value[1].is_number_integer()) {
    if (error) {
      *error = "expected metal_threadgroup as [x,y]";
    }
    return false;
  }
  if (tg_x) {
    *tg_x = value[0].get<int>();
  }
  if (tg_y) {
    *tg_y = value[1].get<int>();
  }
  return true;
}

bool ReadShapeVectorField(const json& j, const char* key,
                          double* x, double* y, double* z,
                          std::string* error) {
  if (!j.contains(key)) {
    return true;
  }
  const json& value = j.at(key);
  if (!value.is_array() || value.size() < 2 || value.size() > 3 ||
      !value[0].is_number() || !value[1].is_number() ||
      (value.size() == 3 && !value[2].is_number())) {
    if (error) {
      *error = std::string("expected ") + key + " as [x,y] or [x,y,z]";
    }
    return false;
  }
  if (x) {
    *x = value[0].get<double>();
  }
  if (y) {
    *y = value[1].get<double>();
  }
  if (value.size() == 3 && z) {
    *z = value[2].get<double>();
  }
  return true;
}

json SolverToJson(const RunConfig& config) {
  json solver;
  solver["method"] = config.method;
  solver["max_iter"] = config.solver.max_iter;
  solver["tol"] = config.solver.tol;
  solver["threads"] = config.solver.thread_count;
  solver["residual_interval"] = config.solver.residual_interval;
  solver["sor_omega"] = config.solver.sor_omega;
  solver["gmres_restart"] = config.solver.gmres_restart;
  solver["mg_pre_smooth"] = config.solver.mg_pre_smooth;
  solver["mg_post_smooth"] = config.solver.mg_post_smooth;
  solver["mg_coarse_iters"] = config.solver.mg_coarse_iters;
  solver["mg_max_levels"] = config.solver.mg_max_levels;
  solver["metal_reduce_interval"] = config.solver.metal_reduce_interval;
  solver["metal_threadgroup"] = {config.solver.metal_tg_x, config.solver.metal_tg_y};
  return solver;
}

json TimeToJson(const RunConfig& config) {
  json time;
  time["enabled"] = config.time.enabled;
  time["t_start"] = config.time.t_start;
  time["t_end"] = config.time.t_end;
  time["dt"] = config.time.dt;
  time["frames"] = config.time.frames;
  return time;
}

json RunConfigToJson(const RunConfig& config) {
  json root;
  root["schema_version"] = config.schema_version;

  json pde;
  pde["latex"] = config.pde_latex;
  root["pde"] = pde;

  json domain;
  domain["bounds"] = config.domain_bounds;
  domain["grid"] = config.grid;
  domain["shape"] = config.domain_shape;
  domain["shape_file"] = config.domain_shape_file;
  domain["shape_mask"] = config.domain_shape_mask;
  domain["shape_mask_threshold"] = config.domain_shape_mask_threshold;
  domain["shape_mask_invert"] = config.domain_shape_mask_invert;
  domain["shape_offset"] = {config.shape_transform.offset_x,
                            config.shape_transform.offset_y,
                            config.shape_transform.offset_z};
  domain["shape_scale"] = {config.shape_transform.scale_x,
                           config.shape_transform.scale_y,
                           config.shape_transform.scale_z};
  domain["mesh"] = config.domain_mesh;
  domain["mesh_format"] = config.domain_mesh_format;
  domain["mesh_discretization"] = config.domain_mesh_discretization;
  domain["coord_mode"] = config.coord_mode;
  domain["mode"] = config.domain_mode;
  root["domain"] = domain;

  json boundary;
  boundary["spec"] = config.boundary_spec;
  root["boundary"] = boundary;

  root["backend"] = config.backend;
  root["solver"] = SolverToJson(config);
  root["time"] = TimeToJson(config);

  json output;
  output["path"] = config.output_path;
  output["dir"] = config.output_dir;
  output["format"] = config.output_format;
  root["output"] = output;
  return root;
}

bool ParseSolver(const json& j, RunConfig* config, std::string* error) {
  if (!j.is_object()) {
    if (error) {
      *error = "solver must be an object";
    }
    return false;
  }
  if (!ReadStringField(j, "method", &config->method, error)) return false;
  if (!ReadIntField(j, "max_iter", &config->solver.max_iter, error)) return false;
  if (!ReadDoubleField(j, "tol", &config->solver.tol, error)) return false;
  if (!ReadIntField(j, "threads", &config->solver.thread_count, error)) return false;
  if (!ReadIntField(j, "residual_interval", &config->solver.residual_interval, error)) return false;
  if (!ReadDoubleField(j, "sor_omega", &config->solver.sor_omega, error)) return false;
  if (!ReadIntField(j, "gmres_restart", &config->solver.gmres_restart, error)) return false;
  if (!ReadIntField(j, "mg_pre_smooth", &config->solver.mg_pre_smooth, error)) return false;
  if (!ReadIntField(j, "mg_post_smooth", &config->solver.mg_post_smooth, error)) return false;
  if (!ReadIntField(j, "mg_coarse_iters", &config->solver.mg_coarse_iters, error)) return false;
  if (!ReadIntField(j, "mg_max_levels", &config->solver.mg_max_levels, error)) return false;
  if (!ReadIntField(j, "metal_reduce_interval", &config->solver.metal_reduce_interval, error)) return false;
  if (!ReadThreadgroup(j, &config->solver.metal_tg_x, &config->solver.metal_tg_y, error)) return false;
  return true;
}

bool ParseTime(const json& j, RunConfig* config, std::string* error) {
  if (!j.is_object()) {
    if (error) {
      *error = "time must be an object";
    }
    return false;
  }
  if (!ReadBoolField(j, "enabled", &config->time.enabled, error)) return false;
  if (!ReadDoubleField(j, "t_start", &config->time.t_start, error)) return false;
  if (!ReadDoubleField(j, "t_end", &config->time.t_end, error)) return false;
  if (!ReadDoubleField(j, "dt", &config->time.dt, error)) return false;
  if (!ReadIntField(j, "frames", &config->time.frames, error)) return false;
  return true;
}

bool ParseOutput(const json& j, RunConfig* config, std::string* error) {
  if (!j.is_object()) {
    if (error) {
      *error = "output must be an object";
    }
    return false;
  }
  if (!ReadStringField(j, "path", &config->output_path, error)) return false;
  if (!ReadStringField(j, "dir", &config->output_dir, error)) return false;
  if (!ReadStringField(j, "format", &config->output_format, error)) return false;
  return true;
}

bool ParseDomain(const json& j, RunConfig* config, std::string* error) {
  if (!j.is_object()) {
    if (error) {
      *error = "domain must be an object";
    }
    return false;
  }
  if (!ReadStringField(j, "bounds", &config->domain_bounds, error)) return false;
  if (!ReadStringField(j, "grid", &config->grid, error)) return false;
  if (!ReadStringField(j, "shape", &config->domain_shape, error)) return false;
  if (!ReadStringField(j, "shape_file", &config->domain_shape_file, error)) return false;
  if (!ReadStringField(j, "shape_mask", &config->domain_shape_mask, error)) return false;
  if (!ReadDoubleField(j, "shape_mask_threshold", &config->domain_shape_mask_threshold, error)) return false;
  if (!ReadBoolField(j, "shape_mask_invert", &config->domain_shape_mask_invert, error)) return false;
  if (!ReadShapeVectorField(j, "shape_offset",
                            &config->shape_transform.offset_x,
                            &config->shape_transform.offset_y,
                            &config->shape_transform.offset_z, error)) {
    return false;
  }
  if (!ReadShapeVectorField(j, "shape_scale",
                            &config->shape_transform.scale_x,
                            &config->shape_transform.scale_y,
                            &config->shape_transform.scale_z, error)) {
    return false;
  }
  if (!ReadStringField(j, "mesh", &config->domain_mesh, error)) return false;
  if (!ReadStringField(j, "mesh_format", &config->domain_mesh_format, error)) return false;
  if (!ReadStringField(j, "mesh_discretization", &config->domain_mesh_discretization, error)) {
    return false;
  }
  if (!ReadStringField(j, "coord_mode", &config->coord_mode, error)) return false;
  if (!ReadStringField(j, "mode", &config->domain_mode, error)) return false;
  return true;
}

bool ParseBoundary(const json& j, RunConfig* config, std::string* error) {
  if (j.is_string()) {
    config->boundary_spec = j.get<std::string>();
    return true;
  }
  if (!j.is_object()) {
    if (error) {
      *error = "boundary must be a string or object";
    }
    return false;
  }
  return ReadStringField(j, "spec", &config->boundary_spec, error);
}

bool ParsePDE(const json& j, RunConfig* config, std::string* error) {
  if (j.is_string()) {
    config->pde_latex = j.get<std::string>();
    return true;
  }
  if (!j.is_object()) {
    if (error) {
      *error = "pde must be a string or object";
    }
    return false;
  }
  return ReadStringField(j, "latex", &config->pde_latex, error);
}

bool ParseRunConfigJson(const json& root, RunConfig* config, std::string* error) {
  if (!config) {
    if (error) {
      *error = "missing config output";
    }
    return false;
  }
  RunConfig parsed;
  if (!root.is_object()) {
    if (error) {
      *error = "run config must be a JSON object";
    }
    return false;
  }

  if (!ReadIntField(root, "schema_version", &parsed.schema_version, error)) return false;
  if (parsed.schema_version != 1) {
    if (error) {
      *error = "unsupported schema_version";
    }
    return false;
  }

  if (root.contains("pde")) {
    if (!ParsePDE(root.at("pde"), &parsed, error)) return false;
  } else {
    ReadStringField(root, "pde", &parsed.pde_latex, nullptr);
  }

  if (root.contains("domain")) {
    if (!ParseDomain(root.at("domain"), &parsed, error)) return false;
  } else {
    ReadStringField(root, "domain", &parsed.domain_bounds, nullptr);
    ReadStringField(root, "grid", &parsed.grid, nullptr);
    ReadStringField(root, "shape", &parsed.domain_shape, nullptr);
    ReadStringField(root, "shape_file", &parsed.domain_shape_file, nullptr);
    ReadStringField(root, "shape_mask", &parsed.domain_shape_mask, nullptr);
    ReadDoubleField(root, "shape_mask_threshold", &parsed.domain_shape_mask_threshold, nullptr);
    ReadBoolField(root, "shape_mask_invert", &parsed.domain_shape_mask_invert, nullptr);
    ReadShapeVectorField(root, "shape_offset",
                         &parsed.shape_transform.offset_x,
                         &parsed.shape_transform.offset_y,
                         &parsed.shape_transform.offset_z, nullptr);
    ReadShapeVectorField(root, "shape_scale",
                         &parsed.shape_transform.scale_x,
                         &parsed.shape_transform.scale_y,
                         &parsed.shape_transform.scale_z, nullptr);
    ReadStringField(root, "mesh", &parsed.domain_mesh, nullptr);
    ReadStringField(root, "mesh_format", &parsed.domain_mesh_format, nullptr);
    ReadStringField(root, "mesh_discretization", &parsed.domain_mesh_discretization, nullptr);
  }

  if (root.contains("boundary")) {
    if (!ParseBoundary(root.at("boundary"), &parsed, error)) return false;
  } else {
    ReadStringField(root, "bc", &parsed.boundary_spec, nullptr);
  }

  if (root.contains("solver")) {
    if (!ParseSolver(root.at("solver"), &parsed, error)) return false;
  }
  ReadStringField(root, "backend", &parsed.backend, nullptr);
  ReadStringField(root, "method", &parsed.method, nullptr);

  if (root.contains("time")) {
    if (!ParseTime(root.at("time"), &parsed, error)) return false;
  }

  if (root.contains("output")) {
    if (!ParseOutput(root.at("output"), &parsed, error)) return false;
  } else {
    ReadStringField(root, "out", &parsed.output_path, nullptr);
    ReadStringField(root, "out_dir", &parsed.output_dir, nullptr);
    ReadStringField(root, "format", &parsed.output_format, nullptr);
  }

  if (!ValidateRunConfig(parsed, error)) {
    return false;
  }

  *config = std::move(parsed);
  return true;
}

}  // namespace

std::string DefaultBoundarySpec() {
  return "left:dirichlet:0;right:dirichlet:0;bottom:dirichlet:0;top:dirichlet:0;front:dirichlet:0;back:dirichlet:0";
}

bool LoadRunConfigFromString(const std::string& content,
                             RunConfig* config,
                             std::string* error) {
  json root;
  try {
    root = json::parse(content);
  } catch (const json::exception& e) {
    if (error) {
      *error = std::string("invalid JSON: ") + e.what();
    }
    return false;
  }
  return ParseRunConfigJson(root, config, error);
}

bool LoadRunConfigFromFile(const std::filesystem::path& path,
                           RunConfig* config,
                           std::string* error) {
  std::string content;
  if (!ReadFile(path, &content, error)) {
    return false;
  }
  return LoadRunConfigFromString(content, config, error);
}

bool SaveRunConfigToFile(const std::filesystem::path& path,
                         const RunConfig& config,
                         std::string* error) {
  const std::string payload = SerializeRunConfig(config, 2);
  return WriteFile(path, payload, error);
}

std::string SerializeRunConfig(const RunConfig& config, int indent) {
  json root = RunConfigToJson(config);
  return root.dump(indent);
}

bool ValidateRunConfig(const RunConfig& config, std::string* error) {
  if (config.schema_version != 1) {
    if (error) {
      *error = "unsupported schema_version";
    }
    return false;
  }
  const bool has_mesh = !config.domain_mesh.empty();
  if (!has_mesh) {
    if (config.pde_latex.empty()) {
      if (error) {
        *error = "missing pde.latex";
      }
      return false;
    }
    if (config.domain_bounds.empty()) {
      if (error) {
        *error = "missing domain.bounds";
      }
      return false;
    }
    if (config.grid.empty()) {
      if (error) {
        *error = "missing domain.grid";
      }
      return false;
    }
  }
  if (has_mesh) {
    if (!config.domain_mesh_format.empty()) {
      const std::string format = ToLowerCopy(config.domain_mesh_format);
      if (!IsMeshFormatToken(format)) {
        if (error) {
          *error = "unknown domain.mesh_format token";
        }
        return false;
      }
    }
    if (!config.domain_mesh_discretization.empty()) {
      const std::string token = ToLowerCopy(config.domain_mesh_discretization);
      if (!IsMeshDiscretizationToken(token)) {
        if (error) {
          *error = "unknown domain.mesh_discretization token";
        }
        return false;
      }
    }
  }
  if (!config.backend.empty()) {
    const std::string backend = ToLowerCopy(config.backend);
    if (!IsBackendToken(backend)) {
      if (error) {
        *error = "unknown backend token";
      }
      return false;
    }
  }
  if (!config.method.empty()) {
    const std::string method = ToLowerCopy(config.method);
    if (!IsMethodToken(method)) {
      if (error) {
        *error = "unknown solver method token";
      }
      return false;
    }
  }
  if (!config.output_format.empty()) {
    const std::string format = ToLowerCopy(config.output_format);
    if (format != "vtk" && format != "vti") {
      if (error) {
        *error = "output.format must be vtk or vti";
      }
      return false;
    }
  }
  return true;
}
