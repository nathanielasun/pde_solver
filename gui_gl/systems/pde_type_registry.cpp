#include "pde_type_registry.h"
#include <algorithm>
#include <cmath>

PDETypeRegistry& PDETypeRegistry::Instance() {
  static PDETypeRegistry instance;
  if (!instance.initialized_) {
    instance.InitializeBuiltInTypes();
    instance.initialized_ = true;
  }
  return instance;
}

void PDETypeRegistry::Register(const std::string& type_id, const PDETypeMetadata& metadata) {
  types_[type_id] = metadata;
}

std::vector<std::string> PDETypeRegistry::GetTypes() const {
  std::vector<std::string> result;
  result.reserve(types_.size());
  for (const auto& pair : types_) {
    result.push_back(pair.first);
  }
  return result;
}

const PDETypeMetadata* PDETypeRegistry::GetMetadata(const std::string& type_id) const {
  auto it = types_.find(type_id);
  if (it != types_.end()) {
    return &it->second;
  }
  return nullptr;
}

bool PDETypeRegistry::Validate(const std::string& type_id, const LatexParseResult& parse_result) const {
  const PDETypeMetadata* metadata = GetMetadata(type_id);
  if (!metadata || !metadata->validator) {
    return false;
  }
  return metadata->validator(parse_result);
}

std::string PDETypeRegistry::DetectType(const LatexParseResult& parse_result) const {
  // Heuristic detection based on coefficients
  const auto& c = parse_result.coeffs;
  
  // Check for time derivatives
  bool has_time = std::abs(c.ut) > 1e-12 || std::abs(c.utt) > 1e-12;
  bool has_second_time = std::abs(c.utt) > 1e-12;
  
  // Check for spatial derivatives
  bool has_laplacian = std::abs(c.a) > 1e-12 || std::abs(c.b) > 1e-12 || std::abs(c.az) > 1e-12;
  bool has_first_order = std::abs(c.c) > 1e-12 || std::abs(c.d) > 1e-12 || std::abs(c.dz) > 1e-12;
  bool has_mixed = std::abs(c.ab) > 1e-12 || std::abs(c.ac) > 1e-12 || std::abs(c.bc) > 1e-12;
  
  // Check for nonlinear terms
  bool has_nonlinear = !parse_result.nonlinear.empty();
  bool has_nonlinear_derivatives = !parse_result.nonlinear_derivatives.empty();
  
  // Wave equation: u_tt = c^2 * (u_xx + u_yy)
  if (has_second_time && has_laplacian && !has_first_order && !has_nonlinear && !has_nonlinear_derivatives) {
    return "wave";
  }
  
  // Heat/Diffusion equation: u_t = D * (u_xx + u_yy)
  if (has_time && !has_second_time && has_laplacian && !has_first_order && !has_nonlinear && !has_nonlinear_derivatives) {
    return "heat";
  }
  
  // Poisson equation: u_xx + u_yy = f
  if (!has_time && has_laplacian && !has_first_order && !has_nonlinear && !has_nonlinear_derivatives) {
    return "poisson";
  }
  
  // Laplace equation: u_xx + u_yy = 0
  if (!has_time && has_laplacian && !has_first_order && !has_nonlinear && !has_nonlinear_derivatives && 
      std::abs(c.f) < 1e-12 && parse_result.coeffs.rhs_latex.empty()) {
    return "laplace";
  }
  
  // Advection equation: u_t + c * u_x = 0
  if (has_time && !has_second_time && !has_laplacian && has_first_order && !has_nonlinear && !has_nonlinear_derivatives) {
    return "advection";
  }
  
  // Burgers' equation: u_t + u * u_x = nu * u_xx
  if (has_time && has_laplacian && has_nonlinear_derivatives) {
    return "burgers";
  }
  
  // Transport equation: u_t + c * u_x + d * u_y = 0
  if (has_time && !has_laplacian && has_first_order && !has_nonlinear && !has_nonlinear_derivatives) {
    return "transport";
  }
  
  // Helmholtz equation: u_xx + u_yy + k^2 * u = 0
  if (!has_time && has_laplacian && std::abs(c.e) > 1e-12 && !has_first_order && !has_nonlinear && !has_nonlinear_derivatives) {
    return "helmholtz";
  }
  
  return "unknown";
}

SolveInput PDETypeRegistry::BuildInput(const std::string& type_id, const LatexParseResult& parse_result) const {
  const PDETypeMetadata* metadata = GetMetadata(type_id);
  if (metadata && metadata->input_builder) {
    return metadata->input_builder(parse_result);
  }
  
  // Default: create basic SolveInput from parse result
  SolveInput input;
  input.pde = parse_result.coeffs;
  input.integrals = parse_result.integrals;
  input.nonlinear = parse_result.nonlinear;
  input.nonlinear_derivatives = parse_result.nonlinear_derivatives;
  return input;
}

void PDETypeRegistry::InitializeBuiltInTypes() {
  // Poisson Equation: u_xx + u_yy = f(x,y)
  PDETypeMetadata poisson;
  poisson.name = "Poisson Equation";
  poisson.description = "Elliptic PDE: ∇²u = f. Used in electrostatics, fluid flow, and steady-state problems.";
  poisson.required_terms = {"u_xx", "u_yy"};
  poisson.optional_terms = {"u_zz", "f"};
  poisson.default_coord_system = CoordinateSystem::Cartesian;
  poisson.example_latex = "u_{xx} + u_{yy} = \\sin(x) \\cos(y)";
  poisson.common_applications = {"Electrostatics", "Steady-state heat transfer", "Fluid flow"};
  poisson.validator = [](const LatexParseResult& result) {
    const auto& c = result.coeffs;
    bool has_laplacian = std::abs(c.a) > 1e-12 || std::abs(c.b) > 1e-12 || std::abs(c.az) > 1e-12;
    bool no_time = std::abs(c.ut) < 1e-12 && std::abs(c.utt) < 1e-12;
    return has_laplacian && no_time;
  };
  poisson.input_builder = [](const LatexParseResult& result) {
    SolveInput input;
    input.pde = result.coeffs;
    input.integrals = result.integrals;
    input.nonlinear = result.nonlinear;
    input.nonlinear_derivatives = result.nonlinear_derivatives;
    return input;
  };
  poisson.default_bcs = {
    {"left", "u=0"},
    {"right", "u=0"},
    {"bottom", "u=0"},
    {"top", "u=0"}
  };
  Register("poisson", poisson);
  
  // Laplace Equation: u_xx + u_yy = 0
  PDETypeMetadata laplace;
  laplace.name = "Laplace Equation";
  laplace.description = "Special case of Poisson equation with zero RHS: ∇²u = 0. Harmonic functions.";
  laplace.required_terms = {"u_xx", "u_yy"};
  laplace.optional_terms = {"u_zz"};
  laplace.default_coord_system = CoordinateSystem::Cartesian;
  laplace.example_latex = "u_{xx} + u_{yy} = 0";
  laplace.common_applications = {"Potential theory", "Harmonic functions", "Conformal mapping"};
  laplace.validator = [](const LatexParseResult& result) {
    const auto& c = result.coeffs;
    bool has_laplacian = std::abs(c.a) > 1e-12 || std::abs(c.b) > 1e-12 || std::abs(c.az) > 1e-12;
    bool no_time = std::abs(c.ut) < 1e-12 && std::abs(c.utt) < 1e-12;
    bool zero_rhs = std::abs(c.f) < 1e-12 && result.coeffs.rhs_latex.empty();
    return has_laplacian && no_time && zero_rhs;
  };
  laplace.input_builder = poisson.input_builder;
  laplace.default_bcs = poisson.default_bcs;
  Register("laplace", laplace);
  
  // Heat/Diffusion Equation: u_t = D * (u_xx + u_yy)
  PDETypeMetadata heat;
  heat.name = "Heat/Diffusion Equation";
  heat.description = "Parabolic PDE: u_t = D∇²u. Describes heat conduction, diffusion, and random walks.";
  heat.required_terms = {"u_t", "u_xx", "u_yy"};
  heat.optional_terms = {"u_zz", "f"};
  heat.default_coord_system = CoordinateSystem::Cartesian;
  heat.example_latex = "u_t = 0.1 (u_{xx} + u_{yy})";
  heat.common_applications = {"Heat conduction", "Diffusion processes", "Brownian motion"};
  heat.validator = [](const LatexParseResult& result) {
    const auto& c = result.coeffs;
    bool has_time = std::abs(c.ut) > 1e-12;
    bool has_laplacian = std::abs(c.a) > 1e-12 || std::abs(c.b) > 1e-12 || std::abs(c.az) > 1e-12;
    bool no_second_time = std::abs(c.utt) < 1e-12;
    return has_time && has_laplacian && no_second_time;
  };
  heat.input_builder = poisson.input_builder;
  heat.default_bcs = poisson.default_bcs;
  Register("heat", heat);
  
  // Wave Equation: u_tt = c² * (u_xx + u_yy)
  PDETypeMetadata wave;
  wave.name = "Wave Equation";
  wave.description = "Hyperbolic PDE: u_tt = c²∇²u. Describes wave propagation, vibrations, and oscillations.";
  wave.required_terms = {"u_tt", "u_xx", "u_yy"};
  wave.optional_terms = {"u_zz"};
  wave.default_coord_system = CoordinateSystem::Cartesian;
  wave.example_latex = "u_{tt} = c^2 (u_{xx} + u_{yy})";
  wave.common_applications = {"Wave propagation", "Vibrations", "Acoustics", "Electromagnetic waves"};
  wave.validator = [](const LatexParseResult& result) {
    const auto& c = result.coeffs;
    bool has_second_time = std::abs(c.utt) > 1e-12;
    bool has_laplacian = std::abs(c.a) > 1e-12 || std::abs(c.b) > 1e-12 || std::abs(c.az) > 1e-12;
    return has_second_time && has_laplacian;
  };
  wave.input_builder = poisson.input_builder;
  wave.default_bcs = poisson.default_bcs;
  Register("wave", wave);
  
  // Advection Equation: u_t + c * u_x = 0
  PDETypeMetadata advection;
  advection.name = "Advection Equation";
  advection.description = "First-order hyperbolic PDE: u_t + c·∇u = 0. Describes transport of quantities.";
  advection.required_terms = {"u_t", "u_x"};
  advection.optional_terms = {"u_y", "u_z"};
  advection.default_coord_system = CoordinateSystem::Cartesian;
  advection.example_latex = "u_t + 2 u_x = 0";
  advection.common_applications = {"Fluid transport", "Particle advection", "Signal propagation"};
  advection.validator = [](const LatexParseResult& result) {
    const auto& c = result.coeffs;
    bool has_time = std::abs(c.ut) > 1e-12;
    bool has_first_order = std::abs(c.c) > 1e-12 || std::abs(c.d) > 1e-12 || std::abs(c.dz) > 1e-12;
    bool no_laplacian = std::abs(c.a) < 1e-12 && std::abs(c.b) < 1e-12 && std::abs(c.az) < 1e-12;
    return has_time && has_first_order && no_laplacian;
  };
  advection.input_builder = poisson.input_builder;
  advection.default_bcs = {
    {"left", "u=sin(t)"},
    {"right", "u_x=0"},
    {"bottom", "u_y=0"},
    {"top", "u_y=0"}
  };
  Register("advection", advection);
  
  // Burgers' Equation: u_t + u * u_x = nu * u_xx
  PDETypeMetadata burgers;
  burgers.name = "Burgers' Equation";
  burgers.description = "Nonlinear PDE: u_t + u·u_x = νu_xx. Combines nonlinear advection with diffusion.";
  burgers.required_terms = {"u_t", "u_x", "u_xx"};
  burgers.optional_terms = {"u_y", "u_yy"};
  burgers.default_coord_system = CoordinateSystem::Cartesian;
  burgers.example_latex = "u_t + u u_x = 0.01 u_{xx}";
  burgers.common_applications = {"Fluid dynamics", "Shock waves", "Traffic flow"};
  burgers.validator = [](const LatexParseResult& result) {
    const auto& c = result.coeffs;
    bool has_time = std::abs(c.ut) > 1e-12;
    bool has_laplacian = std::abs(c.a) > 1e-12 || std::abs(c.b) > 1e-12;
    bool has_nonlinear_deriv = !result.nonlinear_derivatives.empty();
    return has_time && has_laplacian && has_nonlinear_deriv;
  };
  burgers.input_builder = poisson.input_builder;
  burgers.default_bcs = {
    {"left", "u=1"},
    {"right", "u=0"},
    {"bottom", "u_y=0"},
    {"top", "u_y=0"}
  };
  Register("burgers", burgers);
  
  // Helmholtz Equation: u_xx + u_yy + k²u = 0
  PDETypeMetadata helmholtz;
  helmholtz.name = "Helmholtz Equation";
  helmholtz.description = "Eigenvalue problem: ∇²u + k²u = 0. Appears in wave problems and eigenmodes.";
  helmholtz.required_terms = {"u_xx", "u_yy", "u"};
  helmholtz.optional_terms = {"u_zz"};
  helmholtz.default_coord_system = CoordinateSystem::Cartesian;
  helmholtz.example_latex = "u_{xx} + u_{yy} + k^2 u = 0";
  helmholtz.common_applications = {"Eigenvalue problems", "Resonance", "Standing waves"};
  helmholtz.validator = [](const LatexParseResult& result) {
    const auto& c = result.coeffs;
    bool has_laplacian = std::abs(c.a) > 1e-12 || std::abs(c.b) > 1e-12 || std::abs(c.az) > 1e-12;
    bool has_zeroth_order = std::abs(c.e) > 1e-12;
    bool no_time = std::abs(c.ut) < 1e-12 && std::abs(c.utt) < 1e-12;
    return has_laplacian && has_zeroth_order && no_time;
  };
  helmholtz.input_builder = poisson.input_builder;
  helmholtz.default_bcs = poisson.default_bcs;
  Register("helmholtz", helmholtz);
}

