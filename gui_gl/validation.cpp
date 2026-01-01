#include "validation.h"
#include "input_parse.h"
#include "coefficient_evaluator.h"
#include "systems/coordinate_system_registry.h"
#include "utils/coordinate_utils.h"
#include <cmath>
#include <algorithm>

namespace {
constexpr int kMinGrid = 3;
constexpr int kMaxGrid = 2048;
constexpr int kRecommendedMinGrid = 16;
constexpr int kRecommendedMaxGrid = 512;
}

ValidationState ValidatePDE(const std::string& pde_text, int coord_mode) {
  ValidationState state;
  
  if (pde_text.empty()) {
    state.pde_status = ValidationStatus::Error;
    state.pde_error = "PDE expression cannot be empty";
    return state;
  }
  
  LatexParser parser;
  std::string pde_for_parse = pde_text;
  
  // Apply coordinate system remapping if needed
  const bool use_surface = (coord_mode == 5 || coord_mode == 7);  // SphericalSurface or ToroidalSurface
  const bool use_axisymmetric = (coord_mode == 3);  // Axisymmetric
  
  if (use_surface) {
    // Remap would be done here - for now just parse
  }
  if (use_axisymmetric) {
    // Remap would be done here - for now just parse
  }
  
  LatexParseResult parse_result = parser.Parse(pde_for_parse);
  if (!parse_result.ok) {
    state.pde_status = ValidationStatus::Error;
    state.pde_error = parse_result.error;
    return state;
  }

  CoefficientEvaluator coeff_eval = BuildCoefficientEvaluator(parse_result.coeffs);
  if (!coeff_eval.ok) {
    state.pde_status = ValidationStatus::Error;
    state.pde_error = coeff_eval.error;
    return state;
  }
  
  // Check for reasonable coefficients
  const auto& c = parse_result.coeffs;
  if (std::abs(c.a) < 1e-12 && std::abs(c.b) < 1e-12 && 
      std::abs(c.az) < 1e-12 && std::abs(c.ut) < 1e-12 && 
      std::abs(c.utt) < 1e-12) {
    state.pde_status = ValidationStatus::Warning;
    state.pde_warning = "PDE has no second-order spatial or time derivatives";
  }
  
  state.pde_status = ValidationStatus::Valid;
  return state;
}

ValidationState ValidateDomain(double xmin, double xmax, double ymin, double ymax,
                               double zmin, double zmax, int coord_mode, bool is_3d) {
  ValidationState state;
  
  // Check bounds ordering
  if (xmin >= xmax) {
    state.domain_status = ValidationStatus::Error;
    state.domain_error = "xmin must be less than xmax";
    return state;
  }
  
  if (ymin >= ymax) {
    state.domain_status = ValidationStatus::Error;
    state.domain_error = "ymin must be less than ymax";
    return state;
  }
  
  if (is_3d && zmin >= zmax) {
    state.domain_status = ValidationStatus::Error;
    state.domain_error = "zmin must be less than zmax";
    return state;
  }
  
  // Check for reasonable domain size
  const double xspan = xmax - xmin;
  const double yspan = ymax - ymin;
  const double zspan = is_3d ? (zmax - zmin) : 0.0;
  
  if (xspan < 1e-10 || yspan < 1e-10 || (is_3d && zspan < 1e-10)) {
    state.domain_status = ValidationStatus::Warning;
    state.domain_warning = "Domain span is very small, may cause numerical issues";
  }
  
  // Coordinate-specific validation using registry
  CoordinateSystem current_system = CoordModeToSystem(coord_mode);
  const CoordinateSystemMetadata* metadata = 
      CoordinateSystemRegistry::Instance().GetMetadata(current_system);
  
  if (metadata && metadata->validate_bounds) {
    // Use registry validation function
    if (!metadata->validate_bounds(xmin, ymin, is_3d ? zmin : 0.0)) {
      state.domain_status = ValidationStatus::Error;
      // Try to get a more specific error message
      const CoordinateSystemAxis* axis0 = CoordinateSystemRegistry::Instance().GetAxis(current_system, 0);
      if (axis0 && axis0->name.find("Radius") != std::string::npos && xmin < 0.0) {
        state.domain_error = axis0->name + " must be non-negative";
      } else {
        state.domain_error = "Invalid bounds for " + metadata->name;
      }
      return state;
    }
  } else {
    // Fallback to old coordinate-specific checks
    if (coord_mode == 2 || coord_mode == 3 || coord_mode == 4 || 
        coord_mode == 6 || coord_mode == 8) {  // Polar, Axisymmetric, Cylindrical, Spherical, Toroidal
      if (xmin < 0.0) {
        state.domain_status = ValidationStatus::Error;
        state.domain_error = "Radial coordinate (r) must be non-negative";
        return state;
      }
    }
  }
  
  state.domain_status = ValidationStatus::Valid;
  return state;
}

ValidationState ValidateGrid(int nx, int ny, int nz, bool is_3d, const Domain& domain) {
  ValidationState state;
  
  if (nx < kMinGrid || ny < kMinGrid || (is_3d && nz < kMinGrid)) {
    state.grid_status = ValidationStatus::Error;
    state.grid_warning = "Grid resolution too low (minimum " + std::to_string(kMinGrid) + ")";
    return state;
  }
  
  if (nx > kMaxGrid || ny > kMaxGrid || (is_3d && nz > kMaxGrid)) {
    state.grid_status = ValidationStatus::Warning;
    state.grid_warning = "Grid resolution very high, may be slow";
    return state;
  }
  
  if (nx < kRecommendedMinGrid || ny < kRecommendedMinGrid || 
      (is_3d && nz < kRecommendedMinGrid)) {
    state.grid_status = ValidationStatus::Warning;
    state.grid_warning = "Grid resolution may be too low for accurate results";
    return state;
  }
  
  // Estimate memory usage
  const size_t grid_size = static_cast<size_t>(nx) * ny * (is_3d ? nz : 1);
  const double memory_mb = (grid_size * sizeof(double)) / (1024.0 * 1024.0);
  if (memory_mb > 1000.0) {
    state.grid_status = ValidationStatus::Warning;
    state.grid_warning = "Large grid size (~" + std::to_string(static_cast<int>(memory_mb)) + " MB)";
  }
  
  state.grid_status = ValidationStatus::Valid;
  return state;
}

ValidationState ValidateBC(const std::string& bc_expr, const std::string& bc_type,
                           int coord_mode, bool is_3d) {
  ValidationState state;
  
  if (bc_expr.empty() && bc_type != "Dirichlet") {
    // Neumann/Robin BCs need expressions
    state.bc_status[0] = ValidationStatus::Warning;
    state.bc_warnings[0] = "Boundary condition expression is empty";
    return state;
  }
  
  // Basic syntax check - try to parse as expression
  // For now, just check if non-empty for non-Dirichlet
  if (bc_type == "Neumann" || bc_type == "Robin") {
    if (bc_expr.empty()) {
      state.bc_status[0] = ValidationStatus::Error;
      state.bc_errors[0] = "Expression required for " + bc_type + " boundary condition";
      return state;
    }
  }
  
  state.bc_status[0] = ValidationStatus::Valid;
  return state;
}

ValidationState ValidateInputs(const std::string& pde_text,
                               double xmin, double xmax, double ymin, double ymax,
                               double zmin, double zmax,
                               int nx, int ny, int nz,
                               int coord_mode,
                               const std::string& bc_left, const std::string& bc_right,
                               const std::string& bc_bottom, const std::string& bc_top,
                               const std::string& bc_front, const std::string& bc_back,
                               const Domain& domain) {
  ValidationState state;
  
  // Validate PDE
  ValidationState pde_state = ValidatePDE(pde_text, coord_mode);
  state.pde_status = pde_state.pde_status;
  state.pde_error = pde_state.pde_error;
  state.pde_warning = pde_state.pde_warning;
  
  // Validate domain
  const bool is_3d = (coord_mode == 1 || coord_mode == 4 || coord_mode == 6 || coord_mode == 8);
  ValidationState domain_state = ValidateDomain(xmin, xmax, ymin, ymax, zmin, zmax, coord_mode, is_3d);
  state.domain_status = domain_state.domain_status;
  state.domain_error = domain_state.domain_error;
  state.domain_warning = domain_state.domain_warning;
  
  // Validate grid
  ValidationState grid_state = ValidateGrid(nx, ny, nz, is_3d, domain);
  state.grid_status = grid_state.grid_status;
  state.grid_warning = grid_state.grid_warning;
  
  // Boundary conditions are validated individually in UI
  // This is a placeholder - full BC validation would parse expressions
  
  return state;
}
