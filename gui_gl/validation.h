#ifndef VALIDATION_H
#define VALIDATION_H

#include <string>
#include "pde_types.h"
#include "latex_parser.h"

// Validation status for UI indicators
enum class ValidationStatus {
  Valid,      // Green - no issues
  Warning,    // Yellow - potential issues
  Error       // Red - critical errors
};

// Validation state structure
struct ValidationState {
  // PDE validation
  ValidationStatus pde_status = ValidationStatus::Valid;
  std::string pde_error;
  std::string pde_warning;
  
  // Domain validation
  ValidationStatus domain_status = ValidationStatus::Valid;
  std::string domain_error;
  std::string domain_warning;
  
  // Grid validation
  ValidationStatus grid_status = ValidationStatus::Valid;
  std::string grid_warning;
  
  // Boundary condition validation (6 boundaries: left, right, bottom, top, front, back)
  ValidationStatus bc_status[6] = {ValidationStatus::Valid, ValidationStatus::Valid,
                                   ValidationStatus::Valid, ValidationStatus::Valid,
                                   ValidationStatus::Valid, ValidationStatus::Valid};
  std::string bc_errors[6];
  std::string bc_warnings[6];
  
  // Overall validation
  bool is_valid() const {
    if (pde_status == ValidationStatus::Error || domain_status == ValidationStatus::Error) {
      return false;
    }
    for (int i = 0; i < 6; ++i) {
      if (bc_status[i] == ValidationStatus::Error) {
        return false;
      }
    }
    return true;
  }
  
  bool has_warnings() const {
    return pde_status == ValidationStatus::Warning || 
           domain_status == ValidationStatus::Warning ||
           grid_status == ValidationStatus::Warning ||
           bc_status[0] == ValidationStatus::Warning ||
           bc_status[1] == ValidationStatus::Warning ||
           bc_status[2] == ValidationStatus::Warning ||
           bc_status[3] == ValidationStatus::Warning ||
           bc_status[4] == ValidationStatus::Warning ||
           bc_status[5] == ValidationStatus::Warning;
  }
};

// Validate PDE input
ValidationState ValidatePDE(const std::string& pde_text, int coord_mode);

// Validate domain bounds
ValidationState ValidateDomain(double xmin, double xmax, double ymin, double ymax,
                               double zmin, double zmax, int coord_mode, bool is_3d);

// Validate grid resolution
ValidationState ValidateGrid(int nx, int ny, int nz, bool is_3d, const Domain& domain);

// Validate boundary condition expression
ValidationState ValidateBC(const std::string& bc_expr, const std::string& bc_type,
                           int coord_mode, bool is_3d);

// Comprehensive validation
ValidationState ValidateInputs(const std::string& pde_text,
                               double xmin, double xmax, double ymin, double ymax,
                               double zmin, double zmax,
                               int nx, int ny, int nz,
                               int coord_mode,
                               const std::string& bc_left, const std::string& bc_right,
                               const std::string& bc_bottom, const std::string& bc_top,
                               const std::string& bc_front, const std::string& bc_back,
                               const Domain& domain);

#endif  // VALIDATION_H

