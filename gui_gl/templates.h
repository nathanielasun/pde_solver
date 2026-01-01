#ifndef TEMPLATES_H
#define TEMPLATES_H

#include <string>
#include <vector>
#include "pde_types.h"

// Problem template structure
struct ProblemTemplate {
  std::string name;
  std::string description;
  std::string pde_latex;
  std::string domain_bounds = "0,1,0,1";  // "xmin,xmax,ymin,ymax" or "xmin,xmax,ymin,ymax,zmin,zmax"
  std::string grid_resolution = "64,64";  // "nx,ny" or "nx,ny,nz"
  CoordinateSystem coord_system = CoordinateSystem::Cartesian;
  std::string bc_left = "0";
  std::string bc_right = "0";
  std::string bc_bottom = "0";
  std::string bc_top = "0";
  std::string bc_front = "0";  // For 3D
  std::string bc_back = "0";  // For 3D
  int bc_left_kind = 0;  // 0=Dirichlet, 1=Neumann, 2=Robin
  int bc_right_kind = 0;
  int bc_bottom_kind = 0;
  int bc_top_kind = 0;
  int bc_front_kind = 0;
  int bc_back_kind = 0;
  SolveMethod recommended_method = SolveMethod::Jacobi;
  std::string notes;  // Additional notes about the template
};

// Get all available templates
std::vector<ProblemTemplate> GetProblemTemplates();

// Get template by name
ProblemTemplate GetTemplate(const std::string& name);

// BoundaryInput is now defined in main.cpp (moved out of anonymous namespace)
// Forward declaration
struct BoundaryInput;

// Apply template to UI state (helper function for UI integration)
// This version works with the actual UI state structure
void ApplyTemplateToState(const ProblemTemplate& t,
                         std::string& pde_text,
                         double& bound_xmin, double& bound_xmax,
                         double& bound_ymin, double& bound_ymax,
                         double& bound_zmin, double& bound_zmax,
                         int& grid_nx, int& grid_ny, int& grid_nz,
                         int& domain_mode, std::string& domain_shape,
                         int& coord_mode,
                         BoundaryInput& bc_left, BoundaryInput& bc_right,
                         BoundaryInput& bc_bottom, BoundaryInput& bc_top,
                         BoundaryInput& bc_front, BoundaryInput& bc_back,
                         int& method_index, double& sor_omega, int& gmres_restart,
                         double& time_start, double& time_end, int& time_frames);

#endif  // TEMPLATES_H
