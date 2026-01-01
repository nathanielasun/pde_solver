#include "templates.h"
#include "app_state.h"
#include <algorithm>
#include <sstream>

namespace {
// Helper to create coordinate mode index from CoordinateSystem
int CoordSystemToMode(CoordinateSystem cs) {
  switch (cs) {
    case CoordinateSystem::Cartesian: return 0;  // kCoordCartesian2D/3D
    case CoordinateSystem::Polar: return 2;  // kCoordPolar
    case CoordinateSystem::Axisymmetric: return 3;  // kCoordAxisymmetric
    case CoordinateSystem::Cylindrical: return 4;  // kCoordCylindricalVolume
    case CoordinateSystem::SphericalSurface: return 5;  // kCoordSphericalSurface
    case CoordinateSystem::SphericalVolume: return 6;  // kCoordSphericalVolume
    case CoordinateSystem::ToroidalSurface: return 7;  // kCoordToroidalSurface
    case CoordinateSystem::ToroidalVolume: return 8;  // kCoordToroidalVolume
    default: return 0;
  }
}

// Helper to convert SolveMethod to method index
int MethodToIndex(SolveMethod method) {
  switch (method) {
    case SolveMethod::Jacobi: return 0;
    case SolveMethod::GaussSeidel: return 1;
    case SolveMethod::SOR: return 2;
    case SolveMethod::CG: return 3;
    case SolveMethod::BiCGStab: return 4;
    case SolveMethod::GMRES: return 5;
    case SolveMethod::MultigridVcycle: return 6;
    default: return 0;
  }
}
}

std::vector<ProblemTemplate> GetProblemTemplates() {
  std::vector<ProblemTemplate> templates;
  
  // 1. Poisson Equation (2D)
  {
    ProblemTemplate t;
    t.name = "Poisson 2D";
    t.description = "2D Poisson equation: -∇²u = f with zero Dirichlet BCs";
    t.pde_latex = "-u_{xx} - u_{yy} = 1";
    t.domain_bounds = "0,1,0,1";
    t.grid_resolution = "64,64";
    t.coord_system = CoordinateSystem::Cartesian;
    t.bc_left = "0";
    t.bc_right = "0";
    t.bc_bottom = "0";
    t.bc_top = "0";
    t.bc_left_kind = 0;  // Dirichlet
    t.bc_right_kind = 0;
    t.bc_bottom_kind = 0;
    t.bc_top_kind = 0;
    t.recommended_method = SolveMethod::CG;
    t.notes = "Classic elliptic PDE. CG or Multigrid work well.";
    templates.push_back(t);
  }
  
  // 2. Heat Equation (2D)
  {
    ProblemTemplate t;
    t.name = "Heat Equation 2D";
    t.description = "2D heat equation: ∂u/∂t = c∇²u with initial condition";
    t.pde_latex = "u_t = 0.1(u_{xx} + u_{yy})";
    t.domain_bounds = "0,1,0,1";
    t.grid_resolution = "64,64";
    t.coord_system = CoordinateSystem::Cartesian;
    t.bc_left = "3";
    t.bc_right = "5";
    t.bc_bottom = "3";
    t.bc_top = "5";
    t.bc_left_kind = 0;
    t.bc_right_kind = 0;
    t.bc_bottom_kind = 0;
    t.bc_top_kind = 0;
    t.recommended_method = SolveMethod::CG;
    t.notes = "Time-dependent parabolic PDE. Requires time stepping.";
    templates.push_back(t);
  }
  
  // 3. Wave Equation (2D)
  {
    ProblemTemplate t;
    t.name = "Wave Equation 2D";
    t.description = "2D wave equation: ∂²u/∂t² = c²∇²u";
    t.pde_latex = "u_{tt} = 1.0(u_{xx} + u_{yy})";
    t.domain_bounds = "0,1,0,1";
    t.grid_resolution = "64,64";
    t.coord_system = CoordinateSystem::Cartesian;
    t.bc_left = "0";
    t.bc_right = "0";
    t.bc_bottom = "0";
    t.bc_top = "0";
    t.bc_left_kind = 0;
    t.bc_right_kind = 0;
    t.bc_bottom_kind = 0;
    t.bc_top_kind = 0;
    t.recommended_method = SolveMethod::CG;
    t.notes = "Hyperbolic PDE. Requires time stepping with appropriate CFL condition.";
    templates.push_back(t);
  }
  
  // 4. Poisson 3D
  {
    ProblemTemplate t;
    t.name = "Poisson 3D";
    t.description = "3D Poisson equation: -∇²u = f";
    t.pde_latex = "-u_{xx} - u_{yy} - u_{zz} = 1";
    t.domain_bounds = "0,1,0,1,0,1";
    t.grid_resolution = "32,32,32";
    t.coord_system = CoordinateSystem::Cartesian;
    t.bc_left = "0";
    t.bc_right = "0";
    t.bc_bottom = "0";
    t.bc_top = "0";
    t.bc_front = "0";
    t.bc_back = "0";
    t.bc_left_kind = 0;
    t.bc_right_kind = 0;
    t.bc_bottom_kind = 0;
    t.bc_top_kind = 0;
    t.bc_front_kind = 0;
    t.bc_back_kind = 0;
    t.recommended_method = SolveMethod::CG;
    t.notes = "3D elliptic PDE. Lower resolution recommended for performance.";
    templates.push_back(t);
  }
  
  // 5. Poisson with Mixed BCs
  {
    ProblemTemplate t;
    t.name = "Poisson Mixed BCs";
    t.description = "Poisson with Dirichlet on left/right, Neumann on top/bottom";
    t.pde_latex = "-u_{xx} - u_{yy} = 1";
    t.domain_bounds = "0,1,0,1";
    t.grid_resolution = "64,64";
    t.coord_system = CoordinateSystem::Cartesian;
    t.bc_left = "0";
    t.bc_right = "0";
    t.bc_bottom = "0";
    t.bc_top = "0";
    t.bc_left_kind = 0;  // Dirichlet
    t.bc_right_kind = 0;  // Dirichlet
    t.bc_bottom_kind = 1;  // Neumann
    t.bc_top_kind = 1;  // Neumann
    t.recommended_method = SolveMethod::CG;
    t.notes = "Mixed boundary conditions example.";
    templates.push_back(t);
  }
  
  // 6. Poisson in Polar Coordinates
  {
    ProblemTemplate t;
    t.name = "Poisson Polar";
    t.description = "Poisson equation in polar coordinates (r, θ)";
    t.pde_latex = "-u_{rr} - (1/r)u_r - (1/r^2)u_{thetatheta} = 1";
    t.domain_bounds = "0.1,1,0,6.28318";  // r: 0.1 to 1, theta: 0 to 2π
    t.grid_resolution = "32,64";
    t.coord_system = CoordinateSystem::Polar;
    t.bc_left = "0";  // r min
    t.bc_right = "sin(theta)";  // r max
    t.bc_bottom = "0";  // theta min (periodic)
    t.bc_top = "0";  // theta max (periodic)
    t.bc_left_kind = 0;
    t.bc_right_kind = 0;
    t.bc_bottom_kind = 0;
    t.bc_top_kind = 0;
    t.recommended_method = SolveMethod::CG;
    t.notes = "Polar coordinates with periodic boundary in theta.";
    templates.push_back(t);
  }
  
  // 7. Reaction-Diffusion
  {
    ProblemTemplate t;
    t.name = "Reaction-Diffusion";
    t.description = "Reaction-diffusion: ∂u/∂t = D∇²u + f(u)";
    t.pde_latex = "u_t = 0.1(u_{xx} + u_{yy}) + u(1-u)";
    t.domain_bounds = "0,10,0,10";
    t.grid_resolution = "64,64";
    t.coord_system = CoordinateSystem::Cartesian;
    t.bc_left = "0";
    t.bc_right = "0";
    t.bc_bottom = "0";
    t.bc_top = "0";
    t.bc_left_kind = 0;
    t.bc_right_kind = 0;
    t.bc_bottom_kind = 0;
    t.bc_top_kind = 0;
    t.recommended_method = SolveMethod::CG;
    t.notes = "Nonlinear reaction term u(1-u). Requires time stepping.";
    templates.push_back(t);
  }
  
  // 8. Helmholtz Equation
  {
    ProblemTemplate t;
    t.name = "Helmholtz";
    t.description = "Helmholtz equation: -∇²u + k²u = f";
    t.pde_latex = "-u_{xx} - u_{yy} + 4u = 1";
    t.domain_bounds = "0,1,0,1";
    t.grid_resolution = "64,64";
    t.coord_system = CoordinateSystem::Cartesian;
    t.bc_left = "0";
    t.bc_right = "0";
    t.bc_bottom = "0";
    t.bc_top = "0";
    t.bc_left_kind = 0;
    t.bc_right_kind = 0;
    t.bc_bottom_kind = 0;
    t.bc_top_kind = 0;
    t.recommended_method = SolveMethod::CG;
    t.notes = "Helmholtz equation with wave number k=2.";
    templates.push_back(t);
  }
  
  return templates;
}

ProblemTemplate GetTemplate(const std::string& name) {
  auto templates = GetProblemTemplates();
  for (const auto& t : templates) {
    if (t.name == name) {
      return t;
    }
  }
  // Return first template as default if not found
  return templates.empty() ? ProblemTemplate{} : templates[0];
}

// Forward declaration - BoundaryInput is defined in main.cpp
struct BoundaryInput;

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
                         double& time_start, double& time_end, int& time_frames) {
  pde_text = t.pde_latex;
  
  // Parse domain bounds
  std::istringstream bounds_stream(t.domain_bounds);
  std::string token;
  std::vector<double> bounds;
  while (std::getline(bounds_stream, token, ',')) {
    bounds.push_back(std::stod(token));
  }
  if (bounds.size() >= 4) {
    bound_xmin = bounds[0];
    bound_xmax = bounds[1];
    bound_ymin = bounds[2];
    bound_ymax = bounds[3];
    if (bounds.size() >= 6) {
      bound_zmin = bounds[4];
      bound_zmax = bounds[5];
    }
  }
  
  // Parse grid resolution
  std::istringstream grid_stream(t.grid_resolution);
  std::vector<int> grid;
  while (std::getline(grid_stream, token, ',')) {
    grid.push_back(std::stoi(token));
  }
  if (grid.size() >= 2) {
    grid_nx = grid[0];
    grid_ny = grid[1];
    if (grid.size() >= 3) {
      grid_nz = grid[2];
    }
  }
  
  coord_mode = CoordSystemToMode(t.coord_system);
  domain_mode = 0;  // Box domain
  domain_shape = "";
  
  // Set boundary conditions
  bc_left.kind = t.bc_left_kind;
  bc_left.value = t.bc_left;
  bc_right.kind = t.bc_right_kind;
  bc_right.value = t.bc_right;
  bc_bottom.kind = t.bc_bottom_kind;
  bc_bottom.value = t.bc_bottom;
  bc_top.kind = t.bc_top_kind;
  bc_top.value = t.bc_top;
  bc_front.kind = t.bc_front_kind;
  bc_front.value = t.bc_front;
  bc_back.kind = t.bc_back_kind;
  bc_back.value = t.bc_back;
  
  method_index = MethodToIndex(t.recommended_method);
  sor_omega = 1.5;  // Default
  gmres_restart = 30;  // Default
  time_start = 0.0;
  time_end = 1.0;
  time_frames = 60;
}

