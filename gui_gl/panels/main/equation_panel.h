#ifndef EQUATION_PANEL_H
#define EQUATION_PANEL_H

#include "app_state.h"
#include "GlViewer.h"
#include "templates.h"
#include "validation.h"
#include "ui_helpers.h"
#include "app_helpers.h"
#include <string>
#include <filesystem>
#include <vector>

// Forward declaration
class CommandHistory;

// State structure for equation panel
struct EquationPanelState {
  // PDE input
  std::string& pde_text;
  
  // Domain bounds (for template application)
  double& bound_xmin;
  double& bound_xmax;
  double& bound_ymin;
  double& bound_ymax;
  double& bound_zmin;
  double& bound_zmax;
  
  // Grid resolution (for template application)
  int& grid_nx;
  int& grid_ny;
  int& grid_nz;
  
  // Domain settings (for template application)
  int& domain_mode;
  std::string& domain_shape;
  int& coord_mode;
  
  // Boundary conditions (for template application)
  BoundaryInput& bc_left;
  BoundaryInput& bc_right;
  BoundaryInput& bc_bottom;
  BoundaryInput& bc_top;
  BoundaryInput& bc_front;
  BoundaryInput& bc_back;
  
  // Solver settings (for template application)
  int& method_index;
  double& sor_omega;
  int& gmres_restart;
  
  // Time settings (for template application)
  double& time_start;
  double& time_end;
  int& time_frames;
  
  // Viewer (for setting view mode)
  GlViewer& viewer;
  
  // Refresh callback (for updating coordinate flags)
  std::function<void()> refresh_coord_flags;
  
  // LaTeX preview
  LatexTexture& pde_preview;
  
  // LaTeX rendering settings
  std::string& python_path;
  std::filesystem::path& script_path;
  std::filesystem::path& cache_dir;
  std::string& latex_color;
  int& latex_font_size;
  
  // UI width
  float input_width;
  
  // Optional command history for undo/redo
  CommandHistory* cmd_history = nullptr;
};

// Render the Equation panel using configured components.
void RenderEquationPanel(EquationPanelState& state, const std::vector<std::string>& components);

#endif // EQUATION_PANEL_H
