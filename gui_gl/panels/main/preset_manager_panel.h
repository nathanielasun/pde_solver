#ifndef PRESET_MANAGER_PANEL_H
#define PRESET_MANAGER_PANEL_H

#include "boundary_types.h"
#include <string>
#include <vector>

struct PresetManagerPanelState {
  float input_width;
  std::string& preset_directory;

  // PDE configuration references for save/load
  std::string& pde_text;
  double& bound_xmin;
  double& bound_xmax;
  double& bound_ymin;
  double& bound_ymax;
  double& bound_zmin;
  double& bound_zmax;
  int& grid_nx;
  int& grid_ny;
  int& grid_nz;
  int& coord_mode;
  int& method_index;
  double& solver_tol;
  int& solver_max_iter;
  BoundaryInput& bc_left;
  BoundaryInput& bc_right;
  BoundaryInput& bc_bottom;
  BoundaryInput& bc_top;
  BoundaryInput& bc_front;
  BoundaryInput& bc_back;
};

void RenderPresetManagerPanel(PresetManagerPanelState& state,
                               const std::vector<std::string>& components);

#endif  // PRESET_MANAGER_PANEL_H
