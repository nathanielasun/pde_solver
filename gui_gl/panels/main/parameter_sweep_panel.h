#ifndef PARAMETER_SWEEP_PANEL_H
#define PARAMETER_SWEEP_PANEL_H

#include <string>
#include <vector>
#include <mutex>

#include "app_state.h"
#include "boundary_types.h"

struct ParameterSweepPanelState {
  float input_width;
  std::string& pde_text;
  int& coord_mode;
  double& bound_xmin;
  double& bound_xmax;
  double& bound_ymin;
  double& bound_ymax;
  double& bound_zmin;
  double& bound_zmax;
  int& grid_nx;
  int& grid_ny;
  int& grid_nz;
  int& domain_mode;
  std::string& domain_shape;
  std::string& domain_shape_file;
  std::string& domain_shape_mask_path;
  ShapeMask& shape_mask;
  double& shape_mask_threshold;
  bool& shape_mask_invert;
  ShapeTransform& shape_transform;
  BoundaryInput& bc_left;
  BoundaryInput& bc_right;
  BoundaryInput& bc_bottom;
  BoundaryInput& bc_top;
  BoundaryInput& bc_front;
  BoundaryInput& bc_back;
  int& backend_index;
  int& method_index;
  double& sor_omega;
  int& gmres_restart;
  int& solver_max_iter;
  double& solver_tol;
  int& solver_residual_interval;
  int& solver_mg_pre_smooth;
  int& solver_mg_post_smooth;
  int& solver_mg_coarse_iters;
  int& solver_mg_max_levels;
  int& thread_count;
  int& metal_reduce_interval;
  int& metal_tg_x;
  int& metal_tg_y;
  double& time_start;
  double& time_end;
  int& time_frames;
  std::string& output_path;
  SharedState& shared_state;
  std::mutex& shared_state_mutex;
};

void RenderParameterSweepPanel(ParameterSweepPanelState& state,
                                const std::vector<std::string>& components);

#endif  // PARAMETER_SWEEP_PANEL_H
