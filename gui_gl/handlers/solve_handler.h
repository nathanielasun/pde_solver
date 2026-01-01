#ifndef SOLVE_HANDLER_H
#define SOLVE_HANDLER_H

#include "app_state.h"
#include "GlViewer.h"
#include "backend.h"
#include "pde_types.h"
#include "input_parse.h"
#include "vtk_io.h"
#include <string>
#include <filesystem>
#include <thread>
#include <atomic>
#include <functional>
#include <mutex>

// State structure for solve handler
struct SolveHandlerState {
  // Input fields
  std::string& pde_text;
  int& coord_mode;
  double& bound_xmin, &bound_xmax, &bound_ymin, &bound_ymax, &bound_zmin, &bound_zmax;
  int& grid_nx, &grid_ny, &grid_nz;
  int& domain_mode;
  std::string& domain_shape;
  std::string& domain_shape_file;
  std::string& domain_shape_mask_path;
  ShapeMask& shape_mask;
  double& shape_mask_threshold;
  bool& shape_mask_invert;
  ShapeTransform& shape_transform;
  BoundaryInput& bc_left, &bc_right, &bc_bottom, &bc_top, &bc_front, &bc_back;
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
  int& metal_tg_x, &metal_tg_y;
  double& time_start, &time_end;
  int& time_frames;
  std::string& output_path;
  
  // Shared state
  SharedState& state;
  std::mutex& state_mutex;
  
  // Viewer
  GlViewer& viewer;
  
  // Thread management
  std::thread* solver_thread;
  std::atomic<bool>& cancel_requested;
  
  // Callbacks
  std::function<void(const std::string&)> report_status;
  std::function<void()> start_solver;
};

// Launch the solve operation
void LaunchSolve(SolveHandlerState& handler_state);

#endif // SOLVE_HANDLER_H
