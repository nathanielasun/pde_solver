#ifndef BOUNDARY_PANEL_H
#define BOUNDARY_PANEL_H

#include "app_state.h"
#include "app_helpers.h"
#include <string>
#include <filesystem>
#include <vector>

// Forward declaration
class CommandHistory;

// State structure for boundary panel
struct BoundaryPanelState {
  // Boundary condition inputs (6 faces)
  BoundaryInput& bc_left;
  BoundaryInput& bc_right;
  BoundaryInput& bc_bottom;
  BoundaryInput& bc_top;
  BoundaryInput& bc_front;
  BoundaryInput& bc_back;
  
  // LaTeX previews for each boundary
  LatexTexture& bc_left_preview;
  LatexTexture& bc_right_preview;
  LatexTexture& bc_bottom_preview;
  LatexTexture& bc_top_preview;
  LatexTexture& bc_front_preview;
  LatexTexture& bc_back_preview;
  
  // Coordinate flags
  bool& use_cartesian_3d;
  bool& use_axisymmetric;
  bool& use_polar_coords;
  bool& use_cylindrical_volume;
  bool& use_surface;
  bool& use_volume;
  
  // LaTeX rendering settings
  std::string& python_path;
  std::filesystem::path& script_path;
  std::filesystem::path& cache_dir;
  std::string& latex_color;
  int& latex_font_size;
  
  // Optional command history for undo/redo
  CommandHistory* cmd_history = nullptr;
};

// Render the Boundary panel using configured components.
void RenderBoundaryPanel(BoundaryPanelState& state, const std::vector<std::string>& components);

#endif // BOUNDARY_PANEL_H
