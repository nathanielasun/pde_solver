#ifndef GRID_PANEL_H
#define GRID_PANEL_H

#include "validation.h"
#include "pde_types.h"
#include <string>
#include <vector>

// Forward declaration
class CommandHistory;

// State structure for grid panel
struct GridPanelState {
  // Grid resolution
  int& grid_nx;
  int& grid_ny;
  int& grid_nz;
  
  // Coordinate flags
  bool& use_cartesian_3d;
  bool& use_cylindrical_volume;
  bool& use_volume;
  bool& use_surface;
  bool& use_axisymmetric;
  bool& use_polar_coords;
  
  // UI width
  float input_width;
  
  // Optional command history for undo/redo
  CommandHistory* cmd_history = nullptr;
};

// Render the Grid panel using configured components.
void RenderGridPanel(GridPanelState& state, const std::vector<std::string>& components);

#endif // GRID_PANEL_H
