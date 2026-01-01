#ifndef DOMAIN_PANEL_H
#define DOMAIN_PANEL_H

#include "app_state.h"
#include "GlViewer.h"
#include "app_helpers.h"
#include <string>
#include <filesystem>
#include <functional>
#include <vector>

// Forward declaration
class CommandHistory;

// State structure for domain panel
struct DomainPanelState {
  // Coordinate system
  int& coord_mode;
  
  // Domain bounds
  double& bound_xmin;
  double& bound_xmax;
  double& bound_ymin;
  double& bound_ymax;
  double& bound_zmin;
  double& bound_zmax;
  
  // Torus geometry (for toroidal coordinates)
  double& torus_major;
  double& torus_minor;
  
  // Domain mode (0 = rectangular, 1 = implicit)
  int& domain_mode;
  
  // Implicit domain shape
  std::string& domain_shape;

  // Optional shape inputs
  std::string& domain_shape_file;
  std::string& domain_shape_mask_path;
  ShapeMask& shape_mask;
  double& shape_mask_threshold;
  bool& shape_mask_invert;
  ShapeTransform& shape_transform;
  
  // Viewer (for setting view mode and torus radii)
  GlViewer& viewer;
  
  // Refresh callback (for updating coordinate flags)
  std::function<void()> refresh_coord_flags;
  
  // Coordinate flags (computed by refresh_coord_flags)
  bool& use_polar_coords;
  bool& use_cartesian_3d;
  bool& use_axisymmetric;
  bool& use_cylindrical_volume;
  bool& use_spherical_surface;
  bool& use_spherical_volume;
  bool& use_toroidal_surface;
  bool& use_toroidal_volume;
  bool& use_surface;
  bool& use_volume;
  
  // LaTeX preview for shape
  LatexTexture& shape_preview;
  
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

// Render the Domain panel using configured components.
void RenderDomainPanel(DomainPanelState& state, const std::vector<std::string>& components);

#endif // DOMAIN_PANEL_H
