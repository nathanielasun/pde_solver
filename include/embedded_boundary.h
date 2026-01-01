#ifndef EMBEDDED_BOUNDARY_H
#define EMBEDDED_BOUNDARY_H

#include "pde_types.h"
#include <vector>

// Information about the embedded boundary crossing a grid edge
struct BoundaryIntercept {
  bool valid = false;           // true if this edge crosses the boundary
  double fraction = 0.0;        // fraction along edge (0 = at start, 1 = at end)
  double x = 0.0, y = 0.0;     // actual boundary point coordinates
  double z = 0.0;               // for 3D
  double normal_x = 0.0;        // outward normal vector components
  double normal_y = 0.0;
  double normal_z = 0.0;
};

// Boundary information for a grid cell
struct CellBoundaryInfo {
  BoundaryIntercept left;       // intercept on left edge (if any)
  BoundaryIntercept right;      // intercept on right edge (if any)
  BoundaryIntercept bottom;     // intercept on bottom edge (if any)
  BoundaryIntercept top;        // intercept on top edge (if any)
  BoundaryIntercept front;      // intercept on front edge (if any, 3D)
  BoundaryIntercept back;       // intercept on back edge (if any, 3D)
  bool is_cut_cell = false;     // true if cell is cut by boundary
  bool center_inside = false;   // cell center is inside implicit domain
  double volume_fraction = 1.0; // fraction of cell volume inside domain (for cut cells)
};

// Build boundary information for all grid cells
// Returns true on success, false on error
bool BuildEmbeddedBoundary2D(
    const Domain& domain,
    const std::string& domain_shape,
    const ShapeMask& shape_mask,
    const ShapeTransform& shape_transform,
    double shape_mask_threshold,
    bool shape_mask_invert,
    std::vector<CellBoundaryInfo>* boundary_info,
    std::string* error);

bool BuildEmbeddedBoundary3D(
    const Domain& domain,
    const std::string& domain_shape,
    const ShapeMask& shape_mask,
    const ShapeTransform& shape_transform,
    double shape_mask_threshold,
    bool shape_mask_invert,
    std::vector<CellBoundaryInfo>* boundary_info,
    std::string* error);

// Apply boundary condition on embedded boundary using ghost-fluid method
// For Dirichlet: sets ghost point value to satisfy BC
// For Neumann: adjusts ghost point to satisfy normal derivative BC
// For Robin: applies mixed BC
void ApplyEmbeddedBoundaryBC2D(
    const BoundaryCondition& bc,
    const CellBoundaryInfo& cell_info,
    int i, int j,
    const Domain& domain,
    double dx, double dy,
    std::vector<double>* grid);

void ApplyEmbeddedBoundaryBC3D(
    const BoundaryCondition& bc,
    const CellBoundaryInfo& cell_info,
    int i, int j, int k,
    const Domain& domain,
    double dx, double dy, double dz,
    std::vector<double>* grid);

#endif  // EMBEDDED_BOUNDARY_H
