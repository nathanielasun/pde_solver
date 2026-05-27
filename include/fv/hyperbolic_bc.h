#ifndef FV_HYPERBOLIC_BC_H
#define FV_HYPERBOLIC_BC_H

#include "pde_types.h"

// Ghost-cell value for FV face reconstruction at a boundary.
double FVGhostCellValue(const BoundaryCondition& bc, double u_interior, double u_neighbor);

#endif  // FV_HYPERBOLIC_BC_H
