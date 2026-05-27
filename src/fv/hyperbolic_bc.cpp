#include "fv/hyperbolic_bc.h"

#include <cmath>

double FVGhostCellValue(const BoundaryCondition& bc, double u_interior, double u_neighbor) {
  switch (bc.kind) {
    case BCKind::Outflow:
    case BCKind::Transmissive:
      return u_interior;
    case BCKind::Inflow:
      if (!bc.value.latex.empty()) {
        return bc.value.constant;
      }
      return bc.value.constant + bc.value.x * u_interior;
    case BCKind::Dirichlet:
      return bc.value.constant;
    case BCKind::Neumann:
      return u_neighbor;
    case BCKind::Robin:
      return u_interior;
  }
  return u_interior;
}
