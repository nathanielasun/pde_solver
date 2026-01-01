#ifndef CPU_SOLVERS_RELAXATION_H
#define CPU_SOLVERS_RELAXATION_H

#include <functional>
#include <vector>

#include "embedded_boundary.h"
#include "pde_types.h"
#include "progress.h"

// Gauss-Seidel / SOR solver for 2D steady problems.
SolveOutput SolveGaussSeidelSor2D(const SolveInput& input,
                                  const Domain& d,
                                  double dx, double dy,
                                  const std::vector<unsigned char>* active,
                                  const std::vector<double>* integral_weights,
                                  bool use_shape,
                                  const std::vector<CellBoundaryInfo>& boundary_info,
                                  const ProgressCallback& progress,
                                  const std::function<bool()>& cancelled);

// Jacobi solver for 2D steady problems.
SolveOutput SolveJacobi2D(const SolveInput& input,
                          const Domain& d,
                          double dx, double dy,
                          const std::vector<unsigned char>* active,
                          const std::vector<double>* integral_weights,
                          bool use_shape,
                          const std::vector<CellBoundaryInfo>& boundary_info,
                          const ProgressCallback& progress,
                          const std::function<bool()>& cancelled);

#endif  // CPU_SOLVERS_RELAXATION_H
