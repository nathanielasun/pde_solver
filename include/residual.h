#ifndef RESIDUAL_H
#define RESIDUAL_H

#include <string>
#include <vector>

#include "pde_types.h"

struct ResidualNorms {
  double l2 = 0.0;
  double linf = 0.0;
};

// Computes discrete residual norms for the current steady-state solve:
// - L2: sqrt(sum_i r_i^2)
// - Linf: max_i |r_i|
//
// Returns false and fills error if residual cannot be computed.
bool ComputeResidualNorms(const SolveInput& input,
                          const std::vector<double>& grid,
                          ResidualNorms* out,
                          std::string* error);

#endif


