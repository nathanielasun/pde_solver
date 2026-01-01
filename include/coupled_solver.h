#ifndef COUPLED_SOLVER_H
#define COUPLED_SOLVER_H

#include <functional>
#include <map>
#include <string>
#include <vector>

#include "pde_types.h"
#include "progress.h"

// Solve a coupled multi-field PDE system.
// This dispatches to the appropriate coupling strategy based on input.coupling.
//
// For Explicit strategy: Solves each field equation once per time step,
// using previous values for cross-field coupling terms.
//
// For Picard strategy: Iteratively solves all field equations until
// the coupling residual falls below coupling_tol or max_coupling_iters is reached.
//
// Returns a SolveOutput with per-field results in field_outputs and
// coupling diagnostics in coupling_diagnostics.
SolveOutput SolveCoupledPDE(const SolveInput& input,
                            const ProgressCallback& progress = ProgressCallback());

// Callback type for coupled time-dependent solves
using CoupledFrameCallback =
    std::function<bool(int frame, double time,
                       const std::map<std::string, std::vector<double>>& field_grids)>;

// Solve a time-dependent coupled multi-field system.
// on_frame is called after each time step with all field solutions.
SolveOutput SolveCoupledPDETimeSeries(const SolveInput& input,
                                      const CoupledFrameCallback& on_frame,
                                      const ProgressCallback& progress = ProgressCallback());

// Compute the L2 norm of the difference between two field grids.
// Used to measure coupling convergence.
double ComputeFieldChangeNorm(const std::vector<double>& old_grid,
                              const std::vector<double>& new_grid);

// Build a SolveInput for a single field from a multi-field input.
// Coupling terms from other fields are added to the RHS using the
// provided field_grids map.
SolveInput BuildSingleFieldInput(const SolveInput& multi_input,
                                 size_t field_index,
                                 const std::map<std::string, std::vector<double>>& field_grids);

#endif  // COUPLED_SOLVER_H
