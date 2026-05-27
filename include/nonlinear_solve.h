#ifndef NONLINEAR_SOLVE_H
#define NONLINEAR_SOLVE_H

#include "pde_types.h"
#include "progress.h"
#include "solver.h"

SolveOutput SolveConservationLawTimeSeries(const SolveInput& input,
                                           const FrameCallback& on_frame,
                                           const ProgressCallback& progress);

bool ShouldUseConservationSolver(const SolveInput& input);

#endif  // NONLINEAR_SOLVE_H
