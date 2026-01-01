#ifndef SOLVER_H
#define SOLVER_H

#include <functional>

#include "pde_types.h"
#include "progress.h"

using FrameCallback =
    std::function<bool(int frame, double time, const std::vector<double>& grid,
                       const std::vector<double>* velocity)>;

SolveOutput SolvePDE(const SolveInput& input, const ProgressCallback& progress = ProgressCallback());
SolveOutput SolvePDETimeSeries(const SolveInput& input, const FrameCallback& on_frame,
                               const ProgressCallback& progress = ProgressCallback());

#endif
