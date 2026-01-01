#ifndef METAL_TIME_SERIES_H
#define METAL_TIME_SERIES_H

#include "pde_types.h"
#include "progress.h"
#include <functional>
#include <vector>

using FrameCallback =
    std::function<bool(int frame, double time, const std::vector<double>& grid,
                       const std::vector<double>* velocity)>;

// Metal GPU-accelerated time-dependent PDE solver
SolveOutput SolvePDETimeSeriesMetal(const SolveInput& input,
                                     const FrameCallback& on_frame,
                                     const ProgressCallback& progress);

#endif // METAL_TIME_SERIES_H
