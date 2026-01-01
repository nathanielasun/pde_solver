#ifndef METAL_SOLVE_H
#define METAL_SOLVE_H

#include "pde_types.h"
#include "progress.h"
#include <functional>
#include <string>
#include <vector>

using FrameCallback =
    std::function<bool(int frame, double time, const std::vector<double>& grid,
                       const std::vector<double>* velocity)>;

// Checks if a Metal-capable device is available.
bool MetalIsAvailable(std::string* note);

// Main dispatch function for the Metal backend (stationary PDEs).
SolveOutput SolvePDEMetal(const SolveInput& input);

// Metal solver for time-dependent PDEs.
SolveOutput SolvePDETimeSeriesMetal(const SolveInput& input,
                                     const FrameCallback& on_frame,
                                     const ProgressCallback& progress);

#endif // METAL_SOLVE_H
