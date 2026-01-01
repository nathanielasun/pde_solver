#ifndef METAL_SOLVER_H
#define METAL_SOLVER_H

#include <string>

#include "pde_types.h"

bool MetalIsAvailable(std::string* note);
SolveOutput SolvePDEMetal(const SolveInput& input);

#endif
