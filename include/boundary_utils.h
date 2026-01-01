// Shared boundary condition helpers used by both Qt and ImGui frontends.
#ifndef BOUNDARY_UTILS_H
#define BOUNDARY_UTILS_H

#include <string>

#include "boundary_types.h"

struct BoundaryLatexResult {
  bool ok = false;
  std::string latex;
  std::string error;
};

// Convert UI input into human-readable LaTeX text.
BoundaryLatexResult BuildBoundaryLatex(const BoundaryInput& input);

// Build a boundary specification string understood by ApplyBoundarySpec.
// Returns true on success, false on failure and fills error.
bool BuildBoundarySpec(const BoundaryInput& left, const BoundaryInput& right,
                       const BoundaryInput& bottom, const BoundaryInput& top,
                       const BoundaryInput& front, const BoundaryInput& back,
                       std::string* spec_out, std::string* error);

// Expression helpers (shared with both UIs).
std::string LatexifyExpr(const std::string& input);
std::string NormalizeLatexExpr(const std::string& input);

#endif  // BOUNDARY_UTILS_H

