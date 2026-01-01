// Shared boundary input types for both Qt and ImGui frontends.
#ifndef BOUNDARY_TYPES_H
#define BOUNDARY_TYPES_H

#include <string>

// BoundaryInput mirrors the UI selections for a single face.
// kind: 0 Dirichlet, 1 Neumann, 2 Robin.
struct BoundaryInput {
  int kind = 0;
  std::string value = "0";
  std::string alpha = "1";
  std::string beta = "1";
  std::string gamma = "0";
};

#endif  // BOUNDARY_TYPES_H

