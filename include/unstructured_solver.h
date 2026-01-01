#ifndef UNSTRUCTURED_SOLVER_H
#define UNSTRUCTURED_SOLVER_H

#include <string>
#include <vector>

#include "mesh_io.h"
#include "pde_types.h"

enum class UnstructuredDiscretization {
  FiniteElement,
  FiniteVolume,
};

struct UnstructuredSolveInput {
  PDECoefficients pde;
  UnstructuredMesh mesh;
  UnstructuredDiscretization discretization = UnstructuredDiscretization::FiniteElement;
};

struct UnstructuredSolveOutput {
  bool ok = false;
  std::string error;
  std::vector<double> point_values;
};

std::string DiscretizationToken(UnstructuredDiscretization discretization);
bool ParseDiscretizationToken(const std::string& token,
                              UnstructuredDiscretization* discretization);
UnstructuredSolveOutput SolveUnstructuredPDE(const UnstructuredSolveInput& input);

#endif  // UNSTRUCTURED_SOLVER_H
