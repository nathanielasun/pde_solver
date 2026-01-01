#include "unstructured_solver.h"

#include "string_utils.h"

std::string DiscretizationToken(UnstructuredDiscretization discretization) {
  switch (discretization) {
    case UnstructuredDiscretization::FiniteElement:
      return "fe";
    case UnstructuredDiscretization::FiniteVolume:
      return "fv";
  }
  return "fe";
}

bool ParseDiscretizationToken(const std::string& token,
                              UnstructuredDiscretization* discretization) {
  if (!discretization) {
    return false;
  }
  const std::string lower = pde::ToLower(token);
  if (lower == "fe" || lower == "fem" || lower == "finite-element") {
    *discretization = UnstructuredDiscretization::FiniteElement;
    return true;
  }
  if (lower == "fv" || lower == "fvm" || lower == "finite-volume") {
    *discretization = UnstructuredDiscretization::FiniteVolume;
    return true;
  }
  return false;
}

UnstructuredSolveOutput SolveUnstructuredPDE(const UnstructuredSolveInput& input) {
  (void)input;
  UnstructuredSolveOutput output;
  output.ok = false;
  output.error = "unstructured solver stub (finite-element backend planned)";
  return output;
}
