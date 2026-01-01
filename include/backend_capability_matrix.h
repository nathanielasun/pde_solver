#ifndef BACKEND_CAPABILITY_MATRIX_H
#define BACKEND_CAPABILITY_MATRIX_H

#include <algorithm>
#include <map>
#include <string>
#include <vector>

#include "backend.h"
#include "pde_types.h"

struct BackendCapabilities {
  std::vector<SolveMethod> supported_methods;
  bool supports_3d = false;
  bool supports_spatial_rhs = false;
  bool supports_nonlinear = false;
  bool supports_integrals = false;
  bool supports_shapes = false;
  bool supports_time_dependent = false;
  bool supports_piecewise_bc = false;
  // Multi-field and coupling support
  bool supports_coupled_fields = false;
  bool supports_explicit_coupling = false;  // One-way coupling (e.g., u_t = v_xx, v_t = v_yy)
  bool supports_implicit_coupling = false;  // Two-way/iterative coupling
  std::map<std::string, std::string> custom_options;

  bool SupportsMethod(SolveMethod method) const {
    return std::find(supported_methods.begin(), supported_methods.end(), method) !=
           supported_methods.end();
  }

  std::string GetDescription() const;
};

BackendCapabilities GetBackendCapabilities(BackendKind kind);
bool BackendSupportsMethod(BackendKind kind, SolveMethod method);
bool BackendSupportsInput(BackendKind kind, const SolveInput& input, std::string* reason);
bool BackendSupportsMethodForInput(BackendKind kind, SolveMethod method,
                                   const SolveInput& input, std::string* reason);
std::vector<BackendKind> BackendsSupportingMethod(SolveMethod method);

// Multi-field coupling support
bool BackendSupportsCoupling(BackendKind kind, CouplingPattern pattern, std::string* reason);
bool BackendSupportsMultiFieldInput(BackendKind kind, const MultiFieldEquation& multi_eq,
                                    const CouplingAnalysis& coupling, std::string* reason);

#endif  // BACKEND_CAPABILITY_MATRIX_H
