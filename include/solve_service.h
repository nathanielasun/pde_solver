// Shared solve orchestration used by both Qt and ImGui frontends.
#ifndef SOLVE_SERVICE_H
#define SOLVE_SERVICE_H

#include <string>
#include <vector>

#include "backend.h"
#include "pde_types.h"
#include "progress.h"
#include "vtk_io.h"

struct SolveRequest {
  SolveInput input;
  BackendKind requested_backend = BackendKind::Auto;
  std::string output_path;
  ProgressCallback progress;
};

struct SolveResponse {
  bool ok = false;
  std::string output_path;
  std::string error;
  BackendKind backend_used = BackendKind::CPU;
  std::string note;
  std::vector<double> grid;
  DerivedFields derived;
  double residual_l2 = 0.0;
  double residual_linf = 0.0;
  std::vector<int> residual_iters;
  std::vector<double> residual_l2_history;
  std::vector<double> residual_linf_history;
};

SolveResponse ExecuteSolve(const SolveRequest& request);

#endif  // SOLVE_SERVICE_H
