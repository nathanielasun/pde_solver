#include "solve_service.h"

#include "solver.h"
#include "input_parse.h"
#include "vtk_io.h"

SolveResponse ExecuteSolve(const SolveRequest& request) {
  SolveResponse response;
  BackendKind selected = BackendKind::CPU;
  std::string note;

  // Validate boundary conditions before solving
  ParseResult bc_validate = ValidateBoundaryConditions(request.input.bc, request.input.domain);
  if (!bc_validate.ok) {
    response.error = "Boundary condition validation failed: " + bc_validate.error;
    response.backend_used = selected;
    response.note = note;
    return response;
  }

  SolveOutput output =
      SolveWithBackend(request.input, request.requested_backend, &selected, &note, request.progress);
  if (!output.error.empty()) {
    response.error = output.error;
    response.backend_used = selected;
    response.note = note;
    return response;
  }

  if (!request.output_path.empty()) {
    VtkWriteResult write_result =
        WriteVtkStructuredPoints(request.output_path, request.input.domain, output.grid,
                                 request.progress);
    if (!write_result.ok) {
      response.error = write_result.error;
      response.backend_used = selected;
      response.note = note;
      return response;
    }
  }

  response.ok = true;
  response.output_path = request.output_path;
  response.backend_used = selected;
  response.note = note;
  response.grid = output.grid;
  response.residual_l2 = output.residual_l2;
  response.residual_linf = output.residual_linf;
  response.residual_iters = output.residual_iters;
  response.residual_l2_history = output.residual_l2_history;
  response.residual_linf_history = output.residual_linf_history;
  response.derived =
      ComputeDerivedFields(request.input.domain, output.grid, request.input.pde.a,
                           request.input.pde.b, request.input.pde.az);
  return response;
}
