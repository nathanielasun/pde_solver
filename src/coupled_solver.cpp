#include "coupled_solver.h"

#include <algorithm>
#include <cmath>
#include <sstream>

#include "solver.h"

namespace {

// Compute L2 norm of vector
double L2Norm(const std::vector<double>& v) {
  double sum = 0.0;
  for (double val : v) {
    sum += val * val;
  }
  return std::sqrt(sum);
}

// Check if cancelled
bool IsCancelled(const SolveInput& input) {
  return input.cancel && input.cancel->load();
}

}  // namespace

double ComputeFieldChangeNorm(const std::vector<double>& old_grid,
                              const std::vector<double>& new_grid) {
  if (old_grid.size() != new_grid.size()) {
    return std::numeric_limits<double>::infinity();
  }
  double sum = 0.0;
  for (size_t i = 0; i < old_grid.size(); ++i) {
    double diff = new_grid[i] - old_grid[i];
    sum += diff * diff;
  }
  return std::sqrt(sum / static_cast<double>(old_grid.size()));
}

SolveInput BuildSingleFieldInput(const SolveInput& multi_input,
                                 size_t field_index,
                                 const std::map<std::string, std::vector<double>>& field_grids) {
  if (field_index >= multi_input.fields.size()) {
    SolveInput empty;
    return empty;
  }

  const FieldDefinition& field = multi_input.fields[field_index];

  // Build single-field input
  SolveInput single;
  single.pde = field.pde;
  single.domain = multi_input.domain;
  single.bc = field.bc;
  single.solver = multi_input.solver;
  single.domain_shape = multi_input.domain_shape;
  single.shape_transform = multi_input.shape_transform;
  single.shape_mask = multi_input.shape_mask;
  single.shape_mask_threshold = multi_input.shape_mask_threshold;
  single.shape_mask_invert = multi_input.shape_mask_invert;
  single.embedded_bc = multi_input.embedded_bc;
  single.time = multi_input.time;
  single.cancel = multi_input.cancel;

  // Use initial grid from field if available, otherwise from field_grids
  if (!field.initial_grid.empty()) {
    single.initial_grid = field.initial_grid;
  } else {
    auto it = field_grids.find(field.name);
    if (it != field_grids.end() && !it->second.empty()) {
      single.initial_grid = it->second;
    }
  }

  if (!field.initial_velocity.empty()) {
    single.initial_velocity = field.initial_velocity;
  }

  // Note: Cross-field coupling terms would need to be added to the RHS here.
  // For now, we handle this by using the field_grids values as source terms.
  // This is a simplified approach; a full implementation would evaluate
  // v_xx, w_yy etc. from field_grids and add to the RHS.

  return single;
}

// Solve using explicit coupling (operator splitting)
// Each field is solved once using current values from other fields
SolveOutput SolveExplicitCoupling(const SolveInput& input,
                                   const ProgressCallback& progress) {
  SolveOutput result;

  if (!IsMultiField(input)) {
    // Fall back to single-field solve
    return SolvePDE(input, progress);
  }

  const size_t num_fields = input.fields.size();
  const int nx = input.domain.nx;
  const int ny = input.domain.ny;
  const int nz = input.domain.nz;
  const size_t grid_size = static_cast<size_t>(nx * ny * std::max(1, nz));

  // Initialize field grids
  std::map<std::string, std::vector<double>> field_grids;
  for (const auto& field : input.fields) {
    if (!field.initial_grid.empty()) {
      field_grids[field.name] = field.initial_grid;
    } else {
      field_grids[field.name] = std::vector<double>(grid_size, 0.0);
    }
  }

  result.field_outputs.resize(num_fields);
  result.coupling_diagnostics.coupling_iters = 1;  // Explicit = single pass
  result.coupling_diagnostics.converged = true;

  // Solve each field sequentially
  for (size_t i = 0; i < num_fields; ++i) {
    if (IsCancelled(input)) {
      result.error = "cancelled";
      return result;
    }

    const FieldDefinition& field = input.fields[i];

    // Build single-field input using current values from other fields
    SolveInput single_input = BuildSingleFieldInput(input, i, field_grids);

    // Create field-specific progress callback
    ProgressCallback field_progress;
    if (progress) {
      field_progress = [&, i](const std::string& phase, double pct) {
        // Scale progress to show overall multi-field progress
        double overall_progress = (static_cast<double>(i) + pct) / static_cast<double>(num_fields);
        progress("field " + field.name + ": " + phase, overall_progress);
      };
    }

    // Solve this field
    SolveOutput field_output = SolvePDE(single_input, field_progress);

    if (!field_output.error.empty()) {
      result.error = "field '" + field.name + "': " + field_output.error;
      return result;
    }

    // Store result
    field_grids[field.name] = field_output.grid;
    result.field_outputs[i].name = field.name;
    result.field_outputs[i].grid = std::move(field_output.grid);
    result.field_outputs[i].residual_l2 = field_output.residual_l2;
    result.field_outputs[i].residual_linf = field_output.residual_linf;

    // Accumulate residual history
    for (int ri : field_output.residual_iters) {
      result.residual_iters.push_back(ri);
    }
    for (double rl2 : field_output.residual_l2_history) {
      result.residual_l2_history.push_back(rl2);
    }
    for (double rlinf : field_output.residual_linf_history) {
      result.residual_linf_history.push_back(rlinf);
    }
  }

  // Set overall residuals from last field (or could aggregate)
  if (!result.field_outputs.empty()) {
    result.residual_l2 = result.field_outputs.back().residual_l2;
    result.residual_linf = result.field_outputs.back().residual_linf;
    result.grid = result.field_outputs[0].grid;  // Primary field grid
  }

  return result;
}

// Solve using Picard iteration (block Gauss-Seidel)
// Iteratively solve all fields until convergence
SolveOutput SolvePicardCoupling(const SolveInput& input,
                                 const ProgressCallback& progress) {
  SolveOutput result;

  if (!IsMultiField(input)) {
    // Fall back to single-field solve
    return SolvePDE(input, progress);
  }

  const size_t num_fields = input.fields.size();
  const int nx = input.domain.nx;
  const int ny = input.domain.ny;
  const int nz = input.domain.nz;
  const size_t grid_size = static_cast<size_t>(nx * ny * std::max(1, nz));

  const CouplingConfig& config = input.coupling;

  // Initialize field grids
  std::map<std::string, std::vector<double>> field_grids;
  std::map<std::string, std::vector<double>> prev_grids;
  for (const auto& field : input.fields) {
    if (!field.initial_grid.empty()) {
      field_grids[field.name] = field.initial_grid;
    } else {
      field_grids[field.name] = std::vector<double>(grid_size, 0.0);
    }
    prev_grids[field.name] = field_grids[field.name];
  }

  result.field_outputs.resize(num_fields);
  result.coupling_diagnostics.converged = false;

  // Picard iteration loop
  for (int coupling_iter = 0; coupling_iter < config.max_coupling_iters; ++coupling_iter) {
    if (IsCancelled(input)) {
      result.error = "cancelled";
      return result;
    }

    // Save previous grids for convergence check
    for (const auto& field : input.fields) {
      prev_grids[field.name] = field_grids[field.name];
    }

    // Solve each field using current values from other fields
    for (size_t i = 0; i < num_fields; ++i) {
      if (IsCancelled(input)) {
        result.error = "cancelled";
        return result;
      }

      const FieldDefinition& field = input.fields[i];

      // Build single-field input with current coupling values
      SolveInput single_input = BuildSingleFieldInput(input, i, field_grids);

      // Use previous solution as initial guess (warm start)
      if (!field_grids[field.name].empty()) {
        single_input.initial_grid = field_grids[field.name];
      }

      // Solve this field
      SolveOutput field_output = SolvePDE(single_input, nullptr);

      if (!field_output.error.empty()) {
        result.error = "coupling iter " + std::to_string(coupling_iter) +
                       ", field '" + field.name + "': " + field_output.error;
        return result;
      }

      // Apply relaxation if enabled
      if (config.use_relaxation && config.relaxation_factor < 1.0) {
        const double omega = config.relaxation_factor;
        const std::vector<double>& old_grid = field_grids[field.name];
        std::vector<double>& new_grid = field_output.grid;
        for (size_t j = 0; j < new_grid.size() && j < old_grid.size(); ++j) {
          new_grid[j] = omega * new_grid[j] + (1.0 - omega) * old_grid[j];
        }
      }

      // Update field grid
      field_grids[field.name] = std::move(field_output.grid);

      // Store latest result
      result.field_outputs[i].name = field.name;
      result.field_outputs[i].grid = field_grids[field.name];
      result.field_outputs[i].residual_l2 = field_output.residual_l2;
      result.field_outputs[i].residual_linf = field_output.residual_linf;
    }

    // Compute coupling residual (max change across all fields)
    double max_change = 0.0;
    for (const auto& field : input.fields) {
      double change = ComputeFieldChangeNorm(prev_grids[field.name], field_grids[field.name]);
      max_change = std::max(max_change, change);

      // Record per-field history
      result.coupling_diagnostics.per_field_history[field.name].push_back(change);
    }

    result.coupling_diagnostics.coupling_residual_history.push_back(max_change);
    result.coupling_diagnostics.coupling_iters = coupling_iter + 1;

    // Report progress
    if (progress) {
      std::ostringstream phase_oss;
      phase_oss << "coupling iter " << (coupling_iter + 1) << "/" << config.max_coupling_iters;
      progress(phase_oss.str(), 1.0 - max_change / std::max(1.0, config.coupling_tol * 10));
    }

    // Check convergence
    if (max_change < config.coupling_tol) {
      result.coupling_diagnostics.converged = true;
      break;
    }
  }

  // Set warning if not converged
  if (!result.coupling_diagnostics.converged) {
    std::ostringstream oss;
    oss << "coupling did not converge after " << result.coupling_diagnostics.coupling_iters
        << " iterations (residual = ";
    if (!result.coupling_diagnostics.coupling_residual_history.empty()) {
      oss << result.coupling_diagnostics.coupling_residual_history.back();
    } else {
      oss << "N/A";
    }
    oss << ", tol = " << config.coupling_tol << ")";
    result.coupling_diagnostics.warning = oss.str();
  }

  // Set overall residuals
  if (!result.field_outputs.empty()) {
    double max_l2 = 0.0;
    double max_linf = 0.0;
    for (const auto& fo : result.field_outputs) {
      max_l2 = std::max(max_l2, fo.residual_l2);
      max_linf = std::max(max_linf, fo.residual_linf);
    }
    result.residual_l2 = max_l2;
    result.residual_linf = max_linf;
    result.grid = result.field_outputs[0].grid;  // Primary field grid
  }

  return result;
}

SolveOutput SolveCoupledPDE(const SolveInput& input,
                            const ProgressCallback& progress) {
  // Dispatch based on coupling strategy
  switch (input.coupling.strategy) {
    case CouplingStrategy::Explicit:
      return SolveExplicitCoupling(input, progress);
    case CouplingStrategy::Picard:
      return SolvePicardCoupling(input, progress);
    default:
      return SolveExplicitCoupling(input, progress);
  }
}

SolveOutput SolveCoupledPDETimeSeries(const SolveInput& input,
                                      const CoupledFrameCallback& on_frame,
                                      const ProgressCallback& progress) {
  SolveOutput result;

  if (!IsMultiField(input)) {
    // Fall back to single-field time series solve
    FrameCallback single_callback = nullptr;
    if (on_frame) {
      single_callback = [&on_frame](int frame, double time,
                                    const std::vector<double>& grid,
                                    const std::vector<double>* velocity) {
        std::map<std::string, std::vector<double>> field_grids;
        field_grids["u"] = grid;
        return on_frame(frame, time, field_grids);
      };
    }
    return SolvePDETimeSeries(input, single_callback, progress);
  }

  const size_t num_fields = input.fields.size();
  const int nx = input.domain.nx;
  const int ny = input.domain.ny;
  const int nz = input.domain.nz;
  const size_t grid_size = static_cast<size_t>(nx * ny * std::max(1, nz));

  const TimeConfig& time_cfg = input.time;
  const int num_steps = static_cast<int>((time_cfg.t_end - time_cfg.t_start) / time_cfg.dt);
  const int frame_interval = std::max(1, num_steps / std::max(1, time_cfg.frames));

  // Initialize field grids
  std::map<std::string, std::vector<double>> field_grids;
  for (const auto& field : input.fields) {
    if (!field.initial_grid.empty()) {
      field_grids[field.name] = field.initial_grid;
    } else {
      field_grids[field.name] = std::vector<double>(grid_size, 0.0);
    }
  }

  result.field_outputs.resize(num_fields);
  result.coupling_diagnostics.coupling_iters = 0;
  result.coupling_diagnostics.converged = true;

  double current_time = time_cfg.t_start;
  int frame = 0;

  // Emit initial frame
  if (on_frame && !on_frame(frame, current_time, field_grids)) {
    result.error = "callback requested stop at frame 0";
    return result;
  }

  // Time stepping loop
  for (int step = 0; step < num_steps; ++step) {
    if (IsCancelled(input)) {
      result.error = "cancelled";
      return result;
    }

    current_time += time_cfg.dt;

    // Create time-step input
    SolveInput step_input = input;

    // Update initial grids for this time step
    for (size_t i = 0; i < num_fields; ++i) {
      step_input.fields[i].initial_grid = field_grids[input.fields[i].name];
    }

    // Solve coupled system for this time step
    SolveOutput step_output = SolveCoupledPDE(step_input, nullptr);

    if (!step_output.error.empty()) {
      result.error = "time step " + std::to_string(step) + ": " + step_output.error;
      return result;
    }

    // Update field grids
    for (const auto& fo : step_output.field_outputs) {
      field_grids[fo.name] = fo.grid;
    }

    // Accumulate coupling diagnostics
    result.coupling_diagnostics.coupling_iters += step_output.coupling_diagnostics.coupling_iters;
    if (!step_output.coupling_diagnostics.converged) {
      result.coupling_diagnostics.converged = false;
      if (result.coupling_diagnostics.warning.empty()) {
        result.coupling_diagnostics.warning = step_output.coupling_diagnostics.warning;
      }
    }

    // Emit frame if needed
    if ((step + 1) % frame_interval == 0 || step == num_steps - 1) {
      ++frame;
      if (on_frame && !on_frame(frame, current_time, field_grids)) {
        result.error = "callback requested stop at frame " + std::to_string(frame);
        return result;
      }
    }

    // Report progress
    if (progress) {
      double pct = static_cast<double>(step + 1) / static_cast<double>(num_steps);
      std::ostringstream phase_oss;
      phase_oss << "time step " << (step + 1) << "/" << num_steps;
      progress(phase_oss.str(), pct);
    }
  }

  // Store final field outputs
  for (size_t i = 0; i < num_fields; ++i) {
    const std::string& name = input.fields[i].name;
    result.field_outputs[i].name = name;
    result.field_outputs[i].grid = field_grids[name];
  }

  if (!result.field_outputs.empty()) {
    result.grid = result.field_outputs[0].grid;
  }

  return result;
}
