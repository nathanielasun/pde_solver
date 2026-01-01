#include "backend.h"

#include <algorithm>
#include <cmath>
#include <utility>
#include <vector>

#include "residual.h"
#include "solver.h"
#include "backend_capability_matrix.h"

#ifdef USE_CUDA
#include "backends/cuda/CudaSolve.h"
#endif
#ifdef USE_METAL
#include "backends/metal/MetalSolve.h"
#endif

namespace {
const BackendStatus* FindStatus(const std::vector<BackendStatus>& statuses, BackendKind kind) {
  auto it = std::find_if(statuses.begin(), statuses.end(), [&](const BackendStatus& status) {
    return status.kind == kind;
  });
  if (it == statuses.end()) {
    return nullptr;
  }
  return &(*it);
}

void MaybeAttachResidualHistory(SolveOutput* out, std::vector<int>* iters,
                                std::vector<double>* l2, std::vector<double>* linf) {
  if (!out || (!iters->empty() && (l2->size() == iters->size()) &&
               (linf->size() == iters->size()))) {
    out->residual_iters = std::move(*iters);
    out->residual_l2_history = std::move(*l2);
    out->residual_linf_history = std::move(*linf);
  }
}

void MaybeComputeResiduals(const SolveInput& input, SolveOutput* out) {
  if (!out || !out->error.empty() || out->grid.empty()) {
    return;
  }
  ResidualNorms norms;
  std::string err;
  if (ComputeResidualNorms(input, out->grid, &norms, &err)) {
    out->residual_l2 = norms.l2;
    out->residual_linf = norms.linf;
  }
}
}  // namespace

SolveOutput SolveWithBackend(const SolveInput& input, BackendKind requested,
                             BackendKind* selected, std::string* selection_note,
                             const ProgressCallback& progress) {
  // Wrap progress to optionally record residual history.
  std::vector<double> residual_l2_hist;
  std::vector<double> residual_linf_hist;
  std::vector<int> residual_iters;
  int residual_sample = 0;
  ProgressCallback wrapped_progress = progress;
  if (input.solver.residual_interval > 0) {
    wrapped_progress = [&](const std::string& phase, double value) {
      if (phase == "residual_l2") {
        residual_l2_hist.push_back(value);
        residual_iters.push_back(residual_sample);
        return;
      }
      if (phase == "residual_linf") {
        residual_linf_hist.push_back(value);
        residual_sample++;
        return;
      }
      if (progress) {
        progress(phase, value);
      }
    };
  }

  if (std::abs(input.pde.ut) > 1e-12 || std::abs(input.pde.utt) > 1e-12) {
    return {"time-dependent PDEs require the time-series solver", {}};
  }

  const auto statuses = DetectBackends();
  std::string extra_note;
  BackendKind choice = requested;
  auto supports_backend = [&](BackendKind kind, std::string* reason) {
    return BackendSupportsMethodForInput(kind, input.solver.method, input, reason);
  };

  if (choice == BackendKind::Auto) {
    const bool interactive = static_cast<bool>(progress);
    const BackendKind order_interactive[] = {BackendKind::CPU, BackendKind::CUDA,
                                             BackendKind::Metal, BackendKind::TPU};
    const BackendKind order_default[] = {BackendKind::CUDA, BackendKind::Metal,
                                         BackendKind::TPU, BackendKind::CPU};
    const BackendKind* order = interactive ? order_interactive : order_default;
    const int order_len = 4;
    std::string last_reason;
    for (int i = 0; i < order_len; ++i) {
      const BackendStatus* status = FindStatus(statuses, order[i]);
      if (!status || !status->available) {
        continue;
      }
      std::string compat_reason;
      if (!supports_backend(order[i], &compat_reason)) {
        last_reason = compat_reason;
        continue;
      }
      choice = order[i];
      if (interactive && choice == BackendKind::CPU && extra_note.empty()) {
        extra_note = "auto: preferring CPU for GUI responsiveness";
      }
      break;
    }
    if (choice == BackendKind::Auto) {
      const std::string detail = last_reason.empty() ? "no compatible backend available" : last_reason;
      return {"no available backend supports the selected solve: " + detail, {}};
    }
  }

  std::string compat_reason;
  if (!supports_backend(choice, &compat_reason)) {
    if (choice != BackendKind::CPU) {
      if (!extra_note.empty()) {
        extra_note += "; ";
      }
      extra_note += compat_reason.empty() ? "requested backend unsupported" : compat_reason;
      choice = BackendKind::CPU;
      compat_reason.clear();
      if (!supports_backend(choice, &compat_reason)) {
        return {"requested solve not supported on any backend: " + compat_reason, {}};
      }
    } else {
      return {"requested solve not supported on CPU: " + compat_reason, {}};
    }
  }

  const BackendStatus* status = FindStatus(statuses, choice);
  if (!status || !status->available) {
    std::string error = "requested backend unavailable: " + BackendKindName(choice);
    return {error, {}};
  }

  if (selected) {
    *selected = choice;
  }
  if (selection_note) {
    *selection_note = status->note;
    if (!extra_note.empty()) {
      if (!selection_note->empty()) {
        *selection_note += "; " + extra_note;
      } else {
        *selection_note = extra_note;
      }
    }
  }

  SolveOutput out;
  switch (choice) {
    case BackendKind::CPU: {
      out = SolvePDE(input, wrapped_progress);
      MaybeComputeResiduals(input, &out);
      break;
    }
    case BackendKind::CUDA: {
#ifdef USE_CUDA
      if (wrapped_progress) {
        wrapped_progress("solve", 0.0);
      }
      out = SolvePDECuda(input);
      if (wrapped_progress) {
        wrapped_progress("solve", 1.0);
      }
      MaybeComputeResiduals(input, &out);
#else
      out = {"CUDA backend not compiled", {}};
#endif
      break;
    }
    case BackendKind::Metal: {
#ifdef USE_METAL
      if (wrapped_progress) {
        wrapped_progress("solve", 0.0);
      }
      out = SolvePDEMetal(input);
      if (wrapped_progress) {
        wrapped_progress("solve", 1.0);
      }
      MaybeComputeResiduals(input, &out);
#else
      out = {"Metal backend not compiled", {}};
#endif
      break;
    }
    case BackendKind::TPU: {
      if (selection_note) {
        *selection_note = status->note + "; using CPU fallback";
      }
      out = SolvePDE(input, progress);
      break;
    }
    case BackendKind::Auto:
    default:
      out = SolvePDE(input, progress);
      break;
  }

  MaybeAttachResidualHistory(&out, &residual_iters, &residual_l2_hist, &residual_linf_hist);
  return out;
}
