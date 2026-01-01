#include "backend.h"

#include <string>
#include <vector>

#ifdef USE_CUDA
#include "cuda_solver.h"
#endif
#ifdef USE_METAL
#include "metal_solver.h"
#endif

namespace {
BackendStatus MakeStatus(BackendKind kind, const std::string& name, bool available,
                         const std::string& note) {
  BackendStatus status;
  status.kind = kind;
  status.name = name;
  status.available = available;
  status.note = note;
  return status;
}
}  // namespace

BackendKind ParseBackendKind(const std::string& text) {
  if (text == "cpu") {
    return BackendKind::CPU;
  }
  if (text == "cuda") {
    return BackendKind::CUDA;
  }
  if (text == "metal") {
    return BackendKind::Metal;
  }
  if (text == "tpu") {
    return BackendKind::TPU;
  }
  return BackendKind::Auto;
}

std::string BackendKindName(BackendKind kind) {
  switch (kind) {
    case BackendKind::CPU:
      return "CPU";
    case BackendKind::CUDA:
      return "CUDA";
    case BackendKind::Metal:
      return "Metal";
    case BackendKind::TPU:
      return "TPU";
    case BackendKind::Auto:
    default:
      return "Auto";
  }
}

std::vector<BackendStatus> DetectBackends() {
  std::vector<BackendStatus> statuses;
  statuses.push_back(MakeStatus(BackendKind::CPU, "CPU", true, "available"));

#ifdef USE_CUDA
  std::string cuda_note;
  const bool cuda_available = CudaIsAvailable(&cuda_note);
  statuses.push_back(MakeStatus(BackendKind::CUDA, "CUDA", cuda_available,
                                cuda_note.empty() ? "compiled with CUDA support" : cuda_note));
#else
  statuses.push_back(MakeStatus(BackendKind::CUDA, "CUDA", false,
                                "not compiled with CUDA support"));
#endif

#ifdef USE_METAL
  std::string metal_note;
  const bool metal_available = MetalIsAvailable(&metal_note);
  statuses.push_back(MakeStatus(BackendKind::Metal, "Metal", metal_available,
                                metal_note.empty() ? "compiled with Metal support" : metal_note));
#else
  statuses.push_back(MakeStatus(BackendKind::Metal, "Metal", false,
                                "not compiled with Metal support"));
#endif

#ifdef USE_TPU
  statuses.push_back(MakeStatus(BackendKind::TPU, "TPU", true,
                                "compiled with TPU support"));
#else
  statuses.push_back(MakeStatus(BackendKind::TPU, "TPU", false,
                                "not compiled with TPU support"));
#endif

  return statuses;
}

