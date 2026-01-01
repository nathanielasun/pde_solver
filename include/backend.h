#ifndef BACKEND_H
#define BACKEND_H

#include <string>
#include <vector>

#include "pde_types.h"
#include "progress.h"

enum class BackendKind {
  Auto,
  CPU,
  CUDA,
  Metal,
  TPU,
};

struct BackendStatus {
  BackendKind kind = BackendKind::CPU;
  std::string name;
  bool available = false;
  std::string note;
};

BackendKind ParseBackendKind(const std::string& text);
std::string BackendKindName(BackendKind kind);
std::vector<BackendStatus> DetectBackends();
SolveOutput SolveWithBackend(const SolveInput& input, BackendKind requested,
                             BackendKind* selected, std::string* selection_note,
                             const ProgressCallback& progress = ProgressCallback());

#endif
