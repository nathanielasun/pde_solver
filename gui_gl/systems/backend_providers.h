#ifndef BACKEND_PROVIDERS_H
#define BACKEND_PROVIDERS_H

#include "backend_capabilities.h"
#include "backend.h"
#include <memory>

// CPU Backend UI Provider
class CPUBackendProvider : public BackendUIProvider {
 public:
  BackendCapabilities GetCapabilities() const override;
  void RenderOptionsUI() override;
  std::string GetBackendName() const override { return "CPU"; }
  std::string GetBackendDescription() const override {
    return "CPU solver with full feature support";
  }
  bool IsAvailable() const override { return true; }
};

// CUDA Backend UI Provider
class CUDABackendProvider : public BackendUIProvider {
 public:
  BackendCapabilities GetCapabilities() const override;
  void RenderOptionsUI() override;
  std::string GetBackendName() const override { return "CUDA"; }
  std::string GetBackendDescription() const override {
    return "CUDA GPU acceleration (NVIDIA GPUs)";
  }
  bool IsAvailable() const override;
};

// Metal Backend UI Provider
class MetalBackendProvider : public BackendUIProvider {
 public:
  BackendCapabilities GetCapabilities() const override;
  void RenderOptionsUI() override;
  std::string GetBackendName() const override { return "Metal"; }
  std::string GetBackendDescription() const override {
    return "Metal GPU acceleration (Apple Silicon/GPUs)";
  }
  bool IsAvailable() const override;
};

// Helper function to initialize all backend providers
void InitializeBackendProviders(BackendUIRegistry& registry);

#endif  // BACKEND_PROVIDERS_H

