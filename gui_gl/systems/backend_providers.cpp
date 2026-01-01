#include "backend_providers.h"
#include "imgui.h"
#include "backend.h"

#ifdef USE_CUDA
#include "cuda_solver.h"
#endif

#ifdef USE_METAL
#include "metal_solver.h"
#endif

BackendCapabilities CPUBackendProvider::GetCapabilities() const {
  return GetBackendCapabilities(BackendKind::CPU);
}

void CPUBackendProvider::RenderOptionsUI() {
  const BackendCapabilities caps = GetCapabilities();
  ImGui::TextDisabled("CPU backend capabilities:");
  ImGui::BulletText("3D domains: %s", caps.supports_3d ? "Supported" : "Not supported");
  ImGui::BulletText("Spatial RHS: %s", caps.supports_spatial_rhs ? "Supported" : "Not supported");
  ImGui::BulletText("Nonlinear terms: %s", caps.supports_nonlinear ? "Supported" : "Not supported");
  ImGui::BulletText("Integral terms: %s", caps.supports_integrals ? "Supported" : "Not supported");
}

BackendCapabilities CUDABackendProvider::GetCapabilities() const {
  return GetBackendCapabilities(BackendKind::CUDA);
}

void CUDABackendProvider::RenderOptionsUI() {
  const BackendCapabilities caps = GetCapabilities();
  ImGui::TextDisabled("CUDA backend capabilities:");
  ImGui::BulletText("3D domains: %s", caps.supports_3d ? "Supported" : "Not supported");
  ImGui::BulletText("Spatial RHS: %s", caps.supports_spatial_rhs ? "Supported" : "Not supported");
  ImGui::BulletText("Nonlinear terms: %s", caps.supports_nonlinear ? "Supported" : "Not supported");
  ImGui::BulletText("Integral terms: %s", caps.supports_integrals ? "Supported" : "Not supported");
  ImGui::BulletText("Implicit shapes: %s", caps.supports_shapes ? "Supported" : "Not supported");
}

bool CUDABackendProvider::IsAvailable() const {
#ifdef USE_CUDA
  std::string note;
  return CudaIsAvailable(&note);
#else
  return false;
#endif
}

BackendCapabilities MetalBackendProvider::GetCapabilities() const {
  return GetBackendCapabilities(BackendKind::Metal);
}

void MetalBackendProvider::RenderOptionsUI() {
  const BackendCapabilities caps = GetCapabilities();
  ImGui::TextDisabled("Metal backend capabilities:");
  ImGui::BulletText("3D domains: %s", caps.supports_3d ? "Supported" : "Not supported");
  ImGui::BulletText("Spatial RHS: %s", caps.supports_spatial_rhs ? "Supported" : "Not supported");
  ImGui::BulletText("Nonlinear terms: %s", caps.supports_nonlinear ? "Supported" : "Not supported");
  ImGui::BulletText("Integral terms: %s", caps.supports_integrals ? "Supported" : "Not supported");
  ImGui::BulletText("Implicit shapes: %s", caps.supports_shapes ? "Supported" : "Not supported");
}

bool MetalBackendProvider::IsAvailable() const {
#ifdef USE_METAL
  std::string note;
  return MetalIsAvailable(&note);
#else
  return false;
#endif
}

void InitializeBackendProviders(BackendUIRegistry& registry) {
  // Always register CPU (always available)
  registry.Register(std::make_unique<CPUBackendProvider>());
  
  // Register CUDA if available
  auto cuda_provider = std::make_unique<CUDABackendProvider>();
  if (cuda_provider->IsAvailable()) {
    registry.Register(std::move(cuda_provider));
  }
  
  // Register Metal if available
  auto metal_provider = std::make_unique<MetalBackendProvider>();
  if (metal_provider->IsAvailable()) {
    registry.Register(std::move(metal_provider));
  }
}
