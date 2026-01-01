#include "backend_capabilities.h"
#include <algorithm>
#include <sstream>

std::string BackendCapabilities::GetDescription() const {
  std::ostringstream oss;
  oss << "Methods: ";
  for (size_t i = 0; i < supported_methods.size(); ++i) {
    if (i > 0) oss << ", ";
    switch (supported_methods[i]) {
      case SolveMethod::Jacobi: oss << "Jacobi"; break;
      case SolveMethod::GaussSeidel: oss << "Gauss-Seidel"; break;
      case SolveMethod::SOR: oss << "SOR"; break;
      case SolveMethod::CG: oss << "CG"; break;
      case SolveMethod::BiCGStab: oss << "BiCGStab"; break;
      case SolveMethod::GMRES: oss << "GMRES"; break;
      case SolveMethod::MultigridVcycle: oss << "Multigrid"; break;
    }
  }
  oss << " | 3D: " << (supports_3d ? "Yes" : "No");
  oss << " | Spatial RHS: " << (supports_spatial_rhs ? "Yes" : "No");
  oss << " | Nonlinear: " << (supports_nonlinear ? "Yes" : "No");
  return oss.str();
}

void BackendUIRegistry::Register(std::unique_ptr<BackendUIProvider> provider) {
  if (!provider) {
    return;
  }
  std::string name = provider->GetBackendName();
  providers_[name] = std::move(provider);
  
  // Map BackendKind to name (if we can determine it)
  // This is a simplified mapping - in practice, you'd want a more robust system
  if (name == "CPU") {
    kind_to_name_[BackendKind::CPU] = name;
  } else if (name == "CUDA") {
    kind_to_name_[BackendKind::CUDA] = name;
  } else if (name == "Metal") {
    kind_to_name_[BackendKind::Metal] = name;
  } else if (name == "TPU") {
    kind_to_name_[BackendKind::TPU] = name;
  }
}

std::vector<std::string> BackendUIRegistry::GetBackendNames() const {
  std::vector<std::string> names;
  names.reserve(providers_.size());
  for (const auto& [name, provider] : providers_) {
    names.push_back(name);
  }
  return names;
}

BackendUIProvider* BackendUIRegistry::GetProvider(const std::string& name) {
  auto it = providers_.find(name);
  if (it != providers_.end()) {
    return it->second.get();
  }
  return nullptr;
}

BackendUIProvider* BackendUIRegistry::GetProvider(BackendKind kind) {
  auto it = kind_to_name_.find(kind);
  if (it != kind_to_name_.end()) {
    return GetProvider(it->second);
  }
  return nullptr;
}

BackendCapabilities BackendUIRegistry::GetCapabilities(const std::string& name) const {
  auto it = providers_.find(name);
  if (it != providers_.end() && it->second) {
    return it->second->GetCapabilities();
  }
  return BackendCapabilities{};  // Empty capabilities
}

BackendCapabilities BackendUIRegistry::GetCapabilities(BackendKind kind) const {
  auto it = kind_to_name_.find(kind);
  if (it != kind_to_name_.end()) {
    return GetCapabilities(it->second);
  }
  return BackendCapabilities{};  // Empty capabilities
}

std::vector<std::string> BackendUIRegistry::GetAvailableBackends() const {
  std::vector<std::string> available;
  for (const auto& [name, provider] : providers_) {
    if (provider && provider->IsAvailable()) {
      available.push_back(name);
    }
  }
  return available;
}

std::vector<SolveMethod> BackendUIRegistry::GetSupportedMethods(const std::string& name) const {
  return GetCapabilities(name).supported_methods;
}

std::vector<SolveMethod> BackendUIRegistry::GetSupportedMethods(BackendKind kind) const {
  return GetCapabilities(kind).supported_methods;
}

