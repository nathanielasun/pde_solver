#ifndef BACKEND_CAPABILITIES_H
#define BACKEND_CAPABILITIES_H

#include "backend_capability_matrix.h"
#include <map>
#include <memory>
#include <string>
#include <vector>

// Backend UI provider interface
class BackendUIProvider {
 public:
  virtual ~BackendUIProvider() = default;
  
  // Get backend capabilities
  virtual BackendCapabilities GetCapabilities() const = 0;
  
  // Render backend-specific options UI
  virtual void RenderOptionsUI() = 0;
  
  // Get backend name
  virtual std::string GetBackendName() const = 0;
  
  // Get backend description
  virtual std::string GetBackendDescription() const = 0;
  
  // Check if backend is available
  virtual bool IsAvailable() const = 0;
};

// Backend registry for managing backend UI providers
class BackendUIRegistry {
 public:
  // Register a backend provider
  void Register(std::unique_ptr<BackendUIProvider> provider);
  
  // Get all backend names
  std::vector<std::string> GetBackendNames() const;
  
  // Get backend provider by name
  BackendUIProvider* GetProvider(const std::string& name);
  BackendUIProvider* GetProvider(BackendKind kind);
  
  // Get capabilities for a backend
  BackendCapabilities GetCapabilities(const std::string& name) const;
  BackendCapabilities GetCapabilities(BackendKind kind) const;
  
  // Get available backends (those that are available)
  std::vector<std::string> GetAvailableBackends() const;
  
  // Get supported methods for a backend
  std::vector<SolveMethod> GetSupportedMethods(const std::string& name) const;
  std::vector<SolveMethod> GetSupportedMethods(BackendKind kind) const;

 private:
  std::map<std::string, std::unique_ptr<BackendUIProvider>> providers_;
  std::map<BackendKind, std::string> kind_to_name_;
};

#endif  // BACKEND_CAPABILITIES_H
