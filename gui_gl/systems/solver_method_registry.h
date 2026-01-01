#ifndef SOLVER_METHOD_REGISTRY_H
#define SOLVER_METHOD_REGISTRY_H

#include "pde_types.h"
#include "backend.h"
#include <string>
#include <vector>
#include <map>
#include <functional>
#include <any>

struct SolverMethodOption {
  std::string name;
  std::string type;  // "double", "int", "bool", "string"
  std::string description;
  std::any default_value;
  std::any min_value;  // Optional, for numeric types
  std::any max_value;  // Optional, for numeric types
};

struct SolverMethodMetadata {
  SolveMethod id;
  std::string name;
  std::string short_name;  // e.g., "jacobi", "gs", "sor"
  std::string description;
  std::string detailed_description;
  std::vector<BackendKind> supported_backends;
  std::vector<SolverMethodOption> options;
  std::function<bool(const SolveInput&)> is_applicable;
  std::string recommendation_reason;  // Why this method might be recommended
  std::vector<std::string> best_use_cases;
  std::vector<std::string> limitations;
  int default_max_iter = 10000;
  double default_tol = 1e-6;
};

class SolverMethodRegistry {
 public:
  static SolverMethodRegistry& Instance();
  
  // Register a new solver method
  void Register(const SolverMethodMetadata& metadata);
  
  // Get all registered solver methods
  std::vector<SolveMethod> GetMethods() const;
  
  // Get metadata for a specific method
  const SolverMethodMetadata* GetMetadata(SolveMethod method) const;
  
  // Get methods supported by a specific backend
  std::vector<SolveMethod> GetMethodsForBackend(BackendKind backend) const;
  
  // Check if a method is applicable to a given problem
  bool IsApplicable(SolveMethod method, const SolveInput& input) const;
  
  // Get recommended method for a problem (simple heuristic)
  SolveMethod RecommendMethod(const SolveInput& input, BackendKind preferred_backend) const;
  
  // Initialize with built-in methods
  void InitializeBuiltInMethods();

 private:
  SolverMethodRegistry() = default;
  ~SolverMethodRegistry() = default;
  SolverMethodRegistry(const SolverMethodRegistry&) = delete;
  SolverMethodRegistry& operator=(const SolverMethodRegistry&) = delete;
  
  std::map<SolveMethod, SolverMethodMetadata> methods_;
  bool initialized_ = false;
};

#endif  // SOLVER_METHOD_REGISTRY_H

