#ifndef PDE_TYPE_REGISTRY_H
#define PDE_TYPE_REGISTRY_H

#include <string>
#include <vector>
#include <map>
#include <functional>
#include "pde_types.h"
#include "latex_parser.h"

struct PDETypeMetadata {
  std::string name;
  std::string description;
  std::vector<std::string> required_terms;      // e.g., ["u_xx", "u_yy"]
  std::vector<std::string> optional_terms;      // e.g., ["u_t", "u_x", "u_y"]
  std::function<bool(const LatexParseResult&)> validator;
  std::function<SolveInput(const LatexParseResult&)> input_builder;
  std::map<std::string, std::string> default_bcs;  // e.g., {"left": "u=0", "right": "u=0"}
  CoordinateSystem default_coord_system = CoordinateSystem::Cartesian;
  std::string example_latex;  // Example LaTeX expression for this PDE type
  std::vector<std::string> common_applications;  // e.g., ["Electrostatics", "Fluid Flow"]
};

class PDETypeRegistry {
 public:
  static PDETypeRegistry& Instance();
  
  // Register a new PDE type
  void Register(const std::string& type_id, const PDETypeMetadata& metadata);
  
  // Get all registered type IDs
  std::vector<std::string> GetTypes() const;
  
  // Get metadata for a specific type
  const PDETypeMetadata* GetMetadata(const std::string& type_id) const;
  
  // Validate a parsed PDE against a type
  bool Validate(const std::string& type_id, const LatexParseResult& parse_result) const;
  
  // Detect PDE type from parse result (heuristic)
  std::string DetectType(const LatexParseResult& parse_result) const;
  
  // Build SolveInput from parse result using type metadata
  SolveInput BuildInput(const std::string& type_id, const LatexParseResult& parse_result) const;
  
  // Initialize with built-in PDE types
  void InitializeBuiltInTypes();

 private:
  PDETypeRegistry() = default;
  ~PDETypeRegistry() = default;
  PDETypeRegistry(const PDETypeRegistry&) = delete;
  PDETypeRegistry& operator=(const PDETypeRegistry&) = delete;
  
  std::map<std::string, PDETypeMetadata> types_;
  bool initialized_ = false;
};

#endif  // PDE_TYPE_REGISTRY_H

