#ifndef COORDINATE_SYSTEM_REGISTRY_H
#define COORDINATE_SYSTEM_REGISTRY_H

#include "pde_types.h"
#include <string>
#include <vector>
#include <map>
#include <functional>

struct CoordinateSystemAxis {
  std::string name;        // e.g., "Radius", "Azimuth", "Polar Angle"
  std::string symbol;     // e.g., "r", "θ", "φ"
  std::string unit;        // e.g., "m", "rad", "deg"
  double default_min = 0.0;
  double default_max = 1.0;
  bool is_periodic = false;
  bool is_angular = false;  // For angles that wrap around
};

struct CoordinateSystemMetadata {
  CoordinateSystem id;
  std::string name;
  std::string description;
  int dimension;  // 2 or 3
  std::vector<CoordinateSystemAxis> axes;
  std::function<bool(double, double, double)> validate_bounds;  // (x, y, z) -> valid
  std::string example_domain;  // Example domain bounds string
  std::vector<std::string> common_applications;  // e.g., ["Circular domains", "Cylindrical symmetry"]
};

class CoordinateSystemRegistry {
 public:
  static CoordinateSystemRegistry& Instance();
  
  // Register a new coordinate system
  void Register(CoordinateSystem id, const CoordinateSystemMetadata& metadata);
  
  // Get all registered coordinate systems
  std::vector<CoordinateSystem> GetSystems() const;
  
  // Get metadata for a specific coordinate system
  const CoordinateSystemMetadata* GetMetadata(CoordinateSystem id) const;
  
  // Validate bounds for a coordinate system
  bool ValidateBounds(CoordinateSystem id, double x, double y, double z = 0.0) const;
  
  // Get axis information for a coordinate system
  const CoordinateSystemAxis* GetAxis(CoordinateSystem id, int axis_index) const;
  
  // Get number of axes for a coordinate system
  int GetAxisCount(CoordinateSystem id) const;
  
  // Initialize with built-in coordinate systems
  void InitializeBuiltInSystems();

 private:
  CoordinateSystemRegistry() = default;
  ~CoordinateSystemRegistry() = default;
  CoordinateSystemRegistry(const CoordinateSystemRegistry&) = delete;
  CoordinateSystemRegistry& operator=(const CoordinateSystemRegistry&) = delete;
  
  std::map<CoordinateSystem, CoordinateSystemMetadata> systems_;
  bool initialized_ = false;
};

#endif  // COORDINATE_SYSTEM_REGISTRY_H

