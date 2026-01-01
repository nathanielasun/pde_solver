#include "coordinate_system_registry.h"
#include <algorithm>
#include <cmath>

CoordinateSystemRegistry& CoordinateSystemRegistry::Instance() {
  static CoordinateSystemRegistry instance;
  if (!instance.initialized_) {
    instance.InitializeBuiltInSystems();
    instance.initialized_ = true;
  }
  return instance;
}

void CoordinateSystemRegistry::Register(CoordinateSystem id, const CoordinateSystemMetadata& metadata) {
  systems_[id] = metadata;
}

std::vector<CoordinateSystem> CoordinateSystemRegistry::GetSystems() const {
  std::vector<CoordinateSystem> result;
  result.reserve(systems_.size());
  for (const auto& pair : systems_) {
    result.push_back(pair.first);
  }
  return result;
}

const CoordinateSystemMetadata* CoordinateSystemRegistry::GetMetadata(CoordinateSystem id) const {
  auto it = systems_.find(id);
  if (it != systems_.end()) {
    return &it->second;
  }
  return nullptr;
}

bool CoordinateSystemRegistry::ValidateBounds(CoordinateSystem id, double x, double y, double z) const {
  const CoordinateSystemMetadata* metadata = GetMetadata(id);
  if (!metadata || !metadata->validate_bounds) {
    return true;  // Default: accept all bounds
  }
  return metadata->validate_bounds(x, y, z);
}

const CoordinateSystemAxis* CoordinateSystemRegistry::GetAxis(CoordinateSystem id, int axis_index) const {
  const CoordinateSystemMetadata* metadata = GetMetadata(id);
  if (!metadata || axis_index < 0 || axis_index >= static_cast<int>(metadata->axes.size())) {
    return nullptr;
  }
  return &metadata->axes[axis_index];
}

int CoordinateSystemRegistry::GetAxisCount(CoordinateSystem id) const {
  const CoordinateSystemMetadata* metadata = GetMetadata(id);
  if (!metadata) {
    return 0;
  }
  return static_cast<int>(metadata->axes.size());
}

void CoordinateSystemRegistry::InitializeBuiltInSystems() {
  // Cartesian 2D: (x, y)
  CoordinateSystemMetadata cartesian_2d;
  cartesian_2d.id = CoordinateSystem::Cartesian;
  cartesian_2d.name = "Cartesian (2D)";
  cartesian_2d.description = "Standard rectangular coordinates (x, y). Most common coordinate system.";
  cartesian_2d.dimension = 2;
  cartesian_2d.axes = {
    {"X", "x", "m", -1.0, 1.0, false, false},
    {"Y", "y", "m", -1.0, 1.0, false, false}
  };
  cartesian_2d.validate_bounds = [](double x, double y, double z) {
    return true;  // No restrictions
  };
  cartesian_2d.example_domain = "x: [-1, 1], y: [-1, 1]";
  cartesian_2d.common_applications = {"Rectangular domains", "General purpose", "Most PDEs"};
  Register(CoordinateSystem::Cartesian, cartesian_2d);
  
  // Cartesian 3D: (x, y, z)
  CoordinateSystemMetadata cartesian_3d;
  cartesian_3d.id = CoordinateSystem::Cartesian;
  cartesian_3d.name = "Cartesian (3D)";
  cartesian_3d.description = "Standard rectangular coordinates (x, y, z). Most common 3D coordinate system.";
  cartesian_3d.dimension = 3;
  cartesian_3d.axes = {
    {"X", "x", "m", -1.0, 1.0, false, false},
    {"Y", "y", "m", -1.0, 1.0, false, false},
    {"Z", "z", "m", -1.0, 1.0, false, false}
  };
  cartesian_3d.validate_bounds = [](double x, double y, double z) {
    return true;  // No restrictions
  };
  cartesian_3d.example_domain = "x: [-1, 1], y: [-1, 1], z: [-1, 1]";
  cartesian_3d.common_applications = {"Rectangular domains", "General purpose", "Most 3D PDEs"};
  // Note: Cartesian is already registered, but we can extend it for 3D
  
  // Polar: (r, θ)
  CoordinateSystemMetadata polar;
  polar.id = CoordinateSystem::Polar;
  polar.name = "Polar";
  polar.description = "Circular coordinates (r, θ). r is radius, θ is azimuth angle. Useful for circular domains.";
  polar.dimension = 2;
  polar.axes = {
    {"Radius", "r", "m", 0.0, 1.0, false, false},
    {"Azimuth", "θ", "rad", 0.0, 2.0 * M_PI, true, true}  // Periodic, angular
  };
  polar.validate_bounds = [](double r, double theta, double z) {
    return r >= 0.0;  // Radius must be non-negative
  };
  polar.example_domain = "r: [0, 1], θ: [0, 2π]";
  polar.common_applications = {"Circular domains", "Radial symmetry", "Polar problems"};
  Register(CoordinateSystem::Polar, polar);
  
  // Axisymmetric: (r, z)
  CoordinateSystemMetadata axisymmetric;
  axisymmetric.id = CoordinateSystem::Axisymmetric;
  axisymmetric.name = "Axisymmetric (Cylindrical)";
  axisymmetric.description = "Cylindrical coordinates with rotational symmetry (r, z). r is radius, z is height.";
  axisymmetric.dimension = 2;
  axisymmetric.axes = {
    {"Radius", "r", "m", 0.0, 1.0, false, false},
    {"Height", "z", "m", -1.0, 1.0, false, false}
  };
  axisymmetric.validate_bounds = [](double r, double z, double unused) {
    return r >= 0.0;  // Radius must be non-negative
  };
  axisymmetric.example_domain = "r: [0, 1], z: [-1, 1]";
  axisymmetric.common_applications = {"Cylindrical symmetry", "Rotational problems", "Pipes and tubes"};
  Register(CoordinateSystem::Axisymmetric, axisymmetric);
  
  // Cylindrical: (r, θ, z)
  CoordinateSystemMetadata cylindrical;
  cylindrical.id = CoordinateSystem::Cylindrical;
  cylindrical.name = "Cylindrical";
  cylindrical.description = "Full 3D cylindrical coordinates (r, θ, z). r is radius, θ is azimuth, z is height.";
  cylindrical.dimension = 3;
  cylindrical.axes = {
    {"Radius", "r", "m", 0.0, 1.0, false, false},
    {"Azimuth", "θ", "rad", 0.0, 2.0 * M_PI, true, true},  // Periodic, angular
    {"Height", "z", "m", -1.0, 1.0, false, false}
  };
  cylindrical.validate_bounds = [](double r, double theta, double z) {
    return r >= 0.0;  // Radius must be non-negative
  };
  cylindrical.example_domain = "r: [0, 1], θ: [0, 2π], z: [-1, 1]";
  cylindrical.common_applications = {"Cylindrical domains", "Pipes", "Rotational symmetry"};
  Register(CoordinateSystem::Cylindrical, cylindrical);
  
  // Spherical Surface: (θ, φ)
  CoordinateSystemMetadata spherical_surface;
  spherical_surface.id = CoordinateSystem::SphericalSurface;
  spherical_surface.name = "Spherical Surface";
  spherical_surface.description = "2D surface coordinates on a sphere (θ, φ). θ is polar angle, φ is azimuth.";
  spherical_surface.dimension = 2;
  spherical_surface.axes = {
    {"Polar Angle", "θ", "rad", 0.0, M_PI, false, true},  // Angular, 0 to π
    {"Azimuth", "φ", "rad", 0.0, 2.0 * M_PI, true, true}  // Periodic, angular
  };
  spherical_surface.validate_bounds = [](double theta, double phi, double unused) {
    return theta >= 0.0 && theta <= M_PI;  // Polar angle must be in [0, π]
  };
  spherical_surface.example_domain = "θ: [0, π], φ: [0, 2π]";
  spherical_surface.common_applications = {"Spherical surfaces", "Planetary problems", "Surface PDEs"};
  Register(CoordinateSystem::SphericalSurface, spherical_surface);
  
  // Spherical Volume: (r, θ, φ)
  CoordinateSystemMetadata spherical_volume;
  spherical_volume.id = CoordinateSystem::SphericalVolume;
  spherical_volume.name = "Spherical Volume";
  spherical_volume.description = "3D spherical coordinates (r, θ, φ). r is radius, θ is polar angle, φ is azimuth.";
  spherical_volume.dimension = 3;
  spherical_volume.axes = {
    {"Radius", "r", "m", 0.0, 1.0, false, false},
    {"Polar Angle", "θ", "rad", 0.0, M_PI, false, true},  // Angular, 0 to π
    {"Azimuth", "φ", "rad", 0.0, 2.0 * M_PI, true, true}  // Periodic, angular
  };
  spherical_volume.validate_bounds = [](double r, double theta, double phi) {
    return r >= 0.0 && theta >= 0.0 && theta <= M_PI;  // Radius non-negative, polar angle in [0, π]
  };
  spherical_volume.example_domain = "r: [0, 1], θ: [0, π], φ: [0, 2π]";
  spherical_volume.common_applications = {"Spherical domains", "Planetary problems", "Radial symmetry"};
  Register(CoordinateSystem::SphericalVolume, spherical_volume);
  
  // Toroidal Surface: (θ, φ)
  CoordinateSystemMetadata toroidal_surface;
  toroidal_surface.id = CoordinateSystem::ToroidalSurface;
  toroidal_surface.name = "Toroidal Surface";
  toroidal_surface.description = "2D surface coordinates on a torus (θ, φ). Both angles are periodic.";
  toroidal_surface.dimension = 2;
  toroidal_surface.axes = {
    {"Poloidal Angle", "θ", "rad", 0.0, 2.0 * M_PI, true, true},  // Periodic, angular
    {"Toroidal Angle", "φ", "rad", 0.0, 2.0 * M_PI, true, true}   // Periodic, angular
  };
  toroidal_surface.validate_bounds = [](double theta, double phi, double unused) {
    return true;  // Both angles are periodic, no restrictions
  };
  toroidal_surface.example_domain = "θ: [0, 2π], φ: [0, 2π]";
  toroidal_surface.common_applications = {"Torus surfaces", "Fusion plasma", "Topological problems"};
  Register(CoordinateSystem::ToroidalSurface, toroidal_surface);
  
  // Toroidal Volume: (r, θ, φ)
  CoordinateSystemMetadata toroidal_volume;
  toroidal_volume.id = CoordinateSystem::ToroidalVolume;
  toroidal_volume.name = "Toroidal Volume";
  toroidal_volume.description = "3D toroidal coordinates (r, θ, φ). r is minor radius, θ and φ are angles.";
  toroidal_volume.dimension = 3;
  toroidal_volume.axes = {
    {"Minor Radius", "r", "m", 0.0, 1.0, false, false},
    {"Poloidal Angle", "θ", "rad", 0.0, 2.0 * M_PI, true, true},  // Periodic, angular
    {"Toroidal Angle", "φ", "rad", 0.0, 2.0 * M_PI, true, true}   // Periodic, angular
  };
  toroidal_volume.validate_bounds = [](double r, double theta, double phi) {
    return r >= 0.0;  // Minor radius must be non-negative
  };
  toroidal_volume.example_domain = "r: [0, 1], θ: [0, 2π], φ: [0, 2π]";
  toroidal_volume.common_applications = {"Torus volumes", "Fusion plasma", "Magnetic confinement"};
  Register(CoordinateSystem::ToroidalVolume, toroidal_volume);
}

