#ifndef COORDINATE_UTILS_H
#define COORDINATE_UTILS_H

#include "GlViewer.h"
#include "pde_types.h"
#include <string>

// Coordinate system constants
namespace CoordMode {
  constexpr int kCartesian2D = 0;
  constexpr int kCartesian3D = 1;
  constexpr int kPolar = 2;
  constexpr int kAxisymmetric = 3;
  constexpr int kCylindricalVolume = 4;
  constexpr int kSphericalSurface = 5;
  constexpr int kSphericalVolume = 6;
  constexpr int kToroidalSurface = 7;
  constexpr int kToroidalVolume = 8;
}

// Coordinate system helpers
GlViewer::ViewMode ViewModeForCoord(int coord_mode);

// Convert coord_mode to CoordinateSystem enum
CoordinateSystem CoordModeToSystem(int coord_mode);

// Axis information for slice planes
struct AxisInfo {
  const char* labels[3];
  int count;
  double min;
  double max;
};

// Compute axis information for slice planes
AxisInfo ComputeAxisInfo(int axis_index, double xmin, double xmax, 
                         double ymin, double ymax, double zmin, double zmax,
                         bool use_volume);

#endif // COORDINATE_UTILS_H

