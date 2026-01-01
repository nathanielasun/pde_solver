#include "coordinate_utils.h"

#include <algorithm>

GlViewer::ViewMode ViewModeForCoord(int coord_mode) {
  switch (coord_mode) {
    case CoordMode::kCartesian3D:
      return GlViewer::ViewMode::Cartesian;
    case CoordMode::kPolar:
      return GlViewer::ViewMode::Polar;
    case CoordMode::kAxisymmetric:
      return GlViewer::ViewMode::Axisymmetric;
    case CoordMode::kCylindricalVolume:
      return GlViewer::ViewMode::CylindricalVolume;
    case CoordMode::kSphericalSurface:
      return GlViewer::ViewMode::SphericalSurface;
    case CoordMode::kSphericalVolume:
      return GlViewer::ViewMode::SphericalVolume;
    case CoordMode::kToroidalSurface:
      return GlViewer::ViewMode::ToroidalSurface;
    case CoordMode::kToroidalVolume:
      return GlViewer::ViewMode::ToroidalVolume;
    case CoordMode::kCartesian2D:
    default:
      return GlViewer::ViewMode::Cartesian;
  }
}

CoordinateSystem CoordModeToSystem(int coord_mode) {
  switch (coord_mode) {
    case CoordMode::kCartesian2D:
    case CoordMode::kCartesian3D:
      return CoordinateSystem::Cartesian;
    case CoordMode::kPolar:
      return CoordinateSystem::Polar;
    case CoordMode::kAxisymmetric:
      return CoordinateSystem::Axisymmetric;
    case CoordMode::kCylindricalVolume:
      return CoordinateSystem::Cylindrical;
    case CoordMode::kSphericalSurface:
      return CoordinateSystem::SphericalSurface;
    case CoordMode::kSphericalVolume:
      return CoordinateSystem::SphericalVolume;
    case CoordMode::kToroidalSurface:
      return CoordinateSystem::ToroidalSurface;
    case CoordMode::kToroidalVolume:
      return CoordinateSystem::ToroidalVolume;
    default:
      return CoordinateSystem::Cartesian;
  }
}

AxisInfo ComputeAxisInfo(int axis_index, double xmin, double xmax, 
                         double ymin, double ymax, double zmin, double zmax,
                         bool use_volume) {
  AxisInfo info;
  info.labels[0] = "x";
  info.labels[1] = "y";
  info.labels[2] = "z";
  info.count = use_volume ? 3 : 2;
  
  double axis_min = xmin;
  double axis_max = xmax;
  if (axis_index == 1) {
    axis_min = ymin;
    axis_max = ymax;
  } else if (axis_index == 2) {
    axis_min = zmin;
    axis_max = zmax;
  }
  if (axis_max < axis_min) {
    std::swap(axis_min, axis_max);
  }
  if (axis_max - axis_min < 1e-12) {
    axis_max = axis_min + 1.0;
  }
  info.min = axis_min;
  info.max = axis_max;
  return info;
}

