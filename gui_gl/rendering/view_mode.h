#ifndef VIEW_MODE_H
#define VIEW_MODE_H

// View modes for 3D visualization - shared between GlViewer and grid building
enum class ViewMode {
  Cartesian,
  Polar,
  Axisymmetric,
  CylindricalVolume,
  SphericalSurface,
  SphericalVolume,
  ToroidalSurface,
  ToroidalVolume,
};

#endif  // VIEW_MODE_H
