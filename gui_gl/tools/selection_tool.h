#ifndef SELECTION_TOOL_H
#define SELECTION_TOOL_H

#include "pde_types.h"
#include <vector>

class SelectionRegion {
 public:
  virtual ~SelectionRegion() = default;
  virtual bool Contains(double x, double y, double z) const = 0;
};

class BoxRegion : public SelectionRegion {
 public:
  BoxRegion(double xmin, double xmax, double ymin, double ymax, double zmin, double zmax)
      : xmin_(xmin), xmax_(xmax), ymin_(ymin), ymax_(ymax), zmin_(zmin), zmax_(zmax) {}
  bool Contains(double x, double y, double z) const override {
    return x >= xmin_ && x <= xmax_ &&
           y >= ymin_ && y <= ymax_ &&
           z >= zmin_ && z <= zmax_;
  }
 private:
  double xmin_, xmax_, ymin_, ymax_, zmin_, zmax_;
};

class SphereRegion : public SelectionRegion {
 public:
  SphereRegion(double cx, double cy, double cz, double r)
      : cx_(cx), cy_(cy), cz_(cz), r2_(r * r) {}
  bool Contains(double x, double y, double z) const override {
    const double dx = x - cx_;
    const double dy = y - cy_;
    const double dz = z - cz_;
    return dx * dx + dy * dy + dz * dz <= r2_;
  }
 private:
  double cx_, cy_, cz_, r2_;
};

std::vector<bool> CreateMaskFromRegion(const Domain& domain, const SelectionRegion& region);

#endif  // SELECTION_TOOL_H

