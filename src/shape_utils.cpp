#include "shape_utils.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace {
const double kEps = 1e-12;

double SafeScale(double scale) {
  return (std::abs(scale) < kEps) ? 1.0 : scale;
}

double Lerp(double a, double b, double t) {
  return a + t * (b - a);
}

double SampleStructuredMask(const ShapeMask& mask,
                            double x, double y, double z,
                            bool* outside) {
  const Domain& d = mask.domain;
  const int nx = d.nx;
  const int ny = d.ny;
  const int nz = std::max(1, d.nz);
  const size_t expected = static_cast<size_t>(nx * ny * nz);
  if (nx < 1 || ny < 1 || nz < 1 || mask.values.size() < expected) {
    if (outside) {
      *outside = true;
    }
    return 1.0;
  }

  const bool use_z = (nz > 1);
  if (x < d.xmin || x > d.xmax || y < d.ymin || y > d.ymax) {
    if (outside) {
      *outside = true;
    }
    return 1.0;
  }
  if (use_z && (z < d.zmin || z > d.zmax)) {
    if (outside) {
      *outside = true;
    }
    return 1.0;
  }

  if (outside) {
    *outside = false;
  }

  const double dx = (d.xmax - d.xmin) / static_cast<double>(std::max(1, nx - 1));
  const double dy = (d.ymax - d.ymin) / static_cast<double>(std::max(1, ny - 1));
  const double dz = (d.zmax - d.zmin) / static_cast<double>(std::max(1, nz - 1));

  const double fx = (nx > 1 && std::abs(dx) > kEps) ? (x - d.xmin) / dx : 0.0;
  const double fy = (ny > 1 && std::abs(dy) > kEps) ? (y - d.ymin) / dy : 0.0;
  const double fz = (use_z && std::abs(dz) > kEps) ? (z - d.zmin) / dz : 0.0;

  const int i0 = std::clamp(static_cast<int>(std::floor(fx)), 0, nx - 1);
  const int j0 = std::clamp(static_cast<int>(std::floor(fy)), 0, ny - 1);
  const int k0 = std::clamp(static_cast<int>(std::floor(fz)), 0, nz - 1);
  const int i1 = std::min(i0 + 1, nx - 1);
  const int j1 = std::min(j0 + 1, ny - 1);
  const int k1 = std::min(k0 + 1, nz - 1);

  const double tx = (nx > 1) ? (fx - static_cast<double>(i0)) : 0.0;
  const double ty = (ny > 1) ? (fy - static_cast<double>(j0)) : 0.0;
  const double tz = (use_z && nz > 1) ? (fz - static_cast<double>(k0)) : 0.0;

  auto idx = [nx, ny](int i, int j, int k) -> size_t {
    return static_cast<size_t>((k * ny + j) * nx + i);
  };

  const double v000 = mask.values[idx(i0, j0, k0)];
  const double v100 = mask.values[idx(i1, j0, k0)];
  const double v010 = mask.values[idx(i0, j1, k0)];
  const double v110 = mask.values[idx(i1, j1, k0)];

  const double v00 = Lerp(v000, v100, tx);
  const double v10 = Lerp(v010, v110, tx);
  const double v0 = Lerp(v00, v10, ty);

  if (!use_z) {
    return v0;
  }

  const double v001 = mask.values[idx(i0, j0, k1)];
  const double v101 = mask.values[idx(i1, j0, k1)];
  const double v011 = mask.values[idx(i0, j1, k1)];
  const double v111 = mask.values[idx(i1, j1, k1)];

  const double v01 = Lerp(v001, v101, tx);
  const double v11 = Lerp(v011, v111, tx);
  const double v1 = Lerp(v01, v11, ty);

  return Lerp(v0, v1, tz);
}

double SamplePointMask(const ShapeMask& mask,
                       double x, double y, double z) {
  if (mask.points.empty()) {
    return 1.0;
  }
  double best_dist = std::numeric_limits<double>::max();
  double best_val = mask.points.front().value;
  for (const auto& pt : mask.points) {
    const double dx = x - pt.x;
    const double dy = y - pt.y;
    const double dz = z - pt.z;
    const double dist = dx * dx + dy * dy + dz * dz;
    if (dist < best_dist) {
      best_dist = dist;
      best_val = pt.value;
    }
  }
  return best_val;
}
}  // namespace

void ApplyShapeTransform(const ShapeTransform& transform,
                         double x, double y, double z,
                         double* out_x, double* out_y, double* out_z) {
  const double sx = SafeScale(transform.scale_x);
  const double sy = SafeScale(transform.scale_y);
  const double sz = SafeScale(transform.scale_z);
  if (out_x) {
    *out_x = (x - transform.offset_x) / sx;
  }
  if (out_y) {
    *out_y = (y - transform.offset_y) / sy;
  }
  if (out_z) {
    *out_z = (z - transform.offset_z) / sz;
  }
}

double SampleShapeMaskPhi(const ShapeMask& mask,
                          double x, double y, double z,
                          double threshold,
                          bool invert) {
  if (!HasShapeMask(mask)) {
    return 1.0;
  }
  bool outside = false;
  double value = 1.0;
  if (!mask.values.empty()) {
    value = SampleStructuredMask(mask, x, y, z, &outside);
  } else {
    value = SamplePointMask(mask, x, y, z);
  }
  if (outside) {
    return 1.0;
  }
  return invert ? (threshold - value) : (value - threshold);
}
