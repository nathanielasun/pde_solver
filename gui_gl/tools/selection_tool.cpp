#include "selection_tool.h"

#include <algorithm>

std::vector<bool> CreateMaskFromRegion(const Domain& domain, const SelectionRegion& region) {
  const int nx = domain.nx;
  const int ny = domain.ny;
  const int nz = std::max(1, domain.nz);
  std::vector<bool> mask(static_cast<size_t>(nx * ny * nz), false);

  const double dx = (domain.xmax - domain.xmin) / std::max(1, nx - 1);
  const double dy = (domain.ymax - domain.ymin) / std::max(1, ny - 1);
  const double dz = (nz > 1) ? ((domain.zmax - domain.zmin) / std::max(1, nz - 1)) : 0.0;

  size_t idx = 0;
  for (int k = 0; k < nz; ++k) {
    const double z = domain.zmin + dz * k;
    for (int j = 0; j < ny; ++j) {
      const double y = domain.ymin + dy * j;
      for (int i = 0; i < nx; ++i, ++idx) {
        const double x = domain.xmin + dx * i;
        mask[idx] = region.Contains(x, y, z);
      }
    }
  }

  return mask;
}

