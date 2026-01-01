#include "visualization.h"

#include <algorithm>
#include <cmath>

namespace {
float Clamp01(float t) {
  return std::min(1.0f, std::max(0.0f, t));
}
}  // namespace

RGBColor ColorRamp(float t) {
  const float clamped = Clamp01(t);
  const float red = clamped;
  const float blue = 1.0f - clamped;
  const float green = 0.2f + 0.6f * (1.0f - std::abs(clamped - 0.5f) * 2.0f);
  return {red, green, blue};
}

std::vector<VertexData> BuildGridVertices(const Domain& domain,
                                          const std::vector<double>& grid,
                                          float data_scale) {
  const int nx = domain.nx;
  const int ny = domain.ny;
  const int nz = std::max(1, domain.nz);
  if (nx < 1 || ny < 1 || grid.size() != static_cast<size_t>(nx * ny * nz)) {
    return {};
  }

  double min_val = std::numeric_limits<double>::infinity();
  double max_val = -std::numeric_limits<double>::infinity();
  for (double v : grid) {
    min_val = std::min(min_val, v);
    max_val = std::max(max_val, v);
  }
  const double x_span = std::max(1e-9, domain.xmax - domain.xmin);
  const double y_span = std::max(1e-9, domain.ymax - domain.ymin);
  const double z_span = std::max(1e-9, domain.zmax - domain.zmin);
  const double v_span = std::max(1e-9, max_val - min_val);
  const double dx = x_span / std::max(1, nx - 1);
  const double dy = y_span / std::max(1, ny - 1);
  const double dz = z_span / std::max(1, nz - 1);

  std::vector<VertexData> vertices;
  vertices.reserve(static_cast<size_t>(nx * ny * nz));
  for (int k = 0; k < nz; ++k) {
    const double z = domain.zmin + dz * k;
    for (int j = 0; j < ny; ++j) {
      for (int i = 0; i < nx; ++i) {
        const size_t idx = static_cast<size_t>((k * ny + j) * nx + i);
        const double value = grid[idx];
        const double x = domain.xmin + dx * i;
        const double y = domain.ymin + dy * j;
        const float tx = static_cast<float>((x - domain.xmin) / x_span);
        const float ty = static_cast<float>((y - domain.ymin) / y_span);
        const float tz = static_cast<float>((z - domain.zmin) / z_span);
        const float t = static_cast<float>((value - min_val) / v_span);
        const float px = (tx - 0.5f) * 2.0f;
        const float py = (ty - 0.5f) * 2.0f;
        const float pz = (nz > 1) ? (tz - 0.5f) * 2.0f : (t - 0.5f) * 0.8f;

        const RGBColor color = ColorRamp(t);
        VertexData vertex{};
        vertex.position[0] = px;
        vertex.position[1] = py;
        vertex.position[2] = pz * data_scale;
        vertex.color[0] = color.r;
        vertex.color[1] = color.g;
        vertex.color[2] = color.b;
        vertices.push_back(vertex);
      }
    }
  }
  return vertices;
}

std::vector<VertexData> BuildPointCloudVertices(const std::vector<PointSample>& points,
                                                const Domain& bounds) {
  if (points.empty()) {
    return {};
  }
  double min_val = std::numeric_limits<double>::infinity();
  double max_val = -std::numeric_limits<double>::infinity();
  for (const auto& pt : points) {
    min_val = std::min(min_val, pt.value);
    max_val = std::max(max_val, pt.value);
  }
  const double x_span = std::max(1e-9, bounds.xmax - bounds.xmin);
  const double y_span = std::max(1e-9, bounds.ymax - bounds.ymin);
  const double z_span = std::max(1e-9, bounds.zmax - bounds.zmin);
  const double v_span = std::max(1e-9, max_val - min_val);

  std::vector<VertexData> vertices;
  vertices.reserve(points.size());
  for (const auto& pt : points) {
    const float tx = static_cast<float>((pt.x - bounds.xmin) / x_span);
    const float ty = static_cast<float>((pt.y - bounds.ymin) / y_span);
    const float tz = static_cast<float>((pt.z - bounds.zmin) / z_span);
    const float t = static_cast<float>((pt.value - min_val) / v_span);
    const float px = (tx - 0.5f) * 2.0f;
    const float py = (ty - 0.5f) * 2.0f;
    const float pz = (tz - 0.5f) * 2.0f;
    const RGBColor color = ColorRamp(t);

    VertexData vertex{};
    vertex.position[0] = px;
    vertex.position[1] = py;
    vertex.position[2] = pz;
    vertex.color[0] = color.r;
    vertex.color[1] = color.g;
    vertex.color[2] = color.b;
    vertices.push_back(vertex);
  }
  return vertices;
}

