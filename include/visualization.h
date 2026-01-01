// Shared visualization helpers for Qt and ImGui viewers.
#ifndef VISUALIZATION_H
#define VISUALIZATION_H

#include <vector>

#include "pde_types.h"

struct RGBColor {
  float r;
  float g;
  float b;
};

struct VertexData {
  float position[3];
  float color[3];
};

RGBColor ColorRamp(float t);

std::vector<VertexData> BuildGridVertices(const Domain& domain,
                                          const std::vector<double>& grid,
                                          float data_scale = 1.0f);

std::vector<VertexData> BuildPointCloudVertices(const std::vector<PointSample>& points,
                                                const Domain& bounds);

#endif  // VISUALIZATION_H

