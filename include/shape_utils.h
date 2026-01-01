#ifndef SHAPE_UTILS_H
#define SHAPE_UTILS_H

#include "pde_types.h"

void ApplyShapeTransform(const ShapeTransform& transform,
                         double x, double y, double z,
                         double* out_x, double* out_y, double* out_z);

double SampleShapeMaskPhi(const ShapeMask& mask,
                          double x, double y, double z,
                          double threshold,
                          bool invert);

#endif  // SHAPE_UTILS_H
