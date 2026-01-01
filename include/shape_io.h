#ifndef SHAPE_IO_H
#define SHAPE_IO_H

#include <string>

#include "pde_types.h"

std::string NormalizeShapeExpression(const std::string& text);

bool LoadShapeExpressionFromFile(const std::string& path,
                                 std::string* expression,
                                 std::string* error);

bool LoadShapeMaskFromVtk(const std::string& path,
                          ShapeMask* mask,
                          std::string* error);

#endif  // SHAPE_IO_H
