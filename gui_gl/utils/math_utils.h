#ifndef MATH_UTILS_H
#define MATH_UTILS_H

#include <string>

// Math helpers
bool ParseIntValue(const std::string& text, int* out);

// Formatting functions
std::string FormatBounds(double xmin, double xmax, double ymin, double ymax);
std::string FormatBounds3D(double xmin, double xmax, double ymin, double ymax, double zmin, double zmax);
std::string FormatGrid(int nx, int ny);
std::string FormatGrid3D(int nx, int ny, int nz);

#endif // MATH_UTILS_H

