#include "math_utils.h"

#include <cstdlib>
#include <iomanip>
#include <sstream>
#include <string>

bool ParseIntValue(const std::string& text, int* out) {
  char* end = nullptr;
  long value = std::strtol(text.c_str(), &end, 10);
  if (!end || *end != '\0') {
    return false;
  }
  if (out) {
    *out = static_cast<int>(value);
  }
  return true;
}

std::string FormatBounds(double xmin, double xmax, double ymin, double ymax) {
  std::ostringstream out;
  out << xmin << "," << xmax << "," << ymin << "," << ymax;
  return out.str();
}

std::string FormatBounds3D(double xmin, double xmax, double ymin, double ymax, double zmin,
                           double zmax) {
  std::ostringstream out;
  out << xmin << "," << xmax << "," << ymin << "," << ymax << "," << zmin << "," << zmax;
  return out.str();
}

std::string FormatGrid(int nx, int ny) {
  std::ostringstream out;
  out << nx << "," << ny;
  return out.str();
}

std::string FormatGrid3D(int nx, int ny, int nz) {
  std::ostringstream out;
  out << nx << "," << ny << "," << nz;
  return out.str();
}

