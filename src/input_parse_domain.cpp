#include "input_parse.h"

#include <cstdlib>
#include <sstream>
#include <string>
#include <vector>

ParseResult ParseDomain(const std::string& text, Domain* out) {
  std::vector<double> values;
  values.clear();
  std::stringstream ss(text);
  std::string item;
  while (std::getline(ss, item, ',')) {
    if (item.empty()) {
      return {false, "invalid domain format"};
    }
    char* end = nullptr;
    double value = std::strtod(item.c_str(), &end);
    if (end == item.c_str()) {
      return {false, "invalid domain format"};
    }
    values.push_back(value);
  }
  if (values.size() != 4 && values.size() != 6) {
    return {false, "invalid domain format"};
  }
  out->xmin = values[0];
  out->xmax = values[1];
  out->ymin = values[2];
  out->ymax = values[3];
  if (values.size() == 6) {
    out->zmin = values[4];
    out->zmax = values[5];
  } else {
    out->zmin = 0.0;
    out->zmax = 1.0;
  }
  return {true, ""};
}

ParseResult ParseGrid(const std::string& text, Domain* out) {
  std::vector<int> values;
  values.clear();
  std::stringstream ss(text);
  std::string item;
  while (std::getline(ss, item, ',')) {
    if (item.empty()) {
      return {false, "invalid grid format"};
    }
    char* end = nullptr;
    long value = std::strtol(item.c_str(), &end, 10);
    if (end == item.c_str()) {
      return {false, "invalid grid format"};
    }
    values.push_back(static_cast<int>(value));
  }
  if (values.size() != 2 && values.size() != 3) {
    return {false, "invalid grid format"};
  }
  out->nx = values[0];
  out->ny = values[1];
  out->nz = values.size() == 3 ? values[2] : 1;
  return {true, ""};
}


