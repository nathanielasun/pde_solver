#include "shape_io.h"

#include <fstream>
#include <sstream>

#include "string_utils.h"
#include "vtk_io.h"

namespace {
bool ReadFile(const std::string& path, std::string* out, std::string* error) {
  std::ifstream in(path);
  if (!in.is_open()) {
    if (error) {
      *error = "failed to open file: " + path;
    }
    return false;
  }
  std::ostringstream buffer;
  buffer << in.rdbuf();
  *out = buffer.str();
  return true;
}
}  // namespace

std::string NormalizeShapeExpression(const std::string& text) {
  std::string trimmed = pde::Trim(text);
  if (trimmed.empty()) {
    return trimmed;
  }

  auto trim = [](const std::string& input) {
    return pde::Trim(input);
  };

  size_t pos = trimmed.find("<=");
  size_t op_len = 2;
  if (pos == std::string::npos) {
    pos = trimmed.find('<');
    op_len = 1;
  }
  if (pos != std::string::npos) {
    const std::string lhs = trim(trimmed.substr(0, pos));
    const std::string rhs = trim(trimmed.substr(pos + op_len));
    if (!rhs.empty()) {
      return lhs + "-(" + rhs + ")";
    }
    return lhs;
  }

  pos = trimmed.find('=');
  if (pos != std::string::npos) {
    const std::string rhs = trim(trimmed.substr(pos + 1));
    if (!rhs.empty()) {
      return rhs;
    }
  }

  return trimmed;
}

bool LoadShapeExpressionFromFile(const std::string& path,
                                 std::string* expression,
                                 std::string* error) {
  std::string content;
  if (!ReadFile(path, &content, error)) {
    return false;
  }
  const std::string normalized = NormalizeShapeExpression(content);
  if (expression) {
    *expression = normalized;
  }
  if (normalized.empty()) {
    if (error) {
      *error = "shape file is empty";
    }
    return false;
  }
  return true;
}

bool LoadShapeMaskFromVtk(const std::string& path,
                          ShapeMask* mask,
                          std::string* error) {
  VtkReadResult result = ReadVtkFile(path);
  if (!result.ok) {
    if (error) {
      *error = result.error.empty() ? "failed to read mask file" : result.error;
    }
    return false;
  }
  if (!mask) {
    return true;
  }
  ShapeMask loaded;
  loaded.domain = result.domain;
  if (result.kind == VtkReadResult::Kind::StructuredPoints) {
    loaded.values = std::move(result.grid);
  } else {
    loaded.points = std::move(result.points);
  }
  *mask = std::move(loaded);
  return true;
}
