#include "vtk_io.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace {
std::string ToUpper(const std::string& input) {
  std::string out = input;
  std::transform(out.begin(), out.end(), out.begin(), [](unsigned char c) {
    return static_cast<char>(std::toupper(c));
  });
  return out;
}

VtkReadResult ReadVtkPointCloud(const std::string& path) {
  std::ifstream in(path);
  if (!in) {
    VtkReadResult result;
    result.ok = false;
    result.error = "failed to open vtk file";
    return result;
  }

  int point_count = 0;
  int point_data = 0;
  bool got_points = false;
  bool got_scalars = false;
  bool got_lookup = false;
  int components = 1;
  std::vector<double> coords;
  std::string token;

  while (in >> token) {
    const std::string upper = ToUpper(token);
    if (upper == "POINTS") {
      std::string type;
      in >> point_count >> type;
      if (point_count <= 0) {
        VtkReadResult result;
        result.ok = false;
        result.error = "invalid point count";
        return result;
      }
      coords.resize(static_cast<size_t>(point_count) * 3);
      for (int i = 0; i < point_count * 3; ++i) {
        if (!(in >> coords[static_cast<size_t>(i)])) {
          VtkReadResult result;
          result.ok = false;
          result.error = "failed to read point coordinates";
          return result;
        }
      }
      got_points = true;
    } else if (upper == "POINT_DATA") {
      in >> point_data;
    } else if (upper == "SCALARS") {
      std::string name;
      std::string type;
      in >> name >> type;
      std::string rest;
      std::getline(in, rest);
      std::istringstream rest_stream(rest);
      int parsed = 0;
      if (rest_stream >> parsed) {
        components = parsed;
      }
      if (components != 1) {
        VtkReadResult result;
        result.ok = false;
        result.error = "only single-component point scalars are supported";
        return result;
      }
      got_scalars = true;
    } else if (upper == "LOOKUP_TABLE") {
      std::string table;
      in >> table;
      got_lookup = true;
      break;
    }
  }

  if (!got_points) {
    VtkReadResult result;
    result.ok = false;
    result.error = "missing POINTS section";
    return result;
  }
  if (!got_scalars || !got_lookup) {
    VtkReadResult result;
    result.ok = false;
    result.error = "missing point scalar data";
    return result;
  }
  if (point_data <= 0) {
    point_data = point_count;
  }
  if (point_data != point_count) {
    VtkReadResult result;
    result.ok = false;
    result.error = "point_data count mismatch";
    return result;
  }

  std::vector<double> values;
  values.reserve(static_cast<size_t>(point_data));
  double value = 0.0;
  while (values.size() < static_cast<size_t>(point_data) && (in >> value)) {
    values.push_back(value);
  }
  if (values.size() != static_cast<size_t>(point_data)) {
    VtkReadResult result;
    result.ok = false;
    result.error = "insufficient scalar values";
    return result;
  }

  Domain domain;
  double xmin = coords[0];
  double xmax = coords[0];
  double ymin = coords[1];
  double ymax = coords[1];
  double zmin = coords[2];
  double zmax = coords[2];
  for (int i = 0; i < point_count; ++i) {
    const double x = coords[static_cast<size_t>(i) * 3];
    const double y = coords[static_cast<size_t>(i) * 3 + 1];
    const double z = coords[static_cast<size_t>(i) * 3 + 2];
    xmin = std::min(xmin, x);
    xmax = std::max(xmax, x);
    ymin = std::min(ymin, y);
    ymax = std::max(ymax, y);
    zmin = std::min(zmin, z);
    zmax = std::max(zmax, z);
  }
  if (std::abs(xmax - xmin) < 1e-12) {
    xmin -= 0.5;
    xmax += 0.5;
  }
  if (std::abs(ymax - ymin) < 1e-12) {
    ymin -= 0.5;
    ymax += 0.5;
  }
  if (std::abs(zmax - zmin) < 1e-12) {
    zmin -= 0.5;
    zmax += 0.5;
  }
  domain.xmin = xmin;
  domain.xmax = xmax;
  domain.ymin = ymin;
  domain.ymax = ymax;
  domain.zmin = zmin;
  domain.zmax = zmax;
  domain.nx = 2;
  domain.ny = 2;
  domain.nz = std::abs(zmax - zmin) > 1e-12 ? 2 : 1;

  std::vector<PointSample> points;
  points.reserve(static_cast<size_t>(point_count));
  for (int i = 0; i < point_count; ++i) {
    PointSample sample;
    sample.x = coords[static_cast<size_t>(i) * 3];
    sample.y = coords[static_cast<size_t>(i) * 3 + 1];
    sample.z = coords[static_cast<size_t>(i) * 3 + 2];
    sample.value = values[static_cast<size_t>(i)];
    points.push_back(sample);
  }

  VtkReadResult result;
  result.ok = true;
  result.kind = VtkReadResult::Kind::PointCloud;
  result.domain = domain;
  result.points = std::move(points);
  return result;
}
}  // namespace

VtkReadResult ReadVtkStructuredPoints(const std::string& path) {
  std::ifstream in(path);
  if (!in) {
    VtkReadResult result;
    result.ok = false;
    result.error = "failed to open vtk file";
    return result;
  }

  std::string token;
  std::string dataset;
  int nx = 0;
  int ny = 0;
  int nz = 0;
  double ox = 0.0;
  double oy = 0.0;
  double oz = 0.0;
  double sx = 1.0;
  double sy = 1.0;
  double sz = 1.0;
  int point_data = 0;

  while (in >> token) {
    const std::string upper = ToUpper(token);
    if (upper == "DATASET") {
      in >> dataset;
      dataset = ToUpper(dataset);
    } else if (upper == "DIMENSIONS") {
      in >> nx >> ny >> nz;
    } else if (upper == "ORIGIN") {
      in >> ox >> oy >> oz;
    } else if (upper == "SPACING") {
      in >> sx >> sy >> sz;
    } else if (upper == "POINT_DATA") {
      in >> point_data;
    } else if (upper == "LOOKUP_TABLE") {
      std::string table_name;
      in >> table_name;
      break;
    }
  }

  if (dataset != "STRUCTURED_POINTS") {
    VtkReadResult result;
    result.ok = false;
    result.error = "unsupported dataset type";
    return result;
  }
  if (nx <= 0 || ny <= 0 || nz <= 0) {
    VtkReadResult result;
    result.ok = false;
    result.error = "invalid dimensions";
    return result;
  }
  if (point_data != nx * ny * nz) {
    VtkReadResult result;
    result.ok = false;
    result.error = "point_data count mismatch";
    return result;
  }

  std::vector<double> values;
  values.reserve(static_cast<size_t>(point_data));
  double value = 0.0;
  while (values.size() < static_cast<size_t>(point_data) && (in >> value)) {
    values.push_back(value);
  }
  if (values.size() != static_cast<size_t>(point_data)) {
    VtkReadResult result;
    result.ok = false;
    result.error = "insufficient scalar values";
    return result;
  }

  Domain domain;
  domain.xmin = ox;
  domain.ymin = oy;
  domain.zmin = oz;
  domain.xmax = ox + sx * static_cast<double>(nx - 1);
  domain.ymax = oy + sy * static_cast<double>(ny - 1);
  domain.zmax = oz + sz * static_cast<double>(nz - 1);
  domain.nx = nx;
  domain.ny = ny;
  domain.nz = nz;

  VtkReadResult result;
  result.ok = true;
  result.kind = VtkReadResult::Kind::StructuredPoints;
  result.domain = domain;
  result.grid = std::move(values);
  return result;
}

VtkReadResult ReadVtkFile(const std::string& path) {
  std::ifstream in(path);
  if (!in) {
    VtkReadResult result;
    result.ok = false;
    result.error = "failed to open vtk file";
    return result;
  }
  std::string token;
  std::string dataset;
  while (in >> token) {
    if (ToUpper(token) == "DATASET") {
      in >> dataset;
      dataset = ToUpper(dataset);
      break;
    }
  }
  if (dataset.empty()) {
    VtkReadResult result;
    result.ok = false;
    result.error = "missing dataset type";
    return result;
  }
  if (dataset == "STRUCTURED_POINTS") {
    return ReadVtkStructuredPoints(path);
  }
  if (dataset == "UNSTRUCTURED_GRID" || dataset == "POLYDATA") {
    return ReadVtkPointCloud(path);
  }
  VtkReadResult result;
  result.ok = false;
  result.error = "unsupported dataset type";
  return result;
}

VtkReadResult ReadCheckpoint(const std::string& path, CheckpointData* checkpoint) {
  std::ifstream in(path);
  if (!in) {
    VtkReadResult result;
    result.ok = false;
    result.error = "failed to open checkpoint file";
    return result;
  }
  if (checkpoint) {
    *checkpoint = CheckpointData{};
  }

  std::string line;
  std::string token;
  int version = 0;

  while (std::getline(in, line)) {
    if (line.empty() || line[0] == '#') {
      continue;
    }
    std::istringstream iss(line);
    if (!(iss >> token)) {
      continue;
    }

    if (token == "CHECKPOINT_VERSION") {
      iss >> version;
      if (version != 1 && version != 2) {
        VtkReadResult result;
        result.ok = false;
        result.error = "unsupported checkpoint version";
        return result;
      }
    } else if (token == "DOMAIN") {
      iss >> checkpoint->domain.xmin >> checkpoint->domain.xmax
          >> checkpoint->domain.ymin >> checkpoint->domain.ymax
          >> checkpoint->domain.zmin >> checkpoint->domain.zmax
          >> checkpoint->domain.nx >> checkpoint->domain.ny >> checkpoint->domain.nz;
    } else if (token == "TIME") {
      iss >> checkpoint->t_current >> checkpoint->frame_index;
    } else if (token == "PDE") {
      iss >> checkpoint->pde.a >> checkpoint->pde.b >> checkpoint->pde.az
          >> checkpoint->pde.c >> checkpoint->pde.d >> checkpoint->pde.dz
          >> checkpoint->pde.e >> checkpoint->pde.f >> checkpoint->pde.ut
          >> checkpoint->pde.utt;
    } else if (token == "RHS_LATEX") {
      std::getline(iss, checkpoint->pde.rhs_latex);
      // Trim leading whitespace
      checkpoint->pde.rhs_latex.erase(0, checkpoint->pde.rhs_latex.find_first_not_of(" \t"));
    } else if (token == "GRID") {
      size_t count = 0;
      iss >> count;
      checkpoint->grid.resize(count);
      for (size_t i = 0; i < count; ++i) {
        if (!(in >> checkpoint->grid[i])) {
          VtkReadResult result;
          result.ok = false;
          result.error = "failed to read grid values";
          return result;
        }
      }
    } else if (token == "VELOCITY") {
      size_t count = 0;
      iss >> count;
      checkpoint->velocity.resize(count);
      for (size_t i = 0; i < count; ++i) {
        if (!(in >> checkpoint->velocity[i])) {
          VtkReadResult result;
          result.ok = false;
          result.error = "failed to read velocity values";
          return result;
        }
      }
    } else if (token == "BC_LEFT") {
      int kind;
      iss >> kind >> checkpoint->bc.left.value.constant;
      checkpoint->bc.left.kind = static_cast<BCKind>(kind);
    } else if (token == "BC_RIGHT") {
      int kind;
      iss >> kind >> checkpoint->bc.right.value.constant;
      checkpoint->bc.right.kind = static_cast<BCKind>(kind);
    } else if (token == "BC_BOTTOM") {
      int kind;
      iss >> kind >> checkpoint->bc.bottom.value.constant;
      checkpoint->bc.bottom.kind = static_cast<BCKind>(kind);
    } else if (token == "BC_TOP") {
      int kind;
      iss >> kind >> checkpoint->bc.top.value.constant;
      checkpoint->bc.top.kind = static_cast<BCKind>(kind);
    } else if (token == "BC_FRONT") {
      int kind;
      iss >> kind >> checkpoint->bc.front.value.constant;
      checkpoint->bc.front.kind = static_cast<BCKind>(kind);
    } else if (token == "BC_BACK") {
      int kind;
      iss >> kind >> checkpoint->bc.back.value.constant;
      checkpoint->bc.back.kind = static_cast<BCKind>(kind);
    }
  }

  if (!checkpoint->velocity.empty() &&
      checkpoint->velocity.size() != checkpoint->grid.size()) {
    VtkReadResult result;
    result.ok = false;
    result.error = "checkpoint velocity size mismatch";
    return result;
  }

  VtkReadResult result;
  result.ok = true;
  result.kind = VtkReadResult::Kind::StructuredPoints;
  result.domain = checkpoint->domain;
  result.grid = checkpoint->grid;
  return result;
}
