#include "mesh_io.h"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "string_utils.h"

namespace {
std::string ToUpper(const std::string& input) {
  std::string out = input;
  std::transform(out.begin(), out.end(), out.begin(), [](unsigned char c) {
    return static_cast<char>(std::toupper(c));
  });
  return out;
}

std::string StripLeadingDot(const std::string& value) {
  if (!value.empty() && value[0] == '.') {
    return value.substr(1);
  }
  return value;
}

struct GmshElementInfo {
  int vtk_type = 0;
  int node_count = 0;
};

bool GetGmshElementInfo(int gmsh_type, GmshElementInfo* info) {
  if (!info) {
    return false;
  }
  switch (gmsh_type) {
    case 1:  // 2-node line
      info->vtk_type = 3;  // VTK_LINE
      info->node_count = 2;
      return true;
    case 2:  // 3-node triangle
      info->vtk_type = 5;  // VTK_TRIANGLE
      info->node_count = 3;
      return true;
    case 3:  // 4-node quad
      info->vtk_type = 9;  // VTK_QUAD
      info->node_count = 4;
      return true;
    case 4:  // 4-node tetra
      info->vtk_type = 10;  // VTK_TETRA
      info->node_count = 4;
      return true;
    case 5:  // 8-node hex
      info->vtk_type = 12;  // VTK_HEXAHEDRON
      info->node_count = 8;
      return true;
    case 6:  // 6-node prism
      info->vtk_type = 13;  // VTK_WEDGE
      info->node_count = 6;
      return true;
    case 7:  // 5-node pyramid
      info->vtk_type = 14;  // VTK_PYRAMID
      info->node_count = 5;
      return true;
    case 15:  // point
      info->vtk_type = 1;  // VTK_VERTEX
      info->node_count = 1;
      return true;
    default:
      return false;
  }
}

void SkipSection(std::istream& in, const std::string& end_token) {
  std::string token;
  while (in >> token) {
    if (token == end_token) {
      break;
    }
  }
}

MeshReadResult ReadLegacyVtkUnstructuredGrid(const std::string& path) {
  std::ifstream in(path);
  if (!in) {
    MeshReadResult result;
    result.ok = false;
    result.error = "failed to open vtk file";
    return result;
  }

  std::string token;
  std::string dataset;
  while (in >> token) {
    const std::string upper = ToUpper(token);
    if (upper == "BINARY") {
      MeshReadResult result;
      result.ok = false;
      result.error = "binary VTK unstructured grid is not supported";
      return result;
    }
    if (upper == "DATASET") {
      in >> dataset;
      dataset = ToUpper(dataset);
      break;
    }
  }
  if (dataset != "UNSTRUCTURED_GRID") {
    MeshReadResult result;
    result.ok = false;
    result.error = "unsupported VTK dataset (expected UNSTRUCTURED_GRID)";
    return result;
  }

  UnstructuredMesh mesh;
  int point_count = 0;
  int cell_count = 0;
  int point_data_count = 0;
  int cell_data_count = 0;
  enum class DataTarget { None, Point, Cell };
  DataTarget data_target = DataTarget::None;

  while (in >> token) {
    const std::string upper = ToUpper(token);
    if (upper == "POINTS") {
      std::string type;
      in >> point_count >> type;
      if (point_count <= 0) {
        MeshReadResult result;
        result.ok = false;
        result.error = "invalid point count";
        return result;
      }
      mesh.points.resize(static_cast<size_t>(point_count) * 3);
      for (int i = 0; i < point_count * 3; ++i) {
        if (!(in >> mesh.points[static_cast<size_t>(i)])) {
          MeshReadResult result;
          result.ok = false;
          result.error = "failed to read point coordinates";
          return result;
        }
      }
    } else if (upper == "CELLS") {
      int total_size = 0;
      in >> cell_count >> total_size;
      if (cell_count < 0 || total_size < 0) {
        MeshReadResult result;
        result.ok = false;
        result.error = "invalid cell section";
        return result;
      }
      mesh.cell_offsets.clear();
      mesh.cell_connectivity.clear();
      mesh.cell_offsets.reserve(static_cast<size_t>(cell_count) + 1);
      mesh.cell_offsets.push_back(0);
      for (int cell = 0; cell < cell_count; ++cell) {
        int vertex_count = 0;
        if (!(in >> vertex_count) || vertex_count <= 0) {
          MeshReadResult result;
          result.ok = false;
          result.error = "invalid cell vertex count";
          return result;
        }
        for (int v = 0; v < vertex_count; ++v) {
          int index = 0;
          if (!(in >> index)) {
            MeshReadResult result;
            result.ok = false;
            result.error = "failed to read cell connectivity";
            return result;
          }
          if (index < 0 || (point_count > 0 && index >= point_count)) {
            MeshReadResult result;
            result.ok = false;
            result.error = "cell connectivity index out of range";
            return result;
          }
          mesh.cell_connectivity.push_back(index);
        }
        mesh.cell_offsets.push_back(static_cast<int>(mesh.cell_connectivity.size()));
      }
    } else if (upper == "CELL_TYPES") {
      int type_count = 0;
      in >> type_count;
      if (type_count < 0) {
        MeshReadResult result;
        result.ok = false;
        result.error = "invalid cell type count";
        return result;
      }
      mesh.cell_types.resize(static_cast<size_t>(type_count));
      for (int i = 0; i < type_count; ++i) {
        if (!(in >> mesh.cell_types[static_cast<size_t>(i)])) {
          MeshReadResult result;
          result.ok = false;
          result.error = "failed to read cell types";
          return result;
        }
      }
    } else if (upper == "POINT_DATA") {
      in >> point_data_count;
      data_target = DataTarget::Point;
    } else if (upper == "CELL_DATA") {
      in >> cell_data_count;
      data_target = DataTarget::Cell;
    } else if (upper == "SCALARS") {
      const int expected_count =
          (data_target == DataTarget::Point) ? point_data_count :
          (data_target == DataTarget::Cell) ? cell_data_count : 0;
      if (expected_count <= 0) {
        MeshReadResult result;
        result.ok = false;
        result.error = "SCALARS section without active POINT_DATA/CELL_DATA";
        return result;
      }
      std::string name;
      std::string type;
      in >> name >> type;
      std::string rest;
      std::getline(in, rest);
      std::istringstream rest_stream(rest);
      int components = 1;
      if (rest_stream >> components) {
        if (components != 1) {
          MeshReadResult result;
          result.ok = false;
          result.error = "only single-component scalars are supported";
          return result;
        }
      }
      std::string lookup_token;
      if (!(in >> lookup_token) || ToUpper(lookup_token) != "LOOKUP_TABLE") {
        MeshReadResult result;
        result.ok = false;
        result.error = "missing LOOKUP_TABLE after SCALARS";
        return result;
      }
      std::string table_name;
      in >> table_name;
      std::vector<double> values;
      values.reserve(static_cast<size_t>(expected_count));
      double value = 0.0;
      while (values.size() < static_cast<size_t>(expected_count) && (in >> value)) {
        values.push_back(value);
      }
      if (values.size() != static_cast<size_t>(expected_count)) {
        MeshReadResult result;
        result.ok = false;
        result.error = "insufficient scalar values";
        return result;
      }
      if (data_target == DataTarget::Point) {
        mesh.point_scalars = std::move(values);
        mesh.point_scalar_name = name;
      } else if (data_target == DataTarget::Cell) {
        mesh.cell_scalars = std::move(values);
        mesh.cell_scalar_name = name;
      }
    }
  }

  if (mesh.points.empty()) {
    MeshReadResult result;
    result.ok = false;
    result.error = "missing POINTS section";
    return result;
  }
  if (mesh.cell_offsets.size() <= 1) {
    MeshReadResult result;
    result.ok = false;
    result.error = "missing CELLS section";
    return result;
  }
  if (!mesh.cell_types.empty() && cell_count > 0 &&
      static_cast<int>(mesh.cell_types.size()) != cell_count) {
    MeshReadResult result;
    result.ok = false;
    result.error = "cell type count mismatch";
    return result;
  }

  const bool has_cell_types = !mesh.cell_types.empty();
  MeshReadResult result;
  result.ok = true;
  result.mesh = std::move(mesh);
  if (!has_cell_types && cell_count > 0) {
    result.warning = "CELL_TYPES section missing; cell topology may be incomplete";
  }
  return result;
}

MeshReadResult ReadGmshMesh(const std::string& path) {
  std::ifstream in(path);
  if (!in) {
    MeshReadResult result;
    result.ok = false;
    result.error = "failed to open msh file";
    return result;
  }

  double version = 0.0;
  int file_type = 0;
  int data_size = 0;
  bool got_format = false;

  UnstructuredMesh mesh;
  std::unordered_map<long long, int> node_index;
  std::string token;

  while (in >> token) {
    if (token == "$MeshFormat") {
      if (!(in >> version >> file_type >> data_size)) {
        MeshReadResult result;
        result.ok = false;
        result.error = "invalid $MeshFormat section";
        return result;
      }
      got_format = true;
      if (file_type != 0) {
        MeshReadResult result;
        result.ok = false;
        result.error = "binary Gmsh files are not supported";
        return result;
      }
      (void)data_size;
      std::string end_token;
      in >> end_token;
    } else if (token == "$Nodes") {
      if (!got_format) {
        MeshReadResult result;
        result.ok = false;
        result.error = "missing $MeshFormat header";
        return result;
      }
      if (version >= 4.0) {
        long long num_blocks = 0;
        long long num_nodes = 0;
        long long min_tag = 0;
        long long max_tag = 0;
        in >> num_blocks >> num_nodes >> min_tag >> max_tag;
        (void)min_tag;
        (void)max_tag;
        if (num_nodes <= 0) {
          MeshReadResult result;
          result.ok = false;
          result.error = "invalid node count";
          return result;
        }
        mesh.points.resize(static_cast<size_t>(num_nodes) * 3);
        node_index.reserve(static_cast<size_t>(num_nodes));
        long long next_index = 0;
        for (long long block = 0; block < num_blocks; ++block) {
          int entity_dim = 0;
          int entity_tag = 0;
          int parametric = 0;
          long long block_nodes = 0;
          in >> entity_dim >> entity_tag >> parametric >> block_nodes;
          (void)entity_tag;
          if (block_nodes <= 0) {
            continue;
          }
          std::vector<long long> tags(static_cast<size_t>(block_nodes));
          for (long long i = 0; i < block_nodes; ++i) {
            in >> tags[static_cast<size_t>(i)];
          }
          for (long long i = 0; i < block_nodes; ++i) {
            double x = 0.0;
            double y = 0.0;
            double z = 0.0;
            in >> x >> y >> z;
            if (parametric) {
              const int param_count = std::max(0, entity_dim);
              double dummy = 0.0;
              for (int p = 0; p < param_count; ++p) {
                in >> dummy;
              }
            }
            if (next_index >= num_nodes) {
              MeshReadResult result;
              result.ok = false;
              result.error = "node count exceeds header declaration";
              return result;
            }
            const size_t idx = static_cast<size_t>(next_index);
            node_index[tags[static_cast<size_t>(i)]] = static_cast<int>(next_index);
            mesh.points[idx * 3] = x;
            mesh.points[idx * 3 + 1] = y;
            mesh.points[idx * 3 + 2] = z;
            ++next_index;
          }
        }
        std::string end_token;
        in >> end_token;
        if (static_cast<size_t>(next_index) != mesh.points.size() / 3) {
          MeshReadResult result;
          result.ok = false;
          result.error = "node count mismatch in $Nodes";
          return result;
        }
      } else {
        int node_count = 0;
        in >> node_count;
        if (node_count <= 0) {
          MeshReadResult result;
          result.ok = false;
          result.error = "invalid node count";
          return result;
        }
        mesh.points.resize(static_cast<size_t>(node_count) * 3);
        node_index.reserve(static_cast<size_t>(node_count));
        for (int i = 0; i < node_count; ++i) {
          long long id = 0;
          double x = 0.0;
          double y = 0.0;
          double z = 0.0;
          if (!(in >> id >> x >> y >> z)) {
            MeshReadResult result;
            result.ok = false;
            result.error = "failed to read node coordinates";
            return result;
          }
          node_index[id] = i;
          mesh.points[static_cast<size_t>(i) * 3] = x;
          mesh.points[static_cast<size_t>(i) * 3 + 1] = y;
          mesh.points[static_cast<size_t>(i) * 3 + 2] = z;
        }
        std::string end_token;
        in >> end_token;
      }
    } else if (token == "$Elements") {
      if (!got_format) {
        MeshReadResult result;
        result.ok = false;
        result.error = "missing $MeshFormat header";
        return result;
      }
      mesh.cell_offsets.clear();
      mesh.cell_connectivity.clear();
      mesh.cell_types.clear();
      mesh.cell_offsets.push_back(0);
      if (version >= 4.0) {
        long long num_blocks = 0;
        long long num_elements = 0;
        long long min_tag = 0;
        long long max_tag = 0;
        in >> num_blocks >> num_elements >> min_tag >> max_tag;
        (void)min_tag;
        (void)max_tag;
        for (long long block = 0; block < num_blocks; ++block) {
          int entity_dim = 0;
          int entity_tag = 0;
          int element_type = 0;
          long long block_elements = 0;
          in >> entity_dim >> entity_tag >> element_type >> block_elements;
          (void)entity_dim;
          (void)entity_tag;
          GmshElementInfo info;
          if (!GetGmshElementInfo(element_type, &info)) {
            MeshReadResult result;
            result.ok = false;
            result.error = "unsupported gmsh element type";
            return result;
          }
          for (long long i = 0; i < block_elements; ++i) {
            long long element_tag = 0;
            in >> element_tag;
            const int start = mesh.cell_offsets.back();
            for (int n = 0; n < info.node_count; ++n) {
              long long node_id = 0;
              in >> node_id;
              auto it = node_index.find(node_id);
              if (it == node_index.end()) {
                MeshReadResult result;
                result.ok = false;
                result.error = "element references unknown node";
                return result;
              }
              mesh.cell_connectivity.push_back(it->second);
            }
            mesh.cell_offsets.push_back(start + info.node_count);
            mesh.cell_types.push_back(info.vtk_type);
          }
        }
        std::string end_token;
        in >> end_token;
      } else {
        int element_count = 0;
        in >> element_count;
        for (int i = 0; i < element_count; ++i) {
          long long element_id = 0;
          int element_type = 0;
          int tag_count = 0;
          in >> element_id >> element_type >> tag_count;
          for (int t = 0; t < tag_count; ++t) {
            int tag = 0;
            in >> tag;
          }
          GmshElementInfo info;
          if (!GetGmshElementInfo(element_type, &info)) {
            MeshReadResult result;
            result.ok = false;
            result.error = "unsupported gmsh element type";
            return result;
          }
          const int start = mesh.cell_offsets.back();
          for (int n = 0; n < info.node_count; ++n) {
            long long node_id = 0;
            in >> node_id;
            auto it = node_index.find(node_id);
            if (it == node_index.end()) {
              MeshReadResult result;
              result.ok = false;
              result.error = "element references unknown node";
              return result;
            }
            mesh.cell_connectivity.push_back(it->second);
          }
          mesh.cell_offsets.push_back(start + info.node_count);
          mesh.cell_types.push_back(info.vtk_type);
        }
        std::string end_token;
        in >> end_token;
      }
    } else if (!token.empty() && token[0] == '$') {
      const std::string end_token = "$End" + token.substr(1);
      SkipSection(in, end_token);
    }
  }

  if (mesh.points.empty()) {
    MeshReadResult result;
    result.ok = false;
    result.error = "missing node data";
    return result;
  }
  if (mesh.cell_offsets.size() <= 1) {
    MeshReadResult result;
    result.ok = false;
    result.error = "missing element data";
    return result;
  }

  MeshReadResult result;
  result.ok = true;
  result.mesh = std::move(mesh);
  return result;
}
}  // namespace

MeshSummary SummarizeUnstructuredMesh(const UnstructuredMesh& mesh) {
  MeshSummary summary;
  const int point_count = static_cast<int>(mesh.points.size() / 3);
  summary.point_count = point_count;
  summary.cell_count = mesh.cell_offsets.empty()
                           ? static_cast<int>(mesh.cell_types.size())
                           : static_cast<int>(mesh.cell_offsets.size()) - 1;
  if (point_count == 0) {
    return summary;
  }
  double xmin = mesh.points[0];
  double xmax = mesh.points[0];
  double ymin = mesh.points[1];
  double ymax = mesh.points[1];
  double zmin = mesh.points[2];
  double zmax = mesh.points[2];
  for (int i = 1; i < point_count; ++i) {
    const double x = mesh.points[static_cast<size_t>(i) * 3];
    const double y = mesh.points[static_cast<size_t>(i) * 3 + 1];
    const double z = mesh.points[static_cast<size_t>(i) * 3 + 2];
    xmin = std::min(xmin, x);
    xmax = std::max(xmax, x);
    ymin = std::min(ymin, y);
    ymax = std::max(ymax, y);
    zmin = std::min(zmin, z);
    zmax = std::max(zmax, z);
  }
  summary.xmin = xmin;
  summary.xmax = xmax;
  summary.ymin = ymin;
  summary.ymax = ymax;
  summary.zmin = zmin;
  summary.zmax = zmax;
  const double dz = zmax - zmin;
  const double dy = ymax - ymin;
  summary.dimension = (dz > 1e-12) ? 3 : (dy > 1e-12 ? 2 : 1);
  return summary;
}

MeshReadResult ReadUnstructuredMesh(const std::string& path,
                                    const std::string& format_hint) {
  std::string format = pde::ToLower(format_hint);
  if (format.empty()) {
    const std::filesystem::path mesh_path(path);
    format = pde::ToLower(StripLeadingDot(mesh_path.extension().string()));
  } else {
    format = StripLeadingDot(format);
  }

  if (format == "vtk") {
    return ReadLegacyVtkUnstructuredGrid(path);
  }
  if (format == "msh") {
    return ReadGmshMesh(path);
  }

  MeshReadResult result;
  result.ok = false;
  result.error = "unsupported mesh format (expected .vtk or .msh)";
  return result;
}

MeshWriteResult WriteVtkUnstructuredGrid(const std::string& path,
                                         const UnstructuredMesh& mesh) {
  const int point_count = static_cast<int>(mesh.points.size() / 3);
  if (point_count <= 0) {
    return {false, "mesh contains no points"};
  }
  if (!mesh.cell_offsets.empty() &&
      static_cast<size_t>(mesh.cell_offsets.size()) < 2) {
    return {false, "mesh cell offsets are incomplete"};
  }
  const int cell_count = mesh.cell_offsets.empty()
                             ? static_cast<int>(mesh.cell_types.size())
                             : static_cast<int>(mesh.cell_offsets.size()) - 1;
  if (cell_count < 0) {
    return {false, "invalid cell count"};
  }
  if (!mesh.cell_types.empty() &&
      static_cast<int>(mesh.cell_types.size()) != cell_count) {
    return {false, "cell type count mismatch"};
  }

  std::ofstream out(path);
  if (!out) {
    return {false, "failed to open output file"};
  }

  out << "# vtk DataFile Version 3.0\n";
  out << "pde_unstructured_mesh\n";
  out << "ASCII\n";
  out << "DATASET UNSTRUCTURED_GRID\n";
  out << "POINTS " << point_count << " float\n";
  out << std::fixed;
  for (int i = 0; i < point_count; ++i) {
    const size_t idx = static_cast<size_t>(i) * 3;
    out << static_cast<float>(mesh.points[idx]) << " "
        << static_cast<float>(mesh.points[idx + 1]) << " "
        << static_cast<float>(mesh.points[idx + 2]) << "\n";
  }

  if (!mesh.cell_offsets.empty()) {
    const int total_size =
        static_cast<int>(mesh.cell_connectivity.size()) + cell_count;
    out << "CELLS " << cell_count << " " << total_size << "\n";
    for (int cell = 0; cell < cell_count; ++cell) {
      const int start = mesh.cell_offsets[static_cast<size_t>(cell)];
      const int end = mesh.cell_offsets[static_cast<size_t>(cell + 1)];
      const int count = end - start;
      out << count;
      for (int i = start; i < end; ++i) {
        out << " " << mesh.cell_connectivity[static_cast<size_t>(i)];
      }
      out << "\n";
    }
  }

  if (!mesh.cell_types.empty()) {
    out << "CELL_TYPES " << cell_count << "\n";
    for (int cell = 0; cell < cell_count; ++cell) {
      out << mesh.cell_types[static_cast<size_t>(cell)] << "\n";
    }
  }

  const bool has_point_scalars =
      mesh.point_scalars.size() == static_cast<size_t>(point_count);
  const bool has_cell_scalars =
      mesh.cell_scalars.size() == static_cast<size_t>(cell_count);

  if (has_point_scalars) {
    const std::string name =
        mesh.point_scalar_name.empty() ? "scalar" : mesh.point_scalar_name;
    out << "POINT_DATA " << point_count << "\n";
    out << "SCALARS " << name << " float 1\n";
    out << "LOOKUP_TABLE default\n";
    for (int i = 0; i < point_count; ++i) {
      out << static_cast<float>(mesh.point_scalars[static_cast<size_t>(i)]) << "\n";
    }
  }

  if (has_cell_scalars) {
    const std::string name =
        mesh.cell_scalar_name.empty() ? "cell_scalar" : mesh.cell_scalar_name;
    out << "CELL_DATA " << cell_count << "\n";
    out << "SCALARS " << name << " float 1\n";
    out << "LOOKUP_TABLE default\n";
    for (int i = 0; i < cell_count; ++i) {
      out << static_cast<float>(mesh.cell_scalars[static_cast<size_t>(i)]) << "\n";
    }
  }

  return {true, ""};
}
