#ifndef MESH_IO_H
#define MESH_IO_H

#include <string>
#include <vector>

struct UnstructuredMesh {
  std::vector<double> points;  // xyz triples.
  std::vector<int> cell_offsets;  // Prefix sum offsets (size = cell_count + 1).
  std::vector<int> cell_connectivity;
  std::vector<int> cell_types;  // VTK cell type codes.

  std::vector<double> point_scalars;
  std::string point_scalar_name;
  std::vector<double> cell_scalars;
  std::string cell_scalar_name;
};

struct MeshSummary {
  int point_count = 0;
  int cell_count = 0;
  int dimension = 0;
  double xmin = 0.0;
  double xmax = 0.0;
  double ymin = 0.0;
  double ymax = 0.0;
  double zmin = 0.0;
  double zmax = 0.0;
};

struct MeshReadResult {
  bool ok = false;
  std::string error;
  std::string warning;
  UnstructuredMesh mesh;
};

struct MeshWriteResult {
  bool ok = false;
  std::string error;
};

MeshSummary SummarizeUnstructuredMesh(const UnstructuredMesh& mesh);
MeshReadResult ReadUnstructuredMesh(const std::string& path,
                                    const std::string& format_hint = "");
MeshWriteResult WriteVtkUnstructuredGrid(const std::string& path,
                                         const UnstructuredMesh& mesh);

#endif  // MESH_IO_H
