#ifndef VTK_IO_H
#define VTK_IO_H

#include <cstddef>
#include <string>
#include <vector>

#include "pde_types.h"
#include "progress.h"

// Forward declaration for multi-field VTK writer
struct DerivedFields;

struct VtkWriteResult {
  bool ok = false;
  std::string error;
};

struct VtkReadResult {
  bool ok = false;
  std::string error;
  enum class Kind {
    StructuredPoints,
    PointCloud,
  };
  Kind kind = Kind::StructuredPoints;
  Domain domain;
  std::vector<double> grid;
  std::vector<PointSample> points;
};

VtkWriteResult WriteVtkStructuredPoints(const std::string& path, const Domain& domain,
                                        const std::vector<double>& grid,
                                        const ProgressCallback& progress = ProgressCallback());
VtkWriteResult WriteVtkXmlImageData(const std::string& path, const Domain& domain,
                                    const std::vector<double>& grid,
                                    const std::vector<std::vector<double>>* derived_fields = nullptr,
                                    const std::vector<std::string>* field_names = nullptr,
                                    const ProgressCallback& progress = ProgressCallback());
// Multi-field VTK writer for coupled PDE systems
VtkWriteResult WriteVtkXmlImageDataMultiField(
    const std::string& path, const Domain& domain,
    const std::vector<FieldOutput>& field_outputs,
    const std::vector<DerivedFields>* per_field_derived = nullptr,
    const ProgressCallback& progress = ProgressCallback());
VtkWriteResult WriteVtkSeriesPvd(const std::string& path,
                                 const std::vector<std::string>& frame_paths,
                                 const std::vector<double>& times);
VtkReadResult ReadVtkStructuredPoints(const std::string& path);
VtkReadResult ReadVtkFile(const std::string& path);
std::string GenerateRandomTag(size_t length);

// Compute derived fields from solution
struct DerivedFields {
  std::vector<double> gradient_x;  // ∂u/∂x
  std::vector<double> gradient_y;  // ∂u/∂y
  std::vector<double> gradient_z;  // ∂u/∂z (3D only)
  std::vector<double> laplacian;   // ∇²u
  std::vector<double> flux_x;     // -a*∂u/∂x (for diffusion flux)
  std::vector<double> flux_y;     // -b*∂u/∂y
  std::vector<double> flux_z;     // -c*∂u/∂z (3D only)
  std::vector<double> energy_norm; // u² (for energy norm computation)
};

DerivedFields ComputeDerivedFields(const Domain& domain, const std::vector<double>& grid,
                                    double a = 1.0, double b = 1.0, double c = 1.0);

// Checkpoint/restart for time-dependent runs
struct CheckpointData {
  Domain domain;
  std::vector<double> grid;
  std::vector<double> velocity;
  double t_current;
  int frame_index;
  PDECoefficients pde;
  BoundarySet bc;

  // Multi-field checkpoint support (version 3+)
  std::vector<FieldDefinition> fields;
  std::vector<std::vector<double>> field_grids;
  std::vector<std::vector<double>> field_velocities;
};

VtkWriteResult WriteCheckpoint(const std::string& path, const CheckpointData& checkpoint);
VtkReadResult ReadCheckpoint(const std::string& path, CheckpointData* checkpoint);

#endif
