#ifndef RUN_CONFIG_H
#define RUN_CONFIG_H

#include <filesystem>
#include <string>

#include "pde_types.h"

struct RunConfig {
  int schema_version = 1;

  std::string pde_latex;
  std::string domain_bounds;  // "xmin,xmax,ymin,ymax[,zmin,zmax]"
  std::string grid;           // "nx,ny[,nz]"
  std::string boundary_spec;  // "left:dirichlet:0;right:..."
  std::string domain_shape;
  std::string domain_shape_file;
  std::string domain_shape_mask;
  double domain_shape_mask_threshold = 0.0;
  bool domain_shape_mask_invert = false;
  ShapeTransform shape_transform;
  std::string domain_mesh;
  std::string domain_mesh_format;
  std::string domain_mesh_discretization;
  std::string coord_mode;     // GUI coordinate mode token (optional).
  std::string domain_mode;    // "box" or "implicit" (optional).

  std::string backend = "auto";        // "auto|cpu|cuda|metal|tpu"
  std::string method = "jacobi";       // "jacobi|gs|sor|cg|bicgstab|gmres|mg"

  SolverConfig solver;
  TimeConfig time;

  std::string output_path;            // explicit output file path
  std::string output_dir;            // output directory (used when output_path empty)
  std::string output_format = "vtk";  // "vtk" or "vti"
};

bool LoadRunConfigFromFile(const std::filesystem::path& path,
                           RunConfig* config,
                           std::string* error);
bool LoadRunConfigFromString(const std::string& content,
                             RunConfig* config,
                             std::string* error);
bool SaveRunConfigToFile(const std::filesystem::path& path,
                         const RunConfig& config,
                         std::string* error);
std::string SerializeRunConfig(const RunConfig& config, int indent = 2);

bool ValidateRunConfig(const RunConfig& config, std::string* error);
std::string DefaultBoundarySpec();

#endif  // RUN_CONFIG_H
