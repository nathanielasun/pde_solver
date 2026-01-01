#include "vtk_io.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <random>
#include <sstream>
#include <string>
#include <vector>

std::string GenerateRandomTag(size_t length) {
  static constexpr char kAlphabet[] = "abcdefghijklmnopqrstuvwxyz0123456789";
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<size_t> dist(0, sizeof(kAlphabet) - 2);
  std::string out;
  out.reserve(length);
  for (size_t i = 0; i < length; ++i) {
    out.push_back(kAlphabet[dist(gen)]);
  }
  return out;
}

VtkWriteResult WriteVtkStructuredPoints(const std::string& path, const Domain& domain,
                                        const std::vector<double>& grid,
                                        const ProgressCallback& progress) {
  const int nx = domain.nx;
  const int ny = domain.ny;
  const int nz = std::max(1, domain.nz);
  if (nx < 1 || ny < 1) {
    return {false, "invalid grid dimensions"};
  }
  if (grid.size() != static_cast<size_t>(nx * ny * nz)) {
    return {false, "grid size does not match domain"};
  }
  const double dx = (domain.xmax - domain.xmin) / static_cast<double>(std::max(1, nx - 1));
  const double dy = (domain.ymax - domain.ymin) / static_cast<double>(std::max(1, ny - 1));
  const double dz = (domain.zmax - domain.zmin) / static_cast<double>(std::max(1, nz - 1));

  std::ofstream out(path);
  if (!out) {
    return {false, "failed to open output file"};
  }

  out << "# vtk DataFile Version 3.0\n";
  out << "pde_solution\n";
  out << "ASCII\n";
  out << "DATASET STRUCTURED_POINTS\n";
  out << "DIMENSIONS " << nx << " " << ny << " " << nz << "\n";
  out << "ORIGIN " << domain.xmin << " " << domain.ymin << " " << domain.zmin << "\n";
  out << "SPACING " << dx << " " << dy << " " << dz << "\n";
  out << "POINT_DATA " << (nx * ny * nz) << "\n";
  out << "SCALARS solution float 1\n";
  out << "LOOKUP_TABLE default\n";

  out << std::setprecision(6) << std::fixed;
  if (progress) {
    progress("write", 0.0);
  }
  int plane_count = 0;
  const int total_planes = std::max(1, ny * nz);
  for (int k = 0; k < nz; ++k) {
    for (int j = 0; j < ny; ++j) {
      for (int i = 0; i < nx; ++i) {
        const size_t idx = static_cast<size_t>((k * ny + j) * nx + i);
        out << static_cast<float>(grid[idx]) << " ";
      }
      out << "\n";
      if (progress) {
        ++plane_count;
        const double fraction = static_cast<double>(plane_count) /
                                static_cast<double>(total_planes);
        progress("write", fraction);
      }
    }
  }

  return {true, ""};
}

VtkWriteResult WriteVtkSeriesPvd(const std::string& path,
                                 const std::vector<std::string>& frame_paths,
                                 const std::vector<double>& times) {
  if (frame_paths.empty()) {
    return {false, "no frames to write"};
  }
  if (!times.empty() && times.size() != frame_paths.size()) {
    return {false, "frame times count mismatch"};
  }

  std::ofstream out(path);
  if (!out) {
    return {false, "failed to open series file"};
  }

  const std::filesystem::path base_dir = std::filesystem::path(path).parent_path();
  out << "<?xml version=\"1.0\"?>\n";
  out << "<VTKFile type=\"Collection\" version=\"0.1\" byte_order=\"LittleEndian\">\n";
  out << "  <Collection>\n";
  out << std::setprecision(6) << std::fixed;

  for (size_t i = 0; i < frame_paths.size(); ++i) {
    const std::filesystem::path frame_path(frame_paths[i]);
    std::string file_name = frame_path.string();
    std::error_code ec;
    const std::filesystem::path rel = std::filesystem::relative(frame_path, base_dir, ec);
    if (!ec) {
      file_name = rel.string();
    }
    const double t = times.empty() ? static_cast<double>(i) : times[i];
    out << "    <DataSet timestep=\"" << t << "\" group=\"\" part=\"0\" file=\""
        << file_name << "\"/>\n";
  }

  out << "  </Collection>\n";
  out << "</VTKFile>\n";
  return {true, ""};
}

namespace {
// Base64 encoding for binary data in VTK XML
std::string Base64Encode(const unsigned char* data, size_t len) {
  static const char kBase64Chars[] =
      "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  std::string result;
  result.reserve(((len + 2) / 3) * 4);
  for (size_t i = 0; i < len; i += 3) {
    unsigned int b = (data[i] << 16);
    if (i + 1 < len) b |= (data[i + 1] << 8);
    if (i + 2 < len) b |= data[i + 2];
    result += kBase64Chars[(b >> 18) & 63];
    result += kBase64Chars[(b >> 12) & 63];
    if (i + 1 < len) {
      result += kBase64Chars[(b >> 6) & 63];
    } else {
      result += '=';
    }
    if (i + 2 < len) {
      result += kBase64Chars[b & 63];
    } else {
      result += '=';
    }
  }
  return result;
}

// Write binary data as base64-encoded (uncompressed for simplicity)
// VTK XML binary format: [header_size (uint64)][data_size (uint64)][data...]
std::string EncodeBinaryData(const float* data, size_t count) {
  const size_t data_size = count * sizeof(float);
  const uint64_t header_size = static_cast<uint64_t>(sizeof(uint64_t));
  const uint64_t total_size = header_size + data_size;
  
  std::vector<unsigned char> buffer(sizeof(uint64_t) * 2 + data_size);
  // Write header_size
  std::memcpy(buffer.data(), &header_size, sizeof(uint64_t));
  // Write data_size
  std::memcpy(buffer.data() + sizeof(uint64_t), &data_size, sizeof(uint64_t));
  // Write data
  std::memcpy(buffer.data() + 2 * sizeof(uint64_t), data, data_size);
  
  (void)total_size;
  return Base64Encode(buffer.data(), buffer.size());
}

int Index2D(int i, int j, int nx) {
  return j * nx + i;
}

int Index3D(int i, int j, int k, int nx, int ny) {
  return (k * ny + j) * nx + i;
}
}  // namespace

VtkWriteResult WriteVtkXmlImageData(const std::string& path, const Domain& domain,
                                    const std::vector<double>& grid,
                                    const std::vector<std::vector<double>>* derived_fields,
                                    const std::vector<std::string>* field_names,
                                    const ProgressCallback& progress) {
  const int nx = domain.nx;
  const int ny = domain.ny;
  const int nz = std::max(1, domain.nz);
  if (nx < 1 || ny < 1) {
    return {false, "invalid grid dimensions"};
  }
  if (grid.size() != static_cast<size_t>(nx * ny * nz)) {
    return {false, "grid size does not match domain"};
  }
  const double dx = (domain.xmax - domain.xmin) / static_cast<double>(std::max(1, nx - 1));
  const double dy = (domain.ymax - domain.ymin) / static_cast<double>(std::max(1, ny - 1));
  const double dz = (domain.zmax - domain.zmin) / static_cast<double>(std::max(1, nz - 1));

  std::ofstream out(path);
  if (!out) {
    return {false, "failed to open output file"};
  }

  out << std::setprecision(9) << std::scientific;
  out << "<?xml version=\"1.0\"?>\n";
  out << "<VTKFile type=\"ImageData\" version=\"1.0\" byte_order=\"LittleEndian\" "
      << "header_type=\"UInt64\">\n";
  out << "  <ImageData WholeExtent=\"0 " << (nx - 1) << " 0 " << (ny - 1) << " 0 " << (nz - 1)
      << "\" Origin=\"" << domain.xmin << " " << domain.ymin << " " << domain.zmin
      << "\" Spacing=\"" << dx << " " << dy << " " << dz << "\">\n";
  out << "    <Piece Extent=\"0 " << (nx - 1) << " 0 " << (ny - 1) << " 0 " << (nz - 1) << "\">\n";
  out << "      <PointData>\n";

  if (progress) {
    progress("write", 0.0);
  }

  // Write solution field
  std::vector<float> float_grid(grid.size());
  for (size_t i = 0; i < grid.size(); ++i) {
    float_grid[i] = static_cast<float>(grid[i]);
  }
  std::string encoded = EncodeBinaryData(float_grid.data(), float_grid.size());
  out << "        <DataArray type=\"Float32\" Name=\"solution\" NumberOfComponents=\"1\" "
      << "format=\"binary\">\n";
  out << encoded << "\n";
  out << "        </DataArray>\n";

  // Write derived fields if provided
  if (derived_fields && field_names && derived_fields->size() == field_names->size()) {
    for (size_t f = 0; f < derived_fields->size(); ++f) {
      const auto& field = (*derived_fields)[f];
      if (field.size() != grid.size()) {
        continue;
      }
      std::vector<float> float_field(field.size());
      for (size_t i = 0; i < field.size(); ++i) {
        float_field[i] = static_cast<float>(field[i]);
      }
      encoded = EncodeBinaryData(float_field.data(), float_field.size());
      out << "        <DataArray type=\"Float32\" Name=\"" << (*field_names)[f]
          << "\" NumberOfComponents=\"1\" format=\"binary\">\n";
      out << encoded << "\n";
      out << "        </DataArray>\n";
    }
  }

  out << "      </PointData>\n";
  out << "      <CellData/>\n";
  out << "    </Piece>\n";
  out << "  </ImageData>\n";
  out << "</VTKFile>\n";

  if (progress) {
    progress("write", 1.0);
  }

  return {true, ""};
}

VtkWriteResult WriteVtkXmlImageDataMultiField(
    const std::string& path, const Domain& domain,
    const std::vector<FieldOutput>& field_outputs,
    const std::vector<DerivedFields>* per_field_derived,
    const ProgressCallback& progress) {
  if (field_outputs.empty()) {
    return {false, "no fields to write"};
  }

  const int nx = domain.nx;
  const int ny = domain.ny;
  const int nz = std::max(1, domain.nz);
  if (nx < 1 || ny < 1) {
    return {false, "invalid grid dimensions"};
  }
  const size_t expected_size = static_cast<size_t>(nx * ny * nz);
  for (const auto& field : field_outputs) {
    if (field.grid.size() != expected_size) {
      return {false, "field '" + field.name + "' size does not match domain"};
    }
  }
  const double dx = (domain.xmax - domain.xmin) / static_cast<double>(std::max(1, nx - 1));
  const double dy = (domain.ymax - domain.ymin) / static_cast<double>(std::max(1, ny - 1));
  const double dz = (domain.zmax - domain.zmin) / static_cast<double>(std::max(1, nz - 1));

  std::ofstream out(path);
  if (!out) {
    return {false, "failed to open output file"};
  }

  out << std::setprecision(9) << std::scientific;
  out << "<?xml version=\"1.0\"?>\n";
  out << "<VTKFile type=\"ImageData\" version=\"1.0\" byte_order=\"LittleEndian\" "
      << "header_type=\"UInt64\">\n";
  out << "  <ImageData WholeExtent=\"0 " << (nx - 1) << " 0 " << (ny - 1) << " 0 " << (nz - 1)
      << "\" Origin=\"" << domain.xmin << " " << domain.ymin << " " << domain.zmin
      << "\" Spacing=\"" << dx << " " << dy << " " << dz << "\">\n";
  out << "    <Piece Extent=\"0 " << (nx - 1) << " 0 " << (ny - 1) << " 0 " << (nz - 1) << "\">\n";
  out << "      <PointData>\n";

  if (progress) {
    progress("write", 0.0);
  }

  // Write each field
  for (size_t f = 0; f < field_outputs.size(); ++f) {
    const auto& field = field_outputs[f];
    // First field is named "solution" for backward compatibility
    std::string field_name = (f == 0) ? "solution" : field.name;

    std::vector<float> float_grid(field.grid.size());
    for (size_t i = 0; i < field.grid.size(); ++i) {
      float_grid[i] = static_cast<float>(field.grid[i]);
    }
    std::string encoded = EncodeBinaryData(float_grid.data(), float_grid.size());
    out << "        <DataArray type=\"Float32\" Name=\"" << field_name
        << "\" NumberOfComponents=\"1\" format=\"binary\">\n";
    out << encoded << "\n";
    out << "        </DataArray>\n";

    // Write derived fields for this field if provided
    if (per_field_derived && f < per_field_derived->size()) {
      const auto& derived = (*per_field_derived)[f];
      std::string prefix = (f == 0) ? "" : field.name + "_";

      auto write_derived = [&](const std::vector<double>& data, const std::string& name) {
        if (data.size() != expected_size) return;
        std::vector<float> float_data(data.size());
        for (size_t i = 0; i < data.size(); ++i) {
          float_data[i] = static_cast<float>(data[i]);
        }
        std::string enc = EncodeBinaryData(float_data.data(), float_data.size());
        out << "        <DataArray type=\"Float32\" Name=\"" << prefix << name
            << "\" NumberOfComponents=\"1\" format=\"binary\">\n";
        out << enc << "\n";
        out << "        </DataArray>\n";
      };

      write_derived(derived.gradient_x, "gradient_x");
      write_derived(derived.gradient_y, "gradient_y");
      if (nz > 1) {
        write_derived(derived.gradient_z, "gradient_z");
      }
      write_derived(derived.laplacian, "laplacian");
      write_derived(derived.flux_x, "flux_x");
      write_derived(derived.flux_y, "flux_y");
      if (nz > 1) {
        write_derived(derived.flux_z, "flux_z");
      }
      write_derived(derived.energy_norm, "energy_norm");
    }

    if (progress) {
      progress("write", static_cast<double>(f + 1) / static_cast<double>(field_outputs.size()));
    }
  }

  out << "      </PointData>\n";
  out << "      <CellData/>\n";
  out << "    </Piece>\n";
  out << "  </ImageData>\n";
  out << "</VTKFile>\n";

  return {true, ""};
}

DerivedFields ComputeDerivedFields(const Domain& domain, const std::vector<double>& grid,
                                    double a, double b, double c) {
  const int nx = domain.nx;
  const int ny = domain.ny;
  const int nz = std::max(1, domain.nz);
  const double dx = (domain.xmax - domain.xmin) / static_cast<double>(std::max(1, nx - 1));
  const double dy = (domain.ymax - domain.ymin) / static_cast<double>(std::max(1, ny - 1));
  const double dz = (domain.zmax - domain.zmin) / static_cast<double>(std::max(1, nz - 1));

  const double inv_dx = dx > 0.0 ? 1.0 / dx : 0.0;
  const double inv_dy = dy > 0.0 ? 1.0 / dy : 0.0;
  const double inv_dz = dz > 0.0 ? 1.0 / dz : 0.0;
  const double inv_dx2 = dx > 0.0 ? 1.0 / (dx * dx) : 0.0;
  const double inv_dy2 = dy > 0.0 ? 1.0 / (dy * dy) : 0.0;
  const double inv_dz2 = dz > 0.0 ? 1.0 / (dz * dz) : 0.0;

  DerivedFields fields;
  fields.gradient_x.resize(grid.size(), 0.0);
  fields.gradient_y.resize(grid.size(), 0.0);
  fields.gradient_z.resize(grid.size(), 0.0);
  fields.laplacian.resize(grid.size(), 0.0);
  fields.flux_x.resize(grid.size(), 0.0);
  fields.flux_y.resize(grid.size(), 0.0);
  fields.flux_z.resize(grid.size(), 0.0);
  fields.energy_norm.resize(grid.size(), 0.0);

  if (domain.nz > 1) {
    // 3D case
    for (int k = 1; k < nz - 1; ++k) {
      for (int j = 1; j < ny - 1; ++j) {
        for (int i = 1; i < nx - 1; ++i) {
          const size_t idx = static_cast<size_t>(Index3D(i, j, k, nx, ny));
          const double u = grid[idx];
          const double u_left = grid[Index3D(i - 1, j, k, nx, ny)];
          const double u_right = grid[Index3D(i + 1, j, k, nx, ny)];
          const double u_down = grid[Index3D(i, j - 1, k, nx, ny)];
          const double u_up = grid[Index3D(i, j + 1, k, nx, ny)];
          const double u_back = grid[Index3D(i, j, k - 1, nx, ny)];
          const double u_front = grid[Index3D(i, j, k + 1, nx, ny)];

          fields.gradient_x[idx] = (u_right - u_left) * 0.5 * inv_dx;
          fields.gradient_y[idx] = (u_up - u_down) * 0.5 * inv_dy;
          fields.gradient_z[idx] = (u_front - u_back) * 0.5 * inv_dz;
          fields.laplacian[idx] = (u_right - 2.0 * u + u_left) * inv_dx2 +
                                  (u_up - 2.0 * u + u_down) * inv_dy2 +
                                  (u_front - 2.0 * u + u_back) * inv_dz2;
          fields.flux_x[idx] = -a * fields.gradient_x[idx];
          fields.flux_y[idx] = -b * fields.gradient_y[idx];
          fields.flux_z[idx] = -c * fields.gradient_z[idx];
          fields.energy_norm[idx] = u * u;
        }
      }
    }
  } else {
    // 2D case
    for (int j = 1; j < ny - 1; ++j) {
      for (int i = 1; i < nx - 1; ++i) {
        const size_t idx = static_cast<size_t>(Index2D(i, j, nx));
        const double u = grid[idx];
        const double u_left = grid[Index2D(i - 1, j, nx)];
        const double u_right = grid[Index2D(i + 1, j, nx)];
        const double u_down = grid[Index2D(i, j - 1, nx)];
        const double u_up = grid[Index2D(i, j + 1, nx)];

        fields.gradient_x[idx] = (u_right - u_left) * 0.5 * inv_dx;
        fields.gradient_y[idx] = (u_up - u_down) * 0.5 * inv_dy;
        fields.laplacian[idx] = (u_right - 2.0 * u + u_left) * inv_dx2 +
                               (u_up - 2.0 * u + u_down) * inv_dy2;
        fields.flux_x[idx] = -a * fields.gradient_x[idx];
        fields.flux_y[idx] = -b * fields.gradient_y[idx];
        fields.energy_norm[idx] = u * u;
      }
    }
  }

  return fields;
}

VtkWriteResult WriteCheckpoint(const std::string& path, const CheckpointData& checkpoint) {
  std::ofstream out(path);
  if (!out) {
    return {false, "failed to open checkpoint file"};
  }
  if (!checkpoint.velocity.empty() && checkpoint.velocity.size() != checkpoint.grid.size()) {
    return {false, "checkpoint velocity size mismatch"};
  }

  out << std::setprecision(17) << std::scientific;
  out << "# PDE Solver Checkpoint\n";
  out << "# Format: domain, grid, optional velocity, time state, PDE coefficients, boundary conditions\n";
  out << "CHECKPOINT_VERSION 2\n";

  // Write domain
  out << "DOMAIN " << checkpoint.domain.xmin << " " << checkpoint.domain.xmax << " "
      << checkpoint.domain.ymin << " " << checkpoint.domain.ymax << " "
      << checkpoint.domain.zmin << " " << checkpoint.domain.zmax << " "
      << checkpoint.domain.nx << " " << checkpoint.domain.ny << " " << checkpoint.domain.nz << "\n";

  // Write time state
  out << "TIME " << checkpoint.t_current << " " << checkpoint.frame_index << "\n";

  // Write PDE coefficients
  out << "PDE " << checkpoint.pde.a << " " << checkpoint.pde.b << " " << checkpoint.pde.az << " "
      << checkpoint.pde.c << " " << checkpoint.pde.d << " " << checkpoint.pde.dz << " "
      << checkpoint.pde.e << " " << checkpoint.pde.f << " " << checkpoint.pde.ut << " "
      << checkpoint.pde.utt << "\n";
  if (!checkpoint.pde.rhs_latex.empty()) {
    out << "RHS_LATEX " << checkpoint.pde.rhs_latex << "\n";
  }

  // Write grid
  out << "GRID " << checkpoint.grid.size() << "\n";
  for (size_t i = 0; i < checkpoint.grid.size(); ++i) {
    out << checkpoint.grid[i] << " ";
    if ((i + 1) % 10 == 0) {
      out << "\n";
    }
  }
  if (checkpoint.grid.size() % 10 != 0) {
    out << "\n";
  }

  if (!checkpoint.velocity.empty()) {
    out << "VELOCITY " << checkpoint.velocity.size() << "\n";
    for (size_t i = 0; i < checkpoint.velocity.size(); ++i) {
      out << checkpoint.velocity[i] << " ";
      if ((i + 1) % 10 == 0) {
        out << "\n";
      }
    }
    if (checkpoint.velocity.size() % 10 != 0) {
      out << "\n";
    }
  }

  // Write boundary conditions (simplified - just write kind and constant value for now)
  out << "BC_LEFT " << static_cast<int>(checkpoint.bc.left.kind) << " "
      << checkpoint.bc.left.value.constant << "\n";
  out << "BC_RIGHT " << static_cast<int>(checkpoint.bc.right.kind) << " "
      << checkpoint.bc.right.value.constant << "\n";
  out << "BC_BOTTOM " << static_cast<int>(checkpoint.bc.bottom.kind) << " "
      << checkpoint.bc.bottom.value.constant << "\n";
  out << "BC_TOP " << static_cast<int>(checkpoint.bc.top.kind) << " "
      << checkpoint.bc.top.value.constant << "\n";
  if (checkpoint.domain.nz > 1) {
    out << "BC_FRONT " << static_cast<int>(checkpoint.bc.front.kind) << " "
        << checkpoint.bc.front.value.constant << "\n";
    out << "BC_BACK " << static_cast<int>(checkpoint.bc.back.kind) << " "
        << checkpoint.bc.back.value.constant << "\n";
  }

  return {true, ""};
}

