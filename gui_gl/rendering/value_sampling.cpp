#include "value_sampling.h"

#include "vtk_io.h"  // for DerivedFields

double SampleFieldValue(size_t idx, GlViewer::FieldType field_type,
                        const std::vector<double>& grid,
                        const struct DerivedFields* derived,
                        bool has_derived) {
  if (field_type == GlViewer::FieldType::Solution || !has_derived || !derived) {
    return idx < grid.size() ? grid[idx] : 0.0;
  }

  switch (field_type) {
    case GlViewer::FieldType::GradientX:
      return idx < derived->gradient_x.size() ? derived->gradient_x[idx] : 0.0;
    case GlViewer::FieldType::GradientY:
      return idx < derived->gradient_y.size() ? derived->gradient_y[idx] : 0.0;
    case GlViewer::FieldType::GradientZ:
      return idx < derived->gradient_z.size() ? derived->gradient_z[idx] : 0.0;
    case GlViewer::FieldType::Laplacian:
      return idx < derived->laplacian.size() ? derived->laplacian[idx] : 0.0;
    case GlViewer::FieldType::FluxX:
      return idx < derived->flux_x.size() ? derived->flux_x[idx] : 0.0;
    case GlViewer::FieldType::FluxY:
      return idx < derived->flux_y.size() ? derived->flux_y[idx] : 0.0;
    case GlViewer::FieldType::FluxZ:
      return idx < derived->flux_z.size() ? derived->flux_z[idx] : 0.0;
    case GlViewer::FieldType::EnergyNorm:
      return idx < derived->energy_norm.size() ? derived->energy_norm[idx] : 0.0;
    default:
      return idx < grid.size() ? grid[idx] : 0.0;
  }
}

