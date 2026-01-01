#include "embedded_boundary.h"

#include <cmath>
#include <optional>
#include "expression_eval.h"
#include "shape_utils.h"

namespace {
const double kEps = 1e-12;
const double kBisectionTol = 1e-10;
const int kBisectionMaxIter = 50;
const double kNormalStep = 1e-6;
const int kSampleGrid2D = 5;
const int kSampleGrid3D = 3;

struct ShapeEvaluator {
  const ExpressionEvaluator* expr = nullptr;
  const ShapeMask* mask = nullptr;
  ShapeTransform transform;
  double mask_threshold = 0.0;
  bool mask_invert = false;
  bool is_3d = false;

  double Eval(double x, double y, double z) const {
    double tx = x;
    double ty = y;
    double tz = z;
    ApplyShapeTransform(transform, x, y, z, &tx, &ty, &tz);
    if (mask && HasShapeMask(*mask)) {
      return SampleShapeMaskPhi(*mask, tx, ty, tz, mask_threshold, mask_invert);
    }
    if (expr) {
      return expr->Eval(tx, ty, is_3d ? tz : 0.0);
    }
    return 1.0;
  }
};

// Helper function to evaluate boundary condition expressions
double EvalBCExpr(const BoundaryCondition::Expression& expr, double x, double y, double z = 0.0) {
  if (!expr.latex.empty()) {
    ExpressionEvaluator evaluator = ExpressionEvaluator::ParseLatex(expr.latex);
    if (evaluator.ok()) {
      return evaluator.Eval(x, y, z);
    }
  }
  return expr.constant + expr.x * x + expr.y * y + expr.z * z;
}

double DistanceSquared(double x0, double y0, double z0, double x1, double y1, double z1) {
  const double dx = x0 - x1;
  const double dy = y0 - y1;
  const double dz = z0 - z1;
  return dx * dx + dy * dy + dz * dz;
}

void SetIntercept(
    const ShapeEvaluator& evaluator,
    double x,
    double y,
    double z,
    double fraction,
    BoundaryIntercept* intercept) {
  intercept->valid = true;
  intercept->fraction = fraction;
  intercept->x = x;
  intercept->y = y;
  intercept->z = z;

  const double inv = 1.0 / (2.0 * kNormalStep);
  const double phi_x = (evaluator.Eval(x + kNormalStep, y, z) -
                        evaluator.Eval(x - kNormalStep, y, z)) * inv;
  const double phi_y = (evaluator.Eval(x, y + kNormalStep, z) -
                        evaluator.Eval(x, y - kNormalStep, z)) * inv;
  const double phi_z = evaluator.is_3d
      ? (evaluator.Eval(x, y, z + kNormalStep) -
         evaluator.Eval(x, y, z - kNormalStep)) * inv
      : 0.0;
  const double norm = std::sqrt(phi_x * phi_x + phi_y * phi_y + phi_z * phi_z);
  if (norm > kEps) {
    intercept->normal_x = phi_x / norm;
    intercept->normal_y = phi_y / norm;
    intercept->normal_z = phi_z / norm;
  } else {
    intercept->normal_x = 1.0;
    intercept->normal_y = 0.0;
    intercept->normal_z = 0.0;
  }
}

double EstimateVolumeFraction2D(
    const ShapeEvaluator& evaluator,
    double x_left,
    double x_right,
    double y_bottom,
    double y_top,
    int samples,
    bool* any_inside,
    bool* any_outside) {
  const double dx = (x_right - x_left) / static_cast<double>(samples);
  const double dy = (y_top - y_bottom) / static_cast<double>(samples);
  int inside_count = 0;
  for (int j = 0; j < samples; ++j) {
      const double y = y_bottom + (j + 0.5) * dy;
      for (int i = 0; i < samples; ++i) {
        const double x = x_left + (i + 0.5) * dx;
        if (evaluator.Eval(x, y, 0.0) <= 0.0) {
          ++inside_count;
        }
      }
  }
  const int total = samples * samples;
  if (any_inside) {
    *any_inside = inside_count > 0;
  }
  if (any_outside) {
    *any_outside = inside_count < total;
  }
  return static_cast<double>(inside_count) / static_cast<double>(total);
}

double EstimateVolumeFraction3D(
    const ShapeEvaluator& evaluator,
    double x_left,
    double x_right,
    double y_bottom,
    double y_top,
    double z_back,
    double z_front,
    int samples,
    bool* any_inside,
    bool* any_outside) {
  const double dx = (x_right - x_left) / static_cast<double>(samples);
  const double dy = (y_top - y_bottom) / static_cast<double>(samples);
  const double dz = (z_front - z_back) / static_cast<double>(samples);
  int inside_count = 0;
  for (int k = 0; k < samples; ++k) {
    const double z = z_back + (k + 0.5) * dz;
    for (int j = 0; j < samples; ++j) {
        const double y = y_bottom + (j + 0.5) * dy;
        for (int i = 0; i < samples; ++i) {
          const double x = x_left + (i + 0.5) * dx;
          if (evaluator.Eval(x, y, z) <= 0.0) {
            ++inside_count;
          }
        }
    }
  }
  const int total = samples * samples * samples;
  if (any_inside) {
    *any_inside = inside_count > 0;
  }
  if (any_outside) {
    *any_outside = inside_count < total;
  }
  return static_cast<double>(inside_count) / static_cast<double>(total);
}
// Find boundary intercept along an edge using bisection
// Returns true if boundary found, false otherwise
bool FindBoundaryIntercept(
    const ShapeEvaluator& evaluator,
    double x0, double y0, double z0,  // start point
    double x1, double y1, double z1,  // end point
    BoundaryIntercept* intercept) {
  const double phi0 = evaluator.Eval(x0, y0, z0);
  const double phi1 = evaluator.Eval(x1, y1, z1);

  // If one endpoint is exactly on boundary, use it
  if (std::abs(phi0) < kEps) {
    SetIntercept(evaluator, x0, y0, z0, 0.0, intercept);
    return true;
  }
  if (std::abs(phi1) < kEps) {
    SetIntercept(evaluator, x1, y1, z1, 1.0, intercept);
    return true;
  }

  // Check if edge crosses boundary (sign change)
  if (phi0 * phi1 > 0.0) {
    return false;  // No crossing
  }

  // Bisection to find boundary
  double t0 = 0.0;
  double t1 = 1.0;
  for (int iter = 0; iter < kBisectionMaxIter; ++iter) {
    const double t_mid = 0.5 * (t0 + t1);
    const double x_mid = x0 + t_mid * (x1 - x0);
    const double y_mid = y0 + t_mid * (y1 - y0);
    const double z_mid = z0 + t_mid * (z1 - z0);
    const double phi_mid = evaluator.Eval(x_mid, y_mid, z_mid);

    if (std::abs(phi_mid) < kEps || std::abs(t1 - t0) < kBisectionTol) {
      SetIntercept(evaluator, x_mid, y_mid, z_mid, t_mid, intercept);
      return true;
    }

    if (phi0 * phi_mid < 0.0) {
      t1 = t_mid;
    } else {
      t0 = t_mid;
    }
  }

  return false;
}

void UpdateClosestIntercept(
    const BoundaryIntercept& candidate,
    double x_center,
    double y_center,
    double z_center,
    BoundaryIntercept* target) {
  if (!candidate.valid) {
    return;
  }
  if (!target->valid) {
    *target = candidate;
    return;
  }
  const double dist_new = DistanceSquared(candidate.x, candidate.y, candidate.z,
                                          x_center, y_center, z_center);
  const double dist_old = DistanceSquared(target->x, target->y, target->z,
                                          x_center, y_center, z_center);
  if (dist_new < dist_old) {
    *target = candidate;
  }
}

void TryUpdateIntercept(
    const ShapeEvaluator& evaluator,
    double x0, double y0, double z0,
    double x1, double y1, double z1,
    double x_center, double y_center, double z_center,
    BoundaryIntercept* target) {
  BoundaryIntercept candidate;
  if (!FindBoundaryIntercept(evaluator, x0, y0, z0, x1, y1, z1, &candidate)) {
    return;
  }
  UpdateClosestIntercept(candidate, x_center, y_center, z_center, target);
}
}  // namespace

int Index2D(int i, int j, int nx) {
  return j * nx + i;
}

int Index3D(int i, int j, int k, int nx, int ny) {
  return (k * ny + j) * nx + i;
}

bool BuildEmbeddedBoundary2D(
    const Domain& domain,
    const std::string& domain_shape,
    const ShapeMask& shape_mask,
    const ShapeTransform& shape_transform,
    double shape_mask_threshold,
    bool shape_mask_invert,
    std::vector<CellBoundaryInfo>* boundary_info,
    std::string* error) {
  if (!boundary_info) {
    if (error) {
      *error = "missing boundary storage for embedded domain";
    }
    return false;
  }
  const bool use_mask = HasShapeMask(shape_mask);
  if (!use_mask && domain_shape.empty()) {
    if (error) {
      *error = "missing implicit domain shape";
    }
    return false;
  }

  std::optional<ExpressionEvaluator> evaluator;
  if (!use_mask) {
    ExpressionEvaluator parsed = ExpressionEvaluator::ParseLatex(domain_shape);
    if (!parsed.ok()) {
      if (error) {
        *error = "invalid domain shape: " + parsed.error();
      }
      return false;
    }
    evaluator.emplace(std::move(parsed));
  }

  ShapeEvaluator shape_eval;
  shape_eval.expr = use_mask ? nullptr : &(*evaluator);
  shape_eval.mask = use_mask ? &shape_mask : nullptr;
  shape_eval.transform = shape_transform;
  shape_eval.mask_threshold = shape_mask_threshold;
  shape_eval.mask_invert = shape_mask_invert;
  shape_eval.is_3d = false;

  const int nx = domain.nx;
  const int ny = domain.ny;
  const double dx = (domain.xmax - domain.xmin) / static_cast<double>(nx - 1);
  const double dy = (domain.ymax - domain.ymin) / static_cast<double>(ny - 1);

  boundary_info->resize(static_cast<size_t>(nx * ny));

  for (int j = 0; j < ny; ++j) {
    const double y = domain.ymin + j * dy;
    for (int i = 0; i < nx; ++i) {
      const double x = domain.xmin + i * dx;
      const size_t idx = static_cast<size_t>(Index2D(i, j, nx));
      CellBoundaryInfo& info = (*boundary_info)[idx];
      info = CellBoundaryInfo();

      const bool center_inside = shape_eval.Eval(x, y, 0.0) <= 0.0;
      info.center_inside = center_inside;

      const double x_left = x - 0.5 * dx;
      const double x_right = x + 0.5 * dx;
      const double y_bottom = y - 0.5 * dy;
      const double y_top = y + 0.5 * dy;

      bool any_inside = false;
      bool any_outside = false;
      info.volume_fraction = EstimateVolumeFraction2D(
          shape_eval, x_left, x_right, y_bottom, y_top,
          kSampleGrid2D, &any_inside, &any_outside);
      info.is_cut_cell = any_inside && any_outside;

      if (!info.is_cut_cell) {
        continue;
      }

      // Find boundary intercepts on edges
      // Left edge: (x_left, y_bottom) to (x_left, y_top)
      if (i > 0) {
        FindBoundaryIntercept(shape_eval, x_left, y_bottom, 0.0, x_left, y_top, 0.0, &info.left);
      }

      // Right edge: (x_right, y_bottom) to (x_right, y_top)
      if (i < nx - 1) {
        FindBoundaryIntercept(shape_eval, x_right, y_bottom, 0.0, x_right, y_top, 0.0, &info.right);
      }

      // Bottom edge: (x_left, y_bottom) to (x_right, y_bottom)
      if (j > 0) {
        FindBoundaryIntercept(shape_eval, x_left, y_bottom, 0.0, x_right, y_bottom, 0.0, &info.bottom);
      }

      // Top edge: (x_left, y_top) to (x_right, y_top)
      if (j < ny - 1) {
        FindBoundaryIntercept(shape_eval, x_left, y_top, 0.0, x_right, y_top, 0.0, &info.top);
      }

    }
  }

  return true;
}

bool BuildEmbeddedBoundary3D(
    const Domain& domain,
    const std::string& domain_shape,
    const ShapeMask& shape_mask,
    const ShapeTransform& shape_transform,
    double shape_mask_threshold,
    bool shape_mask_invert,
    std::vector<CellBoundaryInfo>* boundary_info,
    std::string* error) {
  if (!boundary_info) {
    if (error) {
      *error = "missing boundary storage for embedded domain";
    }
    return false;
  }
  const bool use_mask = HasShapeMask(shape_mask);
  if (!use_mask && domain_shape.empty()) {
    if (error) {
      *error = "missing implicit domain shape";
    }
    return false;
  }

  std::optional<ExpressionEvaluator> evaluator;
  if (!use_mask) {
    ExpressionEvaluator parsed = ExpressionEvaluator::ParseLatex(domain_shape);
    if (!parsed.ok()) {
      if (error) {
        *error = "invalid domain shape: " + parsed.error();
      }
      return false;
    }
    evaluator.emplace(std::move(parsed));
  }

  ShapeEvaluator shape_eval;
  shape_eval.expr = use_mask ? nullptr : &(*evaluator);
  shape_eval.mask = use_mask ? &shape_mask : nullptr;
  shape_eval.transform = shape_transform;
  shape_eval.mask_threshold = shape_mask_threshold;
  shape_eval.mask_invert = shape_mask_invert;
  shape_eval.is_3d = true;

  const int nx = domain.nx;
  const int ny = domain.ny;
  const int nz = domain.nz;
  const double dx = (domain.xmax - domain.xmin) / static_cast<double>(nx - 1);
  const double dy = (domain.ymax - domain.ymin) / static_cast<double>(ny - 1);
  const double dz = (domain.zmax - domain.zmin) / static_cast<double>(nz - 1);

  boundary_info->resize(static_cast<size_t>(nx * ny * nz));

  for (int k = 0; k < nz; ++k) {
    const double z = domain.zmin + k * dz;
    for (int j = 0; j < ny; ++j) {
      const double y = domain.ymin + j * dy;
      for (int i = 0; i < nx; ++i) {
        const double x = domain.xmin + i * dx;
        const size_t idx = static_cast<size_t>(Index3D(i, j, k, nx, ny));
        CellBoundaryInfo& info = (*boundary_info)[idx];
        info = CellBoundaryInfo();

        const bool center_inside = shape_eval.Eval(x, y, z) <= 0.0;
        info.center_inside = center_inside;

        const double x_left = x - 0.5 * dx;
        const double x_right = x + 0.5 * dx;
        const double y_bottom = y - 0.5 * dy;
        const double y_top = y + 0.5 * dy;
        const double z_back = z - 0.5 * dz;
        const double z_front = z + 0.5 * dz;

        bool any_inside = false;
        bool any_outside = false;
        info.volume_fraction = EstimateVolumeFraction3D(
            shape_eval, x_left, x_right, y_bottom, y_top, z_back, z_front,
            kSampleGrid3D, &any_inside, &any_outside);
        info.is_cut_cell = any_inside && any_outside;

        if (!info.is_cut_cell) {
          continue;
        }

        // Find boundary intercepts on edges (12 edges of a cube).
        if (i > 0) {
          TryUpdateIntercept(shape_eval, x_left, y_bottom, z_back,
                             x_left, y_bottom, z_front, x, y, z, &info.left);
          TryUpdateIntercept(shape_eval, x_left, y_top, z_back,
                             x_left, y_top, z_front, x, y, z, &info.left);
          TryUpdateIntercept(shape_eval, x_left, y_bottom, z_back,
                             x_left, y_top, z_back, x, y, z, &info.left);
          TryUpdateIntercept(shape_eval, x_left, y_bottom, z_front,
                             x_left, y_top, z_front, x, y, z, &info.left);
        }
        if (i < nx - 1) {
          TryUpdateIntercept(shape_eval, x_right, y_bottom, z_back,
                             x_right, y_bottom, z_front, x, y, z, &info.right);
          TryUpdateIntercept(shape_eval, x_right, y_top, z_back,
                             x_right, y_top, z_front, x, y, z, &info.right);
          TryUpdateIntercept(shape_eval, x_right, y_bottom, z_back,
                             x_right, y_top, z_back, x, y, z, &info.right);
          TryUpdateIntercept(shape_eval, x_right, y_bottom, z_front,
                             x_right, y_top, z_front, x, y, z, &info.right);
        }
        if (j > 0) {
          TryUpdateIntercept(shape_eval, x_left, y_bottom, z_back,
                             x_right, y_bottom, z_back, x, y, z, &info.bottom);
          TryUpdateIntercept(shape_eval, x_left, y_bottom, z_front,
                             x_right, y_bottom, z_front, x, y, z, &info.bottom);
          TryUpdateIntercept(shape_eval, x_left, y_bottom, z_back,
                             x_left, y_bottom, z_front, x, y, z, &info.bottom);
          TryUpdateIntercept(shape_eval, x_right, y_bottom, z_back,
                             x_right, y_bottom, z_front, x, y, z, &info.bottom);
        }
        if (j < ny - 1) {
          TryUpdateIntercept(shape_eval, x_left, y_top, z_back,
                             x_right, y_top, z_back, x, y, z, &info.top);
          TryUpdateIntercept(shape_eval, x_left, y_top, z_front,
                             x_right, y_top, z_front, x, y, z, &info.top);
          TryUpdateIntercept(shape_eval, x_left, y_top, z_back,
                             x_left, y_top, z_front, x, y, z, &info.top);
          TryUpdateIntercept(shape_eval, x_right, y_top, z_back,
                             x_right, y_top, z_front, x, y, z, &info.top);
        }
        if (k > 0) {
          TryUpdateIntercept(shape_eval, x_left, y_bottom, z_back,
                             x_right, y_bottom, z_back, x, y, z, &info.back);
          TryUpdateIntercept(shape_eval, x_left, y_top, z_back,
                             x_right, y_top, z_back, x, y, z, &info.back);
          TryUpdateIntercept(shape_eval, x_left, y_bottom, z_back,
                             x_left, y_top, z_back, x, y, z, &info.back);
          TryUpdateIntercept(shape_eval, x_right, y_bottom, z_back,
                             x_right, y_top, z_back, x, y, z, &info.back);
        }
        if (k < nz - 1) {
          TryUpdateIntercept(shape_eval, x_left, y_bottom, z_front,
                             x_right, y_bottom, z_front, x, y, z, &info.front);
          TryUpdateIntercept(shape_eval, x_left, y_top, z_front,
                             x_right, y_top, z_front, x, y, z, &info.front);
          TryUpdateIntercept(shape_eval, x_left, y_bottom, z_front,
                             x_left, y_top, z_front, x, y, z, &info.front);
          TryUpdateIntercept(shape_eval, x_right, y_bottom, z_front,
                             x_right, y_top, z_front, x, y, z, &info.front);
        }

      }
    }
  }

  return true;
}

void ApplyEmbeddedBoundaryBC2D(
    const BoundaryCondition& bc,
    const CellBoundaryInfo& cell_info,
    int i, int j,
    const Domain& domain,
    double dx, double dy,
    std::vector<double>* grid) {
  const int nx = domain.nx;
  const int idx = Index2D(i, j, nx);

  const double x = domain.xmin + i * dx;
  const double y = domain.ymin + j * dy;

  // Find the boundary intercept closest to the cell center.
  const BoundaryIntercept* intercept = nullptr;
  double min_dist = 1e30;
  auto consider = [&](const BoundaryIntercept& candidate) {
    if (!candidate.valid) {
      return;
    }
    const double dist = DistanceSquared(candidate.x, candidate.y, 0.0, x, y, 0.0);
    if (dist < min_dist) {
      min_dist = dist;
      intercept = &candidate;
    }
  };
  consider(cell_info.left);
  consider(cell_info.right);
  consider(cell_info.bottom);
  consider(cell_info.top);

  if (!intercept || !intercept->valid) {
    return;
  }

  const double x_b = intercept->x;
  const double y_b = intercept->y;
  const double n_x = intercept->normal_x;
  const double n_y = intercept->normal_y;

  if (bc.kind == BCKind::Dirichlet) {
    // Dirichlet: u = g on boundary
    // Use second-order extrapolation: find ghost point value
    // such that linear interpolation to boundary gives g
    const double g = EvalBCExpr(bc.value, x_b, y_b);
    const double dist_to_boundary = std::sqrt((x_b - x) * (x_b - x) + (y_b - y) * (y_b - y));
    if (dist_to_boundary < kEps) {
      // Cell center is on boundary
      (*grid)[static_cast<size_t>(idx)] = g;
      return;
    }

    // Find neighbor point inside domain
    int i_neighbor = i;
    int j_neighbor = j;
    if (n_x > 0.5) {
      i_neighbor = i - 1;  // boundary is to the right, neighbor is left
    } else if (n_x < -0.5) {
      i_neighbor = i + 1;  // boundary is to the left, neighbor is right
    }
    if (n_y > 0.5) {
      j_neighbor = j - 1;  // boundary is above, neighbor is below
    } else if (n_y < -0.5) {
      j_neighbor = j + 1;  // boundary is below, neighbor is above
    }

    if (i_neighbor >= 0 && i_neighbor < domain.nx && j_neighbor >= 0 && j_neighbor < domain.ny) {
      const int idx_neighbor = Index2D(i_neighbor, j_neighbor, domain.nx);
      const double u_neighbor = (*grid)[static_cast<size_t>(idx_neighbor)];
      const double x_n = domain.xmin + i_neighbor * dx;
      const double y_n = domain.ymin + j_neighbor * dy;
      const double dist_neighbor = std::sqrt((x_b - x_n) * (x_b - x_n) + (y_b - y_n) * (y_b - y_n));
      
      // Linear interpolation: u_b = (dist_to_boundary / total_dist) * u_neighbor + (dist_neighbor / total_dist) * u_ghost
      // Solving for u_ghost: u_ghost = (g * total_dist - dist_to_boundary * u_neighbor) / dist_neighbor
      const double total_dist = dist_to_boundary + dist_neighbor;
      if (total_dist > kEps) {
        const double u_ghost = (g * total_dist - dist_to_boundary * u_neighbor) / dist_neighbor;
        (*grid)[static_cast<size_t>(idx)] = u_ghost;
      } else {
        (*grid)[static_cast<size_t>(idx)] = g;
      }
    } else {
      // No valid neighbor, use direct assignment
      (*grid)[static_cast<size_t>(idx)] = g;
    }
  } else if (bc.kind == BCKind::Neumann) {
    // Neumann: ∂u/∂n = g on boundary
    // Use finite difference: (u_ghost - u_neighbor) / dist = g * n
    const double g = EvalBCExpr(bc.value, x_b, y_b);
    int i_neighbor = i;
    int j_neighbor = j;
    if (n_x > 0.5) {
      i_neighbor = i - 1;
    } else if (n_x < -0.5) {
      i_neighbor = i + 1;
    }
    if (n_y > 0.5) {
      j_neighbor = j - 1;
    } else if (n_y < -0.5) {
      j_neighbor = j + 1;
    }

    if (i_neighbor >= 0 && i_neighbor < domain.nx && j_neighbor >= 0 && j_neighbor < domain.ny) {
      const int idx_neighbor = Index2D(i_neighbor, j_neighbor, domain.nx);
      const double u_neighbor = (*grid)[static_cast<size_t>(idx_neighbor)];
      const double x_n = domain.xmin + i_neighbor * dx;
      const double y_n = domain.ymin + j_neighbor * dy;
      const double dist = std::sqrt((x - x_n) * (x - x_n) + (y - y_n) * (y - y_n));
      if (dist > kEps) {
        // ∂u/∂n = (u_ghost - u_neighbor) / dist * (n · direction)
        // For normal pointing outward: u_ghost = u_neighbor + g * dist
        const double u_ghost = u_neighbor + g * dist;
        (*grid)[static_cast<size_t>(idx)] = u_ghost;
      }
    }
  } else if (bc.kind == BCKind::Robin) {
    // Robin: α*u + β*∂u/∂n = γ on boundary
    // Combine Dirichlet and Neumann approaches
    const double alpha = EvalBCExpr(bc.alpha, x_b, y_b);
    const double beta = EvalBCExpr(bc.beta, x_b, y_b);
    const double gamma = EvalBCExpr(bc.gamma, x_b, y_b);

    int i_neighbor = i;
    int j_neighbor = j;
    if (n_x > 0.5) {
      i_neighbor = i - 1;
    } else if (n_x < -0.5) {
      i_neighbor = i + 1;
    }
    if (n_y > 0.5) {
      j_neighbor = j - 1;
    } else if (n_y < -0.5) {
      j_neighbor = j + 1;
    }

    if (i_neighbor >= 0 && i_neighbor < domain.nx && j_neighbor >= 0 && j_neighbor < domain.ny) {
      const int idx_neighbor = Index2D(i_neighbor, j_neighbor, domain.nx);
      const double u_neighbor = (*grid)[static_cast<size_t>(idx_neighbor)];
      const double x_n = domain.xmin + i_neighbor * dx;
      const double y_n = domain.ymin + j_neighbor * dy;
      const double dist = std::sqrt((x - x_n) * (x - x_n) + (y - y_n) * (y - y_n));
      if (dist > kEps) {
        // α*u_b + β*(u_ghost - u_neighbor)/dist = γ
        // Approximate u_b ≈ (u_ghost + u_neighbor)/2
        // Solve: α*(u_ghost + u_neighbor)/2 + β*(u_ghost - u_neighbor)/dist = γ
        // u_ghost * (α/2 + β/dist) = γ - u_neighbor*(α/2 - β/dist)
        const double coeff = alpha / 2.0 + beta / dist;
        if (std::abs(coeff) > kEps) {
          const double u_ghost = (gamma - u_neighbor * (alpha / 2.0 - beta / dist)) / coeff;
          (*grid)[static_cast<size_t>(idx)] = u_ghost;
        }
      }
    }
  }
}

void ApplyEmbeddedBoundaryBC3D(
    const BoundaryCondition& bc,
    const CellBoundaryInfo& cell_info,
    int i, int j, int k,
    const Domain& domain,
    double dx, double dy, double dz,
    std::vector<double>* grid) {
  // Similar to 2D but extended to 3D
  // For now, use a simplified approach
  const int nx = domain.nx;
  const int ny = domain.ny;
  const int idx = Index3D(i, j, k, nx, ny);

  const double x = domain.xmin + i * dx;
  const double y = domain.ymin + j * dy;
  const double z = domain.zmin + k * dz;

  // Find closest boundary intercept.
  const BoundaryIntercept* intercept = nullptr;
  double min_dist = 1e30;
  auto consider = [&](const BoundaryIntercept& candidate) {
    if (!candidate.valid) {
      return;
    }
    const double dist = DistanceSquared(candidate.x, candidate.y, candidate.z, x, y, z);
    if (dist < min_dist) {
      min_dist = dist;
      intercept = &candidate;
    }
  };
  consider(cell_info.left);
  consider(cell_info.right);
  consider(cell_info.bottom);
  consider(cell_info.top);
  consider(cell_info.back);
  consider(cell_info.front);

  if (!intercept || !intercept->valid) {
    return;
  }
  const double x_b = intercept->x;
  const double y_b = intercept->y;
  const double z_b = intercept->z;

  if (bc.kind == BCKind::Dirichlet) {
    const double g = EvalBCExpr(bc.value, x_b, y_b, z_b);
    (*grid)[static_cast<size_t>(idx)] = g;
  }
  // Neumann and Robin can be extended similarly
}
