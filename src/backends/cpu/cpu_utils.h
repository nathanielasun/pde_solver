#ifndef CPU_UTILS_H
#define CPU_UTILS_H

#include <functional>
#include <string>
#include <vector>

#include "coefficient_evaluator.h"
#include "coordinate_metrics.h"
#include "finite_differences.h"
#include "pde_types.h"

inline int Index(int i, int j, int nx) {
  return j * nx + i;
}

inline int Index3D(int i, int j, int k, int nx, int ny) {
  return (k * ny + j) * nx + i;
}

double Dot(const std::vector<double>& a, const std::vector<double>& b);
double Norm2(const std::vector<double>& v);
double NormInf(const std::vector<double>& v);

double ComputeIntegralValue(const std::vector<double>& grid, int nx, int ny,
                            const std::vector<unsigned char>* active, double dx, double dy);
double ComputeIntegralValue3D(const std::vector<double>& grid, int nx, int ny, int nz,
                              const std::vector<unsigned char>* active, double dx, double dy,
                              double dz);

void Ax2D(const Domain& d,
          const BoundarySet& bc,
          const std::vector<unsigned char>* active,
          const std::vector<double>* integral_weights,
          double ax, double by, double cx, double dyc, double center,
          double ab,
          const std::vector<double>& x,
          std::vector<double>* y_out);

struct LinearOperator2D {
  bool ok = true;
  std::string error;

  const Domain* domain = nullptr;
  const std::vector<unsigned char>* active = nullptr;
  const std::vector<double>* integral_weights = nullptr;
  CoefficientEvaluator coeff_eval;
  bool has_var_coeff = false;
  bool has_integrals = false;
  bool has_mixed = false;
  bool has_high_order = false;
  int nx = 0;
  int ny = 0;
  double dx = 0.0;
  double dy = 0.0;

  double a = 0.0;
  double b = 0.0;
  double c = 0.0;
  double dcoef = 0.0;
  double e = 0.0;
  double ab = 0.0;
  double a3 = 0.0;
  double b3 = 0.0;
  double a4 = 0.0;
  double b4 = 0.0;

  double ax_const = 0.0;
  double by_const = 0.0;
  double cx_const = 0.0;
  double dyc_const = 0.0;
  double center_const = 0.0;

  void Apply(const std::vector<double>& x, std::vector<double>* y_out) const;
};

LinearOperator2D BuildLinearOperator2D(const SolveInput& input,
                                       const Domain& d,
                                       const std::vector<unsigned char>* active,
                                       const std::vector<double>* integral_weights);

// Evaluate condition expression (returns true if condition is satisfied)
bool EvalCondition(const std::string& condition_latex, double x, double y, double z, double t);
bool EvalCondition(const std::string& condition_latex, double x, double y);

// Get the appropriate BC for a point, checking piecewise conditions first
const BoundaryCondition& GetBCForPoint(
    const BoundaryCondition& default_bc,
    const std::vector<PiecewiseBoundaryCondition>& piecewise,
    double x, double y, double z = 0.0, double t = 0.0);

// Centralized boundary condition application
void ApplyDirichletCPU(const SolveInput& input, const Domain& d, double dx, double dy,
                       std::vector<double>* grid,
                       const std::function<bool(int, int)>& is_active, double t = 0.0);
void ApplyNeumannRobinCPU(const SolveInput& input, const Domain& d, double dx, double dy,
                          std::vector<double>* grid,
                          const std::function<bool(int, int)>& is_active, double t = 0.0);

void ApplyDirichletCPU3D(const SolveInput& input, const Domain& d, double dx, double dy, double dz,
                         std::vector<double>* grid,
                         const std::function<bool(int, int, int)>& is_active, double t = 0.0);
void ApplyNeumannRobinCPU3D(const SolveInput& input, const Domain& d, double dx, double dy, double dz,
                            std::vector<double>* grid,
                            const std::function<bool(int, int, int)>& is_active, double t = 0.0);

                       
                       #endif  // CPU_UTILS_H
