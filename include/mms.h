#ifndef MMS_H
#define MMS_H

#include <functional>
#include <string>

#include "pde_types.h"

struct ManufacturedSolution {
  int dimension = 2;
  std::string u_latex;
  std::string u_x_latex;
  std::string u_y_latex;
  std::string u_z_latex;
  std::string u_xx_latex;
  std::string u_yy_latex;
  std::string u_zz_latex;
  std::function<double(double, double, double)> eval;
};

struct ManufacturedRhsResult {
  bool ok = false;
  std::string error;
  std::string rhs_latex;
};

ManufacturedSolution BuildDefaultManufacturedSolution(int dimension);
ManufacturedRhsResult BuildManufacturedRhs(const PDECoefficients& pde,
                                           int dimension,
                                           const ManufacturedSolution& solution);
bool ValidateMmsInput(const SolveInput& input, std::string* error);

#endif  // MMS_H
