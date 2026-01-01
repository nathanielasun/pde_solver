#include "mms.h"

#include <cmath>
#include <iomanip>
#include <sstream>

namespace {
constexpr double kEps = 1e-12;
constexpr double kPi = 3.14159265358979323846;

std::string FormatDouble(double value) {
  if (std::abs(value) < kEps) {
    value = 0.0;
  }
  std::ostringstream oss;
  oss << std::setprecision(17) << value;
  return oss.str();
}

std::string Wrap(const std::string& expr) {
  return "(" + expr + ")";
}

std::string SinPi(const std::string& var) {
  return "\\sin(\\pi*" + var + ")";
}

std::string CosPi(const std::string& var) {
  return "\\cos(\\pi*" + var + ")";
}

void AppendTerm(std::string* expr, double coeff, const std::string& term) {
  if (!expr || term.empty() || std::abs(coeff) < kEps) {
    return;
  }
  const bool negative = coeff < 0.0;
  const double abs_coeff = std::abs(coeff);
  std::string part = Wrap(term);
  if (std::abs(abs_coeff - 1.0) > kEps) {
    part = FormatDouble(abs_coeff) + "*" + part;
  }
  if (expr->empty()) {
    if (negative) {
      expr->push_back('-');
    }
    expr->append(part);
    return;
  }
  expr->push_back(negative ? '-' : '+');
  expr->append(part);
}
}  // namespace

ManufacturedSolution BuildDefaultManufacturedSolution(int dimension) {
  ManufacturedSolution sol;
  sol.dimension = dimension;

  const std::string sin_x = SinPi("x");
  const std::string sin_y = SinPi("y");
  const std::string cos_x = CosPi("x");
  const std::string cos_y = CosPi("y");

  if (dimension <= 2) {
    sol.u_latex = sin_x + "*" + sin_y;
    sol.u_x_latex = "\\pi*" + cos_x + "*" + sin_y;
    sol.u_y_latex = "\\pi*" + sin_x + "*" + cos_y;
    sol.u_xx_latex = "-\\pi^2*" + sin_x + "*" + sin_y;
    sol.u_yy_latex = "-\\pi^2*" + sin_x + "*" + sin_y;
    sol.eval = [](double x, double y, double) {
      return std::sin(kPi * x) * std::sin(kPi * y);
    };
    return sol;
  }

  const std::string sin_z = SinPi("z");
  const std::string cos_z = CosPi("z");

  sol.u_latex = sin_x + "*" + sin_y + "*" + sin_z;
  sol.u_x_latex = "\\pi*" + cos_x + "*" + sin_y + "*" + sin_z;
  sol.u_y_latex = "\\pi*" + sin_x + "*" + cos_y + "*" + sin_z;
  sol.u_z_latex = "\\pi*" + sin_x + "*" + sin_y + "*" + cos_z;
  sol.u_xx_latex = "-\\pi^2*" + sin_x + "*" + sin_y + "*" + sin_z;
  sol.u_yy_latex = "-\\pi^2*" + sin_x + "*" + sin_y + "*" + sin_z;
  sol.u_zz_latex = "-\\pi^2*" + sin_x + "*" + sin_y + "*" + sin_z;
  sol.eval = [](double x, double y, double z) {
    return std::sin(kPi * x) * std::sin(kPi * y) * std::sin(kPi * z);
  };
  return sol;
}

ManufacturedRhsResult BuildManufacturedRhs(const PDECoefficients& pde,
                                           int dimension,
                                           const ManufacturedSolution& solution) {
  ManufacturedRhsResult out;
  std::string lhs;
  AppendTerm(&lhs, pde.a, solution.u_xx_latex);
  AppendTerm(&lhs, pde.b, solution.u_yy_latex);
  if (dimension > 2) {
    AppendTerm(&lhs, pde.az, solution.u_zz_latex);
  }
  AppendTerm(&lhs, pde.c, solution.u_x_latex);
  AppendTerm(&lhs, pde.d, solution.u_y_latex);
  if (dimension > 2) {
    AppendTerm(&lhs, pde.dz, solution.u_z_latex);
  }
  AppendTerm(&lhs, pde.e, solution.u_latex);

  if (lhs.empty()) {
    out.ok = false;
    out.error = "MMS requires at least one supported PDE term";
    return out;
  }

  out.ok = true;
  out.rhs_latex = "-(" + lhs + ")";
  return out;
}

bool ValidateMmsInput(const SolveInput& input, std::string* error) {
  if (input.domain.coord_system != CoordinateSystem::Cartesian) {
    if (error) {
      *error = "MMS only supports Cartesian coordinates";
    }
    return false;
  }
  if (HasImplicitShape(input)) {
    if (error) {
      *error = "MMS does not support implicit domain shapes";
    }
    return false;
  }
  if (input.time.enabled || std::abs(input.pde.ut) > kEps || std::abs(input.pde.utt) > kEps) {
    if (error) {
      *error = "MMS only supports steady-state PDEs";
    }
    return false;
  }
  if (HasVariableCoefficients(input.pde)) {
    if (error) {
      *error = "MMS requires constant coefficients";
    }
    return false;
  }
  if (HasMixedDerivatives(input.pde) || HasHigherOrderDerivatives(input.pde)) {
    if (error) {
      *error = "MMS does not support mixed or higher-order derivatives";
    }
    return false;
  }
  if (!input.integrals.empty()) {
    if (error) {
      *error = "MMS does not support integral terms";
    }
    return false;
  }
  if (!input.nonlinear.empty() || !input.nonlinear_derivatives.empty()) {
    if (error) {
      *error = "MMS does not support nonlinear terms";
    }
    return false;
  }
  if (input.domain.nz <= 1) {
    if (std::abs(input.pde.az) > kEps || std::abs(input.pde.dz) > kEps) {
      if (error) {
        *error = "MMS 2D mode cannot include z-derivative terms";
      }
      return false;
    }
  }
  return true;
}
