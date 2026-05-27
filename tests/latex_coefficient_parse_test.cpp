#include "latex_parser.h"

#include <cstdlib>
#include <iostream>
#include <string>

namespace {

bool ExpectOk(const LatexParseResult& r, const char* label) {
  if (!r.ok) {
    std::cerr << label << " parse failed: " << r.error << "\n";
    return false;
  }
  return true;
}

}  // namespace

int main() {
  LatexParser parser;

  {
    const LatexParseResult r = parser.Parse("k u_{xx} = 0");
    if (!ExpectOk(r, "k u_xx")) {
      return 1;
    }
    if (r.coeffs.a_latex != "k") {
      std::cerr << "expected a_latex=k got '" << r.coeffs.a_latex << "'\n";
      return 1;
    }
  }

  {
    const LatexParseResult r = parser.Parse("0.5*sin(x)*u_{xx} = 0");
    if (!ExpectOk(r, "sin coeff")) {
      return 1;
    }
    if (r.coeffs.a_latex.find("sin") == std::string::npos &&
        std::abs(r.coeffs.a - 0.5) > 1e-6) {
      std::cerr << "expected sin coeff in a_latex or a=0.5, got a_latex='" << r.coeffs.a_latex
                << "' a=" << r.coeffs.a << "\n";
      return 1;
    }
  }

  {
    const LatexParseResult r = parser.Parse("3.14 = 0");
    if (!ExpectOk(r, "numeric rhs")) {
      return 1;
    }
    if (std::abs(r.coeffs.f - 3.14) > 1e-6) {
      std::cerr << "expected f=3.14 got " << r.coeffs.f << "\n";
      return 1;
    }
  }

  {
    const LatexParseResult r = parser.Parse("u_t + u u_x = 0");
    if (!ExpectOk(r, "burgers")) {
      return 1;
    }
  }

  {
    const LatexParseResult r = parser.Parse("\\frac{1}{Re}\\nabla^2 u = 0");
    if (!ExpectOk(r, "laplacian coeff")) {
      return 1;
    }
    if (r.coeffs.a_latex.empty() && r.coeffs.b_latex.empty() && r.coeffs.az_latex.empty()) {
      std::cerr << "expected variable laplacian coefficient latex\n";
      return 1;
    }
  }

  std::cout << "latex_coefficient_parse_test passed\n";
  return 0;
}
