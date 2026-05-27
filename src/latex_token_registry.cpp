#include "latex_token_registry.h"

#include "latex_patterns.h"

#include <sstream>
#include <string>

namespace {
void AppendPatternArray(std::vector<LatexTokenEntry>* out,
                        const char* category,
                        const char* maps_to,
                        const char* display_note,
                        const char* const* patterns,
                        size_t count) {
  for (size_t i = 0; i < count; ++i) {
    if (patterns[i] == nullptr || patterns[i][0] == '\0') {
      continue;
    }
    out->push_back({category, patterns[i], maps_to, display_note});
  }
}

std::vector<LatexTokenEntry> BuildDerivativeEntries() {
  std::vector<LatexTokenEntry> entries;
  AppendPatternArray(&entries, "derivative_d2x", "PDECoefficients.a", "∂²u/∂x²",
                     LatexPatterns::kD2XPatterns, std::size(LatexPatterns::kD2XPatterns));
  AppendPatternArray(&entries, "derivative_d2y", "PDECoefficients.b", "∂²u/∂y²",
                     LatexPatterns::kD2YPatterns, std::size(LatexPatterns::kD2YPatterns));
  AppendPatternArray(&entries, "derivative_d2z", "PDECoefficients.az", "∂²u/∂z²",
                     LatexPatterns::kD2ZPatterns, std::size(LatexPatterns::kD2ZPatterns));
  AppendPatternArray(&entries, "derivative_dx", "PDECoefficients.c", "∂u/∂x",
                     LatexPatterns::kDXPatterns, std::size(LatexPatterns::kDXPatterns));
  AppendPatternArray(&entries, "derivative_dy", "PDECoefficients.d", "∂u/∂y",
                     LatexPatterns::kDYPatterns, std::size(LatexPatterns::kDYPatterns));
  AppendPatternArray(&entries, "derivative_dz", "PDECoefficients.dz", "∂u/∂z",
                     LatexPatterns::kDZPatterns, std::size(LatexPatterns::kDZPatterns));
  AppendPatternArray(&entries, "derivative_dt", "PDECoefficients.ut", "∂u/∂t",
                     LatexPatterns::kDTPatterns, std::size(LatexPatterns::kDTPatterns));
  AppendPatternArray(&entries, "derivative_d2t", "PDECoefficients.utt", "∂²u/∂t²",
                     LatexPatterns::kD2TPatterns, std::size(LatexPatterns::kD2TPatterns));
  AppendPatternArray(&entries, "derivative_mixed_xy", "PDECoefficients.ab", "∂²u/∂x∂y",
                     LatexPatterns::kDXYPatterns, std::size(LatexPatterns::kDXYPatterns));
  AppendPatternArray(&entries, "derivative_mixed_xz", "PDECoefficients.ac", "∂²u/∂x∂z",
                     LatexPatterns::kDXZPatterns, std::size(LatexPatterns::kDXZPatterns));
  AppendPatternArray(&entries, "derivative_mixed_yz", "PDECoefficients.bc", "∂²u/∂y∂z",
                     LatexPatterns::kDYZPatterns, std::size(LatexPatterns::kDYZPatterns));
  AppendPatternArray(&entries, "derivative_d3x", "PDECoefficients.a3", "∂³u/∂x³",
                     LatexPatterns::kD3XPatterns, std::size(LatexPatterns::kD3XPatterns));
  AppendPatternArray(&entries, "derivative_d3y", "PDECoefficients.b3", "∂³u/∂y³",
                     LatexPatterns::kD3YPatterns, std::size(LatexPatterns::kD3YPatterns));
  AppendPatternArray(&entries, "derivative_d3z", "PDECoefficients.az3", "∂³u/∂z³",
                     LatexPatterns::kD3ZPatterns, std::size(LatexPatterns::kD3ZPatterns));
  AppendPatternArray(&entries, "derivative_d4x", "PDECoefficients.a4", "∂⁴u/∂x⁴",
                     LatexPatterns::kD4XPatterns, std::size(LatexPatterns::kD4XPatterns));
  AppendPatternArray(&entries, "derivative_d4y", "PDECoefficients.b4", "∂⁴u/∂y⁴",
                     LatexPatterns::kD4YPatterns, std::size(LatexPatterns::kD4YPatterns));
  AppendPatternArray(&entries, "derivative_d4z", "PDECoefficients.az4", "∂⁴u/∂z⁴",
                     LatexPatterns::kD4ZPatterns, std::size(LatexPatterns::kD4ZPatterns));
  AppendPatternArray(&entries, "laplacian", "PDECoefficients.a+b(+az)", "∇²u",
                     LatexPatterns::kLaplacianPatterns, std::size(LatexPatterns::kLaplacianPatterns));
  entries.push_back({"state_variable", LatexPatterns::kU, "PDECoefficients.e", "u"});
  return entries;
}

std::vector<LatexTokenEntry> BuildNonlinearEntries() {
  return {
      {"nonlinear_power", "u^{n}", "NonlinearTerm.Power", "u^n (n>=2)"},
      {"nonlinear_sin", "\\sin(u)", "NonlinearTerm.Sin", ""},
      {"nonlinear_sin", "sin(u)", "NonlinearTerm.Sin", ""},
      {"nonlinear_cos", "\\cos(u)", "NonlinearTerm.Cos", ""},
      {"nonlinear_cos", "cos(u)", "NonlinearTerm.Cos", ""},
      {"nonlinear_exp", "\\exp(u)", "NonlinearTerm.Exp", ""},
      {"nonlinear_exp", "exp(u)", "NonlinearTerm.Exp", ""},
      {"nonlinear_abs", "|u|", "NonlinearTerm.Abs", ""},
      {"nonlinear_derivative", "u*u_x", "NonlinearDerivativeTerm.UUx", ""},
      {"nonlinear_derivative", "u*u_y", "NonlinearDerivativeTerm.UUy", ""},
      {"nonlinear_derivative", "u*u_z", "NonlinearDerivativeTerm.UUz", ""},
      {"nonlinear_derivative", "u_x^2", "NonlinearDerivativeTerm.UxUx", ""},
      {"nonlinear_derivative", "u_y^2", "NonlinearDerivativeTerm.UyUy", ""},
      {"nonlinear_derivative", "u_z^2", "NonlinearDerivativeTerm.UzUz", ""},
      {"nonlinear_derivative", "|\\nablau|^2", "NonlinearDerivativeTerm.GradSquared", ""},
      {"nonlinear_derivative", "u_x^2+u_y^2", "NonlinearDerivativeTerm.GradSquared", ""},
      {"nonlinear_derivative", "u_x^2+u_y^2+u_z^2", "NonlinearDerivativeTerm.GradSquared", ""},
  };
}

std::vector<LatexTokenEntry> BuildIntegralEntries() {
  return {
      {"integral", "\\int u", "IntegralTerm", "global integral of u"},
      {"integral", "\\int_{domain} u", "IntegralTerm", "domain integral"},
      {"integral_kernel", "\\int K(x,y) u", "IntegralTerm.kernel_latex", "kernel integral"},
  };
}

std::vector<LatexNormalizationRule> BuildNormalizationRules() {
  return {
      {"\\left", ""},
      {"\\right", ""},
      {"\\cdot", "*"},
      {"\\,", ""},
      {"\\;", ""},
      {"\\:", ""},
      {"\\!", ""},
      {"\\quad", ""},
      {"\\qquad", ""},
      {"$", ""},
      {"\\theta", "theta"},
      {"\\phi", "phi"},
      {"^{2}", "^2"},
      {"_{xx}", "_xx"},
      {"_{yy}", "_yy"},
      {"_{zz}", "_zz"},
      {"_{x}", "_x"},
      {"_{y}", "_y"},
      {"_{t}", "_t"},
      {"_{r}", "_r"},
      {"_{theta}", "_theta"},
      {"_{phi}", "_phi"},
  };
}

std::vector<LatexRewriteRule> BuildConservationRewrites() {
  return {
      {"d/dx(0.5*u^2)", "u*u_x", "Burgers flux divergence"},
      {"d/dx(0.5*u*u)", "u*u_x", "Burgers flux divergence"},
      {"\\frac{d}{dx}(0.5*u^2)", "u*u_x", "Burgers flux divergence"},
  };
}

std::vector<LatexTokenEntry> BuildUnsupportedEntries() {
  return {
      {"unsupported_time_mixed", "u_{xt}", "", "time-space mixed derivative"},
      {"unsupported_time_mixed", "u_{tx}", "", ""},
      {"unsupported_time_mixed", "\\partial_{xt}u", "", ""},
      {"unsupported_time_mixed", "\\frac{\\partial^2u}{\\partialx\\partialt}", "", ""},
      {"unsupported_time_mixed", "d^2u/dxdt", "", ""},
  };
}

std::vector<LatexTokenEntry> BuildDisplayMacroEntries() {
  return {
      {"display_macro", "\\frac{a}{b}", "MicroTeX preview", "fraction"},
      {"display_macro", "\\sin", "MicroTeX preview", "sine"},
      {"display_macro", "\\cos", "MicroTeX preview", "cosine"},
      {"display_macro", "\\exp", "MicroTeX preview", "exponential"},
      {"display_macro", "\\sqrt{x}", "MicroTeX preview", "square root"},
      {"display_macro", "\\partial", "MicroTeX preview", "partial derivative"},
      {"display_macro", "\\nabla", "MicroTeX preview", "nabla"},
      {"display_macro", "u_{xx}", "MicroTeX preview", "subscript"},
      {"display_macro", "x^{2}", "MicroTeX preview", "superscript"},
  };
}

std::vector<LatexTokenEntry> BuildExpressionEvalEntries() {
  return {
      {"expr_function", "sin", "ExpressionEvaluator", ""},
      {"expr_function", "cos", "ExpressionEvaluator", ""},
      {"expr_function", "tan", "ExpressionEvaluator", ""},
      {"expr_function", "exp", "ExpressionEvaluator", ""},
      {"expr_function", "log", "ExpressionEvaluator", ""},
      {"expr_function", "sqrt", "ExpressionEvaluator", ""},
      {"expr_function", "abs", "ExpressionEvaluator", ""},
      {"expr_constant", "pi", "ExpressionEvaluator", ""},
      {"expr_variable", "x", "ExpressionEvaluator", ""},
      {"expr_variable", "y", "ExpressionEvaluator", ""},
      {"expr_variable", "z", "ExpressionEvaluator", ""},
      {"expr_variable", "t", "ExpressionEvaluator", ""},
  };
}

std::string JsonEscape(const std::string& text) {
  std::string out;
  out.reserve(text.size() + 8);
  for (char ch : text) {
    switch (ch) {
      case '\\':
        out += "\\\\";
        break;
      case '"':
        out += "\\\"";
        break;
      case '\n':
        out += "\\n";
        break;
      default:
        out += ch;
        break;
    }
  }
  return out;
}

const std::vector<LatexTokenEntry>& Catalog() {
  static const std::vector<LatexTokenEntry> kCatalog = [] {
    std::vector<LatexTokenEntry> all = BuildDerivativeEntries();
    const auto nonlinear = BuildNonlinearEntries();
    const auto integrals = BuildIntegralEntries();
    const auto display = BuildDisplayMacroEntries();
    all.insert(all.end(), nonlinear.begin(), nonlinear.end());
    all.insert(all.end(), integrals.begin(), integrals.end());
    all.insert(all.end(), display.begin(), display.end());
    return all;
  }();
  return kCatalog;
}

}  // namespace

namespace LatexTokenRegistry {

const std::vector<LatexTokenEntry>& AllEntries() { return Catalog(); }

const std::vector<LatexNormalizationRule>& NormalizationRules() {
  static const std::vector<LatexNormalizationRule> kRules = BuildNormalizationRules();
  return kRules;
}

const std::vector<LatexRewriteRule>& ConservationRewrites() {
  static const std::vector<LatexRewriteRule> kRules = BuildConservationRewrites();
  return kRules;
}

const std::vector<LatexTokenEntry>& UnsupportedPatterns() {
  static const std::vector<LatexTokenEntry> kEntries = BuildUnsupportedEntries();
  return kEntries;
}

const std::vector<LatexTokenEntry>& ExpressionEvalTokens() {
  static const std::vector<LatexTokenEntry> kEntries = BuildExpressionEvalEntries();
  return kEntries;
}

const std::vector<std::string>& ReservedSymbols() {
  static const std::vector<std::string> kReserved = {
      "x", "y", "z", "t", "r", "theta", "phi", "rho"};
  return kReserved;
}

std::string ExportCatalogJson() {
  std::ostringstream oss;
  oss << "{\n";
  oss << "  \"entries\": [\n";
  const auto& entries = AllEntries();
  for (size_t i = 0; i < entries.size(); ++i) {
    const auto& e = entries[i];
    oss << "    {\"category\":\"" << JsonEscape(e.category) << "\",\"pattern\":\""
        << JsonEscape(e.pattern) << "\",\"maps_to\":\"" << JsonEscape(e.maps_to)
        << "\",\"display_note\":\"" << JsonEscape(e.display_note) << "\"}";
    oss << (i + 1 < entries.size() ? ",\n" : "\n");
  }
  oss << "  ],\n";
  oss << "  \"normalization\": [\n";
  const auto& norms = NormalizationRules();
  for (size_t i = 0; i < norms.size(); ++i) {
    oss << "    {\"from\":\"" << JsonEscape(norms[i].from) << "\",\"to\":\""
        << JsonEscape(norms[i].to) << "\"}";
    oss << (i + 1 < norms.size() ? ",\n" : "\n");
  }
  oss << "  ],\n";
  oss << "  \"conservation_rewrites\": [\n";
  const auto& rewrites = ConservationRewrites();
  for (size_t i = 0; i < rewrites.size(); ++i) {
    oss << "    {\"from\":\"" << JsonEscape(rewrites[i].from) << "\",\"to\":\""
        << JsonEscape(rewrites[i].to) << "\",\"note\":\"" << JsonEscape(rewrites[i].note)
        << "\"}";
    oss << (i + 1 < rewrites.size() ? ",\n" : "\n");
  }
  oss << "  ]\n";
  oss << "}\n";
  return oss.str();
}

}  // namespace LatexTokenRegistry
