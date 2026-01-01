#include "latex_parser.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <string>

#include "latex_patterns.h"

namespace {
LatexParseResult MakeErrorResult(const std::string& error) {
  LatexParseResult result;
  result.ok = false;
  result.error = error;
  return result;
}

PDEOperator BuildOperatorFromCoefficients(
    const PDECoefficients& coeffs,
    const std::vector<IntegralTerm>& integrals,
    const std::vector<NonlinearTerm>& nonlinear,
    const std::vector<NonlinearDerivativeTerm>& nonlinear_derivatives) {
  PDEOperator op;
  auto add_term = [&](int dx, int dy, int dz, int dt, double coeff, const std::string& coeff_latex) {
    if (coeff == 0.0 && coeff_latex.empty()) {
      return;
    }
    PDETerm term;
    term.dx = dx;
    term.dy = dy;
    term.dz = dz;
    term.dt = dt;
    term.coeff = coeff;
    term.coeff_latex = coeff_latex;
    op.lhs_terms.push_back(std::move(term));
  };

  add_term(2, 0, 0, 0, coeffs.a, coeffs.a_latex);
  add_term(0, 2, 0, 0, coeffs.b, coeffs.b_latex);
  add_term(0, 0, 2, 0, coeffs.az, coeffs.az_latex);
  add_term(1, 0, 0, 0, coeffs.c, coeffs.c_latex);
  add_term(0, 1, 0, 0, coeffs.d, coeffs.d_latex);
  add_term(0, 0, 1, 0, coeffs.dz, coeffs.dz_latex);
  add_term(0, 0, 0, 0, coeffs.e, coeffs.e_latex);
  add_term(0, 0, 0, 1, coeffs.ut, "");
  add_term(0, 0, 0, 2, coeffs.utt, "");
  add_term(1, 1, 0, 0, coeffs.ab, coeffs.ab_latex);
  add_term(1, 0, 1, 0, coeffs.ac, coeffs.ac_latex);
  add_term(0, 1, 1, 0, coeffs.bc, coeffs.bc_latex);
  add_term(3, 0, 0, 0, coeffs.a3, coeffs.a3_latex);
  add_term(0, 3, 0, 0, coeffs.b3, coeffs.b3_latex);
  add_term(0, 0, 3, 0, coeffs.az3, coeffs.az3_latex);
  add_term(4, 0, 0, 0, coeffs.a4, coeffs.a4_latex);
  add_term(0, 4, 0, 0, coeffs.b4, coeffs.b4_latex);
  add_term(0, 0, 4, 0, coeffs.az4, coeffs.az4_latex);

  op.rhs_constant = coeffs.f;
  op.rhs_latex = coeffs.rhs_latex;
  op.integrals = integrals;
  op.nonlinear = nonlinear;
  op.nonlinear_derivatives = nonlinear_derivatives;
  return op;
}

void ReplaceAll(std::string& text, const std::string& needle, const std::string& repl) {
  if (needle.empty()) {
    return;
  }
  size_t pos = 0;
  while ((pos = text.find(needle, pos)) != std::string::npos) {
    text.replace(pos, needle.size(), repl);
    pos += repl.size();
  }
}

std::string NormalizeLatex(const std::string& input) {
  std::string out = input;
  ReplaceAll(out, "\\left", "");
  ReplaceAll(out, "\\right", "");
  ReplaceAll(out, "\\cdot", "*");
  ReplaceAll(out, "\\,", "");
  ReplaceAll(out, "\\;", "");
  ReplaceAll(out, "\\:", "");
  ReplaceAll(out, "\\!", "");
  ReplaceAll(out, "\\quad", "");
  ReplaceAll(out, "\\qquad", "");
  ReplaceAll(out, "$", "");
  ReplaceAll(out, "\\theta", "theta");
  ReplaceAll(out, "\\vartheta", "theta");
  ReplaceAll(out, "\\phi", "phi");
  ReplaceAll(out, "\\varphi", "phi");

  out.erase(std::remove_if(out.begin(), out.end(), [](unsigned char ch) {
    return std::isspace(ch) != 0;
  }), out.end());

  ReplaceAll(out, "^{2}", "^2");
  ReplaceAll(out, "_{x}", "_x");
  ReplaceAll(out, "_{y}", "_y");
  ReplaceAll(out, "_{t}", "_t");
  ReplaceAll(out, "_{r}", "_r");
  ReplaceAll(out, "_{theta}", "_theta");
  ReplaceAll(out, "_{phi}", "_phi");
  return out;
}

bool IsArgumentList(const std::string& text) {
  if (text.size() < 3 || text.front() != '(' || text.back() != ')') {
    return false;
  }
  std::string inner = text.substr(1, text.size() - 2);
  size_t start = 0;
  while (start < inner.size()) {
    size_t end = inner.find(',', start);
    if (end == std::string::npos) {
      end = inner.size();
    }
    size_t left = start;
    while (left < end && std::isspace(static_cast<unsigned char>(inner[left])) != 0) {
      ++left;
    }
    size_t right = end;
    while (right > left && std::isspace(static_cast<unsigned char>(inner[right - 1])) != 0) {
      --right;
    }
    const std::string token = inner.substr(left, right - left);
    if (!token.empty() && token != "x" && token != "y" && token != "z" &&
        token != "t" && token != "r" && token != "theta" && token != "phi") {
      return false;
    }
    start = end + 1;
  }
  return true;
}

void StripTrailingArgumentList(std::string* text) {
  if (!text || text->empty()) {
    return;
  }
  const size_t pos = text->rfind('(');
  if (pos == std::string::npos) {
    return;
  }
  const std::string tail = text->substr(pos);
  if (!IsArgumentList(tail)) {
    return;
  }
  text->erase(pos);
}

bool ParseLatexArgument(const std::string& text, size_t* index, std::string* out) {
  if (!index || *index >= text.size() || text[*index] != '{') {
    return false;
  }
  int depth = 0;
  size_t start = *index + 1;
  for (size_t i = *index; i < text.size(); ++i) {
    if (text[i] == '{') {
      ++depth;
    } else if (text[i] == '}') {
      --depth;
      if (depth == 0) {
        if (out) {
          *out = text.substr(start, i - start);
        }
        *index = i + 1;
        return true;
      }
    }
  }
  return false;
}

bool StripEnclosing(std::string* text, char open, char close) {
  if (!text || text->size() < 2 || (*text)[0] != open ||
      (*text)[text->size() - 1] != close) {
    return false;
  }
  int depth = 0;
  for (size_t i = 0; i + 1 < text->size(); ++i) {
    if ((*text)[i] == open) {
      ++depth;
    } else if ((*text)[i] == close) {
      --depth;
      if (depth == 0 && i + 1 != text->size() - 1) {
        return false;
      }
    }
  }
  if (depth != 1) {
    return false;
  }
  *text = text->substr(1, text->size() - 2);
  return true;
}

// Detect unsupported mixed derivatives (e.g., with time: u_{xt}, u_{yt}, etc.)
// Spatial mixed derivatives (u_{xy}, u_{xz}, u_{yz}) are now supported and handled above.
bool DetectMixedDerivative(const std::string& text, std::string* error) {
  const char* unsupported_mixed_patterns[] = {
      // Time-mixed derivatives (not yet supported)
      "u_{xt}",
      "u_{tx}",
      "u_{yt}",
      "u_{ty}",
      "u_{zt}",
      "u_{tz}",
      "\\partial_{xt}u",
      "\\partial_{tx}u",
      "\\partial_{yt}u",
      "\\partial_{ty}u",
      "\\partial_{zt}u",
      "\\partial_{tz}u",
      "\\partial_x\\partial_tu",
      "\\partial_t\\partial_xu",
      "\\partial_y\\partial_tu",
      "\\partial_t\\partial_yu",
      "\\partial_z\\partial_tu",
      "\\partial_t\\partial_zu",
      "\\frac{\\partial^2u}{\\partialx\\partialt}",
      "\\frac{\\partial^2u}{\\partialt\\partialx}",
      "\\frac{\\partial^2u}{\\partialy\\partialt}",
      "\\frac{\\partial^2u}{\\partialt\\partialy}",
      "\\frac{\\partial^2u}{\\partialz\\partialt}",
      "\\frac{\\partial^2u}{\\partialt\\partialz}",
      "\\partial^2u/\\partialx\\partialt",
      "\\partial^2u/\\partialt\\partialx",
      "\\partial^2u/\\partialy\\partialt",
      "\\partial^2u/\\partialt\\partialy",
      "\\partial^2u/\\partialz\\partialt",
      "\\partial^2u/\\partialt\\partialz",
      "\\frac{d^2u}{dxdt}",
      "\\frac{d^2u}{dtdx}",
      "\\frac{d^2u}{dydt}",
      "\\frac{d^2u}{dtdy}",
      "\\frac{d^2u}{dzdt}",
      "\\frac{d^2u}{dtdz}",
      "d^2u/dxdt",
      "d^2u/dtdx",
      "d^2u/dydt",
      "d^2u/dtdy",
      "d^2u/dzdt",
      "d^2u/dtdz",
      // Coordinate-time mixed (not yet supported)
      "u_{rt}",
      "u_{tr}",
      "u_{thetat}",
      "u_{ttheta}",
      "u_{phit}",
      "u_{tphi}",
      "\\partial_{rt}u",
      "\\partial_{tr}u",
      "\\partial_{thetat}u",
      "\\partial_{ttheta}u",
      "\\partial_{phit}u",
      "\\partial_{tphi}u",
      "\\partial_r\\partial_tu",
      "\\partial_t\\partial_ru",
      "\\partial_theta\\partial_tu",
      "\\partial_t\\partial_thetau",
      "\\partial_phi\\partial_tu",
      "\\partial_t\\partial_phiu",
      "\\frac{\\partial^2u}{\\partialr\\partialt}",
      "\\frac{\\partial^2u}{\\partialt\\partialr}",
      "\\frac{\\partial^2u}{\\partialtheta\\partialt}",
      "\\frac{\\partial^2u}{\\partialt\\partialtheta}",
      "\\frac{\\partial^2u}{\\partialphi\\partialt}",
      "\\frac{\\partial^2u}{\\partialt\\partialphi}",
      "\\partial^2u/\\partialr\\partialt",
      "\\partial^2u/\\partialt\\partialr",
      "\\partial^2u/\\partialtheta\\partialt",
      "\\partial^2u/\\partialt\\partialtheta",
      "\\partial^2u/\\partialphi\\partialt",
      "\\partial^2u/\\partialt\\partialphi",
      "\\frac{d^2u}{drdt}",
      "\\frac{d^2u}{dtdr}",
      "\\frac{d^2u}{dthetadt}",
      "\\frac{d^2u}{dtdtheta}",
      "\\frac{d^2u}{dphidt}",
      "\\frac{d^2u}{dtdphi}",
      "d^2u/drdt",
      "d^2u/dtdr",
      "d^2u/dthetadt",
      "d^2u/dtdtheta",
      "d^2u/dphidt",
      "d^2u/dtdphi",
  };
  for (const char* pattern : unsupported_mixed_patterns) {
    if (text.find(pattern) != std::string::npos) {
      if (error) {
        *error = "time-mixed derivatives are not yet supported: " + text;
      }
      return true;
    }
  }
  return false;
}
}

LatexParseResult LatexParser::Parse(const std::string& latex) const {
  std::string clean = StripDecorations(latex);
  if (clean.empty()) {
    return MakeErrorResult("empty latex input");
  }

  std::string left = clean;
  std::string right;
  size_t eq = clean.find('=');
  if (eq != std::string::npos) {
    left = Trim(clean.substr(0, eq));
    right = Trim(clean.substr(eq + 1));
    if (left.empty() || right.empty()) {
      return MakeErrorResult("equation must have content on both sides");
    }
  }

  LatexParseResult left_result = ParseExpression(left);
  if (!left_result.ok) {
    return left_result;
  }

  PDECoefficients coeffs = left_result.coeffs;
  std::vector<IntegralTerm> integrals = left_result.integrals;
  std::vector<NonlinearTerm> nonlinear = left_result.nonlinear;
  std::vector<NonlinearDerivativeTerm> nonlinear_derivatives = left_result.nonlinear_derivatives;
  if (!right.empty()) {
    LatexParseResult right_result = ParseExpression(right);
    if (!right_result.ok) {
      return right_result;
    }
    coeffs.a -= right_result.coeffs.a;
    coeffs.b -= right_result.coeffs.b;
    coeffs.az -= right_result.coeffs.az;
    coeffs.c -= right_result.coeffs.c;
    coeffs.d -= right_result.coeffs.d;
    coeffs.dz -= right_result.coeffs.dz;
    coeffs.e -= right_result.coeffs.e;
    coeffs.f -= right_result.coeffs.f;
    coeffs.ut -= right_result.coeffs.ut;
    coeffs.utt -= right_result.coeffs.utt;
    coeffs.ab -= right_result.coeffs.ab;
    coeffs.ac -= right_result.coeffs.ac;
    coeffs.bc -= right_result.coeffs.bc;

    for (const auto& term : right_result.integrals) {
      IntegralTerm negated = term;
      negated.coeff = -negated.coeff;
      integrals.push_back(std::move(negated));
    }
    for (const auto& term : right_result.nonlinear) {
      NonlinearTerm negated = term;
      negated.coeff = -negated.coeff;
      nonlinear.push_back(std::move(negated));
    }
  }
  LatexParseResult result;
  result.ok = true;
  result.coeffs = coeffs;
  result.integrals = std::move(integrals);
  result.nonlinear = std::move(nonlinear);
  result.nonlinear_derivatives = std::move(nonlinear_derivatives);
  result.op = BuildOperatorFromCoefficients(result.coeffs, result.integrals,
                                            result.nonlinear, result.nonlinear_derivatives);
  return result;
}

LatexParseResult LatexParser::ParseExpression(const std::string& expr) const {
  std::string error;
  std::vector<std::string> terms = SplitTerms(expr, &error);
  if (!error.empty()) {
    return MakeErrorResult(error);
  }

  PDECoefficients coeffs;
  std::vector<IntegralTerm> integrals;
  std::vector<NonlinearTerm> nonlinear;
  std::vector<NonlinearDerivativeTerm> nonlinear_derivatives;
  for (const auto& term : terms) {
    if (Trim(term).empty()) {
      continue;
    }
    if (!ParseTerm(term, &coeffs, &integrals, &nonlinear, &nonlinear_derivatives, &error)) {
      return MakeErrorResult(error);
    }
  }

  LatexParseResult result;
  result.ok = true;
  result.coeffs = coeffs;
  result.integrals = std::move(integrals);
  result.nonlinear = std::move(nonlinear);
  result.nonlinear_derivatives = std::move(nonlinear_derivatives);
  result.op = BuildOperatorFromCoefficients(result.coeffs, result.integrals,
                                            result.nonlinear, result.nonlinear_derivatives);
  result.primary_field = "u";  // Default to u for backward compatibility
  result.field_coeffs["u"] = result.coeffs;
  result.detected_fields.insert("u");
  return result;
}

LatexParseResult LatexParser::ParseForField(const std::string& latex, const std::string& primary_field) const {
  std::string clean = StripDecorations(latex);
  if (clean.empty()) {
    return MakeErrorResult("empty latex input");
  }

  std::string left = clean;
  std::string right;
  size_t eq = clean.find('=');
  if (eq != std::string::npos) {
    left = Trim(clean.substr(0, eq));
    right = Trim(clean.substr(eq + 1));
    if (left.empty() || right.empty()) {
      return MakeErrorResult("equation must have content on both sides");
    }
  }

  LatexParseResult left_result = ParseExpressionForField(left, primary_field);
  if (!left_result.ok) {
    return left_result;
  }

  if (!right.empty()) {
    LatexParseResult right_result = ParseExpressionForField(right, primary_field);
    if (!right_result.ok) {
      return right_result;
    }
    // Merge detected fields
    for (const auto& field : right_result.detected_fields) {
      left_result.detected_fields.insert(field);
    }
    // Subtract right side coefficients from left side for each field
    for (auto& [field, coeffs] : right_result.field_coeffs) {
      if (left_result.field_coeffs.find(field) == left_result.field_coeffs.end()) {
        left_result.field_coeffs[field] = PDECoefficients();
      }
      auto& left_coeffs = left_result.field_coeffs[field];
      left_coeffs.a -= coeffs.a;
      left_coeffs.b -= coeffs.b;
      left_coeffs.az -= coeffs.az;
      left_coeffs.c -= coeffs.c;
      left_coeffs.d -= coeffs.d;
      left_coeffs.dz -= coeffs.dz;
      left_coeffs.e -= coeffs.e;
      left_coeffs.f -= coeffs.f;
      left_coeffs.ut -= coeffs.ut;
      left_coeffs.utt -= coeffs.utt;
      left_coeffs.ab -= coeffs.ab;
      left_coeffs.ac -= coeffs.ac;
      left_coeffs.bc -= coeffs.bc;
      left_coeffs.a3 -= coeffs.a3;
      left_coeffs.b3 -= coeffs.b3;
      left_coeffs.az3 -= coeffs.az3;
      left_coeffs.a4 -= coeffs.a4;
      left_coeffs.b4 -= coeffs.b4;
      left_coeffs.az4 -= coeffs.az4;
    }
    // Subtract RHS constant
    left_result.coeffs.f -= right_result.coeffs.f;
    // Handle integrals and nonlinear terms
    for (const auto& term : right_result.integrals) {
      IntegralTerm negated = term;
      negated.coeff = -negated.coeff;
      left_result.integrals.push_back(std::move(negated));
    }
    for (const auto& term : right_result.nonlinear) {
      NonlinearTerm negated = term;
      negated.coeff = -negated.coeff;
      left_result.nonlinear.push_back(std::move(negated));
    }
  }

  // Update main coeffs from primary field
  if (left_result.field_coeffs.find(primary_field) != left_result.field_coeffs.end()) {
    left_result.coeffs = left_result.field_coeffs[primary_field];
  }
  left_result.op = BuildOperatorFromCoefficients(left_result.coeffs, left_result.integrals,
                                                 left_result.nonlinear, left_result.nonlinear_derivatives);
  return left_result;
}

LatexParseResult LatexParser::ParseExpressionForField(const std::string& expr,
                                                       const std::string& primary_field) const {
  std::string error;
  std::vector<std::string> terms = SplitTerms(expr, &error);
  if (!error.empty()) {
    return MakeErrorResult(error);
  }

  std::map<std::string, PDECoefficients> field_coeffs;
  std::set<std::string> detected_fields;
  std::vector<IntegralTerm> integrals;
  std::vector<NonlinearTerm> nonlinear;
  std::vector<NonlinearDerivativeTerm> nonlinear_derivatives;

  for (const auto& term : terms) {
    if (Trim(term).empty()) {
      continue;
    }
    if (!ParseTermMultiField(term, primary_field, &field_coeffs, &detected_fields,
                             &integrals, &nonlinear, &nonlinear_derivatives, &error)) {
      return MakeErrorResult(error);
    }
  }

  LatexParseResult result;
  result.ok = true;
  result.primary_field = primary_field;
  result.field_coeffs = std::move(field_coeffs);
  result.detected_fields = std::move(detected_fields);
  result.integrals = std::move(integrals);
  result.nonlinear = std::move(nonlinear);
  result.nonlinear_derivatives = std::move(nonlinear_derivatives);

  // Set main coeffs from primary field
  if (result.field_coeffs.find(primary_field) != result.field_coeffs.end()) {
    result.coeffs = result.field_coeffs[primary_field];
  }
  result.op = BuildOperatorFromCoefficients(result.coeffs, result.integrals,
                                            result.nonlinear, result.nonlinear_derivatives);
  return result;
}

MultiFieldParseResult LatexParser::ParseMultiField(const std::vector<std::string>& equations,
                                                    const std::vector<std::string>& field_names) const {
  MultiFieldParseResult result;

  if (equations.size() != field_names.size()) {
    result.ok = false;
    result.error = "number of equations must match number of field names";
    return result;
  }

  if (equations.empty()) {
    result.ok = false;
    result.error = "at least one equation is required";
    return result;
  }

  // Parse each equation
  for (size_t i = 0; i < equations.size(); ++i) {
    LatexParseResult eq_result = ParseForField(equations[i], field_names[i]);
    if (!eq_result.ok) {
      result.ok = false;
      result.error = "error parsing equation for field '" + field_names[i] + "': " + eq_result.error;
      return result;
    }
    result.equations.push_back(std::move(eq_result));
    for (const auto& field : result.equations.back().detected_fields) {
      result.all_fields.insert(field);
    }
  }

  // Build MultiFieldEquation structure
  for (size_t i = 0; i < equations.size(); ++i) {
    FieldEquationCoefficients eq_coeffs;
    eq_coeffs.field_name = field_names[i];

    const auto& parsed = result.equations[i];
    for (const auto& [field, coeffs] : parsed.field_coeffs) {
      if (field == field_names[i]) {
        eq_coeffs.self_coeffs = coeffs;
      } else {
        CrossFieldCoefficients cross;
        cross.source_field = field;
        cross.coeffs = coeffs;
        eq_coeffs.coupled.push_back(std::move(cross));
      }
    }
    result.multi_field_eq.equations.push_back(std::move(eq_coeffs));
  }

  // Analyze coupling pattern
  result.coupling = AnalyzeCoupling(result);

  result.ok = true;
  return result;
}

CouplingAnalysis LatexParser::AnalyzeCoupling(const MultiFieldParseResult& result) {
  CouplingAnalysis analysis;

  if (!result.ok || result.equations.empty()) {
    return analysis;
  }

  // Collect all fields
  for (const auto& field : result.all_fields) {
    analysis.fields.push_back(field);
  }

  // Build dependency graph
  for (const auto& eq : result.multi_field_eq.equations) {
    std::vector<std::string> deps;
    // Self-dependency if there are self terms
    bool has_self = eq.self_coeffs.a != 0 || eq.self_coeffs.b != 0 || eq.self_coeffs.az != 0 ||
                    eq.self_coeffs.c != 0 || eq.self_coeffs.d != 0 || eq.self_coeffs.dz != 0 ||
                    eq.self_coeffs.e != 0 || eq.self_coeffs.ut != 0 || eq.self_coeffs.utt != 0;
    if (has_self) {
      deps.push_back(eq.field_name);
    }
    // Cross-field dependencies
    for (const auto& cross : eq.coupled) {
      deps.push_back(cross.source_field);
    }
    analysis.dependencies[eq.field_name] = deps;
  }

  // Determine coupling pattern
  if (result.equations.size() == 1) {
    analysis.pattern = CouplingPattern::SingleField;
  } else {
    bool has_coupling = result.multi_field_eq.HasCoupling();
    if (!has_coupling) {
      // Multiple independent equations
      analysis.pattern = CouplingPattern::SingleField;
    } else {
      // Check for circular dependencies
      for (const auto& [field, deps] : analysis.dependencies) {
        for (const auto& dep : deps) {
          if (dep != field && analysis.dependencies.count(dep) > 0) {
            for (const auto& dep2 : analysis.dependencies.at(dep)) {
              if (dep2 == field) {
                analysis.has_circular_dependency = true;
                break;
              }
            }
          }
        }
        if (analysis.has_circular_dependency) break;
      }

      if (analysis.has_circular_dependency) {
        analysis.pattern = CouplingPattern::SymmetricCoupling;
      } else {
        analysis.pattern = CouplingPattern::ExplicitCoupling;
      }
    }
  }

  return analysis;
}
