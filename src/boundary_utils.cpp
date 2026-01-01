#include "boundary_utils.h"

#include <algorithm>
#include <cctype>
#include <sstream>
#include <iostream>

#include "string_utils.h"

using pde::Trim;

std::string NormalizeLatexExpr(const std::string& input) {
  std::string out;
  out.reserve(input.size());
  for (size_t i = 0; i < input.size(); ++i) {
    if (input[i] == '$') {
      continue;
    }
    if (input[i] == '\\' && i + 1 < input.size() && input[i + 1] == ',') {
      ++i;
      continue;
    }
    out.push_back(input[i]);
  }
  auto replace_all = [](std::string* text, const std::string& from, const std::string& to) {
    size_t pos = 0;
    while ((pos = text->find(from, pos)) != std::string::npos) {
      text->replace(pos, from.size(), to);
      pos += to.size();
    }
  };

  replace_all(&out, "\\left", "");
  replace_all(&out, "\\right", "");
  replace_all(&out, "\\cdot", "*");
  replace_all(&out, "\\times", "*");
  replace_all(&out, "\\vartheta", "theta");
  replace_all(&out, "\\theta", "theta");
  replace_all(&out, "\\varphi", "phi");
  replace_all(&out, "\\phi", "phi");
  return Trim(out);
}

std::string LatexifyExpr(const std::string& input) {
  std::string out = Trim(input);
  std::string replaced;
  replaced.reserve(out.size() * 2);
  for (char ch : out) {
    if (ch == '*') {
      replaced.append("\\cdot ");
    } else {
      replaced.push_back(ch);
    }
  }
  return replaced;
}

BoundaryLatexResult BuildBoundaryLatex(const BoundaryInput& input) {
  BoundaryLatexResult result;
  if (input.kind == 0) {  // Dirichlet
    const std::string value = Trim(input.value);
    if (value.empty()) {
      result.error = "dirichlet value is empty";
      return result;
    }
    result.latex = "u = " + LatexifyExpr(value);
    result.ok = true;
    return result;
  }
  if (input.kind == 1) {  // Neumann
    const std::string value = Trim(input.value);
    if (value.empty()) {
      result.error = "neumann value is empty";
      return result;
    }
    result.latex = "\\frac{\\partial u}{\\partial n} = " + LatexifyExpr(value);
    result.ok = true;
    return result;
  }
  if (input.kind == 2) {  // Robin
    const std::string alpha = Trim(input.alpha);
    const std::string beta = Trim(input.beta);
    const std::string gamma = Trim(input.gamma);
    if (alpha.empty() || beta.empty() || gamma.empty()) {
      result.error = "robin requires alpha, beta, gamma";
      return result;
    }
    result.latex = LatexifyExpr(alpha) + " u + " + LatexifyExpr(beta) +
                   " \\frac{\\partial u}{\\partial n} = " + LatexifyExpr(gamma);
    result.ok = true;
    return result;
  }
  result.error = "unsupported boundary kind: " + std::to_string(input.kind);
  return result;
}

static bool BuildBoundarySpecForInput(const BoundaryInput& input, std::string* spec,
                                      std::string* error) {
  if (!spec) {
    return false;
  }
  if (error) {
    error->clear();
  }
  if (input.kind == 0) {
    const std::string value = Trim(input.value);
    if (value.empty()) {
      if (error) {
        *error = "dirichlet value is empty";
      }
      return false;
    }
    *spec = "dirichlet:" + NormalizeLatexExpr(value);
    return true;
  }
  if (input.kind == 1) {
    const std::string value = Trim(input.value);
    if (value.empty()) {
      if (error) {
        *error = "neumann value is empty";
      }
      return false;
    }
    *spec = "neumann:" + NormalizeLatexExpr(value);
    return true;
  }
  if (input.kind == 2) {
    const std::string alpha = Trim(input.alpha);
    const std::string beta = Trim(input.beta);
    const std::string gamma = Trim(input.gamma);
    if (alpha.empty() || beta.empty() || gamma.empty()) {
      if (error) {
        *error = "robin requires alpha, beta, gamma";
      }
      return false;
    }
    std::ostringstream builder;
    builder << "robin:"
            << "alpha=" << NormalizeLatexExpr(alpha) << ","
            << "beta=" << NormalizeLatexExpr(beta) << ","
            << "gamma=" << NormalizeLatexExpr(gamma);
    *spec = builder.str();
    return true;
  }
  if (error) {
    *error = "unsupported boundary kind: " + std::to_string(input.kind);
    std::cerr << "BuildBoundarySpecForInput error: unsupported kind " << input.kind << "\n";
  }
  return false;
}

bool BuildBoundarySpec(const BoundaryInput& left, const BoundaryInput& right,
                       const BoundaryInput& bottom, const BoundaryInput& top,
                       const BoundaryInput& front, const BoundaryInput& back,
                       std::string* spec_out, std::string* error) {
  if (spec_out) {
    spec_out->clear();
  }
  std::string spec_left, spec_right, spec_bottom, spec_top, spec_front, spec_back;
  if (!BuildBoundarySpecForInput(left, &spec_left, error)) return false;
  if (!BuildBoundarySpecForInput(right, &spec_right, error)) return false;
  if (!BuildBoundarySpecForInput(bottom, &spec_bottom, error)) return false;
  if (!BuildBoundarySpecForInput(top, &spec_top, error)) return false;
  if (!BuildBoundarySpecForInput(front, &spec_front, error)) return false;
  if (!BuildBoundarySpecForInput(back, &spec_back, error)) return false;

  std::ostringstream out;
  out << "left:" << spec_left << ";"
      << "right:" << spec_right << ";"
      << "bottom:" << spec_bottom << ";"
      << "top:" << spec_top << ";"
      << "front:" << spec_front << ";"
      << "back:" << spec_back;
  if (spec_out) {
    *spec_out = out.str();
  }
  return true;
}
