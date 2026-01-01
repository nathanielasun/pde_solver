#include "latex_parser.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <string>
#include <vector>

void LatexParser::ReplaceAll(std::string& text, const std::string& needle, const std::string& repl) {
  if (needle.empty()) {
    return;
  }
  size_t pos = 0;
  while ((pos = text.find(needle, pos)) != std::string::npos) {
    text.replace(pos, needle.size(), repl);
    pos += repl.size();
  }
}

std::string LatexParser::NormalizeLatex(const std::string& input) {
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

bool LatexParser::IsArgumentList(const std::string& text) {
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

void LatexParser::StripTrailingArgumentList(std::string* text) {
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

bool LatexParser::ParseLatexArgument(const std::string& text, size_t* index, std::string* out) {
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

bool LatexParser::StripEnclosing(std::string* text, char open, char close) {
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

bool LatexParser::DetectMixedDerivative(const std::string& text, std::string* error) {
  const char* unsupported_mixed_patterns[] = {
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

std::string LatexParser::StripDecorations(const std::string& input) {
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
  return Trim(out);
}

std::vector<std::string> LatexParser::SplitTerms(const std::string& expr, std::string* error) {
  std::vector<std::string> terms;
  std::string current;
  int brace_depth = 0;
  int paren_depth = 0;
  int bracket_depth = 0;
  for (size_t i = 0; i < expr.size(); ++i) {
    char ch = expr[i];
    if (ch == '{') {
      brace_depth++;
    } else if (ch == '}') {
      brace_depth--;
      if (brace_depth < 0) {
        if (error) {
          *error = "mismatched braces in latex";
        }
        return {};
      }
    } else if (ch == '(') {
      paren_depth++;
    } else if (ch == ')') {
      paren_depth--;
      if (paren_depth < 0) {
        if (error) {
          *error = "mismatched parentheses in latex";
        }
        return {};
      }
    } else if (ch == '[') {
      bracket_depth++;
    } else if (ch == ']') {
      bracket_depth--;
      if (bracket_depth < 0) {
        if (error) {
          *error = "mismatched brackets in latex";
        }
        return {};
      }
    }

    if (brace_depth == 0 && paren_depth == 0 && bracket_depth == 0 && (ch == '+' || ch == '-')) {
      if (!current.empty()) {
        terms.push_back(current);
        current.clear();
      }
      current.push_back(ch);
      continue;
    }
    current.push_back(ch);
  }

  if (brace_depth != 0) {
    if (error) {
      *error = "mismatched braces in latex";
    }
    return {};
  }
  if (paren_depth != 0) {
    if (error) {
      *error = "mismatched parentheses in latex";
    }
    return {};
  }
  if (bracket_depth != 0) {
    if (error) {
      *error = "mismatched brackets in latex";
    }
    return {};
  }

  if (!current.empty()) {
    terms.push_back(current);
  }
  return terms;
}

bool LatexParser::ParseLeadingNumber(const std::string& text, double* out_value,
                                     size_t* out_length) {
  if (!out_value || !out_length) {
    return false;
  }
  size_t index = 0;
  int sign = 1;
  if (index < text.size() && (text[index] == '+' || text[index] == '-')) {
    sign = text[index] == '-' ? -1 : 1;
    ++index;
  }
  if (index >= text.size()) {
    return false;
  }

  const std::string prefixes[] = {"\\frac", "\\tfrac", "\\dfrac"};
  for (const auto& prefix : prefixes) {
    if (text.compare(index, prefix.size(), prefix) == 0) {
      size_t local = index + prefix.size();
      std::string numerator;
      std::string denominator;
      if (!ParseLatexArgument(text, &local, &numerator)) {
        return false;
      }
      if (!ParseLatexArgument(text, &local, &denominator)) {
        return false;
      }
      double num = 0.0;
      double den = 0.0;
      if (!ParseNumber(numerator, &num) || !ParseNumber(denominator, &den)) {
        return false;
      }
      if (den == 0.0) {
        return false;
      }
      *out_value = sign * (num / den);
      *out_length = local;
      return true;
    }
  }

  const char* start = text.c_str() + index;
  char* end = nullptr;
  const double parsed = std::strtod(start, &end);
  if (end == start) {
    return false;
  }
  *out_value = sign * parsed;
  *out_length = static_cast<size_t>(end - text.c_str());
  return true;
}

bool LatexParser::MatchTerm(const std::string& term, const std::string& pattern,
                            double* out_coeff, std::string* error) {
  std::string trimmed = Trim(term);
  size_t pos = trimmed.find(pattern);
  if (pos == std::string::npos) {
    return false;
  }

  if (trimmed.find(pattern, pos + 1) != std::string::npos) {
    if (error) {
      *error = "term repeats derivative: " + trimmed;
    }
    return false;
  }

  std::string coeff_str = trimmed;
  coeff_str.erase(pos, pattern.size());
  coeff_str = Trim(coeff_str);
  coeff_str.erase(std::remove(coeff_str.begin(), coeff_str.end(), '*'), coeff_str.end());
  coeff_str = Trim(coeff_str);
  StripTrailingArgumentList(&coeff_str);
  coeff_str = Trim(coeff_str);

  if (coeff_str.empty() || coeff_str == "+") {
    *out_coeff = 1.0;
    return true;
  }
  if (coeff_str == "-") {
    *out_coeff = -1.0;
    return true;
  }

  double value = 0.0;
  if (ParseNumber(coeff_str, &value)) {
    *out_coeff = value;
    return true;
  }

  int sign = 1;
  if (!coeff_str.empty() && (coeff_str[0] == '+' || coeff_str[0] == '-')) {
    sign = coeff_str[0] == '-' ? -1 : 1;
    std::string rest = Trim(coeff_str.substr(1));
    if (IsArgumentList(rest)) {
      *out_coeff = static_cast<double>(sign);
      return true;
    }
  }
  if (IsArgumentList(coeff_str)) {
    *out_coeff = 1.0;
    return true;
  }

  if (error) {
    *error = "invalid coefficient: " + coeff_str;
  }
  return false;
}

bool LatexParser::ParseNumber(const std::string& text, double* out_value) {
  std::string trimmed = Trim(text);
  if (trimmed.empty()) {
    return false;
  }

  bool stripped = true;
  while (stripped) {
    stripped = StripEnclosing(&trimmed, '(', ')') || StripEnclosing(&trimmed, '{', '}');
    trimmed = Trim(trimmed);
  }

  int sign = 1;
  if (!trimmed.empty() && (trimmed[0] == '+' || trimmed[0] == '-')) {
    sign = trimmed[0] == '-' ? -1 : 1;
    trimmed = Trim(trimmed.substr(1));
  }
  if (trimmed.empty()) {
    return false;
  }
  stripped = true;
  while (stripped) {
    stripped = StripEnclosing(&trimmed, '(', ')') || StripEnclosing(&trimmed, '{', '}');
    trimmed = Trim(trimmed);
  }
  if (trimmed.empty()) {
    return false;
  }

  double value = 0.0;
  auto parse_fraction_token = [&](const std::string& token, double* out) -> bool {
    if (!out) {
      return false;
    }
    const std::string prefixes[] = {"\\frac", "\\tfrac", "\\dfrac"};
    for (const auto& prefix : prefixes) {
      if (token.rfind(prefix, 0) != 0) {
        continue;
      }
      size_t index = prefix.size();
      std::string numerator;
      std::string denominator;
      if (!ParseLatexArgument(token, &index, &numerator)) {
        return false;
      }
      if (!ParseLatexArgument(token, &index, &denominator)) {
        return false;
      }
      if (index != token.size()) {
        return false;
      }
      double num = 0.0;
      double den = 0.0;
      if (!ParseNumber(numerator, &num) || !ParseNumber(denominator, &den)) {
        return false;
      }
      if (den == 0.0) {
        return false;
      }
      *out = num / den;
      return true;
    }
    return false;
  };
  auto parse_slash_fraction = [&](const std::string& token, double* out) -> bool {
    size_t slash = token.find('/');
    if (slash == std::string::npos || token.find('/', slash + 1) != std::string::npos) {
      return false;
    }
    std::string left = token.substr(0, slash);
    std::string right = token.substr(slash + 1);
    if (left.empty() || right.empty()) {
      return false;
    }
    double num = 0.0;
    double den = 0.0;
    if (!ParseNumber(left, &num) || !ParseNumber(right, &den)) {
      return false;
    }
    if (den == 0.0) {
      return false;
    }
    *out = num / den;
    return true;
  };
  if (parse_fraction_token(trimmed, &value) || parse_slash_fraction(trimmed, &value)) {
    *out_value = sign * value;
    return true;
  }

  char* end = nullptr;
  const double parsed = std::strtod(trimmed.c_str(), &end);
  if (!end || *end != '\0') {
    return false;
  }
  *out_value = sign * parsed;
  return true;
}

std::string LatexParser::Trim(const std::string& input) {
  auto start = std::find_if_not(input.begin(), input.end(), [](unsigned char c) {
    return std::isspace(c);
  });
  auto end = std::find_if_not(input.rbegin(), input.rend(), [](unsigned char c) {
    return std::isspace(c);
  }).base();
  if (start >= end) {
    return "";
  }
  return std::string(start, end);
}


