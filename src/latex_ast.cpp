#include "latex_parser.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "latex_patterns.h"

// Helper to check if a term matches a pattern for any field
static bool MatchTermForField(const std::string& term, const std::string& pattern,
                              double* out_coeff, std::string* error) {
  std::string trimmed = term;
  // Remove leading/trailing whitespace
  auto start = trimmed.find_first_not_of(" \t\n\r");
  auto end = trimmed.find_last_not_of(" \t\n\r");
  if (start == std::string::npos) {
    trimmed = "";
  } else {
    trimmed = trimmed.substr(start, end - start + 1);
  }

  size_t pos = trimmed.find(pattern);
  if (pos == std::string::npos) {
    return false;
  }

  // Check for repeated pattern
  if (trimmed.find(pattern, pos + 1) != std::string::npos) {
    if (error) {
      *error = "term repeats derivative: " + trimmed;
    }
    return false;
  }

  std::string coeff_str = trimmed;
  coeff_str.erase(pos, pattern.size());

  // Trim and remove multiplication signs
  start = coeff_str.find_first_not_of(" \t\n\r");
  end = coeff_str.find_last_not_of(" \t\n\r");
  if (start == std::string::npos) {
    coeff_str = "";
  } else {
    coeff_str = coeff_str.substr(start, end - start + 1);
  }
  coeff_str.erase(std::remove(coeff_str.begin(), coeff_str.end(), '*'), coeff_str.end());
  start = coeff_str.find_first_not_of(" \t\n\r");
  end = coeff_str.find_last_not_of(" \t\n\r");
  if (start == std::string::npos) {
    coeff_str = "";
  } else {
    coeff_str = coeff_str.substr(start, end - start + 1);
  }

  if (coeff_str.empty() || coeff_str == "+") {
    *out_coeff = 1.0;
    return true;
  }
  if (coeff_str == "-") {
    *out_coeff = -1.0;
    return true;
  }

  // Try to parse as number
  double value = 0.0;
  if (LatexParser::ParseNumber(coeff_str, &value)) {
    *out_coeff = value;
    return true;
  }

  // Handle sign prefix
  int sign = 1;
  if (!coeff_str.empty() && (coeff_str[0] == '+' || coeff_str[0] == '-')) {
    sign = coeff_str[0] == '-' ? -1 : 1;
    std::string rest = coeff_str.substr(1);
    start = rest.find_first_not_of(" \t\n\r");
    end = rest.find_last_not_of(" \t\n\r");
    if (start != std::string::npos) {
      rest = rest.substr(start, end - start + 1);
    }
    if (LatexParser::IsArgumentList(rest)) {
      *out_coeff = static_cast<double>(sign);
      return true;
    }
  }
  if (LatexParser::IsArgumentList(coeff_str)) {
    *out_coeff = 1.0;
    return true;
  }

  if (error) {
    *error = "invalid coefficient: " + coeff_str;
  }
  return false;
}

bool LatexParser::ParseTerm(const std::string& term, PDECoefficients* coeffs,
                            std::vector<IntegralTerm>* integrals,
                            std::vector<NonlinearTerm>* nonlinear,
                            std::vector<NonlinearDerivativeTerm>* nonlinear_derivatives,
                            std::string* error) {
  std::string original = Trim(term);
  if (original.empty()) {
    return true;
  }
  std::string trimmed = original;
  int leading_sign = 1;
  if (!trimmed.empty() && (trimmed[0] == '+' || trimmed[0] == '-')) {
    leading_sign = trimmed[0] == '-' ? -1 : 1;
    trimmed = Trim(trimmed.substr(1));
  }
  // Check for coefficient before parentheses: a(...), sin(x)(...), (1+x^2)(...), etc.
  // Pattern: [coefficient](term1 + term2 + ...)
  size_t paren_pos = trimmed.find('(');
  if (paren_pos != std::string::npos && paren_pos > 0) {
    // Check if this is a multiplication pattern (coefficient before parentheses)
    // Look for patterns like: "a(", "sin(x)(", "(1+x^2)(", "2*(", etc.
    std::string before_paren = Trim(trimmed.substr(0, paren_pos));
    std::string after_paren = trimmed.substr(paren_pos);
    
    // Check if after_paren is a balanced parentheses expression
    int paren_count = 0;
    size_t matching_paren = std::string::npos;
    for (size_t i = 0; i < after_paren.size(); ++i) {
      if (after_paren[i] == '(') {
        ++paren_count;
      } else if (after_paren[i] == ')') {
        --paren_count;
        if (paren_count == 0) {
          matching_paren = i;
          break;
        }
      }
    }
    
    // If we found a matching closing parenthesis and there's something before the opening paren
    if (matching_paren != std::string::npos && !before_paren.empty()) {
      // Extract the inner expression
      std::string inner_expr = after_paren.substr(1, matching_paren - 1);
      
      // Check if before_paren looks like a coefficient (not just whitespace or operators)
      // It should contain alphanumeric, functions, or be a number/expression
      bool looks_like_coeff = false;
      std::string coeff_part = Trim(before_paren);
      // Remove trailing * if present
      if (!coeff_part.empty() && coeff_part.back() == '*') {
        coeff_part.pop_back();
        coeff_part = Trim(coeff_part);
      }
      
      // Check if it's a valid coefficient expression
      if (!coeff_part.empty() && coeff_part != "+" && coeff_part != "-") {
        // Check if it contains letters, numbers, functions, or parentheses
        bool has_content = false;
        for (char c : coeff_part) {
          if (std::isalnum(static_cast<unsigned char>(c)) || c == '(' || c == ')' || 
              c == '\\' || c == '^' || c == '_' || c == '.' || c == '+' || c == '-' ||
              c == '*' || c == '/' || c == 's' || c == 'c' || c == 'e' || c == 'l') {
            has_content = true;
            break;
          }
        }
        if (has_content) {
          looks_like_coeff = true;
        }
      }
      
      if (looks_like_coeff) {
        // This is a coefficient multiplied by a parenthesized expression
        // Distribute the coefficient to all terms inside
        std::string inner_error;
        std::vector<std::string> inner_terms = SplitTerms(inner_expr, &inner_error);
        if (!inner_error.empty()) {
          if (error) {
            *error = inner_error;
          }
          return false;
        }
        
        // For each inner term, prepend the coefficient
        for (const auto& inner_term : inner_terms) {
          std::string adjusted = Trim(inner_term);
          if (adjusted.empty()) {
            continue;
          }
          
          // Prepend the coefficient: "coeff * term" or "coeff term"
          std::string term_with_coeff;
          if (leading_sign < 0) {
            term_with_coeff = "-";
          }
          term_with_coeff += coeff_part;
          
          // Add multiplication if needed
          if (adjusted[0] != '+' && adjusted[0] != '-' && 
              !coeff_part.empty() && coeff_part.back() != '*' && coeff_part.back() != '(') {
            term_with_coeff += "*";
          }
          
          // Handle sign of inner term
          if (adjusted[0] == '+') {
            term_with_coeff += adjusted.substr(1);
          } else if (adjusted[0] == '-') {
            if (leading_sign < 0) {
              term_with_coeff = coeff_part;  // Negative * negative = positive
              if (!coeff_part.empty() && coeff_part.back() != '*') {
                term_with_coeff += "*";
              }
              term_with_coeff += adjusted.substr(1);
            } else {
              term_with_coeff = "-" + coeff_part;
              if (!coeff_part.empty() && coeff_part.back() != '*') {
                term_with_coeff += "*";
              }
              term_with_coeff += adjusted.substr(1);
            }
          } else {
            term_with_coeff += adjusted;
          }
          
          if (!ParseTerm(term_with_coeff, coeffs, integrals, nonlinear, nonlinear_derivatives, error)) {
            return false;
          }
        }
        return true;
      }
    }
  }
  
  // Original parentheses handling (no coefficient before)
  std::string inner = trimmed;
  if (StripEnclosing(&inner, '(', ')')) {
    std::string inner_error;
    std::vector<std::string> inner_terms = SplitTerms(inner, &inner_error);
    if (!inner_error.empty()) {
      if (error) {
        *error = inner_error;
      }
      return false;
    }
    for (const auto& inner_term : inner_terms) {
      std::string adjusted = Trim(inner_term);
      if (adjusted.empty()) {
        continue;
      }
      if (leading_sign < 0) {
        if (adjusted[0] == '+') {
          adjusted[0] = '-';
        } else if (adjusted[0] == '-') {
          adjusted[0] = '+';
        } else {
          adjusted.insert(adjusted.begin(), '-');
        }
      }
      if (!ParseTerm(adjusted, coeffs, integrals, nonlinear, nonlinear_derivatives, error)) {
        return false;
      }
    }
    return true;
  }

  const std::string normalized = NormalizeLatex(term);
  if (normalized.empty()) {
    return true;
  }

  std::string integral_error;
  if (ParseIntegralTerm(normalized, integrals, &integral_error)) {
    if (!integral_error.empty()) {
      if (error) {
        *error = integral_error;
      }
      return false;
    }
    return true;
  }

  std::string nonlinear_error;
  if (ParseNonlinearTerm(normalized, nonlinear, &nonlinear_error)) {
    if (!nonlinear_error.empty()) {
      if (error) {
        *error = nonlinear_error;
      }
      return false;
    }
    return true;
  }

  std::string nonlinear_deriv_error;
  if (ParseNonlinearDerivativeTerm(normalized, nonlinear_derivatives, &nonlinear_deriv_error)) {
    if (!nonlinear_deriv_error.empty()) {
      if (error) {
        *error = nonlinear_deriv_error;
      }
      return false;
    }
    return true;
  }

  // Helper function to check for and extract variable coefficient expressions
  // Returns the coefficient expression if found, empty string otherwise
  auto ExtractVarCoeff = [](const std::string& term, const std::string& pattern) -> std::string {
    std::string trimmed = Trim(term);
    size_t pos = trimmed.find(pattern);
    if (pos == std::string::npos) {
      return "";
    }
    std::string coeff_str = trimmed;
    coeff_str.erase(pos, pattern.size());
    coeff_str = Trim(coeff_str);
    coeff_str.erase(std::remove(coeff_str.begin(), coeff_str.end(), '*'), coeff_str.end());
    coeff_str = Trim(coeff_str);
    
    // If empty or just a sign, it's a constant coefficient
    if (coeff_str.empty() || coeff_str == "+" || coeff_str == "-") {
      return "";
    }
    
    // Check if it's a numeric coefficient
    double value = 0.0;
    if (ParseNumber(coeff_str, &value)) {
      return "";  // Numeric, not variable
    }
    
    // If it contains parentheses, backslashes (LaTeX commands), or function names, it's likely variable
    if (coeff_str.find('(') != std::string::npos || 
        coeff_str.find('\\') != std::string::npos ||
        coeff_str.find("sin") != std::string::npos ||
        coeff_str.find("cos") != std::string::npos ||
        coeff_str.find("exp") != std::string::npos ||
        coeff_str.find("log") != std::string::npos ||
        coeff_str.find("sqrt") != std::string::npos ||
        coeff_str.find('^') != std::string::npos) {
      return coeff_str;  // Variable coefficient expression
    }
    
    return "";  // Unknown format, treat as constant
  };

  double coeff = 0.0;
  std::string var_coeff;
  
  // Check for variable coefficients in Laplacian terms
  for (const char* pattern : LatexPatterns::kLaplacianPatterns) {
    var_coeff = ExtractVarCoeff(normalized, pattern);
    if (!var_coeff.empty()) {
      // Variable coefficient detected - store in all three second derivative fields
      coeffs->a_latex = var_coeff;
      coeffs->b_latex = var_coeff;
      coeffs->az_latex = var_coeff;
      coeffs->a = 1.0;
      coeffs->b = 1.0;
      coeffs->az = 1.0;
      return true;
    }
    if (MatchTerm(normalized, pattern, &coeff, error)) {
      coeffs->a += coeff;
      coeffs->b += coeff;
      coeffs->az += coeff;
      return true;
    }
  }
  
  // Check for variable coefficients in u_xx terms
  for (const char* pattern : LatexPatterns::kD2XPatterns) {
    var_coeff = ExtractVarCoeff(normalized, pattern);
    if (!var_coeff.empty()) {
      coeffs->a_latex = var_coeff;
      coeffs->a = 1.0;
      return true;
    }
    if (MatchTerm(normalized, pattern, &coeff, error)) {
      coeffs->a += coeff;
      return true;
    }
  }
  // Check for variable coefficients in u_yy terms
  for (const char* pattern : LatexPatterns::kD2YPatterns) {
    var_coeff = ExtractVarCoeff(normalized, pattern);
    if (!var_coeff.empty()) {
      coeffs->b_latex = var_coeff;
      coeffs->b = 1.0;
      return true;
    }
    if (MatchTerm(normalized, pattern, &coeff, error)) {
      coeffs->b += coeff;
      return true;
    }
  }
  
  // Check for variable coefficients in u_zz terms
  for (const char* pattern : LatexPatterns::kD2ZPatterns) {
    var_coeff = ExtractVarCoeff(normalized, pattern);
    if (!var_coeff.empty()) {
      coeffs->az_latex = var_coeff;
      coeffs->az = 1.0;
      return true;
    }
    if (MatchTerm(normalized, pattern, &coeff, error)) {
      coeffs->az += coeff;
      return true;
    }
  }
  for (const char* pattern : LatexPatterns::kD2TPatterns) {
    if (MatchTerm(normalized, pattern, &coeff, error)) {
      coeffs->utt += coeff;
      return true;
    }
  }
  // Check for variable coefficients in u_x terms
  for (const char* pattern : LatexPatterns::kDXPatterns) {
    var_coeff = ExtractVarCoeff(normalized, pattern);
    if (!var_coeff.empty()) {
      coeffs->c_latex = var_coeff;
      coeffs->c = 1.0;
      return true;
    }
    if (MatchTerm(normalized, pattern, &coeff, error)) {
      coeffs->c += coeff;
      return true;
    }
  }
  
  // Check for variable coefficients in u_y terms
  for (const char* pattern : LatexPatterns::kDYPatterns) {
    var_coeff = ExtractVarCoeff(normalized, pattern);
    if (!var_coeff.empty()) {
      coeffs->d_latex = var_coeff;
      coeffs->d = 1.0;
      return true;
    }
    if (MatchTerm(normalized, pattern, &coeff, error)) {
      coeffs->d += coeff;
      return true;
    }
  }
  
  // Check for variable coefficients in u_z terms
  for (const char* pattern : LatexPatterns::kDZPatterns) {
    var_coeff = ExtractVarCoeff(normalized, pattern);
    if (!var_coeff.empty()) {
      coeffs->dz_latex = var_coeff;
      coeffs->dz = 1.0;
      return true;
    }
    if (MatchTerm(normalized, pattern, &coeff, error)) {
      coeffs->dz += coeff;
      return true;
    }
  }
  for (const char* pattern : LatexPatterns::kDTPatterns) {
    if (MatchTerm(normalized, pattern, &coeff, error)) {
      coeffs->ut += coeff;
      return true;
    }
  }
  // Mixed derivatives (must come before DetectMixedDerivative check)
  // Check for variable coefficients in u_xy terms
  for (const char* pattern : LatexPatterns::kDXYPatterns) {
    var_coeff = ExtractVarCoeff(normalized, pattern);
    if (!var_coeff.empty()) {
      coeffs->ab_latex = var_coeff;
      coeffs->ab = 1.0;
      return true;
    }
    if (MatchTerm(normalized, pattern, &coeff, error)) {
      coeffs->ab += coeff;
      return true;
    }
  }
  
  // Check for variable coefficients in u_xz terms
  for (const char* pattern : LatexPatterns::kDXZPatterns) {
    var_coeff = ExtractVarCoeff(normalized, pattern);
    if (!var_coeff.empty()) {
      coeffs->ac_latex = var_coeff;
      coeffs->ac = 1.0;
      return true;
    }
    if (MatchTerm(normalized, pattern, &coeff, error)) {
      coeffs->ac += coeff;
      return true;
    }
  }
  
  // Check for variable coefficients in u_yz terms
  for (const char* pattern : LatexPatterns::kDYZPatterns) {
    var_coeff = ExtractVarCoeff(normalized, pattern);
    if (!var_coeff.empty()) {
      coeffs->bc_latex = var_coeff;
      coeffs->bc = 1.0;
      return true;
    }
    if (MatchTerm(normalized, pattern, &coeff, error)) {
      coeffs->bc += coeff;
      return true;
    }
  }
  
  // Third-order derivatives
  // Check for variable coefficients in u_xxx terms
  for (const char* pattern : LatexPatterns::kD3XPatterns) {
    var_coeff = ExtractVarCoeff(normalized, pattern);
    if (!var_coeff.empty()) {
      coeffs->a3_latex = var_coeff;
      coeffs->a3 = 1.0;
      return true;
    }
    if (MatchTerm(normalized, pattern, &coeff, error)) {
      coeffs->a3 += coeff;
      return true;
    }
  }
  
  // Check for variable coefficients in u_yyy terms
  for (const char* pattern : LatexPatterns::kD3YPatterns) {
    var_coeff = ExtractVarCoeff(normalized, pattern);
    if (!var_coeff.empty()) {
      coeffs->b3_latex = var_coeff;
      coeffs->b3 = 1.0;
      return true;
    }
    if (MatchTerm(normalized, pattern, &coeff, error)) {
      coeffs->b3 += coeff;
      return true;
    }
  }
  
  // Check for variable coefficients in u_zzz terms
  for (const char* pattern : LatexPatterns::kD3ZPatterns) {
    var_coeff = ExtractVarCoeff(normalized, pattern);
    if (!var_coeff.empty()) {
      coeffs->az3_latex = var_coeff;
      coeffs->az3 = 1.0;
      return true;
    }
    if (MatchTerm(normalized, pattern, &coeff, error)) {
      coeffs->az3 += coeff;
      return true;
    }
  }
  
  // Fourth-order derivatives
  // Check for variable coefficients in u_xxxx terms
  for (const char* pattern : LatexPatterns::kD4XPatterns) {
    var_coeff = ExtractVarCoeff(normalized, pattern);
    if (!var_coeff.empty()) {
      coeffs->a4_latex = var_coeff;
      coeffs->a4 = 1.0;
      return true;
    }
    if (MatchTerm(normalized, pattern, &coeff, error)) {
      coeffs->a4 += coeff;
      return true;
    }
  }
  
  // Check for variable coefficients in u_yyyy terms
  for (const char* pattern : LatexPatterns::kD4YPatterns) {
    var_coeff = ExtractVarCoeff(normalized, pattern);
    if (!var_coeff.empty()) {
      coeffs->b4_latex = var_coeff;
      coeffs->b4 = 1.0;
      return true;
    }
    if (MatchTerm(normalized, pattern, &coeff, error)) {
      coeffs->b4 += coeff;
      return true;
    }
  }
  
  // Check for variable coefficients in u_zzzz terms
  for (const char* pattern : LatexPatterns::kD4ZPatterns) {
    var_coeff = ExtractVarCoeff(normalized, pattern);
    if (!var_coeff.empty()) {
      coeffs->az4_latex = var_coeff;
      coeffs->az4 = 1.0;
      return true;
    }
    if (MatchTerm(normalized, pattern, &coeff, error)) {
      coeffs->az4 += coeff;
      return true;
    }
  }
  
  // Check for variable coefficients in u terms
  if (MatchTerm(normalized, LatexPatterns::kU, &coeff, error)) {
    // For u terms, check if there's a variable coefficient
    std::string trimmed = Trim(normalized);
    size_t pos = trimmed.find(LatexPatterns::kU);
    if (pos != std::string::npos) {
      std::string coeff_str = trimmed;
      coeff_str.erase(pos, std::strlen(LatexPatterns::kU));
      coeff_str = Trim(coeff_str);
      coeff_str.erase(std::remove(coeff_str.begin(), coeff_str.end(), '*'), coeff_str.end());
      coeff_str = Trim(coeff_str);
      
      double value = 0.0;
      if (!coeff_str.empty() && coeff_str != "+" && coeff_str != "-" && 
          !ParseNumber(coeff_str, &value) &&
          (coeff_str.find('(') != std::string::npos || 
           coeff_str.find('\\') != std::string::npos ||
           coeff_str.find("sin") != std::string::npos ||
           coeff_str.find("cos") != std::string::npos)) {
        coeffs->e_latex = coeff_str;
        coeffs->e = 1.0;
        return true;
      }
    }
    coeffs->e += coeff;
    return true;
  }

  // Check for unsupported mixed derivatives (e.g., with time)
  std::string mixed_error;
  if (DetectMixedDerivative(normalized, &mixed_error)) {
    if (error) {
      *error = mixed_error;
    }
    return false;
  }

  double value = 0.0;
  if (!ParseNumber(normalized, &value)) {
    if (error) {
      *error = "unsupported term: " + original;
    }
    return false;
  }
  coeffs->f += value;
  return true;
}

bool LatexParser::ParseIntegralTerm(const std::string& normalized,
                                    std::vector<IntegralTerm>* integrals,
                                    std::string* error) {
  const size_t pos_iint = normalized.find("\\iint");
  const size_t pos_int = normalized.find("\\int");
  if (pos_iint == std::string::npos && pos_int == std::string::npos) {
    return false;
  }
  const size_t pos = (pos_iint != std::string::npos &&
                      (pos_int == std::string::npos || pos_iint < pos_int))
                         ? pos_iint
                         : pos_int;
  const std::string prefix = normalized.substr(0, pos);
  const std::string body = normalized.substr(pos);

  if (body.find('u') == std::string::npos) {
    if (error) {
      *error = "integral term must include u";
    }
    return true;
  }
  if (body.find("\\partial") != std::string::npos || body.find("u_") != std::string::npos) {
    if (error) {
      *error = "integral of derivatives is not supported";
    }
    return true;
  }
  for (size_t i = 0; i < body.size(); ++i) {
    unsigned char ch = static_cast<unsigned char>(body[i]);
    if (ch == '\\') {
      ++i;
      while (i < body.size() &&
             std::isalpha(static_cast<unsigned char>(body[i])) != 0) {
        ++i;
      }
      if (i >= body.size()) {
        break;
      }
      --i;
      continue;
    }
    if (std::isalpha(ch) != 0) {
      if (body.compare(i, 5, "theta") == 0) {
        i += 4;
        continue;
      }
      if (body.compare(i, 3, "phi") == 0) {
        i += 2;
        continue;
      }
      if (body.compare(i, 2, "dr") == 0) {
        i += 1;
        continue;
      }
      if (body.compare(i, 6, "dtheta") == 0) {
        i += 5;
        continue;
      }
      if (body.compare(i, 4, "dphi") == 0) {
        i += 3;
        continue;
      }
      const char lower = static_cast<char>(std::tolower(ch));
      if (lower != 'u' && lower != 'x' && lower != 'y' && lower != 'z' && lower != 'r' &&
          lower != 't' && lower != 'd') {
        if (error) {
          *error = "integral kernel must be outside the integral";
        }
        return true;
      }
    }
  }

  IntegralTerm term;
  term.coeff = 1.0;
  term.kernel_latex.clear();

  std::string prefix_trim = Trim(prefix);
  if (!prefix_trim.empty()) {
    if (prefix_trim == "+") {
      term.coeff = 1.0;
    } else if (prefix_trim == "-") {
      term.coeff = -1.0;
    } else {
      double full_value = 0.0;
      if (ParseNumber(prefix_trim, &full_value)) {
        term.coeff = full_value;
      } else {
        double leading_value = 0.0;
        size_t consumed = 0;
        if (ParseLeadingNumber(prefix_trim, &leading_value, &consumed)) {
          term.coeff = leading_value;
          std::string remainder = Trim(prefix_trim.substr(consumed));
          if (!remainder.empty() && remainder.front() == '*') {
            remainder.erase(remainder.begin());
          }
          if (!remainder.empty() && remainder.back() == '*') {
            remainder.pop_back();
          }
          remainder = Trim(remainder);
          term.kernel_latex = remainder;
        } else {
          term.coeff = 1.0;
          term.kernel_latex = prefix_trim;
        }
      }
    }
  }

  if (!term.kernel_latex.empty() && term.kernel_latex.find('u') != std::string::npos) {
    if (error) {
      *error = "kernel expressions may not depend on u";
    }
    return true;
  }

  if (integrals) {
    integrals->push_back(term);
  }
  return true;
}

bool LatexParser::ParseNonlinearTerm(const std::string& normalized,
                                     std::vector<NonlinearTerm>* nonlinear,
                                     std::string* error) {
  if (!nonlinear) {
    return false;
  }
  std::string candidate = normalized;
  ReplaceAll(candidate, "u(x,y,t)", "u");
  ReplaceAll(candidate, "u(x,y)", "u");
  ReplaceAll(candidate, "u(x,t)", "u");
  ReplaceAll(candidate, "u(y,t)", "u");
  ReplaceAll(candidate, "u(x)", "u");
  ReplaceAll(candidate, "u(y)", "u");
  ReplaceAll(candidate, "u(t)", "u");
  ReplaceAll(candidate, "u(r,theta,t)", "u");
  ReplaceAll(candidate, "u(r,theta,phi,t)", "u");
  ReplaceAll(candidate, "u(r,theta)", "u");
  ReplaceAll(candidate, "u(r,phi,t)", "u");
  ReplaceAll(candidate, "u(theta,phi,t)", "u");
  ReplaceAll(candidate, "u(r,phi)", "u");
  ReplaceAll(candidate, "u(theta,phi)", "u");
  ReplaceAll(candidate, "u(r,t)", "u");
  ReplaceAll(candidate, "u(theta,t)", "u");
  ReplaceAll(candidate, "u(phi,t)", "u");
  ReplaceAll(candidate, "u(r)", "u");
  ReplaceAll(candidate, "u(theta)", "u");
  ReplaceAll(candidate, "u(phi)", "u");
  ReplaceAll(candidate, "u(z)", "u");

  auto parse_power = [&](int* out_power, double* out_coeff) -> bool {
    size_t pos = candidate.find("u^");
    if (pos == std::string::npos) {
      return false;
    }
    if (candidate.find("u^", pos + 1) != std::string::npos) {
      if (error) {
        *error = "nonlinear term repeats power: " + normalized;
      }
      return true;
    }
    size_t idx = pos + 2;
    int power = 0;
    if (idx < candidate.size() && candidate[idx] == '{') {
      ++idx;
      size_t start = idx;
      while (idx < candidate.size() && std::isdigit(static_cast<unsigned char>(candidate[idx])) != 0) {
        ++idx;
      }
      if (idx == start || idx >= candidate.size() || candidate[idx] != '}') {
        if (error) {
          *error = "invalid power exponent: " + normalized;
        }
        return true;
      }
      power = std::atoi(candidate.substr(start, idx - start).c_str());
      ++idx;
    } else {
      size_t start = idx;
      while (idx < candidate.size() && std::isdigit(static_cast<unsigned char>(candidate[idx])) != 0) {
        ++idx;
      }
      if (idx == start) {
        if (error) {
          *error = "invalid power exponent: " + normalized;
        }
        return true;
      }
      power = std::atoi(candidate.substr(start, idx - start).c_str());
    }
    if (power < 2) {
      return false;
    }
    const std::string pattern = candidate.substr(pos, idx - pos);
    double coeff = 0.0;
    if (!MatchTerm(candidate, pattern, &coeff, error)) {
      return false;
    }
    if (out_power) {
      *out_power = power;
    }
    if (out_coeff) {
      *out_coeff = coeff;
    }
    return true;
  };

  int power = 0;
  double coeff = 0.0;
  if (parse_power(&power, &coeff)) {
    if (!error || error->empty()) {
      NonlinearTerm term;
      term.kind = NonlinearKind::Power;
      term.coeff = coeff;
      term.power = power;
      nonlinear->push_back(term);
    }
    return true;
  }

  if (MatchTerm(candidate, "|u|", &coeff, error)) {
    NonlinearTerm term;
    term.kind = NonlinearKind::Abs;
    term.coeff = coeff;
    term.power = 1;
    nonlinear->push_back(term);
    return true;
  }

  struct FuncPattern {
    const char* pattern;
    NonlinearKind kind;
  };
  const FuncPattern patterns[] = {
      {"\\sin(u)", NonlinearKind::Sin},
      {"sin(u)", NonlinearKind::Sin},
      {"\\sin{u}", NonlinearKind::Sin},
      {"sin{u}", NonlinearKind::Sin},
      {"\\cos(u)", NonlinearKind::Cos},
      {"cos(u)", NonlinearKind::Cos},
      {"\\cos{u}", NonlinearKind::Cos},
      {"cos{u}", NonlinearKind::Cos},
      {"\\exp(u)", NonlinearKind::Exp},
      {"exp(u)", NonlinearKind::Exp},
      {"\\exp{u}", NonlinearKind::Exp},
      {"exp{u}", NonlinearKind::Exp},
      {"\\abs(u)", NonlinearKind::Abs},
      {"abs(u)", NonlinearKind::Abs},
      {"\\abs{u}", NonlinearKind::Abs},
      {"abs{u}", NonlinearKind::Abs},
  };
  for (const auto& entry : patterns) {
    double coeff_value = 0.0;
    if (MatchTerm(candidate, entry.pattern, &coeff_value, error)) {
      NonlinearTerm term;
      term.kind = entry.kind;
      term.coeff = coeff_value;
      term.power = 1;
      nonlinear->push_back(term);
      return true;
    }
  }

  return false;
}

bool LatexParser::ParseNonlinearDerivativeTerm(const std::string& normalized,
                                               std::vector<NonlinearDerivativeTerm>* nonlinear_derivatives,
                                               std::string* error) {
  if (!nonlinear_derivatives) {
    return false;
  }

  std::string candidate = normalized;
  // Normalize u expressions
  ReplaceAll(candidate, "u(x,y,t)", "u");
  ReplaceAll(candidate, "u(x,y)", "u");
  ReplaceAll(candidate, "u(x,t)", "u");
  ReplaceAll(candidate, "u(y,t)", "u");
  ReplaceAll(candidate, "u(x)", "u");
  ReplaceAll(candidate, "u(y)", "u");
  ReplaceAll(candidate, "u(t)", "u");
  ReplaceAll(candidate, "u(z)", "u");

  // Normalize derivative expressions
  ReplaceAll(candidate, "\\partial_xu", "u_x");
  ReplaceAll(candidate, "\\partial_yu", "u_y");
  ReplaceAll(candidate, "\\partial_zu", "u_z");
  ReplaceAll(candidate, "\\partialu/\\partialx", "u_x");
  ReplaceAll(candidate, "\\partialu/\\partialy", "u_y");
  ReplaceAll(candidate, "\\partialu/\\partialz", "u_z");
  ReplaceAll(candidate, "\\frac{\\partialu}{\\partialx}", "u_x");
  ReplaceAll(candidate, "\\frac{\\partialu}{\\partialy}", "u_y");
  ReplaceAll(candidate, "\\frac{\\partialu}{\\partialz}", "u_z");

  // Patterns for nonlinear derivative terms
  struct Pattern {
    const char* pattern;
    NonlinearDerivativeKind kind;
  };

  const Pattern patterns[] = {
    // u * u_x patterns
    {"u*u_x", NonlinearDerivativeKind::UUx},
    {"uu_x", NonlinearDerivativeKind::UUx},
    {"u\\partial_xu", NonlinearDerivativeKind::UUx},
    {"u\\partialu/\\partialx", NonlinearDerivativeKind::UUx},
    // u * u_y patterns
    {"u*u_y", NonlinearDerivativeKind::UUy},
    {"uu_y", NonlinearDerivativeKind::UUy},
    {"u\\partial_yu", NonlinearDerivativeKind::UUy},
    {"u\\partialu/\\partialy", NonlinearDerivativeKind::UUy},
    // u * u_z patterns
    {"u*u_z", NonlinearDerivativeKind::UUz},
    {"uu_z", NonlinearDerivativeKind::UUz},
    {"u\\partial_zu", NonlinearDerivativeKind::UUz},
    {"u\\partialu/\\partialz", NonlinearDerivativeKind::UUz},
    // u_x^2 patterns
    {"u_x^2", NonlinearDerivativeKind::UxUx},
    {"u_x*u_x", NonlinearDerivativeKind::UxUx},
    {"(u_x)^2", NonlinearDerivativeKind::UxUx},
    {"\\partial_xu^2", NonlinearDerivativeKind::UxUx},
    // u_y^2 patterns
    {"u_y^2", NonlinearDerivativeKind::UyUy},
    {"u_y*u_y", NonlinearDerivativeKind::UyUy},
    {"(u_y)^2", NonlinearDerivativeKind::UyUy},
    {"\\partial_yu^2", NonlinearDerivativeKind::UyUy},
    // u_z^2 patterns
    {"u_z^2", NonlinearDerivativeKind::UzUz},
    {"u_z*u_z", NonlinearDerivativeKind::UzUz},
    {"(u_z)^2", NonlinearDerivativeKind::UzUz},
    {"\\partial_zu^2", NonlinearDerivativeKind::UzUz},
    // |∇u|² patterns
    {"|\\nablau|^2", NonlinearDerivativeKind::GradSquared},
    {"|\\nablau|^2", NonlinearDerivativeKind::GradSquared},
    {"(\\nablau)^2", NonlinearDerivativeKind::GradSquared},
    {"\\nablau^2", NonlinearDerivativeKind::GradSquared},
    {"u_x^2+u_y^2", NonlinearDerivativeKind::GradSquared},
    {"u_x^2+u_y^2+u_z^2", NonlinearDerivativeKind::GradSquared},
  };

  for (const auto& entry : patterns) {
    double coeff = 0.0;
    if (MatchTerm(candidate, entry.pattern, &coeff, error)) {
      NonlinearDerivativeTerm term;
      term.kind = entry.kind;
      term.coeff = coeff;
      nonlinear_derivatives->push_back(term);
      return true;
    }
  }

  return false;
}

// Parse a term and route it to the appropriate field's coefficients
bool LatexParser::ParseTermMultiField(const std::string& term,
                                      const std::string& primary_field,
                                      std::map<std::string, PDECoefficients>* field_coeffs,
                                      std::set<std::string>* detected_fields,
                                      std::vector<IntegralTerm>* integrals,
                                      std::vector<NonlinearTerm>* nonlinear,
                                      std::vector<NonlinearDerivativeTerm>* nonlinear_derivatives,
                                      std::string* error) {
  std::string original = Trim(term);
  if (original.empty()) {
    return true;
  }

  const std::string normalized = NormalizeLatex(term);
  if (normalized.empty()) {
    return true;
  }

  // Detect which field variable this term refers to
  std::string detected_field = LatexPatterns::DetectFieldVariable(normalized);

  // If no field detected, try standard parsing (might be a constant or RHS term)
  if (detected_field.empty()) {
    // Check for integrals first
    std::string integral_error;
    if (ParseIntegralTerm(normalized, integrals, &integral_error)) {
      if (!integral_error.empty()) {
        if (error) *error = integral_error;
        return false;
      }
      return true;
    }

    // Try to parse as a number (constant term)
    double value = 0.0;
    if (ParseNumber(normalized, &value)) {
      // Constant goes to RHS (stored as f)
      if (field_coeffs->find(primary_field) == field_coeffs->end()) {
        (*field_coeffs)[primary_field] = PDECoefficients();
      }
      (*field_coeffs)[primary_field].f += value;
      detected_fields->insert(primary_field);
      return true;
    }

    // Unknown term - treat as primary field's term or error
    if (error) {
      *error = "unsupported term: " + original;
    }
    return false;
  }

  // We detected a field variable - parse the term for that field
  detected_fields->insert(detected_field);

  // Ensure we have a coefficients entry for this field
  if (field_coeffs->find(detected_field) == field_coeffs->end()) {
    (*field_coeffs)[detected_field] = PDECoefficients();
  }
  PDECoefficients* coeffs = &(*field_coeffs)[detected_field];

  double coeff = 0.0;

  // Generate and check patterns for this field
  // Laplacian patterns
  for (const auto& pattern : LatexPatterns::GenerateLaplacianPatterns(detected_field)) {
    if (MatchTermForField(normalized, pattern, &coeff, error)) {
      coeffs->a += coeff;
      coeffs->b += coeff;
      coeffs->az += coeff;
      return true;
    }
  }

  // Second derivatives
  for (const auto& pattern : LatexPatterns::GenerateD2XPatterns(detected_field)) {
    if (MatchTermForField(normalized, pattern, &coeff, error)) {
      coeffs->a += coeff;
      return true;
    }
  }
  for (const auto& pattern : LatexPatterns::GenerateD2YPatterns(detected_field)) {
    if (MatchTermForField(normalized, pattern, &coeff, error)) {
      coeffs->b += coeff;
      return true;
    }
  }
  for (const auto& pattern : LatexPatterns::GenerateD2ZPatterns(detected_field)) {
    if (MatchTermForField(normalized, pattern, &coeff, error)) {
      coeffs->az += coeff;
      return true;
    }
  }
  for (const auto& pattern : LatexPatterns::GenerateD2TPatterns(detected_field)) {
    if (MatchTermForField(normalized, pattern, &coeff, error)) {
      coeffs->utt += coeff;
      return true;
    }
  }

  // First derivatives
  for (const auto& pattern : LatexPatterns::GenerateDXPatterns(detected_field)) {
    if (MatchTermForField(normalized, pattern, &coeff, error)) {
      coeffs->c += coeff;
      return true;
    }
  }
  for (const auto& pattern : LatexPatterns::GenerateDYPatterns(detected_field)) {
    if (MatchTermForField(normalized, pattern, &coeff, error)) {
      coeffs->d += coeff;
      return true;
    }
  }
  for (const auto& pattern : LatexPatterns::GenerateDZPatterns(detected_field)) {
    if (MatchTermForField(normalized, pattern, &coeff, error)) {
      coeffs->dz += coeff;
      return true;
    }
  }
  for (const auto& pattern : LatexPatterns::GenerateDTPatterns(detected_field)) {
    if (MatchTermForField(normalized, pattern, &coeff, error)) {
      coeffs->ut += coeff;
      return true;
    }
  }

  // Mixed derivatives
  for (const auto& pattern : LatexPatterns::GenerateDXYPatterns(detected_field)) {
    if (MatchTermForField(normalized, pattern, &coeff, error)) {
      coeffs->ab += coeff;
      return true;
    }
  }
  for (const auto& pattern : LatexPatterns::GenerateDXZPatterns(detected_field)) {
    if (MatchTermForField(normalized, pattern, &coeff, error)) {
      coeffs->ac += coeff;
      return true;
    }
  }
  for (const auto& pattern : LatexPatterns::GenerateDYZPatterns(detected_field)) {
    if (MatchTermForField(normalized, pattern, &coeff, error)) {
      coeffs->bc += coeff;
      return true;
    }
  }

  // Plain field variable (e.g., just "v" or "2*v")
  if (MatchTermForField(normalized, detected_field, &coeff, error)) {
    coeffs->e += coeff;
    return true;
  }

  // If we detected a field but couldn't parse it, report error
  if (error) {
    *error = "unsupported term for field '" + detected_field + "': " + original;
  }
  return false;
}

// Parse a term for a specific field variable
bool LatexParser::ParseTermForField(const std::string& normalized_term,
                                    const std::string& field,
                                    PDECoefficients* coeffs,
                                    double* out_coeff,
                                    std::string* error) {
  if (!coeffs || !out_coeff) {
    return false;
  }

  double coeff = 0.0;

  // Check Laplacian patterns
  for (const auto& pattern : LatexPatterns::GenerateLaplacianPatterns(field)) {
    if (MatchTermForField(normalized_term, pattern, &coeff, error)) {
      coeffs->a += coeff;
      coeffs->b += coeff;
      coeffs->az += coeff;
      *out_coeff = coeff;
      return true;
    }
  }

  // Check second derivatives
  for (const auto& pattern : LatexPatterns::GenerateD2XPatterns(field)) {
    if (MatchTermForField(normalized_term, pattern, &coeff, error)) {
      coeffs->a += coeff;
      *out_coeff = coeff;
      return true;
    }
  }
  for (const auto& pattern : LatexPatterns::GenerateD2YPatterns(field)) {
    if (MatchTermForField(normalized_term, pattern, &coeff, error)) {
      coeffs->b += coeff;
      *out_coeff = coeff;
      return true;
    }
  }
  for (const auto& pattern : LatexPatterns::GenerateD2ZPatterns(field)) {
    if (MatchTermForField(normalized_term, pattern, &coeff, error)) {
      coeffs->az += coeff;
      *out_coeff = coeff;
      return true;
    }
  }

  // Check first derivatives
  for (const auto& pattern : LatexPatterns::GenerateDXPatterns(field)) {
    if (MatchTermForField(normalized_term, pattern, &coeff, error)) {
      coeffs->c += coeff;
      *out_coeff = coeff;
      return true;
    }
  }
  for (const auto& pattern : LatexPatterns::GenerateDYPatterns(field)) {
    if (MatchTermForField(normalized_term, pattern, &coeff, error)) {
      coeffs->d += coeff;
      *out_coeff = coeff;
      return true;
    }
  }
  for (const auto& pattern : LatexPatterns::GenerateDZPatterns(field)) {
    if (MatchTermForField(normalized_term, pattern, &coeff, error)) {
      coeffs->dz += coeff;
      *out_coeff = coeff;
      return true;
    }
  }
  for (const auto& pattern : LatexPatterns::GenerateDTPatterns(field)) {
    if (MatchTermForField(normalized_term, pattern, &coeff, error)) {
      coeffs->ut += coeff;
      *out_coeff = coeff;
      return true;
    }
  }

  // Check time second derivative
  for (const auto& pattern : LatexPatterns::GenerateD2TPatterns(field)) {
    if (MatchTermForField(normalized_term, pattern, &coeff, error)) {
      coeffs->utt += coeff;
      *out_coeff = coeff;
      return true;
    }
  }

  // Check mixed derivatives
  for (const auto& pattern : LatexPatterns::GenerateDXYPatterns(field)) {
    if (MatchTermForField(normalized_term, pattern, &coeff, error)) {
      coeffs->ab += coeff;
      *out_coeff = coeff;
      return true;
    }
  }
  for (const auto& pattern : LatexPatterns::GenerateDXZPatterns(field)) {
    if (MatchTermForField(normalized_term, pattern, &coeff, error)) {
      coeffs->ac += coeff;
      *out_coeff = coeff;
      return true;
    }
  }
  for (const auto& pattern : LatexPatterns::GenerateDYZPatterns(field)) {
    if (MatchTermForField(normalized_term, pattern, &coeff, error)) {
      coeffs->bc += coeff;
      *out_coeff = coeff;
      return true;
    }
  }

  // Check plain field (e.g., "v" alone)
  if (MatchTermForField(normalized_term, field, &coeff, error)) {
    coeffs->e += coeff;
    *out_coeff = coeff;
    return true;
  }

  return false;
}


