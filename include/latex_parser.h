#ifndef LATEX_PARSER_H
#define LATEX_PARSER_H

#include <map>
#include <set>
#include <string>
#include "pde_types.h"

struct LatexParseResult {
  bool ok = false;
  std::string error;
  PDECoefficients coeffs;
  std::vector<IntegralTerm> integrals;
  std::vector<NonlinearTerm> nonlinear;
  std::vector<NonlinearDerivativeTerm> nonlinear_derivatives;
  PDEOperator op;

  // Multi-field extensions: detected field variable and cross-field coupling
  std::string primary_field;                           // Primary field in this equation (e.g., "u")
  std::map<std::string, PDECoefficients> field_coeffs; // Per-field coefficients (including coupling)
  std::set<std::string> detected_fields;               // All field variables detected in the equation
};

// Result for parsing a multi-field system of equations
struct MultiFieldParseResult {
  bool ok = false;
  std::string error;
  std::vector<LatexParseResult> equations;    // One per field equation
  std::set<std::string> all_fields;           // Union of all detected fields
  MultiFieldEquation multi_field_eq;          // Structured multi-field representation
  CouplingAnalysis coupling;                  // Coupling pattern analysis
};

class LatexParser {
 public:
  // Parse a single equation (backward compatible)
  LatexParseResult Parse(const std::string& latex) const;

  // Parse with explicit primary field name (for multi-field systems)
  LatexParseResult ParseForField(const std::string& latex, const std::string& primary_field) const;

  // Parse a system of equations (one per field)
  MultiFieldParseResult ParseMultiField(const std::vector<std::string>& equations,
                                        const std::vector<std::string>& field_names) const;

  // Analyze coupling pattern from parsed results
  static CouplingAnalysis AnalyzeCoupling(const MultiFieldParseResult& result);

  // Utility methods used by multi-field parsing helpers
  static bool ParseNumber(const std::string& text, double* out_value);
  static bool IsArgumentList(const std::string& text);
  static std::string Trim(const std::string& input);

 private:
  LatexParseResult ParseExpression(const std::string& expr) const;
  LatexParseResult ParseExpressionForField(const std::string& expr, const std::string& primary_field) const;
  static std::string StripDecorations(const std::string& input);
  static std::vector<std::string> SplitTerms(const std::string& expr, std::string* error);
  static bool ParseTerm(const std::string& term, PDECoefficients* coeffs,
                        std::vector<IntegralTerm>* integrals,
                        std::vector<NonlinearTerm>* nonlinear,
                        std::vector<NonlinearDerivativeTerm>* nonlinear_derivatives, std::string* error);
  // Multi-field term parsing: parses term and routes to appropriate field's coefficients
  static bool ParseTermMultiField(const std::string& term,
                                  const std::string& primary_field,
                                  std::map<std::string, PDECoefficients>* field_coeffs,
                                  std::set<std::string>* detected_fields,
                                  std::vector<IntegralTerm>* integrals,
                                  std::vector<NonlinearTerm>* nonlinear,
                                  std::vector<NonlinearDerivativeTerm>* nonlinear_derivatives,
                                  std::string* error);
  // Parse a term for a specific field variable (returns true if term matches this field)
  static bool ParseTermForField(const std::string& normalized_term,
                                const std::string& field,
                                PDECoefficients* coeffs,
                                double* out_coeff,
                                std::string* error);
  static bool ParseIntegralTerm(const std::string& normalized, std::vector<IntegralTerm>* integrals,
                                std::string* error);
  static bool ParseNonlinearTerm(const std::string& normalized,
                                 std::vector<NonlinearTerm>* nonlinear, std::string* error);
  static bool ParseNonlinearDerivativeTerm(const std::string& normalized,
                                           std::vector<NonlinearDerivativeTerm>* nonlinear_derivatives,
                                           std::string* error);
  static bool ParseLeadingNumber(const std::string& text, double* out_value, size_t* out_length);
  static bool MatchTerm(const std::string& term, const std::string& pattern, double* out_coeff, std::string* error);

  // Lexer helpers (implemented in src/latex_lexer.cpp)
  static void ReplaceAll(std::string& text, const std::string& needle, const std::string& repl);
  static std::string NormalizeLatex(const std::string& input);
  static void StripTrailingArgumentList(std::string* text);
  static bool ParseLatexArgument(const std::string& text, size_t* index, std::string* out);
  static bool StripEnclosing(std::string* text, char open, char close);
  static bool DetectMixedDerivative(const std::string& text, std::string* error);
};

#endif
