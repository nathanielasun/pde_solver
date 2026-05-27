#ifndef LATEX_TOKEN_REGISTRY_H
#define LATEX_TOKEN_REGISTRY_H

#include <cstddef>
#include <string>
#include <vector>

// Authoritative catalog of LaTeX tokens recognized by the PDE parser, normalizer,
// and related subsystems. Pattern strings for derivatives remain in latex_patterns.h
// during migration; this registry aggregates them for documentation and tooling.

struct LatexTokenEntry {
  const char* category = "";
  const char* pattern = "";
  const char* maps_to = "";
  const char* display_note = "";
};

struct LatexNormalizationRule {
  const char* from = "";
  const char* to = "";
};

struct LatexRewriteRule {
  const char* from = "";
  const char* to = "";
  const char* note = "";
};

namespace LatexTokenRegistry {

// Build the full token catalog (derivative patterns + nonlinear + integrals + etc.).
const std::vector<LatexTokenEntry>& AllEntries();

const std::vector<LatexNormalizationRule>& NormalizationRules();
const std::vector<LatexRewriteRule>& ConservationRewrites();
const std::vector<LatexTokenEntry>& UnsupportedPatterns();
const std::vector<LatexTokenEntry>& ExpressionEvalTokens();
const std::vector<std::string>& ReservedSymbols();

// Export catalog as JSON for tools/generate_latex_token_docs.py
std::string ExportCatalogJson();

}  // namespace LatexTokenRegistry

#endif  // LATEX_TOKEN_REGISTRY_H
