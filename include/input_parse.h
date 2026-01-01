#ifndef INPUT_PARSE_H
#define INPUT_PARSE_H

#include <string>
#include <vector>

#include "pde_types.h"

struct ParseResult {
  bool ok = false;
  std::string error;
};

ParseResult ParseDomain(const std::string& text, Domain* out);
ParseResult ParseGrid(const std::string& text, Domain* out);
ParseResult ApplyBoundarySpec(const std::string& spec, BoundarySet* bc);
ParseResult ValidateBoundaryConditions(const BoundarySet& bc, const Domain& d);

#endif
