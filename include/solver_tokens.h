#ifndef SOLVER_TOKENS_H
#define SOLVER_TOKENS_H

#include <string>

#include "backend.h"
#include "pde_types.h"

bool IsBackendToken(const std::string& text);
std::string BackendToken(BackendKind kind);

bool IsMethodToken(const std::string& text);
SolveMethod ParseSolveMethodToken(const std::string& text);
std::string MethodToken(SolveMethod method);

#endif  // SOLVER_TOKENS_H
