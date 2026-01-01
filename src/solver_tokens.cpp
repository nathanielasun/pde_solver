#include "solver_tokens.h"

#include <algorithm>
#include <cctype>

namespace {
std::string ToLowerCopy(const std::string& text) {
  std::string out = text;
  std::transform(out.begin(), out.end(), out.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  return out;
}
}

bool IsBackendToken(const std::string& text) {
  const std::string token = ToLowerCopy(text);
  return token == "auto" || token == "cpu" || token == "cuda" || token == "metal" || token == "tpu";
}

std::string BackendToken(BackendKind kind) {
  switch (kind) {
    case BackendKind::CPU:
      return "cpu";
    case BackendKind::CUDA:
      return "cuda";
    case BackendKind::Metal:
      return "metal";
    case BackendKind::TPU:
      return "tpu";
    case BackendKind::Auto:
    default:
      return "auto";
  }
}

bool IsMethodToken(const std::string& text) {
  const std::string token = ToLowerCopy(text);
  return token == "jacobi" || token == "gs" || token == "sor" || token == "cg" ||
         token == "bicgstab" || token == "gmres" || token == "mg";
}

SolveMethod ParseSolveMethodToken(const std::string& text) {
  const std::string token = ToLowerCopy(text);
  if (token == "gs") {
    return SolveMethod::GaussSeidel;
  }
  if (token == "sor") {
    return SolveMethod::SOR;
  }
  if (token == "cg") {
    return SolveMethod::CG;
  }
  if (token == "bicgstab") {
    return SolveMethod::BiCGStab;
  }
  if (token == "gmres") {
    return SolveMethod::GMRES;
  }
  if (token == "mg") {
    return SolveMethod::MultigridVcycle;
  }
  return SolveMethod::Jacobi;
}

std::string MethodToken(SolveMethod method) {
  switch (method) {
    case SolveMethod::GaussSeidel:
      return "gs";
    case SolveMethod::SOR:
      return "sor";
    case SolveMethod::CG:
      return "cg";
    case SolveMethod::BiCGStab:
      return "bicgstab";
    case SolveMethod::GMRES:
      return "gmres";
    case SolveMethod::MultigridVcycle:
      return "mg";
    case SolveMethod::Jacobi:
    default:
      return "jacobi";
  }
}
