#include "solve_status.h"

int SolveStatusExitCode(SolveStatus status) {
  return static_cast<int>(status);
}

std::string SolveStatusToString(SolveStatus status) {
  switch (status) {
    case SolveStatus::Ok:
      return "ok";
    case SolveStatus::ParseError:
      return "parse_error";
    case SolveStatus::UnsupportedFeature:
      return "unsupported_feature";
    case SolveStatus::NewtonFailed:
      return "newton_failed";
    case SolveStatus::CflViolated:
      return "cfl_violated";
    case SolveStatus::Diverged:
      return "diverged";
    case SolveStatus::Cancelled:
      return "cancelled";
    case SolveStatus::ValidationFailed:
      return "validation_failed";
  }
  return "unknown";
}
