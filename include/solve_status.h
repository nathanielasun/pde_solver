#ifndef SOLVE_STATUS_H
#define SOLVE_STATUS_H

#include <string>

enum class SolveStatus {
  Ok = 0,
  ParseError = 1,
  UnsupportedFeature = 2,
  NewtonFailed = 3,
  CflViolated = 4,
  Diverged = 5,
  Cancelled = 6,
  ValidationFailed = 7,
};

int SolveStatusExitCode(SolveStatus status);
std::string SolveStatusToString(SolveStatus status);

#endif  // SOLVE_STATUS_H
