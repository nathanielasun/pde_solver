#ifndef RESIDUAL_OPERATOR_H
#define RESIDUAL_OPERATOR_H

#include <string>

#include "latex_parser.h"
#include "pde_types.h"

struct ProblemClassification {
  ProblemForm form = ProblemForm::LinearOperator;
  Discretization recommended_discretization = Discretization::FiniteDifference;
  std::string flux_latex;  // populated for conservation laws
  std::string note;
};

ProblemClassification ClassifyProblem(const LatexParseResult& parse);
void ApplyClassificationToInput(const ProblemClassification& classification, SolveInput* input);

std::string ProblemFormToString(ProblemForm form);
std::string DiscretizationToString(Discretization disc);

#endif  // RESIDUAL_OPERATOR_H
