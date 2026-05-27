#include "residual_operator.h"

#include <cmath>

namespace {
bool HasNonlinearDerivativeTerms(const LatexParseResult& parse) {
  return !parse.nonlinear_derivatives.empty();
}

bool IsConservationFluxPattern(const LatexParseResult& parse) {
  if (!parse.op.rhs_latex.empty()) {
    return false;
  }
  if (!HasNonlinearDerivativeTerms(parse)) {
    return false;
  }
  for (const auto& term : parse.nonlinear_derivatives) {
    if (term.kind == NonlinearDerivativeKind::UUx ||
        term.kind == NonlinearDerivativeKind::UUy ||
        term.kind == NonlinearDerivativeKind::UUz) {
      return true;
    }
  }
  return false;
}
}  // namespace

ProblemClassification ClassifyProblem(const LatexParseResult& parse) {
  ProblemClassification out;
  const bool time_dep = std::abs(parse.coeffs.ut) > 1e-12 || std::abs(parse.coeffs.utt) > 1e-12;

  if (parse.conservation_divergence && !parse.conservation_flux_latex.empty() && time_dep) {
    out.form = ProblemForm::ConservationLaw;
    out.recommended_discretization = Discretization::FiniteVolume;
    out.flux_latex = parse.conservation_flux_latex;
    out.note = "conservation divergence form detected in LaTeX";
    return out;
  }

  if (!parse.nonlinear.empty() && HasNonlinearDerivativeTerms(parse)) {
    out.form = ProblemForm::NonlinearResidual;
    out.recommended_discretization = Discretization::FiniteDifference;
    out.note = "combined pointwise and derivative nonlinearities";
    return out;
  }

  if (HasNonlinearDerivativeTerms(parse) && time_dep) {
    if (IsConservationFluxPattern(parse)) {
      out.form = ProblemForm::ConservationLaw;
      out.recommended_discretization = Discretization::FiniteVolume;
      out.flux_latex = "0.5*u^2";
      out.note = "hyperbolic conservation law (Burgers-type)";
      return out;
    }
    out.form = ProblemForm::NonlinearResidual;
    out.recommended_discretization = Discretization::FiniteDifference;
    out.note = "nonlinear derivative terms";
    return out;
  }

  if (!parse.nonlinear.empty()) {
    out.form = ProblemForm::NonlinearResidual;
    out.recommended_discretization = Discretization::FiniteDifference;
    out.note = "nonlinear reaction terms";
    return out;
  }

  out.form = ProblemForm::LinearOperator;
  out.recommended_discretization = Discretization::FiniteDifference;
  return out;
}

void ApplyClassificationToInput(const ProblemClassification& classification, SolveInput* input) {
  if (!input) {
    return;
  }
  input->problem_form = classification.form;
  if (input->discretization == Discretization::FiniteDifference &&
      classification.recommended_discretization == Discretization::FiniteVolume) {
    input->discretization = classification.recommended_discretization;
  }
  if (!classification.flux_latex.empty() && input->conservation.flux_latex.empty()) {
    input->conservation.flux_latex = classification.flux_latex;
  }
}

std::string ProblemFormToString(ProblemForm form) {
  switch (form) {
    case ProblemForm::LinearOperator:
      return "linear_operator";
    case ProblemForm::NonlinearResidual:
      return "nonlinear_residual";
    case ProblemForm::ConservationLaw:
      return "conservation_law";
  }
  return "unknown";
}

std::string DiscretizationToString(Discretization disc) {
  switch (disc) {
    case Discretization::FiniteDifference:
      return "fd";
    case Discretization::FiniteVolume:
      return "fv";
  }
  return "unknown";
}
