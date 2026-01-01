#ifndef COEFFICIENT_EVALUATOR_H
#define COEFFICIENT_EVALUATOR_H

#include <optional>
#include <string>

#include "expression_eval.h"
#include "pde_types.h"

struct CoefficientEvaluator {
  bool ok = true;
  std::string error;
  bool has_variable = false;

  std::optional<ExpressionEvaluator> a;
  std::optional<ExpressionEvaluator> b;
  std::optional<ExpressionEvaluator> az;
  std::optional<ExpressionEvaluator> c;
  std::optional<ExpressionEvaluator> d;
  std::optional<ExpressionEvaluator> dz;
  std::optional<ExpressionEvaluator> e;
  std::optional<ExpressionEvaluator> ab;
  std::optional<ExpressionEvaluator> ac;
  std::optional<ExpressionEvaluator> bc;
  std::optional<ExpressionEvaluator> a3;
  std::optional<ExpressionEvaluator> b3;
  std::optional<ExpressionEvaluator> az3;
  std::optional<ExpressionEvaluator> a4;
  std::optional<ExpressionEvaluator> b4;
  std::optional<ExpressionEvaluator> az4;
};

CoefficientEvaluator BuildCoefficientEvaluator(const PDECoefficients& pde);
double EvalCoefficient(const std::optional<ExpressionEvaluator>& evaluator,
                       double constant, double x, double y, double z, double t);

#endif  // COEFFICIENT_EVALUATOR_H
