#ifndef EXPRESSION_EVAL_H
#define EXPRESSION_EVAL_H

#include <memory>
#include <string>

class ExpressionEvaluator {
 public:
  struct Node;

  static ExpressionEvaluator ParseLatex(const std::string& text);

  ~ExpressionEvaluator();
  ExpressionEvaluator(const ExpressionEvaluator&) = delete;
  ExpressionEvaluator& operator=(const ExpressionEvaluator&) = delete;
  ExpressionEvaluator(ExpressionEvaluator&&) noexcept;
  ExpressionEvaluator& operator=(ExpressionEvaluator&&) noexcept;

  bool ok() const;
  const std::string& error() const;
  double Eval(double x, double y) const;
  double Eval(double x, double y, double z) const;
  double Eval(double x, double y, double z, double t) const;

 private:
  explicit ExpressionEvaluator(std::unique_ptr<Node> root, std::string error);

  double EvalNode(const Node* node, double x, double y, double z, double t = 0.0) const;

  std::unique_ptr<Node> root_;
  std::string error_;
};

#endif
