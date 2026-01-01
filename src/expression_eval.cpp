#include "expression_eval.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "string_utils.h"

constexpr double kPi = 3.14159265358979323846;

struct ExpressionEvaluator::Node {
  enum class Type {
    Number,
    VarX,
    VarY,
    VarZ,
    VarT,
    Add,
    Sub,
    Mul,
    Div,
    Pow,
    Neg,
    Func,
  };

  Type type = Type::Number;
  double value = 0.0;
  std::string func;
  std::unique_ptr<Node> left;
  std::unique_ptr<Node> right;
};

namespace {
using Node = ExpressionEvaluator::Node;

struct Token {
  enum class Type {
    Number,
    Ident,
    Plus,
    Minus,
    Mul,
    Div,
    Pow,
    LParen,
    RParen,
    End,
  };

  Type type = Type::End;
  double value = 0.0;
  std::string text;
};

bool IsIdentifierStart(char ch) {
  return std::isalpha(static_cast<unsigned char>(ch)) != 0;
}

bool IsIdentifierChar(char ch) {
  return std::isalpha(static_cast<unsigned char>(ch)) != 0;
}

using pde::ToLower;

bool ParseGroup(const std::string& input, size_t* index, char open, char close,
                std::string* group, std::string* error) {
  if (*index >= input.size() || input[*index] != open) {
    if (error) {
      *error = std::string("expected '") + open + "'";
    }
    return false;
  }
  size_t i = *index + 1;
  int depth = 0;
  for (; i < input.size(); ++i) {
    if (input[i] == open) {
      ++depth;
    } else if (input[i] == close) {
      if (depth == 0) {
        *group = input.substr(*index + 1, i - (*index + 1));
        *index = i + 1;
        return true;
      }
      --depth;
    }
  }
  if (error) {
    *error = std::string("unterminated '") + open + "'";
  }
  return false;
}

void SkipSpaces(const std::string& input, size_t* index) {
  while (*index < input.size() &&
         std::isspace(static_cast<unsigned char>(input[*index])) != 0) {
    ++(*index);
  }
}

std::string ConvertLatexToInfix(const std::string& input, std::string* error);

bool ParseCommand(const std::string& input, size_t* index, std::string* output,
                  std::string* error);

bool ParseLatexArgument(const std::string& input, size_t* index, std::string* arg,
                        std::string* error) {
  SkipSpaces(input, index);
  if (*index >= input.size()) {
    if (error) {
      *error = "missing function argument";
    }
    return false;
  }
  if (input[*index] == '{') {
    std::string group;
    if (!ParseGroup(input, index, '{', '}', &group, error)) {
      return false;
    }
    *arg = ConvertLatexToInfix(group, error);
    return error ? error->empty() : true;
  }
  if (input[*index] == '(') {
    std::string group;
    if (!ParseGroup(input, index, '(', ')', &group, error)) {
      return false;
    }
    *arg = ConvertLatexToInfix(group, error);
    return error ? error->empty() : true;
  }
  if (input[*index] == '\\') {
    std::string command_out;
    if (!ParseCommand(input, index, &command_out, error)) {
      return false;
    }
    if (command_out.empty()) {
      if (error) {
        *error = "invalid function argument";
      }
      return false;
    }
    *arg = command_out;
    return true;
  }
  if (IsIdentifierStart(input[*index])) {
    *arg = std::string(1, input[*index]);
    ++(*index);
    return true;
  }
  if (std::isdigit(static_cast<unsigned char>(input[*index])) || input[*index] == '.') {
    const char* start = input.c_str() + *index;
    char* end = nullptr;
    std::strtod(start, &end);
    if (end == start) {
      if (error) {
        *error = "invalid number in argument";
      }
      return false;
    }
    *arg = input.substr(*index, static_cast<size_t>(end - start));
    *index += static_cast<size_t>(end - start);
    return true;
  }
  if (error) {
    *error = "unsupported function argument";
  }
  return false;
}

bool ParseCommand(const std::string& input, size_t* index, std::string* output,
                  std::string* error) {
  if (*index >= input.size() || input[*index] != '\\') {
    if (error) {
      *error = "expected latex command";
    }
    return false;
  }
  size_t start = *index + 1;
  size_t end = start;
  while (end < input.size() && std::isalpha(static_cast<unsigned char>(input[end])) != 0) {
    ++end;
  }
  if (end == start) {
    if (end < input.size() && input[end] == ',') {
      *index = end + 1;
      output->clear();
      return true;
    }
    if (error) {
      *error = "invalid latex command";
    }
    return false;
  }

  const std::string cmd = input.substr(start, end - start);
  *index = end;

  if (cmd == "left" || cmd == "right") {
    output->clear();
    return true;
  }
  if (cmd == "cdot" || cmd == "times") {
    *output = "*";
    return true;
  }
  if (cmd == "pi") {
    *output = "pi";
    return true;
  }
  if (cmd == "theta" || cmd == "vartheta") {
    *output = "theta";
    return true;
  }
  if (cmd == "phi" || cmd == "varphi") {
    *output = "phi";
    return true;
  }
  if (cmd == "frac") {
    SkipSpaces(input, index);
    std::string num;
    std::string den;
    if (!ParseGroup(input, index, '{', '}', &num, error)) {
      return false;
    }
    SkipSpaces(input, index);
    if (!ParseGroup(input, index, '{', '}', &den, error)) {
      return false;
    }
    std::string num_conv = ConvertLatexToInfix(num, error);
    if (error && !error->empty()) {
      return false;
    }
    std::string den_conv = ConvertLatexToInfix(den, error);
    if (error && !error->empty()) {
      return false;
    }
    *output = "(" + num_conv + ")/(" + den_conv + ")";
    return true;
  }
  if (cmd == "sqrt") {
    SkipSpaces(input, index);
    std::string group;
    if (!ParseGroup(input, index, '{', '}', &group, error)) {
      return false;
    }
    std::string arg = ConvertLatexToInfix(group, error);
    if (error && !error->empty()) {
      return false;
    }
    *output = "sqrt(" + arg + ")";
    return true;
  }

  const std::string func = cmd == "ln" ? "log" : cmd;
  if (func == "sin" || func == "cos" || func == "tan" || func == "exp" || func == "log" ||
      func == "abs") {
    std::string arg;
    if (!ParseLatexArgument(input, index, &arg, error)) {
      return false;
    }
    *output = func + "(" + arg + ")";
    return true;
  }

  if (error) {
    *error = "unsupported latex command: \\" + cmd;
  }
  return false;
}

std::string ConvertLatexToInfix(const std::string& input, std::string* error) {
  std::string out;
  out.reserve(input.size());
  size_t i = 0;
  while (i < input.size()) {
    const char ch = input[i];
    if (std::isspace(static_cast<unsigned char>(ch)) != 0 || ch == '$') {
      ++i;
      continue;
    }
    if (ch == '\\') {
      std::string cmd_out;
      if (!ParseCommand(input, &i, &cmd_out, error)) {
        return "";
      }
      out.append(cmd_out);
      continue;
    }
    if (ch == '{') {
      std::string group;
      if (!ParseGroup(input, &i, '{', '}', &group, error)) {
        return "";
      }
      std::string inner = ConvertLatexToInfix(group, error);
      if (error && !error->empty()) {
        return "";
      }
      out.append("(").append(inner).append(")");
      continue;
    }
    if (ch == '}') {
      if (error) {
        *error = "unexpected '}' in expression";
      }
      return "";
    }
    if (ch == '\\') {
      if (error) {
        *error = "unsupported latex command";
      }
      return "";
    }
    out.push_back(ch);
    ++i;
  }
  return out;
}

std::vector<Token> Tokenize(const std::string& text, std::string* error) {
  std::vector<Token> tokens;
  size_t i = 0;
  while (i < text.size()) {
    const char ch = text[i];
    if (std::isspace(static_cast<unsigned char>(ch)) != 0) {
      ++i;
      continue;
    }
    if (std::isdigit(static_cast<unsigned char>(ch)) || ch == '.') {
      const char* start = text.c_str() + i;
      char* end = nullptr;
      const double value = std::strtod(start, &end);
      if (end == start) {
        if (error) {
          *error = "invalid number in expression";
        }
        return {};
      }
      tokens.push_back({Token::Type::Number, value, {}});
      i += static_cast<size_t>(end - start);
      continue;
    }
    if (IsIdentifierStart(ch)) {
      size_t start = i;
      ++i;
      while (i < text.size() && IsIdentifierChar(text[i])) {
        ++i;
      }
      tokens.push_back({Token::Type::Ident, 0.0, ToLower(text.substr(start, i - start))});
      continue;
    }
    if (ch == '+') {
      tokens.push_back({Token::Type::Plus, 0.0, {}});
      ++i;
      continue;
    }
    if (ch == '-') {
      tokens.push_back({Token::Type::Minus, 0.0, {}});
      ++i;
      continue;
    }
    if (ch == '*') {
      tokens.push_back({Token::Type::Mul, 0.0, {}});
      ++i;
      continue;
    }
    if (ch == '/') {
      tokens.push_back({Token::Type::Div, 0.0, {}});
      ++i;
      continue;
    }
    if (ch == '^') {
      tokens.push_back({Token::Type::Pow, 0.0, {}});
      ++i;
      continue;
    }
    if (ch == '(') {
      tokens.push_back({Token::Type::LParen, 0.0, {}});
      ++i;
      continue;
    }
    if (ch == ')') {
      tokens.push_back({Token::Type::RParen, 0.0, {}});
      ++i;
      continue;
    }
    if (error) {
      *error = std::string("unexpected token: ") + ch;
    }
    return {};
  }
  tokens.push_back({Token::Type::End, 0.0, {}});
  return tokens;
}

class Parser {
 public:
  explicit Parser(std::vector<Token> tokens) : tokens_(std::move(tokens)) {}

  std::unique_ptr<Node> Parse(std::string* error) {
    auto expr = ParseExpression(error);
    if (!expr) {
      return nullptr;
    }
    if (Peek().type != Token::Type::End) {
      if (error) {
        *error = "unexpected trailing tokens";
      }
      return nullptr;
    }
    return expr;
  }

 private:
  const Token& Peek() const { return tokens_[index_]; }

  const Token& Advance() {
    if (index_ < tokens_.size()) {
      ++index_;
    }
    return tokens_[std::min(index_, tokens_.size() - 1)];
  }

  bool Match(Token::Type type) {
    if (Peek().type == type) {
      Advance();
      return true;
    }
    return false;
  }

  bool NextStartsPrimary() const {
    const Token::Type type = Peek().type;
    return type == Token::Type::Number || type == Token::Type::Ident ||
           type == Token::Type::LParen;
  }

  std::unique_ptr<Node> ParseExpression(std::string* error) {
    auto left = ParseTerm(error);
    if (!left) {
      return nullptr;
    }
    while (true) {
      if (Match(Token::Type::Plus)) {
        auto right = ParseTerm(error);
        if (!right) {
          return nullptr;
        }
        auto node = std::make_unique<Node>();
        node->type = Node::Type::Add;
        node->left = std::move(left);
        node->right = std::move(right);
        left = std::move(node);
      } else if (Match(Token::Type::Minus)) {
        auto right = ParseTerm(error);
        if (!right) {
          return nullptr;
        }
        auto node = std::make_unique<Node>();
        node->type = Node::Type::Sub;
        node->left = std::move(left);
        node->right = std::move(right);
        left = std::move(node);
      } else {
        break;
      }
    }
    return left;
  }

  std::unique_ptr<Node> ParseTerm(std::string* error) {
    auto left = ParsePower(error);
    if (!left) {
      return nullptr;
    }
    while (true) {
      if (Match(Token::Type::Mul)) {
        auto right = ParsePower(error);
        if (!right) {
          return nullptr;
        }
        auto node = std::make_unique<Node>();
        node->type = Node::Type::Mul;
        node->left = std::move(left);
        node->right = std::move(right);
        left = std::move(node);
      } else if (Match(Token::Type::Div)) {
        auto right = ParsePower(error);
        if (!right) {
          return nullptr;
        }
        auto node = std::make_unique<Node>();
        node->type = Node::Type::Div;
        node->left = std::move(left);
        node->right = std::move(right);
        left = std::move(node);
      } else if (NextStartsPrimary()) {
        auto right = ParsePower(error);
        if (!right) {
          return nullptr;
        }
        auto node = std::make_unique<Node>();
        node->type = Node::Type::Mul;
        node->left = std::move(left);
        node->right = std::move(right);
        left = std::move(node);
      } else {
        break;
      }
    }
    return left;
  }

  std::unique_ptr<Node> ParsePower(std::string* error) {
    auto left = ParseUnary(error);
    if (!left) {
      return nullptr;
    }
    if (Match(Token::Type::Pow)) {
      auto right = ParsePower(error);
      if (!right) {
        return nullptr;
      }
      auto node = std::make_unique<Node>();
      node->type = Node::Type::Pow;
      node->left = std::move(left);
      node->right = std::move(right);
      return node;
    }
    return left;
  }

  std::unique_ptr<Node> ParseUnary(std::string* error) {
    if (Match(Token::Type::Plus)) {
      return ParseUnary(error);
    }
    if (Match(Token::Type::Minus)) {
      auto child = ParseUnary(error);
      if (!child) {
        return nullptr;
      }
      auto node = std::make_unique<Node>();
      node->type = Node::Type::Neg;
      node->left = std::move(child);
      return node;
    }
    return ParsePrimary(error);
  }

  std::unique_ptr<Node> ParsePrimary(std::string* error) {
    if (Match(Token::Type::Number)) {
      const Token& prev = tokens_[index_ - 1];
      auto node = std::make_unique<Node>();
      node->type = Node::Type::Number;
      node->value = prev.value;
      return node;
    }
    if (Match(Token::Type::Ident)) {
      const Token& prev = tokens_[index_ - 1];
      const std::string name = prev.text;
      if (name == "x") {
        auto node = std::make_unique<Node>();
        node->type = Node::Type::VarX;
        return node;
      }
      if (name == "r") {
        auto node = std::make_unique<Node>();
        node->type = Node::Type::VarX;
        return node;
      }
      if (name == "y") {
        auto node = std::make_unique<Node>();
        node->type = Node::Type::VarY;
        return node;
      }
      if (name == "theta") {
        auto node = std::make_unique<Node>();
        node->type = Node::Type::VarY;
        return node;
      }
      if (name == "z") {
        auto node = std::make_unique<Node>();
        node->type = Node::Type::VarZ;
        return node;
      }
      if (name == "phi") {
        auto node = std::make_unique<Node>();
        node->type = Node::Type::VarZ;
        return node;
      }
      if (name == "t") {
        auto node = std::make_unique<Node>();
        node->type = Node::Type::VarT;
        return node;
      }
      if (name == "pi") {
        auto node = std::make_unique<Node>();
        node->type = Node::Type::Number;
        node->value = kPi;
        return node;
      }
      if (name == "e") {
        auto node = std::make_unique<Node>();
        node->type = Node::Type::Number;
        node->value = std::exp(1.0);
        return node;
      }
      if (!Match(Token::Type::LParen)) {
        if (error) {
          *error = "unknown identifier: " + name;
        }
        return nullptr;
      }
      auto arg = ParseExpression(error);
      if (!arg) {
        return nullptr;
      }
      if (!Match(Token::Type::RParen)) {
        if (error) {
          *error = "missing closing parenthesis for function";
        }
        return nullptr;
      }
      auto node = std::make_unique<Node>();
      node->type = Node::Type::Func;
      node->func = name;
      node->left = std::move(arg);
      return node;
    }
    if (Match(Token::Type::LParen)) {
      auto inside = ParseExpression(error);
      if (!inside) {
        return nullptr;
      }
      if (!Match(Token::Type::RParen)) {
        if (error) {
          *error = "missing closing parenthesis";
        }
        return nullptr;
      }
      return inside;
    }
    if (error) {
      *error = "unexpected token in expression";
    }
    return nullptr;
  }

  std::vector<Token> tokens_;
  size_t index_ = 0;
};
}  // namespace

ExpressionEvaluator::ExpressionEvaluator(std::unique_ptr<Node> root, std::string error)
    : root_(std::move(root)), error_(std::move(error)) {}

ExpressionEvaluator::ExpressionEvaluator(ExpressionEvaluator&&) noexcept = default;
ExpressionEvaluator& ExpressionEvaluator::operator=(ExpressionEvaluator&&) noexcept = default;
ExpressionEvaluator::~ExpressionEvaluator() = default;

ExpressionEvaluator ExpressionEvaluator::ParseLatex(const std::string& text) {
  std::string error;
  std::string infix = ConvertLatexToInfix(text, &error);
  if (!error.empty()) {
    return ExpressionEvaluator(nullptr, error);
  }
  if (infix.empty()) {
    return ExpressionEvaluator(nullptr, "empty expression");
  }

  std::string token_error;
  std::vector<Token> tokens = Tokenize(infix, &token_error);
  if (!token_error.empty()) {
    return ExpressionEvaluator(nullptr, token_error);
  }

  Parser parser(std::move(tokens));
  std::string parse_error;
  std::unique_ptr<Node> root = parser.Parse(&parse_error);
  if (!root) {
    return ExpressionEvaluator(nullptr, parse_error);
  }
  return ExpressionEvaluator(std::move(root), "");
}

bool ExpressionEvaluator::ok() const {
  return error_.empty() && root_ != nullptr;
}

const std::string& ExpressionEvaluator::error() const {
  return error_;
}

double ExpressionEvaluator::Eval(double x, double y) const {
  return Eval(x, y, 0.0, 0.0);
}

double ExpressionEvaluator::Eval(double x, double y, double z) const {
  return Eval(x, y, z, 0.0);
}

double ExpressionEvaluator::Eval(double x, double y, double z, double t) const {
  if (!root_) {
    return 0.0;
  }
  return EvalNode(root_.get(), x, y, z, t);
}

double ExpressionEvaluator::EvalNode(const Node* node, double x, double y, double z, double t) const {
  if (!node) {
    return 0.0;
  }
  switch (node->type) {
    case Node::Type::Number:
      return node->value;
    case Node::Type::VarX:
      return x;
    case Node::Type::VarY:
      return y;
    case Node::Type::VarZ:
      return z;
    case Node::Type::VarT:
      return t;
    case Node::Type::Add:
      return EvalNode(node->left.get(), x, y, z, t) + EvalNode(node->right.get(), x, y, z, t);
    case Node::Type::Sub:
      return EvalNode(node->left.get(), x, y, z, t) - EvalNode(node->right.get(), x, y, z, t);
    case Node::Type::Mul:
      return EvalNode(node->left.get(), x, y, z, t) * EvalNode(node->right.get(), x, y, z, t);
    case Node::Type::Div: {
      const double denom = EvalNode(node->right.get(), x, y, z, t);
      if (std::abs(denom) < 1e-12) {
        return 0.0;
      }
      return EvalNode(node->left.get(), x, y, z, t) / denom;
    }
    case Node::Type::Pow:
      return std::pow(EvalNode(node->left.get(), x, y, z, t),
                      EvalNode(node->right.get(), x, y, z, t));
    case Node::Type::Neg:
      return -EvalNode(node->left.get(), x, y, z, t);
    case Node::Type::Func: {
      const double v = EvalNode(node->left.get(), x, y, z, t);
      if (node->func == "sin") {
        return std::sin(v);
      }
      if (node->func == "cos") {
        return std::cos(v);
      }
      if (node->func == "tan") {
        return std::tan(v);
      }
      if (node->func == "exp") {
        return std::exp(v);
      }
      if (node->func == "log") {
        return std::log(std::max(1e-12, v));
      }
      if (node->func == "sqrt") {
        return std::sqrt(std::max(0.0, v));
      }
      if (node->func == "abs") {
        return std::abs(v);
      }
      return 0.0;
    }
    default:
      return 0.0;
  }
}
