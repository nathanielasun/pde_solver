#include "input_parse.h"

#include <cctype>
#include <cmath>
#include <cstdlib>
#include <sstream>

#include "expression_eval.h"

namespace {
bool ParseDoubles(const std::string& text, int expected, std::vector<double>* out) {
  out->clear();
  std::stringstream ss(text);
  std::string item;
  while (std::getline(ss, item, ',')) {
    if (item.empty()) {
      return false;
    }
    char* end = nullptr;
    double value = std::strtod(item.c_str(), &end);
    if (!end || *end != '\0') {
      return false;
    }
    out->push_back(value);
  }
  return static_cast<int>(out->size()) == expected;
}

bool ParseInts(const std::string& text, int expected, std::vector<int>* out) {
  out->clear();
  std::stringstream ss(text);
  std::string item;
  while (std::getline(ss, item, ',')) {
    if (item.empty()) {
      return false;
    }
    char* end = nullptr;
    long value = std::strtol(item.c_str(), &end, 10);
    if (!end || *end != '\0') {
      return false;
    }
    out->push_back(static_cast<int>(value));
  }
  return static_cast<int>(out->size()) == expected;
}

bool ParseBoundaryKind(const std::string& text, BCKind* kind) {
  if (text == "dirichlet") {
    *kind = BCKind::Dirichlet;
    return true;
  }
  if (text == "neumann") {
    *kind = BCKind::Neumann;
    return true;
  }
  if (text == "robin") {
    *kind = BCKind::Robin;
    return true;
  }
  return false;
}

struct LinearExpr {
  double constant = 0.0;
  double x = 0.0;
  double y = 0.0;
  double z = 0.0;
};

bool IsScalar(const LinearExpr& expr) {
  return std::abs(expr.x) < 1e-12 && std::abs(expr.y) < 1e-12 &&
         std::abs(expr.z) < 1e-12;
}

LinearExpr Add(const LinearExpr& a, const LinearExpr& b) {
  return {a.constant + b.constant, a.x + b.x, a.y + b.y, a.z + b.z};
}

LinearExpr Sub(const LinearExpr& a, const LinearExpr& b) {
  return {a.constant - b.constant, a.x - b.x, a.y - b.y, a.z - b.z};
}

bool Mul(const LinearExpr& a, const LinearExpr& b, LinearExpr* out, std::string* error) {
  if (IsScalar(a)) {
    *out = {b.constant * a.constant, b.x * a.constant, b.y * a.constant, b.z * a.constant};
    return true;
  }
  if (IsScalar(b)) {
    *out = {a.constant * b.constant, a.x * b.constant, a.y * b.constant, a.z * b.constant};
    return true;
  }
  if (error) {
    *error = "nonlinear term: only scalar * (x, y, z, r, theta, or phi) supported";
  }
  return false;
}

bool Div(const LinearExpr& a, const LinearExpr& b, LinearExpr* out, std::string* error) {
  if (!IsScalar(b) || std::abs(b.constant) < 1e-12) {
    if (error) {
      *error = "division requires a nonzero scalar denominator";
    }
    return false;
  }
  const double inv = 1.0 / b.constant;
  *out = {a.constant * inv, a.x * inv, a.y * inv, a.z * inv};
  return true;
}

bool ParseBraceGroup(const std::string& input, size_t* index, std::string* group,
                     std::string* error) {
  if (*index >= input.size() || input[*index] != '{') {
    if (error) {
      *error = "expected '{' after \\frac";
    }
    return false;
  }
  size_t i = *index + 1;
  int depth = 0;
  for (; i < input.size(); ++i) {
    if (input[i] == '{') {
      ++depth;
    } else if (input[i] == '}') {
      if (depth == 0) {
        *group = input.substr(*index + 1, i - (*index + 1));
        *index = i + 1;
        return true;
      }
      --depth;
    }
  }
  if (error) {
    *error = "unterminated '{' in \\frac";
  }
  return false;
}

std::string ConvertLatexToInfix(const std::string& input, std::string* error);

std::string ConvertLatexToInfix(const std::string& input, std::string* error) {
  std::string out;
  out.reserve(input.size());
  size_t i = 0;
  while (i < input.size()) {
    const char ch = input[i];
    if (std::isspace(static_cast<unsigned char>(ch))) {
      ++i;
      continue;
    }
    if (ch == '$') {
      ++i;
      continue;
    }
    if (ch == '\\') {
      if (input.compare(i, 5, "\\frac") == 0) {
        i += 5;
        std::string num;
        std::string den;
        if (!ParseBraceGroup(input, &i, &num, error)) {
          return "";
        }
        if (!ParseBraceGroup(input, &i, &den, error)) {
          return "";
        }
        std::string num_conv = ConvertLatexToInfix(num, error);
        if (error && !error->empty()) {
          return "";
        }
        std::string den_conv = ConvertLatexToInfix(den, error);
        if (error && !error->empty()) {
          return "";
        }
        out.append("(").append(num_conv).append(")/(").append(den_conv).append(")");
        continue;
      }
      if (input.compare(i, 5, "\\cdot") == 0) {
        out.push_back('*');
        i += 5;
        continue;
      }
      if (input.compare(i, 6, "\\times") == 0) {
        out.push_back('*');
        i += 6;
        continue;
      }
      if (input.compare(i, 6, "\\theta") == 0) {
        out.append("theta");
        i += 6;
        continue;
      }
      if (input.compare(i, 4, "\\phi") == 0) {
        out.append("phi");
        i += 4;
        continue;
      }
      if (input.compare(i, 8, "\\vartheta") == 0) {
        out.append("theta");
        i += 8;
        continue;
      }
      if (input.compare(i, 7, "\\varphi") == 0) {
        out.append("phi");
        i += 7;
        continue;
      }
      if (input.compare(i, 5, "\\left") == 0) {
        i += 5;
        continue;
      }
      if (input.compare(i, 6, "\\right") == 0) {
        i += 6;
        continue;
      }
      if (input.compare(i, 2, "\\,") == 0) {
        i += 2;
        continue;
      }
      size_t end = i + 1;
      while (end < input.size() && std::isalpha(static_cast<unsigned char>(input[end]))) {
        ++end;
      }
      if (error) {
        *error = "unsupported latex command: " + input.substr(i, end - i);
      }
      return "";
    }
    if (ch == '{') {
      out.push_back('(');
      ++i;
      continue;
    }
    if (ch == '}') {
      out.push_back(')');
      ++i;
      continue;
    }
    out.push_back(ch);
    ++i;
  }
  return out;
}

enum class TokenType {
  Number,
  VarX,
  VarY,
  VarZ,
  Plus,
  Minus,
  Mul,
  Div,
  LParen,
  RParen,
  End,
};

struct Token {
  TokenType type = TokenType::End;
  double value = 0.0;
};

std::vector<Token> Tokenize(const std::string& text, std::string* error) {
  std::vector<Token> tokens;
  size_t i = 0;
  while (i < text.size()) {
    const char ch = text[i];
    if (std::isspace(static_cast<unsigned char>(ch))) {
      ++i;
      continue;
    }
    if (std::isdigit(static_cast<unsigned char>(ch)) || ch == '.') {
      const char* start = text.c_str() + i;
      char* end = nullptr;
      double value = std::strtod(start, &end);
      if (end == start) {
        if (error) {
          *error = "invalid number in expression";
        }
        return {};
      }
      i += static_cast<size_t>(end - start);
      tokens.push_back({TokenType::Number, value});
      continue;
    }
    if (std::isalpha(static_cast<unsigned char>(ch))) {
      size_t start = i;
      ++i;
      while (i < text.size() && std::isalpha(static_cast<unsigned char>(text[i]))) {
        ++i;
      }
      std::string ident = text.substr(start, i - start);
      for (char& c : ident) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
      }
      if (ident == "x" || ident == "r") {
        tokens.push_back({TokenType::VarX, 0.0});
        continue;
      }
      if (ident == "y" || ident == "theta") {
        tokens.push_back({TokenType::VarY, 0.0});
        continue;
      }
      if (ident == "z" || ident == "phi") {
        tokens.push_back({TokenType::VarZ, 0.0});
        continue;
      }
      if (error) {
        *error = "unknown identifier: " + ident;
      }
      return {};
    }
    if (ch == '+') {
      tokens.push_back({TokenType::Plus, 0.0});
      ++i;
      continue;
    }
    if (ch == '-') {
      tokens.push_back({TokenType::Minus, 0.0});
      ++i;
      continue;
    }
    if (ch == '*') {
      tokens.push_back({TokenType::Mul, 0.0});
      ++i;
      continue;
    }
    if (ch == '/') {
      tokens.push_back({TokenType::Div, 0.0});
      ++i;
      continue;
    }
    if (ch == '(') {
      tokens.push_back({TokenType::LParen, 0.0});
      ++i;
      continue;
    }
    if (ch == ')') {
      tokens.push_back({TokenType::RParen, 0.0});
      ++i;
      continue;
    }
    if (error) {
      *error = std::string("unexpected token: ") + ch;
    }
    return {};
  }
  tokens.push_back({TokenType::End, 0.0});
  return tokens;
}

class ExprParser {
 public:
  explicit ExprParser(const std::vector<Token>& tokens) : tokens_(tokens) {}

  bool Parse(LinearExpr* out, std::string* error) {
    if (!ParseExpression(out, error)) {
      return false;
    }
    if (Peek().type != TokenType::End) {
      if (error) {
        *error = "unexpected trailing tokens in expression";
      }
      return false;
    }
    return true;
  }

 private:
  const std::vector<Token>& tokens_;
  size_t index_ = 0;

  const Token& Peek() const {
    return tokens_[index_];
  }

  const Token& Advance() {
    if (index_ < tokens_.size()) {
      ++index_;
    }
    return tokens_[std::min(index_, tokens_.size() - 1)];
  }

  bool Match(TokenType type) {
    if (Peek().type == type) {
      Advance();
      return true;
    }
    return false;
  }

  bool NextStartsFactor() const {
    const TokenType type = Peek().type;
    return type == TokenType::Number || type == TokenType::VarX || type == TokenType::VarY ||
           type == TokenType::VarZ || type == TokenType::LParen;
  }

  bool ParseExpression(LinearExpr* out, std::string* error) {
    LinearExpr left;
    if (!ParseTerm(&left, error)) {
      return false;
    }
    while (true) {
      if (Match(TokenType::Plus)) {
        LinearExpr right;
        if (!ParseTerm(&right, error)) {
          return false;
        }
        left = Add(left, right);
      } else if (Match(TokenType::Minus)) {
        LinearExpr right;
        if (!ParseTerm(&right, error)) {
          return false;
        }
        left = Sub(left, right);
      } else {
        break;
      }
    }
    *out = left;
    return true;
  }

  bool ParseTerm(LinearExpr* out, std::string* error) {
    LinearExpr left;
    if (!ParseFactor(&left, error)) {
      return false;
    }
    while (true) {
      if (Match(TokenType::Mul)) {
        LinearExpr right;
        if (!ParseFactor(&right, error)) {
          return false;
        }
        LinearExpr result;
        if (!Mul(left, right, &result, error)) {
          return false;
        }
        left = result;
      } else if (Match(TokenType::Div)) {
        LinearExpr right;
        if (!ParseFactor(&right, error)) {
          return false;
        }
        LinearExpr result;
        if (!Div(left, right, &result, error)) {
          return false;
        }
        left = result;
      } else if (NextStartsFactor()) {
        LinearExpr right;
        if (!ParseFactor(&right, error)) {
          return false;
        }
        LinearExpr result;
        if (!Mul(left, right, &result, error)) {
          return false;
        }
        left = result;
      } else {
        break;
      }
    }
    *out = left;
    return true;
  }

  bool ParseFactor(LinearExpr* out, std::string* error) {
    if (Match(TokenType::Plus)) {
      return ParseFactor(out, error);
    }
    if (Match(TokenType::Minus)) {
      LinearExpr value;
      if (!ParseFactor(&value, error)) {
        return false;
      }
      value.constant = -value.constant;
      value.x = -value.x;
      value.y = -value.y;
      value.z = -value.z;
      *out = value;
      return true;
    }
    if (Match(TokenType::Number)) {
      const Token& prev = tokens_[index_ - 1];
      *out = {prev.value, 0.0, 0.0, 0.0};
      return true;
    }
    if (Match(TokenType::VarX)) {
      *out = {0.0, 1.0, 0.0, 0.0};
      return true;
    }
    if (Match(TokenType::VarY)) {
      *out = {0.0, 0.0, 1.0, 0.0};
      return true;
    }
    if (Match(TokenType::VarZ)) {
      *out = {0.0, 0.0, 0.0, 1.0};
      return true;
    }
    if (Match(TokenType::LParen)) {
      LinearExpr inside;
      if (!ParseExpression(&inside, error)) {
        return false;
      }
      if (!Match(TokenType::RParen)) {
        if (error) {
          *error = "missing closing parenthesis";
        }
        return false;
      }
      *out = inside;
      return true;
    }
    if (error) {
      *error = "unexpected token in expression";
    }
    return false;
  }
};

bool ParseLinearExpr(const std::string& text, BoundaryCondition::Expression* expr,
                     std::string* error) {
  *expr = {};
  if (text.empty()) {
    if (error) {
      *error = "empty expression";
    }
    return false;
  }

  std::string latex_error;
  std::string infix = ConvertLatexToInfix(text, &latex_error);
  if (!latex_error.empty()) {
    if (error) {
      *error = latex_error;
    }
    return false;
  }
  if (infix.empty()) {
    if (error) {
      *error = "empty expression";
    }
    return false;
  }

  std::string token_error;
  std::vector<Token> tokens = Tokenize(infix, &token_error);
  if (!token_error.empty()) {
    if (error) {
      *error = token_error;
    }
    return false;
  }

  LinearExpr result;
  ExprParser parser(tokens);
  std::string parse_error;
  if (!parser.Parse(&result, &parse_error)) {
    if (error) {
      *error = parse_error;
    }
    return false;
  }

  expr->constant = result.constant;
  expr->x = result.x;
  expr->y = result.y;
  expr->z = result.z;
  return true;
}
}

ParseResult ApplyBoundarySpec(const std::string& spec, BoundarySet* bc) {
  std::stringstream ss(spec);
  std::string entry;
  while (std::getline(ss, entry, ';')) {
    if (entry.empty()) {
      continue;
    }
    std::stringstream es(entry);
    std::string side;
    std::string kind_str;
    std::string value_str;
    if (!std::getline(es, side, ':') || !std::getline(es, kind_str, ':') ||
        !std::getline(es, value_str, ':')) {
      return {false, "invalid boundary entry: " + entry};
    }

    BCKind kind;
    if (!ParseBoundaryKind(kind_str, &kind)) {
      return {false, "unknown boundary kind: " + kind_str};
    }

    BoundaryCondition condition;
    condition.kind = kind;

    if (kind == BCKind::Robin) {
      bool has_alpha = false;
      bool has_beta = false;
      bool has_gamma = false;
      std::stringstream params(value_str);
      std::string param;
      while (std::getline(params, param, ',')) {
        if (param.empty()) {
          continue;
        }
        const size_t eq = param.find('=');
        if (eq == std::string::npos) {
          return {false, "invalid robin param: " + param};
        }
        std::string key = param.substr(0, eq);
        std::string val = param.substr(eq + 1);
        BoundaryCondition::Expression expr;
        std::string expr_error;
        if (!ParseLinearExpr(val, &expr, &expr_error)) {
          return {false, "invalid robin expression: " + expr_error};
        }
        if (key == "alpha") {
          condition.alpha = expr;
          has_alpha = true;
        } else if (key == "beta") {
          condition.beta = expr;
          has_beta = true;
        } else if (key == "gamma") {
          condition.gamma = expr;
          has_gamma = true;
        } else {
          return {false, "unknown robin parameter: " + key};
        }
      }
      if (!has_alpha || !has_beta || !has_gamma) {
        return {false, "robin requires alpha,beta,gamma"};
      }
    } else {
      BoundaryCondition::Expression expr;
      std::string expr_error;
      if (!ParseLinearExpr(value_str, &expr, &expr_error)) {
        return {false, "invalid boundary expression: " + expr_error};
      }
      condition.value = expr;
    }

    if (side == "left") {
      bc->left = condition;
    } else if (side == "right") {
      bc->right = condition;
    } else if (side == "bottom") {
      bc->bottom = condition;
    } else if (side == "top") {
      bc->top = condition;
    } else if (side == "front") {
      bc->front = condition;
    } else if (side == "back") {
      bc->back = condition;
    } else {
      return {false, "unknown boundary side: " + side};
    }
  }
  return {true, ""};
}
