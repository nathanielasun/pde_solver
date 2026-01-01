#include "string_utils.h"

#include <algorithm>
#include <cctype>
#include <sstream>
#include <string>
#include <vector>

std::string Trim(const std::string& input) {
  size_t start = 0;
  while (start < input.size() && std::isspace(static_cast<unsigned char>(input[start])) != 0) {
    ++start;
  }
  size_t end = input.size();
  while (end > start && std::isspace(static_cast<unsigned char>(input[end - 1])) != 0) {
    --end;
  }
  return input.substr(start, end - start);
}

std::string ToLower(const std::string& input) {
  std::string out = input;
  std::transform(out.begin(), out.end(), out.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  return out;
}

std::vector<std::string> Split(const std::string& input, char delimiter) {
  std::vector<std::string> parts;
  std::stringstream ss(input);
  std::string item;
  while (std::getline(ss, item, delimiter)) {
    parts.push_back(item);
  }
  return parts;
}

void ReplaceAll(std::string* text, const std::string& from, const std::string& to) {
  if (!text || from.empty()) {
    return;
  }
  size_t start = 0;
  while ((start = text->find(from, start)) != std::string::npos) {
    text->replace(start, from.size(), to);
    start += to.size();
  }
}

