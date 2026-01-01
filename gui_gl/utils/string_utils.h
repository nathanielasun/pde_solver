#ifndef STRING_UTILS_H
#define STRING_UTILS_H

#include <string>
#include <vector>

// String utility functions
std::string Trim(const std::string& input);
std::string ToLower(const std::string& input);
std::vector<std::string> Split(const std::string& input, char delimiter);
void ReplaceAll(std::string* text, const std::string& from, const std::string& to);

#endif // STRING_UTILS_H

