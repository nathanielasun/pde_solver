#ifndef STRING_UTILS_H
#define STRING_UTILS_H

#include <string>

namespace pde {

// Trim leading and trailing whitespace.
std::string Trim(const std::string& text);

// Lowercase a string (ASCII-safe).
std::string ToLower(const std::string& text);

}  // namespace pde

#endif

