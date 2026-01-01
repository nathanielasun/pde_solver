#include "log_service.h"

#include <algorithm>
#include <cctype>

namespace {

bool ContainsCaseInsensitive(const std::string& text, const std::string& needle) {
  if (needle.empty()) return true;
  auto to_lower = [](const std::string& in) {
    std::string out(in.size(), '\0');
    std::transform(in.begin(), in.end(), out.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return out;
  };
  const std::string hay_lower = to_lower(text);
  const std::string needle_lower = to_lower(needle);
  return hay_lower.find(needle_lower) != std::string::npos;
}

bool IsError(const std::string& line) {
  return ContainsCaseInsensitive(line, "error");
}

bool IsWarning(const std::string& line) {
  return ContainsCaseInsensitive(line, "warning");
}

}  // namespace

LogService::LogService(const LogService& other) {
  std::lock_guard<std::mutex> lock(other.mutex_);
  logs_ = other.logs_;
}

LogService& LogService::operator=(const LogService& other) {
  if (this == &other) {
    return *this;
  }
  std::scoped_lock lock(mutex_, other.mutex_);
  logs_ = other.logs_;
  return *this;
}

void LogService::Append(const std::string& category, const std::string& message) {
  const std::string trimmed = message;
  if (trimmed.empty()) {
    return;
  }
  std::lock_guard<std::mutex> lock(mutex_);
  logs_.push_back("[" + category + "] " + message);
  if (logs_.size() > static_cast<size_t>(kMaxLogs)) {
    const size_t start = logs_.size() - static_cast<size_t>(kMaxLogs);
    logs_.erase(logs_.begin(), logs_.begin() + static_cast<std::ptrdiff_t>(start));
  }
}

void LogService::Clear() {
  std::lock_guard<std::mutex> lock(mutex_);
  logs_.clear();
}

std::vector<std::string> LogService::GetFiltered(const FilterOptions& opts) const {
  std::vector<std::string> out;
  std::lock_guard<std::mutex> lock(mutex_);
  for (const auto& line : logs_) {
    const bool error_line = IsError(line);
    const bool warning_line = IsWarning(line);

    if (error_line && !opts.show_errors) continue;
    if (warning_line && !opts.show_warnings) continue;
    if (!error_line && !warning_line && !opts.show_info) continue;

    if (!opts.search_text.empty() && !ContainsCaseInsensitive(line, opts.search_text)) {
      continue;
    }
    out.push_back(line);
  }
  return out;
}

