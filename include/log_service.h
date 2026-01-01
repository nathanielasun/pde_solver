// Thread-safe logging service shared by Qt and ImGui frontends.
#ifndef LOG_SERVICE_H
#define LOG_SERVICE_H

#include <mutex>
#include <string>
#include <vector>

class LogService {
 public:
  struct FilterOptions {
    bool show_errors = true;
    bool show_warnings = true;
    bool show_info = true;
    std::string search_text;
  };

  LogService() = default;
  LogService(const LogService& other);
  LogService& operator=(const LogService& other);
  LogService(LogService&&) = delete;
  LogService& operator=(LogService&&) = delete;
  ~LogService() = default;

  void Append(const std::string& category, const std::string& message);
  void Clear();

  std::vector<std::string> GetFiltered(const FilterOptions& opts) const;

  std::mutex& GetMutex() { return mutex_; }
  const std::vector<std::string>& GetLogs() const { return logs_; }

 private:
  std::vector<std::string> logs_;
  mutable std::mutex mutex_;
  static constexpr int kMaxLogs = 2000;
};

#endif  // LOG_SERVICE_H

