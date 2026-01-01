#ifndef LOG_PANEL_H
#define LOG_PANEL_H

#include "ui_component.h"
#include "log_service.h"
#include <vector>
#include <string>
#include <mutex>
#include <functional>

// Log panel component for displaying solver logs
class LogPanelComponent : public UIComponent {
 public:
  LogPanelComponent();
  
  void Render() override;
  std::string GetName() const override { return "LogPanel"; }
  
  // Set log data source (preferred: LogService)
  void SetService(LogService* service);
  void SetLogs(const std::vector<std::string>* logs, std::mutex* mutex);
  
  // Set filter options
  void SetFilter(bool show_errors, bool show_warnings, bool show_info);
  void SetSearchText(const std::string& text);
  
  // Get filter state
  bool GetShowErrors() const { return show_errors_; }
  bool GetShowWarnings() const { return show_warnings_; }
  bool GetShowInfo() const { return show_info_; }
  std::string GetSearchText() const { return search_text_; }

 private:
  LogService* service_ = nullptr;
  const std::vector<std::string>* logs_ = nullptr;
  std::mutex* logs_mutex_ = nullptr;
  bool show_errors_ = true;
  bool show_warnings_ = true;
  bool show_info_ = true;
  std::string search_text_;
  static constexpr int kLogLimit = 300;
  
  bool ShouldShowLog(const std::string& log) const;
};

#endif  // LOG_PANEL_H

