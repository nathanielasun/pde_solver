#include "log_panel.h"
#include "imgui.h"
#include "imgui_stdlib.h"
#include "styles/ui_style.h"
#include <algorithm>
#include <cctype>

LogPanelComponent::LogPanelComponent() {
  visible_ = true;
}

void LogPanelComponent::SetService(LogService* service) {
  service_ = service;
}

void LogPanelComponent::SetLogs(const std::vector<std::string>* logs, std::mutex* mutex) {
  logs_ = logs;
  logs_mutex_ = mutex;
}

void LogPanelComponent::SetFilter(bool show_errors, bool show_warnings, bool show_info) {
  show_errors_ = show_errors;
  show_warnings_ = show_warnings;
  show_info_ = show_info;
}

void LogPanelComponent::SetSearchText(const std::string& text) {
  search_text_ = text;
}

bool LogPanelComponent::ShouldShowLog(const std::string& log) const {
  // Check type filter
  bool is_error = log.find("error") != std::string::npos || 
                  log.find("Error") != std::string::npos ||
                  log.find("ERROR") != std::string::npos;
  bool is_warning = log.find("warning") != std::string::npos ||
                    log.find("Warning") != std::string::npos ||
                    log.find("WARNING") != std::string::npos;
  
  if (is_error && !show_errors_) {
    return false;
  }
  if (is_warning && !show_warnings_) {
    return false;
  }
  if (!is_error && !is_warning && !show_info_) {
    return false;
  }
  
  // Check search text
  if (!search_text_.empty()) {
    std::string log_lower = log;
    std::string search_lower = search_text_;
    std::transform(log_lower.begin(), log_lower.end(), log_lower.begin(), ::tolower);
    std::transform(search_lower.begin(), search_lower.end(), search_lower.begin(), ::tolower);
    if (log_lower.find(search_lower) == std::string::npos) {
      return false;
    }
  }
  
  return true;
}

void LogPanelComponent::Render() {
  if (!IsVisible()) {
    return;
  }
  
  // Filter controls - row 1: checkboxes
  ImGui::Text("Filter:");
  ImGui::SameLine();
  ImGui::Checkbox("Errors", &show_errors_);
  ImGui::SameLine();
  ImGui::Checkbox("Warnings", &show_warnings_);
  ImGui::SameLine();
  ImGui::Checkbox("Info", &show_info_);

  // Filter controls - row 2: search
  ImGui::SetNextItemWidth(-1);  // Use full available width
  ImGui::InputTextWithHint("##log_search", "Search logs...", &search_text_);

  ImGui::Spacing();
  ImGui::BeginChild("LogPanel", ImVec2(-1, -1), true);  // Use remaining height
  
  if (service_) {
    LogService::FilterOptions opts;
    opts.show_errors = show_errors_;
    opts.show_warnings = show_warnings_;
    opts.show_info = show_info_;
    opts.search_text = search_text_;
    const std::vector<std::string> filtered = service_->GetFiltered(opts);
    for (const auto& line : filtered) {
      if (line.find("error") != std::string::npos ||
          line.find("Error") != std::string::npos ||
          line.find("ERROR") != std::string::npos) {
        ImGui::TextColored(ImVec4(1.0f, 0.5f, 0.5f, 1.0f), "%s", line.c_str());
      } else if (line.find("warning") != std::string::npos ||
                 line.find("Warning") != std::string::npos ||
                 line.find("WARNING") != std::string::npos) {
        ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.3f, 1.0f), "%s", line.c_str());
      } else {
        ImGui::TextUnformatted(line.c_str());
      }
    }
  } else if (logs_ && logs_mutex_) {
    std::lock_guard<std::mutex> lock(*logs_mutex_);
    for (const auto& line : *logs_) {
      if (!ShouldShowLog(line)) {
        continue;
      }
      if (line.find("error") != std::string::npos ||
          line.find("Error") != std::string::npos ||
          line.find("ERROR") != std::string::npos) {
        ImGui::TextColored(ImVec4(1.0f, 0.5f, 0.5f, 1.0f), "%s", line.c_str());
      } else if (line.find("warning") != std::string::npos ||
                 line.find("Warning") != std::string::npos ||
                 line.find("WARNING") != std::string::npos) {
        ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.3f, 1.0f), "%s", line.c_str());
      } else {
        ImGui::TextUnformatted(line.c_str());
      }
    }
  }
  
  ImGui::EndChild();
}
