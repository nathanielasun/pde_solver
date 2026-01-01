#include "app_state.h"
#include <mutex>
#include <algorithm>

void AddLog(SharedState& state, std::mutex& state_mutex, const std::string& message) {
  std::lock_guard<std::mutex> lock(state_mutex);
  state.logs.push_back(message);
  if (static_cast<int>(state.logs.size()) > 300) {  // kLogLimit
    state.logs.erase(state.logs.begin());
  }
  state.log_service.Append("log", message);
}

