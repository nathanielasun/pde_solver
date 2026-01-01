#include "view_state_registry.h"
#include <algorithm>

ViewStateRegistry& ViewStateRegistry::Instance() {
  static ViewStateRegistry instance;
  return instance;
}

void ViewStateRegistry::Remove(const std::string& instance_id) {
  std::lock_guard<std::mutex> lock(mutex_);

  // Remove all entries with matching instance_id
  for (auto it = states_.begin(); it != states_.end(); ) {
    if (it->first.second == instance_id) {
      it = states_.erase(it);
    } else {
      ++it;
    }
  }
}

void ViewStateRegistry::Clear() {
  std::lock_guard<std::mutex> lock(mutex_);
  states_.clear();
}

size_t ViewStateRegistry::Size() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return states_.size();
}
