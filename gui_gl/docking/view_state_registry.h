#ifndef VIEW_STATE_REGISTRY_H
#define VIEW_STATE_REGISTRY_H

#include <any>
#include <map>
#include <string>
#include <typeindex>
#include <mutex>

// Registry for per-instance view state
// Allows views to store and retrieve state by instance ID
class ViewStateRegistry {
public:
  // Singleton access
  static ViewStateRegistry& Instance();

  // Get or create state for a view instance
  // Creates default-constructed state if not found
  template<typename T>
  T& Get(const std::string& instance_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto key = std::make_pair(std::type_index(typeid(T)), instance_id);

    auto it = states_.find(key);
    if (it == states_.end()) {
      it = states_.emplace(key, std::make_any<T>()).first;
    }

    return std::any_cast<T&>(it->second);
  }

  // Check if state exists for an instance
  template<typename T>
  bool Has(const std::string& instance_id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto key = std::make_pair(std::type_index(typeid(T)), instance_id);
    return states_.find(key) != states_.end();
  }

  // Remove state for a specific instance (all types)
  void Remove(const std::string& instance_id);

  // Remove state for a specific instance and type
  template<typename T>
  void Remove(const std::string& instance_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto key = std::make_pair(std::type_index(typeid(T)), instance_id);
    states_.erase(key);
  }

  // Clear all state
  void Clear();

  // Get count of stored states
  size_t Size() const;

private:
  ViewStateRegistry() = default;
  ViewStateRegistry(const ViewStateRegistry&) = delete;
  ViewStateRegistry& operator=(const ViewStateRegistry&) = delete;

  using StateKey = std::pair<std::type_index, std::string>;
  std::map<StateKey, std::any> states_;
  mutable std::mutex mutex_;
};

// Convenience function to get state
template<typename T>
T& GetViewState(const std::string& instance_id) {
  return ViewStateRegistry::Instance().Get<T>(instance_id);
}

#endif // VIEW_STATE_REGISTRY_H
