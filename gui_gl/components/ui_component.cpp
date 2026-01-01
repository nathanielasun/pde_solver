#include "ui_component.h"
#include <algorithm>

void ComponentManager::RegisterComponent(std::unique_ptr<UIComponent> component) {
  if (!component) {
    return;
  }
  std::string name = component->GetName();
  components_[name] = std::move(component);
}

void ComponentManager::RenderAll() {
  for (auto& [name, component] : components_) {
    if (component && component->IsVisible()) {
      component->Render();
    }
  }
}

UIComponent* ComponentManager::GetComponent(const std::string& name) {
  auto it = components_.find(name);
  if (it != components_.end()) {
    return it->second.get();
  }
  return nullptr;
}

std::vector<std::string> ComponentManager::GetComponentNames() const {
  std::vector<std::string> names;
  names.reserve(components_.size());
  for (const auto& [name, component] : components_) {
    names.push_back(name);
  }
  return names;
}

void ComponentManager::RemoveComponent(const std::string& name) {
  components_.erase(name);
}

void ComponentManager::Clear() {
  components_.clear();
}

