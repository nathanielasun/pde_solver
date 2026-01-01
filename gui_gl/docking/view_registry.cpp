#include "view_registry.h"
#include "imgui.h"
#include <algorithm>
#include <set>

ViewRegistry& ViewRegistry::Instance() {
  static ViewRegistry instance;
  return instance;
}

void ViewRegistry::Register(ViewType type, ViewTypeInfo info) {
  registry_[type] = std::move(info);
}

bool ViewRegistry::IsRegistered(ViewType type) const {
  return registry_.find(type) != registry_.end();
}

const ViewTypeInfo* ViewRegistry::GetInfo(ViewType type) const {
  auto it = registry_.find(type);
  return it != registry_.end() ? &it->second : nullptr;
}

std::vector<ViewType> ViewRegistry::GetAllTypes() const {
  std::vector<ViewType> types;
  types.reserve(registry_.size());
  for (const auto& [type, info] : registry_) {
    types.push_back(type);
  }
  // Sort by display name for consistent ordering
  std::sort(types.begin(), types.end(), [this](ViewType a, ViewType b) {
    const auto* info_a = GetInfo(a);
    const auto* info_b = GetInfo(b);
    if (info_a && info_b) {
      return info_a->display_name < info_b->display_name;
    }
    return static_cast<int>(a) < static_cast<int>(b);
  });
  return types;
}

std::vector<ViewType> ViewRegistry::GetTypesByCategory(const std::string& category) const {
  std::vector<ViewType> types;
  for (const auto& [type, info] : registry_) {
    if (info.category == category) {
      types.push_back(type);
    }
  }
  // Sort by display name
  std::sort(types.begin(), types.end(), [this](ViewType a, ViewType b) {
    const auto* info_a = GetInfo(a);
    const auto* info_b = GetInfo(b);
    if (info_a && info_b) {
      return info_a->display_name < info_b->display_name;
    }
    return static_cast<int>(a) < static_cast<int>(b);
  });
  return types;
}

std::vector<std::string> ViewRegistry::GetCategories() const {
  std::set<std::string> categories;
  for (const auto& [type, info] : registry_) {
    if (!info.category.empty()) {
      categories.insert(info.category);
    }
  }
  return std::vector<std::string>(categories.begin(), categories.end());
}

void ViewRegistry::Render(ViewType type, ViewRenderContext& ctx) {
  auto it = registry_.find(type);
  if (it != registry_.end() && it->second.renderer) {
    it->second.renderer(ctx);
  } else {
    // Fallback for unregistered views
    ImGui::TextDisabled("View type '%s' not registered",
                        GetViewTypeDisplayName(type).c_str());
  }
}

void ViewRegistry::Clear() {
  registry_.clear();
}
