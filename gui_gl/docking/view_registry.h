#ifndef VIEW_REGISTRY_H
#define VIEW_REGISTRY_H

#include "view_types.h"
#include "../core/app_services.h"
#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

// Context passed to view renderers
struct ViewRenderContext {
  AppServices& services;
  const std::string& instance_id;
  float available_width;
  float available_height;
  bool is_focused;
};

// Function signature for view renderers
using ViewRenderer = std::function<void(ViewRenderContext&)>;

// Information about a view type
struct ViewTypeInfo {
  std::string display_name;
  std::string icon;           // Unicode icon
  std::string category;       // For grouping in dropdown
  ViewRenderer renderer;
  bool allow_multiple;        // Can have multiple instances?
};

// Registry for all view types and their renderers
class ViewRegistry {
public:
  // Singleton access
  static ViewRegistry& Instance();

  // Register a view type with its renderer and metadata
  void Register(ViewType type, ViewTypeInfo info);

  // Check if a view type is registered
  bool IsRegistered(ViewType type) const;

  // Get info for a view type (returns nullptr if not registered)
  const ViewTypeInfo* GetInfo(ViewType type) const;

  // Get all registered view types
  std::vector<ViewType> GetAllTypes() const;

  // Get view types by category
  std::vector<ViewType> GetTypesByCategory(const std::string& category) const;

  // Get all unique categories
  std::vector<std::string> GetCategories() const;

  // Render a view of the given type
  void Render(ViewType type, ViewRenderContext& ctx);

  // Clear all registrations (for testing)
  void Clear();

private:
  ViewRegistry() = default;
  ViewRegistry(const ViewRegistry&) = delete;
  ViewRegistry& operator=(const ViewRegistry&) = delete;

  std::unordered_map<ViewType, ViewTypeInfo> registry_;
};

// Macro for easy view registration in static initializers
#define REGISTER_VIEW(type, display, icon_char, cat, allow_multi, render_fn) \
  namespace { \
    struct ViewRegistrar_##type { \
      ViewRegistrar_##type() { \
        ViewRegistry::Instance().Register(ViewType::type, { \
          display, icon_char, cat, render_fn, allow_multi \
        }); \
      } \
    }; \
    static ViewRegistrar_##type g_registrar_##type; \
  }

#endif // VIEW_REGISTRY_H
