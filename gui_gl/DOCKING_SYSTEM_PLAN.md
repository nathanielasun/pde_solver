# Splittable Window & Docking System Implementation Plan

## Overview

This document outlines the implementation plan for transforming the current fixed-layout UI into a VS Code/Blender-style splittable, draggable, and customizable window system.

---

## Current Architecture Summary

### Existing Structure
```
Application
├── Tab Bar (Main | Inspect | Preferences)
├── Per-Tab Layout:
│   ├── Left Panel (collapsible, fixed panels in order)
│   ├── Splitter (drag to resize)
│   └── Right Panel (visualization + timeline)
└── Panel Registry (maps panel IDs to render functions)
```

### Current Limitations
1. **Fixed panel positions** - Panels only reorder within their container
2. **No splitting** - Cannot split a view into multiple sub-views
3. **No panel type switching** - Cannot change what a panel displays
4. **Tab-bound panels** - Panels are locked to their assigned tabs
5. **Single visualization** - Only one viewer instance

---

## Target Architecture

### Design Goals (VS Code / Blender Style)
1. **Splittable containers** - Any view can split horizontally or vertically
2. **Draggable panels** - Drag panels between any containers
3. **Type-switchable views** - Dropdown to change view type
4. **Resizable everywhere** - All splitters are draggable
5. **Layout persistence** - Save/restore complex layouts
6. **Modular code** - Each view type is self-contained

### Target Structure
```
Application
├── DockingContext (manages all dock nodes)
│   ├── DockNode (recursive tree structure)
│   │   ├── SplitContainer (horizontal or vertical)
│   │   │   ├── DockNode (child 1)
│   │   │   └── DockNode (child 2)
│   │   └── LeafNode (single view)
│   │       ├── ViewType (enum: Viewer, ControlPanel, Timeline, etc.)
│   │       └── ViewState (per-instance state)
│   └── FloatingWindows (detached panels)
├── ViewRegistry (maps ViewType to render function)
└── LayoutManager (save/load layouts)
```

---

## Phase 1: Core Data Structures

### 1.1 ViewType Enumeration

```cpp
// gui_gl/docking/view_types.h
enum class ViewType {
  // Visualization
  Viewer3D,           // Main OpenGL visualization
  Timeline,           // Animation timeline + playback controls

  // Main tab panels (solver configuration)
  EquationEditor,     // PDE input + LaTeX preview
  DomainSettings,     // Coordinate system + bounds
  GridSettings,       // Resolution configuration
  BoundaryConditions, // BC inputs for all faces
  SolverConfig,       // Backend + method selection
  TimeSettings,       // Time-dependent parameters
  RunControls,        // Solve/Stop + progress
  LogView,            // Debug output

  // Inspect tab panels
  FieldSelector,      // Field type dropdown
  SliceControls,      // Slice plane settings
  IsosurfaceControls, // Isosurface settings
  ImageExport,        // Export functionality
  AdvancedInspection, // Probes, line profiles
  ComparisonTools,    // Field comparison

  // Preferences panels
  Appearance,         // Color preferences
  ViewerSettings,     // Camera, grid, rendering
  IOPaths,            // Input/output directories
  LatexSettings,      // LaTeX preview configuration
  Benchmarks,         // Benchmark runner
  UIConfiguration,    // UI config editor

  // Meta
  Empty,              // Placeholder for empty slots
  Custom,             // User-defined (future extensibility)
};
```

### 1.2 DockNode Structure

```cpp
// gui_gl/docking/dock_node.h
#include <memory>
#include <variant>
#include <vector>

struct DockNode;

// Split direction
enum class SplitDirection {
  None,        // Leaf node
  Horizontal,  // Left | Right
  Vertical,    // Top | Bottom
};

// Leaf node content
struct LeafContent {
  ViewType view_type = ViewType::Empty;
  std::string instance_id;  // Unique ID for this view instance
  float scroll_y = 0.0f;    // Per-view scroll state
  // View-specific state stored in ViewStateRegistry
};

// Split container content
struct SplitContent {
  SplitDirection direction;
  float split_ratio = 0.5f;  // 0.0 to 1.0
  std::unique_ptr<DockNode> first;
  std::unique_ptr<DockNode> second;
};

// Unified dock node
struct DockNode {
  std::string id;  // Unique node ID (for serialization)
  std::variant<LeafContent, SplitContent> content;

  bool IsLeaf() const;
  bool IsSplit() const;
  LeafContent& AsLeaf();
  SplitContent& AsSplit();

  // Tree operations
  DockNode* FindNode(const std::string& node_id);
  void SplitLeaf(SplitDirection dir, ViewType new_view);
  void ReplaceChild(DockNode* old_child, std::unique_ptr<DockNode> new_child);
  void Collapse();  // Remove empty nodes and merge single-child splits
};
```

### 1.3 DockingContext

```cpp
// gui_gl/docking/docking_context.h
class DockingContext {
public:
  DockingContext();

  // Root node management
  DockNode& GetRoot();
  void SetRoot(std::unique_ptr<DockNode> root);

  // Node operations
  DockNode* FindNode(const std::string& id);
  void SplitNode(const std::string& node_id, SplitDirection dir, ViewType new_view);
  void CloseNode(const std::string& node_id);
  void SwapNodes(const std::string& id1, const std::string& id2);

  // Drag and drop
  void BeginDrag(const std::string& source_node_id);
  void UpdateDrag(const ImVec2& mouse_pos);
  void EndDrag(const std::string& target_node_id, DropZone zone);
  void CancelDrag();
  bool IsDragging() const;

  // Layout persistence
  void SaveLayout(const std::string& name);
  void LoadLayout(const std::string& name);
  void ResetToDefault();

  // Rendering
  void Render(const ImVec2& available_size, AppServices& services);

private:
  std::unique_ptr<DockNode> root_;
  std::string dragging_node_id_;
  DropPreview drop_preview_;
  std::unordered_map<std::string, LayoutData> saved_layouts_;
};
```

### 1.4 ViewRegistry

```cpp
// gui_gl/docking/view_registry.h
#include <functional>

struct ViewRenderContext {
  AppServices& services;
  const std::string& instance_id;
  float available_width;
  float available_height;
  bool is_focused;
};

using ViewRenderer = std::function<void(ViewRenderContext&)>;

struct ViewTypeInfo {
  std::string display_name;
  std::string icon;  // Unicode icon or texture ID
  std::string category;  // For grouping in dropdown
  ViewRenderer renderer;
  bool allow_multiple;  // Can have multiple instances?
};

class ViewRegistry {
public:
  static ViewRegistry& Instance();

  void Register(ViewType type, ViewTypeInfo info);
  const ViewTypeInfo* GetInfo(ViewType type) const;
  std::vector<ViewType> GetAllTypes() const;
  std::vector<ViewType> GetTypesByCategory(const std::string& category) const;

  void Render(ViewType type, ViewRenderContext& ctx);

private:
  std::unordered_map<ViewType, ViewTypeInfo> registry_;
};
```

---

## Phase 2: Rendering System

### 2.1 Recursive Node Rendering

```cpp
// gui_gl/docking/dock_renderer.cpp

void RenderDockNode(DockNode& node, const ImVec2& pos, const ImVec2& size,
                    AppServices& services, DockingContext& ctx) {
  if (node.IsLeaf()) {
    RenderLeafNode(node, pos, size, services, ctx);
  } else {
    RenderSplitNode(node, pos, size, services, ctx);
  }
}

void RenderSplitNode(DockNode& node, const ImVec2& pos, const ImVec2& size,
                     AppServices& services, DockingContext& ctx) {
  auto& split = node.AsSplit();

  // Calculate child sizes
  ImVec2 first_size, second_size;
  ImVec2 first_pos = pos, second_pos;
  float splitter_thickness = 4.0f;

  if (split.direction == SplitDirection::Horizontal) {
    float first_width = size.x * split.split_ratio - splitter_thickness / 2;
    first_size = ImVec2(first_width, size.y);
    second_pos = ImVec2(pos.x + first_width + splitter_thickness, pos.y);
    second_size = ImVec2(size.x - first_width - splitter_thickness, size.y);
  } else {
    float first_height = size.y * split.split_ratio - splitter_thickness / 2;
    first_size = ImVec2(size.x, first_height);
    second_pos = ImVec2(pos.x, pos.y + first_height + splitter_thickness);
    second_size = ImVec2(size.x, size.y - first_height - splitter_thickness);
  }

  // Render children recursively
  RenderDockNode(*split.first, first_pos, first_size, services, ctx);
  RenderDockNode(*split.second, second_pos, second_size, services, ctx);

  // Render splitter with drag interaction
  RenderSplitter(node, pos, size, split, ctx);
}

void RenderLeafNode(DockNode& node, const ImVec2& pos, const ImVec2& size,
                    AppServices& services, DockingContext& ctx) {
  auto& leaf = node.AsLeaf();

  ImGui::SetCursorScreenPos(pos);
  ImGui::BeginChild(node.id.c_str(), size, true, ImGuiWindowFlags_NoScrollbar);

  // Header with view type dropdown
  RenderViewHeader(node, leaf, ctx);

  // Content area
  ImVec2 content_size = ImGui::GetContentRegionAvail();
  ViewRenderContext render_ctx{
    services,
    leaf.instance_id,
    content_size.x,
    content_size.y,
    ImGui::IsWindowFocused(ImGuiFocusedFlags_ChildWindows)
  };

  ViewRegistry::Instance().Render(leaf.view_type, render_ctx);

  // Drop zones for drag-and-drop
  if (ctx.IsDragging()) {
    RenderDropZones(node, pos, size, ctx);
  }

  ImGui::EndChild();
}
```

### 2.2 View Header with Dropdown

```cpp
void RenderViewHeader(DockNode& node, LeafContent& leaf, DockingContext& ctx) {
  ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(4, 2));

  // View type info
  const auto* info = ViewRegistry::Instance().GetInfo(leaf.view_type);
  std::string header_text = info ? info->display_name : "Empty";

  // Drag handle (left side)
  ImGui::Button(":::");
  if (ImGui::BeginDragDropSource()) {
    ImGui::SetDragDropPayload("DOCK_NODE", node.id.c_str(), node.id.size() + 1);
    ImGui::Text("Move %s", header_text.c_str());
    ctx.BeginDrag(node.id);
    ImGui::EndDragDropSource();
  }

  ImGui::SameLine();
  ImGui::Text("%s", header_text.c_str());

  // View type selector dropdown (right side)
  ImGui::SameLine(ImGui::GetContentRegionAvail().x - 60);
  if (ImGui::BeginCombo("##viewtype", "", ImGuiComboFlags_NoPreview)) {
    // Group by category
    for (const auto& category : {"Visualization", "Configuration", "Inspection", "Settings"}) {
      if (ImGui::BeginMenu(category)) {
        for (ViewType type : ViewRegistry::Instance().GetTypesByCategory(category)) {
          const auto* type_info = ViewRegistry::Instance().GetInfo(type);
          if (ImGui::MenuItem(type_info->display_name.c_str())) {
            leaf.view_type = type;
            leaf.instance_id = GenerateInstanceId(type);
          }
        }
        ImGui::EndMenu();
      }
    }
    ImGui::EndCombo();
  }

  // Split buttons
  ImGui::SameLine();
  if (ImGui::Button("|")) {  // Split horizontal
    ctx.SplitNode(node.id, SplitDirection::Horizontal, ViewType::Empty);
  }
  ImGui::SameLine();
  if (ImGui::Button("-")) {  // Split vertical
    ctx.SplitNode(node.id, SplitDirection::Vertical, ViewType::Empty);
  }
  ImGui::SameLine();
  if (ImGui::Button("X")) {  // Close
    ctx.CloseNode(node.id);
  }

  ImGui::PopStyleVar();
  ImGui::Separator();
}
```

### 2.3 Drop Zone Visualization

```cpp
enum class DropZone {
  None,
  Center,     // Replace content
  Left,       // Split horizontal, new on left
  Right,      // Split horizontal, new on right
  Top,        // Split vertical, new on top
  Bottom,     // Split vertical, new on bottom
  Tab,        // Add as tab (future: tabbed panels)
};

void RenderDropZones(DockNode& node, const ImVec2& pos, const ImVec2& size,
                     DockingContext& ctx) {
  ImDrawList* draw_list = ImGui::GetWindowDrawList();
  ImVec2 center = ImVec2(pos.x + size.x / 2, pos.y + size.y / 2);

  // Zone dimensions
  float zone_size = 40.0f;
  float preview_alpha = 0.3f;
  ImU32 highlight_color = IM_COL32(100, 150, 255, 200);
  ImU32 preview_color = IM_COL32(100, 150, 255, 80);

  // Check mouse position and highlight appropriate zone
  ImVec2 mouse = ImGui::GetMousePos();
  DropZone hovered = DetectDropZone(pos, size, mouse);

  // Draw zone indicators
  struct ZoneRect { DropZone zone; ImVec2 min; ImVec2 max; };
  std::vector<ZoneRect> zones = {
    {DropZone::Center, ImVec2(center.x - zone_size/2, center.y - zone_size/2),
                       ImVec2(center.x + zone_size/2, center.y + zone_size/2)},
    {DropZone::Left,   ImVec2(pos.x, pos.y), ImVec2(pos.x + size.x * 0.25f, pos.y + size.y)},
    {DropZone::Right,  ImVec2(pos.x + size.x * 0.75f, pos.y), ImVec2(pos.x + size.x, pos.y + size.y)},
    {DropZone::Top,    ImVec2(pos.x, pos.y), ImVec2(pos.x + size.x, pos.y + size.y * 0.25f)},
    {DropZone::Bottom, ImVec2(pos.x, pos.y + size.y * 0.75f), ImVec2(pos.x + size.x, pos.y + size.y)},
  };

  for (const auto& zone : zones) {
    bool is_hovered = (zone.zone == hovered);
    ImU32 color = is_hovered ? highlight_color : preview_color;
    draw_list->AddRectFilled(zone.min, zone.max, color, 4.0f);
  }

  ctx.SetDropPreview(node.id, hovered);
}
```

---

## Phase 3: View Implementations

### 3.1 Convert Existing Panels to Views

Each panel needs to be wrapped as a self-contained view:

```cpp
// gui_gl/views/viewer_3d_view.cpp
void RegisterViewer3DView() {
  ViewRegistry::Instance().Register(ViewType::Viewer3D, {
    .display_name = "3D Viewer",
    .icon = "🎬",
    .category = "Visualization",
    .renderer = [](ViewRenderContext& ctx) {
      // Get or create viewer state for this instance
      auto& state = ViewStateRegistry::Get<Viewer3DState>(ctx.instance_id);

      // Render visualization
      GlViewer& viewer = *ctx.services.viewer;

      // Camera controls, rendering, mouse interaction
      ImVec2 size(ctx.available_width, ctx.available_height);
      ImTextureID tex_id = viewer.GetTextureID();
      ImGui::Image(tex_id, size);

      // Mouse interaction for rotation/zoom
      if (ImGui::IsItemHovered()) {
        HandleViewerInteraction(viewer, state);
      }
    },
    .allow_multiple = true,  // Multiple viewer instances allowed
  });
}

// gui_gl/views/equation_editor_view.cpp
void RegisterEquationEditorView() {
  ViewRegistry::Instance().Register(ViewType::EquationEditor, {
    .display_name = "Equation Editor",
    .icon = "∂",
    .category = "Configuration",
    .renderer = [](ViewRenderContext& ctx) {
      auto& state = ViewStateRegistry::Get<EquationEditorState>(ctx.instance_id);

      // PDE input
      ImGui::Text("PDE Expression");
      ImGui::SetNextItemWidth(ctx.available_width);
      ImGui::InputTextMultiline("##pde", &state.pde_text,
                                ImVec2(ctx.available_width, 100));

      // LaTeX preview
      if (state.preview_texture) {
        ImGui::Image(state.preview_texture, ImVec2(ctx.available_width, 60));
      }

      // Template picker
      RenderTemplatePicker(state);
    },
    .allow_multiple = false,  // Only one equation editor needed
  });
}
```

### 3.2 View State Registry

```cpp
// gui_gl/docking/view_state_registry.h
class ViewStateRegistry {
public:
  template<typename T>
  static T& Get(const std::string& instance_id) {
    auto& registry = Instance();
    auto key = std::make_pair(std::type_index(typeid(T)), instance_id);

    if (registry.states_.find(key) == registry.states_.end()) {
      registry.states_[key] = std::make_any<T>();
    }

    return std::any_cast<T&>(registry.states_[key]);
  }

  static void Remove(const std::string& instance_id) {
    auto& registry = Instance();
    std::erase_if(registry.states_, [&](const auto& pair) {
      return pair.first.second == instance_id;
    });
  }

private:
  static ViewStateRegistry& Instance();
  std::map<std::pair<std::type_index, std::string>, std::any> states_;
};
```

---

## Phase 4: Layout Persistence

### 4.1 Layout Serialization

```cpp
// gui_gl/docking/layout_serializer.h
#include <nlohmann/json.hpp>

class LayoutSerializer {
public:
  // Serialize dock tree to JSON
  static nlohmann::json SerializeNode(const DockNode& node) {
    nlohmann::json j;
    j["id"] = node.id;

    if (node.IsLeaf()) {
      const auto& leaf = node.AsLeaf();
      j["type"] = "leaf";
      j["view_type"] = static_cast<int>(leaf.view_type);
      j["instance_id"] = leaf.instance_id;
    } else {
      const auto& split = node.AsSplit();
      j["type"] = "split";
      j["direction"] = split.direction == SplitDirection::Horizontal ? "h" : "v";
      j["ratio"] = split.split_ratio;
      j["first"] = SerializeNode(*split.first);
      j["second"] = SerializeNode(*split.second);
    }

    return j;
  }

  // Deserialize JSON to dock tree
  static std::unique_ptr<DockNode> DeserializeNode(const nlohmann::json& j) {
    auto node = std::make_unique<DockNode>();
    node->id = j["id"];

    if (j["type"] == "leaf") {
      LeafContent leaf;
      leaf.view_type = static_cast<ViewType>(j["view_type"].get<int>());
      leaf.instance_id = j["instance_id"];
      node->content = leaf;
    } else {
      SplitContent split;
      split.direction = j["direction"] == "h" ?
                        SplitDirection::Horizontal : SplitDirection::Vertical;
      split.split_ratio = j["ratio"];
      split.first = DeserializeNode(j["first"]);
      split.second = DeserializeNode(j["second"]);
      node->content = split;
    }

    return node;
  }
};
```

### 4.2 Preset Layouts

```cpp
// gui_gl/docking/preset_layouts.h
class PresetLayouts {
public:
  static std::unique_ptr<DockNode> CreateDefault() {
    // Main split: Left (controls) | Right (viewer + timeline)
    auto root = CreateHorizontalSplit(0.35f,
      // Left: stacked control panels
      CreateVerticalSplit(0.5f,
        CreateLeaf(ViewType::EquationEditor),
        CreateVerticalSplit(0.5f,
          CreateLeaf(ViewType::DomainSettings),
          CreateLeaf(ViewType::SolverConfig)
        )
      ),
      // Right: viewer on top, timeline on bottom
      CreateVerticalSplit(0.85f,
        CreateLeaf(ViewType::Viewer3D),
        CreateLeaf(ViewType::Timeline)
      )
    );
    return root;
  }

  static std::unique_ptr<DockNode> CreateInspection() {
    // Optimized for post-processing visualization
    return CreateHorizontalSplit(0.25f,
      CreateVerticalSplit(0.5f,
        CreateLeaf(ViewType::FieldSelector),
        CreateLeaf(ViewType::SliceControls)
      ),
      CreateVerticalSplit(0.85f,
        CreateLeaf(ViewType::Viewer3D),
        CreateLeaf(ViewType::Timeline)
      )
    );
  }

  static std::unique_ptr<DockNode> CreateDualViewer() {
    // Two viewers side by side for comparison
    return CreateVerticalSplit(0.85f,
      CreateHorizontalSplit(0.5f,
        CreateLeaf(ViewType::Viewer3D),
        CreateLeaf(ViewType::Viewer3D)
      ),
      CreateLeaf(ViewType::Timeline)
    );
  }
};
```

---

## Phase 5: Integration

### 5.1 Application Integration

```cpp
// gui_gl/core/application.h (additions)
class Application {
  // ... existing members ...

private:
  // New docking system
  std::unique_ptr<DockingContext> docking_ctx_;
  ViewRegistry view_registry_;

  void InitializeDockingSystem();
  void RenderDockingUI();
  void RegisterAllViews();
};

// gui_gl/core/application.cpp
void Application::InitializeDockingSystem() {
  docking_ctx_ = std::make_unique<DockingContext>();

  // Register all view types
  RegisterAllViews();

  // Try to load saved layout, or use default
  if (!docking_ctx_->LoadLayout("last_session")) {
    docking_ctx_->SetRoot(PresetLayouts::CreateDefault());
  }
}

void Application::RegisterAllViews() {
  // Visualization
  RegisterViewer3DView();
  RegisterTimelineView();

  // Configuration (Main tab panels)
  RegisterEquationEditorView();
  RegisterDomainSettingsView();
  RegisterGridSettingsView();
  RegisterBoundaryConditionsView();
  RegisterSolverConfigView();
  RegisterTimeSettingsView();
  RegisterRunControlsView();
  RegisterLogView();

  // Inspection
  RegisterFieldSelectorView();
  RegisterSliceControlsView();
  RegisterIsosurfaceControlsView();
  RegisterImageExportView();
  RegisterAdvancedInspectionView();
  RegisterComparisonToolsView();

  // Preferences
  RegisterAppearanceView();
  RegisterViewerSettingsView();
  RegisterIOPathsView();
  RegisterLatexSettingsView();
  RegisterBenchmarksView();
  RegisterUIConfigurationView();
}

void Application::RenderDockingUI() {
  ImVec2 available = ImGui::GetContentRegionAvail();
  docking_ctx_->Render(available, services_);
}
```

### 5.2 Menu Integration

```cpp
void Application::RenderMenuBar() {
  if (ImGui::BeginMenuBar()) {
    // ... existing menus ...

    if (ImGui::BeginMenu("Layout")) {
      if (ImGui::MenuItem("Reset to Default")) {
        docking_ctx_->SetRoot(PresetLayouts::CreateDefault());
      }
      if (ImGui::MenuItem("Inspection Layout")) {
        docking_ctx_->SetRoot(PresetLayouts::CreateInspection());
      }
      if (ImGui::MenuItem("Dual Viewer")) {
        docking_ctx_->SetRoot(PresetLayouts::CreateDualViewer());
      }

      ImGui::Separator();

      if (ImGui::BeginMenu("Saved Layouts")) {
        for (const auto& name : docking_ctx_->GetSavedLayoutNames()) {
          if (ImGui::MenuItem(name.c_str())) {
            docking_ctx_->LoadLayout(name);
          }
        }
        ImGui::EndMenu();
      }

      if (ImGui::MenuItem("Save Current Layout...")) {
        // Open save dialog
      }

      ImGui::EndMenu();
    }

    ImGui::EndMenuBar();
  }
}
```

---

## Phase 6: Migration Strategy

### Step 1: Parallel Implementation
- Build docking system alongside existing tab-based UI
- Add feature flag: `use_docking_ui` in config
- Both systems share underlying panel renderers

### Step 2: View Wrapper Creation
- Wrap each existing panel as a View
- Maintain backward compatibility with panel_registry_
- Views call same render functions as panels

### Step 3: State Management Migration
- Move scattered Application members to ViewState structs
- Create adapters for shared state access
- Ensure thread safety for multi-viewer scenarios

### Step 4: Testing & Refinement
- Test all view combinations
- Verify layout save/restore
- Profile performance with multiple viewers

### Step 5: Default Transition
- Set docking UI as default
- Deprecate tab-based UI (keep for fallback)
- Document new workflow

---

## File Structure

```
gui_gl/
├── docking/
│   ├── dock_node.h/cpp           # Tree structure
│   ├── docking_context.h/cpp     # Manager class
│   ├── dock_renderer.h/cpp       # Rendering logic
│   ├── view_registry.h/cpp       # View type registry
│   ├── view_state_registry.h/cpp # Per-instance state
│   ├── layout_serializer.h/cpp   # JSON save/load
│   ├── preset_layouts.h/cpp      # Default layouts
│   ├── drop_zones.h/cpp          # Drag-drop logic
│   └── view_types.h              # ViewType enum
├── views/
│   ├── viewer_3d_view.cpp
│   ├── timeline_view.cpp
│   ├── equation_editor_view.cpp
│   ├── domain_settings_view.cpp
│   ├── grid_settings_view.cpp
│   ├── boundary_conditions_view.cpp
│   ├── solver_config_view.cpp
│   ├── time_settings_view.cpp
│   ├── run_controls_view.cpp
│   ├── log_view.cpp
│   ├── field_selector_view.cpp
│   ├── slice_controls_view.cpp
│   ├── isosurface_controls_view.cpp
│   ├── image_export_view.cpp
│   ├── advanced_inspection_view.cpp
│   ├── comparison_tools_view.cpp
│   ├── appearance_view.cpp
│   ├── viewer_settings_view.cpp
│   ├── io_paths_view.cpp
│   ├── latex_settings_view.cpp
│   ├── benchmarks_view.cpp
│   └── ui_configuration_view.cpp
└── ... (existing files)
```

---

## Estimated Effort

| Phase | Description | Complexity |
|-------|-------------|------------|
| 1 | Core Data Structures | Medium |
| 2 | Rendering System | High |
| 3 | View Implementations | Medium (per view) |
| 4 | Layout Persistence | Low |
| 5 | Integration | Medium |
| 6 | Migration | High |

**Total estimated scope**: Major refactoring effort

---

## Key Design Decisions

1. **No ImGui Docking Branch**: Build custom solution for full control
2. **Recursive Tree Structure**: Simple, flexible, serializable
3. **View Registry Pattern**: Decouples view types from rendering
4. **Instance-based State**: Supports multiple viewers/editors
5. **JSON Layout Storage**: Human-readable, versionable configs
6. **Preset Layouts**: Quick access to common configurations
7. **Gradual Migration**: Parallel systems during transition

---

## Dependencies

- Existing: ImGui, nlohmann/json
- No new external dependencies required

---

## Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| Performance with many views | Lazy rendering, view culling |
| State synchronization bugs | Centralized state registry |
| Complex drag-drop edge cases | Comprehensive testing |
| Layout corruption | Validation on load, fallback to default |
| GlViewer multi-instance issues | Shared texture with per-view camera |

---

## Success Criteria

1. Any view can be split horizontally or vertically
2. Views can be dragged between any containers
3. View type can be changed via dropdown
4. Layouts persist across sessions
5. Performance matches current implementation
6. All existing functionality preserved
