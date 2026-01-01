#include "docking_context.h"
#include "view_registry.h"
#include "view_state_registry.h"
#include "layout_serializer.h"
#include "imgui.h"
#include <algorithm>
#include <fstream>

namespace {
  constexpr float kSplitterThickness = 4.0f;
  constexpr float kMinPaneSize = 50.0f;
  constexpr float kHeaderHeight = 24.0f;
  constexpr float kDropZoneSize = 40.0f;
}

DockingContext::DockingContext() {
  // Start with a single empty leaf node
  root_ = CreateLeafNode(ViewType::Empty);
}

DockingContext::~DockingContext() = default;

DockNode* DockingContext::GetRoot() {
  return root_.get();
}

const DockNode* DockingContext::GetRoot() const {
  return root_.get();
}

void DockingContext::SetRoot(std::unique_ptr<DockNode> root) {
  root_ = std::move(root);
}

DockNode* DockingContext::FindNode(const std::string& id) {
  return root_ ? root_->FindNode(id) : nullptr;
}

void DockingContext::SplitNode(const std::string& node_id, SplitDirection dir,
                                ViewType new_view) {
  DockNode* node = FindNode(node_id);
  if (node && node->IsLeaf()) {
    node->SplitLeaf(dir, new_view);
  }
}

void DockingContext::CloseNode(const std::string& node_id) {
  DockNode* node = FindNode(node_id);
  if (node && node->IsLeaf()) {
    // Remove view state for this instance
    ViewStateRegistry::Instance().Remove(node->AsLeaf().instance_id);
    // Mark as empty
    node->AsLeaf().view_type = ViewType::Empty;
    // Cleanup tree
    Cleanup();
  }
}

void DockingContext::SetNodeViewType(const std::string& node_id, ViewType type) {
  DockNode* node = FindNode(node_id);
  if (node && node->IsLeaf()) {
    auto& leaf = node->AsLeaf();
    // Remove old state
    ViewStateRegistry::Instance().Remove(leaf.instance_id);
    // Set new type and generate new instance ID
    leaf.view_type = type;
    leaf.instance_id = GenerateInstanceId(type);
    leaf.scroll_y = 0.0f;
  }
}

void DockingContext::SwapNodes(const std::string& id1, const std::string& id2) {
  DockNode* node1 = FindNode(id1);
  DockNode* node2 = FindNode(id2);

  if (node1 && node2 && node1->IsLeaf() && node2->IsLeaf()) {
    std::swap(node1->content, node2->content);
  }
}

void DockingContext::Cleanup() {
  if (!root_) return;

  // Recursively cleanup the tree
  if (root_->IsSplit()) {
    auto& split = root_->AsSplit();
    CleanupNode(nullptr, root_);
  }
}

void DockingContext::CleanupNode(DockNode* parent, std::unique_ptr<DockNode>& node_ptr) {
  if (!node_ptr) return;

  DockNode& node = *node_ptr;

  if (node.IsSplit()) {
    auto& split = node.AsSplit();

    // Recursively cleanup children
    CleanupNode(&node, split.first);
    CleanupNode(&node, split.second);

    // If both children are gone, this node becomes empty leaf
    if (!split.first && !split.second) {
      node_ptr = CreateLeafNode(ViewType::Empty);
      return;
    }

    // If only one child remains, promote it
    if (!split.first) {
      node_ptr = std::move(split.second);
      return;
    }
    if (!split.second) {
      node_ptr = std::move(split.first);
      return;
    }

    // If one child is an empty leaf, collapse
    if (split.first->IsLeaf() && split.first->AsLeaf().view_type == ViewType::Empty) {
      node_ptr = std::move(split.second);
      return;
    }
    if (split.second->IsLeaf() && split.second->AsLeaf().view_type == ViewType::Empty) {
      node_ptr = std::move(split.first);
      return;
    }
  }
}

DockNode* DockingContext::FindParent(const std::string& child_id) {
  if (!root_ || root_->id == child_id) return nullptr;

  DockNode* result = nullptr;
  root_->ForEach([&](DockNode& node) {
    if (node.IsSplit()) {
      auto& split = node.AsSplit();
      if ((split.first && split.first->id == child_id) ||
          (split.second && split.second->id == child_id)) {
        result = &node;
      }
    }
  });
  return result;
}

// Drag and drop
void DockingContext::BeginDrag(const std::string& source_node_id) {
  is_dragging_ = true;
  dragging_node_id_ = source_node_id;
  drop_preview_ = DropPreview{};
}

void DockingContext::UpdateDrag(const ImVec2& mouse_pos) {
  // Drop preview is set during node rendering
}

void DockingContext::EndDrag(const std::string& target_node_id, DropZone zone) {
  if (!is_dragging_ || dragging_node_id_.empty() || target_node_id.empty()) {
    CancelDrag();
    return;
  }

  if (dragging_node_id_ == target_node_id) {
    CancelDrag();
    return;
  }

  DockNode* source = FindNode(dragging_node_id_);
  DockNode* target = FindNode(target_node_id);

  if (!source || !target || !source->IsLeaf() || !target->IsLeaf()) {
    CancelDrag();
    return;
  }

  switch (zone) {
    case DropZone::Center:
      // Swap contents
      std::swap(source->content, target->content);
      break;

    case DropZone::Left:
      // Split target horizontally, put source content on left
      {
        LeafContent source_content = std::move(source->AsLeaf());
        source->AsLeaf().view_type = ViewType::Empty;

        target->SplitLeaf(SplitDirection::Horizontal, ViewType::Empty);
        auto& split = target->AsSplit();
        split.first->content = std::move(source_content);
      }
      break;

    case DropZone::Right:
      // Split target horizontally, put source content on right
      {
        LeafContent source_content = std::move(source->AsLeaf());
        source->AsLeaf().view_type = ViewType::Empty;

        target->SplitLeaf(SplitDirection::Horizontal, ViewType::Empty);
        auto& split = target->AsSplit();
        // First already has target content, second should get source
        std::swap(split.first->content, split.second->content);
        split.second->content = std::move(source_content);
      }
      break;

    case DropZone::Top:
      // Split target vertically, put source content on top
      {
        LeafContent source_content = std::move(source->AsLeaf());
        source->AsLeaf().view_type = ViewType::Empty;

        target->SplitLeaf(SplitDirection::Vertical, ViewType::Empty);
        auto& split = target->AsSplit();
        split.first->content = std::move(source_content);
      }
      break;

    case DropZone::Bottom:
      // Split target vertically, put source content on bottom
      {
        LeafContent source_content = std::move(source->AsLeaf());
        source->AsLeaf().view_type = ViewType::Empty;

        target->SplitLeaf(SplitDirection::Vertical, ViewType::Empty);
        auto& split = target->AsSplit();
        std::swap(split.first->content, split.second->content);
        split.second->content = std::move(source_content);
      }
      break;

    default:
      break;
  }

  Cleanup();
  CancelDrag();
}

void DockingContext::CancelDrag() {
  is_dragging_ = false;
  dragging_node_id_.clear();
  drop_preview_ = DropPreview{};
}

bool DockingContext::IsDragging() const {
  return is_dragging_;
}

const std::string& DockingContext::GetDraggingNodeId() const {
  return dragging_node_id_;
}

const DropPreview& DockingContext::GetDropPreview() const {
  return drop_preview_;
}

void DockingContext::SetDropPreview(const std::string& node_id, DropZone zone,
                                     const ImVec2& min, const ImVec2& max) {
  drop_preview_.target_node_id = node_id;
  drop_preview_.zone = zone;
  drop_preview_.preview_min = min;
  drop_preview_.preview_max = max;
}

// Layout persistence
std::filesystem::path DockingContext::GetLayoutDirectory(
    const std::filesystem::path& project_dir) {
  return project_dir / "gui_gl" / "layouts";
}

std::filesystem::path DockingContext::GetLayoutFilePath(
    const std::string& name, const std::filesystem::path& project_dir) {
  return GetLayoutDirectory(project_dir) / (name + ".json");
}

bool DockingContext::SaveLayout(const std::string& name,
                                 const std::filesystem::path& project_dir) {
  if (!root_) return false;

  std::filesystem::path layout_dir = GetLayoutDirectory(project_dir);
  std::error_code ec;
  std::filesystem::create_directories(layout_dir, ec);
  if (ec) return false;

  std::filesystem::path file_path = GetLayoutFilePath(name, project_dir);
  return LayoutSerializer::SaveToFile(*root_, file_path.string());
}

bool DockingContext::LoadLayout(const std::string& name,
                                 const std::filesystem::path& project_dir) {
  std::filesystem::path file_path = GetLayoutFilePath(name, project_dir);

  std::error_code ec;
  if (!std::filesystem::exists(file_path, ec)) {
    return false;
  }

  auto loaded = LayoutSerializer::LoadFromFile(file_path.string());
  if (loaded) {
    root_ = std::move(loaded);
    return true;
  }
  return false;
}

std::vector<std::string> DockingContext::GetSavedLayoutNames(
    const std::filesystem::path& project_dir) const {
  std::vector<std::string> names;
  std::filesystem::path layout_dir = GetLayoutDirectory(project_dir);

  std::error_code ec;
  if (!std::filesystem::exists(layout_dir, ec)) {
    return names;
  }

  for (const auto& entry : std::filesystem::directory_iterator(layout_dir, ec)) {
    if (entry.path().extension() == ".json") {
      names.push_back(entry.path().stem().string());
    }
  }

  std::sort(names.begin(), names.end());
  return names;
}

bool DockingContext::DeleteLayout(const std::string& name,
                                   const std::filesystem::path& project_dir) {
  std::filesystem::path file_path = GetLayoutFilePath(name, project_dir);
  std::error_code ec;
  return std::filesystem::remove(file_path, ec);
}

// Rendering
void DockingContext::Render(const ImVec2& position, const ImVec2& available_size,
                             AppServices& services) {
  if (!root_) return;

  // Clear drop preview at start of frame
  if (!is_dragging_) {
    drop_preview_ = DropPreview{};
  }

  // Render the dock tree
  RenderNode(*root_, position, available_size, services);

  // Draw drop preview overlay
  if (is_dragging_ && drop_preview_.zone != DropZone::None) {
    ImDrawList* draw_list = ImGui::GetForegroundDrawList();
    ImU32 preview_color = IM_COL32(100, 150, 255, 100);
    draw_list->AddRectFilled(drop_preview_.preview_min, drop_preview_.preview_max,
                             preview_color, 4.0f);
  }

  // Handle drag cancellation when mouse is released outside a valid drop zone
  if (is_dragging_) {
    if (ImGui::IsMouseReleased(ImGuiMouseButton_Left)) {
      // If no drop zone was detected this frame, cancel the drag
      if (drop_preview_.zone == DropZone::None) {
        CancelDrag();
      }
    }
    // Also cancel if user presses Escape
    if (ImGui::IsKeyPressed(ImGuiKey_Escape)) {
      CancelDrag();
    }
  }
}

void DockingContext::RenderNode(DockNode& node, const ImVec2& pos,
                                 const ImVec2& size, AppServices& services) {
  if (node.IsLeaf()) {
    RenderLeafNode(node, pos, size, services);
  } else {
    RenderSplitNode(node, pos, size, services);
  }
}

void DockingContext::RenderSplitNode(DockNode& node, const ImVec2& pos,
                                      const ImVec2& size, AppServices& services) {
  auto& split = node.AsSplit();

  // Calculate child sizes
  ImVec2 first_pos = pos;
  ImVec2 first_size, second_size, second_pos;

  if (split.direction == SplitDirection::Horizontal) {
    float first_width = size.x * split.split_ratio - kSplitterThickness / 2;
    first_width = std::max(kMinPaneSize, std::min(first_width, size.x - kMinPaneSize - kSplitterThickness));

    first_size = ImVec2(first_width, size.y);
    second_pos = ImVec2(pos.x + first_width + kSplitterThickness, pos.y);
    second_size = ImVec2(size.x - first_width - kSplitterThickness, size.y);
  } else {
    float first_height = size.y * split.split_ratio - kSplitterThickness / 2;
    first_height = std::max(kMinPaneSize, std::min(first_height, size.y - kMinPaneSize - kSplitterThickness));

    first_size = ImVec2(size.x, first_height);
    second_pos = ImVec2(pos.x, pos.y + first_height + kSplitterThickness);
    second_size = ImVec2(size.x, size.y - first_height - kSplitterThickness);
  }

  // Render children
  if (split.first) {
    RenderNode(*split.first, first_pos, first_size, services);
  }
  if (split.second) {
    RenderNode(*split.second, second_pos, second_size, services);
  }

  // Render splitter
  RenderSplitter(node, pos, size, split);
}

void DockingContext::RenderSplitter(DockNode& node, const ImVec2& pos,
                                     const ImVec2& size, SplitContent& split) {
  ImVec2 splitter_pos, splitter_size;

  if (split.direction == SplitDirection::Horizontal) {
    float x = pos.x + size.x * split.split_ratio - kSplitterThickness / 2;
    splitter_pos = ImVec2(x, pos.y);
    splitter_size = ImVec2(kSplitterThickness, size.y);
  } else {
    float y = pos.y + size.y * split.split_ratio - kSplitterThickness / 2;
    splitter_pos = ImVec2(pos.x, y);
    splitter_size = ImVec2(size.x, kSplitterThickness);
  }

  ImGui::SetCursorScreenPos(splitter_pos);
  std::string splitter_id = "##splitter_" + node.id;
  ImGui::InvisibleButton(splitter_id.c_str(), splitter_size);

  bool hovered = ImGui::IsItemHovered();
  bool active = ImGui::IsItemActive();

  if (hovered || active) {
    ImGui::SetMouseCursor(split.direction == SplitDirection::Horizontal
                          ? ImGuiMouseCursor_ResizeEW
                          : ImGuiMouseCursor_ResizeNS);
  }

  // Handle dragging
  if (active) {
    ImGuiIO& io = ImGui::GetIO();
    if (split.direction == SplitDirection::Horizontal) {
      float new_ratio = (io.MousePos.x - pos.x) / size.x;
      split.split_ratio = std::max(0.1f, std::min(0.9f, new_ratio));
    } else {
      float new_ratio = (io.MousePos.y - pos.y) / size.y;
      split.split_ratio = std::max(0.1f, std::min(0.9f, new_ratio));
    }
  }

  // Draw splitter
  ImDrawList* draw_list = ImGui::GetWindowDrawList();
  ImU32 color = active ? IM_COL32(120, 120, 120, 255)
                       : (hovered ? IM_COL32(100, 100, 100, 255)
                                  : IM_COL32(60, 60, 60, 255));
  draw_list->AddRectFilled(splitter_pos,
                           ImVec2(splitter_pos.x + splitter_size.x,
                                  splitter_pos.y + splitter_size.y),
                           color);
}

void DockingContext::RenderLeafNode(DockNode& node, const ImVec2& pos,
                                     const ImVec2& size, AppServices& services) {
  auto& leaf = node.AsLeaf();

  ImGui::SetCursorScreenPos(pos);

  // Create child window for this leaf
  std::string child_id = "##dock_leaf_" + node.id;
  ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(4, 4));
  ImGui::PushStyleVar(ImGuiStyleVar_ChildBorderSize, 1.0f);

  // Allow scrolling for panel content - disable only horizontal scrollbar
  ImGuiWindowFlags leaf_flags = ImGuiWindowFlags_None;
  // For viewer, disable scrolling; for panels, allow vertical scrolling
  if (leaf.view_type == ViewType::Viewer3D || leaf.view_type == ViewType::Timeline) {
    leaf_flags = ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse;
  }

  ImGui::BeginChild(child_id.c_str(), size, true, leaf_flags);

  // Render header
  RenderViewHeader(node, leaf);

  // Render content
  ImVec2 content_size = ImGui::GetContentRegionAvail();
  bool is_focused = ImGui::IsWindowFocused(ImGuiFocusedFlags_ChildWindows);

  if (leaf.view_type != ViewType::Empty) {
    ViewRenderContext ctx{
      services,
      leaf.instance_id,
      content_size.x,
      content_size.y,
      is_focused
    };
    ViewRegistry::Instance().Render(leaf.view_type, ctx);
  } else {
    ImGui::TextDisabled("Empty - select a view type from the dropdown");
  }

  // Drop zones when dragging
  if (is_dragging_ && dragging_node_id_ != node.id) {
    RenderDropZones(node, pos, size);
  }

  ImGui::EndChild();
  ImGui::PopStyleVar(2);
}

void DockingContext::RenderViewHeader(DockNode& node, LeafContent& leaf) {
  ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(4, 2));
  ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(4, 2));

  // Drag handle
  ImGui::Button(":::");
  if (ImGui::BeginDragDropSource(ImGuiDragDropFlags_SourceAllowNullID)) {
    ImGui::SetDragDropPayload("DOCK_NODE", node.id.c_str(), node.id.size() + 1);
    ImGui::Text("Move %s", GetViewTypeDisplayName(leaf.view_type).c_str());
    BeginDrag(node.id);
    ImGui::EndDragDropSource();
  }

  ImGui::SameLine();

  // View title
  std::string title = GetViewTypeDisplayName(leaf.view_type);
  ImGui::Text("%s", title.c_str());

  // Right-aligned buttons
  float button_width = 20.0f;
  float combo_width = 20.0f;
  float right_edge = ImGui::GetContentRegionAvail().x + ImGui::GetCursorPosX();

  ImGui::SameLine(right_edge - button_width * 3 - combo_width - 16);

  // View type dropdown
  ImGui::PushItemWidth(combo_width);
  if (ImGui::BeginCombo("##viewtype", "", ImGuiComboFlags_NoPreview)) {
    for (const std::string& category : ViewRegistry::Instance().GetCategories()) {
      if (ImGui::BeginMenu(category.c_str())) {
        for (ViewType type : ViewRegistry::Instance().GetTypesByCategory(category)) {
          const auto* info = ViewRegistry::Instance().GetInfo(type);
          if (info) {
            bool selected = (type == leaf.view_type);
            if (ImGui::MenuItem(info->display_name.c_str(), nullptr, selected)) {
              SetNodeViewType(node.id, type);
            }
          }
        }
        ImGui::EndMenu();
      }
    }
    ImGui::EndCombo();
  }
  ImGui::PopItemWidth();

  ImGui::SameLine();

  // Split horizontal button
  if (ImGui::Button("|", ImVec2(button_width, 0))) {
    SplitNode(node.id, SplitDirection::Horizontal, ViewType::Empty);
  }
  if (ImGui::IsItemHovered()) {
    ImGui::SetTooltip("Split Horizontal");
  }

  ImGui::SameLine();

  // Split vertical button
  if (ImGui::Button("-", ImVec2(button_width, 0))) {
    SplitNode(node.id, SplitDirection::Vertical, ViewType::Empty);
  }
  if (ImGui::IsItemHovered()) {
    ImGui::SetTooltip("Split Vertical");
  }

  ImGui::SameLine();

  // Close button
  if (ImGui::Button("X", ImVec2(button_width, 0))) {
    CloseNode(node.id);
  }
  if (ImGui::IsItemHovered()) {
    ImGui::SetTooltip("Close");
  }

  ImGui::PopStyleVar(2);
  ImGui::Separator();
}

void DockingContext::RenderDropZones(DockNode& node, const ImVec2& pos,
                                      const ImVec2& size) {
  ImVec2 mouse = ImGui::GetMousePos();
  ImVec2 center = ImVec2(pos.x + size.x / 2, pos.y + size.y / 2);

  // Detect which zone the mouse is in
  DropZone detected = DropZone::None;
  ImVec2 preview_min, preview_max;

  // Check zones in order of priority
  float zone_margin = size.x * 0.25f;
  float zone_margin_y = size.y * 0.25f;

  if (mouse.x >= pos.x && mouse.x < pos.x + zone_margin &&
      mouse.y >= pos.y && mouse.y < pos.y + size.y) {
    detected = DropZone::Left;
    preview_min = pos;
    preview_max = ImVec2(pos.x + size.x * 0.5f, pos.y + size.y);
  } else if (mouse.x >= pos.x + size.x - zone_margin && mouse.x < pos.x + size.x &&
             mouse.y >= pos.y && mouse.y < pos.y + size.y) {
    detected = DropZone::Right;
    preview_min = ImVec2(pos.x + size.x * 0.5f, pos.y);
    preview_max = ImVec2(pos.x + size.x, pos.y + size.y);
  } else if (mouse.y >= pos.y && mouse.y < pos.y + zone_margin_y &&
             mouse.x >= pos.x && mouse.x < pos.x + size.x) {
    detected = DropZone::Top;
    preview_min = pos;
    preview_max = ImVec2(pos.x + size.x, pos.y + size.y * 0.5f);
  } else if (mouse.y >= pos.y + size.y - zone_margin_y && mouse.y < pos.y + size.y &&
             mouse.x >= pos.x && mouse.x < pos.x + size.x) {
    detected = DropZone::Bottom;
    preview_min = ImVec2(pos.x, pos.y + size.y * 0.5f);
    preview_max = ImVec2(pos.x + size.x, pos.y + size.y);
  } else if (mouse.x >= pos.x && mouse.x < pos.x + size.x &&
             mouse.y >= pos.y && mouse.y < pos.y + size.y) {
    detected = DropZone::Center;
    preview_min = pos;
    preview_max = ImVec2(pos.x + size.x, pos.y + size.y);
  }

  if (detected != DropZone::None) {
    SetDropPreview(node.id, detected, preview_min, preview_max);

    // Handle drop on mouse release
    if (ImGui::IsMouseReleased(ImGuiMouseButton_Left)) {
      EndDrag(node.id, detected);
    }
  }
}

size_t DockingContext::GetNodeCount() const {
  return root_ ? root_->CountNodes() : 0;
}

size_t DockingContext::GetLeafCount() const {
  return root_ ? root_->CountLeaves() : 0;
}
