#ifndef DOCKING_CONTEXT_H
#define DOCKING_CONTEXT_H

#include "dock_node.h"
#include "view_types.h"
#include "../core/app_services.h"
#include "imgui.h"
#include <memory>
#include <string>
#include <vector>
#include <filesystem>

// Drop zone for drag-and-drop
enum class DropZone {
  None,
  Center,     // Replace content
  Left,       // Split horizontal, new on left
  Right,      // Split horizontal, new on right
  Top,        // Split vertical, new on top
  Bottom,     // Split vertical, new on bottom
};

// Preview info for drop target
struct DropPreview {
  std::string target_node_id;
  DropZone zone = DropZone::None;
  ImVec2 preview_min;
  ImVec2 preview_max;
};

// Main docking context - manages the dock tree and interactions
class DockingContext {
public:
  DockingContext();
  ~DockingContext();

  // Non-copyable
  DockingContext(const DockingContext&) = delete;
  DockingContext& operator=(const DockingContext&) = delete;

  // Root node access
  DockNode* GetRoot();
  const DockNode* GetRoot() const;
  void SetRoot(std::unique_ptr<DockNode> root);

  // Node operations
  DockNode* FindNode(const std::string& id);
  void SplitNode(const std::string& node_id, SplitDirection dir, ViewType new_view);
  void CloseNode(const std::string& node_id);
  void SetNodeViewType(const std::string& node_id, ViewType type);
  void SwapNodes(const std::string& id1, const std::string& id2);

  // Collapse empty nodes and merge single-child splits
  void Cleanup();

  // Drag and drop
  void BeginDrag(const std::string& source_node_id);
  void UpdateDrag(const ImVec2& mouse_pos);
  void EndDrag(const std::string& target_node_id, DropZone zone);
  void CancelDrag();
  bool IsDragging() const;
  const std::string& GetDraggingNodeId() const;
  const DropPreview& GetDropPreview() const;
  void SetDropPreview(const std::string& node_id, DropZone zone,
                      const ImVec2& min, const ImVec2& max);

  // Layout persistence (paths are relative to project directory)
  bool SaveLayout(const std::string& name, const std::filesystem::path& project_dir);
  bool LoadLayout(const std::string& name, const std::filesystem::path& project_dir);
  std::vector<std::string> GetSavedLayoutNames(const std::filesystem::path& project_dir) const;
  bool DeleteLayout(const std::string& name, const std::filesystem::path& project_dir);

  // Layout directory management
  static std::filesystem::path GetLayoutDirectory(const std::filesystem::path& project_dir);
  static std::filesystem::path GetLayoutFilePath(const std::string& name,
                                                  const std::filesystem::path& project_dir);

  // Main render function
  void Render(const ImVec2& position, const ImVec2& available_size, AppServices& services);

  // Get statistics
  size_t GetNodeCount() const;
  size_t GetLeafCount() const;

private:
  std::unique_ptr<DockNode> root_;

  // Drag state
  bool is_dragging_ = false;
  std::string dragging_node_id_;
  DropPreview drop_preview_;

  // Rendering helpers
  void RenderNode(DockNode& node, const ImVec2& pos, const ImVec2& size,
                  AppServices& services);
  void RenderLeafNode(DockNode& node, const ImVec2& pos, const ImVec2& size,
                      AppServices& services);
  void RenderSplitNode(DockNode& node, const ImVec2& pos, const ImVec2& size,
                       AppServices& services);
  void RenderSplitter(DockNode& node, const ImVec2& pos, const ImVec2& size,
                      SplitContent& split);
  void RenderViewHeader(DockNode& node, LeafContent& leaf);
  void RenderDropZones(DockNode& node, const ImVec2& pos, const ImVec2& size);

  // Tree operations
  void CleanupNode(DockNode* parent, std::unique_ptr<DockNode>& node_ptr);
  DockNode* FindParent(const std::string& child_id);
};

#endif // DOCKING_CONTEXT_H
