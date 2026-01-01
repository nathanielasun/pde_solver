#ifndef DOCK_NODE_H
#define DOCK_NODE_H

#include "view_types.h"
#include <memory>
#include <string>
#include <variant>
#include <vector>
#include <functional>

// Forward declaration
struct DockNode;

// Split direction for container nodes
enum class SplitDirection {
  None,        // Leaf node (no split)
  Horizontal,  // Left | Right
  Vertical,    // Top | Bottom
};

// Content for a leaf node (displays a single view)
struct LeafContent {
  ViewType view_type = ViewType::Empty;
  std::string instance_id;  // Unique ID for this view instance
  float scroll_y = 0.0f;    // Per-view scroll state

  LeafContent() = default;
  explicit LeafContent(ViewType type);
};

// Content for a split container node
struct SplitContent {
  SplitDirection direction = SplitDirection::Horizontal;
  float split_ratio = 0.5f;  // 0.0 to 1.0 (position of splitter)
  std::unique_ptr<DockNode> first;   // First child (left or top)
  std::unique_ptr<DockNode> second;  // Second child (right or bottom)

  SplitContent() = default;
  SplitContent(SplitDirection dir, float ratio);
  SplitContent(SplitContent&& other) noexcept;
  SplitContent& operator=(SplitContent&& other) noexcept;

  // No copy (unique_ptr members)
  SplitContent(const SplitContent&) = delete;
  SplitContent& operator=(const SplitContent&) = delete;
};

// Unified dock node - either a leaf or a split container
struct DockNode {
  std::string id;  // Unique node ID (for serialization and lookup)
  std::variant<LeafContent, SplitContent> content;

  DockNode();
  explicit DockNode(const std::string& node_id);

  // Move only (due to unique_ptr in SplitContent)
  DockNode(DockNode&& other) noexcept;
  DockNode& operator=(DockNode&& other) noexcept;
  DockNode(const DockNode&) = delete;
  DockNode& operator=(const DockNode&) = delete;

  // Type checks
  bool IsLeaf() const;
  bool IsSplit() const;

  // Safe accessors (throw if wrong type)
  LeafContent& AsLeaf();
  const LeafContent& AsLeaf() const;
  SplitContent& AsSplit();
  const SplitContent& AsSplit() const;

  // Tree operations
  DockNode* FindNode(const std::string& node_id);
  const DockNode* FindNode(const std::string& node_id) const;

  // Split this leaf node in the given direction
  // Returns pointer to the new node created in the split
  DockNode* SplitLeaf(SplitDirection dir, ViewType new_view_type);

  // Close this node (called from parent to remove child)
  // Returns true if the node was successfully marked for removal
  bool Close();

  // Traverse all nodes
  void ForEach(std::function<void(DockNode&)> fn);
  void ForEachConst(std::function<void(const DockNode&)> fn) const;

  // Count total nodes in subtree
  size_t CountNodes() const;

  // Count leaf nodes in subtree
  size_t CountLeaves() const;

  // Deep clone the node tree
  std::unique_ptr<DockNode> Clone() const;
};

// Helper to generate unique node IDs
std::string GenerateNodeId();

// Helper to generate unique instance IDs for views
std::string GenerateInstanceId(ViewType type);

// Factory functions for creating nodes
std::unique_ptr<DockNode> CreateLeafNode(ViewType type);
std::unique_ptr<DockNode> CreateSplitNode(SplitDirection dir, float ratio,
                                          std::unique_ptr<DockNode> first,
                                          std::unique_ptr<DockNode> second);

// Convenience factories
std::unique_ptr<DockNode> CreateHorizontalSplit(float ratio,
                                                 std::unique_ptr<DockNode> left,
                                                 std::unique_ptr<DockNode> right);
std::unique_ptr<DockNode> CreateVerticalSplit(float ratio,
                                               std::unique_ptr<DockNode> top,
                                               std::unique_ptr<DockNode> bottom);

#endif // DOCK_NODE_H
