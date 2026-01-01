#include "dock_node.h"
#include <atomic>
#include <stdexcept>
#include <sstream>

// Thread-safe ID counter
static std::atomic<uint64_t> g_node_id_counter{0};
static std::atomic<uint64_t> g_instance_id_counter{0};

std::string GenerateNodeId() {
  uint64_t id = g_node_id_counter.fetch_add(1);
  std::ostringstream oss;
  oss << "node_" << id;
  return oss.str();
}

std::string GenerateInstanceId(ViewType type) {
  uint64_t id = g_instance_id_counter.fetch_add(1);
  std::ostringstream oss;
  oss << ViewTypeToString(type) << "_" << id;
  return oss.str();
}

// LeafContent implementation
LeafContent::LeafContent(ViewType type)
    : view_type(type), instance_id(GenerateInstanceId(type)), scroll_y(0.0f) {}

// SplitContent implementation
SplitContent::SplitContent(SplitDirection dir, float ratio)
    : direction(dir), split_ratio(ratio), first(nullptr), second(nullptr) {}

SplitContent::SplitContent(SplitContent&& other) noexcept
    : direction(other.direction),
      split_ratio(other.split_ratio),
      first(std::move(other.first)),
      second(std::move(other.second)) {}

SplitContent& SplitContent::operator=(SplitContent&& other) noexcept {
  if (this != &other) {
    direction = other.direction;
    split_ratio = other.split_ratio;
    first = std::move(other.first);
    second = std::move(other.second);
  }
  return *this;
}

// DockNode implementation
DockNode::DockNode() : id(GenerateNodeId()), content(LeafContent{}) {}

DockNode::DockNode(const std::string& node_id)
    : id(node_id), content(LeafContent{}) {}

DockNode::DockNode(DockNode&& other) noexcept
    : id(std::move(other.id)), content(std::move(other.content)) {}

DockNode& DockNode::operator=(DockNode&& other) noexcept {
  if (this != &other) {
    id = std::move(other.id);
    content = std::move(other.content);
  }
  return *this;
}

bool DockNode::IsLeaf() const {
  return std::holds_alternative<LeafContent>(content);
}

bool DockNode::IsSplit() const {
  return std::holds_alternative<SplitContent>(content);
}

LeafContent& DockNode::AsLeaf() {
  if (!IsLeaf()) {
    throw std::runtime_error("DockNode is not a leaf");
  }
  return std::get<LeafContent>(content);
}

const LeafContent& DockNode::AsLeaf() const {
  if (!IsLeaf()) {
    throw std::runtime_error("DockNode is not a leaf");
  }
  return std::get<LeafContent>(content);
}

SplitContent& DockNode::AsSplit() {
  if (!IsSplit()) {
    throw std::runtime_error("DockNode is not a split");
  }
  return std::get<SplitContent>(content);
}

const SplitContent& DockNode::AsSplit() const {
  if (!IsSplit()) {
    throw std::runtime_error("DockNode is not a split");
  }
  return std::get<SplitContent>(content);
}

DockNode* DockNode::FindNode(const std::string& node_id) {
  if (id == node_id) {
    return this;
  }
  if (IsSplit()) {
    auto& split = AsSplit();
    if (split.first) {
      if (DockNode* found = split.first->FindNode(node_id)) {
        return found;
      }
    }
    if (split.second) {
      if (DockNode* found = split.second->FindNode(node_id)) {
        return found;
      }
    }
  }
  return nullptr;
}

const DockNode* DockNode::FindNode(const std::string& node_id) const {
  if (id == node_id) {
    return this;
  }
  if (IsSplit()) {
    const auto& split = AsSplit();
    if (split.first) {
      if (const DockNode* found = split.first->FindNode(node_id)) {
        return found;
      }
    }
    if (split.second) {
      if (const DockNode* found = split.second->FindNode(node_id)) {
        return found;
      }
    }
  }
  return nullptr;
}

DockNode* DockNode::SplitLeaf(SplitDirection dir, ViewType new_view_type) {
  if (!IsLeaf()) {
    return nullptr;  // Can only split leaf nodes
  }

  // Save current leaf content
  LeafContent old_leaf = std::move(AsLeaf());

  // Create new split content
  SplitContent split_content(dir, 0.5f);

  // First child keeps the original content
  split_content.first = std::make_unique<DockNode>();
  split_content.first->content = std::move(old_leaf);

  // Second child gets the new view type
  split_content.second = std::make_unique<DockNode>();
  split_content.second->content = LeafContent(new_view_type);

  // Replace our content with the split
  content = std::move(split_content);

  return AsSplit().second.get();
}

bool DockNode::Close() {
  // Mark for removal - actual removal handled by parent
  if (IsLeaf()) {
    AsLeaf().view_type = ViewType::Empty;
    return true;
  }
  return false;
}

void DockNode::ForEach(std::function<void(DockNode&)> fn) {
  fn(*this);
  if (IsSplit()) {
    auto& split = AsSplit();
    if (split.first) {
      split.first->ForEach(fn);
    }
    if (split.second) {
      split.second->ForEach(fn);
    }
  }
}

void DockNode::ForEachConst(std::function<void(const DockNode&)> fn) const {
  fn(*this);
  if (IsSplit()) {
    const auto& split = AsSplit();
    if (split.first) {
      split.first->ForEachConst(fn);
    }
    if (split.second) {
      split.second->ForEachConst(fn);
    }
  }
}

size_t DockNode::CountNodes() const {
  size_t count = 1;
  if (IsSplit()) {
    const auto& split = AsSplit();
    if (split.first) {
      count += split.first->CountNodes();
    }
    if (split.second) {
      count += split.second->CountNodes();
    }
  }
  return count;
}

size_t DockNode::CountLeaves() const {
  if (IsLeaf()) {
    return 1;
  }
  size_t count = 0;
  const auto& split = AsSplit();
  if (split.first) {
    count += split.first->CountLeaves();
  }
  if (split.second) {
    count += split.second->CountLeaves();
  }
  return count;
}

std::unique_ptr<DockNode> DockNode::Clone() const {
  auto clone = std::make_unique<DockNode>();
  clone->id = GenerateNodeId();  // New ID for clone

  if (IsLeaf()) {
    const auto& leaf = AsLeaf();
    LeafContent new_leaf;
    new_leaf.view_type = leaf.view_type;
    new_leaf.instance_id = GenerateInstanceId(leaf.view_type);
    new_leaf.scroll_y = 0.0f;
    clone->content = std::move(new_leaf);
  } else {
    const auto& split = AsSplit();
    SplitContent new_split;
    new_split.direction = split.direction;
    new_split.split_ratio = split.split_ratio;
    if (split.first) {
      new_split.first = split.first->Clone();
    }
    if (split.second) {
      new_split.second = split.second->Clone();
    }
    clone->content = std::move(new_split);
  }

  return clone;
}

// Factory functions
std::unique_ptr<DockNode> CreateLeafNode(ViewType type) {
  auto node = std::make_unique<DockNode>();
  node->content = LeafContent(type);
  return node;
}

std::unique_ptr<DockNode> CreateSplitNode(SplitDirection dir, float ratio,
                                          std::unique_ptr<DockNode> first,
                                          std::unique_ptr<DockNode> second) {
  auto node = std::make_unique<DockNode>();
  SplitContent split(dir, ratio);
  split.first = std::move(first);
  split.second = std::move(second);
  node->content = std::move(split);
  return node;
}

std::unique_ptr<DockNode> CreateHorizontalSplit(float ratio,
                                                 std::unique_ptr<DockNode> left,
                                                 std::unique_ptr<DockNode> right) {
  return CreateSplitNode(SplitDirection::Horizontal, ratio,
                         std::move(left), std::move(right));
}

std::unique_ptr<DockNode> CreateVerticalSplit(float ratio,
                                               std::unique_ptr<DockNode> top,
                                               std::unique_ptr<DockNode> bottom) {
  return CreateSplitNode(SplitDirection::Vertical, ratio,
                         std::move(top), std::move(bottom));
}
