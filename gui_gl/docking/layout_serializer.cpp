#include "layout_serializer.h"
#include <fstream>
#include <iostream>

nlohmann::json LayoutSerializer::SerializeNode(const DockNode& node) {
  nlohmann::json j;
  j["id"] = node.id;

  if (node.IsLeaf()) {
    const auto& leaf = node.AsLeaf();
    j["type"] = "leaf";
    j["view_type"] = ViewTypeToString(leaf.view_type);
    j["instance_id"] = leaf.instance_id;
    j["scroll_y"] = leaf.scroll_y;
  } else {
    const auto& split = node.AsSplit();
    j["type"] = "split";
    j["direction"] = DirectionToString(split.direction);
    j["ratio"] = split.split_ratio;

    if (split.first) {
      j["first"] = SerializeNode(*split.first);
    }
    if (split.second) {
      j["second"] = SerializeNode(*split.second);
    }
  }

  return j;
}

std::unique_ptr<DockNode> LayoutSerializer::DeserializeNode(const nlohmann::json& j) {
  if (j.empty()) return nullptr;

  auto node = std::make_unique<DockNode>();

  if (j.contains("id")) {
    node->id = j["id"].get<std::string>();
  } else {
    node->id = GenerateNodeId();
  }

  std::string type = j.value("type", "leaf");

  if (type == "leaf") {
    LeafContent leaf;

    if (j.contains("view_type")) {
      leaf.view_type = StringToViewType(j["view_type"].get<std::string>());
    }
    if (j.contains("instance_id")) {
      leaf.instance_id = j["instance_id"].get<std::string>();
    } else {
      leaf.instance_id = GenerateInstanceId(leaf.view_type);
    }
    if (j.contains("scroll_y")) {
      leaf.scroll_y = j["scroll_y"].get<float>();
    }

    node->content = std::move(leaf);
  } else if (type == "split") {
    SplitContent split;

    if (j.contains("direction")) {
      split.direction = StringToDirection(j["direction"].get<std::string>());
    }
    if (j.contains("ratio")) {
      split.split_ratio = j["ratio"].get<float>();
    }
    if (j.contains("first")) {
      split.first = DeserializeNode(j["first"]);
    }
    if (j.contains("second")) {
      split.second = DeserializeNode(j["second"]);
    }

    node->content = std::move(split);
  }

  return node;
}

bool LayoutSerializer::SaveToFile(const DockNode& root, const std::string& file_path) {
  try {
    nlohmann::json j;
    j["version"] = 1;
    j["root"] = SerializeNode(root);

    std::ofstream file(file_path);
    if (!file.is_open()) {
      std::cerr << "Failed to open layout file for writing: " << file_path << std::endl;
      return false;
    }

    file << j.dump(2);
    return true;
  } catch (const std::exception& e) {
    std::cerr << "Failed to save layout: " << e.what() << std::endl;
    return false;
  }
}

std::unique_ptr<DockNode> LayoutSerializer::LoadFromFile(const std::string& file_path) {
  try {
    std::ifstream file(file_path);
    if (!file.is_open()) {
      std::cerr << "Failed to open layout file for reading: " << file_path << std::endl;
      return nullptr;
    }

    nlohmann::json j;
    file >> j;

    if (!j.contains("root")) {
      std::cerr << "Layout file missing root node: " << file_path << std::endl;
      return nullptr;
    }

    return DeserializeNode(j["root"]);
  } catch (const std::exception& e) {
    std::cerr << "Failed to load layout: " << e.what() << std::endl;
    return nullptr;
  }
}

std::string LayoutSerializer::DirectionToString(SplitDirection dir) {
  switch (dir) {
    case SplitDirection::Horizontal: return "horizontal";
    case SplitDirection::Vertical: return "vertical";
    default: return "none";
  }
}

SplitDirection LayoutSerializer::StringToDirection(const std::string& str) {
  if (str == "horizontal" || str == "h") return SplitDirection::Horizontal;
  if (str == "vertical" || str == "v") return SplitDirection::Vertical;
  return SplitDirection::None;
}
