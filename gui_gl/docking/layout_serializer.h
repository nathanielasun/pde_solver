#ifndef LAYOUT_SERIALIZER_H
#define LAYOUT_SERIALIZER_H

#include "dock_node.h"
#include <nlohmann/json.hpp>
#include <string>
#include <memory>

// Serializes dock layouts to/from JSON
class LayoutSerializer {
public:
  // Serialize a dock tree to JSON
  static nlohmann::json SerializeNode(const DockNode& node);

  // Deserialize JSON to a dock tree
  static std::unique_ptr<DockNode> DeserializeNode(const nlohmann::json& j);

  // Save layout to file
  static bool SaveToFile(const DockNode& root, const std::string& file_path);

  // Load layout from file
  static std::unique_ptr<DockNode> LoadFromFile(const std::string& file_path);

private:
  // Serialize split direction
  static std::string DirectionToString(SplitDirection dir);
  static SplitDirection StringToDirection(const std::string& str);
};

#endif // LAYOUT_SERIALIZER_H
