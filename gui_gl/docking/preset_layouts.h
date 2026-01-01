#ifndef PRESET_LAYOUTS_H
#define PRESET_LAYOUTS_H

#include "dock_node.h"
#include <memory>

// Factory for creating preset dock layouts
class PresetLayouts {
public:
  // Default layout: Controls on left, Viewer + Timeline on right
  static std::unique_ptr<DockNode> CreateDefault();

  // Inspection layout: Field controls on left, Viewer on right
  static std::unique_ptr<DockNode> CreateInspection();

  // Dual viewer layout: Two 3D viewers side by side
  static std::unique_ptr<DockNode> CreateDualViewer();

  // Minimal layout: Just the 3D viewer
  static std::unique_ptr<DockNode> CreateMinimal();

  // Full configuration layout: All solver settings visible
  static std::unique_ptr<DockNode> CreateFullConfiguration();
};

#endif // PRESET_LAYOUTS_H
