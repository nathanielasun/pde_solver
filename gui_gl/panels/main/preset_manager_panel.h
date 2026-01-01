#ifndef PRESET_MANAGER_PANEL_H
#define PRESET_MANAGER_PANEL_H

#include <string>
#include <vector>

struct PresetManagerPanelState {
  float input_width;
  std::string& preset_directory;
};

void RenderPresetManagerPanel(PresetManagerPanelState& state,
                               const std::vector<std::string>& components);

#endif  // PRESET_MANAGER_PANEL_H
