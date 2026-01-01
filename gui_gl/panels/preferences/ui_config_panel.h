#ifndef UI_CONFIG_PANEL_H
#define UI_CONFIG_PANEL_H

#include "systems/ui_config.h"
#include <string>
#include <filesystem>
#include <functional>

struct UIConfigPanelState {
  UIConfigManager* config_manager;
  std::filesystem::path config_file_path;
  std::filesystem::path font_dir;
  bool show_validation = false;
  float input_width = 320.0f;
  bool settings_changed = false;

  // Callback to apply settings immediately
  std::function<void()> on_apply_settings;
};

void RenderUIConfigPanel(UIConfigPanelState& state);

#endif  // UI_CONFIG_PANEL_H
