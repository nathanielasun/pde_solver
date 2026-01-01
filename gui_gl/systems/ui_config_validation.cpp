#include "ui_config.h"
#include <filesystem>
#include <set>

ValidationResult UIConfigManager::Validate() const {
  ValidationResult result;

  constexpr int kSchemaVersion = 1;
  if (config_.metadata.schema_version != kSchemaVersion) {
    result.warnings.push_back("Unsupported UI config schema_version: " +
                              std::to_string(config_.metadata.schema_version));
  }
  if (config_.metadata.component_version < 1) {
    result.warnings.push_back("Invalid component_version in metadata");
  }
  
  // Validate panels
  std::set<std::string> panel_ids;
  std::set<std::string> tab_order_set;
  for (const auto& tab : config_.tab_order) {
    tab_order_set.insert(tab);
  }
  for (const auto& panel : config_.panels) {
    if (panel.id.empty()) {
      result.errors.push_back("Panel has empty ID");
      result.valid = false;
    } else if (panel_ids.count(panel.id) > 0) {
      result.errors.push_back("Duplicate panel ID: " + panel.id);
      result.valid = false;
    } else {
      panel_ids.insert(panel.id);
    }
    
    if (panel.name.empty()) {
      result.warnings.push_back("Panel '" + panel.id + "' has empty name");
    }

    if (!panel.tab.empty() && tab_order_set.count(panel.tab) == 0) {
      result.warnings.push_back("Panel '" + panel.id + "' references unknown tab: " + panel.tab);
    }
    
    // Validate components referenced by panel
    for (const auto& comp_id : panel.components) {
      if (config_.components.find(comp_id) == config_.components.end()) {
        result.warnings.push_back("Panel '" + panel.id + "' references unknown component: " + comp_id);
      }
    }
  }
  
  // Validate tab order
  std::set<std::string> tab_seen;
  for (const auto& tab : config_.tab_order) {
    if (tab_seen.count(tab) > 0) {
      result.warnings.push_back("Duplicate tab in tab_order: " + tab);
    }
    tab_seen.insert(tab);
  }

  // Validate component metadata
  for (const auto& [id, comp] : config_.components) {
    if (comp.version < 1) {
      result.warnings.push_back("Component '" + id + "' has invalid version");
    }
  }
  
  // Validate theme
  if (config_.theme.font_size <= 0 || config_.theme.font_size > 100) {
    result.warnings.push_back("Theme font_size is out of reasonable range");
  }
  if (!config_.theme.font_path.empty()) {
    std::filesystem::path font_path(config_.theme.font_path);
    if (font_path.is_absolute() && !std::filesystem::exists(font_path)) {
      result.warnings.push_back("Theme font_path does not exist: " + config_.theme.font_path);
    }
  }
  if (config_.theme.input_width <= 0 || config_.theme.input_width > 5000) {
    result.warnings.push_back("Theme input_width is out of reasonable range");
  }
  
  // Validate layout
  if (config_.left_panel_min_width >= config_.left_panel_max_width) {
    result.errors.push_back("left_panel_min_width must be < left_panel_max_width");
    result.valid = false;
  }
  if (config_.left_panel_min_width <= 0 || config_.right_panel_min_width <= 0) {
    result.errors.push_back("Panel min widths must be > 0");
    result.valid = false;
  }
  
  return result;
}

std::vector<std::string> UIConfigManager::GetAvailableThemes() const {
  // Built-in themes
  return {"Dark", "Light", "Classic", "Custom"};
}

bool UIConfigManager::ApplyTheme(const std::string& theme_name) {
  if (theme_name == "Dark") {
    config_.theme.name = "Dark";
    config_.theme.colors["window_bg"] = "#262626";
    config_.theme.colors["panel_bg"] = "#333333";
    config_.theme.colors["input_bg"] = "#404040";
    config_.theme.colors["text_color"] = "#FFFFFF";
    config_.theme.colors["accent_primary"] = "#4299FF";
    return true;
  } else if (theme_name == "Light") {
    config_.theme.name = "Light";
    config_.theme.colors["window_bg"] = "#F5F5F5";
    config_.theme.colors["panel_bg"] = "#FFFFFF";
    config_.theme.colors["input_bg"] = "#E0E0E0";
    config_.theme.colors["text_color"] = "#000000";
    config_.theme.colors["accent_primary"] = "#0066CC";
    return true;
  } else if (theme_name == "Classic") {
    config_.theme.name = "Classic";
    config_.theme.colors["window_bg"] = "#3F3F3F";
    config_.theme.colors["panel_bg"] = "#464646";
    config_.theme.colors["input_bg"] = "#505050";
    config_.theme.colors["text_color"] = "#FFFFFF";
    config_.theme.colors["accent_primary"] = "#4A9EFF";
    return true;
  }
  return false;  // Unknown theme or "Custom"
}
