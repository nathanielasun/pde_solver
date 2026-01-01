#ifndef UI_CONFIG_H
#define UI_CONFIG_H

#include <string>
#include <vector>
#include <map>
#include <optional>
#include <set>
#include <nlohmann/json.hpp>

/**
 * UI Configuration System
 * 
 * Allows customization of UI layout, components, themes, and feature flags
 * via JSON configuration files.
 */

struct UIConfigMetadata {
  int schema_version = 1;
  int component_version = 1;
  std::string config_version = "1.0";
};

struct ComponentConfig {
  std::string name;
  std::string type;
  bool enabled = true;
  int version = 1;
  float width = 0.0f;  // 0 = auto
  float height = 0.0f;  // 0 = auto
  std::map<std::string, std::string> properties;  // Component-specific properties
};

struct PanelConfig {
  std::string name;
  std::string id;  // Unique identifier
  std::vector<std::string> components;  // Component IDs in order
  float width = 0.0f;  // 0 = auto
  bool collapsible = true;
  bool default_collapsed = false;
  int order = 0;  // Display order (lower = first)
  std::string tab;  // Which tab this panel belongs to
};

struct ThemeConfig {
  std::string name;
  std::map<std::string, std::string> colors;  // Color name -> hex value
  std::string font_path;  // Optional font file path for Unicode UI text
  float font_size = 13.0f;
  float input_width = 320.0f;
  float panel_spacing = 8.0f;
};

struct UIConfig {
  // Metadata for schema/versioning
  UIConfigMetadata metadata;

  // Panel configurations
  std::vector<PanelConfig> panels;
  
  // Component configurations
  std::map<std::string, ComponentConfig> components;
  
  // Theme configuration
  ThemeConfig theme;
  
  // Feature flags
  std::map<std::string, bool> feature_flags;
  
  // Layout settings
  float left_panel_min_width = 320.0f;
  float left_panel_max_width = 1200.0f;
  float right_panel_min_width = 320.0f;
  float splitter_width = 8.0f;
  
  // Tab order
  std::vector<std::string> tab_order;
  
  // Default values
  UIConfig();
};

struct ValidationResult {
  bool valid = true;
  std::vector<std::string> errors;
  std::vector<std::string> warnings;
};

class UIConfigManager {
 public:
  static UIConfigManager& Instance();
  
  // Load configuration from file
  bool LoadFromFile(const std::string& filepath);
  
  // Load default configuration
  void LoadDefault();
  
  // Save configuration to file
  bool SaveToFile(const std::string& filepath) const;
  
  // Get current configuration
  const UIConfig& GetConfig() const { return config_; }
  UIConfig& GetMutableConfig() { return config_; }
  
  // Get panel configuration by ID
  const PanelConfig* GetPanel(const std::string& panel_id) const;
  PanelConfig* GetMutablePanel(const std::string& panel_id);
  
  // Reorder panel within a tab
  void ReorderPanel(const std::string& tab_name, const std::string& source_id, const std::string& target_id);
  
  // Get component configuration by ID
  const ComponentConfig* GetComponent(const std::string& component_id) const;
  
  // Check feature flag
  bool IsFeatureEnabled(const std::string& feature_name) const;
  
  // Get panels for a specific tab, sorted by order
  std::vector<const PanelConfig*> GetPanelsForTab(const std::string& tab_name) const;
  
  // Validate configuration
  ValidationResult Validate() const;
  
  // Get available theme names
  std::vector<std::string> GetAvailableThemes() const;
  
  // Apply a theme by name
  bool ApplyTheme(const std::string& theme_name);

  // Ensure all default panels exist (adds missing ones)
  void EnsureDefaultPanels();

 private:
  UIConfigManager() = default;
  ~UIConfigManager() = default;
  UIConfigManager(const UIConfigManager&) = delete;
  UIConfigManager& operator=(const UIConfigManager&) = delete;
  
  UIConfig config_;
  bool loaded_ = false;
  
  // Parse JSON (using simple string-based parser for now, or nlohmann/json)
  bool ParseJSON(const std::string& json_content);
  std::string ToJSON() const;
};

#endif  // UI_CONFIG_H
