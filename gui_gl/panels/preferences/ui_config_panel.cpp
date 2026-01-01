#include "ui_config_panel.h"

#include "styles/ui_style.h"
#include "utils/file_dialog.h"
#include "imgui.h"
#include "imgui_stdlib.h"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <vector>

namespace {

struct FontEntry {
  std::filesystem::path path;
  std::string label;
};

std::string ToLower(std::string value) {
  for (char& c : value) {
    c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  }
  return value;
}

bool HasFontExtension(const std::filesystem::path& path) {
  const std::string ext = ToLower(path.extension().string());
  return ext == ".ttf" || ext == ".otf" || ext == ".ttc";
}

std::string BuildFontLabel(const std::filesystem::path& root, const std::filesystem::path& path) {
  std::error_code ec;
  std::filesystem::path rel = std::filesystem::relative(path, root, ec);
  if (!ec && !rel.empty()) {
    return rel.generic_string();
  }
  return path.filename().string();
}

std::vector<FontEntry> CollectFontEntries(const std::filesystem::path& root, std::string* error) {
  std::vector<FontEntry> entries;
  if (root.empty()) {
    if (error) {
      *error = "Font directory not set.";
    }
    return entries;
  }

  std::error_code ec;
  if (!std::filesystem::exists(root, ec)) {
    if (error) {
      *error = "Font directory not found.";
    }
    return entries;
  }

  for (std::filesystem::recursive_directory_iterator it(root, ec), end; it != end && !ec; it.increment(ec)) {
    if (!it->is_regular_file(ec)) {
      continue;
    }
    const std::filesystem::path& path = it->path();
    if (!HasFontExtension(path)) {
      continue;
    }
    entries.push_back(FontEntry{path, BuildFontLabel(root, path)});
  }

  if (ec && error) {
    *error = "Font directory scan failed.";
  }

  std::sort(entries.begin(), entries.end(),
            [](const FontEntry& a, const FontEntry& b) { return a.label < b.label; });
  return entries;
}

bool PathsEquivalent(const std::filesystem::path& a, const std::filesystem::path& b) {
  std::error_code ec;
  if (std::filesystem::equivalent(a, b, ec)) {
    return true;
  }
  return a.lexically_normal() == b.lexically_normal();
}

}  // namespace

void RenderUIConfigPanel(UIConfigPanelState& state) {
  if (!state.config_manager) {
    return;
  }

  UIConfig& config = state.config_manager->GetMutableConfig();

  // Track original values to detect changes
  static float original_font_size = config.theme.font_size;
  static float original_input_width = config.theme.input_width;
  static float original_panel_spacing = config.theme.panel_spacing;
  static std::string original_font_path = config.theme.font_path;

  // Detect if settings have changed
  state.settings_changed =
    (original_font_size != config.theme.font_size) ||
    (original_input_width != config.theme.input_width) ||
    (original_panel_spacing != config.theme.panel_spacing) ||
    (original_font_path != config.theme.font_path);

  // Save/Load/Apply buttons
  ImGui::Text("Configuration File");
  ImGui::BeginGroup();

  // Apply Settings button
  ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(12.0f, 8.0f));
  if (state.settings_changed) {
    if (UIButton::Button("Apply Settings", ImVec2(state.input_width, 0),
                         UIButton::Size::Medium, UIButton::Variant::Primary)) {
      // Update original values
      original_font_size = config.theme.font_size;
      original_input_width = config.theme.input_width;
      original_panel_spacing = config.theme.panel_spacing;
      original_font_path = config.theme.font_path;
      state.settings_changed = false;

      // Trigger immediate application
      if (state.on_apply_settings) {
        state.on_apply_settings();
      }

      UIToast::Show(UIToast::Type::Success, "UI settings applied");
    }
  } else {
    ImGui::BeginDisabled();
    UIButton::Button("Apply Settings", ImVec2(state.input_width, 0),
                     UIButton::Size::Medium, UIButton::Variant::Primary);
    ImGui::EndDisabled();
  }
  ImGui::PopStyleVar();

  ImGui::Spacing();

  if (UIButton::Button("Save Configuration", ImVec2(state.input_width * 0.48f, 0),
                       UIButton::Size::Medium, UIButton::Variant::Secondary)) {
    if (state.config_file_path.empty()) {
      UIToast::Show(UIToast::Type::Warning, "UI config path is empty");
    } else if (state.config_manager->SaveToFile(state.config_file_path.string())) {
      UIToast::Show(UIToast::Type::Success, "UI config saved to file");
    } else {
      UIToast::Show(UIToast::Type::Error, "Failed to save UI config");
    }
  }
  ImGui::SameLine();
  if (UIButton::Button("Load Configuration", ImVec2(state.input_width * 0.48f, 0),
                       UIButton::Size::Medium, UIButton::Variant::Secondary)) {
    if (state.config_file_path.empty()) {
      UIToast::Show(UIToast::Type::Warning, "UI config path is empty");
    } else if (!std::filesystem::exists(state.config_file_path)) {
      UIToast::Show(UIToast::Type::Error, "UI config file not found");
    } else if (state.config_manager->LoadFromFile(state.config_file_path.string())) {
      // Update original values after loading
      original_font_size = config.theme.font_size;
      original_input_width = config.theme.input_width;
      original_panel_spacing = config.theme.panel_spacing;
      original_font_path = config.theme.font_path;
      state.settings_changed = false;

      // Trigger immediate application
      if (state.on_apply_settings) {
        state.on_apply_settings();
      }

      UIToast::Show(UIToast::Type::Success, "UI config loaded and applied");
    } else {
      UIToast::Show(UIToast::Type::Error, "Failed to load UI config");
    }
  }
  ImGui::EndGroup();

  // File path
  std::string path_str = state.config_file_path.string();
  ImGui::Text("Path:");
  ImGui::SetNextItemWidth(state.input_width);
  if (UIInput::InputText("##config_path", &path_str)) {
    state.config_file_path = path_str;
  }

  ImGui::Separator();

  // Metadata
  ImGui::Text("Metadata");
  ImGui::TextDisabled("Schema: v%d", config.metadata.schema_version);
  ImGui::TextDisabled("Components: v%d", config.metadata.component_version);
  ImGui::TextDisabled("Config version: %s", config.metadata.config_version.c_str());
  ImGui::TextDisabled("Theme colors are configured in Appearance.");

  ImGui::Separator();

  // Theme settings (sizing only)
  ImGui::Text("Sizing");
  ImGui::SetNextItemWidth(state.input_width);
  if (UIInput::InputFloat("Font Size", &config.theme.font_size, 0.5f, 1.0f, "%.1f")) {
    config.theme.font_size = std::max(8.0f, std::min(72.0f, config.theme.font_size));
  }

  ImGui::SetNextItemWidth(state.input_width);
  if (UIInput::InputFloat("Input Width", &config.theme.input_width, 10.0f, 50.0f, "%.0f")) {
    config.theme.input_width = std::max(100.0f, std::min(2000.0f, config.theme.input_width));
  }

  ImGui::SetNextItemWidth(state.input_width);
  if (UIInput::InputFloat("Panel Spacing", &config.theme.panel_spacing, 1.0f, 5.0f, "%.1f")) {
    config.theme.panel_spacing = std::max(0.0f, std::min(50.0f, config.theme.panel_spacing));
  }

  ImGui::Separator();

  // Typography settings
  ImGui::Text("Typography");
  ImGui::Text("Program Font");
  ImGui::BeginGroup();
  static std::filesystem::path cached_font_dir;
  static std::vector<FontEntry> cached_fonts;
  static std::string cached_error;
  static bool needs_refresh = true;

  if (state.font_dir != cached_font_dir) {
    cached_font_dir = state.font_dir;
    needs_refresh = true;
  }

  ImGui::TextDisabled("Local font directory:");
  if (!state.font_dir.empty()) {
    ImGui::SameLine();
    ImGui::Text("%s", state.font_dir.string().c_str());
  } else {
    ImGui::SameLine();
    ImGui::TextDisabled("Not found.");
  }

  ImGui::SameLine();
  if (UIButton::Button("Refresh##font_list", UIButton::Size::Small, UIButton::Variant::Secondary)) {
    needs_refresh = true;
  }

  if (needs_refresh) {
    cached_error.clear();
    cached_fonts = CollectFontEntries(state.font_dir, &cached_error);
    needs_refresh = false;
  }

  std::string current_label = config.theme.font_path.empty() ? "System Default" : config.theme.font_path;
  if (!config.theme.font_path.empty()) {
    std::filesystem::path current_path(config.theme.font_path);
    if (current_path.is_relative() && !state.font_dir.empty()) {
      current_path = state.font_dir / current_path;
    }
    for (const auto& entry : cached_fonts) {
      if (PathsEquivalent(current_path, entry.path)) {
        current_label = entry.label;
        break;
      }
    }
  }

  ImGui::SetNextItemWidth(state.input_width);
  if (ImGui::BeginCombo("Local Font", current_label.c_str())) {
    if (ImGui::Selectable("System Default", config.theme.font_path.empty())) {
      config.theme.font_path.clear();
    }
    for (const auto& entry : cached_fonts) {
      bool selected = false;
      if (!config.theme.font_path.empty()) {
        std::filesystem::path config_path(config.theme.font_path);
        if (config_path.is_relative() && !state.font_dir.empty()) {
          config_path = state.font_dir / config_path;
        }
        selected = PathsEquivalent(config_path, entry.path);
      }
      if (ImGui::Selectable(entry.label.c_str(), selected)) {
        std::error_code ec;
        std::filesystem::path rel = std::filesystem::relative(entry.path, state.font_dir, ec);
        if (!ec && !rel.empty()) {
          config.theme.font_path = rel.string();
        } else {
          config.theme.font_path = entry.path.string();
        }
      }
    }
    ImGui::EndCombo();
  }

  if (!cached_error.empty()) {
    ImGui::TextColored(ImVec4(1.0f, 0.4f, 0.4f, 1.0f), "%s", cached_error.c_str());
  }

  ImGui::Spacing();
  ImGui::Text("Custom Font Path");
  ImGui::BeginGroup();
  const float font_input_width = std::max(120.0f, state.input_width - 140.0f);
  ImGui::SetNextItemWidth(font_input_width);
  UIInput::InputText("##ui_font_path", &config.theme.font_path);
  ImGui::SameLine();
  if (state.config_manager->IsFeatureEnabled("file_dialogs")) {
    if (UIButton::Button("Browse##ui_font", UIButton::Size::Small, UIButton::Variant::Secondary)) {
      std::filesystem::path default_path = config.theme.font_path.empty() ?
          std::filesystem::current_path() : std::filesystem::path(config.theme.font_path);
      if (default_path.is_relative() && !state.font_dir.empty()) {
        default_path = state.font_dir / default_path;
      }
      auto selected = FileDialog::PickFile(
          "Select Font File",
          default_path,
          "Font Files",
          {"ttf", "otf", "ttc"});
      if (selected) {
        std::error_code rel_ec;
        std::filesystem::path rel = state.font_dir.empty()
            ? std::filesystem::path()
            : std::filesystem::relative(*selected, state.font_dir, rel_ec);
        if (!rel_ec && !rel.empty()) {
          config.theme.font_path = rel.string();
        } else {
          config.theme.font_path = selected->string();
        }
      }
    }
  } else {
    ImGui::BeginDisabled();
    UIButton::Button("Browse##ui_font", UIButton::Size::Small, UIButton::Variant::Secondary);
    ImGui::EndDisabled();
  }
  ImGui::SameLine();
  if (UIButton::Button("Clear##ui_font", UIButton::Size::Small, UIButton::Variant::Secondary)) {
    config.theme.font_path.clear();
  }
  ImGui::EndGroup();
  ImGui::EndGroup();
  ImGui::TextDisabled("Relative paths resolve against the local font directory.");
  std::filesystem::path font_path_check = config.theme.font_path;
  if (font_path_check.is_relative() && !state.font_dir.empty()) {
    font_path_check = state.font_dir / font_path_check;
  }
  if (!config.theme.font_path.empty() && !std::filesystem::exists(font_path_check)) {
    ImGui::TextColored(ImVec4(1.0f, 0.4f, 0.4f, 1.0f), "Font file not found.");
  } else {
    ImGui::TextDisabled("Use a Unicode-capable font for Greek + math symbols.");
  }
  ImGui::TextDisabled("Recommended: Noto Sans, DejaVu Sans, STIX Two Text/Math.");

  ImGui::Separator();

  // Layout settings
  ImGui::Text("Layout Settings");
  ImGui::SetNextItemWidth(state.input_width);
  if (UIInput::InputFloat("Left Panel Min Width", &config.left_panel_min_width, 10.0f, 50.0f, "%.0f")) {
    config.left_panel_min_width = std::max(100.0f, std::min(2000.0f, config.left_panel_min_width));
  }

  ImGui::SetNextItemWidth(state.input_width);
  if (UIInput::InputFloat("Left Panel Max Width", &config.left_panel_max_width, 10.0f, 50.0f, "%.0f")) {
    config.left_panel_max_width = std::max(config.left_panel_min_width + 100.0f,
                                           std::min(5000.0f, config.left_panel_max_width));
  }

  ImGui::SetNextItemWidth(state.input_width);
  if (UIInput::InputFloat("Right Panel Min Width", &config.right_panel_min_width, 10.0f, 50.0f, "%.0f")) {
    config.right_panel_min_width = std::max(100.0f, std::min(2000.0f, config.right_panel_min_width));
  }

  ImGui::SetNextItemWidth(state.input_width);
  if (UIInput::InputFloat("Splitter Width", &config.splitter_width, 1.0f, 5.0f, "%.0f")) {
    config.splitter_width = std::max(4.0f, std::min(50.0f, config.splitter_width));
  }

  ImGui::Separator();

  // Feature flags
  ImGui::Text("Feature Flags");
  if (ImGui::BeginChild("FeatureFlags", ImVec2(0, 200), true)) {
    for (auto& [key, value] : config.feature_flags) {
      ImGui::Checkbox(key.c_str(), &value);
    }
  }
  ImGui::EndChild();

  // Add new feature flag
  static std::string new_flag_name;
  ImGui::Text("Add Feature Flag");
  ImGui::SetNextItemWidth(state.input_width * 0.7f);
  UIInput::InputText("##new_flag", &new_flag_name);
  ImGui::SameLine();
  if (UIButton::Button("Add", ImVec2(state.input_width * 0.28f, 0),
                       UIButton::Size::Small, UIButton::Variant::Secondary)) {
    if (!new_flag_name.empty() && config.feature_flags.find(new_flag_name) == config.feature_flags.end()) {
      config.feature_flags[new_flag_name] = true;
      new_flag_name.clear();
    }
  }

  ImGui::Separator();

  // Tab order
  ImGui::Text("Tab Order");
  if (ImGui::BeginChild("TabOrder", ImVec2(0, 150), true)) {
    for (size_t i = 0; i < config.tab_order.size(); ++i) {
      ImGui::PushID(static_cast<int>(i));
      ImGui::Text("%zu.", i + 1);
      ImGui::SameLine();
      ImGui::SetNextItemWidth(state.input_width - 40);
      std::string tab_name = config.tab_order[i];
      if (UIInput::InputText("##tab", &tab_name)) {
        config.tab_order[i] = tab_name;
      }
      ImGui::SameLine();
      if (i > 0 && UIButton::Button("↑", UIButton::Size::Small, UIButton::Variant::Secondary)) {
        std::swap(config.tab_order[i], config.tab_order[i - 1]);
      }
      ImGui::SameLine();
      if (i < config.tab_order.size() - 1 && UIButton::Button("↓", UIButton::Size::Small, UIButton::Variant::Secondary)) {
        std::swap(config.tab_order[i], config.tab_order[i + 1]);
      }
      ImGui::SameLine();
      if (UIButton::Button("×", UIButton::Size::Small, UIButton::Variant::Danger)) {
        config.tab_order.erase(config.tab_order.begin() + i);
        --i;
      }
      ImGui::PopID();
    }
  }
  ImGui::EndChild();

  // Add new tab
  static std::string new_tab_name;
  ImGui::Text("Add Tab");
  ImGui::SetNextItemWidth(state.input_width * 0.7f);
  UIInput::InputText("##new_tab", &new_tab_name);
  ImGui::SameLine();
  if (UIButton::Button("Add##tab", ImVec2(state.input_width * 0.28f, 0),
                       UIButton::Size::Small, UIButton::Variant::Secondary)) {
    if (!new_tab_name.empty()) {
      config.tab_order.push_back(new_tab_name);
      new_tab_name.clear();
    }
  }

  ImGui::Separator();

  // Validation
  if (UIButton::Button("Validate Configuration", ImVec2(state.input_width, 0),
                       UIButton::Size::Small, UIButton::Variant::Secondary)) {
    state.show_validation = !state.show_validation;
  }

  if (state.show_validation) {
    auto validation = state.config_manager->Validate();
    if (validation.valid) {
      ImGui::TextColored(ImVec4(0.0f, 1.0f, 0.0f, 1.0f), "✓ Configuration is valid");
    } else {
      ImGui::TextColored(ImVec4(1.0f, 0.0f, 0.0f, 1.0f), "✗ Configuration has errors:");
      for (const auto& error : validation.errors) {
        ImGui::BulletText("%s", error.c_str());
      }
    }

    if (!validation.warnings.empty()) {
      ImGui::TextColored(ImVec4(1.0f, 0.7f, 0.0f, 1.0f), "Warnings:");
      for (const auto& warning : validation.warnings) {
        ImGui::BulletText("%s", warning.c_str());
      }
    }
  }

  ImGui::Separator();

  // Panel configuration
  ImGui::Text("Panels");
  if (ImGui::BeginChild("PanelsList", ImVec2(0, 300), true)) {
    for (auto& panel : config.panels) {
      if (ImGui::TreeNode(panel.id.c_str(), "%s (%s)", panel.name.c_str(), panel.id.c_str())) {
        ImGui::SetNextItemWidth(state.input_width);
        UIInput::InputText("Name", &panel.name);
        ImGui::SetNextItemWidth(state.input_width);
        UIInput::InputText("Tab", &panel.tab);
        ImGui::SetNextItemWidth(state.input_width);
        UIInput::InputInt("Order", &panel.order, 1, 1);
        ImGui::Checkbox("Collapsible", &panel.collapsible);
        ImGui::Checkbox("Default Collapsed", &panel.default_collapsed);
        ImGui::SetNextItemWidth(state.input_width);
        UIInput::InputFloat("Width", &panel.width, 10.0f, 50.0f, "%.0f");

        ImGui::Text("Components:");
        for (size_t i = 0; i < panel.components.size(); ++i) {
          ImGui::PushID(static_cast<int>(i));
          ImGui::Text("%zu. %s", i + 1, panel.components[i].c_str());
          ImGui::SameLine();
          if (UIButton::Button("Remove", UIButton::Size::Small, UIButton::Variant::Secondary)) {
            panel.components.erase(panel.components.begin() + i);
            --i;
          }
          ImGui::PopID();
        }

        ImGui::TreePop();
      }
    }
  }
  ImGui::EndChild();
}
