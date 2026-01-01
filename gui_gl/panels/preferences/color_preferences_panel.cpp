#include "color_preferences_panel.h"
#include "components/color_picker.h"
#include "styles/ui_style.h"
#include "imgui.h"
#include <algorithm>

// Get default color preferences (Dark theme)
ColorPreferences GetDefaultColorPreferences() {
  ColorPreferences colors;
  // Defaults are already set in struct definition in app_state.h
  return colors;
}

// Load theme preset
void LoadThemePreset(int preset_index, ColorPreferences& colors) {
  switch (preset_index) {
    case 0: {  // Dark Theme (default)
      colors = GetDefaultColorPreferences();
      break;
    }
    case 1: {  // Light Theme
      colors.window_bg = ImVec4(0.96f, 0.96f, 0.96f, 1.0f);
      colors.panel_bg = ImVec4(1.0f, 1.0f, 1.0f, 1.0f);
      colors.input_bg = ImVec4(0.98f, 0.98f, 0.98f, 1.0f);
      colors.text_color = ImVec4(0.13f, 0.13f, 0.13f, 1.0f);
      colors.text_disabled = ImVec4(0.50f, 0.50f, 0.50f, 1.0f);
      colors.border_color = ImVec4(0.60f, 0.60f, 0.60f, 0.50f);
      colors.accent_primary = ImVec4(0.10f, 0.46f, 0.82f, 1.0f);  // Material Blue Dark
      colors.accent_hover = ImVec4(0.15f, 0.55f, 0.90f, 1.0f);
      colors.accent_active = ImVec4(0.08f, 0.40f, 0.75f, 1.0f);
      colors.color_success = ImVec4(0.18f, 0.49f, 0.20f, 1.0f);  // Material Green Dark
      colors.color_warning = ImVec4(0.96f, 0.49f, 0.0f, 1.0f);   // Material Orange
      colors.color_error = ImVec4(0.78f, 0.16f, 0.16f, 1.0f);   // Material Red Dark
      colors.color_info = ImVec4(0.10f, 0.46f, 0.82f, 1.0f);
      colors.grid_color = ImVec4(0.2f, 0.2f, 0.2f, 0.9f);
      colors.axis_label_color = ImVec4(0.2f, 0.2f, 0.2f, 0.9f);
      colors.splitter_color = ImVec4(0.60f, 0.60f, 0.60f, 0.71f);
      colors.clear_color = ImVec4(0.96f, 0.96f, 0.96f, 1.0f);
      colors.latex_text = ImVec4(0.13f, 0.13f, 0.13f, 1.0f);
      colors.latex_bg = ImVec4(0.98f, 0.98f, 0.98f, 1.0f);
      break;
    }
    case 2: {  // High Contrast Theme
      colors.window_bg = ImVec4(0.0f, 0.0f, 0.0f, 1.0f);
      colors.panel_bg = ImVec4(0.10f, 0.10f, 0.10f, 1.0f);
      colors.input_bg = ImVec4(0.18f, 0.18f, 0.18f, 1.0f);
      colors.text_color = ImVec4(1.0f, 1.0f, 1.0f, 1.0f);
      colors.text_disabled = ImVec4(0.70f, 0.70f, 0.70f, 1.0f);
      colors.border_color = ImVec4(1.0f, 1.0f, 1.0f, 1.0f);
      colors.accent_primary = ImVec4(0.0f, 1.0f, 1.0f, 1.0f);  // Cyan
      colors.accent_hover = ImVec4(0.2f, 1.0f, 1.0f, 1.0f);
      colors.accent_active = ImVec4(0.0f, 0.8f, 0.8f, 1.0f);
      colors.color_success = ImVec4(0.0f, 1.0f, 0.0f, 1.0f);  // Bright Green
      colors.color_warning = ImVec4(1.0f, 1.0f, 0.0f, 1.0f);   // Yellow
      colors.color_error = ImVec4(1.0f, 0.0f, 0.0f, 1.0f);     // Red
      colors.color_info = ImVec4(0.0f, 0.5f, 1.0f, 1.0f);      // Bright Blue
      colors.grid_color = ImVec4(1.0f, 1.0f, 1.0f, 1.0f);
      colors.axis_label_color = ImVec4(1.0f, 1.0f, 1.0f, 1.0f);
      colors.splitter_color = ImVec4(1.0f, 1.0f, 1.0f, 1.0f);
      colors.clear_color = ImVec4(0.0f, 0.0f, 0.0f, 1.0f);
      colors.latex_text = ImVec4(1.0f, 1.0f, 1.0f, 1.0f);
      colors.latex_bg = ImVec4(0.10f, 0.10f, 0.10f, 1.0f);
      break;
    }
    case 3: {  // Custom - don't change, user has customized
      // Keep current values
      break;
    }
  }
}

// Apply color preferences to ImGui style
void ApplyColorPreferences(const ColorPreferences& colors) {
  ImGuiStyle& style = ImGui::GetStyle();
  
  // Core colors
  style.Colors[ImGuiCol_WindowBg] = colors.window_bg;
  style.Colors[ImGuiCol_ChildBg] = colors.panel_bg;
  style.Colors[ImGuiCol_FrameBg] = colors.input_bg;
  style.Colors[ImGuiCol_FrameBgHovered] = ImVec4(
    colors.input_bg.x * 1.1f,
    colors.input_bg.y * 1.1f,
    colors.input_bg.z * 1.1f,
    colors.input_bg.w
  );
  style.Colors[ImGuiCol_FrameBgActive] = ImVec4(
    colors.input_bg.x * 1.2f,
    colors.input_bg.y * 1.2f,
    colors.input_bg.z * 1.2f,
    colors.input_bg.w
  );
  style.Colors[ImGuiCol_Text] = colors.text_color;
  style.Colors[ImGuiCol_TextDisabled] = colors.text_disabled;
  style.Colors[ImGuiCol_Border] = colors.border_color;
  style.Colors[ImGuiCol_BorderShadow] = ImVec4(0.0f, 0.0f, 0.0f, 0.0f);
  
  // Button colors (use accent)
  style.Colors[ImGuiCol_Button] = colors.accent_primary;
  style.Colors[ImGuiCol_ButtonHovered] = colors.accent_hover;
  style.Colors[ImGuiCol_ButtonActive] = colors.accent_active;
  
  // Header colors (collapsible sections)
  style.Colors[ImGuiCol_Header] = ImVec4(
    colors.accent_primary.x * 0.3f,
    colors.accent_primary.y * 0.3f,
    colors.accent_primary.z * 0.3f,
    colors.accent_primary.w * 0.5f
  );
  style.Colors[ImGuiCol_HeaderHovered] = ImVec4(
    colors.accent_primary.x * 0.4f,
    colors.accent_primary.y * 0.4f,
    colors.accent_primary.z * 0.4f,
    colors.accent_primary.w * 0.6f
  );
  style.Colors[ImGuiCol_HeaderActive] = ImVec4(
    colors.accent_primary.x * 0.5f,
    colors.accent_primary.y * 0.5f,
    colors.accent_primary.z * 0.5f,
    colors.accent_primary.w * 0.7f
  );
  
  // Tab colors
  style.Colors[ImGuiCol_Tab] = ImVec4(
    colors.panel_bg.x * 0.8f,
    colors.panel_bg.y * 0.8f,
    colors.panel_bg.z * 0.8f,
    colors.panel_bg.w
  );
  style.Colors[ImGuiCol_TabHovered] = ImVec4(
    colors.accent_primary.x * 0.2f,
    colors.accent_primary.y * 0.2f,
    colors.accent_primary.z * 0.2f,
    colors.accent_primary.w * 0.3f
  );
  style.Colors[ImGuiCol_TabActive] = colors.accent_primary;
  
  // Scrollbar colors
  style.Colors[ImGuiCol_ScrollbarBg] = ImVec4(
    colors.panel_bg.x * 0.9f,
    colors.panel_bg.y * 0.9f,
    colors.panel_bg.z * 0.9f,
    colors.panel_bg.w
  );
  style.Colors[ImGuiCol_ScrollbarGrab] = ImVec4(
    colors.border_color.x,
    colors.border_color.y,
    colors.border_color.z,
    colors.border_color.w * 0.8f
  );
  style.Colors[ImGuiCol_ScrollbarGrabHovered] = colors.accent_primary;
  style.Colors[ImGuiCol_ScrollbarGrabActive] = colors.accent_active;
  
  // Slider colors
  style.Colors[ImGuiCol_SliderGrab] = colors.accent_primary;
  style.Colors[ImGuiCol_SliderGrabActive] = colors.accent_active;
  
  // Checkbox/Radio colors
  style.Colors[ImGuiCol_CheckMark] = colors.accent_primary;
  
  // Plot colors
  style.Colors[ImGuiCol_PlotLines] = colors.accent_primary;
  style.Colors[ImGuiCol_PlotLinesHovered] = colors.accent_hover;
  style.Colors[ImGuiCol_PlotHistogram] = colors.accent_primary;
  style.Colors[ImGuiCol_PlotHistogramHovered] = colors.accent_hover;
  
  // Popup colors
  style.Colors[ImGuiCol_PopupBg] = colors.panel_bg;
  style.Colors[ImGuiCol_ModalWindowDimBg] = ImVec4(0.0f, 0.0f, 0.0f, 0.6f);
}

// Render the color preferences panel
void RenderColorPreferencesPanel(ColorPreferencesPanelState& state) {
  ColorPreferences& colors = state.colors;
  bool& prefs_changed = state.prefs_changed;
  const float input_width = state.input_width;
  
  // Panel is wrapped in collapsible header by caller, just render content
  
  // Theme Presets
  ImGui::Indent(20.0f);  // Indent to show this is a subgroup
  bool theme_expanded = ImGui::CollapsingHeader("Theme Presets");
  if (theme_expanded) {
    const char* theme_items[] = {"Dark", "Light", "High Contrast", "Custom"};
    int old_preset = colors.theme_preset;
    ImGui::SetNextItemWidth(input_width);
    if (UIInput::Combo("Theme", &colors.theme_preset, theme_items, 4)) {
      if (colors.theme_preset != 3) {  // Not Custom
        LoadThemePreset(colors.theme_preset, colors);
        ApplyColorPreferences(colors);
        prefs_changed = true;
      } else {
        // Switching to Custom - keep current colors
        colors.theme_preset = 3;
      }
    }
    
    if (colors.theme_preset != old_preset && colors.theme_preset != 3) {
      ImGui::TextDisabled("Theme applied. Customize individual colors below.");
    }
    
    if (colors.theme_preset == 3) {
      ImGui::TextDisabled("Custom theme active. Modify colors below.");
    }
  }
  ImGui::Unindent(20.0f);
  
  ImGui::Spacing();
  
  // Core Colors
  ImGui::Indent(20.0f);  // Indent to show this is a subgroup
  bool core_expanded = ImGui::CollapsingHeader("Core Colors");
  if (core_expanded) {
    if (ColorPicker("Window Background", &colors.window_bg, 
                    GetDefaultColorPreferences().window_bg)) {
      ApplyColorPreferences(colors);
      colors.theme_preset = 3;  // Switch to Custom
      prefs_changed = true;
    }
    
    if (ColorPicker("Panel Background", &colors.panel_bg,
                    GetDefaultColorPreferences().panel_bg)) {
      ApplyColorPreferences(colors);
      colors.theme_preset = 3;
      prefs_changed = true;
    }
    
    if (ColorPicker("Input Background", &colors.input_bg,
                    GetDefaultColorPreferences().input_bg)) {
      ApplyColorPreferences(colors);
      colors.theme_preset = 3;
      prefs_changed = true;
    }
    
    if (ColorPicker("Text Color", &colors.text_color,
                    GetDefaultColorPreferences().text_color)) {
      ApplyColorPreferences(colors);
      colors.theme_preset = 3;
      prefs_changed = true;
    }
    
    if (ColorPicker("Text Disabled", &colors.text_disabled,
                    GetDefaultColorPreferences().text_disabled)) {
      ApplyColorPreferences(colors);
      colors.theme_preset = 3;
      prefs_changed = true;
    }
    
    if (ColorPicker("Border Color", &colors.border_color,
                    GetDefaultColorPreferences().border_color)) {
      ApplyColorPreferences(colors);
      colors.theme_preset = 3;
      prefs_changed = true;
    }
  }
  ImGui::Unindent(20.0f);
  
  ImGui::Spacing();
  
  // Accent Colors
  ImGui::Indent(20.0f);  // Indent to show this is a subgroup
  bool accent_expanded = ImGui::CollapsingHeader("Accent Colors");
  if (accent_expanded) {
    if (ColorPicker("Primary Accent", &colors.accent_primary,
                    GetDefaultColorPreferences().accent_primary)) {
      ApplyColorPreferences(colors);
      colors.theme_preset = 3;
      prefs_changed = true;
    }
    
    if (ColorPicker("Accent Hover", &colors.accent_hover,
                    GetDefaultColorPreferences().accent_hover)) {
      ApplyColorPreferences(colors);
      colors.theme_preset = 3;
      prefs_changed = true;
    }
    
    if (ColorPicker("Accent Active", &colors.accent_active,
                    GetDefaultColorPreferences().accent_active)) {
      ApplyColorPreferences(colors);
      colors.theme_preset = 3;
      prefs_changed = true;
    }
  }
  ImGui::Unindent(20.0f);
  
  ImGui::Spacing();
  
  // Status Colors
  ImGui::Indent(20.0f);  // Indent to show this is a subgroup
  bool status_expanded = ImGui::CollapsingHeader("Status Colors");
  if (status_expanded) {
    if (ColorPicker("Success", &colors.color_success,
                    GetDefaultColorPreferences().color_success)) {
      colors.theme_preset = 3;
      prefs_changed = true;
    }
    
    if (ColorPicker("Warning", &colors.color_warning,
                    GetDefaultColorPreferences().color_warning)) {
      colors.theme_preset = 3;
      prefs_changed = true;
    }
    
    if (ColorPicker("Error", &colors.color_error,
                    GetDefaultColorPreferences().color_error)) {
      colors.theme_preset = 3;
      prefs_changed = true;
    }
    
    if (ColorPicker("Info", &colors.color_info,
                    GetDefaultColorPreferences().color_info)) {
      colors.theme_preset = 3;
      prefs_changed = true;
    }
  }
  ImGui::Unindent(20.0f);
  
  ImGui::Spacing();
  
  // Visualization Colors
  ImGui::Indent(20.0f);  // Indent to show this is a subgroup
  bool viz_expanded = ImGui::CollapsingHeader("Visualization Colors");
  if (viz_expanded) {
    if (ColorPicker("Grid Color", &colors.grid_color,
                    GetDefaultColorPreferences().grid_color)) {
      colors.theme_preset = 3;
      prefs_changed = true;
    }
    
    if (ColorPicker("Axis Label Color", &colors.axis_label_color,
                    GetDefaultColorPreferences().axis_label_color)) {
      colors.theme_preset = 3;
      prefs_changed = true;
    }
    
    if (ColorPicker("Splitter Color", &colors.splitter_color,
                    GetDefaultColorPreferences().splitter_color)) {
      colors.theme_preset = 3;
      prefs_changed = true;
    }
    
    if (ColorPicker("Background Clear Color", &colors.clear_color,
                    GetDefaultColorPreferences().clear_color)) {
      colors.theme_preset = 3;
      prefs_changed = true;
    }
  }
  ImGui::Unindent(20.0f);
  
  ImGui::Spacing();
  
  // LaTeX Preview Colors
  ImGui::Indent(20.0f);  // Indent to show this is a subgroup
  bool latex_expanded = ImGui::CollapsingHeader("LaTeX Preview Colors");
  if (latex_expanded) {
    if (ColorPicker("LaTeX Text Color", &colors.latex_text,
                    GetDefaultColorPreferences().latex_text)) {
      colors.theme_preset = 3;
      prefs_changed = true;
    }
    
    if (ColorPicker("LaTeX Background", &colors.latex_bg,
                    GetDefaultColorPreferences().latex_bg)) {
      colors.theme_preset = 3;
      prefs_changed = true;
    }
  }
  ImGui::Unindent(20.0f);
}
