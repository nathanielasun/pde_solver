#ifndef COLOR_PREFERENCES_PANEL_H
#define COLOR_PREFERENCES_PANEL_H

#include "app_state.h"
#include <string>

// State structure for color preferences panel
struct ColorPreferencesPanelState {
  ColorPreferences& colors;
  bool& prefs_changed;  // Set to true when colors change
  float input_width;
};

// Render the color preferences panel
void RenderColorPreferencesPanel(ColorPreferencesPanelState& state);

// Apply color preferences to ImGui style
void ApplyColorPreferences(const ColorPreferences& colors);

// Load theme preset into color preferences
void LoadThemePreset(int preset_index, ColorPreferences& colors);

// Get default color preferences (Dark theme)
ColorPreferences GetDefaultColorPreferences();

#endif // COLOR_PREFERENCES_PANEL_H

