#ifndef COLOR_PICKER_H
#define COLOR_PICKER_H

#include "imgui.h"
#include <string>

// Color picker widget for ImGui
// Returns true if color was changed
bool ColorPicker(const char* label, ImVec4* color, 
                 const ImVec4& default_color = ImVec4(0, 0, 0, 1),
                 bool show_alpha = true,
                 bool show_reset = true);

// Compact color picker (smaller, inline version)
bool ColorPickerCompact(const char* label, ImVec4* color,
                        const ImVec4& default_color = ImVec4(0, 0, 0, 1));

// Color preview button (shows color, opens picker on click)
bool ColorButton(const char* label, const ImVec4& color, 
                 ImVec2 size = ImVec2(0, 0));

#endif // COLOR_PICKER_H

