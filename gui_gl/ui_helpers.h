#ifndef UI_HELPERS_H
#define UI_HELPERS_H

#include "imgui.h"
#include "validation.h"
#include <string>

struct Preferences;

// Draw validation indicator icon next to a widget
// Returns true if tooltip should be shown
bool DrawValidationIndicator(ValidationStatus status, const std::string& error_msg = "", 
                             const std::string& warning_msg = "");

// Helper to draw input with validation indicator
void DrawValidatedInput(const char* label, ValidationStatus status, 
                       const std::string& error_msg, const std::string& warning_msg);

// Draw status icon (circle) with color based on validation status
void DrawStatusIcon(ValidationStatus status, float size = 8.0f);

// Collapsing header that remembers open/closed state across sessions using Preferences.
// - id: stable key (do not use label if label may change)
// - changed: set to true if the stored state was updated this frame
bool CollapsingHeaderWithMemory(const char* label,
                                const char* id,
                                Preferences& prefs,
                                bool default_open,
                                bool* changed = nullptr,
                                ImGuiTreeNodeFlags flags = 0);

// Draw a lightweight placeholder for unknown component IDs.
void DrawUnknownComponentPlaceholder(const char* component_id);

#endif  // UI_HELPERS_H
