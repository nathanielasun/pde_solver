#include "ui_helpers.h"

#include "app_state.h"

#include <cmath>

bool DrawValidationIndicator(ValidationStatus status, const std::string& error_msg, 
                             const std::string& warning_msg) {
  ImDrawList* draw_list = ImGui::GetWindowDrawList();
  ImVec2 pos = ImGui::GetCursorScreenPos();
  const float size = 8.0f;
  const float spacing = 4.0f;
  
  ImU32 color;
  std::string tooltip_text;
  
  switch (status) {
    case ValidationStatus::Valid:
      color = IM_COL32(50, 200, 50, 255);  // Green
      break;
    case ValidationStatus::Warning:
      color = IM_COL32(255, 200, 50, 255);  // Yellow
      tooltip_text = warning_msg;
      break;
    case ValidationStatus::Error:
      color = IM_COL32(255, 50, 50, 255);  // Red
      tooltip_text = error_msg;
      break;
  }
  
  // Draw circle
  draw_list->AddCircleFilled(ImVec2(pos.x + size * 0.5f, pos.y + ImGui::GetTextLineHeight() * 0.5f),
                            size * 0.5f, color);
  
  // Move cursor for next widget
  ImGui::SetCursorPosX(ImGui::GetCursorPosX() + size + spacing);
  
  // Return true if tooltip should be shown
  return !tooltip_text.empty() && ImGui::IsItemHovered();
}

void DrawStatusIcon(ValidationStatus status, float size) {
  ImDrawList* draw_list = ImGui::GetWindowDrawList();
  ImVec2 pos = ImGui::GetCursorScreenPos();
  
  ImU32 color;
  switch (status) {
    case ValidationStatus::Valid:
      color = IM_COL32(50, 200, 50, 255);
      break;
    case ValidationStatus::Warning:
      color = IM_COL32(255, 200, 50, 255);
      break;
    case ValidationStatus::Error:
      color = IM_COL32(255, 50, 50, 255);
      break;
  }
  
  draw_list->AddCircleFilled(ImVec2(pos.x + size * 0.5f, pos.y + size * 0.5f),
                            size * 0.5f, color);
}

void DrawValidatedInput(const char* label, ValidationStatus status,
                       const std::string& error_msg, const std::string& warning_msg) {
  ImGui::Text("%s", label);
  ImGui::SameLine();
  DrawValidationIndicator(status, error_msg, warning_msg);
  if (ImGui::IsItemHovered() && (!error_msg.empty() || !warning_msg.empty())) {
    ImGui::BeginTooltip();
    if (!error_msg.empty()) {
      ImGui::TextColored(ImVec4(1.0f, 0.5f, 0.5f, 1.0f), "%s", error_msg.c_str());
    } else if (!warning_msg.empty()) {
      ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.3f, 1.0f), "%s", warning_msg.c_str());
    }
    ImGui::EndTooltip();
  }
}

bool CollapsingHeaderWithMemory(const char* label,
                                const char* id,
                                Preferences& prefs,
                                bool default_open,
                                bool* changed,
                                ImGuiTreeNodeFlags flags) {
  if (changed) {
    *changed = false;
  }

  bool desired_open = default_open;
  if (id && *id) {
    auto it = prefs.ui_section_open.find(id);
    if (it != prefs.ui_section_open.end()) {
      desired_open = it->second;
    }
  }

  // Drive ImGui state from our persisted state.
  ImGui::SetNextItemOpen(desired_open, ImGuiCond_Always);
  const bool now_open = ImGui::CollapsingHeader(label, flags);

  if (id && *id && now_open != desired_open) {
    prefs.ui_section_open[id] = now_open;
    if (changed) {
      *changed = true;
    }
  }
  return now_open;
}

void DrawUnknownComponentPlaceholder(const char* component_id) {
  if (!component_id || !*component_id) {
    ImGui::TextDisabled("Missing component (empty id)");
    return;
  }
  ImGui::TextDisabled("Missing component: %s", component_id);
}
