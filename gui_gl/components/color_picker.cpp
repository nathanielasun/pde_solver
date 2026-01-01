#include "color_picker.h"
#include "imgui.h"
#include "imgui_stdlib.h"
#include <algorithm>
#include <sstream>
#include <iomanip>

bool ColorButton(const char* label, const ImVec4& color, ImVec2 size) {
  ImGui::PushID(label);
  ImVec2 button_size = size;
  if (button_size.x == 0) {
    button_size.x = ImGui::GetFrameHeight();
  }
  if (button_size.y == 0) {
    button_size.y = ImGui::GetFrameHeight();
  }
  
  bool clicked = ImGui::ColorButton(label, color, 
                                    ImGuiColorEditFlags_NoTooltip,
                                    button_size);
  ImGui::PopID();
  return clicked;
}

bool ColorPickerCompact(const char* label, ImVec4* color,
                        const ImVec4& default_color) {
  bool changed = false;
  ImGui::PushID(label);
  
  // Color preview button
  ImVec2 button_size(ImGui::GetFrameHeight(), ImGui::GetFrameHeight());
  if (ImGui::ColorButton("##preview", *color, 0, button_size)) {
    ImGui::OpenPopup("##picker");
  }
  
  ImGui::SameLine();
  ImGui::Text("%s", label);
  
  // Color picker popup
  if (ImGui::BeginPopup("##picker")) {
    ImGuiColorEditFlags flags = ImGuiColorEditFlags_None;
    if (ImGui::ColorPicker4("##picker", (float*)color, flags)) {
      changed = true;
    }
    
    // Reset button
    if (ImGui::Button("Reset")) {
      *color = default_color;
      changed = true;
    }
    
    ImGui::EndPopup();
  }
  
  ImGui::PopID();
  return changed;
}

bool ColorPicker(const char* label, ImVec4* color, 
                 const ImVec4& default_color,
                 bool show_alpha,
                 bool show_reset) {
  bool changed = false;
  ImGui::PushID(label);
  
  ImGui::Text("%s", label);
  
  // Color preview button
  ImVec2 button_size(ImGui::GetFrameHeight() * 1.5f, ImGui::GetFrameHeight() * 1.5f);
  if (ImGui::ColorButton("##preview", *color, 0, button_size)) {
    ImGui::OpenPopup("##picker");
  }
  
  ImGui::SameLine();
  
  // RGB inputs
  ImGui::PushItemWidth(60.0f);
  if (ImGui::DragFloat("R##rgb", &color->x, 0.01f, 0.0f, 1.0f, "%.2f")) {
    color->x = std::max(0.0f, std::min(1.0f, color->x));
    changed = true;
  }
  ImGui::SameLine();
  if (ImGui::DragFloat("G##rgb", &color->y, 0.01f, 0.0f, 1.0f, "%.2f")) {
    color->y = std::max(0.0f, std::min(1.0f, color->y));
    changed = true;
  }
  ImGui::SameLine();
  if (ImGui::DragFloat("B##rgb", &color->z, 0.01f, 0.0f, 1.0f, "%.2f")) {
    color->z = std::max(0.0f, std::min(1.0f, color->z));
    changed = true;
  }
  if (show_alpha) {
    ImGui::SameLine();
    if (ImGui::DragFloat("A##rgb", &color->w, 0.01f, 0.0f, 1.0f, "%.2f")) {
      color->w = std::max(0.0f, std::min(1.0f, color->w));
      changed = true;
    }
  }
  ImGui::PopItemWidth();
  
  // Hex input
  ImGui::PushItemWidth(100.0f);
  std::string hex_str = "#";
  std::ostringstream oss;
  oss << std::hex << std::setfill('0') << std::setw(2) 
      << static_cast<int>(color->x * 255)
      << std::setw(2) << static_cast<int>(color->y * 255)
      << std::setw(2) << static_cast<int>(color->z * 255);
  if (show_alpha) {
    oss << std::setw(2) << static_cast<int>(color->w * 255);
  }
  hex_str += oss.str();
  
  // Use ImGui::PushID to create unique ID for each color picker's hex input
  ImGui::PushID("hex_input");
  static std::string hex_input;
  hex_input = hex_str;
  if (ImGui::InputText("Hex##hex", &hex_input)) {
    // Parse hex input
    if (hex_input.size() >= 7 && hex_input[0] == '#') {
      unsigned int hex_val = 0;
      std::istringstream iss(hex_input.substr(1));
      if (iss >> std::hex >> hex_val) {
        if (show_alpha && hex_input.size() >= 9) {
          // RGBA
          color->w = ((hex_val >> 24) & 0xFF) / 255.0f;
          color->z = ((hex_val >> 16) & 0xFF) / 255.0f;
          color->y = ((hex_val >> 8) & 0xFF) / 255.0f;
          color->x = (hex_val & 0xFF) / 255.0f;
        } else {
          // RGB
          color->z = ((hex_val >> 16) & 0xFF) / 255.0f;
          color->y = ((hex_val >> 8) & 0xFF) / 255.0f;
          color->x = (hex_val & 0xFF) / 255.0f;
        }
        changed = true;
      }
    }
  }
  ImGui::PopID();
  ImGui::PopItemWidth();
  
  // Reset button
  if (show_reset) {
    ImGui::SameLine();
    if (ImGui::SmallButton("Reset")) {
      *color = default_color;
      changed = true;
    }
  }
  
  // Color picker popup
  if (ImGui::BeginPopup("##picker")) {
    ImGuiColorEditFlags flags = ImGuiColorEditFlags_None;
    if (!show_alpha) {
      flags |= ImGuiColorEditFlags_NoAlpha;
    }
    if (ImGui::ColorPicker4("##picker", (float*)color, flags)) {
      changed = true;
    }
    ImGui::EndPopup();
  }
  
  ImGui::PopID();
  return changed;
}

