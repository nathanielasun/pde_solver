#include "preferences_io.h"

#include "../utils/math_utils.h"
#include "../utils/string_utils.h"
#include "app_state.h"
#include "imgui.h"

#include <filesystem>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>

bool LoadPreferences(const std::filesystem::path& path,
                     Preferences* prefs,
                     std::string* error) {
  if (!prefs) {
    return false;
  }
  std::error_code ec;
  if (!std::filesystem::exists(path, ec)) {
    return true;
  }
  std::ifstream file(path);
  if (!file) {
    if (error) {
      *error = "failed to open prefs file: " + path.string();
    }
    return false;
  }
  std::string line;
  int line_no = 0;
  while (std::getline(file, line)) {
    ++line_no;
    std::string trimmed = Trim(line);
    if (trimmed.empty() || trimmed[0] == '#') {
      continue;
    }
    const size_t eq = trimmed.find('=');
    if (eq == std::string::npos) {
      continue;
    }
    std::string key = ToLower(Trim(trimmed.substr(0, eq)));
    std::string value = Trim(trimmed.substr(eq + 1));
    int parsed = 0;
    if (key == "metal_reduce_interval") {
      if (!ParseIntValue(value, &parsed)) {
        if (error) {
          *error = "invalid metal_reduce_interval at line " + std::to_string(line_no);
        }
        return false;
      }
      prefs->metal_reduce_interval = std::max(0, parsed);
    } else if (key == "metal_tg_x") {
      if (!ParseIntValue(value, &parsed)) {
        if (error) {
          *error = "invalid metal_tg_x at line " + std::to_string(line_no);
        }
        return false;
      }
      prefs->metal_tg_x = std::max(0, parsed);
    } else if (key == "metal_tg_y") {
      if (!ParseIntValue(value, &parsed)) {
        if (error) {
          *error = "invalid metal_tg_y at line " + std::to_string(line_no);
        }
        return false;
      }
      prefs->metal_tg_y = std::max(0, parsed);
    } else if (key == "latex_font_size") {
      if (!ParseIntValue(value, &parsed)) {
        if (error) {
          *error = "invalid latex_font_size at line " + std::to_string(line_no);
        }
        return false;
      }
      prefs->latex_font_size = std::max(8, parsed);
    } else if (key == "method_index") {
      if (!ParseIntValue(value, &parsed)) {
        if (error) {
          *error = "invalid method_index at line " + std::to_string(line_no);
        }
        return false;
      }
      prefs->method_index = std::max(0, std::min(6, parsed));
    } else if (key == "gmres_restart") {
      if (!ParseIntValue(value, &parsed)) {
        if (error) {
          *error = "invalid gmres_restart at line " + std::to_string(line_no);
        }
        return false;
      }
      prefs->gmres_restart = std::max(1, parsed);
    } else if (key == "sor_omega") {
      char* end = nullptr;
      const double omega = std::strtod(value.c_str(), &end);
      if (!end || *end != '\0') {
        if (error) {
          *error = "invalid sor_omega at line " + std::to_string(line_no);
        }
        return false;
      }
      prefs->sor_omega = omega;
    } else if (key == "theme_preset") {
      if (!ParseIntValue(value, &parsed)) {
        if (error) {
          *error = "invalid theme_preset at line " + std::to_string(line_no);
        }
        return false;
      }
      prefs->colors.theme_preset = std::max(0, std::min(3, parsed));
    } else if (key.find("section_open.") == 0) {
      // UI collapsible state: section_open.<id>=0/1 (also accept true/false)
      std::string id = key.substr(std::string("section_open.").size());
      bool open = false;
      if (ParseIntValue(value, &parsed)) {
        open = (parsed != 0);
      } else {
        std::string v = ToLower(Trim(value));
        open = (v == "true" || v == "yes" || v == "on");
      }
      if (!id.empty()) {
        prefs->ui_section_open[id] = open;
      }
    } else if (key.find("color_") == 0) {
      // Parse color values (format: "r,g,b,a" or "r,g,b")
      std::vector<std::string> parts;
      std::string part;
      std::istringstream iss(value);
      while (std::getline(iss, part, ',')) {
        parts.push_back(Trim(part));
      }
      if (parts.size() >= 3) {
        char* end = nullptr;
        float r = std::strtof(parts[0].c_str(), &end);
        if (!end || *end != '\0') continue;
        float g = std::strtof(parts[1].c_str(), &end);
        if (!end || *end != '\0') continue;
        float b = std::strtof(parts[2].c_str(), &end);
        if (!end || *end != '\0') continue;
        float a = 1.0f;
        if (parts.size() >= 4) {
          a = std::strtof(parts[3].c_str(), &end);
          if (!end || *end != '\0') a = 1.0f;
        }
        
        // Map color keys to ColorPreferences members
        if (key == "color_window_bg") {
          prefs->colors.window_bg = ImVec4(r, g, b, a);
        } else if (key == "color_panel_bg") {
          prefs->colors.panel_bg = ImVec4(r, g, b, a);
        } else if (key == "color_input_bg") {
          prefs->colors.input_bg = ImVec4(r, g, b, a);
        } else if (key == "color_text") {
          prefs->colors.text_color = ImVec4(r, g, b, a);
        } else if (key == "color_text_disabled") {
          prefs->colors.text_disabled = ImVec4(r, g, b, a);
        } else if (key == "color_border") {
          prefs->colors.border_color = ImVec4(r, g, b, a);
        } else if (key == "color_accent_primary") {
          prefs->colors.accent_primary = ImVec4(r, g, b, a);
        } else if (key == "color_accent_hover") {
          prefs->colors.accent_hover = ImVec4(r, g, b, a);
        } else if (key == "color_accent_active") {
          prefs->colors.accent_active = ImVec4(r, g, b, a);
        } else if (key == "color_success") {
          prefs->colors.color_success = ImVec4(r, g, b, a);
        } else if (key == "color_warning") {
          prefs->colors.color_warning = ImVec4(r, g, b, a);
        } else if (key == "color_error") {
          prefs->colors.color_error = ImVec4(r, g, b, a);
        } else if (key == "color_info") {
          prefs->colors.color_info = ImVec4(r, g, b, a);
        } else if (key == "color_grid") {
          prefs->colors.grid_color = ImVec4(r, g, b, a);
        } else if (key == "color_axis_label") {
          prefs->colors.axis_label_color = ImVec4(r, g, b, a);
        } else if (key == "color_splitter") {
          prefs->colors.splitter_color = ImVec4(r, g, b, a);
        } else if (key == "color_clear") {
          prefs->colors.clear_color = ImVec4(r, g, b, a);
        } else if (key == "color_latex_text") {
          prefs->colors.latex_text = ImVec4(r, g, b, a);
        } else if (key == "color_latex_bg") {
          prefs->colors.latex_bg = ImVec4(r, g, b, a);
        }
      }
    }
  }
  return true;
}

bool SavePreferences(const std::filesystem::path& path,
                     const Preferences& prefs,
                     std::string* error) {
  std::error_code ec;
  if (path.has_parent_path()) {
    std::filesystem::create_directories(path.parent_path(), ec);
  }
  std::ofstream file(path, std::ios::trunc);
  if (!file) {
    if (error) {
      *error = "failed to write prefs file: " + path.string();
    }
    return false;
  }
  file << "# pde_gui preferences\n";
  file << "metal_reduce_interval=" << std::max(0, prefs.metal_reduce_interval) << "\n";
  file << "metal_tg_x=" << std::max(0, prefs.metal_tg_x) << "\n";
  file << "metal_tg_y=" << std::max(0, prefs.metal_tg_y) << "\n";
  file << "latex_font_size=" << std::max(8, prefs.latex_font_size) << "\n";
  file << "method_index=" << std::max(0, std::min(6, prefs.method_index)) << "\n";
  file << "sor_omega=" << prefs.sor_omega << "\n";
  file << "gmres_restart=" << std::max(1, prefs.gmres_restart) << "\n";

  // Save UI section state
  if (!prefs.ui_section_open.empty()) {
    file << "\n# UI section expanded/collapsed state\n";
    for (const auto& [id, open] : prefs.ui_section_open) {
      file << "section_open." << id << "=" << (open ? 1 : 0) << "\n";
    }
  }
  
  // Save color preferences
  file << "\n# Color preferences\n";
  file << "theme_preset=" << std::max(0, std::min(3, prefs.colors.theme_preset)) << "\n";
  file << "color_window_bg=" << prefs.colors.window_bg.x << "," << prefs.colors.window_bg.y 
       << "," << prefs.colors.window_bg.z << "," << prefs.colors.window_bg.w << "\n";
  file << "color_panel_bg=" << prefs.colors.panel_bg.x << "," << prefs.colors.panel_bg.y 
       << "," << prefs.colors.panel_bg.z << "," << prefs.colors.panel_bg.w << "\n";
  file << "color_input_bg=" << prefs.colors.input_bg.x << "," << prefs.colors.input_bg.y 
       << "," << prefs.colors.input_bg.z << "," << prefs.colors.input_bg.w << "\n";
  file << "color_text=" << prefs.colors.text_color.x << "," << prefs.colors.text_color.y 
       << "," << prefs.colors.text_color.z << "," << prefs.colors.text_color.w << "\n";
  file << "color_text_disabled=" << prefs.colors.text_disabled.x << "," << prefs.colors.text_disabled.y 
       << "," << prefs.colors.text_disabled.z << "," << prefs.colors.text_disabled.w << "\n";
  file << "color_border=" << prefs.colors.border_color.x << "," << prefs.colors.border_color.y 
       << "," << prefs.colors.border_color.z << "," << prefs.colors.border_color.w << "\n";
  file << "color_accent_primary=" << prefs.colors.accent_primary.x << "," << prefs.colors.accent_primary.y 
       << "," << prefs.colors.accent_primary.z << "," << prefs.colors.accent_primary.w << "\n";
  file << "color_accent_hover=" << prefs.colors.accent_hover.x << "," << prefs.colors.accent_hover.y 
       << "," << prefs.colors.accent_hover.z << "," << prefs.colors.accent_hover.w << "\n";
  file << "color_accent_active=" << prefs.colors.accent_active.x << "," << prefs.colors.accent_active.y 
       << "," << prefs.colors.accent_active.z << "," << prefs.colors.accent_active.w << "\n";
  file << "color_success=" << prefs.colors.color_success.x << "," << prefs.colors.color_success.y 
       << "," << prefs.colors.color_success.z << "," << prefs.colors.color_success.w << "\n";
  file << "color_warning=" << prefs.colors.color_warning.x << "," << prefs.colors.color_warning.y 
       << "," << prefs.colors.color_warning.z << "," << prefs.colors.color_warning.w << "\n";
  file << "color_error=" << prefs.colors.color_error.x << "," << prefs.colors.color_error.y 
       << "," << prefs.colors.color_error.z << "," << prefs.colors.color_error.w << "\n";
  file << "color_info=" << prefs.colors.color_info.x << "," << prefs.colors.color_info.y 
       << "," << prefs.colors.color_info.z << "," << prefs.colors.color_info.w << "\n";
  file << "color_grid=" << prefs.colors.grid_color.x << "," << prefs.colors.grid_color.y 
       << "," << prefs.colors.grid_color.z << "," << prefs.colors.grid_color.w << "\n";
  file << "color_axis_label=" << prefs.colors.axis_label_color.x << "," << prefs.colors.axis_label_color.y 
       << "," << prefs.colors.axis_label_color.z << "," << prefs.colors.axis_label_color.w << "\n";
  file << "color_splitter=" << prefs.colors.splitter_color.x << "," << prefs.colors.splitter_color.y 
       << "," << prefs.colors.splitter_color.z << "," << prefs.colors.splitter_color.w << "\n";
  file << "color_clear=" << prefs.colors.clear_color.x << "," << prefs.colors.clear_color.y 
       << "," << prefs.colors.clear_color.z << "," << prefs.colors.clear_color.w << "\n";
  file << "color_latex_text=" << prefs.colors.latex_text.x << "," << prefs.colors.latex_text.y 
       << "," << prefs.colors.latex_text.z << "," << prefs.colors.latex_text.w << "\n";
  file << "color_latex_bg=" << prefs.colors.latex_bg.x << "," << prefs.colors.latex_bg.y 
       << "," << prefs.colors.latex_bg.z << "," << prefs.colors.latex_bg.w << "\n";
  
  return true;
}

