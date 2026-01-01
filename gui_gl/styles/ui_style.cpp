#include "ui_style.h"
#include "imgui.h"
#include "imgui_stdlib.h"
#include "imgui_internal.h"
#include <algorithm>
#include <vector>
#include <chrono>
#include <map>
#include <functional>
#include <cmath>
#include <cstring>
#include <string>

// ============================================================================
// BUTTON STYLING IMPLEMENTATION
// ============================================================================

namespace UIButton {
  static int style_stack_depth = 0;
  
  void PushStyle(Size size, Variant variant) {
    ImGuiStyle& style = ImGui::GetStyle();
    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(
      UISpacing::ButtonPaddingX,
      size == Size::Small ? 6.0f : (size == Size::Large ? 14.0f : UISpacing::ButtonPaddingY)
    ));
    
    ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 6.0f);
    
    // Set button colors based on variant
    ImVec4 button_color;
    ImVec4 button_hover;
    ImVec4 button_active;
    ImVec4 text_color = ImGui::GetStyleColorVec4(ImGuiCol_Text);
    
    switch (variant) {
      case Variant::Primary: {
        // Use accent color from style
        button_color = ImGui::GetStyleColorVec4(ImGuiCol_Button);
        button_hover = ImGui::GetStyleColorVec4(ImGuiCol_ButtonHovered);
        button_active = ImGui::GetStyleColorVec4(ImGuiCol_ButtonActive);
        break;
      }
      case Variant::Secondary: {
        // Outlined style
        button_color = ImVec4(0.0f, 0.0f, 0.0f, 0.0f); // Transparent
        button_hover = ImVec4(0.2f, 0.2f, 0.2f, 0.5f);
        button_active = ImVec4(0.3f, 0.3f, 0.3f, 0.7f);
        ImGui::PushStyleColor(ImGuiCol_Border, ImGui::GetStyleColorVec4(ImGuiCol_Button));
        ImGui::PushStyleVar(ImGuiStyleVar_FrameBorderSize, 1.0f);
        break;
      }
      case Variant::Tertiary: {
        // Text-only
        button_color = ImVec4(0.0f, 0.0f, 0.0f, 0.0f);
        button_hover = ImVec4(0.2f, 0.2f, 0.2f, 0.3f);
        button_active = ImVec4(0.3f, 0.3f, 0.3f, 0.5f);
        break;
      }
      case Variant::Danger: {
        // Red variant
        button_color = ImVec4(0.8f, 0.2f, 0.2f, 1.0f);
        button_hover = ImVec4(0.9f, 0.3f, 0.3f, 1.0f);
        button_active = ImVec4(0.7f, 0.15f, 0.15f, 1.0f);
        break;
      }
    }
    
    ImGui::PushStyleColor(ImGuiCol_Button, button_color);
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, button_hover);
    ImGui::PushStyleColor(ImGuiCol_ButtonActive, button_active);
    
    if (variant == Variant::Secondary) {
      style_stack_depth += 2; // Border style var + colors
    } else {
      style_stack_depth += 1; // Just colors
    }
  }
  
  void PopStyle() {
    ImGui::PopStyleColor(3); // Button, Hovered, Active
    ImGui::PopStyleVar(2);   // Padding, Rounding
    
    if (style_stack_depth > 1) {
      ImGui::PopStyleVar(1); // Border size
      ImGui::PopStyleColor(1); // Border color
      style_stack_depth -= 2;
    } else {
      style_stack_depth -= 1;
    }
  }
  
  bool Button(const char* label, Size size, Variant variant) {
    PushStyle(size, variant);
    bool result = ImGui::Button(label);
    PopStyle();
    return result;
  }
  
  bool Button(const char* label, const ImVec2& size, Size button_size, Variant variant) {
    PushStyle(button_size, variant);
    bool result = ImGui::Button(label, size);
    PopStyle();
    return result;
  }
}

// ============================================================================
// INPUT FIELD STYLING IMPLEMENTATION
// ============================================================================

namespace UIInput {
  static int style_stack_depth = 0;
  
  void PushStyle(ValidationState state) {
    ImGuiStyle& style = ImGui::GetStyle();
    
    // Apply padding
    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(
      UISpacing::InputPaddingX,
      UISpacing::InputPaddingY
    ));
    
    // Apply rounding
    ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 4.0f);
    
    // Apply validation colors
    if (state != ValidationState::None) {
      ImVec4 frame_bg = ImGui::GetStyleColorVec4(ImGuiCol_FrameBg);
      ImVec4 frame_bg_hover = ImGui::GetStyleColorVec4(ImGuiCol_FrameBgHovered);
      ImVec4 frame_bg_active = ImGui::GetStyleColorVec4(ImGuiCol_FrameBgActive);
      ImVec4 border_color = ImGui::GetStyleColorVec4(ImGuiCol_Border);
      
      switch (state) {
        case ValidationState::Valid:
          // Green tint
          frame_bg = ImVec4(0.2f, 0.6f, 0.3f, frame_bg.w);
          frame_bg_hover = ImVec4(0.25f, 0.65f, 0.35f, frame_bg_hover.w);
          frame_bg_active = ImVec4(0.3f, 0.7f, 0.4f, frame_bg_active.w);
          border_color = ImVec4(0.3f, 0.7f, 0.4f, 1.0f);
          break;
        case ValidationState::Warning:
          // Yellow tint
          frame_bg = ImVec4(0.7f, 0.6f, 0.2f, frame_bg.w);
          frame_bg_hover = ImVec4(0.75f, 0.65f, 0.25f, frame_bg_hover.w);
          frame_bg_active = ImVec4(0.8f, 0.7f, 0.3f, frame_bg_active.w);
          border_color = ImVec4(0.9f, 0.7f, 0.2f, 1.0f);
          break;
        case ValidationState::Error:
          // Red tint
          frame_bg = ImVec4(0.6f, 0.2f, 0.2f, frame_bg.w);
          frame_bg_hover = ImVec4(0.65f, 0.25f, 0.25f, frame_bg_hover.w);
          frame_bg_active = ImVec4(0.7f, 0.3f, 0.3f, frame_bg_active.w);
          border_color = ImVec4(0.9f, 0.3f, 0.3f, 1.0f);
          break;
        case ValidationState::None:
          break;
      }
      
      ImGui::PushStyleColor(ImGuiCol_FrameBg, frame_bg);
      ImGui::PushStyleColor(ImGuiCol_FrameBgHovered, frame_bg_hover);
      ImGui::PushStyleColor(ImGuiCol_FrameBgActive, frame_bg_active);
      ImGui::PushStyleColor(ImGuiCol_Border, border_color);
      ImGui::PushStyleVar(ImGuiStyleVar_FrameBorderSize, 1.5f);
      style_stack_depth = 5;
    } else {
      style_stack_depth = 2;
    }
  }
  
  void PopStyle() {
    if (style_stack_depth == 5) {
      ImGui::PopStyleVar(1); // Border size
      ImGui::PopStyleColor(4); // Frame colors + border
    }
    ImGui::PopStyleVar(2); // Padding + rounding
  }
  
  bool InputText(const char* label, std::string* str, ImGuiInputTextFlags flags) {
    PushStyle(ValidationState::None);
    bool result = ImGui::InputText(label, str, flags);
    PopStyle();
    return result;
  }

  bool InputTextMultiline(const char* label, std::string* str, const ImVec2& size,
                          ImGuiInputTextFlags flags) {
    PushStyle(ValidationState::None);
    bool result = ImGui::InputTextMultiline(label, str, size, flags);
    PopStyle();
    return result;
  }

  bool InputFloat(const char* label, float* v, float step, float step_fast, const char* format) {
    PushStyle(ValidationState::None);
    bool result = ImGui::InputFloat(label, v, step, step_fast, format);
    PopStyle();
    return result;
  }
  
  bool InputDouble(const char* label, double* v, double step, double step_fast, const char* format) {
    PushStyle(ValidationState::None);
    bool result = ImGui::InputDouble(label, v, step, step_fast, format);
    PopStyle();
    return result;
  }
  
  bool InputInt(const char* label, int* v, int step, int step_fast) {
    PushStyle(ValidationState::None);
    bool result = ImGui::InputInt(label, v, step, step_fast);
    PopStyle();
    return result;
  }

  bool InputInt2(const char* label, int v[2]) {
    PushStyle(ValidationState::None);
    bool result = ImGui::InputInt2(label, v);
    PopStyle();
    return result;
  }
  
  bool SliderFloat(const char* label, float* v, float v_min, float v_max, const char* format) {
    PushStyle(ValidationState::None);
    bool result = ImGui::SliderFloat(label, v, v_min, v_max, format);
    PopStyle();
    return result;
  }
  
  bool SliderInt(const char* label, int* v, int v_min, int v_max) {
    PushStyle(ValidationState::None);
    bool result = ImGui::SliderInt(label, v, v_min, v_max);
    PopStyle();
    return result;
  }

  bool Combo(const char* label, int* current_item, const char* const items[], int items_count) {
    PushStyle(ValidationState::None);
    bool result = ImGui::Combo(label, current_item, items, items_count);
    PopStyle();
    return result;
  }
}

// ============================================================================
// TOAST NOTIFICATION IMPLEMENTATION
// ============================================================================

namespace UIToast {
  static std::vector<Toast> toasts;
  static int next_id = 1;
  static bool initialized = false;
  
  void Initialize() {
    if (!initialized) {
      toasts.clear();
      next_id = 1;
      initialized = true;
    }
  }
  
  void Show(Type type, const std::string& message, float duration) {
    Toast toast;
    toast.message = message;
    toast.type = type;
    toast.duration = duration > 0.0f ? duration : 3.0f;
    toast.elapsed = 0.0f;
    toast.id = next_id++;
    toasts.push_back(toast);
  }
  
  void Render() {
    if (toasts.empty()) return;
    
    ImGuiIO& io = ImGui::GetIO();
    float delta_time = io.DeltaTime;
    
    const float toast_width = 350.0f;
    const float toast_padding = 16.0f;
    const float toast_spacing = 8.0f;
    const float start_y = 20.0f;
    const float start_x = io.DisplaySize.x - toast_width - 20.0f;
    
    float current_y = start_y;
    
    // Update and render toasts
    for (auto it = toasts.begin(); it != toasts.end();) {
      Toast& toast = *it;
      toast.elapsed += delta_time;
      
      // Remove expired toasts
      if (toast.elapsed >= toast.duration) {
        it = toasts.erase(it);
        continue;
      }
      
      // Calculate fade (fade out in last 0.3 seconds)
      float alpha = 1.0f;
      if (toast.elapsed > toast.duration - 0.3f) {
        alpha = 1.0f - ((toast.elapsed - (toast.duration - 0.3f)) / 0.3f);
      }
      
      // Set position
      ImGui::SetNextWindowPos(ImVec2(start_x, current_y), ImGuiCond_Always);
      ImGui::SetNextWindowSize(ImVec2(toast_width, 0.0f), ImGuiCond_Always);
      
      // Set colors based on type
      ImVec4 bg_color;
      ImVec4 text_color = ImVec4(1.0f, 1.0f, 1.0f, alpha);
      ImVec4 border_color;
      
      switch (toast.type) {
        case Type::Success:
          bg_color = ImVec4(0.2f, 0.7f, 0.3f, 0.95f * alpha);
          border_color = ImVec4(0.3f, 0.8f, 0.4f, alpha);
          break;
        case Type::Warning:
          bg_color = ImVec4(0.9f, 0.7f, 0.2f, 0.95f * alpha);
          border_color = ImVec4(1.0f, 0.8f, 0.3f, alpha);
          break;
        case Type::Error:
          bg_color = ImVec4(0.8f, 0.2f, 0.2f, 0.95f * alpha);
          border_color = ImVec4(0.9f, 0.3f, 0.3f, alpha);
          break;
        case Type::Info:
          bg_color = ImVec4(0.2f, 0.5f, 0.9f, 0.95f * alpha);
          border_color = ImVec4(0.3f, 0.6f, 1.0f, alpha);
          break;
      }
      
      // Window flags
      ImGuiWindowFlags flags = 
        ImGuiWindowFlags_NoTitleBar |
        ImGuiWindowFlags_NoResize |
        ImGuiWindowFlags_NoMove |
        ImGuiWindowFlags_NoScrollbar |
        ImGuiWindowFlags_NoCollapse |
        ImGuiWindowFlags_AlwaysAutoResize |
        ImGuiWindowFlags_NoFocusOnAppearing |
        ImGuiWindowFlags_NoNav;
      
      ImGui::PushStyleVar(ImGuiStyleVar_Alpha, alpha);
      ImGui::PushStyleColor(ImGuiCol_WindowBg, bg_color);
      ImGui::PushStyleColor(ImGuiCol_Border, border_color);
      ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 1.0f);
      ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 8.0f);
      ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(toast_padding, toast_padding));
      
      char window_id[32];
      snprintf(window_id, sizeof(window_id), "Toast_%d", toast.id);
      
      if (ImGui::Begin(window_id, nullptr, flags)) {
        ImGui::PushStyleColor(ImGuiCol_Text, text_color);
        ImGui::TextWrapped("%s", toast.message.c_str());
        ImGui::PopStyleColor();
      }
      ImGui::End();
      
      ImGui::PopStyleVar(4);
      ImGui::PopStyleColor(2);
      
      current_y += ImGui::GetWindowHeight() + toast_spacing;
      ++it;
    }
  }
  
  void Clear() {
    toasts.clear();
  }
}

// ============================================================================
// STYLE SYSTEM INITIALIZATION
// ============================================================================

namespace UIStyle {
  void ApplySpacing() {
    ImGuiStyle& style = ImGui::GetStyle();
    
    // Item spacing (horizontal and vertical)
    style.ItemSpacing = ImVec2(UISpacing::SM, UISpacing::SM);
    style.ItemInnerSpacing = ImVec2(UISpacing::XS, UISpacing::XS);
    
    // Window padding
    style.WindowPadding = ImVec2(UISpacing::WindowPadding, UISpacing::WindowPadding);
    style.FramePadding = ImVec2(UISpacing::InputPaddingX, UISpacing::InputPaddingY);
    style.CellPadding = ImVec2(UISpacing::XS, UISpacing::XS);
    
    // Indent
    style.IndentSpacing = 20.0f;
    style.ScrollbarSize = 14.0f;
    
    // Child window padding
    style.ChildRounding = 6.0f;
    style.ChildBorderSize = 1.0f;
  }
  
  void ApplyTypography() {
    ImGuiStyle& style = ImGui::GetStyle();
    
    // Note: Actual font loading would be done separately
    // This sets up the style for typography
    
    // Button text alignment
    style.ButtonTextAlign = ImVec2(0.5f, 0.5f);
    
    // Selectable text alignment
    style.SelectableTextAlign = ImVec2(0.0f, 0.0f);
  }
  
  void ApplyProfessionalStyle() {
    ImGuiStyle& style = ImGui::GetStyle();
    
    // Apply spacing
    ApplySpacing();
    
    // Apply typography
    ApplyTypography();
    
    // Rounding
    style.WindowRounding = 8.0f;
    style.FrameRounding = 6.0f;
    style.GrabRounding = 6.0f;
    style.ScrollbarRounding = 6.0f;
    style.TabRounding = 6.0f;
    style.ChildRounding = 6.0f;
    style.PopupRounding = 8.0f;
    
    // Borders
    style.WindowBorderSize = 1.0f;
    style.FrameBorderSize = 1.0f;
    style.PopupBorderSize = 1.0f;
    style.ChildBorderSize = 1.0f;
    
    // Anti-aliasing
    style.AntiAliasedLines = true;
    style.AntiAliasedFill = true;
    style.AntiAliasedLinesUseTex = true;

    // Disable drag-drop target yellow border (we use custom blue line indicator instead)
    style.Colors[ImGuiCol_DragDropTarget] = ImVec4(0.0f, 0.0f, 0.0f, 0.0f);
  }
  
  void ApplyPanelStyling() {
    ImGuiStyle& style = ImGui::GetStyle();
    
    // Enhanced panel styling
    style.ChildRounding = 8.0f;
    style.ChildBorderSize = 1.0f;
    style.WindowRounding = 8.0f;
    style.PopupRounding = 8.0f;
    
    // Frame styling for panels
    style.FrameRounding = 6.0f;
    style.FrameBorderSize = 1.0f;
  }
  
  void Initialize() {
    UIToast::Initialize();
    UIAnimation::Initialize();
    UIHelp::Initialize();
    // UIAccessibility::GetSettings();  // Phase 3 - not yet implemented
    ApplyProfessionalStyle();
    ApplyPanelStyling();
  }
}

// ============================================================================
// HELP SYSTEM IMPLEMENTATION (Phase 3 - minimal, but functional)
// ============================================================================

namespace UIHelp {
namespace {
  static bool g_initialized = false;
  static bool g_show_search = false;
  static std::string g_search_query;
  static std::string g_current_help_id;  // last explicitly requested id

  static std::map<std::string, HelpEntry> g_entries;

  static bool ContainsCaseInsensitive(const std::string& haystack, const std::string& needle) {
    if (needle.empty()) return true;
    auto lower = [](unsigned char c) { return static_cast<char>(std::tolower(c)); };
    std::string h;
    std::string n;
    h.reserve(haystack.size());
    n.reserve(needle.size());
    for (char c : haystack) h.push_back(lower(static_cast<unsigned char>(c)));
    for (char c : needle) n.push_back(lower(static_cast<unsigned char>(c)));
    return h.find(n) != std::string::npos;
  }

  static bool EntryMatches(const HelpEntry& e, const std::string& q) {
    if (q.empty()) return true;
    if (ContainsCaseInsensitive(e.id, q)) return true;
    if (ContainsCaseInsensitive(e.title, q)) return true;
    if (ContainsCaseInsensitive(e.content, q)) return true;
    if (ContainsCaseInsensitive(e.category, q)) return true;
    if (ContainsCaseInsensitive(e.shortcut, q)) return true;
    for (const auto& k : e.keywords) {
      if (ContainsCaseInsensitive(k, q)) return true;
    }
    return false;
  }

  static void RenderEntry(const HelpEntry& e) {
    ImGui::Text("%s", e.title.c_str());
    if (!e.shortcut.empty()) {
      ImGui::SameLine();
      ImGui::TextDisabled("(%s)", e.shortcut.c_str());
    }
    if (!e.category.empty()) {
      ImGui::SameLine();
      ImGui::TextDisabled("— %s", e.category.c_str());
    }
    if (!e.content.empty()) {
      ImGui::TextWrapped("%s", e.content.c_str());
    }
  }
}  // namespace

void Initialize() {
  if (g_initialized) return;
  g_entries.clear();
  g_search_query.clear();
  g_current_help_id.clear();
  g_show_search = false;
  g_initialized = true;
}

void OpenSearch() {
  g_show_search = true;
}

void RegisterHelp(const HelpEntry& entry) {
  if (!g_initialized) {
    Initialize();
  }
  if (entry.id.empty()) {
    return;
  }
  g_entries[entry.id] = entry;
}

const HelpEntry* GetHelp(const char* help_id) {
  if (!help_id) return nullptr;
  auto it = g_entries.find(std::string(help_id));
  if (it == g_entries.end()) return nullptr;
  return &it->second;
}

void ShowTooltip(const char* help_id) {
  const HelpEntry* e = GetHelp(help_id);
  if (!e) return;
  ImGui::BeginTooltip();
  ImGui::Text("%s", e->title.c_str());
  if (!e->content.empty()) {
    ImGui::Separator();
    ImGui::TextWrapped("%s", e->content.c_str());
  }
  if (!e->shortcut.empty()) {
    ImGui::Separator();
    ImGui::TextDisabled("Shortcut: %s", e->shortcut.c_str());
  }
  ImGui::EndTooltip();
}

bool HelpIcon(const char* help_id, const char* tooltip_text) {
  // Compact "(?)" icon; keep it simple and avoid requiring icon fonts.
  ImGui::PushStyleColor(ImGuiCol_Text, ImGui::GetStyleColorVec4(ImGuiCol_TextDisabled));
  bool clicked = ImGui::SmallButton("?");
  ImGui::PopStyleColor();

  if (ImGui::IsItemHovered()) {
    if (tooltip_text && *tooltip_text) {
      ImGui::BeginTooltip();
      ImGui::TextWrapped("%s", tooltip_text);
      ImGui::EndTooltip();
    } else {
      ShowTooltip(help_id);
    }
  }

  if (clicked) {
    if (help_id) {
      g_current_help_id = help_id;
    }
    g_show_search = true;
  }
  return clicked;
}

void ShowHelpPanel(const char* help_id) {
  const HelpEntry* e = GetHelp(help_id);
  if (!e) return;
  RenderEntry(*e);
}

void ShowHelpSearch() {
  if (!g_show_search) {
    return;
  }

  ImGui::OpenPopup("Help");
  if (ImGui::BeginPopupModal("Help", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
    ImGui::TextDisabled("Search help (F1)");
    ImGui::SetNextItemWidth(520.0f);
    ImGui::InputText("##help_search", &g_search_query);

    ImGui::Separator();

    // Results
    ImGui::BeginChild("##help_results", ImVec2(520.0f, 300.0f), true);
    int shown = 0;
    for (const auto& [id, e] : g_entries) {
      if (!EntryMatches(e, g_search_query)) continue;
      // If we have a "preferred" id (e.g. clicked icon), show it first-ish by marking it.
      if (!g_current_help_id.empty() && id == g_current_help_id) {
        ImGui::PushStyleColor(ImGuiCol_Text, ImGui::GetStyleColorVec4(ImGuiCol_Text));
        RenderEntry(e);
        ImGui::PopStyleColor();
      } else {
        RenderEntry(e);
      }
      ImGui::Separator();
      ++shown;
      if (shown >= 50) {
        ImGui::TextDisabled("Showing first 50 matches...");
        break;
      }
    }
    if (shown == 0) {
      ImGui::TextDisabled("No matches.");
    }
    ImGui::EndChild();

    if (ImGui::Button("Close")) {
      g_show_search = false;
      g_current_help_id.clear();
      ImGui::CloseCurrentPopup();
    }
    ImGui::EndPopup();
  }
}
}  // namespace UIHelp

// ============================================================================
// ICON SYSTEM IMPLEMENTATION (Phase 2)
// ============================================================================

namespace UIIcon {
  void Text(const char* icon, const char* text, Size size) {
    ImGui::PushFont(nullptr); // Use current font
    float icon_size = static_cast<float>(size);
    ImGui::Text("%s", icon);
    ImGui::SameLine(0, UISpacing::XS);
    ImGui::Text("%s", text);
  }
  
  void Icon(const char* icon, Size size, const ImVec4& color) {
    ImVec4 old_color = ImGui::GetStyleColorVec4(ImGuiCol_Text);
    ImGui::PushStyleColor(ImGuiCol_Text, color);
    ImGui::Text("%s", icon);
    ImGui::PopStyleColor();
  }
  
  bool Button(const char* icon, const char* label, Size size) {
    if (label) {
      ImGui::Text("%s", icon);
      ImGui::SameLine(0, UISpacing::XS);
      return ImGui::Button(label);
    } else {
      // Icon-only button
      float icon_size = static_cast<float>(size);
      ImVec2 button_size(icon_size + UISpacing::SM * 2, icon_size + UISpacing::SM * 2);
      return ImGui::Button(icon, button_size);
    }
  }
}

// ============================================================================
// PANEL STYLING IMPLEMENTATION (Phase 2)
// ============================================================================

namespace UIPanel {
  static std::map<const char*, Elevation> panel_elevations;
  
  void DrawShadow(const ImVec2& min, const ImVec2& max, Elevation elevation, float rounding) {
    if (elevation == Elevation::Flat) return;
    
    ImDrawList* draw_list = ImGui::GetWindowDrawList();
    ImVec4 shadow_color;
    float shadow_size;
    
    switch (elevation) {
      case Elevation::Subtle:
        shadow_color = ImVec4(0.0f, 0.0f, 0.0f, 0.1f);
        shadow_size = 2.0f;
        break;
      case Elevation::Medium:
        shadow_color = ImVec4(0.0f, 0.0f, 0.0f, 0.15f);
        shadow_size = 4.0f;
        break;
      case Elevation::Strong:
        shadow_color = ImVec4(0.0f, 0.0f, 0.0f, 0.25f);
        shadow_size = 8.0f;
        break;
      default:
        return;
    }
    
    // Draw shadow (simplified - multiple passes for blur effect)
    for (int i = 0; i < 3; ++i) {
      float offset = shadow_size * (i + 1) / 3.0f;
      float alpha = shadow_color.w * (1.0f - i * 0.3f);
      ImVec4 color = ImVec4(shadow_color.x, shadow_color.y, shadow_color.z, alpha);
      draw_list->AddRectFilled(
        ImVec2(min.x + offset, min.y + offset),
        ImVec2(max.x + offset, max.y + offset),
        ImGui::ColorConvertFloat4ToU32(color),
        rounding
      );
    }
  }
  
  bool Begin(const char* name, bool* p_open, ImGuiWindowFlags flags, Elevation elevation) {
    panel_elevations[name] = elevation;
    return ImGui::Begin(name, p_open, flags);
  }
  
  void End() {
    ImGui::End();
  }
  
  bool BeginChild(const char* str_id, const ImVec2& size, bool border, ImGuiWindowFlags flags, Elevation elevation) {
    panel_elevations[str_id] = elevation;
    return ImGui::BeginChild(str_id, size, border, flags);
  }
  
  void EndChild() {
    ImGui::EndChild();
  }
  
  void BeginCard(const char* label, bool border, Elevation elevation) {
    ImGui::PushStyleVar(ImGuiStyleVar_ChildRounding, 8.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_ChildBorderSize, border ? 1.0f : 0.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(UISpacing::PanelPadding, UISpacing::PanelPadding));
    
    if (label) {
      ImGui::BeginChild(label, ImVec2(0, 0), border, 0);
    }
  }
  
  void EndCard() {
    // Check if we're inside a child window
    if (ImGui::GetCurrentWindow()->Flags & ImGuiWindowFlags_ChildWindow) {
      ImGui::EndChild();
    }
    ImGui::PopStyleVar(3);
  }
}

// ============================================================================
// ANIMATION SYSTEM IMPLEMENTATION (Phase 2)
// ============================================================================

namespace UIAnimation {
  struct AnimationState {
    float current = 0.0f;
    float target = 0.0f;
    float speed = 5.0f;
    bool active = false;
  };
  
  struct ColorAnimationState {
    ImVec4 current = ImVec4(0, 0, 0, 0);
    ImVec4 target = ImVec4(0, 0, 0, 0);
    float speed = 5.0f;
    bool active = false;
  };
  
  static std::map<std::string, AnimationState> animations;
  static std::map<std::string, ColorAnimationState> color_animations;
  static bool initialized = false;
  
  void Initialize() {
    if (!initialized) {
      animations.clear();
      color_animations.clear();
      initialized = true;
    }
  }
  
  float EaseInOut(float t) {
    return t < 0.5f ? 2.0f * t * t : 1.0f - pow(-2.0f * t + 2.0f, 2.0f) / 2.0f;
  }
  
  float EaseOut(float t) {
    return 1.0f - pow(1.0f - t, 3.0f);
  }
  
  float EaseIn(float t) {
    return t * t * t;
  }
  
  void Update(float delta_time) {
    // Update float animations
    for (auto& [id, state] : animations) {
      if (state.active) {
        float diff = state.target - state.current;
        if (std::abs(diff) > 0.001f) {
          state.current += diff * state.speed * delta_time;
          if (std::abs(state.target - state.current) < 0.001f) {
            state.current = state.target;
            state.active = false;
          }
        } else {
          state.active = false;
        }
      }
    }
    
    // Update color animations
    for (auto& [id, state] : color_animations) {
      if (state.active) {
        ImVec4 diff = ImVec4(
          state.target.x - state.current.x,
          state.target.y - state.current.y,
          state.target.z - state.current.z,
          state.target.w - state.current.w
        );
        float max_diff = std::max({std::abs(diff.x), std::abs(diff.y), std::abs(diff.z), std::abs(diff.w)});
        if (max_diff > 0.001f) {
          state.current.x += diff.x * state.speed * delta_time;
          state.current.y += diff.y * state.speed * delta_time;
          state.current.z += diff.z * state.speed * delta_time;
          state.current.w += diff.w * state.speed * delta_time;
          if (max_diff < 0.001f) {
            state.current = state.target;
            state.active = false;
          }
        } else {
          state.active = false;
        }
      }
    }
  }
  
  float Animate(const char* id, float target, float speed) {
    std::string id_str(id);
    auto& state = animations[id_str];
    state.target = target;
    state.speed = speed;
    state.active = true;
    
    if (std::abs(state.current - state.target) < 0.001f) {
      state.current = state.target;
      state.active = false;
    }
    
    return state.current;
  }
  
  ImVec4 AnimateColor(const char* id, const ImVec4& target, float speed) {
    std::string id_str(id);
    auto& state = color_animations[id_str];
    state.target = target;
    state.speed = speed;
    state.active = true;
    
    float max_diff = std::max({
      std::abs(state.current.x - state.target.x),
      std::abs(state.current.y - state.target.y),
      std::abs(state.current.z - state.target.z),
      std::abs(state.current.w - state.target.w)
    });
    
    if (max_diff < 0.001f) {
      state.current = state.target;
      state.active = false;
    }
    
    return state.current;
  }
  
  float AnimateFloat(const char* id, float target, float speed) {
    return Animate(id, target, speed);
  }
  
  void Reset(const char* id) {
    std::string id_str(id);
    animations.erase(id_str);
    color_animations.erase(id_str);
  }
}

// ============================================================================
// EMPTY STATES IMPLEMENTATION (Phase 2)
// ============================================================================

namespace UIEmptyState {
  void Show(Type type, const char* title, const char* message, const char* action_label, std::function<void()> action) {
    ImVec2 region = ImGui::GetContentRegionAvail();
    float center_x = ImGui::GetCursorPosX() + region.x * 0.5f;
    float center_y = ImGui::GetCursorPosY() + region.y * 0.5f;
    
    ImGui::SetCursorPosY(center_y - 60.0f);
    
    // Icon
    const char* icon = "";
    ImVec4 icon_color = ImVec4(0.5f, 0.5f, 0.5f, 1.0f);
    
    switch (type) {
      case Type::NoData:
        icon = "📊";
        break;
      case Type::NoResults:
        icon = "🔍";
        break;
      case Type::NoLogs:
        icon = "📝";
        break;
      case Type::Loading:
        icon = "⏳";
        break;
      case Type::Error:
        icon = UIIcon::Symbol::Error;
        icon_color = ImVec4(0.9f, 0.3f, 0.3f, 1.0f);
        break;
      case Type::Empty:
        icon = "○";
        break;
    }
    
    ImGui::SetCursorPosX(center_x - 20.0f);
    ImGui::PushStyleColor(ImGuiCol_Text, icon_color);
    ImGui::Text("%s", icon);
    ImGui::PopStyleColor();
    
    // Title
    if (title) {
      ImGui::SetCursorPosX(center_x - ImGui::CalcTextSize(title).x * 0.5f);
      ImGui::Text("%s", title);
    }
    
    // Message
    if (message) {
      ImGui::Spacing();
      ImGui::SetCursorPosX(center_x - ImGui::CalcTextSize(message).x * 0.5f);
      ImGui::TextDisabled("%s", message);
    }
    
    // Action button
    if (action_label && action) {
      ImGui::Spacing();
      ImGui::SetCursorPosX(center_x - 60.0f);
      if (UIButton::Button(action_label, UIButton::Size::Medium, UIButton::Variant::Primary)) {
        action();
      }
    }
  }
  
  void ShowSkeleton(int lines) {
    ImVec2 region = ImGui::GetContentRegionAvail();
    float line_height = ImGui::GetTextLineHeight() + UISpacing::SM;
    
    for (int i = 0; i < lines; ++i) {
      float width = region.x * (0.7f + (i % 3) * 0.1f);
      ImVec2 pos = ImGui::GetCursorScreenPos();
      ImDrawList* draw_list = ImGui::GetWindowDrawList();
      
      ImVec4 bg_color = ImGui::GetStyleColorVec4(ImGuiCol_FrameBg);
      bg_color.w *= 0.5f;
      
      draw_list->AddRectFilled(
        pos,
        ImVec2(pos.x + width, pos.y + line_height),
        ImGui::ColorConvertFloat4ToU32(bg_color),
        4.0f
      );
      
      ImGui::SetCursorPosY(ImGui::GetCursorPosY() + line_height);
    }
  }
}

// ============================================================================
// STATUS BADGES IMPLEMENTATION (Phase 2)
// ============================================================================

namespace UIBadge {
  void Draw(const char* text, Type type, bool small) {
    ImVec4 bg_color;
    ImVec4 text_color = ImVec4(1.0f, 1.0f, 1.0f, 1.0f);
    
    switch (type) {
      case Type::Success:
        bg_color = ImVec4(0.2f, 0.7f, 0.3f, 1.0f);
        break;
      case Type::Warning:
        bg_color = ImVec4(0.9f, 0.7f, 0.2f, 1.0f);
        break;
      case Type::Error:
        bg_color = ImVec4(0.8f, 0.2f, 0.2f, 1.0f);
        break;
      case Type::Info:
        bg_color = ImVec4(0.2f, 0.5f, 0.9f, 1.0f);
        break;
      case Type::Neutral:
        bg_color = ImVec4(0.4f, 0.4f, 0.4f, 1.0f);
        break;
    }
    
    ImVec2 padding = small ? ImVec2(4.0f, 2.0f) : ImVec2(8.0f, 4.0f);
    ImVec2 text_size = ImGui::CalcTextSize(text);
    ImVec2 badge_size = ImVec2(text_size.x + padding.x * 2, text_size.y + padding.y * 2);
    
    ImVec2 pos = ImGui::GetCursorScreenPos();
    ImDrawList* draw_list = ImGui::GetWindowDrawList();
    
    // Draw badge background
    draw_list->AddRectFilled(
      pos,
      ImVec2(pos.x + badge_size.x, pos.y + badge_size.y),
      ImGui::ColorConvertFloat4ToU32(bg_color),
      4.0f
    );
    
    // Draw text
    ImVec2 cursor_pos = ImGui::GetCursorPos();
    ImGui::SetCursorPos(ImVec2(cursor_pos.x + padding.x, cursor_pos.y + padding.y));
    ImGui::PushStyleColor(ImGuiCol_Text, text_color);
    ImGui::Text("%s", text);
    ImGui::PopStyleColor();
    cursor_pos = ImGui::GetCursorPos();
    ImGui::SetCursorPos(ImVec2(cursor_pos.x, cursor_pos.y + padding.y));
  }
  
  void DrawPill(const char* text, Type type) {
    Draw(text, type, true);
  }
}
