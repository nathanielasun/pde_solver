#include "application.h"

#include "imgui.h"

#include <algorithm>
#include <cstring>

#include "../ui_helpers.h"
#include "../styles/ui_style.h"

void Application::RenderMainTab(bool time_dependent) {
  ImGuiIO& io = ImGui::GetIO();
  
  if (left_collapsed_) {
    if (ImGui::SmallButton(">>")) {
      left_collapsed_ = false;
      left_panel_width_ = 400.0f; // Restore to default width
    }
    ImGui::SameLine();
    RenderVisualizationPanel();
  } else {
    // Left panel
    float max_left = std::min(left_panel_max_width_, io.DisplaySize.x * 0.6f);
    left_panel_width_ = std::max(left_panel_min_width_, std::min(left_panel_width_, max_left));
    
    ImGui::BeginChild("LeftPanel", ImVec2(left_panel_width_, -1), false);
    
    if (ImGui::SmallButton("<<")) {
      left_collapsed_ = true;
    }
    RenderPanelsForTab("Main", "main", time_dependent);
    
    ImGui::EndChild();
    ImGui::SameLine();
    
    // Splitter
    ImGui::InvisibleButton("splitter", ImVec2(splitter_width_, -1));
    const bool splitter_hovered = ImGui::IsItemHovered();
    const bool splitter_active = ImGui::IsItemActive();
    if (splitter_hovered || splitter_active) {
      ImGui::SetMouseCursor(ImGuiMouseCursor_ResizeEW);
    }
    ImDrawList* draw_list = ImGui::GetWindowDrawList();
    ImVec2 splitter_min = ImGui::GetItemRectMin();
    ImVec2 splitter_max = ImGui::GetItemRectMax();
    const ImVec4& splitter_base = prefs_.colors.splitter_color;
    ImVec4 splitter_active_col = ImVec4(
      std::min(1.0f, splitter_base.x * 1.3f),
      std::min(1.0f, splitter_base.y * 1.3f),
      std::min(1.0f, splitter_base.z * 1.3f),
      splitter_base.w);
    ImVec4 splitter_hover_col = ImVec4(
      std::min(1.0f, splitter_base.x * 1.15f),
      std::min(1.0f, splitter_base.y * 1.15f),
      std::min(1.0f, splitter_base.z * 1.15f),
      splitter_base.w);
    const ImU32 splitter_col =
        splitter_active ? ImGui::ColorConvertFloat4ToU32(splitter_active_col)
                        : (splitter_hovered ? ImGui::ColorConvertFloat4ToU32(splitter_hover_col)
                                            : ImGui::ColorConvertFloat4ToU32(splitter_base));
    draw_list->AddRectFilled(splitter_min, splitter_max, splitter_col);
    if (ImGui::IsItemActive()) {
      left_panel_width_ += io.MouseDelta.x;
      left_panel_width_ = std::max(left_panel_min_width_, std::min(left_panel_width_, max_left));
    }
    ImGui::SameLine();
    
    // Right panel (visualization)
    RenderVisualizationPanel();
  }
}

void Application::RenderInspectTab() {
  ImGuiIO& io = ImGui::GetIO();
  
  if (inspect_left_collapsed_) {
    if (ImGui::SmallButton(">>")) {
      inspect_left_collapsed_ = false;
      inspect_left_panel_width_ = 400.0f; // Restore to default width
    }
    ImGui::SameLine();
    RenderVisualizationPanel();
  } else {
    float max_left = std::min(left_panel_max_width_, io.DisplaySize.x * 0.6f);
    inspect_left_panel_width_ = std::max(left_panel_min_width_, std::min(inspect_left_panel_width_, max_left));
    
    ImGui::BeginChild("InspectLeftPanel", ImVec2(inspect_left_panel_width_, -1), false);
    
    if (ImGui::SmallButton("<<")) {
      inspect_left_collapsed_ = true;
    }
    RenderPanelsForTab("Inspect", "inspect", false);
    
    ImGui::EndChild();
    ImGui::SameLine();
    
    // Visualization panel (same as main tab)
    RenderVisualizationPanel();
  }
}

void Application::RenderPreferencesTab() {
  ImGui::BeginChild("PreferencesPanel", ImVec2(-1, -1), false);
  RenderPanelsForTab("Preferences", "prefs", false);

  ImGui::Spacing();
  ImGui::TextDisabled("Prefs file: %s", prefs_path_.string().c_str());
  
  ImGui::EndChild();
}

void Application::RenderPanelsForTab(const std::string& tab_name, const char* tab_key,
                                     bool time_dependent) {
  const auto panel_ptrs = ui_config_mgr_.GetPanelsForTab(tab_name);
  std::vector<PanelConfig> panels;
  panels.reserve(panel_ptrs.size());
  for (const auto* panel_ptr : panel_ptrs) {
    if (panel_ptr) {
      panels.push_back(*panel_ptr);
    }
  }
  if (panels.empty()) {
    ImGui::TextDisabled("No panels configured for %s.", tab_name.c_str());
    return;
  }

  const float spacing = std::max(0.0f, ui_config_mgr_.GetConfig().theme.panel_spacing);
  const std::string tab_prefix = tab_key ? tab_key : tab_name;

  // Track drag and drop state for visual feedback
  static std::string dragging_panel_id;
  static int drop_target_index = -1;
  const ImGuiPayload* drag_payload = ImGui::GetDragDropPayload();
  bool is_dragging = drag_payload && strcmp(drag_payload->DataType, "PANEL_REORDER") == 0;

  if (!is_dragging) {
    dragging_panel_id.clear();
    drop_target_index = -1;
  }

  for (size_t i = 0; i < panels.size(); ++i) {
    const PanelConfig& panel = panels[i];
    const char* header_label = panel.name.empty() ? panel.id.c_str() : panel.name.c_str();

    ImGui::PushID(panel.id.c_str());

    // Draw drop indicator line above this panel if hovering
    if (is_dragging && drop_target_index == static_cast<int>(i)) {
      ImDrawList* draw_list = ImGui::GetWindowDrawList();
      ImVec2 cursor_pos = ImGui::GetCursorScreenPos();
      float line_width = ImGui::GetContentRegionAvail().x;
      ImU32 line_color = ImGui::ColorConvertFloat4ToU32(ImVec4(0.3f, 0.6f, 1.0f, 0.8f));

      // Draw animated drop indicator
      draw_list->AddRectFilled(
        ImVec2(cursor_pos.x, cursor_pos.y - 2.0f),
        ImVec2(cursor_pos.x + line_width, cursor_pos.y + 2.0f),
        line_color,
        2.0f
      );

      // Add visual spacing
      ImGui::Dummy(ImVec2(0.0f, 4.0f));
    }

    // Animate panel position during reordering
    std::string anim_id = "panel_offset_" + tab_name + "_" + panel.id;
    float y_offset = UIAnimation::AnimateFloat(anim_id.c_str(), 0.0f, 12.0f);

    if (std::abs(y_offset) > 0.1f) {
      ImGui::SetCursorPosY(ImGui::GetCursorPosY() + y_offset);
    }

    // Grip button
    ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0, 0, 0, 0));
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.3f, 0.3f, 0.3f, 0.3f));
    ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.4f, 0.4f, 0.4f, 0.5f));
    ImGui::Button(":::");
    ImGui::PopStyleColor(3);

    // Drag Source
    if (ImGui::BeginDragDropSource(ImGuiDragDropFlags_SourceAllowNullID)) {
      ImGui::SetDragDropPayload("PANEL_REORDER", panel.id.c_str(), panel.id.size() + 1);
      ImGui::Text("Move %s", header_label);
      dragging_panel_id = panel.id;
      ImGui::EndDragDropSource();
    }

    // Drag Target (Grip) - detect hover
    if (ImGui::BeginDragDropTarget()) {
      drop_target_index = static_cast<int>(i);
      if (const ImGuiPayload* payload = ImGui::AcceptDragDropPayload("PANEL_REORDER")) {
        std::string source_id((const char*)payload->Data);
        if (source_id != panel.id) {
          ui_config_mgr_.ReorderPanel(tab_name, source_id, panel.id);
        }
      }
      ImGui::EndDragDropTarget();
    }

    ImGui::SameLine();

    // Add indentation for content alignment
    float content_indent = ImGui::GetCursorPosX();

    bool open = true;
    if (panel.collapsible) {
      std::string header_id = tab_prefix + "." + panel.id;
      open = CollapsingHeaderWithMemory(header_label, header_id.c_str(), prefs_,
                                        !panel.default_collapsed, &prefs_changed_);

      // Drag Target (Header)
      if (ImGui::BeginDragDropTarget()) {
        drop_target_index = static_cast<int>(i);
        if (const ImGuiPayload* payload = ImGui::AcceptDragDropPayload("PANEL_REORDER")) {
          std::string source_id((const char*)payload->Data);
          if (source_id != panel.id) {
            ui_config_mgr_.ReorderPanel(tab_name, source_id, panel.id);
          }
        }
        ImGui::EndDragDropTarget();
      }
    } else if (!panel.name.empty()) {
      ImGui::Text("%s", header_label);
      ImGui::Separator();
    }

    if (!panel.collapsible || open) {
      // Add proper indentation for panel content (double indent for better spacing)
      const float double_indent = ImGui::GetStyle().IndentSpacing * 2.0f;
      ImGui::Indent(double_indent);

      // Limit content width to prevent overflow
      float available_width = ImGui::GetContentRegionAvail().x;
      ImGui::PushItemWidth(std::min(available_width, input_width_));

      auto it = panel_registry_.find(panel.id);
      if (it != panel_registry_.end()) {
        it->second(panel, time_dependent);
      } else {
        ImGui::TextDisabled("Missing panel: %s", panel.id.c_str());
      }

      ImGui::PopItemWidth();
      ImGui::Unindent(double_indent);
    }

    ImGui::PopID();

    if (i + 1 < panels.size() && spacing > 0.0f) {
      ImGui::Dummy(ImVec2(0.0f, spacing));
    }
  }

  // Draw drop indicator at the end if hovering below all panels
  if (is_dragging && drop_target_index == static_cast<int>(panels.size())) {
    ImDrawList* draw_list = ImGui::GetWindowDrawList();
    ImVec2 cursor_pos = ImGui::GetCursorScreenPos();
    float line_width = ImGui::GetContentRegionAvail().x;
    ImU32 line_color = ImGui::ColorConvertFloat4ToU32(ImVec4(0.3f, 0.6f, 1.0f, 0.8f));

    draw_list->AddRectFilled(
      ImVec2(cursor_pos.x, cursor_pos.y - 2.0f),
      ImVec2(cursor_pos.x + line_width, cursor_pos.y + 2.0f),
      line_color,
      2.0f
    );
  }
}

