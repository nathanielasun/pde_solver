#include "animation_export_panel.h"
#include "imgui.h"

static int s_format = 0;
static int s_width = 1280;
static int s_height = 720;
static int s_fps = 24;
static int s_start_frame = 0;
static int s_end_frame = -1;
static bool s_show_timestamp = true;

void RenderAnimationExportPanel(AnimationExportPanelState& state,
                                 const std::vector<std::string>& components) {
  ImGui::Text("Animation Export");
  ImGui::Separator();
  ImGui::Spacing();

  if (state.frame_paths.empty()) {
    ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f),
                       "No time series data available.");
    ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f),
                       "Solve a time-dependent PDE first.");
    return;
  }

  ImGui::Text("Frames available: %zu", state.frame_paths.size());
  ImGui::Spacing();

  // Format selection
  ImGui::Text("Export Format:");
  ImGui::RadioButton("GIF", &s_format, 0);
  ImGui::SameLine();
  ImGui::RadioButton("PNG Sequence", &s_format, 1);
  ImGui::SameLine();
  ImGui::RadioButton("MP4", &s_format, 2);

  ImGui::Spacing();
  ImGui::Separator();
  ImGui::Spacing();

  // Resolution
  ImGui::Text("Resolution:");
  ImGui::SetNextItemWidth(80.0f);
  ImGui::InputInt("Width##anim", &s_width);
  ImGui::SameLine();
  ImGui::SetNextItemWidth(80.0f);
  ImGui::InputInt("Height##anim", &s_height);

  // Presets
  if (ImGui::Button("720p")) { s_width = 1280; s_height = 720; }
  ImGui::SameLine();
  if (ImGui::Button("1080p")) { s_width = 1920; s_height = 1080; }
  ImGui::SameLine();
  if (ImGui::Button("4K")) { s_width = 3840; s_height = 2160; }

  ImGui::Spacing();

  // FPS
  ImGui::SetNextItemWidth(80.0f);
  ImGui::InputInt("FPS", &s_fps);

  // Frame range
  int max_frame = static_cast<int>(state.frame_paths.size()) - 1;
  ImGui::SetNextItemWidth(80.0f);
  ImGui::InputInt("Start Frame", &s_start_frame);
  ImGui::SetNextItemWidth(80.0f);
  ImGui::InputInt("End Frame", &s_end_frame);
  if (s_end_frame < 0) s_end_frame = max_frame;

  ImGui::Spacing();

  // Options
  ImGui::Checkbox("Show timestamp overlay", &s_show_timestamp);

  ImGui::Spacing();
  ImGui::Separator();
  ImGui::Spacing();

  if (ImGui::Button("Export Animation", ImVec2(-1, 30))) {
    // TODO: Implement export
  }

  ImGui::Spacing();
  if (s_format == 2) {
    ImGui::TextColored(ImVec4(1.0f, 0.7f, 0.3f, 1.0f),
                       "MP4 export requires ffmpeg.");
  }
  ImGui::TextColored(ImVec4(1.0f, 0.7f, 0.3f, 1.0f),
                     "Video export coming soon.");
}
