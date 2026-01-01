#include "error_dialog.h"

#include "../utils/file_dialog.h"
#include <filesystem>
#include <fstream>

namespace {
std::string JoinLines(const std::vector<std::string>& lines) {
  std::string out;
  for (const auto& l : lines) {
    out.append(l);
    out.push_back('\n');
  }
  return out;
}
}  // namespace

void ErrorDialogComponent::Render(SharedState& state, std::mutex& state_mutex) {
  bool open_requested = false;
  std::vector<std::string> logs_snapshot;
  {
    std::lock_guard<std::mutex> lock(state_mutex);
    if (state.error_dialog_open && state.last_error.has_value()) {
      active_error_ = state.last_error;
      state.error_dialog_open = false;
      open_requested = true;
    }
    logs_snapshot = state.logs;
  }

  if (!active_error_.has_value()) {
    return;
  }

  if (open_requested) {
    ImGui::OpenPopup("Error");
  }

  if (ImGui::BeginPopupModal("Error", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
    const ErrorInfo& err = *active_error_;

    ImGui::TextColored(ImVec4(1.0f, 0.35f, 0.35f, 1.0f), "%s", err.title.c_str());
    ImGui::Spacing();

    if (!err.message.empty()) {
      ImGui::TextWrapped("%s", err.message.c_str());
    }

    if (!err.suggestions.empty()) {
      ImGui::Spacing();
      ImGui::Text("Suggestions:");
      for (const auto& s : err.suggestions) {
        ImGui::BulletText("%s", s.c_str());
      }
    }

    if (!err.details.empty()) {
      ImGui::Spacing();
      if (ImGui::TreeNode("Details")) {
        ImGui::PushTextWrapPos(520.0f);
        ImGui::TextUnformatted(err.details.c_str());
        ImGui::PopTextWrapPos();
        ImGui::TreePop();
      }
    }

    ImGui::Separator();

    if (ImGui::Button("Export logs...")) {
      std::filesystem::path default_path = std::filesystem::current_path();
      auto selected = FileDialog::SaveFile(
        "Save Logs",
        default_path,
        "pde_gui_logs.txt",
        "Text file",
        {".txt"}
      );
      if (selected) {
        std::error_code ec;
        std::filesystem::create_directories(selected->parent_path(), ec);
        std::ofstream out(*selected, std::ios::trunc);
        if (out) {
          out << JoinLines(logs_snapshot);
        }
      }
    }

    ImGui::SameLine();
    if (ImGui::Button("Close")) {
      active_error_.reset();
      ImGui::CloseCurrentPopup();
    }

    ImGui::EndPopup();
  }
}


