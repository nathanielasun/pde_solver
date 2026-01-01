#include "export_panel.h"
#include "ui_helpers.h"
#include "io/image_export.h"
#include "utils/file_dialog.h"
#include "styles/ui_style.h"
#include "imgui.h"
#include <algorithm>
#include <ctime>
#include <filesystem>
#include <iomanip>
#include <sstream>
#include <cfloat>

namespace {

std::vector<std::string> DefaultComponents() {
  return {"image_export"};
}

bool UseCompactLayout(float input_width) {
  const float avail = ImGui::GetContentRegionAvail().x;
  return avail < std::max(320.0f, input_width * 1.2f);
}

void RenderImageExportSection(ExportPanelState& state) {
  static ImageExport::ExportOptions export_options;
  static std::string default_export_dir = "outputs";
  static std::string export_filename = "pde_export";
  static int export_format = 0;
  static int resolution_preset = 0;

  const bool compact = UseCompactLayout(state.input_width);

  const char* format_items[] = {"PNG", "JPEG"};
  ImGui::Text("Format");
  ImGui::SetNextItemWidth(state.input_width);
  UIInput::Combo("##export_format", &export_format, format_items, 2);

  ImGui::Text("Resolution");
  const char* resolution_items[] = {
    "1920x1080 (Full HD)",
    "1280x720 (HD)",
    "2560x1440 (2K)",
    "3840x2160 (4K)",
    "Custom"
  };
  ImGui::SetNextItemWidth(state.input_width);
  if (UIInput::Combo("##resolution_preset", &resolution_preset, resolution_items, 5)) {
    switch (resolution_preset) {
      case 0: export_options.width = 1920; export_options.height = 1080; break;
      case 1: export_options.width = 1280; export_options.height = 720; break;
      case 2: export_options.width = 2560; export_options.height = 1440; break;
      case 3: export_options.width = 3840; export_options.height = 2160; break;
      case 4: break;
    }
  }

  if (resolution_preset == 4) {
    if (!compact && ImGui::BeginTable("export_resolution", 2, ImGuiTableFlags_SizingStretchProp)) {
      ImGui::TableNextColumn();
      ImGui::Text("Width");
      ImGui::SetNextItemWidth(-FLT_MIN);
      UIInput::InputInt("##export_width", &export_options.width, 100, 1000);
      ImGui::TableNextColumn();
      ImGui::Text("Height");
      ImGui::SetNextItemWidth(-FLT_MIN);
      UIInput::InputInt("##export_height", &export_options.height, 100, 1000);
      ImGui::EndTable();
    } else {
      ImGui::Text("Width");
      ImGui::SetNextItemWidth(state.input_width);
      UIInput::InputInt("##export_width", &export_options.width, 100, 1000);
      ImGui::Text("Height");
      ImGui::SetNextItemWidth(state.input_width);
      UIInput::InputInt("##export_height", &export_options.height, 100, 1000);
    }
    export_options.width = std::max(100, std::min(8192, export_options.width));
    export_options.height = std::max(100, std::min(8192, export_options.height));
  }

  ImGui::Checkbox("Include axis labels", &export_options.include_axis_labels);

  if (export_format == 1) {
    ImGui::Text("JPEG Quality");
    ImGui::SetNextItemWidth(state.input_width);
    ImGui::SliderInt("##jpeg_quality", &export_options.jpeg_quality, 1, 100);
  }

  ImGui::Spacing();
  ImGui::Text("Filename (without extension)");
  ImGui::SetNextItemWidth(state.input_width);
  UIInput::InputText("##export_filename", &export_filename);

  ImGui::BeginDisabled(!state.viewer.has_data());
  if (UIButton::Button("Export Image...", ImVec2(state.input_width, 0),
                       UIButton::Size::Medium, UIButton::Variant::Primary)) {
    std::string default_name = export_filename;
    if (default_name.empty()) {
      std::time_t now = std::time(nullptr);
      std::tm* timeinfo = std::localtime(&now);
      std::ostringstream filename_stream;
      filename_stream << "pde_export_" << std::put_time(timeinfo, "%Y%m%d_%H%M%S");
      default_name = filename_stream.str();
    }

    default_name += (export_format == 0 ? ".png" : ".jpg");
    std::filesystem::path default_path = default_export_dir.empty()
      ? std::filesystem::current_path()
      : std::filesystem::path(default_export_dir);

    std::vector<std::string> extensions;
    if (export_format == 0) {
      extensions = {".png"};
    } else {
      extensions = {".jpg", ".jpeg"};
    }

    auto selected = FileDialog::SaveFile(
      "Save Image",
      default_path,
      default_name,
      export_format == 0 ? "PNG Image" : "JPEG Image",
      extensions
    );

    if (selected) {
      default_export_dir = selected->parent_path().string();
      std::filesystem::create_directories(selected->parent_path());

      static std::string saved_path;
      if (ImageExport::ExportImage(state.viewer, *selected, export_options)) {
        saved_path = selected->string();
        ImGui::OpenPopup("Export Success");
      } else {
        ImGui::OpenPopup("Export Error");
      }
    }
  }
  ImGui::EndDisabled();

  if (!state.viewer.has_data()) {
    ImGui::TextDisabled("No data to export. Solve PDE or load VTK file first.");
  }

  static std::string saved_path;
  if (ImGui::BeginPopupModal("Export Success", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
    ImGui::Text("Image exported successfully!");
    if (!saved_path.empty()) {
      ImGui::Text("Saved to: %s", saved_path.c_str());
    }
    if (UIButton::Button("OK", UIButton::Size::Small, UIButton::Variant::Secondary)) {
      ImGui::CloseCurrentPopup();
    }
    ImGui::EndPopup();
  }

  if (ImGui::BeginPopupModal("Export Error", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
    ImGui::TextColored(ImVec4(1.0f, 0.0f, 0.0f, 1.0f), "Failed to export image!");
    ImGui::Text("Please check that:");
    ImGui::BulletText("The export directory is valid");
    ImGui::BulletText("You have write permissions");
    ImGui::BulletText("The viewer has data loaded");
    if (UIButton::Button("OK", UIButton::Size::Small, UIButton::Variant::Secondary)) {
      ImGui::CloseCurrentPopup();
    }
    ImGui::EndPopup();
  }
}

} // namespace

void RenderExportPanel(ExportPanelState& state, const std::vector<std::string>& components) {
  const std::vector<std::string> default_components = DefaultComponents();
  const std::vector<std::string>& component_list =
      components.empty() ? default_components : components;

  for (size_t i = 0; i < component_list.size(); ++i) {
    const std::string& id = component_list[i];
    if (id == "image_export") {
      RenderImageExportSection(state);
    } else {
      DrawUnknownComponentPlaceholder(id.c_str());
    }

    if (i + 1 < component_list.size()) {
      ImGui::Spacing();
    }
  }
}

