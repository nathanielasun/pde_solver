#include "advanced_panel.h"
#include "ui_helpers.h"
#include "components/inspection_tools.h"
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
  return {"advanced_inspection"};
}

bool UseCompactLayout(float input_width) {
  const float avail = ImGui::GetContentRegionAvail().x;
  return avail < std::max(320.0f, input_width * 1.2f);
}

void RenderAdvancedInspectionSection(AdvancedPanelState& state) {
  InspectionToolsComponent* inspection_component = GetInspectionComponentSingleton();
  inspection_component->SetViewer(&state.viewer);

  const Domain* domain_ptr = nullptr;
  const std::vector<double>* grid_ptr = nullptr;
  const DerivedFields* derived_ptr = nullptr;
  {
    std::lock_guard<std::mutex> lock(state.state_mutex);
    domain_ptr = &state.current_domain;
    grid_ptr = &state.current_grid;
    if (state.has_derived_fields) {
      derived_ptr = &state.derived_fields;
    }
  }

  inspection_component->SetData(domain_ptr, grid_ptr, derived_ptr);
  inspection_component->Render();

  ImGui::Spacing();
  ImGui::Separator();
  ImGui::Spacing();

  if (ImGui::TreeNode("Export Line Plot Images")) {
    static std::string lineplot_default_dir = "outputs";
    static std::string lineplot_filename = "lineplot";
    static int lineplot_export_format = 0;
    static ImageExport::LinePlotExportOptions lineplot_export_options;
    static std::string lineplot_saved_path;

    const auto& line_plots = inspection_component->GetLinePlots();
    if (line_plots.empty()) {
      ImGui::TextDisabled("No line plots available. Add a line plot first.");
    } else {
      static int selected_plot_index = 0;
      std::vector<const char*> plot_names;
      plot_names.reserve(line_plots.size());
      for (const auto& plot : line_plots) {
        plot_names.push_back(plot.name.empty() ? "Unnamed Plot" : plot.name.c_str());
      }

      ImGui::Text("Select Line Plot");
      ImGui::SetNextItemWidth(state.input_width);
      if (UIInput::Combo("##plot_selector", &selected_plot_index,
                         plot_names.data(), static_cast<int>(plot_names.size()))) {
        selected_plot_index = std::max(0, std::min(selected_plot_index,
                                                   static_cast<int>(line_plots.size()) - 1));
      }

      if (selected_plot_index >= 0 && selected_plot_index < static_cast<int>(line_plots.size())) {
        const auto& selected_plot = line_plots[static_cast<size_t>(selected_plot_index)];

        if (!selected_plot.enabled || selected_plot.values.empty()) {
          ImGui::TextColored(ImVec4(1.0f, 0.5f, 0.0f, 1.0f),
                             "Selected plot is disabled or has no data.");
        } else {
          const char* format_items[] = {"PNG", "JPEG"};
          ImGui::Text("Format");
          ImGui::SetNextItemWidth(state.input_width);
          UIInput::Combo("##lineplot_format", &lineplot_export_format, format_items, 2);

          ImGui::Text("Resolution");
          if (!UseCompactLayout(state.input_width) &&
              ImGui::BeginTable("lineplot_resolution", 2, ImGuiTableFlags_SizingStretchProp)) {
            ImGui::TableNextColumn();
            ImGui::Text("Width");
            ImGui::SetNextItemWidth(-FLT_MIN);
            UIInput::InputInt("##lineplot_width", &lineplot_export_options.width, 0, 0);
            ImGui::TableNextColumn();
            ImGui::Text("Height");
            ImGui::SetNextItemWidth(-FLT_MIN);
            UIInput::InputInt("##lineplot_height", &lineplot_export_options.height, 0, 0);
            ImGui::EndTable();
          } else {
            ImGui::Text("Width");
            ImGui::SetNextItemWidth(state.input_width);
            UIInput::InputInt("##lineplot_width", &lineplot_export_options.width, 0, 0);
            ImGui::Text("Height");
            ImGui::SetNextItemWidth(state.input_width);
            UIInput::InputInt("##lineplot_height", &lineplot_export_options.height, 0, 0);
          }

          lineplot_export_options.width =
              std::max(400, std::min(8192, lineplot_export_options.width));
          lineplot_export_options.height =
              std::max(300, std::min(8192, lineplot_export_options.height));

          ImGui::Checkbox("Include grid", &lineplot_export_options.include_grid);
          ImGui::Checkbox("Include axis labels", &lineplot_export_options.include_axis_labels);
          ImGui::Checkbox("Include metadata", &lineplot_export_options.include_metadata);

          if (lineplot_export_options.include_metadata) {
            ImGui::SetNextItemWidth(state.input_width);
            UIInput::InputInt("Metadata height", &lineplot_export_options.metadata_height, 10, 20);
            lineplot_export_options.metadata_height =
                std::max(60, std::min(300, lineplot_export_options.metadata_height));
          }

          if (lineplot_export_format == 1) {
            ImGui::Text("JPEG Quality");
            ImGui::SetNextItemWidth(state.input_width);
            ImGui::SliderInt("##lineplot_jpeg_quality", &lineplot_export_options.jpeg_quality, 1, 100);
          }

          ImGui::Spacing();
          ImGui::Text("Filename (without extension)");
          ImGui::SetNextItemWidth(state.input_width);
          UIInput::InputText("##lineplot_filename", &lineplot_filename);

          if (UIButton::Button("Export Line Plot Image...", ImVec2(state.input_width, 0),
                               UIButton::Size::Medium, UIButton::Variant::Primary)) {
            std::string default_name = lineplot_filename;
            if (default_name.empty()) {
              std::time_t now = std::time(nullptr);
              std::tm* timeinfo = std::localtime(&now);
              std::ostringstream filename_stream;
              filename_stream << "lineplot_"
                              << std::put_time(timeinfo, "%Y%m%d_%H%M%S");
              default_name = filename_stream.str();
            }

            default_name += (lineplot_export_format == 0 ? ".png" : ".jpg");

            std::filesystem::path default_path = lineplot_default_dir.empty()
              ? std::filesystem::current_path()
              : std::filesystem::path(lineplot_default_dir);

            std::vector<std::string> extensions;
            if (lineplot_export_format == 0) {
              extensions = {".png"};
            } else {
              extensions = {".jpg", ".jpeg"};
            }

            auto selected = FileDialog::SaveFile(
              "Save Line Plot Image",
              default_path,
              default_name,
              lineplot_export_format == 0 ? "PNG Image" : "JPEG Image",
              extensions
            );

            if (selected) {
              lineplot_default_dir = selected->parent_path().string();
              std::filesystem::create_directories(selected->parent_path());

              GlViewer::FieldType current_field = state.viewer.GetFieldType();
              if (ImageExport::ExportLinePlotImage(selected_plot, state.current_domain,
                                                   current_field, *selected,
                                                   lineplot_export_options)) {
                lineplot_saved_path = selected->string();
                ImGui::OpenPopup("LinePlot Export Success");
              } else {
                ImGui::OpenPopup("LinePlot Export Error");
              }
            }
          }

          if (ImGui::BeginPopupModal("LinePlot Export Success", nullptr,
                                     ImGuiWindowFlags_AlwaysAutoResize)) {
            ImGui::Text("Line plot image exported successfully!");
            if (!lineplot_saved_path.empty()) {
              ImGui::Text("Saved to: %s", lineplot_saved_path.c_str());
            }
            if (UIButton::Button("OK", UIButton::Size::Small, UIButton::Variant::Secondary)) {
              ImGui::CloseCurrentPopup();
            }
            ImGui::EndPopup();
          }

          if (ImGui::BeginPopupModal("LinePlot Export Error", nullptr,
                                     ImGuiWindowFlags_AlwaysAutoResize)) {
            ImGui::TextColored(ImVec4(1.0f, 0.0f, 0.0f, 1.0f),
                               "Failed to export line plot image!");
            ImGui::Text("Please check that:");
            ImGui::BulletText("The export directory is valid");
            ImGui::BulletText("You have write permissions");
            ImGui::BulletText("The line plot has data");
            if (UIButton::Button("OK", UIButton::Size::Small, UIButton::Variant::Secondary)) {
              ImGui::CloseCurrentPopup();
            }
            ImGui::EndPopup();
          }
        }
      }
    }

    ImGui::TreePop();
  }
}

} // namespace

void RenderAdvancedPanel(AdvancedPanelState& state, const std::vector<std::string>& components) {
  const std::vector<std::string> default_components = DefaultComponents();
  const std::vector<std::string>& component_list =
      components.empty() ? default_components : components;

  for (size_t i = 0; i < component_list.size(); ++i) {
    const std::string& id = component_list[i];
    if (id == "advanced_inspection") {
      RenderAdvancedInspectionSection(state);
    } else {
      DrawUnknownComponentPlaceholder(id.c_str());
    }

    if (i + 1 < component_list.size()) {
      ImGui::Spacing();
    }
  }
}

