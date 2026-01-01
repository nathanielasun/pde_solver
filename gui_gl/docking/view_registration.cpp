#include "view_registration.h"
#include "view_registry.h"
#include "view_types.h"
#include "../core/app_services.h"
#include "../core/application.h"
#include "../GlViewer.h"
#include "../app_helpers.h"
#include "imgui.h"
#include <algorithm>

namespace {

// Helper to create a placeholder view for types not yet implemented
void RenderPlaceholder(const std::string& view_name, ViewRenderContext& ctx) {
  ImGui::BeginChild("##placeholder", ImVec2(ctx.available_width, ctx.available_height));
  ImGui::TextDisabled("%s", view_name.c_str());
  ImGui::TextDisabled("(View not yet integrated)");
  ImGui::EndChild();
}

// Wrapper to render a panel through the Application's panel registry
void RenderPanelView(const std::string& panel_id, ViewRenderContext& ctx) {
  if (!ctx.services.app) {
    RenderPlaceholder(panel_id, ctx);
    return;
  }

  // Let Application render the panel
  ctx.services.app->RenderPanelById(panel_id, ctx.available_width);
}

} // namespace

void RegisterAllViews() {
  auto& registry = ViewRegistry::Instance();

  // === Visualization Views ===

  registry.Register(ViewType::Viewer3D, {
    .display_name = "3D Viewer",
    .icon = "",
    .category = "Visualization",
    .renderer = [](ViewRenderContext& ctx) {
      if (!ctx.services.viewer) {
        RenderPlaceholder("3D Viewer", ctx);
        return;
      }
      // Render the 3D viewer to texture and display
      GlViewer& viewer = *ctx.services.viewer;
      viewer.RenderToTexture(
          static_cast<int>(ctx.available_width),
          static_cast<int>(ctx.available_height));

      ImTextureID tex_id = (ImTextureID)(uint64_t)viewer.texture();
      if (tex_id) {
        // Flip UV vertically as OpenGL textures are inverted
        ImGui::Image(tex_id, ImVec2(ctx.available_width, ctx.available_height),
                     ImVec2(0, 1), ImVec2(1, 0));

        // Get image bounds for interaction
        ImVec2 image_min = ImGui::GetItemRectMin();
        ImVec2 image_max = ImGui::GetItemRectMax();
        ImVec4 label_color(0.86f, 0.88f, 0.92f, 0.9f);
        if (ctx.services.app) {
          label_color = ctx.services.app->prefs().colors.axis_label_color;
        }
        DrawAxisLabels(viewer, image_min, image_max, label_color);
        ImGuiIO& io = ImGui::GetIO();

        // Handle mouse interaction for the viewer (check before gimbal to avoid conflict)
        bool hovered = ImGui::IsItemHovered();
        if (hovered) {
          // Mouse drag for rotation
          if (ImGui::IsMouseDragging(ImGuiMouseButton_Left)) {
            ImVec2 delta = io.MouseDelta;
            if (delta.x != 0 || delta.y != 0) {
              viewer.Rotate(delta.x, delta.y);
            }
          }

          // Mouse wheel for zoom
          if (io.MouseWheel != 0) {
            viewer.Zoom(io.MouseWheel);
          }
        }

        // Draw gimbal overlay in bottom-right corner
        // Use foreground draw list to ensure it renders on top
        constexpr float gimbal_size = 80.0f;
        constexpr float gimbal_padding = 12.0f;
        ImVec2 gimbal_top_right(image_max.x - gimbal_padding,
                                 image_max.y - gimbal_size - gimbal_padding);
        DrawGimbalForeground(viewer, gimbal_top_right, gimbal_size, io);
      }
    },
    .allow_multiple = true,
  });

  registry.Register(ViewType::Timeline, {
    .display_name = "Timeline",
    .icon = "",
    .category = "Visualization",
    .renderer = [](ViewRenderContext& ctx) {
      if (!ctx.services.app) {
        RenderPlaceholder("Timeline", ctx);
        return;
      }
      ctx.services.app->RenderTimeline(ctx.available_width, ctx.available_height);
    },
    .allow_multiple = false,
  });

  // === Configuration Views (Main Tab Panels) ===

  registry.Register(ViewType::EquationEditor, {
    .display_name = "Equation Editor",
    .icon = "",
    .category = "Configuration",
    .renderer = [](ViewRenderContext& ctx) {
      RenderPanelView("equation", ctx);
    },
    .allow_multiple = false,
  });

  registry.Register(ViewType::DomainSettings, {
    .display_name = "Domain Settings",
    .icon = "",
    .category = "Configuration",
    .renderer = [](ViewRenderContext& ctx) {
      RenderPanelView("domain", ctx);
    },
    .allow_multiple = false,
  });

  registry.Register(ViewType::GridSettings, {
    .display_name = "Grid Settings",
    .icon = "",
    .category = "Configuration",
    .renderer = [](ViewRenderContext& ctx) {
      RenderPanelView("grid", ctx);
    },
    .allow_multiple = false,
  });

  registry.Register(ViewType::BoundaryConditions, {
    .display_name = "Boundary Conditions",
    .icon = "",
    .category = "Configuration",
    .renderer = [](ViewRenderContext& ctx) {
      RenderPanelView("boundary", ctx);
    },
    .allow_multiple = false,
  });

  registry.Register(ViewType::SolverConfig, {
    .display_name = "Solver Configuration",
    .icon = "",
    .category = "Configuration",
    .renderer = [](ViewRenderContext& ctx) {
      RenderPanelView("compute", ctx);
    },
    .allow_multiple = false,
  });

  registry.Register(ViewType::TimeSettings, {
    .display_name = "Time Settings",
    .icon = "",
    .category = "Configuration",
    .renderer = [](ViewRenderContext& ctx) {
      RenderPanelView("time", ctx);
    },
    .allow_multiple = false,
  });

  registry.Register(ViewType::RunControls, {
    .display_name = "Run Controls",
    .icon = "",
    .category = "Configuration",
    .renderer = [](ViewRenderContext& ctx) {
      RenderPanelView("run", ctx);
    },
    .allow_multiple = false,
  });

  registry.Register(ViewType::LogView, {
    .display_name = "Log",
    .icon = "",
    .category = "Configuration",
    .renderer = [](ViewRenderContext& ctx) {
      RenderPanelView("log", ctx);
    },
    .allow_multiple = false,
  });

  registry.Register(ViewType::InitialConditions, {
    .display_name = "Initial Conditions",
    .icon = "",
    .category = "Configuration",
    .renderer = [](ViewRenderContext& ctx) {
      RenderPanelView("initial_conditions", ctx);
    },
    .allow_multiple = false,
  });

  registry.Register(ViewType::PresetManager, {
    .display_name = "Preset Manager",
    .icon = "",
    .category = "Configuration",
    .renderer = [](ViewRenderContext& ctx) {
      RenderPanelView("preset_manager", ctx);
    },
    .allow_multiple = false,
  });

  registry.Register(ViewType::SourceTermEditor, {
    .display_name = "Source Term Editor",
    .icon = "",
    .category = "Configuration",
    .renderer = [](ViewRenderContext& ctx) {
      RenderPanelView("source_term", ctx);
    },
    .allow_multiple = false,
  });

  registry.Register(ViewType::MaterialProperties, {
    .display_name = "Material Properties",
    .icon = "",
    .category = "Configuration",
    .renderer = [](ViewRenderContext& ctx) {
      RenderPanelView("material_properties", ctx);
    },
    .allow_multiple = false,
  });

  registry.Register(ViewType::MeshPreview, {
    .display_name = "Mesh Preview",
    .icon = "",
    .category = "Configuration",
    .renderer = [](ViewRenderContext& ctx) {
      RenderPanelView("mesh_preview", ctx);
    },
    .allow_multiple = false,
  });

  registry.Register(ViewType::ParameterSweep, {
    .display_name = "Parameter Sweep",
    .icon = "",
    .category = "Configuration",
    .renderer = [](ViewRenderContext& ctx) {
      RenderPanelView("parameter_sweep", ctx);
    },
    .allow_multiple = false,
  });

  // === Inspection Views ===

  registry.Register(ViewType::FieldSelector, {
    .display_name = "Field Selector",
    .icon = "",
    .category = "Inspection",
    .renderer = [](ViewRenderContext& ctx) {
      RenderPanelView("field_panel", ctx);
    },
    .allow_multiple = false,
  });

  registry.Register(ViewType::SliceControls, {
    .display_name = "Slice Controls",
    .icon = "",
    .category = "Inspection",
    .renderer = [](ViewRenderContext& ctx) {
      RenderPanelView("slice_panel", ctx);
    },
    .allow_multiple = false,
  });

  registry.Register(ViewType::IsosurfaceControls, {
    .display_name = "Isosurface Controls",
    .icon = "",
    .category = "Inspection",
    .renderer = [](ViewRenderContext& ctx) {
      RenderPanelView("iso_panel", ctx);
    },
    .allow_multiple = false,
  });

  registry.Register(ViewType::ImageExport, {
    .display_name = "Image Export",
    .icon = "",
    .category = "Inspection",
    .renderer = [](ViewRenderContext& ctx) {
      RenderPanelView("export_panel", ctx);
    },
    .allow_multiple = false,
  });

  registry.Register(ViewType::AdvancedInspection, {
    .display_name = "Advanced Inspection",
    .icon = "",
    .category = "Inspection",
    .renderer = [](ViewRenderContext& ctx) {
      RenderPanelView("advanced_panel", ctx);
    },
    .allow_multiple = false,
  });

  registry.Register(ViewType::ComparisonTools, {
    .display_name = "Comparison Tools",
    .icon = "",
    .category = "Inspection",
    .renderer = [](ViewRenderContext& ctx) {
      RenderPanelView("comparison_panel", ctx);
    },
    .allow_multiple = false,
  });

  registry.Register(ViewType::ConvergencePlot, {
    .display_name = "Convergence Plot",
    .icon = "",
    .category = "Inspection",
    .renderer = [](ViewRenderContext& ctx) {
      RenderPanelView("convergence_panel", ctx);
    },
    .allow_multiple = false,
  });

  registry.Register(ViewType::PointProbe, {
    .display_name = "Point Probe",
    .icon = "",
    .category = "Inspection",
    .renderer = [](ViewRenderContext& ctx) {
      RenderPanelView("point_probe", ctx);
    },
    .allow_multiple = false,
  });

  registry.Register(ViewType::StatisticsPanel, {
    .display_name = "Statistics",
    .icon = "",
    .category = "Inspection",
    .renderer = [](ViewRenderContext& ctx) {
      RenderPanelView("statistics_panel", ctx);
    },
    .allow_multiple = false,
  });

  registry.Register(ViewType::AnimationExport, {
    .display_name = "Animation Export",
    .icon = "",
    .category = "Inspection",
    .renderer = [](ViewRenderContext& ctx) {
      RenderPanelView("animation_export", ctx);
    },
    .allow_multiple = false,
  });

  // === Settings Views ===

  registry.Register(ViewType::Appearance, {
    .display_name = "Appearance",
    .icon = "",
    .category = "Settings",
    .renderer = [](ViewRenderContext& ctx) {
      RenderPanelView("colors", ctx);
    },
    .allow_multiple = false,
  });

  registry.Register(ViewType::ViewerSettings, {
    .display_name = "Viewer Settings",
    .icon = "",
    .category = "Settings",
    .renderer = [](ViewRenderContext& ctx) {
      RenderPanelView("viewer", ctx);
    },
    .allow_multiple = false,
  });

  registry.Register(ViewType::IOPaths, {
    .display_name = "I/O Paths",
    .icon = "",
    .category = "Settings",
    .renderer = [](ViewRenderContext& ctx) {
      RenderPanelView("io", ctx);
    },
    .allow_multiple = false,
  });

  registry.Register(ViewType::LatexSettings, {
    .display_name = "LaTeX Settings",
    .icon = "",
    .category = "Settings",
    .renderer = [](ViewRenderContext& ctx) {
      RenderPanelView("latex", ctx);
    },
    .allow_multiple = false,
  });

  registry.Register(ViewType::Benchmarks, {
    .display_name = "Benchmarks",
    .icon = "",
    .category = "Settings",
    .renderer = [](ViewRenderContext& ctx) {
      RenderPanelView("benchmark", ctx);
    },
    .allow_multiple = false,
  });

  registry.Register(ViewType::UIConfiguration, {
    .display_name = "UI Configuration",
    .icon = "",
    .category = "Settings",
    .renderer = [](ViewRenderContext& ctx) {
      RenderPanelView("ui_config", ctx);
    },
    .allow_multiple = false,
  });

  // === Meta Views ===

  registry.Register(ViewType::Empty, {
    .display_name = "Empty",
    .icon = "",
    .category = "Other",
    .renderer = [](ViewRenderContext& ctx) {
      ImGui::BeginChild("##empty", ImVec2(ctx.available_width, ctx.available_height));
      ImGui::TextDisabled("Select a view from the dropdown");
      ImGui::EndChild();
    },
    .allow_multiple = true,
  });
}
