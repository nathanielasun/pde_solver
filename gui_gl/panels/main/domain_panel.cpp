#include "domain_panel.h"
#include "imgui.h"
#include "imgui_stdlib.h"
#include "app_helpers.h"
#include "validation.h"
#include "systems/coordinate_system_registry.h"
#include "systems/command_history.h"
#include "utils/coordinate_utils.h"
#include "styles/ui_style.h"
#include "ui_helpers.h"
#include "shape_io.h"
#include "utils/file_dialog.h"
#include <string>
#include <vector>
#include <sstream>
#include <cmath>
#include <utility>

namespace {

std::vector<std::string> DefaultComponents() {
  return {"coord_system", "bounds", "implicit_shape"};
}

void RenderCoordSystemSection(DomainPanelState& state) {
  ImGui::Text("Coordinate System");
  
  static std::vector<std::string> coord_item_strings;
  static std::vector<const char*> coord_items;
  static bool items_initialized = false;
  
  if (!items_initialized) {
    const int coord_mode_to_system[] = {
      CoordMode::kCartesian2D,
      CoordMode::kCartesian3D,
      CoordMode::kPolar,
      CoordMode::kAxisymmetric,
      CoordMode::kCylindricalVolume,
      CoordMode::kSphericalSurface,
      CoordMode::kSphericalVolume,
      CoordMode::kToroidalSurface,
      CoordMode::kToroidalVolume,
    };
    
    coord_item_strings.clear();
    coord_items.clear();
    coord_item_strings.reserve(9);
    
    const char* fallback_names[] = {
      "Cartesian (x, y)",
      "Cartesian (x, y, z)",
      "Polar (r, theta)",
      "Axisymmetric (r, z)",
      "Cylindrical volume (r, theta, z)",
      "Spherical surface (theta, phi)",
      "Spherical volume (r, theta, phi)",
      "Toroidal surface (theta, phi)",
      "Toroidal volume (r, theta, phi)",
    };
    
    for (int i = 0; i < 9; ++i) {
      CoordinateSystem cs = CoordModeToSystem(coord_mode_to_system[i]);
      const CoordinateSystemMetadata* metadata =
          CoordinateSystemRegistry::Instance().GetMetadata(cs);
      if (metadata) {
        coord_item_strings.push_back(metadata->name);
      } else {
        coord_item_strings.push_back(fallback_names[i]);
      }
    }
    
    coord_items.reserve(coord_item_strings.size());
    for (const auto& item : coord_item_strings) {
      coord_items.push_back(item.c_str());
    }
    items_initialized = true;
  }
  
  int current_coord_mode = state.coord_mode;
  if (current_coord_mode < 0 || current_coord_mode >= static_cast<int>(coord_items.size())) {
    current_coord_mode = 0;
    state.coord_mode = current_coord_mode;
  }
  
  int old_coord_mode = state.coord_mode;
  ImGui::SetNextItemWidth(state.input_width);
  if (UIInput::Combo("System", &current_coord_mode, coord_items.data(),
                     static_cast<int>(coord_items.size()))) {
    if (current_coord_mode < 0) {
      current_coord_mode = 0;
    } else if (current_coord_mode >= static_cast<int>(coord_items.size())) {
      current_coord_mode = static_cast<int>(coord_items.size()) - 1;
    }
    
    if (state.cmd_history && current_coord_mode != old_coord_mode) {
      auto cmd = std::make_unique<SetIntCommand>(&state.coord_mode, current_coord_mode,
                                                 "Change coordinate system");
      state.cmd_history->Execute(std::move(cmd));
    } else {
      state.coord_mode = current_coord_mode;
    }
    
    state.viewer.SetViewMode(ViewModeForCoord(state.coord_mode));
    state.viewer.SetTorusRadii(static_cast<float>(state.torus_major),
                               static_cast<float>(state.torus_minor));
    if (state.refresh_coord_flags) {
      state.refresh_coord_flags();
    }
  }
  
  CoordinateSystem current_system = CoordModeToSystem(state.coord_mode);
  const CoordinateSystemMetadata* metadata =
      CoordinateSystemRegistry::Instance().GetMetadata(current_system);
  if (metadata && ImGui::IsItemHovered()) {
    ImGui::BeginTooltip();
    ImGui::TextWrapped("%s", metadata->description.c_str());
    if (!metadata->common_applications.empty()) {
      ImGui::Separator();
      ImGui::Text("Common Applications:");
      for (const auto& app : metadata->common_applications) {
        ImGui::BulletText("%s", app.c_str());
      }
    }
    if (!metadata->example_domain.empty()) {
      ImGui::Separator();
      ImGui::Text("Example Domain:");
      ImGui::Text("%s", metadata->example_domain.c_str());
    }
    ImGui::EndTooltip();
  }
}

void RenderBoundsSection(DomainPanelState& state) {
  CoordinateSystem current_system = CoordModeToSystem(state.coord_mode);
  const CoordinateSystemMetadata* metadata =
      CoordinateSystemRegistry::Instance().GetMetadata(current_system);
  
  std::string bounds_label = "Bounding Box";
  if (metadata) {
    bounds_label = metadata->dimension == 3 ? metadata->name + " Bounds (3D)" : metadata->name + " Bounds";
  } else if (state.use_polar_coords) {
    bounds_label = "Radial Bounds";
  } else if (state.use_cartesian_3d) {
    bounds_label = "Bounding Box (3D)";
  } else if (state.use_axisymmetric) {
    bounds_label = "Axisymmetric Bounds";
  } else if (state.use_cylindrical_volume) {
    bounds_label = "Cylindrical Bounds";
  } else if (state.use_toroidal_surface) {
    bounds_label = "Toroidal Surface Bounds";
  } else if (state.use_toroidal_volume) {
    bounds_label = "Toroidal Volume Bounds";
  } else if (state.use_surface) {
    bounds_label = "Angular Bounds";
  } else if (state.use_volume) {
    bounds_label = "Spherical Bounds";
  }
  
  ImGui::Text("%s", bounds_label.c_str());
  
  const float avail = ImGui::GetContentRegionAvail().x;
  const bool compact = avail < 360.0f;
  const float field_width = std::max(110.0f, std::min(180.0f, state.input_width * (compact ? 0.9f : 0.45f)));
  
  std::string x_min_label = "xmin";
  std::string x_max_label = "xmax";
  std::string y_min_label = "ymin";
  std::string y_max_label = "ymax";
  std::string z_min_label = "z min";
  std::string z_max_label = "z max";
  
  if (metadata && metadata->axes.size() >= 2) {
    x_min_label = metadata->axes[0].symbol + " min";
    x_max_label = metadata->axes[0].symbol + " max";
    y_min_label = metadata->axes[1].symbol + " min";
    y_max_label = metadata->axes[1].symbol + " max";
    if (metadata->dimension == 3 && metadata->axes.size() >= 3) {
      z_min_label = metadata->axes[2].symbol + " min";
      z_max_label = metadata->axes[2].symbol + " max";
    }
  } else if (state.use_axisymmetric) {
    x_min_label = "r min";
    x_max_label = "r max";
    y_min_label = "z min";
    y_max_label = "z max";
  } else if (state.use_polar_coords) {
    x_min_label = "r min";
    x_max_label = "r max";
    y_min_label = "theta min";
    y_max_label = "theta max";
  } else if (state.use_cylindrical_volume) {
    x_min_label = "r min";
    x_max_label = "r max";
    y_min_label = "theta min";
    y_max_label = "theta max";
    z_min_label = "z min";
    z_max_label = "z max";
  } else if (state.use_surface) {
    x_min_label = "theta min";
    x_max_label = "theta max";
    y_min_label = "phi min";
    y_max_label = "phi max";
  } else if (state.use_volume) {
    x_min_label = "r min";
    x_max_label = "r max";
    y_min_label = "theta min";
    y_max_label = "theta max";
    z_min_label = "phi min";
    z_max_label = "phi max";
  }
  
  double old_xmin = state.bound_xmin;
  double old_xmax = state.bound_xmax;
  double old_ymin = state.bound_ymin;
  double old_ymax = state.bound_ymax;
  double old_zmin = state.bound_zmin;
  double old_zmax = state.bound_zmax;
  
  ImGui::PushItemWidth(field_width);
  if (UIInput::InputDouble(x_min_label.c_str(), &state.bound_xmin, 0.0, 0.0, "%.6g")) {
    if (state.cmd_history && state.bound_xmin != old_xmin) {
      auto cmd = std::make_unique<SetDoubleCommand>(&state.bound_xmin, state.bound_xmin,
                                                    "Change " + x_min_label);
      state.cmd_history->Execute(std::move(cmd));
    }
  }
  if (!compact) {
    ImGui::SameLine();
  }
  if (UIInput::InputDouble(x_max_label.c_str(), &state.bound_xmax, 0.0, 0.0, "%.6g")) {
    if (state.cmd_history && state.bound_xmax != old_xmax) {
      auto cmd = std::make_unique<SetDoubleCommand>(&state.bound_xmax, state.bound_xmax,
                                                    "Change " + x_max_label);
      state.cmd_history->Execute(std::move(cmd));
    }
  }
  if (UIInput::InputDouble(y_min_label.c_str(), &state.bound_ymin, 0.0, 0.0, "%.6g")) {
    if (state.cmd_history && state.bound_ymin != old_ymin) {
      auto cmd = std::make_unique<SetDoubleCommand>(&state.bound_ymin, state.bound_ymin,
                                                    "Change " + y_min_label);
      state.cmd_history->Execute(std::move(cmd));
    }
  }
  if (!compact) {
    ImGui::SameLine();
  }
  if (UIInput::InputDouble(y_max_label.c_str(), &state.bound_ymax, 0.0, 0.0, "%.6g")) {
    if (state.cmd_history && state.bound_ymax != old_ymax) {
      auto cmd = std::make_unique<SetDoubleCommand>(&state.bound_ymax, state.bound_ymax,
                                                    "Change " + y_max_label);
      state.cmd_history->Execute(std::move(cmd));
    }
  }
  if (state.use_volume || (metadata && metadata->dimension == 3)) {
    if (UIInput::InputDouble(z_min_label.c_str(), &state.bound_zmin, 0.0, 0.0, "%.6g")) {
      if (state.cmd_history && state.bound_zmin != old_zmin) {
        auto cmd = std::make_unique<SetDoubleCommand>(&state.bound_zmin, state.bound_zmin,
                                                      "Change " + z_min_label);
        state.cmd_history->Execute(std::move(cmd));
      }
    }
    if (!compact) {
      ImGui::SameLine();
    }
    if (UIInput::InputDouble(z_max_label.c_str(), &state.bound_zmax, 0.0, 0.0, "%.6g")) {
      if (state.cmd_history && state.bound_zmax != old_zmax) {
        auto cmd = std::make_unique<SetDoubleCommand>(&state.bound_zmax, state.bound_zmax,
                                                      "Change " + z_max_label);
        state.cmd_history->Execute(std::move(cmd));
      }
    }
  }
  ImGui::PopItemWidth();
  
  ValidationState domain_validation = ValidateDomain(
      state.bound_xmin, state.bound_xmax, state.bound_ymin, state.bound_ymax,
      state.bound_zmin, state.bound_zmax, state.coord_mode, state.use_volume);
  
  if (domain_validation.domain_status == ValidationStatus::Error) {
    std::string msg = "Bounds must satisfy xmin < xmax and ymin < ymax.";
    if (metadata) {
      std::ostringstream msg_oss;
      if (metadata->dimension == 2 && metadata->axes.size() >= 2) {
        msg_oss << "Bounds must satisfy " << metadata->axes[0].symbol
                << " min < " << metadata->axes[0].symbol << " max and "
                << metadata->axes[1].symbol << " min < " << metadata->axes[1].symbol << " max.";
      } else if (metadata->dimension == 3 && metadata->axes.size() >= 3) {
        msg_oss << "Bounds must satisfy " << metadata->axes[0].symbol
                << " min < " << metadata->axes[0].symbol << " max, "
                << metadata->axes[1].symbol << " min < " << metadata->axes[1].symbol << " max, and "
                << metadata->axes[2].symbol << " min < " << metadata->axes[2].symbol << " max.";
      }
      if (!msg_oss.str().empty()) {
        msg = msg_oss.str();
      }
    } else if (state.use_axisymmetric) {
      msg = "Bounds must satisfy r min < r max and z min < z max.";
    } else if (state.use_polar_coords) {
      msg = "Bounds must satisfy r min < r max and theta min < theta max.";
    } else if (state.use_cylindrical_volume) {
      msg = "Bounds must satisfy r, theta, z min < max.";
    } else if (state.use_surface) {
      msg = "Bounds must satisfy theta min < theta max and phi min < phi max.";
    } else if (state.use_volume) {
      msg = "Bounds must satisfy r, theta, phi min < max.";
    } else if (state.use_cartesian_3d) {
      msg = "Bounds must satisfy xmin < xmax, ymin < ymax, and z min < z max.";
    }
    ImGui::TextColored(ImVec4(1.0f, 0.55f, 0.55f, 1.0f), "%s", msg.c_str());
  }
  
  if (metadata) {
    bool has_angular_axis = false;
    for (const auto& axis : metadata->axes) {
      if (axis.is_angular) {
        has_angular_axis = true;
        break;
      }
    }
    if (has_angular_axis) {
      ImGui::TextDisabled("Angular coordinates are in radians.");
    }
  } else {
    const bool use_angle_coords =
        state.use_polar_coords || state.use_surface || state.use_cylindrical_volume ||
        state.use_spherical_volume || state.use_toroidal_volume;
    if (use_angle_coords) {
      ImGui::TextDisabled("theta/phi are in radians.");
    }
  }
  
  if (state.use_toroidal_surface || state.use_toroidal_volume) {
    ImGui::Spacing();
    ImGui::Text("Torus Geometry");
    ImGui::PushItemWidth(std::max(120.0f, field_width));
    
    double old_torus_major = state.torus_major;
    double old_torus_minor = state.torus_minor;
    
    bool torus_changed = false;
    if (UIInput::InputDouble("Major radius (R)", &state.torus_major, 0.0, 0.0, "%.6g")) {
      if (state.cmd_history && state.torus_major != old_torus_major) {
        auto cmd = std::make_unique<SetDoubleCommand>(&state.torus_major, state.torus_major,
                                                      "Change torus major radius");
        state.cmd_history->Execute(std::move(cmd));
      }
      torus_changed = true;
    }
    if (state.use_toroidal_surface) {
      if (!compact) {
        ImGui::SameLine();
      }
      if (UIInput::InputDouble("Minor radius (r)", &state.torus_minor, 0.0, 0.0, "%.6g")) {
        if (state.cmd_history && state.torus_minor != old_torus_minor) {
          auto cmd = std::make_unique<SetDoubleCommand>(&state.torus_minor, state.torus_minor,
                                                        "Change torus minor radius");
          state.cmd_history->Execute(std::move(cmd));
        }
        torus_changed = true;
      }
    } else {
      ImGui::TextDisabled("Minor radius uses the r bounds.");
    }
    ImGui::PopItemWidth();
    if (state.torus_major <= 0.0) {
      state.torus_major = 1.0;
      torus_changed = true;
    }
    if (state.torus_minor <= 0.0) {
      state.torus_minor = 0.1;
      torus_changed = true;
    }
    if (torus_changed) {
      state.viewer.SetTorusRadii(static_cast<float>(state.torus_major),
                                 static_cast<float>(state.torus_minor));
    }
  }
}

void RenderImplicitShapeSection(DomainPanelState& state) {
  bool use_implicit = (state.domain_mode == 1);
  int old_domain_mode = state.domain_mode;
  if (ImGui::Checkbox("Use implicit domain condition", &use_implicit)) {
    int new_domain_mode = use_implicit ? 1 : 0;
    if (state.cmd_history && new_domain_mode != old_domain_mode) {
      auto cmd = std::make_unique<SetIntCommand>(&state.domain_mode, new_domain_mode,
                                                 "Change domain mode");
      state.cmd_history->Execute(std::move(cmd));
    } else {
      state.domain_mode = new_domain_mode;
    }
  }
  if (state.domain_mode != 1) {
    return;
  }
  
  const bool show_z = state.use_cartesian_3d || state.use_cylindrical_volume || state.use_volume;
  const float field_width = std::max(70.0f, state.input_width * (show_z ? 0.25f : 0.35f));

  auto update_double = [&](const char* label, double* value, const char* desc) {
    const double old_value = *value;
    if (UIInput::InputDouble(label, value, 0.0, 0.0, "%.6g")) {
      if (state.cmd_history && *value != old_value) {
        auto cmd = std::make_unique<SetDoubleCommand>(value, *value, desc);
        state.cmd_history->Execute(std::move(cmd));
      }
    }
  };

  auto update_bool = [&](const char* label, bool* value, const char* desc) {
    const bool old_value = *value;
    if (ImGui::Checkbox(label, value)) {
      if (state.cmd_history && *value != old_value) {
        auto cmd = std::make_unique<SetBoolCommand>(value, *value, desc);
        state.cmd_history->Execute(std::move(cmd));
      }
    }
  };

  ImGui::Text("Shape Sources");
  if (UIButton::Button("Load Shape File", UIButton::Size::Small, UIButton::Variant::Secondary)) {
    auto selected = FileDialog::PickFile("Select Shape Expression", {},
                                         "Shape Files", {"txt", "tex", "latex", "md"});
    if (selected) {
      std::string expr;
      std::string load_error;
      if (!LoadShapeExpressionFromFile(selected->string(), &expr, &load_error)) {
        UIToast::Show(UIToast::Type::Error,
                      load_error.empty() ? "Failed to load shape file" : load_error);
      } else {
        state.domain_shape_file = selected->string();
        state.domain_shape = expr;
        UIToast::Show(UIToast::Type::Success, "Shape file loaded");
      }
    }
  }
  ImGui::SameLine();
  if (UIButton::Button("Load Mask (VTK)", UIButton::Size::Small, UIButton::Variant::Secondary)) {
    auto selected = FileDialog::PickFile("Select Shape Mask", {},
                                         "VTK Files", {"vtk", "vti"});
    if (selected) {
      ShapeMask loaded;
      std::string load_error;
      if (!LoadShapeMaskFromVtk(selected->string(), &loaded, &load_error)) {
        UIToast::Show(UIToast::Type::Error,
                      load_error.empty() ? "Failed to load shape mask" : load_error);
      } else {
        state.domain_shape_mask_path = selected->string();
        state.shape_mask = std::move(loaded);
        UIToast::Show(UIToast::Type::Success, "Shape mask loaded");
      }
    }
  }

  if (!state.domain_shape_file.empty()) {
    ImGui::TextWrapped("Shape file: %s", state.domain_shape_file.c_str());
    ImGui::SameLine();
    if (UIButton::Button("Clear##shape_file", UIButton::Size::Small,
                         UIButton::Variant::Secondary)) {
      state.domain_shape_file.clear();
    }
  }

  const bool has_mask = HasShapeMask(state.shape_mask);
  if (!state.domain_shape_mask_path.empty()) {
    ImGui::TextWrapped("Mask: %s", state.domain_shape_mask_path.c_str());
    ImGui::SameLine();
    if (UIButton::Button("Clear##shape_mask", UIButton::Size::Small,
                         UIButton::Variant::Secondary)) {
      state.domain_shape_mask_path.clear();
      state.shape_mask = ShapeMask();
    }
  }

  if (has_mask) {
    const Domain& md = state.shape_mask.domain;
    ImGui::Text("Mask grid: %d x %d x %d", md.nx, md.ny, std::max(1, md.nz));
    ImGui::Text("Mask bounds: [%.3g, %.3g] x [%.3g, %.3g]",
                md.xmin, md.xmax, md.ymin, md.ymax);
    if (show_z) {
      ImGui::Text("Mask z: [%.3g, %.3g]", md.zmin, md.zmax);
    }
  }

  ImGui::Spacing();
  ImGui::Text("Domain Condition");
  ImGui::SetNextItemWidth(state.input_width);
  const char* shape_label = "f(x,y) <= 0";
  if (state.use_axisymmetric) {
    shape_label = "f(r,z) <= 0";
  } else if (state.use_polar_coords) {
    shape_label = "f(r,theta) <= 0";
  } else if (state.use_cylindrical_volume) {
    shape_label = "f(r,theta,z) <= 0";
  } else if (state.use_surface) {
    shape_label = "f(theta,phi) <= 0";
  } else if (state.use_volume) {
    shape_label = "f(r,theta,phi) <= 0";
  } else if (state.use_cartesian_3d) {
    shape_label = "f(x,y,z) <= 0";
  }
  
  std::string old_domain_shape = state.domain_shape;
  if (UIInput::InputTextMultiline(shape_label, &state.domain_shape,
                                  ImVec2(state.input_width, 48),
                                  ImGuiInputTextFlags_WordWrap)) {
    if (state.cmd_history && state.domain_shape != old_domain_shape) {
      auto cmd = std::make_unique<SetStringCommand>(&state.domain_shape, state.domain_shape,
                                                    "Change domain shape");
      state.cmd_history->Execute(std::move(cmd));
    }
  }
  
  std::string shape_prefix;
  if (state.use_axisymmetric) {
    shape_prefix = "f(r,z) = ";
  } else if (state.use_polar_coords) {
    shape_prefix = "f(r,\\theta) = ";
  } else if (state.use_cylindrical_volume) {
    shape_prefix = "f(r,\\theta,z) = ";
  } else if (state.use_surface) {
    shape_prefix = "f(\\theta,\\phi) = ";
  } else if (state.use_volume) {
    shape_prefix = "f(r,\\theta,\\phi) = ";
  } else if (state.use_cartesian_3d) {
    shape_prefix = "f(x,y,z) = ";
  } else {
    shape_prefix = "f(x,y) = ";
  }
  const std::string shape_latex =
      state.domain_shape.empty() ? std::string() : shape_prefix + state.domain_shape;
  UpdateLatexTexture(state.shape_preview, shape_latex, state.python_path,
                     state.script_path, state.cache_dir, state.latex_color,
                     state.latex_font_size);
  if (!state.shape_preview.error.empty()) {
    ImGui::TextColored(ImVec4(1.0f, 0.5f, 0.5f, 1.0f), "%s", state.shape_preview.error.c_str());
  } else if (state.shape_preview.texture != 0) {
    DrawLatexPreview(state.shape_preview, state.input_width, 110.0f);
  }
  if (has_mask) {
    ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.3f, 1.0f),
                       "Mask overrides the expression.");
  }

  if (has_mask || !state.domain_shape_mask_path.empty()) {
    ImGui::Spacing();
    ImGui::Text("Mask Options");
    ImGui::SetNextItemWidth(state.input_width * 0.6f);
    update_double("Threshold", &state.shape_mask_threshold, "Change mask threshold");
    update_bool("Invert mask", &state.shape_mask_invert, "Invert mask");
  }

  ImGui::Spacing();
  ImGui::Text("Shape Transform");
  if (!has_mask) {
    ImGui::BeginDisabled();
  }
  if (UIButton::Button("Fit Mask to Domain", UIButton::Size::Small, UIButton::Variant::Secondary)) {
    const Domain& md = state.shape_mask.domain;
    auto fit_axis = [](double dom_min, double dom_max, double mask_min, double mask_max,
                       double* offset, double* scale) {
      const double dom_span = dom_max - dom_min;
      const double mask_span = mask_max - mask_min;
      if (std::abs(dom_span) < 1e-12 || std::abs(mask_span) < 1e-12) {
        return;
      }
      const double new_scale = dom_span / mask_span;
      if (scale) {
        *scale = new_scale;
      }
      if (offset) {
        *offset = dom_min - new_scale * mask_min;
      }
    };
    fit_axis(state.bound_xmin, state.bound_xmax, md.xmin, md.xmax,
             &state.shape_transform.offset_x, &state.shape_transform.scale_x);
    fit_axis(state.bound_ymin, state.bound_ymax, md.ymin, md.ymax,
             &state.shape_transform.offset_y, &state.shape_transform.scale_y);
    if (show_z) {
      fit_axis(state.bound_zmin, state.bound_zmax, md.zmin, md.zmax,
               &state.shape_transform.offset_z, &state.shape_transform.scale_z);
    }
  }
  if (!has_mask) {
    ImGui::EndDisabled();
  }
  ImGui::SameLine();
  if (UIButton::Button("Reset Transform", UIButton::Size::Small, UIButton::Variant::Secondary)) {
    state.shape_transform = ShapeTransform();
  }

  ImGui::PushItemWidth(field_width);
  ImGui::Text("Offset");
  update_double("X##shape_offset", &state.shape_transform.offset_x, "Change shape offset X");
  ImGui::SameLine();
  update_double("Y##shape_offset", &state.shape_transform.offset_y, "Change shape offset Y");
  if (show_z) {
    ImGui::SameLine();
    update_double("Z##shape_offset", &state.shape_transform.offset_z, "Change shape offset Z");
  }
  ImGui::Text("Scale");
  update_double("X##shape_scale", &state.shape_transform.scale_x, "Change shape scale X");
  ImGui::SameLine();
  update_double("Y##shape_scale", &state.shape_transform.scale_y, "Change shape scale Y");
  if (show_z) {
    ImGui::SameLine();
    update_double("Z##shape_scale", &state.shape_transform.scale_z, "Change shape scale Z");
  }
  ImGui::PopItemWidth();

  ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.3f, 1.0f),
                     "Implicit domains use CPU fallback.");
}

}  // namespace

void RenderDomainPanel(DomainPanelState& state, const std::vector<std::string>& components) {
  const std::vector<std::string> default_components = DefaultComponents();
  const std::vector<std::string>& component_list =
      components.empty() ? default_components : components;
  
  for (size_t i = 0; i < component_list.size(); ++i) {
    const std::string& id = component_list[i];
    if (id == "coord_system") {
      RenderCoordSystemSection(state);
    } else if (id == "bounds") {
      RenderBoundsSection(state);
    } else if (id == "implicit_shape") {
      RenderImplicitShapeSection(state);
    } else {
      DrawUnknownComponentPlaceholder(id.c_str());
    }
    
    if (i + 1 < component_list.size()) {
      ImGui::Spacing();
    }
  }
}
