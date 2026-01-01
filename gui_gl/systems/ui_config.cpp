#include "ui_config.h"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

UIConfig::UIConfig() {
  // Set default theme
  theme.name = "Dark";
  theme.font_path.clear();
  theme.font_size = 13.0f;
  theme.input_width = 320.0f;
  theme.panel_spacing = 8.0f;
  
  // Default tab order
  tab_order = {"Main", "Inspect", "Preferences"};
  
  // Default feature flags
  feature_flags["undo_redo"] = true;
  feature_flags["comparison_tools"] = true;
  feature_flags["inspection_tools"] = true;
  feature_flags["file_dialogs"] = true;
  feature_flags["image_export"] = true;
  feature_flags["latex_preview"] = true;
  feature_flags["ui_config_editor"] = true;
}

UIConfigManager& UIConfigManager::Instance() {
  static UIConfigManager instance;
  if (!instance.loaded_) {
    instance.LoadDefault();
    instance.loaded_ = true;
  }
  return instance;
}

bool UIConfigManager::LoadFromFile(const std::string& filepath) {
  std::ifstream file(filepath);
  if (!file.is_open()) {
    return false;
  }

  std::stringstream buffer;
  buffer << file.rdbuf();
  std::string content = buffer.str();

  bool result = ParseJSON(content);
  if (result) {
    // Add any missing default panels (for backwards compatibility with old configs)
    EnsureDefaultPanels();
  }
  return result;
}

void UIConfigManager::LoadDefault() {
  // Reset to defaults
  config_ = UIConfig();
  
  // Add default panels
  PanelConfig eq_panel;
  eq_panel.name = "Equation";
  eq_panel.id = "equation";
  eq_panel.components = {"pde_input", "pde_preview", "pde_templates"};
  eq_panel.collapsible = true;
  eq_panel.default_collapsed = false;
  eq_panel.order = 0;
  eq_panel.tab = "Main";
  config_.panels.push_back(eq_panel);

  PanelConfig domain_panel;
  domain_panel.name = "Domain";
  domain_panel.id = "domain";
  domain_panel.components = {"coord_system", "bounds", "implicit_shape"};
  domain_panel.collapsible = true;
  domain_panel.default_collapsed = false;
  domain_panel.order = 1;
  domain_panel.tab = "Main";
  config_.panels.push_back(domain_panel);

  PanelConfig grid_panel;
  grid_panel.name = "Grid";
  grid_panel.id = "grid";
  grid_panel.components = {"grid_resolution"};
  grid_panel.collapsible = true;
  grid_panel.default_collapsed = false;
  grid_panel.order = 2;
  grid_panel.tab = "Main";
  config_.panels.push_back(grid_panel);

  PanelConfig boundary_panel;
  boundary_panel.name = "Boundary Conditions";
  boundary_panel.id = "boundary";
  boundary_panel.components = {"bc_inputs"};
  boundary_panel.collapsible = true;
  boundary_panel.default_collapsed = false;
  boundary_panel.order = 3;
  boundary_panel.tab = "Main";
  config_.panels.push_back(boundary_panel);

  PanelConfig compute_panel;
  compute_panel.name = "Solver Configuration";
  compute_panel.id = "compute";
  compute_panel.components = {"backend", "solver_method", "solver_options"};
  compute_panel.collapsible = true;
  compute_panel.default_collapsed = false;
  compute_panel.order = 4;
  compute_panel.tab = "Main";
  config_.panels.push_back(compute_panel);

  PanelConfig time_panel;
  time_panel.name = "Time";
  time_panel.id = "time";
  time_panel.components = {"time_controls"};
  time_panel.collapsible = true;
  time_panel.default_collapsed = false;
  time_panel.order = 5;
  time_panel.tab = "Main";
  config_.panels.push_back(time_panel);

  PanelConfig run_panel;
  run_panel.name = "Run";
  run_panel.id = "run";
  run_panel.components = {"run_controls"};
  run_panel.collapsible = true;
  run_panel.default_collapsed = false;
  run_panel.order = 6;
  run_panel.tab = "Main";
  config_.panels.push_back(run_panel);

  PanelConfig log_panel;
  log_panel.name = "Log";
  log_panel.id = "log";
  log_panel.components = {"log_view"};
  log_panel.collapsible = true;
  log_panel.default_collapsed = false;
  log_panel.order = 7;
  log_panel.tab = "Main";
  config_.panels.push_back(log_panel);

  PanelConfig ic_panel;
  ic_panel.name = "Initial Conditions";
  ic_panel.id = "initial_conditions";
  ic_panel.components = {"initial_conditions_editor"};
  ic_panel.collapsible = true;
  ic_panel.default_collapsed = true;
  ic_panel.order = 8;
  ic_panel.tab = "Main";
  config_.panels.push_back(ic_panel);

  PanelConfig preset_panel;
  preset_panel.name = "Preset Manager";
  preset_panel.id = "preset_manager";
  preset_panel.components = {"preset_manager_tools"};
  preset_panel.collapsible = true;
  preset_panel.default_collapsed = true;
  preset_panel.order = 9;
  preset_panel.tab = "Main";
  config_.panels.push_back(preset_panel);

  PanelConfig source_panel;
  source_panel.name = "Source Term";
  source_panel.id = "source_term";
  source_panel.components = {"source_term_editor"};
  source_panel.collapsible = true;
  source_panel.default_collapsed = true;
  source_panel.order = 10;
  source_panel.tab = "Main";
  config_.panels.push_back(source_panel);

  PanelConfig material_panel;
  material_panel.name = "Material Properties";
  material_panel.id = "material_properties";
  material_panel.components = {"material_properties_editor"};
  material_panel.collapsible = true;
  material_panel.default_collapsed = true;
  material_panel.order = 11;
  material_panel.tab = "Main";
  config_.panels.push_back(material_panel);

  PanelConfig mesh_panel;
  mesh_panel.name = "Mesh Preview";
  mesh_panel.id = "mesh_preview";
  mesh_panel.components = {"mesh_preview_viewer"};
  mesh_panel.collapsible = true;
  mesh_panel.default_collapsed = true;
  mesh_panel.order = 12;
  mesh_panel.tab = "Main";
  config_.panels.push_back(mesh_panel);

  PanelConfig sweep_panel;
  sweep_panel.name = "Parameter Sweep";
  sweep_panel.id = "parameter_sweep";
  sweep_panel.components = {"parameter_sweep_tools"};
  sweep_panel.collapsible = true;
  sweep_panel.default_collapsed = true;
  sweep_panel.order = 13;
  sweep_panel.tab = "Main";
  config_.panels.push_back(sweep_panel);

  PanelConfig field_panel;
  field_panel.name = "Field Selector";
  field_panel.id = "field_panel";
  field_panel.components = {"field_selector"};
  field_panel.collapsible = true;
  field_panel.default_collapsed = false;
  field_panel.order = 0;
  field_panel.tab = "Inspect";
  config_.panels.push_back(field_panel);

  PanelConfig slice_panel;
  slice_panel.name = "Slice Controls";
  slice_panel.id = "slice_panel";
  slice_panel.components = {"slice_controls"};
  slice_panel.collapsible = true;
  slice_panel.default_collapsed = false;
  slice_panel.order = 1;
  slice_panel.tab = "Inspect";
  config_.panels.push_back(slice_panel);

  PanelConfig iso_panel;
  iso_panel.name = "Isosurface Controls";
  iso_panel.id = "iso_panel";
  iso_panel.components = {"isosurface_controls"};
  iso_panel.collapsible = true;
  iso_panel.default_collapsed = false;
  iso_panel.order = 2;
  iso_panel.tab = "Inspect";
  config_.panels.push_back(iso_panel);

  PanelConfig export_panel;
  export_panel.name = "Image Export";
  export_panel.id = "export_panel";
  export_panel.components = {"image_export"};
  export_panel.collapsible = true;
  export_panel.default_collapsed = false;
  export_panel.order = 3;
  export_panel.tab = "Inspect";
  config_.panels.push_back(export_panel);

  PanelConfig advanced_panel;
  advanced_panel.name = "Advanced Inspection";
  advanced_panel.id = "advanced_panel";
  advanced_panel.components = {"advanced_inspection"};
  advanced_panel.collapsible = true;
  advanced_panel.default_collapsed = true;
  advanced_panel.order = 4;
  advanced_panel.tab = "Inspect";
  config_.panels.push_back(advanced_panel);

  PanelConfig comparison_panel;
  comparison_panel.name = "Comparison Tools";
  comparison_panel.id = "comparison_panel";
  comparison_panel.components = {"comparison_tools"};
  comparison_panel.collapsible = true;
  comparison_panel.default_collapsed = true;
  comparison_panel.order = 5;
  comparison_panel.tab = "Inspect";
  config_.panels.push_back(comparison_panel);

  PanelConfig convergence_panel;
  convergence_panel.name = "Convergence Plot";
  convergence_panel.id = "convergence_panel";
  convergence_panel.components = {"convergence_plot"};
  convergence_panel.collapsible = true;
  convergence_panel.default_collapsed = false;
  convergence_panel.order = 6;
  convergence_panel.tab = "Inspect";
  config_.panels.push_back(convergence_panel);

  PanelConfig statistics_panel;
  statistics_panel.name = "Statistics";
  statistics_panel.id = "statistics_panel";
  statistics_panel.components = {"statistics_view"};
  statistics_panel.collapsible = true;
  statistics_panel.default_collapsed = false;
  statistics_panel.order = 7;
  statistics_panel.tab = "Inspect";
  config_.panels.push_back(statistics_panel);

  PanelConfig point_probe_panel;
  point_probe_panel.name = "Point Probe";
  point_probe_panel.id = "point_probe";
  point_probe_panel.components = {"point_probe_tools"};
  point_probe_panel.collapsible = true;
  point_probe_panel.default_collapsed = true;
  point_probe_panel.order = 8;
  point_probe_panel.tab = "Inspect";
  config_.panels.push_back(point_probe_panel);

  PanelConfig animation_export_panel;
  animation_export_panel.name = "Animation Export";
  animation_export_panel.id = "animation_export";
  animation_export_panel.components = {"animation_export_tools"};
  animation_export_panel.collapsible = true;
  animation_export_panel.default_collapsed = true;
  animation_export_panel.order = 9;
  animation_export_panel.tab = "Inspect";
  config_.panels.push_back(animation_export_panel);

  PanelConfig colors_panel;
  colors_panel.name = "Appearance";
  colors_panel.id = "colors";
  colors_panel.components = {"color_preferences"};
  colors_panel.collapsible = true;
  colors_panel.default_collapsed = false;
  colors_panel.order = 0;
  colors_panel.tab = "Preferences";
  config_.panels.push_back(colors_panel);

  PanelConfig viewer_panel;
  viewer_panel.name = "Viewer";
  viewer_panel.id = "viewer";
  viewer_panel.components = {"viewer_controls"};
  viewer_panel.collapsible = true;
  viewer_panel.default_collapsed = false;
  viewer_panel.order = 1;
  viewer_panel.tab = "Preferences";
  config_.panels.push_back(viewer_panel);

  PanelConfig io_panel;
  io_panel.name = "I/O";
  io_panel.id = "io";
  io_panel.components = {"io_paths"};
  io_panel.collapsible = true;
  io_panel.default_collapsed = false;
  io_panel.order = 2;
  io_panel.tab = "Preferences";
  config_.panels.push_back(io_panel);

  PanelConfig latex_panel;
  latex_panel.name = "LaTeX Preview";
  latex_panel.id = "latex";
  latex_panel.components = {"latex_settings"};
  latex_panel.collapsible = true;
  latex_panel.default_collapsed = true;
  latex_panel.order = 3;
  latex_panel.tab = "Preferences";
  config_.panels.push_back(latex_panel);

  PanelConfig benchmark_panel;
  benchmark_panel.name = "Benchmarks";
  benchmark_panel.id = "benchmark";
  benchmark_panel.components = {"benchmark_controls"};
  benchmark_panel.collapsible = true;
  benchmark_panel.default_collapsed = true;
  benchmark_panel.order = 4;
  benchmark_panel.tab = "Preferences";
  config_.panels.push_back(benchmark_panel);

  PanelConfig ui_config_panel;
  ui_config_panel.name = "UI Configuration";
  ui_config_panel.id = "ui_config";
  ui_config_panel.components = {"ui_config_editor"};
  ui_config_panel.collapsible = true;
  ui_config_panel.default_collapsed = true;
  ui_config_panel.order = 5;
  ui_config_panel.tab = "Preferences";
  config_.panels.push_back(ui_config_panel);
  
  // Add default components
  ComponentConfig pde_input;
  pde_input.name = "PDE Input";
  pde_input.type = "text_input";
  pde_input.enabled = true;
  config_.components["pde_input"] = pde_input;
  
  ComponentConfig pde_preview;
  pde_preview.name = "PDE Preview";
  pde_preview.type = "latex_preview";
  pde_preview.enabled = true;
  config_.components["pde_preview"] = pde_preview;

  ComponentConfig pde_templates;
  pde_templates.name = "PDE Templates";
  pde_templates.type = "template_picker";
  pde_templates.enabled = true;
  config_.components["pde_templates"] = pde_templates;
  
  ComponentConfig coord_system;
  coord_system.name = "Coordinate System";
  coord_system.type = "combo";
  coord_system.enabled = true;
  config_.components["coord_system"] = coord_system;
  
  ComponentConfig bounds;
  bounds.name = "Domain Bounds";
  bounds.type = "bounds_input";
  bounds.enabled = true;
  config_.components["bounds"] = bounds;

  ComponentConfig implicit_shape;
  implicit_shape.name = "Implicit Shape";
  implicit_shape.type = "shape_input";
  implicit_shape.enabled = true;
  config_.components["implicit_shape"] = implicit_shape;
  
  ComponentConfig grid_resolution;
  grid_resolution.name = "Grid Resolution";
  grid_resolution.type = "int_input";
  grid_resolution.enabled = true;
  config_.components["grid_resolution"] = grid_resolution;
  
  ComponentConfig bc_inputs;
  bc_inputs.name = "Boundary Condition Inputs";
  bc_inputs.type = "bc_inputs";
  bc_inputs.enabled = true;
  config_.components["bc_inputs"] = bc_inputs;
  
  ComponentConfig solver_method;
  solver_method.name = "Solver Method";
  solver_method.type = "combo";
  solver_method.enabled = true;
  config_.components["solver_method"] = solver_method;
  
  ComponentConfig backend;
  backend.name = "Backend";
  backend.type = "combo";
  backend.enabled = true;
  config_.components["backend"] = backend;

  ComponentConfig solver_options;
  solver_options.name = "Solver Options";
  solver_options.type = "solver_options";
  solver_options.enabled = true;
  config_.components["solver_options"] = solver_options;
  
  ComponentConfig field_selector;
  field_selector.name = "Field Selector";
  field_selector.type = "combo";
  field_selector.enabled = true;
  config_.components["field_selector"] = field_selector;
  
  ComponentConfig slice_controls;
  slice_controls.name = "Slice Controls";
  slice_controls.type = "slice_controls";
  slice_controls.enabled = true;
  config_.components["slice_controls"] = slice_controls;
  
  ComponentConfig isosurface_controls;
  isosurface_controls.name = "Isosurface Controls";
  isosurface_controls.type = "isosurface_controls";
  isosurface_controls.enabled = true;
  config_.components["isosurface_controls"] = isosurface_controls;

  ComponentConfig image_export;
  image_export.name = "Image Export";
  image_export.type = "image_export";
  image_export.enabled = true;
  config_.components["image_export"] = image_export;

  ComponentConfig advanced_inspection;
  advanced_inspection.name = "Advanced Inspection";
  advanced_inspection.type = "inspection_tools";
  advanced_inspection.enabled = true;
  config_.components["advanced_inspection"] = advanced_inspection;

  ComponentConfig comparison_tools;
  comparison_tools.name = "Comparison Tools";
  comparison_tools.type = "comparison_tools";
  comparison_tools.enabled = true;
  config_.components["comparison_tools"] = comparison_tools;

  ComponentConfig time_controls;
  time_controls.name = "Time Controls";
  time_controls.type = "time_controls";
  time_controls.enabled = true;
  config_.components["time_controls"] = time_controls;

  ComponentConfig run_controls;
  run_controls.name = "Run Controls";
  run_controls.type = "run_controls";
  run_controls.enabled = true;
  config_.components["run_controls"] = run_controls;

  ComponentConfig log_view;
  log_view.name = "Log View";
  log_view.type = "log_view";
  log_view.enabled = true;
  config_.components["log_view"] = log_view;

  ComponentConfig color_preferences;
  color_preferences.name = "Color Preferences";
  color_preferences.type = "color_preferences";
  color_preferences.enabled = true;
  config_.components["color_preferences"] = color_preferences;

  ComponentConfig viewer_controls;
  viewer_controls.name = "Viewer Controls";
  viewer_controls.type = "viewer_controls";
  viewer_controls.enabled = true;
  config_.components["viewer_controls"] = viewer_controls;

  ComponentConfig io_paths;
  io_paths.name = "I/O Paths";
  io_paths.type = "io_paths";
  io_paths.enabled = true;
  config_.components["io_paths"] = io_paths;

  ComponentConfig latex_settings;
  latex_settings.name = "LaTeX Settings";
  latex_settings.type = "latex_settings";
  latex_settings.enabled = true;
  config_.components["latex_settings"] = latex_settings;

  ComponentConfig benchmark_controls;
  benchmark_controls.name = "Benchmark Controls";
  benchmark_controls.type = "benchmark_controls";
  benchmark_controls.enabled = true;
  config_.components["benchmark_controls"] = benchmark_controls;

  ComponentConfig ui_config_editor;
  ui_config_editor.name = "UI Config Editor";
  ui_config_editor.type = "ui_config_editor";
  ui_config_editor.enabled = true;
  config_.components["ui_config_editor"] = ui_config_editor;

  ComponentConfig convergence_plot;
  convergence_plot.name = "Convergence Plot";
  convergence_plot.type = "convergence_plot";
  convergence_plot.enabled = true;
  config_.components["convergence_plot"] = convergence_plot;

  ComponentConfig statistics_view;
  statistics_view.name = "Statistics View";
  statistics_view.type = "statistics_view";
  statistics_view.enabled = true;
  config_.components["statistics_view"] = statistics_view;

  ComponentConfig point_probe_tools;
  point_probe_tools.name = "Point Probe Tools";
  point_probe_tools.type = "point_probe_tools";
  point_probe_tools.enabled = true;
  config_.components["point_probe_tools"] = point_probe_tools;

  ComponentConfig animation_export_tools;
  animation_export_tools.name = "Animation Export Tools";
  animation_export_tools.type = "animation_export_tools";
  animation_export_tools.enabled = true;
  config_.components["animation_export_tools"] = animation_export_tools;

  ComponentConfig initial_conditions_editor;
  initial_conditions_editor.name = "Initial Conditions Editor";
  initial_conditions_editor.type = "initial_conditions_editor";
  initial_conditions_editor.enabled = true;
  config_.components["initial_conditions_editor"] = initial_conditions_editor;

  ComponentConfig preset_manager_tools;
  preset_manager_tools.name = "Preset Manager Tools";
  preset_manager_tools.type = "preset_manager_tools";
  preset_manager_tools.enabled = true;
  config_.components["preset_manager_tools"] = preset_manager_tools;

  ComponentConfig source_term_editor;
  source_term_editor.name = "Source Term Editor";
  source_term_editor.type = "source_term_editor";
  source_term_editor.enabled = true;
  config_.components["source_term_editor"] = source_term_editor;

  ComponentConfig material_properties_editor;
  material_properties_editor.name = "Material Properties Editor";
  material_properties_editor.type = "material_properties_editor";
  material_properties_editor.enabled = true;
  config_.components["material_properties_editor"] = material_properties_editor;

  ComponentConfig mesh_preview_viewer;
  mesh_preview_viewer.name = "Mesh Preview Viewer";
  mesh_preview_viewer.type = "mesh_preview_viewer";
  mesh_preview_viewer.enabled = true;
  config_.components["mesh_preview_viewer"] = mesh_preview_viewer;

  ComponentConfig parameter_sweep_tools;
  parameter_sweep_tools.name = "Parameter Sweep Tools";
  parameter_sweep_tools.type = "parameter_sweep_tools";
  parameter_sweep_tools.enabled = true;
  config_.components["parameter_sweep_tools"] = parameter_sweep_tools;
}

bool UIConfigManager::SaveToFile(const std::string& filepath) const {
  std::ofstream file(filepath);
  if (!file.is_open()) {
    return false;
  }
  
  file << ToJSON();
  return file.good();
}

const PanelConfig* UIConfigManager::GetPanel(const std::string& panel_id) const {
  for (const auto& panel : config_.panels) {
    if (panel.id == panel_id) {
      return &panel;
    }
  }
  return nullptr;
}

PanelConfig* UIConfigManager::GetMutablePanel(const std::string& panel_id) {
  for (auto& panel : config_.panels) {
    if (panel.id == panel_id) {
      return &panel;
    }
  }
  return nullptr;
}

void UIConfigManager::ReorderPanel(const std::string& tab_name, const std::string& source_id, const std::string& target_id) {
  PanelConfig* src = GetMutablePanel(source_id);
  PanelConfig* dst = GetMutablePanel(target_id);
  
  if (src && dst && src->tab == tab_name && dst->tab == tab_name) {
    // Swap orders
    std::swap(src->order, dst->order);
    
    // Normalize orders for the tab to be safe (0, 1, 2...)
    auto panels = GetPanelsForTab(tab_name); // This returns const pointers, sorted by order
    // But since we just swapped, the sort order in GetPanelsForTab might be inconsistent if we don't refresh or if we rely on it.
    // Actually, simply swapping order integers is enough if they are distinct.
    // However, let's re-normalize to be clean.
    
    // We need mutable access to re-normalize.
    std::vector<PanelConfig*> tab_panels;
    for (auto& panel : config_.panels) {
      if (panel.tab == tab_name) {
        tab_panels.push_back(&panel);
      }
    }
    std::sort(tab_panels.begin(), tab_panels.end(),
              [](const PanelConfig* a, const PanelConfig* b) {
                return a->order < b->order;
              });
              
    for (size_t i = 0; i < tab_panels.size(); ++i) {
      tab_panels[i]->order = static_cast<int>(i);
    }
  }
}

const ComponentConfig* UIConfigManager::GetComponent(const std::string& component_id) const {
  auto it = config_.components.find(component_id);
  if (it != config_.components.end()) {
    return &it->second;
  }
  return nullptr;
}

bool UIConfigManager::IsFeatureEnabled(const std::string& feature_name) const {
  auto it = config_.feature_flags.find(feature_name);
  if (it != config_.feature_flags.end()) {
    return it->second;
  }
  return false;  // Default to disabled if not specified
}

std::vector<const PanelConfig*> UIConfigManager::GetPanelsForTab(const std::string& tab_name) const {
  std::vector<const PanelConfig*> result;
  for (const auto& panel : config_.panels) {
    if (panel.tab == tab_name) {
      result.push_back(&panel);
    }
  }
  // Sort by order
  std::sort(result.begin(), result.end(),
            [](const PanelConfig* a, const PanelConfig* b) {
              return a->order < b->order;
            });
  return result;
}

bool UIConfigManager::ParseJSON(const std::string& json_content) {
  try {
    json j = json::parse(json_content);
    
    // Parse metadata
    if (j.contains("metadata")) {
      const auto& meta_obj = j["metadata"];
      if (meta_obj.contains("schema_version")) config_.metadata.schema_version = meta_obj["schema_version"];
      if (meta_obj.contains("component_version")) config_.metadata.component_version = meta_obj["component_version"];
      if (meta_obj.contains("config_version")) config_.metadata.config_version = meta_obj["config_version"];
    }

    // Parse theme
    if (j.contains("theme")) {
      const auto& theme_obj = j["theme"];
      if (theme_obj.contains("name")) config_.theme.name = theme_obj["name"];
      if (theme_obj.contains("font_path")) config_.theme.font_path = theme_obj["font_path"];
      if (theme_obj.contains("font_size")) config_.theme.font_size = theme_obj["font_size"];
      if (theme_obj.contains("input_width")) config_.theme.input_width = theme_obj["input_width"];
      if (theme_obj.contains("panel_spacing")) config_.theme.panel_spacing = theme_obj["panel_spacing"];
      if (theme_obj.contains("colors")) {
        for (const auto& [key, value] : theme_obj["colors"].items()) {
          config_.theme.colors[key] = value;
        }
      }
    }
    
    // Parse panels
    if (j.contains("panels") && j["panels"].is_array()) {
      config_.panels.clear();
      for (const auto& panel_obj : j["panels"]) {
        PanelConfig panel;
        if (panel_obj.contains("id")) panel.id = panel_obj["id"];
        if (panel_obj.contains("name")) panel.name = panel_obj["name"];
        if (panel_obj.contains("tab")) panel.tab = panel_obj["tab"];
        if (panel_obj.contains("order")) panel.order = panel_obj["order"];
        if (panel_obj.contains("collapsible")) panel.collapsible = panel_obj["collapsible"];
        if (panel_obj.contains("default_collapsed")) panel.default_collapsed = panel_obj["default_collapsed"];
        if (panel_obj.contains("width")) panel.width = panel_obj["width"];
        if (panel_obj.contains("components") && panel_obj["components"].is_array()) {
          for (const auto& comp : panel_obj["components"]) {
            panel.components.push_back(comp);
          }
        }
        config_.panels.push_back(panel);
      }
    }
    
    // Parse components
    if (j.contains("components")) {
      config_.components.clear();
      for (const auto& [key, comp_obj] : j["components"].items()) {
        ComponentConfig comp;
        if (comp_obj.contains("name")) comp.name = comp_obj["name"];
        if (comp_obj.contains("type")) comp.type = comp_obj["type"];
        if (comp_obj.contains("enabled")) comp.enabled = comp_obj["enabled"];
        if (comp_obj.contains("version")) comp.version = comp_obj["version"];
        if (comp_obj.contains("width")) comp.width = comp_obj["width"];
        if (comp_obj.contains("height")) comp.height = comp_obj["height"];
        if (comp_obj.contains("properties")) {
          for (const auto& [prop_key, prop_value] : comp_obj["properties"].items()) {
            comp.properties[prop_key] = prop_value;
          }
        }
        config_.components[key] = comp;
      }
    }
    
    // Parse feature flags
    if (j.contains("feature_flags")) {
      config_.feature_flags.clear();
      for (const auto& [key, value] : j["feature_flags"].items()) {
        config_.feature_flags[key] = value;
      }
    }
    
    // Parse layout
    if (j.contains("layout")) {
      const auto& layout_obj = j["layout"];
      if (layout_obj.contains("left_panel_min_width")) config_.left_panel_min_width = layout_obj["left_panel_min_width"];
      if (layout_obj.contains("left_panel_max_width")) config_.left_panel_max_width = layout_obj["left_panel_max_width"];
      if (layout_obj.contains("right_panel_min_width")) config_.right_panel_min_width = layout_obj["right_panel_min_width"];
      if (layout_obj.contains("splitter_width")) config_.splitter_width = layout_obj["splitter_width"];
    }
    
    // Parse tab order
    if (j.contains("tab_order") && j["tab_order"].is_array()) {
      config_.tab_order.clear();
      for (const auto& tab : j["tab_order"]) {
        config_.tab_order.push_back(tab);
      }
    }
    
    return true;
  } catch (const json::exception& e) {
    // Log error if we had a logger
    return false;
  }
}

std::string UIConfigManager::ToJSON() const {
  try {
    json j;
    
    // Metadata
    j["metadata"]["schema_version"] = config_.metadata.schema_version;
    j["metadata"]["component_version"] = config_.metadata.component_version;
    j["metadata"]["config_version"] = config_.metadata.config_version;

    // Theme
    j["theme"]["name"] = config_.theme.name;
    j["theme"]["font_path"] = config_.theme.font_path;
    j["theme"]["font_size"] = config_.theme.font_size;
    j["theme"]["input_width"] = config_.theme.input_width;
    j["theme"]["panel_spacing"] = config_.theme.panel_spacing;
    for (const auto& [key, value] : config_.theme.colors) {
      j["theme"]["colors"][key] = value;
    }
    
    // Panels
    j["panels"] = json::array();
    for (const auto& panel : config_.panels) {
      json panel_obj;
      panel_obj["id"] = panel.id;
      panel_obj["name"] = panel.name;
      panel_obj["tab"] = panel.tab;
      panel_obj["order"] = panel.order;
      panel_obj["collapsible"] = panel.collapsible;
      panel_obj["default_collapsed"] = panel.default_collapsed;
      panel_obj["width"] = panel.width;
      panel_obj["components"] = panel.components;
      j["panels"].push_back(panel_obj);
    }
    
    // Components
    for (const auto& [key, comp] : config_.components) {
      json comp_obj;
      comp_obj["name"] = comp.name;
      comp_obj["type"] = comp.type;
      comp_obj["enabled"] = comp.enabled;
      comp_obj["version"] = comp.version;
      comp_obj["width"] = comp.width;
      comp_obj["height"] = comp.height;
      for (const auto& [prop_key, prop_value] : comp.properties) {
        comp_obj["properties"][prop_key] = prop_value;
      }
      j["components"][key] = comp_obj;
    }
    
    // Feature flags
    for (const auto& [key, value] : config_.feature_flags) {
      j["feature_flags"][key] = value;
    }
    
    // Layout
    j["layout"]["left_panel_min_width"] = config_.left_panel_min_width;
    j["layout"]["left_panel_max_width"] = config_.left_panel_max_width;
    j["layout"]["right_panel_min_width"] = config_.right_panel_min_width;
    j["layout"]["splitter_width"] = config_.splitter_width;
    
    // Tab order
    j["tab_order"] = config_.tab_order;
    
    return j.dump(2);  // Pretty print with 2-space indent
  } catch (const json::exception& e) {
    return "{}";  // Return empty JSON on error
  }
}

void UIConfigManager::EnsureDefaultPanels() {
  // Define all default panels that should exist
  struct DefaultPanel {
    std::string id;
    std::string name;
    std::string tab;
    int order;
    bool default_collapsed;
    std::vector<std::string> components;
  };

  std::vector<DefaultPanel> default_panels = {
    // Main tab panels
    {"equation", "Equation", "Main", 0, false, {"pde_input", "pde_preview", "pde_templates"}},
    {"domain", "Domain", "Main", 1, false, {"coord_system", "bounds", "implicit_shape"}},
    {"grid", "Grid", "Main", 2, false, {"grid_resolution"}},
    {"boundary", "Boundary Conditions", "Main", 3, false, {"bc_inputs"}},
    {"compute", "Solver Configuration", "Main", 4, false, {"backend", "solver_method", "solver_options"}},
    {"time", "Time", "Main", 5, false, {"time_controls"}},
    {"run", "Run", "Main", 6, false, {"run_controls"}},
    {"log", "Log", "Main", 7, false, {"log_view"}},
    {"initial_conditions", "Initial Conditions", "Main", 8, true, {"initial_conditions_editor"}},
    {"preset_manager", "Preset Manager", "Main", 9, true, {"preset_manager_tools"}},
    {"source_term", "Source Term", "Main", 10, true, {"source_term_editor"}},
    {"material_properties", "Material Properties", "Main", 11, true, {"material_properties_editor"}},
    {"mesh_preview", "Mesh Preview", "Main", 12, true, {"mesh_preview_viewer"}},
    {"parameter_sweep", "Parameter Sweep", "Main", 13, true, {"parameter_sweep_tools"}},
    {"testing", "Testing", "Main", 14, true, {"testing_tools"}},

    // Inspect tab panels
    {"field_panel", "Field Selector", "Inspect", 0, false, {"field_selector"}},
    {"slice_panel", "Slice Controls", "Inspect", 1, false, {"slice_controls"}},
    {"iso_panel", "Isosurface Controls", "Inspect", 2, false, {"isosurface_controls"}},
    {"export_panel", "Image Export", "Inspect", 3, false, {"image_export"}},
    {"advanced_panel", "Advanced Inspection", "Inspect", 4, true, {"advanced_inspection"}},
    {"comparison_panel", "Comparison Tools", "Inspect", 5, true, {"comparison_tools"}},
    {"convergence_panel", "Convergence Plot", "Inspect", 6, false, {"convergence_plot"}},
    {"statistics_panel", "Statistics", "Inspect", 7, false, {"statistics_view"}},
    {"point_probe", "Point Probe", "Inspect", 8, true, {"point_probe_tools"}},
    {"animation_export", "Animation Export", "Inspect", 9, true, {"animation_export_tools"}},

    // Preferences tab panels
    {"colors", "Appearance", "Preferences", 0, false, {"color_preferences"}},
    {"viewer", "Viewer", "Preferences", 1, false, {"viewer_controls"}},
    {"io", "I/O", "Preferences", 2, false, {"io_paths"}},
    {"latex", "LaTeX Preview", "Preferences", 3, true, {"latex_settings"}},
    {"benchmark", "Benchmarks", "Preferences", 4, true, {"benchmark_controls"}},
    {"ui_config", "UI Configuration", "Preferences", 5, true, {"ui_config_editor"}},
  };

  // Check which panels are missing and add them
  std::set<std::string> existing_ids;
  for (const auto& panel : config_.panels) {
    existing_ids.insert(panel.id);
  }

  for (const auto& def : default_panels) {
    if (existing_ids.find(def.id) == existing_ids.end()) {
      // Panel is missing, add it
      PanelConfig panel;
      panel.id = def.id;
      panel.name = def.name;
      panel.tab = def.tab;
      panel.order = def.order;
      panel.collapsible = true;
      panel.default_collapsed = def.default_collapsed;
      panel.components = def.components;
      config_.panels.push_back(panel);
    }
  }
}
