#ifndef VIEW_TYPES_H
#define VIEW_TYPES_H

#include <string>
#include <unordered_map>

// All available view types in the docking system
enum class ViewType {
  // Visualization
  Viewer3D,           // Main OpenGL visualization
  Timeline,           // Animation timeline + playback controls

  // Main tab panels (solver configuration)
  EquationEditor,     // PDE input + LaTeX preview
  DomainSettings,     // Coordinate system + bounds
  GridSettings,       // Resolution configuration
  BoundaryConditions, // BC inputs for all faces
  SolverConfig,       // Backend + method selection
  TimeSettings,       // Time-dependent parameters
  RunControls,        // Solve/Stop + progress
  LogView,            // Debug output
  InitialConditions,  // Initial condition u(x,y,z,0)
  PresetManager,      // Save/load problem presets
  SourceTermEditor,   // Source function f(x,y,z,t)
  MaterialProperties, // Variable coefficients
  MeshPreview,        // Grid wireframe preview
  ParameterSweep,     // Batch parameter studies

  // Inspect tab panels
  FieldSelector,      // Field type dropdown
  SliceControls,      // Slice plane settings
  IsosurfaceControls, // Isosurface settings
  ImageExport,        // Export functionality
  AdvancedInspection, // Probes, line profiles
  ComparisonTools,    // Field comparison
  ConvergencePlot,    // Residual history graph
  PointProbe,         // Point value query
  StatisticsPanel,    // Field statistics
  AnimationExport,    // Video/GIF export

  // Preferences panels
  Appearance,         // Color preferences
  ViewerSettings,     // Camera, grid, rendering
  IOPaths,            // Input/output directories
  LatexSettings,      // LaTeX preview configuration
  Benchmarks,         // Benchmark runner
  UIConfiguration,    // UI config editor

  // Meta
  Empty,              // Placeholder for empty slots

  COUNT               // Number of view types
};

// Convert ViewType to string for serialization
inline std::string ViewTypeToString(ViewType type) {
  static const std::unordered_map<ViewType, std::string> names = {
    {ViewType::Viewer3D, "Viewer3D"},
    {ViewType::Timeline, "Timeline"},
    {ViewType::EquationEditor, "EquationEditor"},
    {ViewType::DomainSettings, "DomainSettings"},
    {ViewType::GridSettings, "GridSettings"},
    {ViewType::BoundaryConditions, "BoundaryConditions"},
    {ViewType::SolverConfig, "SolverConfig"},
    {ViewType::TimeSettings, "TimeSettings"},
    {ViewType::RunControls, "RunControls"},
    {ViewType::LogView, "LogView"},
    {ViewType::InitialConditions, "InitialConditions"},
    {ViewType::PresetManager, "PresetManager"},
    {ViewType::SourceTermEditor, "SourceTermEditor"},
    {ViewType::MaterialProperties, "MaterialProperties"},
    {ViewType::MeshPreview, "MeshPreview"},
    {ViewType::ParameterSweep, "ParameterSweep"},
    {ViewType::FieldSelector, "FieldSelector"},
    {ViewType::SliceControls, "SliceControls"},
    {ViewType::IsosurfaceControls, "IsosurfaceControls"},
    {ViewType::ImageExport, "ImageExport"},
    {ViewType::AdvancedInspection, "AdvancedInspection"},
    {ViewType::ComparisonTools, "ComparisonTools"},
    {ViewType::ConvergencePlot, "ConvergencePlot"},
    {ViewType::PointProbe, "PointProbe"},
    {ViewType::StatisticsPanel, "StatisticsPanel"},
    {ViewType::AnimationExport, "AnimationExport"},
    {ViewType::Appearance, "Appearance"},
    {ViewType::ViewerSettings, "ViewerSettings"},
    {ViewType::IOPaths, "IOPaths"},
    {ViewType::LatexSettings, "LatexSettings"},
    {ViewType::Benchmarks, "Benchmarks"},
    {ViewType::UIConfiguration, "UIConfiguration"},
    {ViewType::Empty, "Empty"},
  };
  auto it = names.find(type);
  return it != names.end() ? it->second : "Unknown";
}

// Convert string to ViewType for deserialization
inline ViewType StringToViewType(const std::string& name) {
  static const std::unordered_map<std::string, ViewType> types = {
    {"Viewer3D", ViewType::Viewer3D},
    {"Timeline", ViewType::Timeline},
    {"EquationEditor", ViewType::EquationEditor},
    {"DomainSettings", ViewType::DomainSettings},
    {"GridSettings", ViewType::GridSettings},
    {"BoundaryConditions", ViewType::BoundaryConditions},
    {"SolverConfig", ViewType::SolverConfig},
    {"TimeSettings", ViewType::TimeSettings},
    {"RunControls", ViewType::RunControls},
    {"LogView", ViewType::LogView},
    {"InitialConditions", ViewType::InitialConditions},
    {"PresetManager", ViewType::PresetManager},
    {"SourceTermEditor", ViewType::SourceTermEditor},
    {"MaterialProperties", ViewType::MaterialProperties},
    {"MeshPreview", ViewType::MeshPreview},
    {"ParameterSweep", ViewType::ParameterSweep},
    {"FieldSelector", ViewType::FieldSelector},
    {"SliceControls", ViewType::SliceControls},
    {"IsosurfaceControls", ViewType::IsosurfaceControls},
    {"ImageExport", ViewType::ImageExport},
    {"AdvancedInspection", ViewType::AdvancedInspection},
    {"ComparisonTools", ViewType::ComparisonTools},
    {"ConvergencePlot", ViewType::ConvergencePlot},
    {"PointProbe", ViewType::PointProbe},
    {"StatisticsPanel", ViewType::StatisticsPanel},
    {"AnimationExport", ViewType::AnimationExport},
    {"Appearance", ViewType::Appearance},
    {"ViewerSettings", ViewType::ViewerSettings},
    {"IOPaths", ViewType::IOPaths},
    {"LatexSettings", ViewType::LatexSettings},
    {"Benchmarks", ViewType::Benchmarks},
    {"UIConfiguration", ViewType::UIConfiguration},
    {"Empty", ViewType::Empty},
  };
  auto it = types.find(name);
  return it != types.end() ? it->second : ViewType::Empty;
}

// Get display name for UI
inline std::string GetViewTypeDisplayName(ViewType type) {
  static const std::unordered_map<ViewType, std::string> display_names = {
    {ViewType::Viewer3D, "3D Viewer"},
    {ViewType::Timeline, "Timeline"},
    {ViewType::EquationEditor, "Equation Editor"},
    {ViewType::DomainSettings, "Domain Settings"},
    {ViewType::GridSettings, "Grid Settings"},
    {ViewType::BoundaryConditions, "Boundary Conditions"},
    {ViewType::SolverConfig, "Solver Configuration"},
    {ViewType::TimeSettings, "Time Settings"},
    {ViewType::RunControls, "Run Controls"},
    {ViewType::LogView, "Log"},
    {ViewType::InitialConditions, "Initial Conditions"},
    {ViewType::PresetManager, "Preset Manager"},
    {ViewType::SourceTermEditor, "Source Term Editor"},
    {ViewType::MaterialProperties, "Material Properties"},
    {ViewType::MeshPreview, "Mesh Preview"},
    {ViewType::ParameterSweep, "Parameter Sweep"},
    {ViewType::FieldSelector, "Field Selector"},
    {ViewType::SliceControls, "Slice Controls"},
    {ViewType::IsosurfaceControls, "Isosurface Controls"},
    {ViewType::ImageExport, "Image Export"},
    {ViewType::AdvancedInspection, "Advanced Inspection"},
    {ViewType::ComparisonTools, "Comparison Tools"},
    {ViewType::ConvergencePlot, "Convergence Plot"},
    {ViewType::PointProbe, "Point Probe"},
    {ViewType::StatisticsPanel, "Statistics"},
    {ViewType::AnimationExport, "Animation Export"},
    {ViewType::Appearance, "Appearance"},
    {ViewType::ViewerSettings, "Viewer Settings"},
    {ViewType::IOPaths, "I/O Paths"},
    {ViewType::LatexSettings, "LaTeX Settings"},
    {ViewType::Benchmarks, "Benchmarks"},
    {ViewType::UIConfiguration, "UI Configuration"},
    {ViewType::Empty, "Empty"},
  };
  auto it = display_names.find(type);
  return it != display_names.end() ? it->second : "Unknown";
}

// Get category for grouping in dropdown
inline std::string GetViewTypeCategory(ViewType type) {
  switch (type) {
    case ViewType::Viewer3D:
    case ViewType::Timeline:
      return "Visualization";

    case ViewType::EquationEditor:
    case ViewType::DomainSettings:
    case ViewType::GridSettings:
    case ViewType::BoundaryConditions:
    case ViewType::SolverConfig:
    case ViewType::TimeSettings:
    case ViewType::RunControls:
    case ViewType::LogView:
    case ViewType::InitialConditions:
    case ViewType::PresetManager:
    case ViewType::SourceTermEditor:
    case ViewType::MaterialProperties:
    case ViewType::MeshPreview:
    case ViewType::ParameterSweep:
      return "Configuration";

    case ViewType::FieldSelector:
    case ViewType::SliceControls:
    case ViewType::IsosurfaceControls:
    case ViewType::ImageExport:
    case ViewType::AdvancedInspection:
    case ViewType::ComparisonTools:
    case ViewType::ConvergencePlot:
    case ViewType::PointProbe:
    case ViewType::StatisticsPanel:
    case ViewType::AnimationExport:
      return "Inspection";

    case ViewType::Appearance:
    case ViewType::ViewerSettings:
    case ViewType::IOPaths:
    case ViewType::LatexSettings:
    case ViewType::Benchmarks:
    case ViewType::UIConfiguration:
      return "Settings";

    default:
      return "Other";
  }
}

// Check if multiple instances of this view type are allowed
inline bool AllowsMultipleInstances(ViewType type) {
  switch (type) {
    case ViewType::Viewer3D:
    case ViewType::Empty:
      return true;
    default:
      return false;
  }
}

#endif // VIEW_TYPES_H
