#ifndef INSPECTION_TOOLS_H
#define INSPECTION_TOOLS_H

#include "ui_component.h"
#include "GlViewer.h"
#include "pde_types.h"
#include "vtk_io.h"
#include "tools/statistics_compute.h"
#include <vector>
#include <string>

// Slice plane definition
struct SlicePlane {
  int axis = 0;  // 0=x, 1=y, 2=z
  double value = 0.0;
  double thickness = 0.01;
  bool enabled = false;
  std::string name;
  
  SlicePlane() = default;
  SlicePlane(int a, double v, double t, const std::string& n) 
    : axis(a), value(v), thickness(t), name(n), enabled(true) {}
};

// Probe point for value inspection
struct ProbePoint {
  double x = 0.0;
  double y = 0.0;
  double z = 0.0;
  double value = 0.0;
  bool active = false;
  std::string label;
  
  ProbePoint() = default;
  ProbePoint(double x_, double y_, double z_, const std::string& lbl = "")
    : x(x_), y(y_), z(z_), label(lbl), active(true) {}
};

// Line plot definition (1D cross-section)
struct LinePlot {
  double x0 = 0.0, y0 = 0.0, z0 = 0.0;
  double x1 = 0.0, y1 = 0.0, z1 = 0.0;
  int num_points = 100;
  bool enabled = false;
  std::string name;
  std::vector<double> positions;  // 1D positions along line
  std::vector<double> values;    // Values at each position
  
  LinePlot() = default;
  LinePlot(double x0_, double y0_, double z0_, double x1_, double y1_, double z1_,
           int n, const std::string& n_)
    : x0(x0_), y0(y0_), z0(z0_), x1(x1_), y1(y1_), z1(z1_), 
      num_points(n), name(n_), enabled(true) {}
};

// Advanced inspection tools component
class InspectionToolsComponent : public UIComponent {
 public:
  InspectionToolsComponent();
  
  void Render() override;
  std::string GetName() const override { return "InspectionTools"; }
  
  // Set viewer reference
  void SetViewer(GlViewer* viewer) { viewer_ = viewer; }
  
  // Set domain and grid data
  void SetData(const Domain* domain, const std::vector<double>* grid,
               const struct DerivedFields* derived = nullptr);
  
  // Slice management
  void AddSlice(const SlicePlane& slice);
  void RemoveSlice(size_t index);
  void ClearSlices();
  const std::vector<SlicePlane>& GetSlices() const { return slices_; }
  
  // Probe management
  void AddProbe(const ProbePoint& probe);
  void RemoveProbe(size_t index);
  void ClearProbes();
  const std::vector<ProbePoint>& GetProbes() const { return probes_; }
  
  // Line plot management
  void AddLinePlot(const LinePlot& plot);
  void RemoveLinePlot(size_t index);
  void ClearLinePlots();
  const std::vector<LinePlot>& GetLinePlots() const { return line_plots_; }
  
  // Mask management (region-based statistics)
  void SetMask(const std::vector<bool>& mask) { mask_ = mask; stats_dirty_ = true; }
  const std::vector<bool>& GetMask() const { return mask_; }
  
  // Update computed values (call after data changes)
  void UpdateProbeValues();
  void UpdateLinePlots();

 private:
  GlViewer* viewer_ = nullptr;
  const Domain* domain_ = nullptr;
  const std::vector<double>* grid_ = nullptr;
  const struct DerivedFields* derived_fields_ = nullptr;
  bool has_derived_ = false;
  
  std::vector<SlicePlane> slices_;
  std::vector<ProbePoint> probes_;
  std::vector<LinePlot> line_plots_;
  
  // UI state
  int selected_slice_axis_ = 0;
  double selected_slice_value_ = 0.0;
  double selected_slice_thickness_ = 0.01;
  
  double probe_x_ = 0.0, probe_y_ = 0.0, probe_z_ = 0.0;
  
  double line_x0_ = 0.0, line_y0_ = 0.0, line_z0_ = 0.0;
  double line_x1_ = 1.0, line_y1_ = 0.0, line_z1_ = 0.0;
  int line_num_points_ = 100;
  
  // Statistics & histogram
  std::vector<bool> mask_;
  int selected_field_index_ = 0;
  int histogram_bins_ = 20;
  bool stats_dirty_ = true;
  struct StatsCache {
    bool valid = false;
    FieldStatistics stats;
    Histogram hist;
  } stats_cache_;

  // Region selection UI
  int region_type_ = 0;  // 0 = box, 1 = sphere
  double box_xmin_ = 0.0, box_xmax_ = 1.0;
  double box_ymin_ = 0.0, box_ymax_ = 1.0;
  double box_zmin_ = 0.0, box_zmax_ = 1.0;
  double sphere_cx_ = 0.5, sphere_cy_ = 0.5, sphere_cz_ = 0.0, sphere_r_ = 0.5;
  
  // Helper functions
  double GetValueAtPoint(double x, double y, double z) const;
  void ComputeLinePlot(LinePlot& plot) const;
  std::string FormatValue(double value) const;
  const std::vector<double>* GetFieldDataByIndex(int idx) const;
  void EnsureStatisticsComputed();
};

// Shared singleton accessor (used by both InspectPanel and main viewer for click-probes)
InspectionToolsComponent* GetInspectionComponentSingleton();

#endif  // INSPECTION_TOOLS_H

