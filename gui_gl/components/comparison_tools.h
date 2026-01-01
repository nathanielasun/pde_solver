#ifndef COMPARISON_TOOLS_COMPONENT_H
#define COMPARISON_TOOLS_COMPONENT_H

#include "ui_component.h"
#include "tools/comparison_tools.h"
#include "GlViewer.h"
#include "pde_types.h"
#include "vtk_io.h"
#include <string>
#include <optional>

// Comparison tools component for Phase 2.4
class ComparisonToolsComponent : public UIComponent {
 public:
  ComparisonToolsComponent();
  
  void Render() override;
  std::string GetName() const override { return "ComparisonTools"; }
  
  // Set viewer reference
  void SetViewer(GlViewer* viewer) { viewer_ = viewer; }
  
  // Set current solution (for time step comparison)
  void SetCurrentSolution(const Domain& domain, const std::vector<double>& grid);
  
  // Load solution A from file
  bool LoadSolutionA(const std::string& filepath);
  
  // Load solution B from file
  bool LoadSolutionB(const std::string& filepath);
  
  // Get comparison result (difference field)
  const std::vector<double>& GetDifferenceField() const { return difference_field_; }
  
  // Get comparison result (relative error field)
  const std::vector<double>& GetRelativeErrorField() const { return relative_error_field_; }
  
  // Get comparison statistics
  const ComparisonStatistics& GetStatistics() const { return statistics_; }
  
  // Check if comparison is ready
  bool IsComparisonReady() const { return comparator_.IsReady() && comparator_.AreDomainsCompatible(); }
  
  // Get domain for visualization
  Domain GetDomain() const { return comparator_.GetDomain(); }
  
  // Clear all solutions
  void ClearAll();

 private:
  GlViewer* viewer_ = nullptr;
  SolutionComparator comparator_;
  
  std::vector<double> difference_field_;
  std::vector<double> relative_error_field_;
  ComparisonStatistics statistics_;
  
  // UI state
  std::string solution_a_path_;
  std::string solution_b_path_;
  bool show_difference_ = true;
  bool show_relative_error_ = false;
  bool auto_update_ = true;
  
  // Time step comparison
  std::optional<Domain> current_domain_;
  std::optional<std::vector<double>> current_grid_;
  int time_step_mode_ = 0;  // 0 = file comparison, 1 = time step comparison
  
  // Helper functions
  void UpdateComparison();
  void RenderFileComparison();
  void RenderTimeStepComparison();
  void RenderStatistics();
  std::string FormatNumber(double value, int precision = 6) const;
};

#endif  // COMPARISON_TOOLS_COMPONENT_H

