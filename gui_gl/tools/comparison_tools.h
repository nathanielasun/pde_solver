#ifndef COMPARISON_TOOLS_H
#define COMPARISON_TOOLS_H

#include "pde_types.h"
#include <vector>
#include <string>
#include <optional>

// Statistics for field comparison
struct ComparisonStatistics {
  double min_diff = 0.0;
  double max_diff = 0.0;
  double mean_diff = 0.0;
  double l2_diff = 0.0;
  double linf_diff = 0.0;
  double relative_error_l2 = 0.0;
  double relative_error_linf = 0.0;
  size_t valid_points = 0;
  size_t total_points = 0;
};

// Solution comparator for comparing two solutions
class SolutionComparator {
 public:
  SolutionComparator();
  
  // Set solution A (reference solution)
  void SetSolutionA(const Domain& domain, const std::vector<double>& grid);
  
  // Set solution B (solution to compare)
  void SetSolutionB(const Domain& domain, const std::vector<double>& grid);
  
  // Check if both solutions are set
  bool IsReady() const { return has_solution_a_ && has_solution_b_; }
  
  // Check if domains are compatible
  bool AreDomainsCompatible() const;
  
  // Compute difference field (B - A)
  std::vector<double> ComputeDifference() const;
  
  // Compute relative error field (|B - A| / |A|) where A != 0
  std::vector<double> ComputeRelativeError() const;
  
  // Compute comparison statistics
  ComparisonStatistics ComputeStatistics() const;
  
  // Get domain (assumes both domains are compatible)
  Domain GetDomain() const;
  
  // Clear solutions
  void ClearSolutionA();
  void ClearSolutionB();
  void ClearAll();

 private:
  bool has_solution_a_ = false;
  bool has_solution_b_ = false;
  Domain domain_a_;
  Domain domain_b_;
  std::vector<double> grid_a_;
  std::vector<double> grid_b_;
  
  // Helper to interpolate value at point (x, y, z) from grid
  double InterpolateValue(const Domain& domain, const std::vector<double>& grid,
                          double x, double y, double z) const;
  
  // Helper to get grid index
  size_t GetGridIndex(const Domain& domain, int i, int j, int k) const;
  
  // Helper to check if point is in domain
  bool IsPointInDomain(const Domain& domain, double x, double y, double z) const;
};

#endif  // COMPARISON_TOOLS_H

