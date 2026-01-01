#include "comparison_tools.h"
#include <cmath>
#include <algorithm>
#include <numeric>

SolutionComparator::SolutionComparator() = default;

void SolutionComparator::SetSolutionA(const Domain& domain, const std::vector<double>& grid) {
  domain_a_ = domain;
  grid_a_ = grid;
  has_solution_a_ = true;
}

void SolutionComparator::SetSolutionB(const Domain& domain, const std::vector<double>& grid) {
  domain_b_ = domain;
  grid_b_ = grid;
  has_solution_b_ = true;
}

bool SolutionComparator::AreDomainsCompatible() const {
  if (!has_solution_a_ || !has_solution_b_) {
    return false;
  }
  
  // Check if grid dimensions match
  if (domain_a_.nx != domain_b_.nx || 
      domain_a_.ny != domain_b_.ny || 
      domain_a_.nz != domain_b_.nz) {
    return false;
  }
  
  // Check if coordinate systems match
  if (domain_a_.coord_system != domain_b_.coord_system) {
    return false;
  }
  
  // Check if domain bounds are close (within 1% tolerance)
  const double tol = 0.01;
  if (std::abs(domain_a_.xmin - domain_b_.xmin) > tol * std::abs(domain_a_.xmin) ||
      std::abs(domain_a_.xmax - domain_b_.xmax) > tol * std::abs(domain_a_.xmax) ||
      std::abs(domain_a_.ymin - domain_b_.ymin) > tol * std::abs(domain_a_.ymin) ||
      std::abs(domain_a_.ymax - domain_b_.ymax) > tol * std::abs(domain_a_.ymax) ||
      std::abs(domain_a_.zmin - domain_b_.zmin) > tol * std::abs(domain_a_.zmin) ||
      std::abs(domain_a_.zmax - domain_b_.zmax) > tol * std::abs(domain_a_.zmax)) {
    return false;
  }
  
  return true;
}

std::vector<double> SolutionComparator::ComputeDifference() const {
  if (!AreDomainsCompatible()) {
    return {};
  }
  
  const size_t n = grid_a_.size();
  std::vector<double> diff(n);
  
  for (size_t i = 0; i < n; ++i) {
    diff[i] = grid_b_[i] - grid_a_[i];
  }
  
  return diff;
}

std::vector<double> SolutionComparator::ComputeRelativeError() const {
  if (!AreDomainsCompatible()) {
    return {};
  }
  
  const size_t n = grid_a_.size();
  std::vector<double> rel_error(n);
  
  for (size_t i = 0; i < n; ++i) {
    const double a_val = grid_a_[i];
    if (std::abs(a_val) > 1e-10) {
      rel_error[i] = std::abs(grid_b_[i] - a_val) / std::abs(a_val);
    } else {
      // If reference is near zero, use absolute difference
      rel_error[i] = std::abs(grid_b_[i] - a_val);
    }
  }
  
  return rel_error;
}

ComparisonStatistics SolutionComparator::ComputeStatistics() const {
  ComparisonStatistics stats;
  
  if (!AreDomainsCompatible()) {
    return stats;
  }
  
  const auto diff = ComputeDifference();
  const auto rel_error = ComputeRelativeError();
  
  if (diff.empty()) {
    return stats;
  }
  
  stats.total_points = diff.size();
  stats.valid_points = diff.size();
  
  // Compute min/max/mean difference
  auto [min_it, max_it] = std::minmax_element(diff.begin(), diff.end());
  stats.min_diff = *min_it;
  stats.max_diff = *max_it;
  
  const double sum = std::accumulate(diff.begin(), diff.end(), 0.0);
  stats.mean_diff = sum / diff.size();
  
  // Compute L2 and L-infinity norms of difference
  double sum_sq = 0.0;
  double max_abs = 0.0;
  for (double d : diff) {
    sum_sq += d * d;
    max_abs = std::max(max_abs, std::abs(d));
  }
  stats.l2_diff = std::sqrt(sum_sq / diff.size());
  stats.linf_diff = max_abs;
  
  // Compute relative error norms
  double rel_sum_sq = 0.0;
  double rel_max = 0.0;
  for (double r : rel_error) {
    rel_sum_sq += r * r;
    rel_max = std::max(rel_max, r);
  }
  stats.relative_error_l2 = std::sqrt(rel_sum_sq / rel_error.size());
  stats.relative_error_linf = rel_max;
  
  return stats;
}

Domain SolutionComparator::GetDomain() const {
  if (has_solution_a_) {
    return domain_a_;
  }
  return domain_b_;
}

void SolutionComparator::ClearSolutionA() {
  has_solution_a_ = false;
  domain_a_ = Domain();
  grid_a_.clear();
}

void SolutionComparator::ClearSolutionB() {
  has_solution_b_ = false;
  domain_b_ = Domain();
  grid_b_.clear();
}

void SolutionComparator::ClearAll() {
  ClearSolutionA();
  ClearSolutionB();
}

double SolutionComparator::InterpolateValue(const Domain& domain, const std::vector<double>& grid,
                                            double x, double y, double z) const {
  // Simple nearest-neighbor interpolation
  const double dx = (domain.xmax - domain.xmin) / domain.nx;
  const double dy = (domain.ymax - domain.ymin) / domain.ny;
  const double dz = (domain.zmax - domain.zmin) / domain.nz;
  
  int i = static_cast<int>((x - domain.xmin) / dx);
  int j = static_cast<int>((y - domain.ymin) / dy);
  int k = static_cast<int>((z - domain.zmin) / dz);
  
  // Clamp to valid range
  i = std::max(0, std::min(i, domain.nx - 1));
  j = std::max(0, std::min(j, domain.ny - 1));
  k = std::max(0, std::min(k, domain.nz - 1));
  
  return grid[GetGridIndex(domain, i, j, k)];
}

size_t SolutionComparator::GetGridIndex(const Domain& domain, int i, int j, int k) const {
  return i + j * domain.nx + k * domain.nx * domain.ny;
}

bool SolutionComparator::IsPointInDomain(const Domain& domain, double x, double y, double z) const {
  return x >= domain.xmin && x <= domain.xmax &&
         y >= domain.ymin && y <= domain.ymax &&
         z >= domain.zmin && z <= domain.zmax;
}

