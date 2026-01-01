#include "conserved_monitor.h"

#include <algorithm>
#include <cmath>

namespace {
constexpr double kBaselineEps = 1e-12;
constexpr double kDriftWarn = 1e-2;
constexpr double kGrowthEps = 1e-10;
constexpr double kGrowthThreshold = 0.35;
constexpr int kGrowthConsecutive = 3;

struct KahanSum {
  double sum = 0.0;
  double c = 0.0;
  void Add(double value) {
    const double y = value - c;
    const double t = sum + y;
    c = (t - sum) - y;
    sum = t;
  }
};

ConservedSample ComputeSample(const Domain& d, const std::vector<double>& grid) {
  ConservedSample sample;
  const int nx = d.nx;
  const int ny = d.ny;
  const int nz = std::max(1, d.nz);
  const size_t expected = static_cast<size_t>(nx) * static_cast<size_t>(ny) * static_cast<size_t>(nz);
  if (grid.size() < expected || nx <= 0 || ny <= 0 || nz <= 0) {
    return sample;
  }
  const double dx = (d.xmax - d.xmin) / static_cast<double>(std::max(1, nx - 1));
  const double dy = (d.ymax - d.ymin) / static_cast<double>(std::max(1, ny - 1));
  const double dz = (d.zmax - d.zmin) / static_cast<double>(std::max(1, nz - 1));
  const double cell_volume = (nz > 1) ? (dx * dy * dz) : (dx * dy);

  KahanSum mass_sum;
  KahanSum energy_sum;
  double max_abs = 0.0;
  for (size_t idx = 0; idx < expected; ++idx) {
    const double value = grid[idx];
    mass_sum.Add(value);
    energy_sum.Add(value * value);
    max_abs = std::max(max_abs, std::abs(value));
  }

  sample.ok = true;
  sample.mass = mass_sum.sum * cell_volume;
  sample.energy = energy_sum.sum * cell_volume;
  sample.max_abs = max_abs;
  return sample;
}
}  // namespace

ConservedSample UpdateConservedMonitor(const Domain& d, int frame,
                                       const std::vector<double>& grid,
                                       ConservedMonitor* monitor) {
  ConservedSample sample = ComputeSample(d, grid);
  if (!sample.ok || !monitor) {
    return sample;
  }

  if (!monitor->has_baseline) {
    monitor->has_baseline = true;
    monitor->mass0 = sample.mass;
    monitor->energy0 = sample.energy;
    monitor->last_max_abs = sample.max_abs;
  }

  const bool mass_valid = std::abs(monitor->mass0) > kBaselineEps;
  const bool energy_valid = std::abs(monitor->energy0) > kBaselineEps;
  if (mass_valid) {
    sample.mass_drift = (sample.mass - monitor->mass0) / std::abs(monitor->mass0);
  }
  if (energy_valid) {
    sample.energy_drift = (sample.energy - monitor->energy0) / std::abs(monitor->energy0);
  }

  monitor->mass_history.push_back(sample.mass);
  monitor->energy_history.push_back(sample.energy);
  monitor->max_abs_history.push_back(sample.max_abs);
  monitor->mass_drift_history.push_back(sample.mass_drift);
  monitor->energy_drift_history.push_back(sample.energy_drift);

  if (!monitor->mass_warning && mass_valid && std::abs(sample.mass_drift) > kDriftWarn) {
    monitor->mass_warning = true;
    monitor->mass_warning_frame = frame;
  }
  if (!monitor->energy_warning && energy_valid && std::abs(sample.energy_drift) > kDriftWarn) {
    monitor->energy_warning = true;
    monitor->energy_warning_frame = frame;
  }

  double ratio = 1.0;
  if (monitor->last_max_abs > kGrowthEps) {
    ratio = sample.max_abs / monitor->last_max_abs;
    if (ratio > 1.0 + kGrowthThreshold) {
      monitor->growth_count++;
    } else {
      monitor->growth_count = 0;
    }
    if (!monitor->blowup_warning && monitor->growth_count >= kGrowthConsecutive) {
      monitor->blowup_warning = true;
      monitor->blowup_warning_frame = frame;
      monitor->blowup_ratio = ratio;
      monitor->blowup_max = sample.max_abs;
    }
  }
  monitor->last_max_abs = sample.max_abs;

  return sample;
}
