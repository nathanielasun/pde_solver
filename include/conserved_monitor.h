#ifndef CONSERVED_MONITOR_H
#define CONSERVED_MONITOR_H

#include <vector>

#include "pde_types.h"

struct ConservedSample {
  bool ok = false;
  double mass = 0.0;
  double energy = 0.0;
  double max_abs = 0.0;
  double mass_drift = 0.0;
  double energy_drift = 0.0;
};

struct ConservedMonitor {
  double mass0 = 0.0;
  double energy0 = 0.0;
  double last_max_abs = 0.0;
  bool has_baseline = false;
  int growth_count = 0;

  bool mass_warning = false;
  bool energy_warning = false;
  bool blowup_warning = false;
  int mass_warning_frame = -1;
  int energy_warning_frame = -1;
  int blowup_warning_frame = -1;
  double blowup_ratio = 0.0;
  double blowup_max = 0.0;

  std::vector<double> mass_history;
  std::vector<double> energy_history;
  std::vector<double> max_abs_history;
  std::vector<double> mass_drift_history;
  std::vector<double> energy_drift_history;
};

ConservedSample UpdateConservedMonitor(const Domain& d, int frame,
                                       const std::vector<double>& grid,
                                       ConservedMonitor* monitor);

#endif  // CONSERVED_MONITOR_H
