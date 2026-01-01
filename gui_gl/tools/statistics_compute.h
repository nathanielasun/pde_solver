#ifndef STATISTICS_COMPUTE_H
#define STATISTICS_COMPUTE_H

#include <vector>

struct FieldStatistics {
  double min = 0.0;
  double max = 0.0;
  double mean = 0.0;
  double median = 0.0;
  double stddev = 0.0;
  double rms = 0.0;
  double l2_norm = 0.0;
  size_t count = 0;
};

struct Histogram {
  double min = 0.0;
  double max = 0.0;
  std::vector<double> bins;   // Bin edges (size = bins + 1)
  std::vector<int> counts;    // Bin counts (size = bins)
};

FieldStatistics ComputeStatistics(const std::vector<double>& data,
                                  const std::vector<bool>& mask = {});

Histogram ComputeHistogram(const std::vector<double>& data, int bins,
                           const std::vector<bool>& mask = {});

#endif  // STATISTICS_COMPUTE_H

