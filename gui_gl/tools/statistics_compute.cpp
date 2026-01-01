#include "statistics_compute.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace {

bool UseValue(size_t idx, const std::vector<bool>& mask) {
  if (mask.empty()) return true;
  return idx < mask.size() && mask[idx];
}

}  // namespace

FieldStatistics ComputeStatistics(const std::vector<double>& data,
                                  const std::vector<bool>& mask) {
  FieldStatistics stats{};
  if (data.empty()) return stats;

  double min_v = std::numeric_limits<double>::infinity();
  double max_v = -std::numeric_limits<double>::infinity();
  double sum = 0.0;
  double sum_sq = 0.0;
  size_t count = 0;
  std::vector<double> values;
  values.reserve(data.size());

  for (size_t i = 0; i < data.size(); ++i) {
    if (!UseValue(i, mask)) continue;
    double v = data[i];
    if (!std::isfinite(v)) continue;
    min_v = std::min(min_v, v);
    max_v = std::max(max_v, v);
    sum += v;
    sum_sq += v * v;
    values.push_back(v);
    ++count;
  }

  if (count == 0) return stats;

  stats.count = count;
  stats.min = min_v;
  stats.max = max_v;
  stats.mean = sum / static_cast<double>(count);
  stats.l2_norm = std::sqrt(sum_sq);
  stats.rms = std::sqrt(sum_sq / static_cast<double>(count));

  std::sort(values.begin(), values.end());
  if (count % 2 == 1) {
    stats.median = values[count / 2];
  } else {
    stats.median = 0.5 * (values[count / 2 - 1] + values[count / 2]);
  }

  double var = 0.0;
  for (double v : values) {
    double d = v - stats.mean;
    var += d * d;
  }
  var /= static_cast<double>(count);
  stats.stddev = std::sqrt(var);

  return stats;
}

Histogram ComputeHistogram(const std::vector<double>& data, int bins,
                           const std::vector<bool>& mask) {
  Histogram hist{};
  if (data.empty() || bins <= 0) return hist;

  double min_v = std::numeric_limits<double>::infinity();
  double max_v = -std::numeric_limits<double>::infinity();
  size_t count = 0;
  for (size_t i = 0; i < data.size(); ++i) {
    if (!UseValue(i, mask)) continue;
    double v = data[i];
    if (!std::isfinite(v)) continue;
    min_v = std::min(min_v, v);
    max_v = std::max(max_v, v);
    ++count;
  }
  if (count == 0) return hist;

  hist.min = min_v;
  hist.max = max_v;
  double range = max_v - min_v;
  if (range == 0.0) {
    hist.bins = {min_v, max_v};
    hist.counts = std::vector<int>(1, static_cast<int>(count));
    return hist;
  }

  hist.bins.resize(static_cast<size_t>(bins) + 1);
  hist.counts.assign(static_cast<size_t>(bins), 0);
  for (int b = 0; b <= bins; ++b) {
    hist.bins[static_cast<size_t>(b)] = min_v + range * (static_cast<double>(b) / bins);
  }

  for (size_t i = 0; i < data.size(); ++i) {
    if (!UseValue(i, mask)) continue;
    double v = data[i];
    if (!std::isfinite(v)) continue;
    int bin = static_cast<int>(std::floor((v - min_v) / range * bins));
    if (bin < 0) bin = 0;
    if (bin >= bins) bin = bins - 1;
    ++hist.counts[static_cast<size_t>(bin)];
  }

  return hist;
}

