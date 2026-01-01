#ifndef PLOT_WIDGET_H
#define PLOT_WIDGET_H

#include <vector>
#include <string>
#include "imgui.h"

// Simple line plot widget for real-time data visualization
class PlotWidget {
 public:
  PlotWidget() = default;

  // Set complete data series
  void SetData(const std::vector<float>& x, const std::vector<float>& y);

  // Append a single point (for real-time updates)
  void AppendPoint(float x, float y);

  // Clear all data
  void Clear();

  // Render the plot
  void Render(float width, float height);

  // Configuration
  bool auto_scale = true;
  bool log_scale_y = false;
  float y_min = 0.0f;
  float y_max = 1.0f;

  std::string title;
  std::string x_label;
  std::string y_label;

  ImU32 line_color = IM_COL32(100, 180, 255, 255);
  ImU32 grid_color = IM_COL32(60, 60, 60, 255);
  float line_thickness = 1.5f;

  // Multi-series support
  struct Series {
    std::vector<float> x;
    std::vector<float> y;
    ImU32 color;
    std::string label;
    bool visible = true;
  };

  void AddSeries(const std::string& label, ImU32 color);
  void SetSeriesData(size_t index, const std::vector<float>& x, const std::vector<float>& y);
  void AppendToSeries(size_t index, float x, float y);
  void ClearSeries(size_t index);
  size_t GetSeriesCount() const { return series_.size(); }

 private:
  // Primary data (single series mode)
  std::vector<float> x_data_;
  std::vector<float> y_data_;

  // Multi-series data
  std::vector<Series> series_;

  // Cached bounds
  float x_min_ = 0.0f;
  float x_max_ = 1.0f;
  float y_min_auto_ = 0.0f;
  float y_max_auto_ = 1.0f;
  bool bounds_dirty_ = true;

  void UpdateBounds();
  void RenderGrid(ImDrawList* draw_list, ImVec2 pos, ImVec2 size,
                  float x_min, float x_max, float y_min, float y_max);
  void RenderSeries(ImDrawList* draw_list, ImVec2 pos, ImVec2 size,
                    float x_min, float x_max, float y_min, float y_max,
                    const std::vector<float>& x, const std::vector<float>& y,
                    ImU32 color);
};

#endif // PLOT_WIDGET_H
