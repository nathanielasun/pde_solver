#include "plot_widget.h"
#include <algorithm>
#include <cmath>
#include <sstream>
#include <iomanip>

void PlotWidget::SetData(const std::vector<float>& x, const std::vector<float>& y) {
  x_data_ = x;
  y_data_ = y;
  bounds_dirty_ = true;
}

void PlotWidget::AppendPoint(float x, float y) {
  x_data_.push_back(x);
  y_data_.push_back(y);
  bounds_dirty_ = true;
}

void PlotWidget::Clear() {
  x_data_.clear();
  y_data_.clear();
  for (auto& s : series_) {
    s.x.clear();
    s.y.clear();
  }
  bounds_dirty_ = true;
}

void PlotWidget::AddSeries(const std::string& label, ImU32 color) {
  Series s;
  s.label = label;
  s.color = color;
  series_.push_back(std::move(s));
}

void PlotWidget::SetSeriesData(size_t index, const std::vector<float>& x, const std::vector<float>& y) {
  if (index < series_.size()) {
    series_[index].x = x;
    series_[index].y = y;
    bounds_dirty_ = true;
  }
}

void PlotWidget::AppendToSeries(size_t index, float x, float y) {
  if (index < series_.size()) {
    series_[index].x.push_back(x);
    series_[index].y.push_back(y);
    bounds_dirty_ = true;
  }
}

void PlotWidget::ClearSeries(size_t index) {
  if (index < series_.size()) {
    series_[index].x.clear();
    series_[index].y.clear();
    bounds_dirty_ = true;
  }
}

void PlotWidget::UpdateBounds() {
  if (!bounds_dirty_) return;

  x_min_ = std::numeric_limits<float>::max();
  x_max_ = std::numeric_limits<float>::lowest();
  y_min_auto_ = std::numeric_limits<float>::max();
  y_max_auto_ = std::numeric_limits<float>::lowest();

  // Check primary data
  for (size_t i = 0; i < x_data_.size(); ++i) {
    x_min_ = std::min(x_min_, x_data_[i]);
    x_max_ = std::max(x_max_, x_data_[i]);
  }
  for (size_t i = 0; i < y_data_.size(); ++i) {
    float val = y_data_[i];
    if (log_scale_y && val > 0) {
      val = std::log10(val);
    }
    y_min_auto_ = std::min(y_min_auto_, val);
    y_max_auto_ = std::max(y_max_auto_, val);
  }

  // Check series data
  for (const auto& s : series_) {
    if (!s.visible) continue;
    for (size_t i = 0; i < s.x.size(); ++i) {
      x_min_ = std::min(x_min_, s.x[i]);
      x_max_ = std::max(x_max_, s.x[i]);
    }
    for (size_t i = 0; i < s.y.size(); ++i) {
      float val = s.y[i];
      if (log_scale_y && val > 0) {
        val = std::log10(val);
      }
      y_min_auto_ = std::min(y_min_auto_, val);
      y_max_auto_ = std::max(y_max_auto_, val);
    }
  }

  // Handle empty or single-point data
  if (x_min_ >= x_max_) {
    x_min_ = 0.0f;
    x_max_ = 1.0f;
  }
  if (y_min_auto_ >= y_max_auto_) {
    y_min_auto_ = 0.0f;
    y_max_auto_ = 1.0f;
  }

  // Add padding
  float y_range = y_max_auto_ - y_min_auto_;
  y_min_auto_ -= y_range * 0.05f;
  y_max_auto_ += y_range * 0.05f;

  bounds_dirty_ = false;
}

void PlotWidget::RenderGrid(ImDrawList* draw_list, ImVec2 pos, ImVec2 size,
                            float x_min, float x_max, float y_min, float y_max) {
  const int grid_lines = 5;

  // Horizontal grid lines
  for (int i = 0; i <= grid_lines; ++i) {
    float t = static_cast<float>(i) / grid_lines;
    float y = pos.y + size.y - t * size.y;
    draw_list->AddLine(ImVec2(pos.x, y), ImVec2(pos.x + size.x, y), grid_color);

    // Y-axis labels
    float val = y_min + t * (y_max - y_min);
    if (log_scale_y) {
      val = std::pow(10.0f, val);
    }
    std::ostringstream oss;
    if (std::abs(val) < 1e-3 || std::abs(val) >= 1e4) {
      oss << std::scientific << std::setprecision(1) << val;
    } else {
      oss << std::fixed << std::setprecision(2) << val;
    }
    draw_list->AddText(ImVec2(pos.x + 2, y - 6), IM_COL32(180, 180, 180, 255), oss.str().c_str());
  }

  // Vertical grid lines
  for (int i = 0; i <= grid_lines; ++i) {
    float t = static_cast<float>(i) / grid_lines;
    float x = pos.x + t * size.x;
    draw_list->AddLine(ImVec2(x, pos.y), ImVec2(x, pos.y + size.y), grid_color);

    // X-axis labels
    float val = x_min + t * (x_max - x_min);
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(0) << val;
    draw_list->AddText(ImVec2(x - 10, pos.y + size.y + 2), IM_COL32(180, 180, 180, 255), oss.str().c_str());
  }
}

void PlotWidget::RenderSeries(ImDrawList* draw_list, ImVec2 pos, ImVec2 size,
                              float x_min, float x_max, float y_min, float y_max,
                              const std::vector<float>& x, const std::vector<float>& y,
                              ImU32 color) {
  if (x.size() < 2 || y.size() < 2) return;

  size_t count = std::min(x.size(), y.size());
  float x_range = x_max - x_min;
  float y_range = y_max - y_min;

  if (x_range <= 0 || y_range <= 0) return;

  ImVec2 prev;
  bool has_prev = false;

  for (size_t i = 0; i < count; ++i) {
    float val = y[i];
    if (log_scale_y) {
      if (val <= 0) continue;
      val = std::log10(val);
    }

    float px = pos.x + (x[i] - x_min) / x_range * size.x;
    float py = pos.y + size.y - (val - y_min) / y_range * size.y;

    // Clamp to plot area
    px = std::clamp(px, pos.x, pos.x + size.x);
    py = std::clamp(py, pos.y, pos.y + size.y);

    ImVec2 pt(px, py);
    if (has_prev) {
      draw_list->AddLine(prev, pt, color, line_thickness);
    }
    prev = pt;
    has_prev = true;
  }
}

void PlotWidget::Render(float width, float height) {
  UpdateBounds();

  float y_min_use = auto_scale ? y_min_auto_ : (log_scale_y ? std::log10(y_min) : y_min);
  float y_max_use = auto_scale ? y_max_auto_ : (log_scale_y ? std::log10(y_max) : y_max);

  // Reserve space for labels
  const float margin_left = 60.0f;
  const float margin_bottom = 20.0f;
  const float margin_top = title.empty() ? 5.0f : 20.0f;
  const float margin_right = 10.0f;

  ImVec2 cursor = ImGui::GetCursorScreenPos();
  ImVec2 plot_pos(cursor.x + margin_left, cursor.y + margin_top);
  ImVec2 plot_size(width - margin_left - margin_right, height - margin_top - margin_bottom);

  if (plot_size.x <= 0 || plot_size.y <= 0) return;

  ImDrawList* draw_list = ImGui::GetWindowDrawList();

  // Background
  draw_list->AddRectFilled(plot_pos, ImVec2(plot_pos.x + plot_size.x, plot_pos.y + plot_size.y),
                           IM_COL32(25, 25, 30, 255));
  draw_list->AddRect(plot_pos, ImVec2(plot_pos.x + plot_size.x, plot_pos.y + plot_size.y),
                     IM_COL32(80, 80, 80, 255));

  // Title
  if (!title.empty()) {
    ImVec2 text_size = ImGui::CalcTextSize(title.c_str());
    draw_list->AddText(ImVec2(cursor.x + (width - text_size.x) * 0.5f, cursor.y),
                       IM_COL32(220, 220, 220, 255), title.c_str());
  }

  // Grid
  RenderGrid(draw_list, plot_pos, plot_size, x_min_, x_max_, y_min_use, y_max_use);

  // Primary data
  if (!x_data_.empty() && !y_data_.empty()) {
    RenderSeries(draw_list, plot_pos, plot_size, x_min_, x_max_, y_min_use, y_max_use,
                 x_data_, y_data_, line_color);
  }

  // Multi-series data
  for (const auto& s : series_) {
    if (s.visible && !s.x.empty() && !s.y.empty()) {
      RenderSeries(draw_list, plot_pos, plot_size, x_min_, x_max_, y_min_use, y_max_use,
                   s.x, s.y, s.color);
    }
  }

  // Axis labels
  if (!x_label.empty()) {
    ImVec2 text_size = ImGui::CalcTextSize(x_label.c_str());
    draw_list->AddText(ImVec2(cursor.x + (width - text_size.x) * 0.5f, cursor.y + height - 15),
                       IM_COL32(180, 180, 180, 255), x_label.c_str());
  }

  if (!y_label.empty()) {
    // Y-axis label (rotated text would require more work, so we just place it at top-left)
    draw_list->AddText(ImVec2(cursor.x, cursor.y + margin_top),
                       IM_COL32(180, 180, 180, 255), y_label.c_str());
  }

  // Reserve the space
  ImGui::Dummy(ImVec2(width, height));
}
