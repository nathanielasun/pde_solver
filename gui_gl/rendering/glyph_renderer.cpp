#include "glyph_renderer.h"

void GlyphRenderer2D::Clear() {
  starts_.clear();
  ends_.clear();
}

void GlyphRenderer2D::SetArrows(const std::vector<ImVec2>& starts,
                                const std::vector<ImVec2>& ends) {
  starts_ = starts;
  ends_ = ends;
}

void GlyphRenderer2D::Render(ImDrawList* draw_list, ImU32 color, float thickness) const {
  if (!draw_list || starts_.size() != ends_.size()) return;
  for (size_t i = 0; i < starts_.size(); ++i) {
    const ImVec2& a = starts_[i];
    const ImVec2& b = ends_[i];
    draw_list->AddLine(a, b, color, thickness);
    // Arrow head
    ImVec2 dir = ImVec2(b.x - a.x, b.y - a.y);
    float len = std::sqrt(dir.x * dir.x + dir.y * dir.y);
    if (len > 1e-4f) {
      dir.x /= len;
      dir.y /= len;
      const float head = 6.0f;
      ImVec2 left = ImVec2(-dir.y, dir.x);
      ImVec2 right = ImVec2(dir.y, -dir.x);
      ImVec2 head_base = ImVec2(b.x - dir.x * head, b.y - dir.y * head);
      draw_list->AddLine(b, ImVec2(head_base.x + left.x * head * 0.6f,
                                   head_base.y + left.y * head * 0.6f),
                         color, thickness);
      draw_list->AddLine(b, ImVec2(head_base.x + right.x * head * 0.6f,
                                   head_base.y + right.y * head * 0.6f),
                         color, thickness);
    }
  }
}

