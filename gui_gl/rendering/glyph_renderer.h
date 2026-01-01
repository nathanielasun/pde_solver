#ifndef GLYPH_RENDERER_H
#define GLYPH_RENDERER_H

#include "imgui.h"
#include <vector>

// Simple 2D arrow renderer drawn with ImGui draw lists.
class GlyphRenderer2D {
 public:
  void Clear();
  void SetArrows(const std::vector<ImVec2>& starts, const std::vector<ImVec2>& ends);
  void Render(ImDrawList* draw_list, ImU32 color, float thickness = 1.0f) const;

 private:
  std::vector<ImVec2> starts_;
  std::vector<ImVec2> ends_;
};

#endif  // GLYPH_RENDERER_H

