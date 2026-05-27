#ifndef LATEX_BITMAP_H
#define LATEX_BITMAP_H

#include <cstdint>
#include <string>
#include <vector>

struct LatexBitmap {
  int width = 0;
  int height = 0;
  std::vector<uint8_t> rgba;
  std::string error;
};

struct LatexRenderStyle {
  int font_size = 18;
  std::string fg_hex = "#e0e0e0";
  int padding = 8;
};

// CPU rasterization (mathtext layout; MicroTeX when USE_MICROTEX_LATEX_PREVIEW is enabled).
LatexBitmap RenderLatexToBitmap(const std::string& latex, const LatexRenderStyle& style);

#endif  // LATEX_BITMAP_H
