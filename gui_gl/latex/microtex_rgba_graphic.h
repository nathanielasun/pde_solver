#ifndef MICROTEX_RGBA_GRAPHIC_H
#define MICROTEX_RGBA_GRAPHIC_H

#include "latex/latex_bitmap.h"

#include <cstdint>
#include <string>
#include <vector>

namespace tex {
class TeXRender;
}

namespace microtex_rgba {

extern std::string g_res_root;

class RgbaSurface {
 public:
  RgbaSurface(int width, int height);
  int width() const { return width_; }
  int height() const { return height_; }
  std::vector<uint8_t>& pixels() { return pixels_; }
  const std::vector<uint8_t>& pixels() const { return pixels_; }

  void clear(uint8_t r, uint8_t g, uint8_t b, uint8_t a);
  void blendPixel(int x, int y, uint8_t r, uint8_t g, uint8_t b, uint8_t a);
  void fillRect(int x, int y, int w, int h, uint8_t r, uint8_t g, uint8_t b, uint8_t a);
  void drawLine(int x0, int y0, int x1, int y1, uint8_t r, uint8_t g, uint8_t b, uint8_t a,
                float width);

  LatexBitmap toBitmap() const;

 private:
  int width_ = 0;
  int height_ = 0;
  std::vector<uint8_t> pixels_;
};

void SetResRoot(const std::string& path);
void DrawTeXRender(tex::TeXRender* render, RgbaSurface* surface, int padding);

}  // namespace microtex_rgba

#ifdef USE_MICROTEX_LATEX_PREVIEW
void MicroTeXInit(const std::string& res_root);
void MicroTeXShutdown();
LatexBitmap RenderMicroTeXToBitmap(const std::string& latex, const LatexRenderStyle& style);
#endif

#endif  // MICROTEX_RGBA_GRAPHIC_H
