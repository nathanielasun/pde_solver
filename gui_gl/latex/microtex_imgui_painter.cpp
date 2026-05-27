#include "latex/latex_bitmap.h"

#ifdef USE_MICROTEX_LATEX_PREVIEW
// MicroTeX integration: link third_party/MicroTeX when submodule and deps are present.
// Falls through to mathtext when render returns empty (build without full MicroTeX yet).
extern LatexBitmap RenderMathtextToBitmap(const std::string& latex, const LatexRenderStyle& style);
extern LatexBitmap RenderMicroTeXToBitmap(const std::string& latex, const LatexRenderStyle& style);

LatexBitmap RenderLatexToBitmap(const std::string& latex, const LatexRenderStyle& style) {
  LatexBitmap mt = RenderMicroTeXToBitmap(latex, style);
  if (!mt.error.empty() || mt.rgba.empty()) {
    return RenderMathtextToBitmap(latex, style);
  }
  return mt;
}
#else
extern LatexBitmap RenderMathtextToBitmap(const std::string& latex, const LatexRenderStyle& style);

LatexBitmap RenderLatexToBitmap(const std::string& latex, const LatexRenderStyle& style) {
  return RenderMathtextToBitmap(latex, style);
}
#endif
