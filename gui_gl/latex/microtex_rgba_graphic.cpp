#include "latex/microtex_rgba_graphic.h"

#include "latex/preview_font_paths.h"

#include "config.h"

#if defined(BUILD_RGBA) && !defined(MEM_CHECK)

#include "graphic/graphic.h"
#include "utils/utf.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#define STB_TRUETYPE_IMPLEMENTATION
#define STBTT_STATIC
#include "imstb_truetype.h"

namespace microtex_rgba {

std::string g_res_root;

void SetResRoot(const std::string& path) {
  g_res_root = path;
}

RgbaSurface::RgbaSurface(int width, int height) : width_(width), height_(height) {
  pixels_.assign(static_cast<size_t>(width * height * 4), 0);
}

void RgbaSurface::clear(uint8_t r, uint8_t g, uint8_t b, uint8_t a) {
  for (size_t i = 0; i < pixels_.size(); i += 4) {
    pixels_[i] = r;
    pixels_[i + 1] = g;
    pixels_[i + 2] = b;
    pixels_[i + 3] = a;
  }
}

void RgbaSurface::blendPixel(int x, int y, uint8_t r, uint8_t g, uint8_t b, uint8_t a) {
  if (x < 0 || y < 0 || x >= width_ || y >= height_ || a == 0) {
    return;
  }
  uint8_t* p = &pixels_[static_cast<size_t>((y * width_ + x) * 4)];
  const float af = a / 255.f;
  const float inv = 1.f - af;
  p[0] = static_cast<uint8_t>(std::min(255.f, p[0] * inv + r * af));
  p[1] = static_cast<uint8_t>(std::min(255.f, p[1] * inv + g * af));
  p[2] = static_cast<uint8_t>(std::min(255.f, p[2] * inv + b * af));
  p[3] = static_cast<uint8_t>(std::min(255.f, static_cast<float>(p[3]) + static_cast<float>(a)));
}

void RgbaSurface::fillRect(int x, int y, int w, int h, uint8_t r, uint8_t g, uint8_t b,
                           uint8_t a) {
  const int x1 = std::max(0, x);
  const int y1 = std::max(0, y);
  const int x2 = std::min(width_, x + w);
  const int y2 = std::min(height_, y + h);
  for (int py = y1; py < y2; ++py) {
    for (int px = x1; px < x2; ++px) {
      blendPixel(px, py, r, g, b, a);
    }
  }
}

void RgbaSurface::drawLine(int x0, int y0, int x1, int y1, uint8_t r, uint8_t g, uint8_t b,
                           uint8_t a, float width) {
  const int steps = std::max(std::abs(x1 - x0), std::abs(y1 - y0));
  const int thick = std::max(1, static_cast<int>(std::lround(width)));
  for (int i = 0; i <= steps; ++i) {
    const float t = steps == 0 ? 0.f : static_cast<float>(i) / static_cast<float>(steps);
    const int x = static_cast<int>(std::lround(x0 + (x1 - x0) * t));
    const int y = static_cast<int>(std::lround(y0 + (y1 - y0) * t));
    for (int dy = -thick / 2; dy <= thick / 2; ++dy) {
      for (int dx = -thick / 2; dx <= thick / 2; ++dx) {
        blendPixel(x + dx, y + dy, r, g, b, a);
      }
    }
  }
}

LatexBitmap RgbaSurface::toBitmap() const {
  LatexBitmap out;
  out.width = width_;
  out.height = height_;
  out.rgba = pixels_;
  return out;
}

}  // namespace microtex_rgba

namespace {

bool IsMathSymbolFontPath(const std::string& path) {
  return path.find("cmsy") != std::string::npos || path.find("msam") != std::string::npos ||
         path.find("msbm") != std::string::npos || path.find("cmex") != std::string::npos ||
         path.find("stmary") != std::string::npos || path.find("rsfs") != std::string::npos ||
         path.find("cmbsy") != std::string::npos || path.find("special.ttf") != std::string::npos;
}

std::string BundledSansBoldPath() {
  if (microtex_rgba::g_res_root.empty()) {
    return {};
  }
  const std::string cmssbx = microtex_rgba::g_res_root + "/fonts/latin/optional/cmssbx10.ttf";
  if (std::filesystem::exists(cmssbx)) {
    return cmssbx;
  }
  const std::string cmss = microtex_rgba::g_res_root + "/fonts/latin/optional/cmss10.ttf";
  if (std::filesystem::exists(cmss)) {
    return cmss;
  }
  return {};
}

std::string RemapToReadableFont(const std::string& resolved) {
  if (resolved.empty() || IsMathSymbolFontPath(resolved)) {
    return resolved;
  }
  const std::string sansBold = FindPreviewSansBoldFont();
  if (!sansBold.empty()) {
    return sansBold;
  }
  const std::string bundled = BundledSansBoldPath();
  if (!bundled.empty()) {
    return bundled;
  }
  return resolved;
}

std::string ResolveFontFile(const std::string& name) {
  if (name == "SansSerif" || name == "Serif" || name == "sans-serif" || name == "serif") {
    const std::string preferred = FindPreviewSansBoldFont();
    if (!preferred.empty()) {
      return preferred;
    }
    const std::string bundled = BundledSansBoldPath();
    if (!bundled.empty()) {
      return bundled;
    }
  }

  std::string resolved;
  if (!name.empty() && std::filesystem::exists(name)) {
    resolved = name;
  } else if (!microtex_rgba::g_res_root.empty() &&
             std::filesystem::exists(microtex_rgba::g_res_root + "/" + name)) {
    resolved = microtex_rgba::g_res_root + "/" + name;
  } else {
    const std::string cmr = microtex_rgba::g_res_root + "/fonts/latin/cmr10.ttf";
    if (std::filesystem::exists(cmr)) {
      resolved = cmr;
    } else {
      resolved = name;
    }
  }
  return RemapToReadableFont(resolved);
}

struct TtfData {
  std::vector<unsigned char> bytes;
  stbtt_fontinfo info{};
  bool ok = false;
};

std::mutex g_font_cache_mutex;
std::map<std::string, std::shared_ptr<TtfData>> g_font_cache;

std::shared_ptr<TtfData> LoadTtf(const std::string& path) {
  std::lock_guard<std::mutex> lock(g_font_cache_mutex);
  const auto it = g_font_cache.find(path);
  if (it != g_font_cache.end()) {
    return it->second;
  }
  auto data = std::make_shared<TtfData>();
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    g_font_cache[path] = data;
    return data;
  }
  in.seekg(0, std::ios::end);
  const std::streamsize sz = in.tellg();
  in.seekg(0, std::ios::beg);
  if (sz <= 0) {
    g_font_cache[path] = data;
    return data;
  }
  data->bytes.resize(static_cast<size_t>(sz));
  if (!in.read(reinterpret_cast<char*>(data->bytes.data()), sz)) {
    g_font_cache[path] = data;
    return data;
  }
  if (!stbtt_InitFont(&data->info, data->bytes.data(),
                      stbtt_GetFontOffsetForIndex(data->bytes.data(), 0))) {
    data->bytes.clear();
    g_font_cache[path] = data;
    return data;
  }
  data->ok = true;
  g_font_cache[path] = data;
  return data;
}

}  // namespace

namespace tex {

class Font_rgba : public Font {
 public:
  std::shared_ptr<TtfData> ttf;
  std::string path;
  int style = PLAIN;
  float size = 20.f;
  float scale = 1.f;
  int ascent = 0;
  int descent = 0;

  Font_rgba(const std::string& file, float s) : path(ResolveFontFile(file)), size(s) {
    ttf = LoadTtf(path);
    if (ttf && ttf->ok) {
      scale = stbtt_ScaleForPixelHeight(&ttf->info, size);
      int a = 0, d = 0, gap = 0;
      stbtt_GetFontVMetrics(&ttf->info, &a, &d, &gap);
      ascent = static_cast<int>(std::lround(a * scale));
      descent = static_cast<int>(std::lround(d * scale));
    }
  }

  Font_rgba(const std::string& name, int st, float s) : style(st), size(s) {
    path = ResolveFontFile(name);
    ttf = LoadTtf(path);
    if (ttf && ttf->ok) {
      scale = stbtt_ScaleForPixelHeight(&ttf->info, size);
      int a = 0, d = 0, gap = 0;
      stbtt_GetFontVMetrics(&ttf->info, &a, &d, &gap);
      ascent = static_cast<int>(std::lround(a * scale));
      descent = static_cast<int>(std::lround(d * scale));
    }
  }

  float getSize() const override { return size; }

  sptr<Font> deriveFont(int st) const override {
    return sptrOf<Font_rgba>(path, st, size);
  }

  bool operator==(const Font& o) const override {
    const auto& f = static_cast<const Font_rgba&>(o);
    return path == f.path && style == f.style && size == f.size;
  }

  bool operator!=(const Font& o) const override { return !(*this == o); }
};

class TextLayout_rgba : public TextLayout {
 public:
  std::wstring text;
  sptr<Font_rgba> font;

  TextLayout_rgba(const std::wstring& src, const sptr<Font_rgba>& f) : text(src), font(f) {}

  void getBounds(Rect& bounds) override {
    bounds = {0, 0, 0, 0};
    if (!font || !font->ttf || !font->ttf->ok) {
      return;
    }
    float w = 0;
    for (wchar_t ch : text) {
      int adv = 0;
      stbtt_GetCodepointHMetrics(&font->ttf->info, static_cast<int>(ch), &adv, nullptr);
      w += adv * font->scale;
    }
    bounds.w = w;
    bounds.h = font->ascent - font->descent;
    bounds.y = -static_cast<float>(font->ascent);
  }

  void draw(Graphics2D& g2, float x, float y) override { g2.drawText(text, x, y); }
};

class Graphics2D_rgba : public Graphics2D {
  struct Affine {
    float a = 1, b = 0, c = 0, d = 1, e = 0, f = 0;
    void reset() {
      a = d = 1;
      b = c = e = f = 0;
    }
    Affine multiplied(const Affine& o) const {
      Affine r;
      r.a = a * o.a + c * o.b;
      r.b = b * o.a + d * o.b;
      r.c = a * o.c + c * o.d;
      r.d = b * o.c + d * o.d;
      r.e = a * o.e + c * o.f + e;
      r.f = b * o.e + d * o.f + f;
      return r;
    }
    void map(float& x, float& y) const {
      x = a * x + c * y + e;
      y = b * x + d * y + f;
    }
  };

 public:
  ::microtex_rgba::RgbaSurface* surface = nullptr;
  color fg = black;
  Stroke stroke;
  const Font_rgba* font = nullptr;
  Affine transform;
  std::vector<Affine> stack;
  float sx_acc = 1.f;
  float sy_acc = 1.f;

  explicit Graphics2D_rgba(::microtex_rgba::RgbaSurface* s) : surface(s) {}

  void setColor(color c) override { fg = c; }
  color getColor() const override { return fg; }
  void setStroke(const Stroke& s) override { stroke = s; }
  const Stroke& getStroke() const override { return stroke; }
  void setStrokeWidth(float w) override { stroke.lineWidth = w; }
  const Font* getFont() const override { return font; }
  void setFont(const Font* f) override { font = static_cast<const Font_rgba*>(f); }

  void translate(float dx, float dy) override {
    Affine t;
    t.e = dx;
    t.f = dy;
    transform = transform.multiplied(t);
  }

  void scale(float sx, float sy) override {
    sx_acc *= sx;
    sy_acc *= sy;
    Affine s;
    s.a = sx;
    s.d = sy;
    transform = transform.multiplied(s);
  }

  void rotate(float angle) override {
    const float c = std::cos(angle);
    const float s = std::sin(angle);
    Affine r;
    r.a = c;
    r.b = s;
    r.c = -s;
    r.d = c;
    transform = transform.multiplied(r);
  }

  void rotate(float angle, float px, float py) override {
    translate(px, py);
    rotate(angle);
    translate(-px, -py);
  }

  void reset() override {
    transform.reset();
    stack.clear();
    sx_acc = sy_acc = 1.f;
  }

  float sx() const override { return sx_acc; }
  float sy() const override { return sy_acc; }

  void drawChar(wchar_t c, float x, float y) override {
    std::wstring s(1, c);
    drawText(s, x, y);
  }

  void drawText(const std::wstring& text, float x, float y) override {
    if (!surface || !font || !font->ttf || !font->ttf->ok) {
      return;
    }
    const uint8_t r = static_cast<uint8_t>(color_r(fg));
    const uint8_t g = static_cast<uint8_t>(color_g(fg));
    const uint8_t b = static_cast<uint8_t>(color_b(fg));
  const uint8_t a = static_cast<uint8_t>(color_a(fg));

    float pen_x = x;
    float pen_y = y;
    transform.map(pen_x, pen_y);
    const float glyph_scale = font->scale * sx_acc;

    for (wchar_t ch : text) {
      int advance = 0, lsb = 0;
      stbtt_GetCodepointHMetrics(&font->ttf->info, static_cast<int>(ch), &advance, &lsb);
      int x0 = 0, y0 = 0, x1 = 0, y1 = 0;
      stbtt_GetCodepointBitmapBox(&font->ttf->info, static_cast<int>(ch), glyph_scale, glyph_scale,
                                  &x0, &y0, &x1, &y1);
      const int bw = x1 - x0;
      const int bh = y1 - y0;
      if (bw > 0 && bh > 0) {
        std::vector<unsigned char> glyph(static_cast<size_t>(bw * bh));
        stbtt_MakeCodepointBitmap(&font->ttf->info, glyph.data(), bw, bh, bw, glyph_scale,
                                  glyph_scale, static_cast<int>(ch));
        const int dst_x = static_cast<int>(std::lround(pen_x + x0));
        const int dst_y = static_cast<int>(std::lround(pen_y + y0));
        for (int gy = 0; gy < bh; ++gy) {
          for (int gx = 0; gx < bw; ++gx) {
            const uint8_t ga = glyph[static_cast<size_t>(gy * bw + gx)];
            if (ga == 0) {
              continue;
            }
            const uint8_t fa = static_cast<uint8_t>((ga * a) / 255);
            surface->blendPixel(dst_x + gx, dst_y + gy, r, g, b, fa);
          }
        }
      }
      pen_x += advance * glyph_scale;
    }
  }

  void drawLine(float x1, float y1, float x2, float y2) override {
    if (!surface) {
      return;
    }
    transform.map(x1, y1);
    transform.map(x2, y2);
    surface->drawLine(static_cast<int>(std::lround(x1)), static_cast<int>(std::lround(y1)),
                      static_cast<int>(std::lround(x2)), static_cast<int>(std::lround(y2)),
                      static_cast<uint8_t>(color_r(fg)), static_cast<uint8_t>(color_g(fg)),
                      static_cast<uint8_t>(color_b(fg)), static_cast<uint8_t>(color_a(fg)),
                      stroke.lineWidth);
  }

  void drawRect(float x, float y, float w, float h) override { drawLine(x, y, x + w, y); }

  void fillRect(float x, float y, float w, float h) override {
    if (!surface) {
      return;
    }
    float corners[4][2] = {{x, y}, {x + w, y}, {x + w, y + h}, {x, y + h}};
    float minx = corners[0][0], maxx = corners[0][0];
    float miny = corners[0][1], maxy = corners[0][1];
    for (auto& c : corners) {
      transform.map(c[0], c[1]);
      minx = std::min(minx, c[0]);
      maxx = std::max(maxx, c[0]);
      miny = std::min(miny, c[1]);
      maxy = std::max(maxy, c[1]);
    }
    surface->fillRect(static_cast<int>(std::floor(minx)), static_cast<int>(std::floor(miny)),
                      static_cast<int>(std::ceil(maxx - minx)), static_cast<int>(std::ceil(maxy - miny)),
                      static_cast<uint8_t>(color_r(fg)), static_cast<uint8_t>(color_g(fg)),
                      static_cast<uint8_t>(color_b(fg)), static_cast<uint8_t>(color_a(fg)));
  }

  void drawRoundRect(float x, float y, float w, float h, float rx, float ry) override {
    drawRect(x, y, w, h);
  }

  void fillRoundRect(float x, float y, float w, float h, float rx, float ry) override {
    fillRect(x, y, w, h);
  }
};

Font* Font::create(const std::string& file, float size) {
  return new Font_rgba(file, size);
}

sptr<Font> Font::_create(const std::string& name, int style, float size) {
  return sptrOf<Font_rgba>(name, style, size);
}

sptr<TextLayout> TextLayout::create(const std::wstring& src, const sptr<Font>& font) {
  sptr<Font_rgba> f = static_pointer_cast<Font_rgba>(font);
  return sptrOf<TextLayout_rgba>(src, f);
}

}  // namespace tex

#include "render.h"

namespace microtex_rgba {

void DrawTeXRender(tex::TeXRender* render, RgbaSurface* surface, int padding) {
  if (!render || !surface) {
    return;
  }
  tex::Graphics2D_rgba g2(surface);
  render->draw(g2, padding, padding);
}

}  // namespace microtex_rgba

#endif  // BUILD_RGBA

#ifdef USE_MICROTEX_LATEX_PREVIEW

#include "latex.h"
#include "render.h"
#include "utils/exceptions.h"
#include "utils/utf.h"

#include <cctype>
#include <mutex>

namespace {

std::mutex g_init_mutex;
bool g_ready = false;

tex::color HexToColor(const std::string& hex) {
  std::string s = hex;
  if (!s.empty() && s[0] == '#') {
    s.erase(s.begin());
  }
  if (s.size() == 6) {
    return tex::decode("#FF" + s);
  }
  if (s.size() == 8) {
    return tex::decode("#" + s);
  }
  return tex::rgb(224, 224, 224);
}

}  // namespace

void MicroTeXInit(const std::string& res_root) {
  std::lock_guard<std::mutex> lock(g_init_mutex);
  if (g_ready) {
    return;
  }
  microtex_rgba::SetResRoot(res_root);
  tex::LaTeX::init(res_root);
  g_ready = true;
}

void MicroTeXShutdown() {
  std::lock_guard<std::mutex> lock(g_init_mutex);
  if (!g_ready) {
    return;
  }
  tex::LaTeX::release();
  g_ready = false;
}

LatexBitmap RenderMicroTeXToBitmap(const std::string& latex, const LatexRenderStyle& style) {
  LatexBitmap out;
  if (latex.empty()) {
    out.error = "empty latex";
    return out;
  }
  std::lock_guard<std::mutex> lock(g_init_mutex);
  if (!g_ready) {
    out.error = "MicroTeX not initialized";
    return out;
  }

  const std::wstring wide = tex::utf82wide(latex);
  tex::TeXRender* render = nullptr;
  try {
    render = tex::LaTeX::parse(wide, 2048, static_cast<float>(std::max(8, style.font_size)),
                               static_cast<float>(std::max(8, style.font_size)) / 3.f,
                               HexToColor(style.fg_hex));
  } catch (const tex::ex_parse& ex) {
    out.error = ex.what();
    return out;
  } catch (const std::exception& ex) {
    out.error = ex.what();
    return out;
  } catch (...) {
    out.error = "MicroTeX parse failed";
    return out;
  }

  const int w = std::max(1, render->getWidth() + style.padding * 2);
  const int h = std::max(1, render->getHeight() + style.padding * 2);
  microtex_rgba::RgbaSurface surface(w, h);
  surface.clear(0, 0, 0, 0);
  try {
    microtex_rgba::DrawTeXRender(render, &surface, style.padding);
  } catch (...) {
    delete render;
    out.error = "MicroTeX draw failed";
    return out;
  }
  delete render;
  out = surface.toBitmap();
  if (out.rgba.empty()) {
    out.error = "empty bitmap";
  }
  return out;
}

#endif  // USE_MICROTEX_LATEX_PREVIEW
