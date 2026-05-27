#include "latex/latex_bitmap.h"
#include "latex/preview_font_paths.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <mutex>
#include <vector>

#define STB_TRUETYPE_IMPLEMENTATION
#define STBTT_STATIC
#include "imstb_truetype.h"

namespace {

struct FontAtlas {
  std::vector<unsigned char> ttf;
  stbtt_fontinfo info{};
  bool ok = false;
  float scale = 1.0f;
  int ascent = 0;
  int descent = 0;
};

std::mutex g_font_mutex;
FontAtlas g_font;

bool ParseHexColor(const std::string& hex, uint8_t* r, uint8_t* g, uint8_t* b) {
  std::string s = hex;
  if (!s.empty() && s[0] == '#') {
    s.erase(s.begin());
  }
  if (s.size() != 6) {
    return false;
  }
  unsigned int rv = 0;
  unsigned int gv = 0;
  unsigned int bv = 0;
  if (std::sscanf(s.c_str(), "%2x%2x%2x", &rv, &gv, &bv) != 3) {
    return false;
  }
  *r = static_cast<uint8_t>(rv);
  *g = static_cast<uint8_t>(gv);
  *b = static_cast<uint8_t>(bv);
  return true;
}

bool LoadFontOnce() {
  std::lock_guard<std::mutex> lock(g_font_mutex);
  if (g_font.ok) {
    return true;
  }
  const std::vector<std::filesystem::path> candidates = {
      FindPreviewSansBoldFont(),
      "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
      "/System/Library/Fonts/Supplemental/Comic Sans MS Bold.ttf",
      "gui_gl/assets/fonts/DejaVuSans-Bold.ttf",
      "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
      "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
      "/System/Library/Fonts/Supplemental/Arial.ttf",
      "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
  };
  for (const auto& path : candidates) {
    if (!std::filesystem::exists(path)) {
      continue;
    }
    FILE* f = std::fopen(path.string().c_str(), "rb");
    if (!f) {
      continue;
    }
    std::fseek(f, 0, SEEK_END);
    const long sz = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    if (sz <= 0) {
      std::fclose(f);
      continue;
    }
    g_font.ttf.resize(static_cast<size_t>(sz));
    if (std::fread(g_font.ttf.data(), 1, g_font.ttf.size(), f) != g_font.ttf.size()) {
      std::fclose(f);
      g_font.ttf.clear();
      continue;
    }
    std::fclose(f);
    if (!stbtt_InitFont(&g_font.info, g_font.ttf.data(),
                        stbtt_GetFontOffsetForIndex(g_font.ttf.data(), 0))) {
      g_font.ttf.clear();
      continue;
    }
    g_font.ok = true;
    return true;
  }
  return false;
}

void EnsureFontScale(int font_size) {
  if (!g_font.ok) {
    return;
  }
  g_font.scale = stbtt_ScaleForPixelHeight(&g_font.info, static_cast<float>(font_size));
  int a = 0;
  int d = 0;
  int line_gap = 0;
  stbtt_GetFontVMetrics(&g_font.info, &a, &d, &line_gap);
  g_font.ascent = static_cast<int>(std::lround(static_cast<float>(a) * g_font.scale));
  g_font.descent = static_cast<int>(std::lround(static_cast<float>(d) * g_font.scale));
}

std::string PreprocessLatex(std::string s) {
  auto replace_all = [&](const std::string& from, const std::string& to) {
    size_t pos = 0;
    while ((pos = s.find(from, pos)) != std::string::npos) {
      s.replace(pos, from.size(), to);
      pos += to.size();
    }
  };
  if (s.size() >= 2 && s.front() == '$' && s.back() == '$') {
    s = s.substr(1, s.size() - 2);
  }
  replace_all("\\partial", "\xE2\x88\x82");  // ∂
  replace_all("\\nabla", "\xE2\x88\x87");    // ∇
  replace_all("\\cdot", "\xC2\xB7");        // ·
  replace_all("\\times", "\xC3\x97");        // ×
  replace_all("\\leq", "\xE2\x89\xA4");
  replace_all("\\geq", "\xE2\x89\xA5");
  replace_all("\\neq", "\xE2\x89\xA0");
  replace_all("\\infty", "\xE2\x88\x9E");
  replace_all("\\alpha", "\xCE\xB1");
  replace_all("\\beta", "\xCE\xB2");
  replace_all("\\gamma", "\xCE\xB3");
  replace_all("\\Delta", "\xCE\x94");
  replace_all("\\pi", "\xCF\x80");
  replace_all("\\theta", "\xCE\xB8");
  replace_all("\\lambda", "\xCE\xBB");
  replace_all("\\mu", "\xCE\xBC");
  replace_all("\\nu", "\xCE\xBD");
  replace_all("\\rho", "\xCF\x81");
  replace_all("\\sigma", "\xCF\x83");
  replace_all("\\phi", "\xCF\x86");
  replace_all("\\psi", "\xCF\x88");
  replace_all("\\omega", "\xCF\x89");
  replace_all("\\left", "");
  replace_all("\\right", "");
  replace_all("\\,", " ");
  replace_all("\\;", " ");
  replace_all("\\quad", "  ");
  replace_all("\\qquad", "    ");
  replace_all("{", "");
  replace_all("}", "");
  return s;
}

struct GlyphRun {
  std::string text;
  float x = 0.0f;
  float y = 0.0f;
  float scale = 1.0f;
};

int NextCodepoint(const std::string& text, size_t& index) {
  if (index >= text.size()) {
    return 0;
  }
  const unsigned char c0 = static_cast<unsigned char>(text[index]);
  if (c0 < 0x80) {
    return text[index++];
  }
  if ((c0 & 0xE0) == 0xC0 && index + 1 < text.size()) {
    const int cp = ((c0 & 0x1F) << 6) | (static_cast<unsigned char>(text[index + 1]) & 0x3F);
    index += 2;
    return cp;
  }
  if ((c0 & 0xF0) == 0xE0 && index + 2 < text.size()) {
    const int cp = ((c0 & 0x0F) << 12) |
                   ((static_cast<unsigned char>(text[index + 1]) & 0x3F) << 6) |
                   (static_cast<unsigned char>(text[index + 2]) & 0x3F);
    index += 3;
    return cp;
  }
  return text[index++];
}

float MeasureTextWidth(const std::string& text, float scale) {
  float w = 0.0f;
  for (size_t i = 0; i < text.size();) {
    const int cp = NextCodepoint(text, i);
    int advance = 0;
    stbtt_GetCodepointHMetrics(&g_font.info, cp, &advance, nullptr);
    w += static_cast<float>(advance) * scale;
  }
  return w;
}

std::vector<GlyphRun> LayoutRuns(const std::string& latex, int font_size) {
  std::vector<GlyphRun> runs;
  const std::string text = PreprocessLatex(latex);
  float x = 0.0f;
  const float base_scale = 1.0f;
  const float sub_scale = 0.72f;
  const float sup_scale = 0.72f;
  const float sub_drop = static_cast<float>(font_size) * 0.28f;
  const float sup_rise = static_cast<float>(font_size) * 0.55f;

  std::string current;
  float y = 0.0f;
  float scale = base_scale;
  bool in_sub = false;
  bool in_sup = false;

  auto flush = [&]() {
    if (current.empty()) {
      return;
    }
    runs.push_back({current, x, y, scale});
    x += MeasureTextWidth(current, g_font.scale * scale);
    current.clear();
  };

  for (size_t i = 0; i < text.size(); ++i) {
    const char c = text[i];
    if (c == '_') {
      flush();
      in_sub = true;
      in_sup = false;
      y = sub_drop;
      scale = sub_scale;
      continue;
    }
    if (c == '^') {
      flush();
      in_sup = true;
      in_sub = false;
      y = -sup_rise;
      scale = sup_scale;
      continue;
    }
    if (c == ' ' && !in_sub && !in_sup) {
      flush();
      x += static_cast<float>(font_size) * 0.25f;
      continue;
    }
    if ((c == '+' || c == '=' || c == '-') && (in_sub || in_sup)) {
      flush();
      in_sub = false;
      in_sup = false;
      y = 0.0f;
      scale = base_scale;
    }
    current.push_back(c);
  }
  flush();
  return runs;
}

}  // namespace

LatexBitmap RenderMathtextToBitmap(const std::string& latex, const LatexRenderStyle& style) {
  LatexBitmap out;
  if (latex.empty()) {
    out.error = "empty latex";
    return out;
  }
  if (!LoadFontOnce()) {
    out.error = "no system font found for LaTeX preview (install Arial Bold or DejaVu Sans Bold)";
    return out;
  }

  const int font_size = std::max(8, style.font_size);
  EnsureFontScale(font_size);

  const std::vector<GlyphRun> runs = LayoutRuns(latex, font_size);
  float total_w = static_cast<float>(style.padding) * 2.0f;
  for (const auto& run : runs) {
    total_w += MeasureTextWidth(run.text, g_font.scale * run.scale);
  }
  const int h = font_size + style.padding * 2;
  const int w = std::max(1, static_cast<int>(std::ceil(total_w)));

  out.width = w;
  out.height = h;
  out.rgba.assign(static_cast<size_t>(w * h * 4), 0);

  uint8_t fr = 224;
  uint8_t fg = 224;
  uint8_t fb = 224;
  if (!ParseHexColor(style.fg_hex, &fr, &fg, &fb)) {
    fr = fg = fb = 224;
  }

  float pen_x = static_cast<float>(style.padding);
  const float baseline = static_cast<float>(style.padding + g_font.ascent);

  for (const auto& run : runs) {
    const float run_scale = g_font.scale * run.scale;
    float x = pen_x;
    for (size_t i = 0; i < run.text.size();) {
      const int cp = NextCodepoint(run.text, i);
      if (cp == 0) {
        break;
      }
      int advance = 0;
      int lsb = 0;
      stbtt_GetCodepointHMetrics(&g_font.info, cp, &advance, &lsb);

      int gw = 0;
      int gh = 0;
      int x0 = 0;
      int y0 = 0;
      int x1 = 0;
      int y1 = 0;
      stbtt_GetCodepointBitmapBox(&g_font.info, cp, run_scale, run_scale, &x0, &y0, &x1, &y1);
      const int bw = x1 - x0;
      const int bh = y1 - y0;
      if (bw > 0 && bh > 0) {
        std::vector<unsigned char> glyph(static_cast<size_t>(bw * bh), 0);
        stbtt_MakeCodepointBitmap(&g_font.info, glyph.data(), bw, bh, bw, run_scale, run_scale, cp);
        const int dst_x = static_cast<int>(std::lround(x + static_cast<float>(x0)));
        const int dst_y =
            static_cast<int>(std::lround(baseline + run.y + static_cast<float>(y0)));
        for (int gy = 0; gy < bh; ++gy) {
          const int py = dst_y + gy;
          if (py < 0 || py >= h) {
            continue;
          }
          for (int gx = 0; gx < bw; ++gx) {
            const int px = dst_x + gx;
            if (px < 0 || px >= w) {
              continue;
            }
            const uint8_t alpha = glyph[static_cast<size_t>(gy * bw + gx)];
            if (alpha == 0) {
              continue;
            }
            uint8_t* pixel = &out.rgba[static_cast<size_t>((py * w + px) * 4)];
            pixel[0] = fr;
            pixel[1] = fg;
            pixel[2] = fb;
            pixel[3] = std::max(pixel[3], alpha);
          }
        }
      }
      x += static_cast<float>(advance) * run_scale;
    }
    pen_x = x;
  }
  return out;
}
