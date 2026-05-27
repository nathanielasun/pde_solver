#include "latex/latex_bitmap.h"

#ifdef USE_MICROTEX_LATEX_PREVIEW
#include "latex/microtex_rgba_graphic.h"
#endif

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>

namespace {

bool RenderOk(const LatexBitmap& bmp, std::string* error) {
  if (!bmp.error.empty()) {
    *error = bmp.error;
    return false;
  }
  if (bmp.width <= 0 || bmp.height <= 0 || bmp.rgba.empty()) {
    *error = "empty bitmap";
    return false;
  }
  return true;
}

std::filesystem::path FindResDir() {
  const std::filesystem::path cwd = std::filesystem::current_path();
  if (std::filesystem::exists(cwd / "res" / "fonts")) {
    return cwd / "res";
  }
  const std::filesystem::path dev = cwd / "third_party" / "MicroTeX" / "res";
  if (std::filesystem::exists(dev / "fonts")) {
    return dev;
  }
  return {};
}

}  // namespace

int main() {
#ifdef USE_MICROTEX_LATEX_PREVIEW
  const std::filesystem::path res_dir = FindResDir();
  if (res_dir.empty()) {
    std::cerr << "MicroTeX res/ not found\n";
    return 1;
  }
  MicroTeXInit(res_dir.string());
#endif

  LatexRenderStyle style;
  style.font_size = 20;
  style.fg_hex = "#e8e8e8";

  struct Case {
    const char* name;
    const char* latex;
    int min_height;
  };
  const Case cases[] = {
      {"burgers", "u_t + u u_x = 0.01 u_{xx}", 20},
      {"fraction", R"(\frac{xy}{y^2\sin(x)})", 28},
      {"coeff_deriv", R"(k\sin(x)\,\partial_{xx} u)", 20},
  };

  for (const auto& c : cases) {
    const auto t0 = std::chrono::steady_clock::now();
    LatexBitmap bmp = RenderLatexToBitmap(c.latex, style);
    const auto t1 = std::chrono::steady_clock::now();
    std::string error;
    if (!RenderOk(bmp, &error)) {
      std::cerr << c.name << " failed: " << error << "\n";
      return 1;
    }
    if (bmp.height < c.min_height) {
      std::cerr << c.name << " height " << bmp.height << " < " << c.min_height << "\n";
      return 1;
    }
    const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    std::cout << c.name << " ok " << bmp.width << "x" << bmp.height << " ms=" << ms << "\n";
  }

#ifdef USE_MICROTEX_LATEX_PREVIEW
  MicroTeXShutdown();
#endif
  std::cout << "latex_render_smoke_test passed\n";
  return 0;
}
