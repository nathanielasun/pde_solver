#ifndef LATEX_TEXTURE_H
#define LATEX_TEXTURE_H

#include <chrono>
#include <string>

struct LatexTexture {
  std::string source;
  std::string last_rendered;
  std::string error;
  std::string color;
  unsigned int texture = 0;
  int width = 0;
  int height = 0;
  int font_size = 18;
  bool dirty = false;
  bool pending = false;
  std::chrono::steady_clock::time_point last_edit;
};

#endif  // LATEX_TEXTURE_H
