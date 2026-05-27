#ifndef LATEX_PREVIEW_DRAW_H
#define LATEX_PREVIEW_DRAW_H

#include "latex/latex_texture.h"
#include <filesystem>
#include <string>
#include <vector>

bool UploadTextureFromRGBA(unsigned int* texture, int* width, int* height,
                           const std::vector<uint8_t>& rgba, int w, int h, std::string* error);

bool LoadTextureFromFile(const std::filesystem::path& path, unsigned int* texture, int* width,
                         int* height, std::string* error);

void DrawLatexPreview(const LatexTexture& tex, float max_width, float max_height);

void DrawLatexPreviewError(const LatexTexture& tex, const std::string& parse_message,
                           bool parse_ok, float max_width);

#endif  // LATEX_PREVIEW_DRAW_H
