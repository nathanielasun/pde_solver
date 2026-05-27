#include "latex/latex_preview_draw.h"

#include <cstring>

#include "imgui.h"

#include <OpenGL/gl3.h>

#include "stb_image.h"

bool UploadTextureFromRGBA(unsigned int* texture, int* width, int* height,
                           const std::vector<uint8_t>& rgba, int w, int h, std::string* error) {
  if (w <= 0 || h <= 0 || rgba.size() < static_cast<size_t>(w * h * 4)) {
    if (error) {
      *error = "invalid bitmap dimensions";
    }
    return false;
  }
  if (*texture == 0) {
    glGenTextures(1, reinterpret_cast<GLuint*>(texture));
  }
  glBindTexture(GL_TEXTURE_2D, *reinterpret_cast<GLuint*>(texture));
  glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, rgba.data());
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glBindTexture(GL_TEXTURE_2D, 0);
  if (width) {
    *width = w;
  }
  if (height) {
    *height = h;
  }
  return true;
}

bool LoadTextureFromFile(const std::filesystem::path& path, unsigned int* texture, int* width,
                         int* height, std::string* error) {
  int w = 0;
  int h = 0;
  int channels = 0;
  stbi_uc* data = stbi_load(path.string().c_str(), &w, &h, &channels, 4);
  if (!data) {
    if (error) {
      *error = stbi_failure_reason() ? stbi_failure_reason() : "failed to load png";
    }
    return false;
  }
  std::vector<uint8_t> rgba(static_cast<size_t>(w * h * 4));
  std::memcpy(rgba.data(), data, rgba.size());
  stbi_image_free(data);
  return UploadTextureFromRGBA(texture, width, height, rgba, w, h, error);
}

void DrawLatexPreview(const LatexTexture& tex, float max_width, float max_height) {
  if (tex.texture == 0) {
    return;
  }
  float w = static_cast<float>(tex.width);
  float h = static_cast<float>(tex.height);
  if (w <= 0.0f || h <= 0.0f) {
    w = max_width;
    h = max_height;
  }
  float scale = 1.0f;
  if (w > max_width) {
    scale = max_width / w;
  }
  if (h * scale > max_height) {
    scale = std::min(scale, max_height / h);
  }
  ImGui::Image(reinterpret_cast<void*>(static_cast<intptr_t>(tex.texture)),
               ImVec2(w * scale, h * scale), ImVec2(0, 0), ImVec2(1, 1));
}

void DrawLatexPreviewError(const LatexTexture& tex, const std::string& parse_message,
                           bool parse_ok, float max_width) {
  if (!tex.error.empty()) {
    ImGui::TextColored(ImVec4(1.0f, 0.35f, 0.35f, 1.0f), "Render: %s", tex.error.c_str());
  }
  if (!parse_message.empty()) {
    const ImVec4 color = parse_ok ? ImVec4(0.4f, 0.9f, 0.5f, 1.0f) : ImVec4(1.0f, 0.75f, 0.3f, 1.0f);
    ImGui::TextColored(color, "Parse: %s", parse_message.c_str());
  }
  if (!tex.source.empty()) {
    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.75f, 0.75f, 0.8f, 1.0f));
    ImGui::PushFont(ImGui::GetIO().Fonts->Fonts.empty() ? nullptr : ImGui::GetIO().Fonts->Fonts[0]);
    ImGui::TextWrapped("%s", tex.source.c_str());
    ImGui::PopFont();
    ImGui::PopStyleColor();
  }
  (void)max_width;
}
