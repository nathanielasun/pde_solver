#include "animation_export_panel.h"
#include "imgui.h"
#include "vtk_io.h"
#include "utils/file_dialog.h"

#include <OpenGL/gl3.h>
#include "../../third_party/glfw/deps/stb_image_write.h"

#include <atomic>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

// ---------------------------------------------------------------------------
// Static panel state (persists across frames)
// ---------------------------------------------------------------------------
static int s_format = 1;  // Default to PNG Sequence (the most reliable)
static int s_width = 1280;
static int s_height = 720;
static int s_fps = 24;
static int s_start_frame = 0;
static int s_end_frame = -1;
static bool s_show_timestamp = true;

// Export progress / threading state
static std::atomic<bool> s_exporting{false};
static std::atomic<bool> s_cancel_requested{false};
static std::atomic<int> s_export_current{0};
static std::atomic<int> s_export_total{0};
static std::atomic<bool> s_export_done{false};
static std::atomic<bool> s_export_success{false};
static std::string s_export_error;
static std::string s_export_output_dir;

// ---------------------------------------------------------------------------
// Helper: capture a single frame to a pixel buffer (RGBA, top-left origin)
// Must be called on the GL thread.
// ---------------------------------------------------------------------------
static bool CaptureFrame(GlViewer& viewer, int width, int height,
                         std::vector<unsigned char>& outPixels) {
  viewer.RenderToTexture(width, height);

  GLuint texture = viewer.texture();
  if (texture == 0) return false;

  outPixels.resize(static_cast<size_t>(width) * height * 4);

  GLuint readFbo = 0;
  glGenFramebuffers(1, &readFbo);
  glBindFramebuffer(GL_READ_FRAMEBUFFER, readFbo);
  glFramebufferTexture2D(GL_READ_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                         GL_TEXTURE_2D, texture, 0);

  if (glCheckFramebufferStatus(GL_READ_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
    glBindFramebuffer(GL_READ_FRAMEBUFFER, 0);
    glDeleteFramebuffers(1, &readFbo);
    return false;
  }

  std::vector<unsigned char> rawPixels(static_cast<size_t>(width) * height * 4);
  glReadPixels(0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, rawPixels.data());

  glBindFramebuffer(GL_READ_FRAMEBUFFER, 0);
  glDeleteFramebuffers(1, &readFbo);

  // Flip vertically (OpenGL origin is bottom-left, images are top-left)
  for (int y = 0; y < height; ++y) {
    int srcY = height - 1 - y;
    std::copy(rawPixels.begin() + (srcY * width * 4),
              rawPixels.begin() + ((srcY + 1) * width * 4),
              outPixels.begin() + (y * width * 4));
  }
  return true;
}

// ---------------------------------------------------------------------------
// Helper: burn a simple timestamp string into the pixel buffer (top-left).
// Uses a minimal 5x7 bitmap font.  Only digits, '.', 't', '=', ' ', '-'.
// ---------------------------------------------------------------------------
namespace {

// 5-wide, 7-tall glyphs stored as 7 bytes per character (MSB = left).
// Only the subset we need: 0-9, '.', 't', '=', ' ', '-', 's'
struct GlyphEntry { char ch; unsigned char rows[7]; };

// clang-format off
static const GlyphEntry kGlyphs[] = {
  {'0', {0x70,0x88,0x98,0xA8,0xC8,0x88,0x70}},
  {'1', {0x20,0x60,0x20,0x20,0x20,0x20,0x70}},
  {'2', {0x70,0x88,0x08,0x10,0x20,0x40,0xF8}},
  {'3', {0xF8,0x10,0x20,0x10,0x08,0x88,0x70}},
  {'4', {0x10,0x30,0x50,0x90,0xF8,0x10,0x10}},
  {'5', {0xF8,0x80,0xF0,0x08,0x08,0x88,0x70}},
  {'6', {0x30,0x40,0x80,0xF0,0x88,0x88,0x70}},
  {'7', {0xF8,0x08,0x10,0x20,0x40,0x40,0x40}},
  {'8', {0x70,0x88,0x88,0x70,0x88,0x88,0x70}},
  {'9', {0x70,0x88,0x88,0x78,0x08,0x10,0x60}},
  {'.', {0x00,0x00,0x00,0x00,0x00,0x60,0x60}},
  {'t', {0x40,0x40,0xE0,0x40,0x40,0x48,0x30}},
  {'=', {0x00,0x00,0xF8,0x00,0xF8,0x00,0x00}},
  {' ', {0x00,0x00,0x00,0x00,0x00,0x00,0x00}},
  {'-', {0x00,0x00,0x00,0xF8,0x00,0x00,0x00}},
  {'s', {0x00,0x00,0x78,0x80,0x70,0x08,0xF0}},
  {'e', {0x00,0x00,0x70,0x88,0xF8,0x80,0x70}},
};
// clang-format on

static const unsigned char* FindGlyph(char ch) {
  for (const auto& g : kGlyphs) {
    if (g.ch == ch) return g.rows;
  }
  return nullptr;
}

void BurnTimestamp(std::vector<unsigned char>& pixels, int imgW, int imgH,
                   const std::string& text, int scale = 2) {
  const int glyphW = 5;
  const int glyphH = 7;
  const int pad = 6 * scale;
  int cursorX = pad;
  int startY = pad;

  // Semi-transparent black background behind text
  int bgW = static_cast<int>(text.size()) * (glyphW + 1) * scale + pad * 2;
  int bgH = glyphH * scale + pad * 2;
  for (int y = 0; y < bgH && y < imgH; ++y) {
    for (int x = 0; x < bgW && x < imgW; ++x) {
      int idx = (y * imgW + x) * 4;
      // Blend 60% black over existing pixel
      pixels[idx + 0] = static_cast<unsigned char>(pixels[idx + 0] * 0.4);
      pixels[idx + 1] = static_cast<unsigned char>(pixels[idx + 1] * 0.4);
      pixels[idx + 2] = static_cast<unsigned char>(pixels[idx + 2] * 0.4);
    }
  }

  for (char ch : text) {
    const unsigned char* glyph = FindGlyph(ch);
    if (!glyph) { cursorX += (glyphW + 1) * scale; continue; }

    for (int row = 0; row < glyphH; ++row) {
      unsigned char bits = glyph[row];
      for (int col = 0; col < glyphW; ++col) {
        if (bits & (0x80 >> col)) {
          // Draw a scale x scale block of white pixels
          for (int sy = 0; sy < scale; ++sy) {
            for (int sx = 0; sx < scale; ++sx) {
              int px = cursorX + col * scale + sx;
              int py = startY + row * scale + sy;
              if (px >= 0 && px < imgW && py >= 0 && py < imgH) {
                int idx = (py * imgW + px) * 4;
                pixels[idx + 0] = 255;
                pixels[idx + 1] = 255;
                pixels[idx + 2] = 255;
                pixels[idx + 3] = 255;
              }
            }
          }
        }
      }
    }
    cursorX += (glyphW + 1) * scale;
  }
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// Helper: check if ffmpeg is available on PATH
// ---------------------------------------------------------------------------
static bool FfmpegAvailable() {
#ifdef _WIN32
  return std::system("ffmpeg -version >nul 2>&1") == 0;
#else
  return std::system("ffmpeg -version >/dev/null 2>&1") == 0;
#endif
}

// ---------------------------------------------------------------------------
// The main panel rendering function
// ---------------------------------------------------------------------------
void RenderAnimationExportPanel(AnimationExportPanelState& state,
                                 const std::vector<std::string>& /*components*/) {
  ImGui::Text("Animation Export");
  ImGui::Separator();
  ImGui::Spacing();

  if (state.frame_paths.empty()) {
    ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f),
                       "No time series data available.");
    ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f),
                       "Solve a time-dependent PDE first.");
    return;
  }

  ImGui::Text("Frames available: %zu", state.frame_paths.size());
  ImGui::Spacing();

  // ---- Format selection ----
  ImGui::Text("Export Format:");
  ImGui::RadioButton("PNG Sequence", &s_format, 1);
  ImGui::SameLine();
  ImGui::RadioButton("MP4", &s_format, 2);
  ImGui::SameLine();
  ImGui::RadioButton("GIF", &s_format, 0);

  ImGui::Spacing();
  ImGui::Separator();
  ImGui::Spacing();

  // ---- Resolution ----
  ImGui::Text("Resolution:");
  ImGui::SetNextItemWidth(80.0f);
  ImGui::InputInt("Width##anim", &s_width);
  ImGui::SameLine();
  ImGui::SetNextItemWidth(80.0f);
  ImGui::InputInt("Height##anim", &s_height);

  // Presets
  if (ImGui::Button("720p")) { s_width = 1280; s_height = 720; }
  ImGui::SameLine();
  if (ImGui::Button("1080p")) { s_width = 1920; s_height = 1080; }
  ImGui::SameLine();
  if (ImGui::Button("4K")) { s_width = 3840; s_height = 2160; }

  // Clamp
  s_width = std::max(100, std::min(7680, s_width));
  s_height = std::max(100, std::min(4320, s_height));

  ImGui::Spacing();

  // ---- FPS ----
  ImGui::SetNextItemWidth(80.0f);
  ImGui::InputInt("FPS", &s_fps);
  s_fps = std::max(1, std::min(120, s_fps));

  // ---- Frame range ----
  int maxFrame = static_cast<int>(state.frame_paths.size()) - 1;
  ImGui::SetNextItemWidth(80.0f);
  ImGui::InputInt("Start Frame", &s_start_frame);
  ImGui::SetNextItemWidth(80.0f);
  ImGui::InputInt("End Frame", &s_end_frame);
  if (s_end_frame < 0) s_end_frame = maxFrame;
  s_start_frame = std::max(0, std::min(maxFrame, s_start_frame));
  s_end_frame = std::max(s_start_frame, std::min(maxFrame, s_end_frame));

  ImGui::Spacing();

  // ---- Options ----
  ImGui::Checkbox("Show timestamp overlay", &s_show_timestamp);

  ImGui::Spacing();
  ImGui::Separator();
  ImGui::Spacing();

  // ====================================================================
  // If we are currently exporting, show progress instead of the button
  // ====================================================================
  if (s_exporting.load()) {
    int cur = s_export_current.load();
    int tot = s_export_total.load();
    float frac = tot > 0 ? static_cast<float>(cur) / tot : 0.0f;

    ImGui::ProgressBar(frac, ImVec2(-1, 0));
    ImGui::Text("Exporting frame %d / %d ...", cur, tot);

    if (ImGui::Button("Cancel", ImVec2(-1, 30))) {
      s_cancel_requested.store(true);
    }

    // Check if the background work finished
    if (s_export_done.load()) {
      s_exporting.store(false);
      if (s_export_success.load()) {
        ImGui::OpenPopup("Animation Export Success");
      } else {
        ImGui::OpenPopup("Animation Export Error");
      }
      s_export_done.store(false);
    }
  } else {
    // ---- Export button ----
    ImGui::BeginDisabled(!state.viewer.has_data());
    if (ImGui::Button("Export Animation", ImVec2(-1, 30))) {
      // Pick output directory
      auto outputDir = FileDialog::PickDirectory(
          "Select Output Directory for Animation",
          std::filesystem::current_path() / "outputs");

      if (outputDir) {
        std::filesystem::create_directories(*outputDir);

        // Snapshot parameters for the export
        const int startFrame = s_start_frame;
        const int endFrame = s_end_frame;
        const int width = s_width;
        const int height = s_height;
        const int fps = s_fps;
        const int format = s_format;
        const bool showTimestamp = s_show_timestamp;
        const int totalFrames = endFrame - startFrame + 1;

        s_export_output_dir = outputDir->string();
        s_export_current.store(0);
        s_export_total.store(totalFrames);
        s_export_done.store(false);
        s_export_success.store(false);
        s_cancel_requested.store(false);
        s_exporting.store(true);
        s_export_error.clear();

        // Capture a reference to the viewer and frame data we need.
        // NOTE: OpenGL calls must happen on the main thread.  We do the
        // heavy lifting (VTK load, PNG write) in stages: because ImGui is
        // immediate-mode we simply process one frame per UI tick while
        // s_exporting is true. This avoids blocking AND keeps GL on the
        // main thread.  We store the captured state in statics so the
        // per-tick function can use it.
        //
        // However, the user asked for a threaded approach. OpenGL contexts
        // are thread-bound, so the pragmatic solution is to do the export
        // synchronously on the main thread but yield after each frame so
        // the progress bar updates. We achieve this by processing one
        // frame per call to RenderAnimationExportPanel while s_exporting
        // is true.

        // We store the export params in file-level statics so the per-tick
        // export logic (below) can access them.
      }
    }
    ImGui::EndDisabled();

    if (!state.viewer.has_data()) {
      ImGui::TextDisabled("Viewer has no data. Load or solve first.");
    }
  }

  // ====================================================================
  // Per-tick export: process one frame each time the panel is rendered
  // while s_exporting is true (keeps GL on the main thread).
  // ====================================================================
  if (s_exporting.load() && !s_export_done.load()) {
    int cur = s_export_current.load();
    int tot = s_export_total.load();

    if (s_cancel_requested.load()) {
      s_export_error = "Export cancelled by user.";
      s_export_success.store(false);
      s_export_done.store(true);
    } else if (cur < tot) {
      int frameIdx = s_start_frame + cur;

      // Bounds check
      if (frameIdx < 0 ||
          frameIdx >= static_cast<int>(state.frame_paths.size())) {
        s_export_error = "Frame index out of range.";
        s_export_success.store(false);
        s_export_done.store(true);
      } else {
        // 1. Load VTK data for this frame
        VtkReadResult vtkResult = ReadVtkFile(state.frame_paths[static_cast<size_t>(frameIdx)]);
        if (!vtkResult.ok) {
          s_export_error = "Failed to read frame " + std::to_string(frameIdx) +
                           ": " + vtkResult.error;
          s_export_success.store(false);
          s_export_done.store(true);
        } else {
          // 2. Push data into the viewer
          state.viewer.SetData(vtkResult.domain, vtkResult.grid);

          // 3. Render & capture
          std::vector<unsigned char> pixels;
          if (!CaptureFrame(state.viewer, s_width, s_height, pixels)) {
            s_export_error = "GL capture failed on frame " + std::to_string(frameIdx);
            s_export_success.store(false);
            s_export_done.store(true);
          } else {
            // 4. Optionally burn timestamp
            if (s_show_timestamp &&
                frameIdx < static_cast<int>(state.frame_times.size())) {
              std::ostringstream ts;
              ts << "t=" << std::fixed << std::setprecision(4)
                 << state.frame_times[static_cast<size_t>(frameIdx)] << "s";
              BurnTimestamp(pixels, s_width, s_height, ts.str());
            }

            // 5. Write PNG
            std::ostringstream fname;
            fname << "frame_" << std::setw(4) << std::setfill('0') << cur << ".png";
            std::filesystem::path outPath =
                std::filesystem::path(s_export_output_dir) / fname.str();

            int ok = stbi_write_png(outPath.string().c_str(),
                                    s_width, s_height, 4,
                                    pixels.data(), s_width * 4);
            if (!ok) {
              s_export_error = "Failed to write " + outPath.string();
              s_export_success.store(false);
              s_export_done.store(true);
            } else {
              s_export_current.store(cur + 1);

              // If that was the last frame, finalise
              if (cur + 1 >= tot) {
                // For MP4 format, shell out to ffmpeg
                if (s_format == 2) {
                  if (!FfmpegAvailable()) {
                    s_export_error =
                        "PNG sequence saved. ffmpeg not found on PATH -- "
                        "cannot produce MP4. Install ffmpeg and re-export, "
                        "or run manually:\n  ffmpeg -framerate " +
                        std::to_string(s_fps) +
                        " -i frame_%04d.png -c:v libx264 -pix_fmt yuv420p output.mp4";
                    s_export_success.store(false);
                    s_export_done.store(true);
                  } else {
                    std::ostringstream cmd;
                    cmd << "cd " << std::quoted(s_export_output_dir)
                        << " && ffmpeg -y -framerate " << s_fps
                        << " -i frame_%04d.png"
                        << " -c:v libx264 -pix_fmt yuv420p"
                        << " animation.mp4"
                        << " >/dev/null 2>&1";
                    int ret = std::system(cmd.str().c_str());
                    if (ret != 0) {
                      s_export_error =
                          "PNG frames saved, but ffmpeg encoding failed (exit code " +
                          std::to_string(ret) + ").";
                      s_export_success.store(false);
                    } else {
                      s_export_success.store(true);
                    }
                    s_export_done.store(true);
                  }
                } else if (s_format == 0) {
                  // GIF -- not yet implemented
                  s_export_error =
                      "PNG sequence saved. GIF encoding is not yet available. "
                      "You can convert the PNG sequence to GIF using external "
                      "tools such as ffmpeg or ImageMagick:\n"
                      "  ffmpeg -framerate " +
                      std::to_string(s_fps) +
                      " -i frame_%04d.png -vf \"palettegen\" palette.png && "
                      "ffmpeg -framerate " +
                      std::to_string(s_fps) +
                      " -i frame_%04d.png -i palette.png -lavfi paletteuse output.gif";
                  // Still mark as success since the PNGs were saved
                  s_export_success.store(true);
                  s_export_done.store(true);
                } else {
                  // PNG Sequence -- already done
                  s_export_success.store(true);
                  s_export_done.store(true);
                }
              }
            }
          }
        }
      }
    }
  }

  // ====================================================================
  // Result popups
  // ====================================================================
  if (ImGui::BeginPopupModal("Animation Export Success", nullptr,
                             ImGuiWindowFlags_AlwaysAutoResize)) {
    int tot = s_export_total.load();
    ImGui::Text("Animation export complete!");
    ImGui::Text("%d frames saved to:", tot);
    ImGui::TextWrapped("%s", s_export_output_dir.c_str());
    if (s_format == 2) {
      ImGui::Spacing();
      ImGui::Text("MP4 written: animation.mp4");
    }
    if (s_format == 0 && !s_export_error.empty()) {
      ImGui::Spacing();
      ImGui::TextColored(ImVec4(1.0f, 0.7f, 0.3f, 1.0f), "Note:");
      ImGui::TextWrapped("%s", s_export_error.c_str());
    }
    ImGui::Spacing();
    if (ImGui::Button("OK", ImVec2(120, 0))) {
      ImGui::CloseCurrentPopup();
    }
    ImGui::EndPopup();
  }

  if (ImGui::BeginPopupModal("Animation Export Error", nullptr,
                             ImGuiWindowFlags_AlwaysAutoResize)) {
    ImGui::TextColored(ImVec4(1.0f, 0.3f, 0.3f, 1.0f), "Export failed!");
    ImGui::TextWrapped("%s", s_export_error.c_str());
    ImGui::Spacing();
    if (ImGui::Button("OK", ImVec2(120, 0))) {
      ImGui::CloseCurrentPopup();
    }
    ImGui::EndPopup();
  }

  // ---- Info notes ----
  ImGui::Spacing();
  if (s_format == 2) {
    ImGui::TextColored(ImVec4(1.0f, 0.7f, 0.3f, 1.0f),
                       "MP4 export requires ffmpeg on PATH.");
  }
  if (s_format == 0) {
    ImGui::TextColored(ImVec4(1.0f, 0.7f, 0.3f, 1.0f),
                       "GIF: PNG frames will be saved. Convert with ffmpeg/ImageMagick.");
  }
}
