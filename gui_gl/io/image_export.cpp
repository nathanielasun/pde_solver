#include "image_export.h"
#include "rendering/projection.h"
#include <OpenGL/gl3.h>
#include <vector>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <sstream>

// Include stb_image_write for image saving
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../../third_party/glfw/deps/stb_image_write.h"

namespace ImageExport {

// Render text label on image (simple bitmap rendering)
// This is a placeholder - in a full implementation, you'd use a proper font rendering library
// For now, we'll use a simple approach with OpenGL or skip text rendering in the image
void RenderLabelOnImage(std::vector<unsigned char>& image, int width, int height,
                        int x, int y, const std::string& text, bool align_right) {
  // Note: This is a simplified placeholder
  // In a full implementation, you would:
  // 1. Use a font rendering library (like FreeType)
  // 2. Or use ImGui's font rendering capabilities
  // 3. Or use OpenGL text rendering
  
  // For now, we'll add the labels as metadata or use a simpler approach
  // The labels will be included via the axis label overlay
}

bool ExportImage(GlViewer& viewer, const std::filesystem::path& filepath, 
                 const ExportOptions& options) {
  if (!viewer.has_data()) {
    std::cerr << "ImageExport: No data in viewer\n";
    return false;
  }
  
  // Render viewer to texture at specified resolution
  viewer.RenderToTexture(options.width, options.height);
  
  // Read pixels from OpenGL framebuffer
  std::vector<unsigned char> pixels(options.width * options.height * 4);
  
  // Bind the framebuffer texture
  GLuint texture = viewer.texture();
  if (texture == 0) {
    std::cerr << "ImageExport: Invalid texture\n";
    return false;
  }
  
  // Create a temporary FBO to read from the texture
  GLuint read_fbo = 0;
  glGenFramebuffers(1, &read_fbo);
  glBindFramebuffer(GL_READ_FRAMEBUFFER, read_fbo);
  glFramebufferTexture2D(GL_READ_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, 
                         GL_TEXTURE_2D, texture, 0);
  
  if (glCheckFramebufferStatus(GL_READ_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
    std::cerr << "ImageExport: Framebuffer not complete\n";
    glBindFramebuffer(GL_READ_FRAMEBUFFER, 0);
    glDeleteFramebuffers(1, &read_fbo);
    return false;
  }
  
  // Read pixels (flip vertically because OpenGL origin is bottom-left)
  glReadPixels(0, 0, options.width, options.height, GL_RGBA, GL_UNSIGNED_BYTE, 
               pixels.data());
  
  // Flip image vertically (OpenGL has origin at bottom-left, images at top-left)
  std::vector<unsigned char> flipped_pixels(options.width * options.height * 4);
  for (int y = 0; y < options.height; ++y) {
    int src_y = options.height - 1 - y;
    std::copy(pixels.begin() + (src_y * options.width * 4),
              pixels.begin() + ((src_y + 1) * options.width * 4),
              flipped_pixels.begin() + (y * options.width * 4));
  }
  
  // Add axis labels if requested
  if (options.include_axis_labels) {
    // Get axis labels from viewer
    std::vector<GlViewer::ScreenLabel> labels = viewer.AxisLabels();
    
    // For now, we'll render labels as simple colored rectangles
    // In a full implementation, you'd render actual text
    // This is a placeholder that shows where labels would go
    for (const auto& label : labels) {
      int x = static_cast<int>(label.x);
      int y = static_cast<int>(label.y);
      
      // Clamp to image bounds
      x = std::max(0, std::min(options.width - 1, x));
      y = std::max(0, std::min(options.height - 1, y));
      
      // Draw a small marker at label position (white pixel)
      // In a real implementation, you'd render the text here
      if (x >= 0 && x < options.width && y >= 0 && y < options.height) {
        int idx = (y * options.width + x) * 4;
        if (idx + 3 < static_cast<int>(flipped_pixels.size())) {
          // Mark label position with a white pixel (placeholder)
          flipped_pixels[idx] = 255;     // R
          flipped_pixels[idx + 1] = 255;  // G
          flipped_pixels[idx + 2] = 255;  // B
          flipped_pixels[idx + 3] = 255;  // A
        }
      }
    }
  }
  
  // Clean up
  glBindFramebuffer(GL_READ_FRAMEBUFFER, 0);
  glDeleteFramebuffers(1, &read_fbo);
  
  // Determine file format from extension
  std::string ext = filepath.extension().string();
  std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
  
  bool success = false;
  
  if (ext == ".png" || ext == ".PNG") {
    // Save as PNG
    success = stbi_write_png(filepath.string().c_str(), 
                              options.width, options.height, 4,
                              flipped_pixels.data(), 
                              options.width * 4) != 0;
  } else if (ext == ".jpg" || ext == ".jpeg" || ext == ".JPG" || ext == ".JPEG") {
    // Save as JPEG
    success = stbi_write_jpg(filepath.string().c_str(), 
                              options.width, options.height, 4,
                              flipped_pixels.data(), 
                              options.jpeg_quality) != 0;
  } else {
    std::cerr << "ImageExport: Unsupported file format: " << ext << "\n";
    std::cerr << "ImageExport: Supported formats: .png, .jpg, .jpeg\n";
    return false;
  }
  
  if (!success) {
    std::cerr << "ImageExport: Failed to write image file\n";
    return false;
  }
  
  return true;
}

std::string GetSuggestedFilename(const GlViewer& viewer, const std::string& extension) {
  // Generate a filename based on current state
  std::string filename = "pde_export";
  
  // Add field type
  switch (viewer.GetFieldType()) {
    case GlViewer::FieldType::Solution:
      filename += "_solution";
      break;
    case GlViewer::FieldType::GradientX:
      filename += "_gradient_x";
      break;
    case GlViewer::FieldType::GradientY:
      filename += "_gradient_y";
      break;
    case GlViewer::FieldType::GradientZ:
      filename += "_gradient_z";
      break;
    case GlViewer::FieldType::Laplacian:
      filename += "_laplacian";
      break;
    case GlViewer::FieldType::FluxX:
      filename += "_flux_x";
      break;
    case GlViewer::FieldType::FluxY:
      filename += "_flux_y";
      break;
    case GlViewer::FieldType::FluxZ:
      filename += "_flux_z";
      break;
    case GlViewer::FieldType::EnergyNorm:
      filename += "_energy_norm";
      break;
  }
  
  // Add view mode
  switch (viewer.view_mode()) {
    case GlViewer::ViewMode::Polar:
      filename += "_polar";
      break;
    case GlViewer::ViewMode::Axisymmetric:
      filename += "_axisymmetric";
      break;
    case GlViewer::ViewMode::CylindricalVolume:
      filename += "_cylindrical";
      break;
    case GlViewer::ViewMode::SphericalSurface:
      filename += "_spherical_surface";
      break;
    case GlViewer::ViewMode::SphericalVolume:
      filename += "_spherical_volume";
      break;
    case GlViewer::ViewMode::ToroidalSurface:
      filename += "_toroidal_surface";
      break;
    case GlViewer::ViewMode::ToroidalVolume:
      filename += "_toroidal_volume";
      break;
    default:
      break;
  }
  
  filename += "." + extension;
  return filename;
}

// Helper function to format value for display
static std::string FormatValue(double value) {
  std::ostringstream oss;
  if (std::abs(value) < 1e-3 || std::abs(value) > 1e6) {
    oss << std::scientific << std::setprecision(6) << value;
  } else {
    oss << std::fixed << std::setprecision(6) << value;
  }
  return oss.str();
}

// Helper function to get field type name
static std::string GetFieldTypeName(GlViewer::FieldType field_type) {
  switch (field_type) {
    case GlViewer::FieldType::Solution: return "Solution (u)";
    case GlViewer::FieldType::GradientX: return "Gradient X (∂u/∂x)";
    case GlViewer::FieldType::GradientY: return "Gradient Y (∂u/∂y)";
    case GlViewer::FieldType::GradientZ: return "Gradient Z (∂u/∂z)";
    case GlViewer::FieldType::Laplacian: return "Laplacian (∇²u)";
    case GlViewer::FieldType::FluxX: return "Flux X";
    case GlViewer::FieldType::FluxY: return "Flux Y";
    case GlViewer::FieldType::FluxZ: return "Flux Z";
    case GlViewer::FieldType::EnergyNorm: return "Energy Norm (u²)";
    default: return "Unknown";
  }
}

// Draw a simple line on the image
static void DrawLine(std::vector<unsigned char>& image, int width, int height,
                     int x0, int y0, int x1, int y1,
                     unsigned char r, unsigned char g, unsigned char b) {
  // Bresenham's line algorithm
  int dx = std::abs(x1 - x0);
  int dy = std::abs(y1 - y0);
  int sx = (x0 < x1) ? 1 : -1;
  int sy = (y0 < y1) ? 1 : -1;
  int err = dx - dy;
  
  int x = x0, y = y0;
  while (true) {
    if (x >= 0 && x < width && y >= 0 && y < height) {
      int idx = (y * width + x) * 4;
      if (idx + 3 < static_cast<int>(image.size())) {
        image[idx] = r;
        image[idx + 1] = g;
        image[idx + 2] = b;
        image[idx + 3] = 255;
      }
    }
    
    if (x == x1 && y == y1) break;
    
    int e2 = 2 * err;
    if (e2 > -dy) {
      err -= dy;
      x += sx;
    }
    if (e2 < dx) {
      err += dx;
      y += sy;
    }
  }
}

// Simple 8x8 bitmap font data (ASCII characters 32-126)
// Each character is 8 bytes, each byte is a row of 8 bits
static const unsigned char font_8x8[95][8] = {
  {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00}, // space (32)
  {0x18, 0x3C, 0x3C, 0x18, 0x18, 0x00, 0x18, 0x00}, // !
  {0x36, 0x36, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00}, // "
  {0x36, 0x36, 0x7F, 0x36, 0x7F, 0x36, 0x36, 0x00}, // #
  {0x0C, 0x3E, 0x03, 0x1E, 0x30, 0x1F, 0x0C, 0x00}, // $
  {0x00, 0x63, 0x33, 0x18, 0x0C, 0x66, 0x63, 0x00}, // %
  {0x1C, 0x36, 0x1C, 0x6E, 0x3B, 0x33, 0x6E, 0x00}, // &
  {0x06, 0x06, 0x03, 0x00, 0x00, 0x00, 0x00, 0x00}, // '
  {0x18, 0x0C, 0x06, 0x06, 0x06, 0x0C, 0x18, 0x00}, // (
  {0x06, 0x0C, 0x18, 0x18, 0x18, 0x0C, 0x06, 0x00}, // )
  {0x00, 0x66, 0x3C, 0xFF, 0x3C, 0x66, 0x00, 0x00}, // *
  {0x00, 0x0C, 0x0C, 0x3F, 0x0C, 0x0C, 0x00, 0x00}, // +
  {0x00, 0x00, 0x00, 0x00, 0x00, 0x0C, 0x06, 0x00}, // ,
  {0x00, 0x00, 0x00, 0x3F, 0x00, 0x00, 0x00, 0x00}, // -
  {0x00, 0x00, 0x00, 0x00, 0x00, 0x0C, 0x0C, 0x00}, // .
  {0x60, 0x30, 0x18, 0x0C, 0x06, 0x03, 0x01, 0x00}, // /
  {0x3E, 0x63, 0x73, 0x7B, 0x6F, 0x67, 0x3E, 0x00}, // 0
  {0x0C, 0x0E, 0x0C, 0x0C, 0x0C, 0x0C, 0x3F, 0x00}, // 1
  {0x1E, 0x33, 0x30, 0x1C, 0x06, 0x33, 0x3F, 0x00}, // 2
  {0x1E, 0x33, 0x30, 0x1C, 0x30, 0x33, 0x1E, 0x00}, // 3
  {0x38, 0x3C, 0x36, 0x33, 0x7F, 0x30, 0x78, 0x00}, // 4
  {0x3F, 0x03, 0x1F, 0x30, 0x30, 0x33, 0x1E, 0x00}, // 5
  {0x1C, 0x06, 0x03, 0x1F, 0x33, 0x33, 0x1E, 0x00}, // 6
  {0x3F, 0x33, 0x30, 0x18, 0x0C, 0x0C, 0x0C, 0x00}, // 7
  {0x1E, 0x33, 0x33, 0x1E, 0x33, 0x33, 0x1E, 0x00}, // 8
  {0x1E, 0x33, 0x33, 0x3E, 0x30, 0x18, 0x0E, 0x00}, // 9
  {0x00, 0x0C, 0x0C, 0x00, 0x00, 0x0C, 0x0C, 0x00}, // :
  {0x00, 0x0C, 0x0C, 0x00, 0x00, 0x0C, 0x06, 0x00}, // ;
  {0x18, 0x0C, 0x06, 0x03, 0x06, 0x0C, 0x18, 0x00}, // <
  {0x00, 0x00, 0x3F, 0x00, 0x00, 0x3F, 0x00, 0x00}, // =
  {0x06, 0x0C, 0x18, 0x30, 0x18, 0x0C, 0x06, 0x00}, // >
  {0x1E, 0x33, 0x30, 0x18, 0x0C, 0x00, 0x0C, 0x00}, // ?
  {0x3E, 0x63, 0x7B, 0x7B, 0x7B, 0x03, 0x1E, 0x00}, // @
  {0x0C, 0x1E, 0x33, 0x33, 0x3F, 0x33, 0x33, 0x00}, // A
  {0x3F, 0x66, 0x66, 0x3E, 0x66, 0x66, 0x3F, 0x00}, // B
  {0x3C, 0x66, 0x03, 0x03, 0x03, 0x66, 0x3C, 0x00}, // C
  {0x1F, 0x36, 0x66, 0x66, 0x66, 0x36, 0x1F, 0x00}, // D
  {0x7F, 0x06, 0x06, 0x3E, 0x06, 0x06, 0x7F, 0x00}, // E
  {0x7F, 0x06, 0x06, 0x3E, 0x06, 0x06, 0x06, 0x00}, // F
  {0x3C, 0x66, 0x03, 0x03, 0x73, 0x66, 0x7C, 0x00}, // G
  {0x33, 0x33, 0x33, 0x3F, 0x33, 0x33, 0x33, 0x00}, // H
  {0x1E, 0x0C, 0x0C, 0x0C, 0x0C, 0x0C, 0x1E, 0x00}, // I
  {0x78, 0x30, 0x30, 0x30, 0x33, 0x33, 0x1E, 0x00}, // J
  {0x67, 0x66, 0x36, 0x1E, 0x36, 0x66, 0x67, 0x00}, // K
  {0x06, 0x06, 0x06, 0x06, 0x06, 0x06, 0x7F, 0x00}, // L
  {0x63, 0x77, 0x7F, 0x6B, 0x63, 0x63, 0x63, 0x00}, // M
  {0x63, 0x67, 0x6F, 0x7B, 0x73, 0x63, 0x63, 0x00}, // N
  {0x1C, 0x36, 0x63, 0x63, 0x63, 0x36, 0x1C, 0x00}, // O
  {0x3F, 0x66, 0x66, 0x3E, 0x06, 0x06, 0x06, 0x00}, // P
  {0x1E, 0x33, 0x33, 0x33, 0x3B, 0x1E, 0x38, 0x00}, // Q
  {0x3F, 0x66, 0x66, 0x3E, 0x36, 0x66, 0x67, 0x00}, // R
  {0x1E, 0x33, 0x07, 0x0E, 0x38, 0x33, 0x1E, 0x00}, // S
  {0x3F, 0x2D, 0x0C, 0x0C, 0x0C, 0x0C, 0x1E, 0x00}, // T
  {0x33, 0x33, 0x33, 0x33, 0x33, 0x33, 0x3F, 0x00}, // U
  {0x33, 0x33, 0x33, 0x33, 0x33, 0x1E, 0x0C, 0x00}, // V
  {0x63, 0x63, 0x63, 0x6B, 0x7F, 0x77, 0x63, 0x00}, // W
  {0x63, 0x63, 0x36, 0x1C, 0x1C, 0x36, 0x63, 0x00}, // X
  {0x33, 0x33, 0x33, 0x1E, 0x0C, 0x0C, 0x1E, 0x00}, // Y
  {0x7F, 0x63, 0x31, 0x18, 0x4C, 0x66, 0x7F, 0x00}, // Z
  {0x1E, 0x06, 0x06, 0x06, 0x06, 0x06, 0x1E, 0x00}, // [
  {0x03, 0x06, 0x0C, 0x18, 0x30, 0x60, 0x40, 0x00}, // backslash
  {0x1E, 0x18, 0x18, 0x18, 0x18, 0x18, 0x1E, 0x00}, // ]
  {0x08, 0x1C, 0x36, 0x63, 0x00, 0x00, 0x00, 0x00}, // ^
  {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xFF}, // _
  {0x0C, 0x0C, 0x18, 0x00, 0x00, 0x00, 0x00, 0x00}, // `
  {0x00, 0x00, 0x1E, 0x30, 0x3E, 0x33, 0x6E, 0x00}, // a
  {0x07, 0x06, 0x06, 0x3E, 0x66, 0x66, 0x3B, 0x00}, // b
  {0x00, 0x00, 0x1E, 0x33, 0x03, 0x33, 0x1E, 0x00}, // c
  {0x38, 0x30, 0x30, 0x3E, 0x33, 0x33, 0x6E, 0x00}, // d
  {0x00, 0x00, 0x1E, 0x33, 0x3F, 0x03, 0x1E, 0x00}, // e
  {0x1C, 0x36, 0x06, 0x0F, 0x06, 0x06, 0x0F, 0x00}, // f
  {0x00, 0x00, 0x6E, 0x33, 0x33, 0x3E, 0x30, 0x1F}, // g
  {0x07, 0x06, 0x36, 0x6E, 0x66, 0x66, 0x67, 0x00}, // h
  {0x0C, 0x00, 0x0E, 0x0C, 0x0C, 0x0C, 0x1E, 0x00}, // i
  {0x30, 0x00, 0x30, 0x30, 0x30, 0x33, 0x33, 0x1E}, // j
  {0x07, 0x06, 0x66, 0x36, 0x1E, 0x36, 0x67, 0x00}, // k
  {0x0E, 0x0C, 0x0C, 0x0C, 0x0C, 0x0C, 0x1E, 0x00}, // l
  {0x00, 0x00, 0x33, 0x7F, 0x7F, 0x6B, 0x63, 0x00}, // m
  {0x00, 0x00, 0x1F, 0x33, 0x33, 0x33, 0x33, 0x00}, // n
  {0x00, 0x00, 0x1E, 0x33, 0x33, 0x33, 0x1E, 0x00}, // o
  {0x00, 0x00, 0x3B, 0x66, 0x66, 0x3E, 0x06, 0x0F}, // p
  {0x00, 0x00, 0x6E, 0x33, 0x33, 0x3E, 0x30, 0x78}, // q
  {0x00, 0x00, 0x3B, 0x6E, 0x66, 0x06, 0x0F, 0x00}, // r
  {0x00, 0x00, 0x3E, 0x03, 0x1E, 0x30, 0x1F, 0x00}, // s
  {0x08, 0x0C, 0x3E, 0x0C, 0x0C, 0x2C, 0x18, 0x00}, // t
  {0x00, 0x00, 0x33, 0x33, 0x33, 0x33, 0x6E, 0x00}, // u
  {0x00, 0x00, 0x33, 0x33, 0x33, 0x1E, 0x0C, 0x00}, // v
  {0x00, 0x00, 0x63, 0x6B, 0x7F, 0x7F, 0x36, 0x00}, // w
  {0x00, 0x00, 0x63, 0x36, 0x1C, 0x36, 0x63, 0x00}, // x
  {0x00, 0x00, 0x33, 0x33, 0x33, 0x3E, 0x30, 0x1F}, // y
  {0x00, 0x00, 0x3F, 0x19, 0x0C, 0x26, 0x3F, 0x00}, // z
  {0x38, 0x0C, 0x0C, 0x07, 0x0C, 0x0C, 0x38, 0x00}, // {
  {0x18, 0x18, 0x18, 0x00, 0x18, 0x18, 0x18, 0x00}, // |
  {0x07, 0x0C, 0x0C, 0x38, 0x0C, 0x0C, 0x07, 0x00}, // }
  {0x6E, 0x3B, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00}, // ~
};

// Render a single character using the bitmap font
static void DrawChar(std::vector<unsigned char>& image, int width, int height,
                     int x, int y, char c, unsigned char r, unsigned char g, unsigned char b,
                     int scale = 1) {
  if (c < 32 || c > 126) {
    return;  // Only render printable ASCII
  }
  
  const unsigned char* font_data = font_8x8[c - 32];
  
  for (int row = 0; row < 8; ++row) {
    unsigned char row_data = font_data[row];
    for (int col = 0; col < 8; ++col) {
      // If characters appear mirrored, the font data likely has LSB (bit 0) on the left
      // So we check bit col (0=left, 7=right) and draw at position col (left to right)
      // This fixes the mirroring issue
      if (row_data & (1 << col)) {
        // Draw pixel at (x + col*scale, y + row*scale) with scaling
        for (int sy = 0; sy < scale; ++sy) {
          for (int sx = 0; sx < scale; ++sx) {
            int px = x + col * scale + sx;
            int py = y + row * scale + sy;
            if (px >= 0 && px < width && py >= 0 && py < height) {
              int idx = (py * width + px) * 4;
              if (idx + 3 < static_cast<int>(image.size())) {
                image[idx] = r;
                image[idx + 1] = g;
                image[idx + 2] = b;
                image[idx + 3] = 255;
              }
            }
          }
        }
      }
    }
  }
}

// Render a string of text
static void DrawText(std::vector<unsigned char>& image, int width, int height,
                     int x, int y, const std::string& text, 
                     unsigned char r, unsigned char g, unsigned char b,
                     int scale = 1) {
  int current_x = x;
  for (char c : text) {
    DrawChar(image, width, height, current_x, y, c, r, g, b, scale);
    current_x += 8 * scale + 1;  // Character width + spacing
    if (current_x >= width - 8 * scale) {
      break;  // Don't overflow
    }
  }
}

bool ExportLinePlotImage(const LinePlot& plot, const Domain& domain,
                         GlViewer::FieldType field_type,
                         const std::filesystem::path& filepath,
                         const LinePlotExportOptions& options) {
  if (plot.positions.empty() || plot.values.empty()) {
    std::cerr << "ImageExport: Line plot has no data\n";
    return false;
  }
  
  // Calculate plot area (excluding metadata)
  const int plot_height = options.height - (options.include_metadata ? options.metadata_height : 0);
  const int plot_width = options.width;
  const int margin_left = 80;   // Space for Y-axis labels
  const int margin_bottom = 40;  // Space for X-axis labels
  const int margin_right = 20;
  const int margin_top = options.include_metadata ? options.metadata_height : 20;
  
  const int plot_area_x = margin_left;
  const int plot_area_y = margin_top;
  const int plot_area_w = plot_width - margin_left - margin_right;
  const int plot_area_h = plot_height - margin_top - margin_bottom;
  
  // Create image buffer (white background)
  std::vector<unsigned char> image(options.width * options.height * 4, 255);
  
  // Find value range
  double min_val = *std::min_element(plot.values.begin(), plot.values.end());
  double max_val = *std::max_element(plot.values.begin(), plot.values.end());
  double val_range = max_val - min_val;
  if (val_range < 1e-12) {
    val_range = 1.0;
    min_val -= 0.5;
    max_val += 0.5;
  }
  
  // Find position range
  double min_pos = plot.positions.empty() ? 0.0 : plot.positions[0];
  double max_pos = plot.positions.empty() ? 1.0 : plot.positions.back();
  double pos_range = max_pos - min_pos;
  if (pos_range < 1e-12) {
    pos_range = 1.0;
  }
  
  // Draw grid if requested
  if (options.include_grid) {
    // Horizontal grid lines (value axis)
    const int num_h_lines = 10;
    for (int i = 0; i <= num_h_lines; ++i) {
      double val = min_val + (max_val - min_val) * i / num_h_lines;
      int y = plot_area_y + plot_area_h - static_cast<int>((val - min_val) / val_range * plot_area_h);
      DrawLine(image, options.width, options.height,
               plot_area_x, y, plot_area_x + plot_area_w, y,
               200, 200, 200);  // Light gray
    }
    
    // Vertical grid lines (position axis)
    const int num_v_lines = 10;
    for (int i = 0; i <= num_v_lines; ++i) {
      double pos = min_pos + (max_pos - min_pos) * i / num_v_lines;
      int x = plot_area_x + static_cast<int>((pos - min_pos) / pos_range * plot_area_w);
      DrawLine(image, options.width, options.height,
               x, plot_area_y, x, plot_area_y + plot_area_h,
               200, 200, 200);  // Light gray
    }
  }
  
  // Draw axes
  DrawLine(image, options.width, options.height,
           plot_area_x, plot_area_y, plot_area_x, plot_area_y + plot_area_h,
           0, 0, 0);  // Black Y-axis
  DrawLine(image, options.width, options.height,
           plot_area_x, plot_area_y + plot_area_h, 
           plot_area_x + plot_area_w, plot_area_y + plot_area_h,
           0, 0, 0);  // Black X-axis
  
  // Draw plot line
  if (plot.positions.size() > 1) {
    for (size_t i = 0; i < plot.positions.size() - 1; ++i) {
      int x0 = plot_area_x + static_cast<int>((plot.positions[i] - min_pos) / pos_range * plot_area_w);
      int y0 = plot_area_y + plot_area_h - static_cast<int>((plot.values[i] - min_val) / val_range * plot_area_h);
      int x1 = plot_area_x + static_cast<int>((plot.positions[i + 1] - min_pos) / pos_range * plot_area_w);
      int y1 = plot_area_y + plot_area_h - static_cast<int>((plot.values[i + 1] - min_val) / val_range * plot_area_h);
      
      DrawLine(image, options.width, options.height, x0, y0, x1, y1, 0, 0, 255);  // Blue line
    }
  }
  
  // Draw axis labels with text
  if (options.include_axis_labels) {
    const int font_scale = 1;  // Can increase for larger text
    
    // Y-axis labels (value)
    const int num_y_labels = 5;
    for (int i = 0; i <= num_y_labels; ++i) {
      double val = min_val + (max_val - min_val) * i / num_y_labels;
      int y = plot_area_y + plot_area_h - static_cast<int>((val - min_val) / val_range * plot_area_h);
      
      // Format value string
      std::string label_str = FormatValue(val);
      
      // Render text, right-aligned to the left of the axis
      int text_width = static_cast<int>(label_str.length()) * (8 * font_scale + 1);
      int text_x = plot_area_x - text_width - 5;
      int text_y = y - 4 * font_scale;  // Center vertically on tick
      
      if (text_x >= 0 && text_y >= 0 && text_y + 8 * font_scale < options.height) {
        DrawText(image, options.width, options.height, text_x, text_y, 
                 label_str, 0, 0, 0, font_scale);
      }
      
      // Draw tick mark
      DrawLine(image, options.width, options.height,
               plot_area_x - 5, y, plot_area_x, y, 0, 0, 0);
    }
    
    // X-axis labels (position)
    const int num_x_labels = 5;
    for (int i = 0; i <= num_x_labels; ++i) {
      double pos = min_pos + (max_pos - min_pos) * i / num_x_labels;
      int x = plot_area_x + static_cast<int>((pos - min_pos) / pos_range * plot_area_w);
      
      // Format position string
      std::string label_str = FormatValue(pos);
      
      // Render text, centered on tick
      int text_width = static_cast<int>(label_str.length()) * (8 * font_scale + 1);
      int text_x = x - text_width / 2;
      int text_y = plot_area_y + plot_area_h + 5;
      
      if (text_x >= 0 && text_x + text_width < options.width && 
          text_y >= 0 && text_y + 8 * font_scale < options.height) {
        DrawText(image, options.width, options.height, text_x, text_y, 
                 label_str, 0, 0, 0, font_scale);
      }
      
      // Draw tick mark
      DrawLine(image, options.width, options.height,
               x, plot_area_y + plot_area_h, x, plot_area_y + plot_area_h + 5, 0, 0, 0);
    }
    
    // Draw axis titles
    // Y-axis title (rotated would be ideal, but for simplicity, place it to the left)
    std::string y_label = "Value";
    int y_title_x = 10;
    int y_title_y = plot_area_y + plot_area_h / 2 - 4 * font_scale;
    if (y_title_x >= 0 && y_title_y >= 0) {
      DrawText(image, options.width, options.height, y_title_x, y_title_y,
               y_label, 0, 0, 0, font_scale);
    }
    
    // X-axis title
    std::string x_label = "Position";
    int x_title_x = plot_area_x + plot_area_w / 2 - static_cast<int>(x_label.length()) * (8 * font_scale + 1) / 2;
    int x_title_y = plot_area_y + plot_area_h + 25;
    if (x_title_x >= 0 && x_title_y >= 0 && x_title_y + 8 * font_scale < options.height) {
      DrawText(image, options.width, options.height, x_title_x, x_title_y,
               x_label, 0, 0, 0, font_scale);
    }
  }
  
  // Draw metadata text at top
  if (options.include_metadata) {
    // Draw metadata background (light gray)
    for (int y = 0; y < options.metadata_height; ++y) {
      for (int x = 0; x < options.width; ++x) {
        int idx = (y * options.width + x) * 4;
        if (idx + 3 < static_cast<int>(image.size())) {
          image[idx] = 240;
          image[idx + 1] = 240;
          image[idx + 2] = 240;
          image[idx + 3] = 255;
        }
      }
    }
    
    // Draw metadata separator line
    DrawLine(image, options.width, options.height,
             0, options.metadata_height - 1, options.width, options.metadata_height - 1,
             180, 180, 180);
    
    // Prepare metadata text
    std::ostringstream metadata;
    metadata << "Field: " << GetFieldTypeName(field_type) << "  |  ";
    metadata << "Start: (" << FormatValue(plot.x0) << ", " 
             << FormatValue(plot.y0) << ", " << FormatValue(plot.z0) << ")  |  ";
    metadata << "End: (" << FormatValue(plot.x1) << ", " 
             << FormatValue(plot.y1) << ", " << FormatValue(plot.z1) << ")  |  ";
    metadata << "Value Range: [" << FormatValue(min_val) << ", " << FormatValue(max_val) << "]  |  ";
    metadata << "Points: " << plot.num_points << "  |  ";
    metadata << "Length: " << FormatValue(max_pos - min_pos);
    
    std::string metadata_str = metadata.str();
    
    // Render metadata text using bitmap font
    const int font_scale = 1;
    const int text_x = 10;
    const int line_spacing = 12;
    
    // Split metadata into multiple lines
    const int max_line_width = options.width - 20;
    const int char_width = 8 * font_scale + 1;
    const int max_chars_per_line = max_line_width / char_width;
    
    std::vector<std::string> lines;
    
    // Split by "  |  " separator first
    std::vector<std::string> segments;
    size_t pos = 0;
    while (pos < metadata_str.length()) {
      size_t next_sep = metadata_str.find("  |  ", pos);
      if (next_sep == std::string::npos) {
        segments.push_back(metadata_str.substr(pos));
        break;
      }
      segments.push_back(metadata_str.substr(pos, next_sep - pos));
      pos = next_sep + 5;  // Skip "  |  "
    }
    
    // Build lines, wrapping as needed
    std::string current_line;
    for (const auto& segment : segments) {
      std::string to_add = current_line.empty() ? segment : "  |  " + segment;
      
      if (current_line.length() + to_add.length() - current_line.length() < static_cast<size_t>(max_chars_per_line)) {
        // Fits on current line
        current_line = current_line.empty() ? segment : current_line + "  |  " + segment;
      } else {
        // Doesn't fit, start new line
        if (!current_line.empty()) {
          lines.push_back(current_line);
        }
        current_line = segment;
      }
    }
    if (!current_line.empty()) {
      lines.push_back(current_line);
    }
    
    // Render each line
    int line_y = 10;
    for (const auto& line_text : lines) {
      if (line_y + 8 * font_scale < options.metadata_height - 5) {
        DrawText(image, options.width, options.height, text_x, line_y,
                 line_text, 0, 0, 0, font_scale);
        line_y += line_spacing;
      } else {
        break;  // No more room
      }
    }
  }
  
  // Determine file format from extension
  std::string ext = filepath.extension().string();
  std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
  
  bool success = false;
  
  if (ext == ".png" || ext == ".PNG") {
    success = stbi_write_png(filepath.string().c_str(), 
                              options.width, options.height, 4,
                              image.data(), 
                              options.width * 4) != 0;
  } else if (ext == ".jpg" || ext == ".jpeg" || ext == ".JPG" || ext == ".JPEG") {
    success = stbi_write_jpg(filepath.string().c_str(), 
                              options.width, options.height, 4,
                              image.data(), 
                              options.jpeg_quality) != 0;
  } else {
    std::cerr << "ImageExport: Unsupported file format: " << ext << "\n";
    return false;
  }
  
  if (!success) {
    std::cerr << "ImageExport: Failed to write line plot image file\n";
    return false;
  }
  
  return true;
}

std::string GetLinePlotFilename(const LinePlot& plot, const std::string& extension) {
  std::ostringstream filename;
  filename << "lineplot_"
           << std::fixed << std::setprecision(2) 
           << plot.x0 << "_" << plot.y0 << "_" << plot.z0
           << "_to_"
           << plot.x1 << "_" << plot.y1 << "_" << plot.z1
           << "." << extension;
  return filename.str();
}

}  // namespace ImageExport

