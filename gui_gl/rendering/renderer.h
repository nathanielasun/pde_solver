#ifndef RENDERER_H
#define RENDERER_H

#include "GlViewer.h"
#include "grid_builder.h"
#include "render_types.h"
#include <OpenGL/gl3.h>

// Rendering orchestration and FBO management
namespace Renderer {

// FBO (Framebuffer Object) management
struct FboState {
  unsigned int fbo = 0;
  unsigned int color_tex = 0;
  unsigned int depth_rb = 0;
  int width = 0;
  int height = 0;
};

// Ensure FBO is created and sized correctly
void EnsureFbo(FboState* fbo_state, int width, int height);

// Cleanup FBO resources
void DestroyFbo(FboState* fbo_state);

// Render parameters
struct RenderParams {
  const float* clear_color = nullptr;  // 3-element RGB
  unsigned int program = 0;
  unsigned int mesh_vao = 0;
  unsigned int grid_vao = 0;
  int mesh_vertex_count = 0;
  int grid_vertex_count = 0;
  bool grid_enabled = true;
  const GridBuildResult* grid_build_result = nullptr;
  
  // View parameters
  float yaw = 0.0f;
  float pitch = 0.0f;
  float distance = 3.6f;
  float point_scale = 1.0f;
  bool use_ortho = true;
  
  // Scene bounds
  float scene_center_x = 0.0f;
  float scene_center_y = 0.0f;
  float scene_center_z = 0.0f;
  float scene_radius = 1.0f;
  
  // Output
  float* mvp_out = nullptr;  // 16-element matrix output
  float* last_aspect_out = nullptr;  // Output for aspect ratio
};

// Render scene to texture
void RenderToTexture(const FboState& fbo, const RenderParams& params,
                    int width, int height);

}  // namespace Renderer

#endif  // RENDERER_H

