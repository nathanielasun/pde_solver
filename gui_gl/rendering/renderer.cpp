#include "renderer.h"
#include "render_types.h"
#include <algorithm>
#include <cmath>

namespace Renderer {

void EnsureFbo(FboState* fbo_state, int width, int height) {
  if (!fbo_state || width <= 0 || height <= 0) {
    return;
  }
  
  if (fbo_state->width == width && fbo_state->height == height && fbo_state->fbo != 0) {
    return;
  }
  
  fbo_state->width = width;
  fbo_state->height = height;
  
  if (!fbo_state->fbo) {
    glGenFramebuffers(1, &fbo_state->fbo);
  }
  glBindFramebuffer(GL_FRAMEBUFFER, fbo_state->fbo);
  
  if (!fbo_state->color_tex) {
    glGenTextures(1, &fbo_state->color_tex);
  }
  glBindTexture(GL_TEXTURE_2D, fbo_state->color_tex);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE,
               nullptr);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, 
                        fbo_state->color_tex, 0);
  
  if (!fbo_state->depth_rb) {
    glGenRenderbuffers(1, &fbo_state->depth_rb);
  }
  glBindRenderbuffer(GL_RENDERBUFFER, fbo_state->depth_rb);
  glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, width, height);
  glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER,
                            fbo_state->depth_rb);
  
  if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
    // Error handling - framebuffer incomplete
  }
  glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void DestroyFbo(FboState* fbo_state) {
  if (!fbo_state) {
    return;
  }
  
  if (fbo_state->depth_rb) {
    glDeleteRenderbuffers(1, &fbo_state->depth_rb);
    fbo_state->depth_rb = 0;
  }
  if (fbo_state->color_tex) {
    glDeleteTextures(1, &fbo_state->color_tex);
    fbo_state->color_tex = 0;
  }
  if (fbo_state->fbo) {
    glDeleteFramebuffers(1, &fbo_state->fbo);
    fbo_state->fbo = 0;
  }
  fbo_state->width = 0;
  fbo_state->height = 0;
}

void RenderToTexture(const FboState& fbo, const RenderParams& params,
                    int width, int height) {
  EnsureFbo(const_cast<FboState*>(&fbo), width, height);
  
  glBindFramebuffer(GL_FRAMEBUFFER, fbo.fbo);
  glViewport(0, 0, width, height);
  glEnable(GL_DEPTH_TEST);
  glDepthFunc(GL_LEQUAL);
  glEnable(GL_PROGRAM_POINT_SIZE);
  glDisable(GL_CULL_FACE);
  
  if (params.clear_color) {
    glClearColor(params.clear_color[0], params.clear_color[1], params.clear_color[2], 1.0f);
  } else {
    glClearColor(0.08f, 0.09f, 0.11f, 1.0f);
  }
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  
  if ((params.mesh_vertex_count > 0) || (params.grid_vertex_count > 0 && params.grid_enabled)) {
    glUseProgram(params.program);
    
    const float aspect = width > 0 && height > 0 ? static_cast<float>(width) / height : 1.0f;
    if (params.last_aspect_out) {
      *params.last_aspect_out = aspect;
    }
    
    const float view_distance = params.use_ortho ? params.distance : params.distance * params.point_scale;
    const float radius = std::max(0.05f, params.scene_radius);
    float near_plane = view_distance - radius * 4.0f;
    if (near_plane < 0.001f) {
      near_plane = 0.001f;
    }
    float far_plane = view_distance + radius * 4.0f;
    if (far_plane <= near_plane + 0.01f) {
      far_plane = near_plane + 0.01f;
    }
    
    Mat4 proj{};
    if (params.use_ortho) {
      float half = std::max(0.01f, radius) * std::max(0.1f, params.point_scale) * 1.2f;
      float half_x = half;
      float half_y = half;
      if (aspect >= 1.0f) {
        half_x = half * aspect;
      } else {
        half_y = half / aspect;
      }
      proj = RenderMath::Orthographic(-half_x, half_x, -half_y, half_y, near_plane, far_plane);
    } else {
      proj = RenderMath::Perspective(RenderMath::kPi / 3.5f, aspect, near_plane, far_plane);
    }
    
    Mat4 view = RenderMath::Translate(0.0f, 0.0f, -view_distance);
    Mat4 model = RenderMath::Multiply(RenderMath::RotateY(params.yaw), 
                                      RenderMath::RotateX(params.pitch));
    model = RenderMath::Multiply(model,
                                 RenderMath::Translate(-params.scene_center_x,
                                                       -params.scene_center_y,
                                                       -params.scene_center_z));
    Mat4 mvp = RenderMath::Multiply(RenderMath::Multiply(proj, view), model);
    
    if (params.mvp_out) {
      std::copy(std::begin(mvp.m), std::end(mvp.m), params.mvp_out);
    }
    
    const GLint loc = glGetUniformLocation(params.program, "uMVP");
    glUniformMatrix4fv(loc, 1, GL_FALSE, mvp.m);
    
    if (params.grid_enabled && params.grid_vertex_count > 0) {
      glBindVertexArray(params.grid_vao);
      glDrawArrays(GL_LINES, 0, params.grid_vertex_count);
      glBindVertexArray(0);
    }
    
    if (params.mesh_vertex_count > 0) {
      glBindVertexArray(params.mesh_vao);
      glDrawArrays(GL_POINTS, 0, params.mesh_vertex_count);
      glBindVertexArray(0);
    }
    
    glUseProgram(0);
  }
  
  glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

}  // namespace Renderer

