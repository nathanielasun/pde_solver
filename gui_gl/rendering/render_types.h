#ifndef RENDER_TYPES_H
#define RENDER_TYPES_H

// Common types used across rendering modules

struct Vertex {
  float x;
  float y;
  float z;
  float r;
  float g;
  float b;
};

struct Mat4 {
  float m[16];
};

namespace RenderMath {
constexpr float kPi = 3.14159265358979323846f;

Mat4 Identity();
Mat4 Multiply(const Mat4& a, const Mat4& b);
Mat4 Translate(float x, float y, float z);
Mat4 Scale(float s);
Mat4 RotateX(float radians);
Mat4 RotateY(float radians);
Mat4 Perspective(float fovy_radians, float aspect, float z_near, float z_far);
Mat4 Orthographic(float left, float right, float bottom, float top, 
                  float z_near, float z_far);
}  // namespace RenderMath

#endif  // RENDER_TYPES_H

