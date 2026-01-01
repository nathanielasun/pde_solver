#include "render_types.h"
#include <cmath>

namespace RenderMath {

Mat4 Identity() {
  Mat4 out{};
  out.m[0] = 1.0f;
  out.m[5] = 1.0f;
  out.m[10] = 1.0f;
  out.m[15] = 1.0f;
  return out;
}

Mat4 Multiply(const Mat4& a, const Mat4& b) {
  Mat4 out{};
  for (int col = 0; col < 4; ++col) {
    for (int row = 0; row < 4; ++row) {
      out.m[col * 4 + row] =
          a.m[0 * 4 + row] * b.m[col * 4 + 0] +
          a.m[1 * 4 + row] * b.m[col * 4 + 1] +
          a.m[2 * 4 + row] * b.m[col * 4 + 2] +
          a.m[3 * 4 + row] * b.m[col * 4 + 3];
    }
  }
  return out;
}

Mat4 Translate(float x, float y, float z) {
  Mat4 out = Identity();
  out.m[12] = x;
  out.m[13] = y;
  out.m[14] = z;
  return out;
}

Mat4 Scale(float s) {
  Mat4 out{};
  out.m[0] = s;
  out.m[5] = s;
  out.m[10] = s;
  out.m[15] = 1.0f;
  return out;
}

Mat4 RotateX(float radians) {
  Mat4 out = Identity();
  const float c = std::cos(radians);
  const float s = std::sin(radians);
  out.m[5] = c;
  out.m[6] = s;
  out.m[9] = -s;
  out.m[10] = c;
  return out;
}

Mat4 RotateY(float radians) {
  Mat4 out = Identity();
  const float c = std::cos(radians);
  const float s = std::sin(radians);
  out.m[0] = c;
  out.m[2] = -s;
  out.m[8] = s;
  out.m[10] = c;
  return out;
}

Mat4 Perspective(float fovy_radians, float aspect, float z_near, float z_far) {
  Mat4 out{};
  const float f = 1.0f / std::tan(fovy_radians * 0.5f);
  out.m[0] = f / aspect;
  out.m[5] = f;
  out.m[10] = (z_far + z_near) / (z_near - z_far);
  out.m[11] = -1.0f;
  out.m[14] = (2.0f * z_far * z_near) / (z_near - z_far);
  return out;
}

Mat4 Orthographic(float left, float right, float bottom, float top, 
                  float z_near, float z_far) {
  Mat4 out{};
  out.m[0] = 2.0f / (right - left);
  out.m[5] = 2.0f / (top - bottom);
  out.m[10] = -2.0f / (z_far - z_near);
  out.m[12] = -(right + left) / (right - left);
  out.m[13] = -(top + bottom) / (top - bottom);
  out.m[14] = -(z_far + z_near) / (z_far - z_near);
  out.m[15] = 1.0f;
  return out;
}

}  // namespace RenderMath

