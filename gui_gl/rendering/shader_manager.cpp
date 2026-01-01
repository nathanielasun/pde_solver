#include "shader_manager.h"
#include <iostream>
#include <string>

namespace ShaderManager {

const char* GetDefaultVertexShader() {
  return "#version 330 core\n"
         "layout(location=0) in vec3 aPos;\n"
         "layout(location=1) in vec3 aColor;\n"
         "uniform mat4 uMVP;\n"
         "out vec3 vColor;\n"
         "void main() {\n"
         "  vColor = aColor;\n"
         "  gl_Position = uMVP * vec4(aPos, 1.0);\n"
         "  gl_PointSize = 2.5;\n"
         "}\n";
}

const char* GetDefaultFragmentShader() {
  return "#version 330 core\n"
         "in vec3 vColor;\n"
         "out vec4 FragColor;\n"
         "void main() {\n"
         "  FragColor = vec4(vColor, 1.0);\n"
         "}\n";
}

unsigned int CompileShader(GLenum type, const char* source) {
  GLuint shader = glCreateShader(type);
  glShaderSource(shader, 1, &source, nullptr);
  glCompileShader(shader);
  
  GLint success = 0;
  glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
  if (success == GL_FALSE) {
    GLint log_len = 0;
    glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &log_len);
    std::string log(static_cast<size_t>(log_len), '\0');
    glGetShaderInfoLog(shader, log_len, nullptr, log.data());
    std::cerr << "shader compile error: " << log << "\n";
    glDeleteShader(shader);
    return 0;
  }
  return shader;
}

unsigned int LinkProgram(unsigned int vertex_shader, unsigned int fragment_shader) {
  GLuint program = glCreateProgram();
  glAttachShader(program, vertex_shader);
  glAttachShader(program, fragment_shader);
  glLinkProgram(program);
  
  GLint success = 0;
  glGetProgramiv(program, GL_LINK_STATUS, &success);
  if (success == GL_FALSE) {
    GLint log_len = 0;
    glGetProgramiv(program, GL_INFO_LOG_LENGTH, &log_len);
    std::string log(static_cast<size_t>(log_len), '\0');
    glGetProgramInfoLog(program, log_len, nullptr, log.data());
    std::cerr << "shader link error: " << log << "\n";
    glDeleteProgram(program);
    return 0;
  }
  return program;
}

unsigned int CreateProgram(const char* vertex_source, const char* fragment_source) {
  unsigned int vs = CompileShader(GL_VERTEX_SHADER, vertex_source);
  if (!vs) {
    return 0;
  }
  
  unsigned int fs = CompileShader(GL_FRAGMENT_SHADER, fragment_source);
  if (!fs) {
    glDeleteShader(vs);
    return 0;
  }
  
  unsigned int program = LinkProgram(vs, fs);
  glDeleteShader(vs);
  glDeleteShader(fs);
  return program;
}

unsigned int CreateDefaultProgram() {
  return CreateProgram(GetDefaultVertexShader(), GetDefaultFragmentShader());
}

}  // namespace ShaderManager

