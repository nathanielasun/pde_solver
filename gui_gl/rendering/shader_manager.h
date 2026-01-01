#ifndef SHADER_MANAGER_H
#define SHADER_MANAGER_H

#include <OpenGL/gl3.h>
#include <string>

// Shader compilation and management utilities
namespace ShaderManager {

// Compile a shader from source code
// Returns shader ID on success, 0 on failure
unsigned int CompileShader(GLenum type, const char* source);

// Link vertex and fragment shaders into a program
// Returns program ID on success, 0 on failure
unsigned int LinkProgram(unsigned int vertex_shader, unsigned int fragment_shader);

// Compile a complete shader program from vertex and fragment source
// Returns program ID on success, 0 on failure
// Automatically cleans up intermediate shaders
unsigned int CreateProgram(const char* vertex_source, const char* fragment_source);

// Get the default vertex shader source for point rendering
const char* GetDefaultVertexShader();

// Get the default fragment shader source
const char* GetDefaultFragmentShader();

// Create the default shader program used by GlViewer
// Returns program ID on success, 0 on failure
unsigned int CreateDefaultProgram();

}  // namespace ShaderManager

#endif  // SHADER_MANAGER_H

