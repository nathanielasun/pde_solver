#include "window_manager.h"
#include <iostream>

WindowManager::~WindowManager() {
  Shutdown();
}

bool WindowManager::Init(const Config& config) {
  if (window_) {
    return true;  // Already initialized
  }

  if (!glfwInit()) {
    std::cerr << "Failed to initialize GLFW\n";
    return false;
  }
  glfw_initialized_ = true;

  // Set OpenGL version hints
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, config.gl_major);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, config.gl_minor);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#ifdef __APPLE__
  glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GLFW_TRUE);
#endif

  // Create window
  window_ = glfwCreateWindow(config.width, config.height, config.title.c_str(), 
                              nullptr, nullptr);
  if (!window_) {
    std::cerr << "Failed to create GLFW window\n";
    glfwTerminate();
    glfw_initialized_ = false;
    return false;
  }

  // Make context current
  glfwMakeContextCurrent(window_);

  // Set vsync
  glfwSwapInterval(config.vsync ? 1 : 0);

  return true;
}

void WindowManager::Shutdown() {
  if (window_) {
    glfwDestroyWindow(window_);
    window_ = nullptr;
  }
  if (glfw_initialized_) {
    glfwTerminate();
    glfw_initialized_ = false;
  }
}

bool WindowManager::ShouldClose() const {
  return window_ && glfwWindowShouldClose(window_);
}

void WindowManager::SetShouldClose(bool value) {
  if (window_) {
    glfwSetWindowShouldClose(window_, value ? GLFW_TRUE : GLFW_FALSE);
  }
}

void WindowManager::SwapBuffers() {
  if (window_) {
    glfwSwapBuffers(window_);
  }
}

void WindowManager::PollEvents() {
  glfwPollEvents();
}

int WindowManager::GetWidth() const {
  if (!window_) return 0;
  int width, height;
  glfwGetWindowSize(window_, &width, &height);
  return width;
}

int WindowManager::GetHeight() const {
  if (!window_) return 0;
  int width, height;
  glfwGetWindowSize(window_, &width, &height);
  return height;
}

void WindowManager::GetFramebufferSize(int* width, int* height) const {
  if (window_) {
    glfwGetFramebufferSize(window_, width, height);
  } else {
    if (width) *width = 0;
    if (height) *height = 0;
  }
}

