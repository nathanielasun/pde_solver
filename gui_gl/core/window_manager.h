#ifndef WINDOW_MANAGER_H
#define WINDOW_MANAGER_H

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#include <string>

/**
 * WindowManager - Encapsulates GLFW window creation and management.
 * 
 * Responsibilities:
 * - Initialize GLFW
 * - Create and configure the application window
 * - Handle window lifecycle (resize, close)
 * - Manage OpenGL context
 */
class WindowManager {
 public:
  struct Config {
    int width;
    int height;
    std::string title;
    int gl_major;
    int gl_minor;
    bool vsync;
    Config()
        : width(1400),
          height(900),
          title("PDE Solver"),
          gl_major(3),
          gl_minor(3),
          vsync(true) {}
  };

  WindowManager() = default;
  ~WindowManager();

  // Non-copyable
  WindowManager(const WindowManager&) = delete;
  WindowManager& operator=(const WindowManager&) = delete;

  // Initialize GLFW and create window
  bool Init(const Config& config);

  // Shutdown and cleanup
  void Shutdown();

  // Check if window should close
  bool ShouldClose() const;

  // Set window should close flag
  void SetShouldClose(bool value);

  // Swap buffers and poll events
  void SwapBuffers();
  void PollEvents();

  // Get window dimensions
  int GetWidth() const;
  int GetHeight() const;
  void GetFramebufferSize(int* width, int* height) const;

  // Get raw GLFW window pointer (for ImGui init, etc.)
  GLFWwindow* GetWindow() const { return window_; }

  // Check if initialized
  bool IsInitialized() const { return window_ != nullptr; }

 private:
  GLFWwindow* window_ = nullptr;
  bool glfw_initialized_ = false;
};

#endif  // WINDOW_MANAGER_H

