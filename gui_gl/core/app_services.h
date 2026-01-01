#ifndef APP_SERVICES_H
#define APP_SERVICES_H

#include <mutex>

// Forward declarations
class Application;

// Lightweight container for dependency injection across GUI components.
// Pointers are owned elsewhere (Application) and must outlive consumers.
struct AppServices {
  struct SharedState* shared_state = nullptr;
  std::mutex* state_mutex = nullptr;
  class GlViewer* viewer = nullptr;
  class SolverManager* solver_manager = nullptr;

  // Application pointer for view rendering (may be null if not in docking context)
  Application* app = nullptr;
};

#endif  // APP_SERVICES_H

