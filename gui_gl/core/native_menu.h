#ifndef NATIVE_MENU_H
#define NATIVE_MENU_H

#include "app_actions.h"

#include <functional>

struct GLFWwindow;

struct NativeMenuCallbacks {
  std::function<void(AppAction)> on_action;
  std::function<bool(AppAction)> can_execute;
  std::function<bool(AppAction)> is_checked;
};

class NativeMenu {
 public:
  static NativeMenu& Instance();

  void Install(GLFWwindow* window, const NativeMenuCallbacks& callbacks);
  bool IsInstalled() const;

 private:
  NativeMenu() = default;
};

#endif  // NATIVE_MENU_H
