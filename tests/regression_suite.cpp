#include "self_test.h"

#include <cstdlib>
#include <iostream>
#include <string>

namespace {
int RunExecutable(const char* path) {
  const std::string cmd = std::string("\"") + path + "\"";
  const int code = std::system(cmd.c_str());
  if (code != 0) {
    std::cerr << "failed: " << path << " (exit " << code << ")\n";
    return 1;
  }
  std::cout << "passed: " << path << "\n";
  return 0;
}
}  // namespace

int main() {
  int failures = 0;
  failures += RunSelfTest();
  failures += RunExecutable("./build/burgers_fd_test");
  failures += RunExecutable("./build/shock_tube_test");
  failures += RunExecutable("./build/reaction_diffusion_imex_test");
  return failures == 0 ? 0 : 1;
}
