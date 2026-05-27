#include "latex_token_registry.h"

#include <iostream>

int main() {
  std::cout << LatexTokenRegistry::ExportCatalogJson();
  return 0;
}
