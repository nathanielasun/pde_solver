#include "latex_token_registry.h"

#include <cstdlib>
#include <iostream>
#include <set>
#include <string>

int main() {
  const auto& entries = LatexTokenRegistry::AllEntries();
  if (entries.empty()) {
    std::cerr << "registry has no entries\n";
    return 1;
  }

  std::set<std::string> patterns;
  for (const auto& e : entries) {
    if (e.pattern == nullptr || e.pattern[0] == '\0') {
      std::cerr << "empty pattern in category " << (e.category ? e.category : "?") << "\n";
      return 1;
    }
    patterns.insert(e.pattern);
  }

  const std::string json = LatexTokenRegistry::ExportCatalogJson();
  if (json.find("\"entries\"") == std::string::npos) {
    std::cerr << "export json missing entries key\n";
    return 1;
  }

  std::cout << "latex_token_registry_test ok (" << entries.size() << " entries, "
            << patterns.size() << " unique patterns)\n";
  return 0;
}
