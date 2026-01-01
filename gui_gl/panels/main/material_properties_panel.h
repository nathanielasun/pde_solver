#ifndef MATERIAL_PROPERTIES_PANEL_H
#define MATERIAL_PROPERTIES_PANEL_H

#include <string>
#include <vector>

struct MaterialPropertiesPanelState {
  float input_width;
};

void RenderMaterialPropertiesPanel(MaterialPropertiesPanelState& state,
                                    const std::vector<std::string>& components);

#endif  // MATERIAL_PROPERTIES_PANEL_H
