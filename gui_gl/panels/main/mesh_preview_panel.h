#ifndef MESH_PREVIEW_PANEL_H
#define MESH_PREVIEW_PANEL_H

#include "pde_types.h"
#include <vector>
#include <string>

struct MeshPreviewPanelState {
  Domain& domain;
  float input_width;
};

void RenderMeshPreviewPanel(MeshPreviewPanelState& state,
                             const std::vector<std::string>& components);

#endif  // MESH_PREVIEW_PANEL_H
