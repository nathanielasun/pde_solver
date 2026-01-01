#ifndef EXPORT_PANEL_H
#define EXPORT_PANEL_H

#include "GlViewer.h"
#include <vector>
#include <string>

struct ExportPanelState {
  GlViewer& viewer;
  float input_width;
};

void RenderExportPanel(ExportPanelState& state, const std::vector<std::string>& components);

#endif // EXPORT_PANEL_H

