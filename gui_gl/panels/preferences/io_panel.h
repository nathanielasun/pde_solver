#ifndef IO_PANEL_H
#define IO_PANEL_H

#include "utils/file_dialog.h"
#include <string>
#include <vector>

// State structure for I/O panel
struct IOPanelState {
  std::string& output_path;
  std::string& input_dir;
  float input_width;
};

// Render the I/O panel using configured components.
void RenderIOPanel(IOPanelState& state, const std::vector<std::string>& components);

#endif  // IO_PANEL_H
