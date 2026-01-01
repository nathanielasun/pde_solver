#ifndef FIELD_PANEL_H
#define FIELD_PANEL_H

#include "GlViewer.h"
#include <mutex>
#include <vector>
#include <string>

struct FieldPanelState {
  GlViewer& viewer;
  std::mutex& state_mutex;
  DerivedFields& derived_fields;
  bool& has_derived_fields;
  int& field_type_index;
  bool use_volume;
  float input_width;

  // Multi-field support
  std::vector<std::string>& field_names;
  int& active_field_index;
  std::vector<DerivedFields>& all_derived_fields;
};

void RenderFieldPanel(FieldPanelState& state, const std::vector<std::string>& components);

#endif // FIELD_PANEL_H

