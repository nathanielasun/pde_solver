#include "preset_layouts.h"

std::unique_ptr<DockNode> PresetLayouts::CreateDefault() {
  // Main split: Left (equation + run) | Right (viewer + timeline)
  // Panels are given ample space to show all their contents.
  // Layout:
  // +------------------------+-------------------------+
  // |                        |                         |
  // |    EquationEditor      |                         |
  // |                        |       Viewer3D          |
  // |------------------------|                         |
  // |                        |                         |
  // |    RunControls         |-------------------------|
  // |                        |       Timeline          |
  // +------------------------+-------------------------+

  return CreateHorizontalSplit(0.40f,
    // Left column: only essential panels with room to breathe
    CreateVerticalSplit(0.60f,
      CreateLeafNode(ViewType::EquationEditor),
      CreateLeafNode(ViewType::RunControls)
    ),
    // Right column: viewer on top, timeline on bottom
    CreateVerticalSplit(0.88f,
      CreateLeafNode(ViewType::Viewer3D),
      CreateLeafNode(ViewType::Timeline)
    )
  );
}

std::unique_ptr<DockNode> PresetLayouts::CreateInspection() {
  // Optimized for post-processing visualization.
  // Fewer panels, each with ample space for controls.
  // Layout:
  // +------------------------+-------------------------+
  // |                        |                         |
  // |    FieldSelector       |                         |
  // |                        |       Viewer3D          |
  // |------------------------|                         |
  // |                        |                         |
  // |    SliceControls       |-------------------------|
  // |                        |       Timeline          |
  // +------------------------+-------------------------+

  return CreateHorizontalSplit(0.35f,
    // Left: primary inspection controls with room to show all options
    CreateVerticalSplit(0.50f,
      CreateLeafNode(ViewType::FieldSelector),
      CreateLeafNode(ViewType::SliceControls)
    ),
    // Right: viewer + timeline
    CreateVerticalSplit(0.88f,
      CreateLeafNode(ViewType::Viewer3D),
      CreateLeafNode(ViewType::Timeline)
    )
  );
}

std::unique_ptr<DockNode> PresetLayouts::CreateDualViewer() {
  // Two viewers side by side for comparison
  // Layout:
  // +------------+------------+
  // |            |            |
  // |  Viewer3D  |  Viewer3D  |
  // |            |            |
  // +------------+------------+
  // |       Timeline          |
  // +-------------------------+

  return CreateVerticalSplit(0.85f,
    CreateHorizontalSplit(0.50f,
      CreateLeafNode(ViewType::Viewer3D),
      CreateLeafNode(ViewType::Viewer3D)
    ),
    CreateLeafNode(ViewType::Timeline)
  );
}

std::unique_ptr<DockNode> PresetLayouts::CreateMinimal() {
  // Just the 3D viewer with timeline
  // Layout:
  // +-------------------------+
  // |                         |
  // |       Viewer3D          |
  // |                         |
  // +-------------------------+
  // |       Timeline          |
  // +-------------------------+

  return CreateVerticalSplit(0.90f,
    CreateLeafNode(ViewType::Viewer3D),
    CreateLeafNode(ViewType::Timeline)
  );
}

std::unique_ptr<DockNode> PresetLayouts::CreateFullConfiguration() {
  // All essential solver configuration panels visible with adequate space.
  // Uses a two-column layout for configuration panels.
  // Layout:
  // +------------------+------------------+------------------------+
  // |                  |                  |                        |
  // |  Equation        |  Domain          |                        |
  // |                  |                  |       Viewer3D         |
  // |------------------|------------------|                        |
  // |                  |                  |                        |
  // |  SolverConfig    |  BoundaryCondns  |------------------------|
  // |                  |                  |       Timeline         |
  // +------------------+------------------+------------------------+

  return CreateHorizontalSplit(0.50f,
    // Left: two columns of essential settings
    CreateHorizontalSplit(0.50f,
      // Left column: equation + solver
      CreateVerticalSplit(0.50f,
        CreateLeafNode(ViewType::EquationEditor),
        CreateLeafNode(ViewType::SolverConfig)
      ),
      // Right column: domain + boundary
      CreateVerticalSplit(0.50f,
        CreateLeafNode(ViewType::DomainSettings),
        CreateLeafNode(ViewType::BoundaryConditions)
      )
    ),
    // Right: viewer + timeline
    CreateVerticalSplit(0.88f,
      CreateLeafNode(ViewType::Viewer3D),
      CreateLeafNode(ViewType::Timeline)
    )
  );
}
