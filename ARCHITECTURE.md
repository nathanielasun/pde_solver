# PDE Solver Architecture (Condensed)

## Goals
- Keep solver, backends, visualization, and UI loosely coupled.
- Make feature upgrades and backend swaps safe and local.
- Provide a single authoritative map of modules to avoid documentation drift.

## Module Overview
- `src/` – Core solver logic, backend dispatch, numeric kernels, VTK I/O.
- `include/` – Public headers for solver/backends/types; keep minimal includes.
- `gui_gl/` – OpenGL/ImGui GUI (panels, rendering, handlers, systems, styles).
- `tools/` – LaTeX preview renderer for the GUI (Python).
- `third_party/` – External libraries (Dear ImGui, GLFW, stb).
- `build/` – CMake outputs (generated).

## Build Targets
- `pde_sim` – CLI solver; parses PDE, configures domain/BCs, dispatches to backends, writes VTK/VTI.
- `pde_gui` – OpenGL GUI; invokes the same solver/backends and visualizes outputs.

## Removed / Deprecated
- Legacy Qt Widgets GUI under `gui/` has been removed to reduce duplication and maintenance cost. All UI work lives in `gui_gl/`.

## Key Architectural Boundaries
- **Parsing**: `latex_parser`, `input_parse`, `expression_eval` produce typed configs.
- **Configuration/Types**: `pde_types`, `progress` shared between CLI and GUI.
- **Backends**: `backend` dispatches to CPU/CUDA/Metal; each backend isolates device-specific code.
- **Numerics**: `coordinate_metrics`, `finite_differences`, `embedded_boundary`, `boundary_utils`, `residual`.
- **I/O**: `vtk_io` handles legacy VTK and VTI, derived fields, checkpoint/restart.
- **Unstructured Mesh**: `mesh_io` provides legacy VTK/Gmsh mesh import/export; `unstructured_solver` is a separate stub engine with finite-element as the first planned backend (FV optional).
- **GUI**: `gui_gl/core` (app loop), `panels`, `handlers`, `rendering`, `systems`, `components`, `styles`.

## Maintenance Practices
- Keep new features behind clear interfaces (headers) with minimal cross-includes.
- Prefer small translation units (200–400 lines per responsibility) and avoid anonymous-namespace duplication.
- Update this file when moving modules or adding subsystems to prevent documentation drift.
