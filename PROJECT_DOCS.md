# Project Documentation Log

This file tracks the purpose of key files and modules in the repository. For a
condensed module map, see `ARCHITECTURE.md`.

## Core C++ Application

- `CMakeLists.txt`: Build configuration for the PDE solver, including optional CUDA/Metal backends.
- `src/main.cpp`: CLI entry point; parses arguments, validates inputs, selects backend, writes VTK output, and handles unstructured mesh previews (`--mesh`). Supports time-dependent solves, checkpoint/restart, output format selection, batch runner (`--batch`), output naming patterns, convergence sweeps, dataset tools, coupled PDE examples (`--list-examples`/`--run-example`), advection tests (`--advection-test`), time integrator tests (`--time-integrator-test`), pressure projection tests (`--projection-test`), lid-driven cavity benchmark (`--lid-cavity [Re]`), and `--validate`/`--dump-operator[=json]` (JSON includes `schema_version`).
- `src/advection.cpp`: Advection discretization schemes (upwind, Lax-Wendroff, Beam-Warming, Fromm, TVD with minmod/superbee/vanLeer/MC limiters). Provides flux computation for 1D/2D/3D and CFL helpers.
- `include/advection.h`: Advection scheme interface with `ComputeAdvectionFlux1D()`, `ComputeAdvectionTerm2D/3D()`, flux limiters, and CFL utilities.
- `src/advection_tests.cpp`: Advection test suite with top-hat, Gaussian, and rotation tests. Provides `RunAdvectionTestSuite()`, `PrintAdvectionTestResults()`, and `CompareAdvectionSchemes()`.
- `include/advection_tests.h`: Advection test interfaces and `AdvectionTestResult` struct.
- `src/time_integrator.cpp`: Time integration methods including explicit (Forward Euler, RK2, RK4, SSPRK2, SSPRK3) and implicit (Backward Euler, Crank-Nicolson) schemes. Provides IMEX methods (IMEXEulerStep, IMEXSSP2Step), operator splitting (OperatorSplitStep), CFL-based dt selection (SuggestStableDt, ComputeCFLAdvectionDiffusion), and adaptive time stepping (AdaptiveTimeStep).
- `include/time_integrator.h`: Time integrator interface with `TimeStep()`, RK and SSP step functions, implicit solver wrappers, IMEX methods for stiff systems, `OperatorSplitConfig` for Strang/Lie-Trotter splitting, and adaptive stepping with error control.
- `src/time_integrator_tests.cpp`: Time integrator test suite with exponential decay, harmonic oscillator, advection MOL, stiff decay, convergence studies, IMEX tests, adaptive stepping tests, and CFL stepping tests. Provides `RunTimeIntegratorTestSuite()`, `PrintTimeIntegratorTestResults()`, and `CompareTimeIntegrators()`.
- `include/time_integrator_tests.h`: Time integrator test interfaces and `TimeIntegratorTestResult` struct.
- `src/pressure_projection.cpp`: Chorin-Temam pressure projection method for incompressible flow. Implements 2D/3D Poisson solvers with Neumann BCs, divergence/gradient operators, velocity projection, and incompressible Navier-Stokes time stepping with advection and viscous terms.
- `include/pressure_projection.h`: Pressure projection interface with `VelocityField2D/3D` structs, `ProjectVelocity2D/3D()`, `SolvePressurePoisson2D/3D()`, `NavierStokesStep2D()`, `IncompressibleNSConfig`, and utility functions for kinetic energy, enstrophy, and vorticity.
- `src/pressure_projection_tests.cpp`: Pressure projection test suite with simple projection, divergence-free preservation, Taylor-Green vortex, and lid-driven cavity benchmarks. Provides `RunPressureProjectionTestSuite()`, `RunLidDrivenCavityTest()`, and convergence studies.
- `include/pressure_projection_tests.h`: Pressure projection test interfaces and `PressureProjectionTestResult`, `LidDrivenCavityResult` structs.
- `src/self_test.cpp`: Self-test and regression suite harness used by `--self-test` and the `regression_suite` target.
- `src/backend.cpp`: Backend selection and hardware detection for CPU/CUDA/Metal/TPU. Handles solver method dispatch and residual computation.
- `include/backend.h`: Backend API declarations used by the CLI and solver.
- `GPU_PARITY_PLAN.md`: Backend capability matrix and GPU parity priorities (Phase 7.1 kickoff).
- `src/backends/cpu/solver.cpp`: CPU solver implementations (Jacobi, Gauss-Seidel, SOR, CG, BiCGStab, GMRES, Multigrid V-cycle). Supports 2D/3D, time-dependent, metric-corrected coordinate systems, embedded boundaries, spatial RHS/nonlinear terms, variable coefficients, and mixed/higher-order derivatives for Jacobi/Gauss-Seidel/SOR in Cartesian coordinates (5-point stencils require nx/ny/nz >= 5).
- `src/coupled_solver.cpp`: Multi-field coupled PDE solver implementing explicit coupling (operator splitting) and Picard iteration (block Gauss-Seidel). Dispatches via `SolveCoupledPDE()` based on `CouplingStrategy`; includes under-relaxation, per-field residual tracking, and time-series support.
- `include/coupled_solver.h`: Coupled solver interface with `SolveCoupledPDE()`, `SolveCoupledPDETimeSeries()`, `ComputeFieldChangeNorm()`, and `BuildSingleFieldInput()` for multi-field PDE systems.
- `src/coupled_examples.cpp`: Coupled PDE example systems including Gray-Scott reaction-diffusion, Heat-Diffusion multi-physics, Brusselator, and Predator-Prey (Lotka-Volterra). Provides `GetCoupledPDEExamples()`, `CreateGrayScottExample()`, `CreateHeatDiffusionExample()`, `CreateBrusselatorExample()`, `CreatePredatorPreyExample()`, and `BuildSolveInputFromExample()`.
- `include/coupled_examples.h`: Coupled PDE example interface with `CoupledPDEExample` struct and factory functions for standard reaction-diffusion and multi-physics systems.
- `src/backends/cpu/cpu_utils.cpp`: CPU helper routines (vector norms, Ax2D, and matrix-free LinearOperator2D apply used by Krylov solvers).
- `include/solver.h`: CPU solver interface (steady-state and time-dependent).
- `src/expression_eval.cpp`: LaTeX-to-expression evaluator used for domain shape masking, RHS expressions, boundary conditions, and variable coefficients.
- `include/expression_eval.h`: Expression evaluator interface for custom domain shapes, spatial RHS, and time-dependent BCs.
- `include/coefficient_evaluator.h`: Parsed coefficient expression cache for variable-coefficient PDE terms.
- `src/coefficient_evaluator.cpp`: Variable coefficient evaluation helpers shared by solver and residuals.
- `src/dataset_tools.cpp`: Dataset indexing and cleanup helpers (scan summary/meta sidecars, aggregate stats, remove orphans).
- `include/dataset_tools.h`: Dataset index/cleanup interfaces and result structs.
- `src/latex_parser.cpp`: Strict LaTeX parser for PDE terms (including variable coefficients), spatial RHS expressions, and nonlinear terms. Supports multi-field parsing with `ParseForField()` and `ParseMultiField()` for coupled PDE systems.
- `include/latex_parser.h`: LaTeX parsing interface. Includes `LatexParseResult` with multi-field extensions (field_coeffs, detected_fields), `MultiFieldParseResult` for coupled systems, and `CouplingAnalysis` for pattern classification.
- `src/latex_ast.cpp`: AST construction for LaTeX terms. Includes `ParseTermMultiField()` for detecting field variables (u, v, w, etc.) and routing coefficients to per-field storage.
- `include/latex_patterns.h`: LaTeX pattern registry for derivative matching. Includes pattern generators (`GenerateD2XPatterns()`, etc.) for any field variable and `DetectFieldVariable()` for multi-field support.
- `src/input_parse.cpp`: Input parsing for domain, grid, and boundary condition specifications (including piecewise BCs).
- `include/input_parse.h`: Input parsing interface.
- `src/residual.cpp`: Residual norm computation (L2 and L-infinity) and error norm computation for manufactured solutions.
- `include/residual.h`: Residual and error norm computation interfaces.
- `include/self_test.h`: Self-test entry point shared by the CLI and regression runner.
- `src/finite_differences.cpp`: Finite difference stencils for mixed derivatives (u_xy, u_xz, u_yz) plus third/fourth-order derivatives (u_xxx/u_yyy/u_zzz, u_xxxx/u_yyyy/u_zzzz).
- `include/finite_differences.h`: Mixed derivative computation interface for cross-derivatives in 2D and 3D.
- `src/coordinate_metrics.cpp`: Metric term computation for non-Cartesian coordinate systems (polar, cylindrical, spherical, toroidal). Handles singularities and periodic boundaries.
- `include/coordinate_metrics.h`: Coordinate system metric computation interface.
- `src/conserved_monitor.cpp`: Conserved quantity monitor helpers (mass/energy/max|u| and drift detection) for time-dependent runs.
- `include/conserved_monitor.h`: Conserved quantity monitor interfaces and sample/monitor structs.
- `src/embedded_boundary.cpp`: Embedded boundary condition implementation (cut-cell/ghost-fluid approach) for implicit domain shapes.
- `include/embedded_boundary.h`: Embedded boundary condition interface.
- `src/vtk_io.cpp`: VTK legacy `STRUCTURED_POINTS` reader/writer, VTK XML ImageData (.vti) writer, derived fields computation, checkpoint/restart I/O (v2 includes velocity for `u_tt`), and point-cloud import for `POLYDATA`/`UNSTRUCTURED_GRID`.
- `include/vtk_io.h`: VTK I/O interfaces, point-cloud reader, derived fields computation, checkpoint/restart structs (including optional velocity), and random output tag helper.
- `src/mesh_io.cpp`: Unstructured mesh import/export for legacy VTK unstructured grids and ASCII Gmsh `.msh` files, plus mesh summary helpers.
- `include/mesh_io.h`: Unstructured mesh structs, summary helpers, and read/write interfaces.
- `src/unstructured_solver.cpp`: Unstructured solver stub (finite-element backend planned).
- `include/unstructured_solver.h`: Unstructured solver stub interface with FE/FV discretization selection.
- `include/pde_types.h`: Shared data types (PDE coefficients, domain, boundary conditions, solver configuration, coordinate systems, piecewise BCs, nonlinear terms, point samples). Includes multi-field types: `CrossFieldCoefficients`, `FieldEquationCoefficients`, `MultiFieldEquation`, `CouplingPattern`, `CouplingAnalysis`, `CouplingStrategy` (Explicit/Picard), `CouplingConfig`, and `CouplingDiagnostics` for coupled PDE systems.
- `include/progress.h`: Progress callback types shared between solver, I/O, and GUI.

## CUDA Backend

- `src/backends/cuda/jacobi/cuda_solver.cu`: CUDA solver implementations (Jacobi, Red-Black Gauss-Seidel, SOR, CG, BiCGStab, GMRES, Multigrid V-cycle). Supports 3D domains, spatial RHS, and basic nonlinear terms.
- `include/cuda_solver.h`: CUDA solver interface.

## Metal Backend

- `src/backends/metal/jacobi/metal_solver.mm`: Metal solver implementation and runtime shader loading. Supports Jacobi, Red-Black Gauss-Seidel, SOR, CG, BiCGStab, GMRES, and Multigrid V-cycle.
- `src/backends/metal/jacobi/metal_kernels.metal`: Metal compute kernels for boundaries, Jacobi steps, vector operations, and multigrid operations.
- `include/metal_solver.h`: Metal solver interface.

## Tools and UI

- `tools/render_latex.py`: Local LaTeX-to-PNG renderer used by the GUI.

## Tests

- `tests/regression_suite.cpp`: CLI-style regression runner for analytic golden outputs.
- `tests/checkpoint_equivalence.cpp`: Checkpoint restart equivalence regression test.

## OpenGL GUI (gui_gl)

Primary UI path (Dear ImGui + GLFW). Requires OpenGL 3.3. UI layout is driven by
`gui_gl/config/default_ui.json` with optional `ui_config.json` overrides. The
former Qt GUI under `gui/` has been removed to reduce duplication; all UI work
now lives in `gui_gl/`.

### Core Application

- `gui_gl/main.cpp`: OpenGL GUI entry point (Dear ImGui + GLFW). Creates application and starts main loop.
- `gui_gl/core/application.h`: High-level GUI coordinator (panel registry, UIConfig layout, main loop wiring).
- `gui_gl/core/application.cpp`: OpenGL GUI runtime entrypoint and panel orchestration (~1593 lines).
- `gui_gl/core/window_manager.h`: GLFW window creation and management.
- `gui_gl/core/window_manager.cpp`: Window manager implementation.
- `gui_gl/core/event_handler.h`: Input event handling (mouse, keyboard).
- `gui_gl/core/event_handler.cpp`: Event handler implementation.
- `gui_gl/app_state.h`: Application state management (solver state, UI state, viewer state).
- `gui_gl/app_state.cpp`: Application state implementation.
- `gui_gl/app_helpers.h`: Helper functions for GUI operations (solver invocation, file I/O, LaTeX rendering).
- `gui_gl/app_helpers.cpp`: Application helper implementations.
- `gui_gl/GlViewer.h`: OpenGL viewer API (view modes, grid, slice/isosurface filters).
- `gui_gl/GlViewer.cpp`: OpenGL 3D point viewer with color gradient rendering to texture (~2252 lines, being refactored into rendering modules).
- `gui_gl/stb_image_impl.cpp`: stb_image implementation for loading LaTeX PNG previews.
- `gui_gl/validation.h`: Input validation for PDE, domain, and boundary conditions.
- `gui_gl/validation.cpp`: Validation implementation.
- `gui_gl/progress_feedback.h`: Progress feedback interface for solver callbacks.

### UI Components

- `gui_gl/components/ui_component.h`: Base UI component interface.
- `gui_gl/components/ui_component.cpp`: Base UI component implementation.
- `gui_gl/components/log_panel.h`: Log panel component for displaying solver messages and status.
- `gui_gl/components/log_panel.cpp`: Log panel implementation.
- `gui_gl/components/inspection_tools.h`: Inspection tools component (slice planes, isosurfaces, point queries).
- `gui_gl/components/inspection_tools.cpp`: Inspection tools implementation.
- `gui_gl/components/comparison_tools.h`: Comparison tools component for solution comparison and difference analysis.
- `gui_gl/components/comparison_tools.cpp`: Comparison tools implementation.
- `gui_gl/components/color_picker.h`: Color picker widgets for appearance settings.
- `gui_gl/components/color_picker.cpp`: Color picker implementation.
- `gui_gl/components/error_dialog.h`: Centralized error dialog component.
- `gui_gl/components/error_dialog.cpp`: Error dialog implementation.

### UI Styling

- `gui_gl/styles/ui_style.h`: Style system declarations (spacing, typography, buttons, toasts).
- `gui_gl/styles/ui_style.cpp`: Style system implementation.
- `gui_gl/styles/fonts/`: Bundled font files available to the UI Configuration panel.

### UI Panels

- `gui_gl/panels/main/equation_panel.h`: PDE equation input panel with LaTeX preview.
- `gui_gl/panels/main/equation_panel.cpp`: Equation panel implementation.
- `gui_gl/panels/main/boundary_panel.h`: Boundary condition input panel with per-side BC configuration and piecewise BC support.
- `gui_gl/panels/main/boundary_panel.cpp`: Boundary panel implementation (clamps invalid boundary kinds to Dirichlet for UI safety).
- `gui_gl/panels/main/domain_panel.h`: Domain configuration panel (bounding box, implicit shapes, coordinate systems).
- `gui_gl/panels/main/domain_panel.cpp`: Domain panel implementation.
- `gui_gl/panels/main/grid_panel.h`: Grid resolution panel.
- `gui_gl/panels/main/grid_panel.cpp`: Grid panel implementation.
- `gui_gl/panels/main/compute_panel.h`: Solver backend/method configuration panel.
- `gui_gl/panels/main/compute_panel.cpp`: Compute panel implementation.
- `gui_gl/panels/main/time_panel.h`: Time range controls for transient solves.
- `gui_gl/panels/main/time_panel.cpp`: Time panel implementation.
- `gui_gl/panels/main/run_panel.h`: Run/stop controls and progress display.
- `gui_gl/panels/main/run_panel.cpp`: Run panel implementation.
- `gui_gl/panels/inspect/field_panel.h`: Field selector panel for derived fields.
- `gui_gl/panels/inspect/field_panel.cpp`: Field panel implementation.
- `gui_gl/panels/inspect/slice_panel.h`: Slice controls panel.
- `gui_gl/panels/inspect/slice_panel.cpp`: Slice panel implementation.
- `gui_gl/panels/inspect/isosurface_panel.h`: Isosurface controls panel.
- `gui_gl/panels/inspect/isosurface_panel.cpp`: Isosurface panel implementation.
- `gui_gl/panels/inspect/export_panel.h`: Image export panel.
- `gui_gl/panels/inspect/export_panel.cpp`: Export panel implementation.
- `gui_gl/panels/inspect/advanced_panel.h`: Advanced inspection tools panel.
- `gui_gl/panels/inspect/advanced_panel.cpp`: Advanced panel implementation.
- `gui_gl/panels/inspect/comparison_panel.h`: Comparison tools panel.
- `gui_gl/panels/inspect/comparison_panel.cpp`: Comparison panel implementation.
- `gui_gl/panels/preferences/viewer_panel.h`: Viewer tuning panel (camera, point size, grid).
- `gui_gl/panels/preferences/viewer_panel.cpp`: Viewer panel implementation.
- `gui_gl/panels/preferences/io_panel.h`: Output/input path panel.
- `gui_gl/panels/preferences/io_panel.cpp`: I/O panel implementation.
- `gui_gl/panels/preferences/benchmark_panel.h`: Benchmark preset panel.
- `gui_gl/panels/preferences/benchmark_panel.cpp`: Benchmark panel implementation.
- `gui_gl/panels/preferences/color_preferences_panel.h`: Appearance (theme/color) panel.
- `gui_gl/panels/preferences/color_preferences_panel.cpp`: Color preferences panel implementation.
- `gui_gl/panels/preferences/ui_config_panel.h`: UIConfig editor panel.
- `gui_gl/panels/preferences/ui_config_panel.cpp`: UIConfig panel implementation.

### Rendering

The rendering subsystem has been extracted from `GlViewer.cpp` into focused modules:

- `gui_gl/rendering/render_types.h`: Rendering type definitions (Vertex, Mat4, matrix math utilities).
- `gui_gl/rendering/render_types.cpp`: Render type implementations and matrix math.
- `gui_gl/rendering/shader_manager.h`: OpenGL shader compilation and program management.
- `gui_gl/rendering/shader_manager.cpp`: Shader manager implementation (~200 lines).
- `gui_gl/rendering/coordinate_transforms.h`: Coordinate system transformation utilities (Polar, Cylindrical, Spherical, Toroidal to Cartesian).
- `gui_gl/rendering/coordinate_transforms.cpp`: Coordinate transformation implementations (~400 lines).
- `gui_gl/rendering/mesh_builder.h`: Mesh construction from point clouds and grid data.
- `gui_gl/rendering/mesh_builder.cpp`: Mesh builder implementation with slice/isosurface filtering (~400 lines).
- `gui_gl/rendering/grid_builder.h`: Grid line generation for visualization.
- `gui_gl/rendering/grid_builder.cpp`: Grid builder implementation (~200 lines, coordinate-system aware grid generation).
- `gui_gl/rendering/projection.h`: 3D to 2D projection and axis labeling.
- `gui_gl/rendering/projection.cpp`: Projection implementation (~200 lines, axis label projection for viewer overlay).
- `gui_gl/rendering/renderer.h`: Main rendering orchestration (FBO, viewport, pipeline).
- `gui_gl/rendering/renderer.cpp`: Renderer implementation (~200 lines).
- `gui_gl/rendering/glyph_renderer.h`: Glyph rendering for vector field visualization.
- `gui_gl/rendering/glyph_renderer.cpp`: Glyph renderer implementation.
- `gui_gl/rendering/value_sampling.h`: Value sampling utilities for inspection.
- `gui_gl/rendering/value_sampling.cpp`: Value sampling implementation.

**Note**: GlViewer.cpp integration with these modules is pending. See `PENDING_UPDATES.md` for status.

### Systems

- `gui_gl/systems/backend_capabilities.h`: Backend capability detection and querying (supported methods, features).
- `gui_gl/systems/backend_capabilities.cpp`: Backend capabilities implementation.
- `gui_gl/systems/backend_providers.h`: Backend provider interface for backend selection and management.
- `gui_gl/systems/backend_providers.cpp`: Backend provider implementations.
- `gui_gl/systems/command_history.h`: Command pattern implementation for undo/redo functionality.
- `gui_gl/systems/command_history.cpp`: Command history management with undo/redo support.
- `gui_gl/systems/ui_config.h`: UI layout configuration schema and registry (theme sizing, font path, panel layout).
- `gui_gl/systems/ui_config.cpp`: UI config load/save and defaults, including theme font path handling.
- `gui_gl/systems/ui_config_validation.cpp`: UI config validation.

### Utilities

- `gui_gl/utils/coordinate_utils.h`: GUI-specific coordinate system constants and view mode mapping.
- `gui_gl/utils/coordinate_utils.cpp`: Coordinate utility implementations.
- `gui_gl/utils/math_utils.h`: Mathematical utility functions for GUI.
- `gui_gl/utils/math_utils.cpp`: Math utility implementations.
- `gui_gl/utils/path_utils.h`: File path manipulation utilities.
- `gui_gl/utils/path_utils.cpp`: Path utility implementations.
- `gui_gl/utils/string_utils.h`: String manipulation utilities.
- `gui_gl/utils/string_utils.cpp`: String utility implementations.
- `gui_gl/utils/format_utils.h`: Formatting utilities (numbers, scientific notation).
- `gui_gl/utils/format_utils.cpp`: Format utility implementations.

### I/O and Templates

- `gui_gl/io/preferences_io.h`: User preferences I/O (saving/loading GUI settings).
- `gui_gl/io/preferences_io.cpp`: Preferences I/O implementation.
- `gui_gl/io/file_utils.h`: File utility functions (latest VTK detection, path resolution).
- `gui_gl/io/file_utils.cpp`: File utility implementations.
- `gui_gl/io/image_export.h`: Image export functionality (PNG, JPG).
- `gui_gl/io/image_export.cpp`: Image export implementation.
- `gui_gl/templates.h`: Template PDE and configuration presets.
- `gui_gl/templates.cpp`: Template implementations.
- `gui_gl/ui_helpers.h`: UI helper functions (ImGui utilities, formatting).
- `gui_gl/ui_helpers.cpp`: UI helper implementations.

### Tools and Solver Integration

- `gui_gl/tools/comparison_tools.h`: Solution comparison engine for computing difference fields and relative errors.
- `gui_gl/tools/comparison_tools.cpp`: Comparison tools implementation.
- `gui_gl/solver/solver_manager.h`: Solver execution management.
- `gui_gl/solver/solver_manager.cpp`: Solver manager implementation.

### Handlers

- `gui_gl/handlers/file_handler.h`: File handling operations (VTK loading, saving, latest file detection).
- `gui_gl/handlers/file_handler.cpp`: File handler implementation.
- `gui_gl/handlers/solve_handler.h`: Solver execution orchestration and thread management.
- `gui_gl/handlers/solve_handler.cpp`: Solve handler implementation.

### Models

- `gui_gl/models/application_state.h`: Application state data model.
- `gui_gl/models/application_state.cpp`: Application state implementation.

## Qt Widgets GUI (Removed)

The legacy Qt Widgets GUI under `gui/` was removed to simplify maintenance. New
UI work should target `gui_gl/` exclusively.

## Third-Party

- `third_party/imgui`: Dear ImGui sources for the OpenGL GUI.
- `third_party/glfw`: GLFW windowing library for OpenGL context creation.
- `third_party/stb/stb_image.h`: stb_image header for PNG loading.

## Documentation

### Primary Documentation
- `README.md`: Build, run, backend, and visualization instructions with complete examples.
- `AGENTS.md`: Contributor guidelines.
- `CLAUDE.md`: Symlink to contributor guidelines.

### Implementation Status
- `COMPLETED_PHASES.md`: **Consolidated summary of all completed implementation phases** (Phase 1 and Phase 2).
- `PROPOSED_PHASES.md`: **Consolidated summary of all proposed work, organized into separable tasks** for parallel agent assignment.

### Detailed References
- `PDE_TERMS_RECOMMENDATIONS.md`: Recommendations for PDE term usage and best practices.
- `PDE_TERMS_IMPLEMENTATION_PLAN.md`: Detailed implementation plan for PDE term support.
- `GUI_UX_IMPROVEMENTS.md`: Original GUI/UX improvement recommendations and specifications.
- `ARCHITECTURE.md`: High-level architectural overview and module boundaries.

**Note**: Documentation has been consolidated:
- **Completed work** → `COMPLETED_PHASES.md`
- **Ongoing/partial work** → `PENDING_UPDATES.md`  
- **Future work** → `PROPOSED_PHASES.md`
- Individual phase/refactoring documents have been removed to reduce duplication.

### Refactoring Status
- Refactoring plans have been consolidated into `PENDING_UPDATES.md`
- See `PENDING_UPDATES.md` for ongoing refactoring work and completion status
