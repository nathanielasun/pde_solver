# Master Roadmap

**Last updated:** 2026-05-27  
**Status:** Active — single source of truth for planned work  
**Audience:** Maintainers and agents working systematically through backlog items

This document merges intent from all planning files in `docs/planning/` and the
current codebase state. Work packages below are ordered for systematic execution:
complete earlier tracks (or individual packages) before dependent ones unless noted.

---

## How to work through this document

1. Start at **Track 0** (documentation hygiene) — low risk, unblocks accurate planning.
2. Proceed through **Tracks 1–5** in order; these address the highest-impact gaps
   identified in the 2026-05-27 project audit.
3. **Tracks 6–8** are longer-horizon; schedule after Track 1–5 milestones close.
4. For each work package (WP):
   - Read **Original intent** to understand design goals.
   - Implement **Tasks** in order.
   - Verify **Acceptance criteria** before marking complete.
   - Update the status cell in the summary table.

**Status legend:** `⬜ Not started` · `🔄 In progress` · `✅ Done` · `⏸ Blocked`

---

## Executive summary

| # | Track | Priority | Status | Outcome |
| --- | --- | --- | --- | --- |
| 0 | Documentation & hygiene | P0 | ⬜ | Docs match code; no broken references |
| 1 | Option 1 completion (linear FD) | P0 | 🔄 ~65% | Full CPU linear operator pipeline + CLI gaps closed |
| 2 | Nonlinear derivative terms | P0 | ⬜ | Parsed terms actually solvable (CPU 2D first) |
| 3 | GPU parity (Phase 7.1) | P1 | ⬜ | CUDA/Metal match CPU for high-use 2D cases |
| 4 | Docking UI productization | P1 | 🔄 ~85% built | Docking UI default; tab UI deprecated |
| 5 | GUI Phase 3–4 remainder | P2 | 🔄 ~45% | Registries finished; advanced viz/export |
| 6 | Option 2 (nonlinear / FV) | P2 | ✅ ~85% CPU MVP | FD nonlinear derivs, FV+Riemann, IMEX, CLI/GUI wiring |
| 7 | Option 3 (FEM / unstructured) | P3 | ⬜ ~5% | FE assembly on imported meshes |
| 8 | Tiers 3–4 specialized terms | P3 | ⬜ | Integro, vector PDEs, fractional, etc. |

---

## Current state (merged audit)

### Solver core — what works today

- **CLI automation:** `--validate`, `--dump-operator`, `--dump-metadata`, `--dump-summary`,
  JSON run configs, batch manifests, dataset index/cleanup.
- **CPU linear FD (Option 1 subset):** mixed derivatives, higher-order derivatives,
  variable coefficients on Jacobi/GS/SOR and BiCGStab/GMRES (Cartesian; grid ≥ 5 where
  5-point stencils apply).
- **CPU extras:** implicit shapes, piecewise BCs, global integrals, scalar nonlinear
  reactions, coupled multi-field (explicit + Picard), advection schemes, time integrators,
  pressure projection, MMS/self-test/regression suite.
- **GPU (CUDA/Metal):** Jacobi/GS/SOR/CG/BiCGStab/GMRES/MG for restricted linear
  subsets; variable coefficients partial; no mixed/higher-order; integrals/shapes/piecewise
  BC fall back to CPU.
- **Parsing ahead of solving:** `NonlinearDerivativeTerm` parsed into `PDEOperator` but
  not evaluated in relaxation or time-stepping paths.

### GUI — what works today

- Modular `gui_gl/` with panels, handlers, rendering modules, styling system, backend
  capability UI, inspection/comparison tools, command history (undo/redo), registries
  (PDE type, coordinate, solver method), JSON UI config.
- **Docking system** implemented under `gui_gl/docking/` but **`use_docking_ui_ = false`**
  by default; tab-based layout remains primary.

### Documentation drift (must fix in Track 0)

- `README.md` still claims mixed derivatives unsupported.
- `COMPLETED_PHASES.md` marks comparison/undo/inspect integration as pending (integrated).
- `PROJECT_DOCS.md` references missing `PENDING_UPDATES.md` and `PROPOSED_PHASES.md`.
- Root-level planning files moved to `docs/planning/` (update cross-references).

---

## Track 0 — Documentation & hygiene

**Original intent:** Keep `README.md`, `PROJECT_DOCS.md`, and planning docs aligned with
code so agents and contributors do not implement against stale specs.

### WP-0.1 — Fix user-facing docs

| | |
| --- | --- |
| **Status** | ⬜ |
| **Depends on** | — |

**Tasks**

1. Update `README.md` LaTeX support section: document mixed/higher-order/variable-coeff
   support on CPU; list GPU restrictions accurately.
2. Update `README.md` WIP list to reflect docking UI, GPU parity, nonlinear-derivative gap.
3. Refresh `PROJECT_DOCS.md` documentation section to point at `docs/planning/`.
4. Remove references to `PENDING_UPDATES.md` / `PROPOSED_PHASES.md` or replace with
   `MASTER_ROADMAP.md`.
5. Update `docs/planning/COMPLETED_PHASES.md` integration notes (2.4–2.6 are integrated).

**Acceptance criteria**

- No broken doc links to moved planning files.
- README PDE support section matches `include/backend_capability_matrix.h` behavior.
- `PROJECT_DOCS.md` lists `docs/planning/MASTER_ROADMAP.md` as primary plan.

### WP-0.2 — Backend capability doc sync

| | |
| --- | --- |
| **Status** | ⬜ |
| **Depends on** | WP-0.1 |

**Tasks**

1. Ensure `docs/planning/GPU_PARITY_PLAN.md` matrix matches
   `src/backend_capability_matrix.cpp`.
2. Add a one-line pointer from `README.md` backends section to GPU parity plan.

**Acceptance criteria**

- GPU parity plan and runtime gating agree on 3D, mixed, variable-coeff, time-dependent.

---

## Track 1 — Option 1 completion (extended linear FD engine)

**Original intent** ([`PDE_EXPANSION_PLANS.md`](PDE_EXPANSION_PLANS.md) Option 1,
[`PDE_TERMS_IMPLEMENTATION_PLAN.md`](PDE_TERMS_IMPLEMENTATION_PLAN.md)):

Support a broader **linear** PDE class (mixed derivatives, variable coefficients,
higher-order terms, metric-aware forms) with a stable CLI, CPU-first implementation,
explicit backend gating, and regression validation — without breaking existing fast paths.

### Option 1 phase map

| Phase | Description | Current | Target WP |
| --- | --- | --- | --- |
| 0 | Scope & compatibility matrix | Mostly done | WP-0.2 |
| 1 | AST + parsing + validate CLI | Done | — |
| 2 | Coefficient evaluation + file input | Partial | WP-1.1 |
| 3 | Generalized stencil / discretization | Partial | WP-1.2, WP-1.3 |
| 4 | Linear operator runtime | Partial (2D Krylov) | WP-1.4 |
| 5 | CLI/config extensions | Partial | WP-1.1, WP-1.5 |
| 6 | GUI alignment | Partial | WP-5.2 |
| 7 | Validation & testing | Partial | WP-1.6 |

### WP-1.1 — Coefficient file input (`--coeff-file`, `--coeff-map`)

| | |
| --- | --- |
| **Status** | ⬜ |
| **Depends on** | — |
| **Source** | PDE_EXPANSION_PLANS Phase 2; PDE_TERMS Step 2.x |

**Original intent:** Allow spatial coefficient fields from VTK/VTI or CSV grids instead
of only LaTeX expressions, e.g. `a=coeff.vti:field_name`.

**Tasks**

1. Add CLI flags `--coeff-file` and `--coeff-map` to `src/main.cpp` help and parser.
2. Implement loader in new `src/coefficient_field_io.{h,cpp}` (reuse `vtk_io` patterns).
3. Extend `SolveInput` or a sidecar struct with per-term grid coefficient fields.
4. Wire `coefficient_evaluator` / relaxation kernels to prefer grid values when present.
5. Extend `--dump-operator` JSON to note file-sourced coefficients.
6. Document in `README.md`.

**Acceptance criteria**

- Poisson with `a(x,y)` from VTI matches LaTeX-evaluated equivalent within tolerance.
- Unsupported backend emits clear error at validate time.
- Self-test or regression case added.

### WP-1.2 — Stencil builder module (extract + divergence form)

| | |
| --- | --- |
| **Status** | ⬜ |
| **Depends on** | — |
| **Source** | PDE_EXPANSION_PLANS Phase 3 |

**Original intent:** Centralize stencil generation for 1st/2nd/mixed derivatives with
support for non-divergence (`a(x) u_xx`) and divergence (`d/dx(a du/dx)`) forms.

**Tasks**

1. Create `src/operator/stencil_builder.{h,cpp}`; move logic from
   `finite_differences.cpp` + relaxation discretization helpers.
2. Add `--operator-form auto|divergence|nondivergence` CLI flag.
3. Auto-detect form when coefficients are variable (document heuristic).
4. Integrate metric terms from `coordinate_metrics.cpp` for non-Cartesian Laplacian paths.
5. Unit-level tests via regression case (manufactured variable-coeff diffusion).

**Acceptance criteria**

- Divergence vs non-divergence forms produce expected differences on variable `a(x)`.
- No regression in existing constant-coeff MMS cases.
- Cartesian mixed/higher-order paths unchanged or improved (no duplication).

### WP-1.3 — Biharmonic operator (`∇⁴u`)

| | |
| --- | --- |
| **Status** | ⬜ |
| **Depends on** | WP-1.2 |
| **Source** | PDE_TERMS Tier 2 §5 |

**Original intent:** Parse `\nabla^4 u`, `\Delta^2 u` and discretize as
`u_xxxx + 2 u_xxyy + u_yyyy` (2D) with appropriate BC handling.

**Tasks**

1. Extend `latex_parser` / `latex_ast` with biharmonic patterns.
2. Map to fourth-order + mixed fourth-order stencil set in stencil builder.
3. Add MMS plate-equation regression case (clamped or simply supported).
4. Gate Krylov/MG: document symmetric positive-definite requirements.

**Acceptance criteria**

- `\nabla^4 u = 1` on unit square converges with expected order on CPU Jacobi/GS/SOR.
- `--dump-operator` shows decomposed terms.

### WP-1.4 — General `LinearOperator` interface

| | |
| --- | --- |
| **Status** | ⬜ |
| **Depends on** | WP-1.2 |
| **Source** | PDE_EXPANSION_PLANS Phase 4 |

**Original intent:** Matrix-free `Apply(u, out)` for all linear term classes; optional
CSR assembly for preconditioners; shared between CPU Krylov and future GPU matrix-free.

**Tasks**

1. Define `LinearOperator` in `include/linear_operator.h` (2D and 3D specializations).
2. Refactor `BuildLinearOperator2D` into factory from `PDEOperator` + domain.
3. Add optional `AssembleCSR()` behind feature flag for small grids / debugging.
4. Route CPU CG/BiCGStab/GMRES through unified interface.
5. Document unsupported combinations in capability matrix.

**Acceptance criteria**

- Existing Krylov tests pass unchanged.
- Mixed + variable-coeff linear operator applies correctly via matrix-free path.

### WP-1.5 — Structured run summary completeness

| | |
| --- | --- |
| **Status** | ⬜ |
| **Depends on** | WP-1.1 |
| **Source** | PDE_EXPANSION_PLANS Phase 5 |

**Tasks**

1. Ensure `.summary.json` includes parsed operator hash, backend eligibility warnings,
   coefficient source (latex vs file), operator-form used.
2. Add deterministic exit codes for validate-only, unsupported feature, convergence failure.
3. GUI: write same summary sidecar on GUI solves (via `solver_manager`).

**Acceptance criteria**

- Agent/automation can classify run outcome from exit code + summary JSON alone.

### WP-1.6 — Option 1 validation suite expansion

| | |
| --- | --- |
| **Status** | ⬜ |
| **Depends on** | WP-1.1–WP-1.3 |
| **Source** | PDE_EXPANSION_PLANS Phase 7 |

**Tasks**

1. Add manufactured solutions: mixed derivative 2D, variable-coeff 2D/3D, biharmonic 2D.
2. Extend `regression_suite` with golden residuals for new cases.
3. Performance baseline script/note in `tests/README.md` (optional timing thresholds).

**Acceptance criteria**

- `./build/regression_suite` covers all new term classes.
- `./build/pde_sim --self-test` passes on CI hardware.

---

## Track 2 — Nonlinear derivative terms (parse → solve)

**Original intent** ([`PDE_TERMS_IMPLEMENTATION_PLAN.md`](PDE_TERMS_IMPLEMENTATION_PLAN.md)
Tier 1 §3): Enable Burgers-type and nonlinear diffusion PDEs (`u u_x`, `|\nabla u|^2`).

**Gap:** Parser produces `NonlinearDerivativeTerm`; solvers reject or ignore them.

### WP-2.1 — Evaluation kernel (CPU 2D steady)

| | |
| --- | --- |
| **Status** | ⬜ |
| **Depends on** | Track 1 WP-1.2 recommended |
| **Source** | PDE_TERMS Steps 3.3–3.4 |

**Tasks**

1. Add `EvalNonlinearDerivative()` in `src/nonlinear_derivatives.{h,cpp}`.
2. Integrate into `relaxation.cpp` 2D Jacobi/GS/SOR iteration (explicit treatment).
3. Reject or route Krylov/MG with clear message (nonlinear ≠ linear operator).
4. Extend `--dump-operator` output for nonlinear derivative terms (already partial).

**Acceptance criteria**

- Steady Burgers-like elliptic test case or pseudo-steady demo runs on CPU 2D.
- Backend gating prevents silent wrong answers on unsupported methods.

### WP-2.2 — Time-dependent nonlinear derivatives (CPU 2D)

| | |
| --- | --- |
| **Status** | ⬜ |
| **Depends on** | WP-2.1 |
| **Source** | PDE_TERMS Step 3.5 |

**Tasks**

1. Remove blanket reject in `SolvePDETimeSeries` for `nonlinear_derivatives`.
2. Evaluate terms explicitly at current time level (Forward Euler / RK stages).
3. CFL warning for advection-dominated nonlinear terms (`u u_x`).
4. Traveling-wave Burgers MMS or benchmark in self-test.

**Acceptance criteria**

- `u_t + u u_x = ν u_xx` runs on CPU 2D with documented CFL limits.
- Time series summary monitors mass/energy warnings appropriately.

### WP-2.3 — Additional scalar nonlinear functions

| | |
| --- | --- |
| **Status** | ⬜ |
| **Depends on** | — |
| **Source** | PDE_TERMS Tier 2 §6 |

**Tasks**

1. Extend `NonlinearKind` with Tanh, Sinh, Cosh, Ln, Max, Min, ReLU.
2. Parser patterns in `latex_ast.cpp`; evaluation in `residual.cpp` / solver paths.
3. Domain checks for `ln(u)` with user-visible warnings.

**Acceptance criteria**

- `u_t = ∇²u + tanh(u)` parses and runs on CPU fallback path.

---

## Track 3 — GPU parity (Phase 7.1)

**Original intent** ([`GPU_PARITY_PLAN.md`](GPU_PARITY_PLAN.md)): Close the gap between
CPU and CUDA/Metal for the **most-used** operator classes before 3D GPU parity.

### Priority order (from GPU_PARITY_PLAN)

1. 2D diffusion/Poisson parity (constant coefficients)
2. Time-dependent 2D advection-diffusion
3. Variable-coefficient diffusion on GPU
4. Nonlinear reaction terms (scalar, non-derivative) on GPU
5. Piecewise BC + integral terms on GPU

### WP-3.1 — 2D Poisson/diffusion parity audit

| | |
| --- | --- |
| **Status** | ⬜ |
| **Depends on** | WP-0.2 |
| **Source** | GPU_PARITY_PLAN §1 |

**Tasks**

1. Run matched CPU/CUDA/Metal cases; log residual and timing deltas.
2. Fix implicit-shape and spatial-RHS gaps if any remain on GPU 2D.
3. Document results table in `GPU_PARITY_PLAN.md`.

**Acceptance criteria**

- Constant-coeff Poisson 2D: GPU residuals match CPU within float tolerance.

### WP-3.2 — GPU time-dependent 2D advection-diffusion

| | |
| --- | --- |
| **Status** | ⬜ |
| **Depends on** | WP-3.1 |
| **Source** | GPU_PARITY_PLAN §2 |

**Tasks**

1. Wire explicit time stepping on CUDA/Metal for 2D heat + advection.
2. CFL monitoring hooks from `time_integrator.cpp`.
3. Frame output parity with CPU on short series.

**Acceptance criteria**

- Heat + advection 2D time series completes on CUDA and Metal without CPU fallback.

### WP-3.3 — GPU variable-coefficient diffusion

| | |
| --- | --- |
| **Status** | ⬜ |
| **Depends on** | WP-3.1, Track 1 WP-1.1 optional |
| **Source** | GPU_PARITY_PLAN §3 |

**Tasks**

1. Extend CUDA/Metal relaxation kernels for full variable-coeff set (including mixed).
2. Pre-evaluate coefficient fields on CPU when cheaper than per-cell LaTeX eval.
3. Update `backend_capability_matrix.cpp` and GUI capability providers.

**Acceptance criteria**

- `\sin(x) u_xx + u_yy` runs on GPU 2D within tolerance vs CPU.

### WP-3.4 — GPU mixed / higher-order (Phase 7.2 prep)

| | |
| --- | --- |
| **Status** | ⬜ |
| **Depends on** | WP-3.3, Track 1 complete |
| **Source** | GPU_PARITY_PLAN out-of-scope note |

**Tasks**

1. Port mixed-derivative stencils to CUDA/Metal relaxation kernels.
2. Gate Krylov/MG on GPU for expanded operators explicitly.
3. Update README backend limitations.

**Acceptance criteria**

- Mixed-derivative 2D case runs on GPU relaxation methods.

### WP-3.5 — GPU 3D parity (deferred milestone)

| | |
| --- | --- |
| **Status** | ⬜ |
| **Depends on** | WP-3.1–WP-3.4 |
| **Source** | GPU_PARITY_PLAN Phase 7.2+ |

**Tasks:** TBD after 2D parity stable — 3D Jacobi/GS/SOR, then Krylov/MG 3D.

---

## Track 4 — Docking UI productization

**Original intent** ([`DOCKING_SYSTEM_PLAN.md`](DOCKING_SYSTEM_PLAN.md)): VS Code/Blender-style
splittable, draggable panels with layout persistence — replacing fixed tab layout.

**Current:** `gui_gl/docking/*` implemented; `use_docking_ui_ = false` in
`gui_gl/core/application.h`.

### WP-4.1 — View integration audit

| | |
| --- | --- |
| **Status** | ⬜ |
| **Depends on** | — |
| **Source** | DOCKING_SYSTEM_PLAN Phase 3 |

**Tasks**

1. Audit `view_registration.cpp` for placeholder views; wire any missing panels.
2. Verify every `ViewType` in `view_types.h` renders or is intentionally hidden.
3. Fix scroll/state issues via `ViewStateRegistry` per panel.

**Acceptance criteria**

- All preset layouts (`Default`, `Inspection`, `DualViewer`, etc.) render without placeholders.

### WP-4.2 — Layout persistence & session restore

| | |
| --- | --- |
| **Status** | ⬜ |
| **Depends on** | WP-4.1 |
| **Source** | DOCKING_SYSTEM_PLAN Phase 4 |

**Tasks**

1. Auto-save `last_session` layout on clean exit (hook in `Application` destructor/shutdown).
2. Load `last_session` on startup when present; fallback to `CreateDefault()`.
3. Validate JSON on load; corrupt files fall back with toast warning.

**Acceptance criteria**

- Restart restores split ratios and view types.

### WP-4.3 — Default to docking UI

| | |
| --- | --- |
| **Status** | ⬜ |
| **Depends on** | WP-4.1, WP-4.2 |
| **Source** | DOCKING_SYSTEM_PLAN Phase 6 |

**Tasks**

1. Set `use_docking_ui_ = true` by default.
2. Add Preferences toggle “Use classic tab layout” for one release cycle.
3. Update `README.md` GUI section and screenshots guidance.

**Acceptance criteria**

- Fresh install opens docking layout; classic tabs opt-in works.

### WP-4.4 — Floating windows (optional)

| | |
| --- | --- |
| **Status** | ⬜ |
| **Depends on** | WP-4.3 |
| **Source** | DOCKING_SYSTEM_PLAN target architecture |

**Tasks**

1. Implement `FloatingWindows` list in `DockingContext`.
2. Detach/reattach drag gesture from leaf header.
3. Serialize floating window geometry in layout JSON.

**Acceptance criteria**

- Viewer can be dragged out to floating window and restored.

### WP-4.5 — Remove tab-based UI (final)

| | |
| --- | --- |
| **Status** | ⬜ |
| **Depends on** | WP-4.3 + one release of opt-in classic |
| **Source** | DOCKING_SYSTEM_PLAN migration Step 5 |

**Tasks**

1. Delete dead tab-bar code paths from `application.cpp`.
2. Remove `use_docking_ui_` flag once classic layout unused.

**Acceptance criteria**

- `application.cpp` has single layout system; CI builds `pde_gui` cleanly.

---

## Track 5 — GUI Phase 3–4 remainder

**Original intent** ([`GUI_UX_IMPROVEMENTS.md`](GUI_UX_IMPROVEMENTS.md),
[`COMPLETED_PHASES.md`](COMPLETED_PHASES.md)): Future-proof GUI via registries,
config-driven layout, advanced inspection/export.

### Already partially built

- `pde_type_registry`, `coordinate_system_registry`, `solver_method_registry`
- `ui_config` + `default_ui.json`
- `application_state` model
- `statistics_panel`, `convergence_panel`, `animation_export_panel`

### WP-5.1 — Finish Phase 3 registries

| | |
| --- | --- |
| **Status** | ⬜ |
| **Depends on** | — |
| **Source** | GUI_UX §Future-Proofing; COMPLETED_PHASES Phase 3 |

**Tasks**

1. Wire `pde_type_registry` into equation panel (template/metadata-driven hints).
2. Wire `solver_method_registry` into compute panel (method recommendations).
3. Enable commented `UIHelp` / `UIAccessibility` sections in `ui_style.cpp`.
4. Add Preferences panel for accessibility settings.

**Acceptance criteria**

- F1 help covers equation, domain, solver, boundary panels.
- Method dropdown shows recommendation tooltip when applicable.

### WP-5.2 — GUI backend / operator warnings (Option 1 Phase 6)

| | |
| --- | --- |
| **Status** | ⬜ |
| **Depends on** | Track 1 |
| **Source** | PDE_EXPANSION_PLANS Phase 6 |

**Tasks**

1. On PDE parse, call `BackendSupportsInput` for selected backend; show inline warnings.
2. Surface `--dump-operator`-equivalent preview in equation panel (collapsible).
3. Persist operator metadata in GUI export directory.

**Acceptance criteria**

- User sees GPU incompatibility before pressing Solve for unsupported PDE.

### WP-5.3 — Advanced visualization (Phase 4)

| | |
| --- | --- |
| **Status** | ⬜ |
| **Depends on** | WP-5.1 |
| **Source** | GUI_UX Inspection §2–3, Phase 4 |

**Tasks**

1. Vector glyph rendering for gradient/flux fields (`glyph_renderer` integration).
2. Histogram / distribution panel in `statistics_panel`.
3. Complete `animation_export_panel` (MP4/GIF or frame sequence + ffmpeg docs).
4. Custom colormap picker in color preferences.

**Acceptance criteria**

- User can export animation of time series from GUI.
- Gradient field displays arrow glyphs in viewer.

### WP-5.4 — GlViewer modular integration cleanup

| | |
| --- | --- |
| **Status** | ⬜ |
| **Depends on** | — |
| **Source** | PROJECT_DOCS rendering note |

**Tasks**

1. Finish migrating remaining logic from `GlViewer.cpp` into `rendering/*` modules.
2. Reduce `GlViewer.cpp` to orchestration-only (~500 lines target).
3. Update `PROJECT_DOCS.md` to remove stale PENDING_UPDATES reference.

**Acceptance criteria**

- No duplicate mesh/shader logic between GlViewer and rendering modules.

---

## Track 6 — Option 2 (nonlinear / conservation-law pipeline)

**Original intent** ([`PDE_EXPANSION_PLANS.md`](PDE_EXPANSION_PLANS.md) Option 2):
Residual form `R(u)=0`, finite-volume fluxes, Riemann solvers, Newton-Krylov, IMEX,
CFL-driven adaptive stepping — CPU-first with GPU explicit FV later.

**Existing adjacent code:** `advection.cpp`, `time_integrator.cpp`, `pressure_projection.cpp`,
`coupled_solver.cpp` — **not** the planned unified nonlinear pipeline.

### WP-6.1 — Residual-form problem definition

| | |
| --- | --- |
| **Status** | ✅ MVP |
| **Depends on** | Track 2 |
| **Source** | PDE_EXPANSION Option 2 Phase A |

**Tasks**

1. Add `ResidualOperator` type alongside `PDEOperator` for nonlinear PDEs.
2. Parser mode or auto-detection for nonlinear products.
3. `--validate` reports residual form vs linear operator form.

**Acceptance criteria**

- Burgers conservation form parsed as residual, not silently linearized.

### WP-6.2 — FV discretization path (1D → 2D)

| | |
| --- | --- |
| **Status** | ✅ MVP (1D/2D scalar, CPU) |
| **Depends on** | WP-6.1 |
| **Source** | Option 2 Phase B |

**Tasks**

1. Create `src/fv/` with flux interfaces, Lax-Friedrichs/HLL, minmod/MC limiters.
2. CLI `--discretization fd|fv` for hyperbolic conservation laws.
3. Shock tube 1D regression benchmark.

**Acceptance criteria**

- Sod shock tube matches reference solution profile at fixed time.

### WP-6.3 — Newton-Krylov + IMEX time integration

| | |
| --- | --- |
| **Status** | 🔄 MVP (diagonal Newton + IMEX in `SolvePDETimeSeries`) |
| **Depends on** | WP-6.2, WP-1.4 |
| **Source** | Option 2 Phase C |

**Tasks**

1. Newton loop with line search; JVP via finite differences or automatic linearization.
2. IMEX split: stiff diffusion implicit, advection explicit.
3. Structured JSON status codes for Newton/CFL failure.

**Acceptance criteria**

- Reaction-diffusion stiff case completes with IMEX without blow-up.

---

## Track 7 — Option 3 (FEM / unstructured)

**Original intent** ([`PDE_EXPANSION_PLANS.md`](PDE_EXPANSION_PLANS.md) Option 3):
Weak-form assembly on unstructured meshes (Gmsh/VTK), external solver integration,
AMR hooks — long-horizon.

**Current:** `mesh_io` done; `unstructured_solver` stub only.

### WP-7.1 — FE mesh data structures

| | |
| --- | --- |
| **Status** | ⬜ |
| **Depends on** | — |
| **Source** | Option 3 Phase A–B |

**Tasks**

1. Topology utilities: nodes, elements, boundary markers from `mesh_io` output.
2. P1 basis + quadrature on simplices.
3. Local element stiffness assembly for Poisson.

**Acceptance criteria**

- Assembly matrix matches FD Poisson on structured triangle mesh.

### WP-7.2 — Linear solve + VTK output

| | |
| --- | --- |
| **Status** | ⬜ |
| **Depends on** | WP-7.1 |
| **Source** | Option 3 Phase C, E |

**Tasks**

1. CG on sparse CSR from FE assembly (CPU).
2. Wire `--mesh-solve` to real solver instead of stub error.
3. Write solution on unstructured VTK.

**Acceptance criteria**

- `./build/pde_sim --mesh ... --mesh-solve` solves Poisson on simple mesh.

---

## Track 8 — Tiers 3–4 specialized terms

**Original intent** ([`PDE_TERMS_IMPLEMENTATION_PLAN.md`](PDE_TERMS_IMPLEMENTATION_PLAN.md)
Tiers 3–4): Integro-differential kernels, vector PDEs, boundary integrals, fractional,
stochastic, delay terms — research-grade extensions.

**Recommendation:** Do not start until Track 1–2 complete. Each tier item becomes its
own WP when prioritized.

| Item | Tier | Suggested WP id | Notes |
| --- | --- | --- | --- |
| Integro-differential kernels | 3 | WP-8.1 | Extend `IntegralTerm` beyond global ∫u |
| Vector / system PDEs | 3 | WP-8.2 | Partial: multi-field coupling exists |
| Boundary integral terms | 3 | WP-8.3 | Requires boundary mesh |
| Fractional derivatives | 4 | WP-8.4 | Research + memory cost |
| Stochastic terms | 4 | WP-8.5 | RNG + ensemble stepping |
| Delay terms | 4 | WP-8.6 | Solution history buffer |
| Time-fractional / distributed order | 4 | WP-8.7 | Extends WP-8.4 |

---

## Cross-cutting engineering rules

From [`PDE_EXPANSION_PLANS.md`](PDE_EXPANSION_PLANS.md) shared considerations — apply to
**every** track:

1. **Backend gating:** Unsupported features fail at `--validate` with actionable messages.
2. **Structured summaries:** JSON run reports include status, warnings, timing, residuals.
3. **Exit codes:** Deterministic codes for automation/agent workflows.
4. **Regression first:** No feature merges without self-test or regression_suite coverage.
5. **Backward compatibility:** Existing Poisson/heat/wave examples unchanged within tolerance.
6. **Minimal scope:** Prefer extending `stencil_builder` / registries over parallel systems.

---

## Suggested execution order (first sprint)

For systematic progress on the audit issues, run work packages in this order:

```
WP-0.1 → WP-0.2
WP-1.1 → WP-1.2 → WP-1.6
WP-2.1 → WP-2.2
WP-3.1 → WP-3.2 → WP-3.3
WP-4.1 → WP-4.2 → WP-4.3
WP-5.2 (parallel with Track 1 closeout)
```

---

## Document maintenance

When completing a work package:

1. Update status in the executive summary table and the WP block.
2. Add a dated one-line note under **Changelog** below.
3. If user-visible behavior changed, update `README.md` in the same PR.

### Changelog

| Date | Change |
| --- | --- |
| 2026-05-27 | Initial master roadmap; planning files consolidated under `docs/planning/` |
