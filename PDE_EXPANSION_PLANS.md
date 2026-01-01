# PDE Expansion Plans (Options 1, 2, and 3)

This document records an in-depth implementation plan for Option 1 and extended
plans for Options 2 and 3. It is intended to be a continuity reference for
future development and parallel agent work.

--------------------------------------------------------------------------------

## Option 1: Extended Finite-Difference Operator Engine (Linear + Variable Coefficients)

### Goals
- Support a broader linear PDE class: mixed derivatives (u_xy, u_xz, u_yz),
  spatially and time-varying coefficients, higher-order linear derivatives, and
  non-Cartesian metric-aware forms.
- Maintain a stable CLI workflow that is fully automatable.
- Preserve existing fast paths for the current subset while expanding capability.

### Core Deliverables
- PDE AST and operator model (linear terms, coefficient expressions, RHS).
- Generalized stencil/linear-operator engine with boundary integration.
- CPU-first implementation with optional GPU matrix-free or limited support.
- Validation suite with manufactured solutions and regression tests.

### Phase 0: Scope and Compatibility Matrix
1) Define the expanded PDE grammar:
   - Allowed terms (derivative order, mixed terms, coefficient forms).
   - Required form (linear only, nonlinearity excluded in Option 1).
   - Coordinate system mapping and metric-aware terms.
2) Create a compatibility matrix:
   - CPU: full support.
   - CUDA/Metal: limited support (2D only, reduced term set) with clear gating.
3) Update CLI help text with explicit supported term list.

### Phase 1: PDE AST and Parsing Extensions
1) Replace or extend `PDECoefficients` with a structured operator representation:
   - New types in `include/pde_types.h`:
     - `PDETerm` (derivative multi-index, coefficient expression, term weight).
     - `PDEOperator` (vector of terms, RHS expression, metadata).
2) Extend LaTeX parsing pipeline:
   - Update `src/latex_lexer.cpp`, `src/latex_ast.cpp`, `src/latex_parser.cpp`.
   - Parse mixed derivatives (u_xy, u_xz, u_yz).
   - Parse coefficient expressions (a(x,y,t) u_xx).
   - Record derivative order and index (x,y,z,t).
3) Provide a canonical term representation:
   - Normalize equivalent terms (u_xy == u_yx).
   - Normalize constant vs expression coefficients.
4) Add CLI "validate only" path:
   - `--validate` prints parsed operator and backend eligibility.

### Phase 2: Expression Evaluation and Coefficient Caching
1) Build a coefficient evaluator:
   - New module: `src/coefficient_evaluator.{h,cpp}`.
   - Parse LaTeX expressions into a compact expression tree.
   - Evaluate at (x,y,z,t) with caching for repeated use.
2) Add coefficient field caching:
   - Precompute coefficient fields when `a(x,y,t)` is time-independent.
   - Use lightweight cache keyed by term id and grid resolution.
3) Add data-source support for coefficients:
   - CLI: `--coeff-file` accepts a VTK/VTI or CSV grid for coefficients.
   - Optional mapping syntax: `a=coeff.vti:field_name`.

### Phase 3: Discretization and Stencil Builder
1) Implement a generalized stencil generator:
   - New module: `src/operator/stencil_builder.{h,cpp}`.
   - Support 1st, 2nd, mixed derivatives using centered differences.
   - Allow variable coefficient forms in:
     - Non-divergence form: a(x) u_xx.
     - Divergence form: d/dx (a(x) d/dx u).
2) Boundary integration:
   - Centralized boundary handler for Dirichlet/Neumann/Robin.
   - Support spatially varying BC expressions.
   - Fallback to one-sided stencils at boundaries.
3) Coordinate metric integration:
   - Integrate metric terms from `src/coordinate_metrics.cpp`.
   - Ensure operator builder uses metric-aware derivative computations.

### Phase 4: Linear Operator Runtime
1) Introduce a `LinearOperator` interface:
   - `Apply(u, out)` for matrix-free iteration.
   - `AssembleCSR()` optional for sparse solves or preconditioners.
2) CPU path:
   - Matrix-free Jacobi/GS/SOR for quick iteration.
   - CSR assembly for CG/GMRES/BiCGStab/MG when needed.
3) GPU path (optional, staged):
   - Matrix-free only for a restricted term set.
   - Clearly gate unsupported terms and fall back to CPU.

### Phase 5: CLI and Config Extensions
1) CLI surface:
   - Add `--operator-form` (divergence, nondivergence, auto).
   - Add `--validate` and `--dump-operator`.
   - Add `--coeff-file` and `--coeff-map`.
2) Configuration output:
   - Write a JSON run summary: parsed operator, backend chosen,
     convergence status, final residuals, timing, and warnings.
3) Input validation:
   - Fail fast on unsupported PDE features for the selected backend.
   - Emit actionable error messages for the user/agent.

### Phase 6: UI Alignment (Optional)
1) GUI updates:
   - Surface coefficient expressions in the PDE preview.
   - Add warnings for GPU incompatibility.
2) Export:
   - Persist operator metadata in the output directory alongside VTK files.

### Phase 7: Validation and Testing
1) Manufactured solutions:
   - Mixed derivatives and variable coefficient tests in 2D and 3D.
   - Coordinate system tests with metric terms.
2) Regression suite:
   - Ensure existing PDEs reproduce previous results within tolerance.
3) Performance checks:
   - Measure solver runtime against current baseline.
   - Identify the cost of coefficient evaluation and caching.

### Milestones
1) Parser and operator model complete; validate-only CLI path works.
2) CPU operator engine supports mixed derivatives and variable coefficients.
3) Full CLI workflow (solve + export + summary) for expanded linear PDEs.
4) Optional GPU matrix-free support for a constrained operator set.

--------------------------------------------------------------------------------

## Option 2: Nonlinear PDE and Conservation-Law Pipeline (FV + Newton-Krylov)

### Goals
- Support nonlinear derivative terms and conservation laws.
- Add robust time integration (implicit/IMEX) and stability controls.
- Provide automated failure detection and solver fallback.

### Extended Plan (Phased)
Phase A: Nonlinear PDE representation
- Replace linear operator-only model with residual-form `R(u) = 0`.
- Extend parser to detect nonlinear products (u u_x, |grad u|^2).
- Add nonlinear term registry and evaluator.

Phase B: Discretization for conservation laws
- Add a finite-volume (FV) path with flux interfaces.
- Implement Riemann solvers (Lax-Friedrichs, HLL) and limiters (minmod, MC).
- Support boundary types for hyperbolic systems (inflow/outflow).

Phase C: Nonlinear solvers and time integration
- Newton-Krylov framework with line search and trust-region options.
- IMEX and adaptive time-step control with CFL monitoring.
- Automatic Jacobian-vector products using analytic or finite-diff Jv.

Phase D: Backend gating and mixed modes
- CPU-first implementation with validation.
- GPU support for explicit FV kernels (restrictive but fast).
- Auto fallback to CPU for stiff or unsupported cases.

Phase E: CLI automation features
- Add solver status codes for convergence, CFL failure, or divergence.
- Output structured JSON summaries for agent workflows.
- Add "dry-run" compatibility and stability checks.

Phase F: Validation suite
- Burgers, reaction-diffusion, and standard shock tube problems.
- Compare against reference solutions or benchmark datasets.

--------------------------------------------------------------------------------

## Option 3: Weak-Form / Finite-Element Track (General PDEs + Geometry)

### Goals
- Support broader PDEs on complex geometries via FEM.
- Enable unstructured meshes and higher-order elements.
- Integrate scalable solvers for large systems.

### Extended Plan (Phased)
Phase A: Mesh and geometry ingestion
- Add mesh readers (Gmsh, VTK unstructured).
- Define mesh data structures and topology utilities.

Phase B: Weak-form DSL and assembly
- Introduce a weak-form expression system.
- Implement basis functions (P1, P2) and quadrature.
- Assemble global system matrices and load vectors.

Phase C: Solver integration
- Integrate external solvers (PETSc or Trilinos) with preconditioners.
- Provide linear and nonlinear solve pipelines.
- Implement adaptive mesh refinement (AMR) hooks.

Phase D: Boundary and interface conditions
- Implement strong and weak BCs (Dirichlet, Neumann, Robin).
- Support mixed systems (u, v, p) for multiphysics.

Phase E: CLI and output
- Extend CLI to accept mesh files, function-space config, and FE options.
- Output VTK with field metadata and solution diagnostics.

Phase F: Validation and performance
- Benchmark against Poisson, heat, Stokes, and elasticity tests.
- Validate convergence rates with mesh refinement studies.

--------------------------------------------------------------------------------

## Shared Engineering Considerations
- Compatibility gating: explicit backend restrictions in all modes.
- Structured summaries for automation: JSON run report with status and metrics.
- Clear error taxonomy and exit codes for agentic workflows.
- Regression testing to preserve current solver behavior.
