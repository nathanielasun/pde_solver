# PDE Simulator (LaTeX -> 2D/3D Finite Differences -> VTK)

This repository contains a C++ solver for a strict subset of 2D/3D PDEs written in LaTeX and a GPU-accelerated OpenGL GUI for authoring and visualizing solutions. Results are stored as legacy VTK `STRUCTURED_POINTS` grids or modern VTK XML ImageData (`.vti`) files. We have chosen VTK files over HDF5 or VDB for simplicity. Time-dependent PDEs with `u_t` or `u_tt` are supported via explicit time stepping with checkpoint/restart capabilities (still a work in progress).

## Project Structure

- `src/`: Core solver, backend selection, coordinate metrics, boundary handling, and VTK I/O
- `include/`: Public headers for solver, backends, and data types
- `gui_gl/`: Primary OpenGL GUI with modular architecture:
  - `core/`: Application initialization, window management, event handling
  - `panels/`: UI panels grouped by tab (`main/`, `inspect/`, `preferences/`) for equation/domain/grid/boundary/solver/time, inspection tools, and preferences
  - `components/`: Reusable UI components (log panel, inspection tools, comparison tools)
  - `rendering/`: OpenGL rendering modules (shaders, mesh/grid builders, coordinate transforms, projection)
  - `handlers/`: Event handlers (file operations, solver execution)
  - `systems/`: System integrations (backend capabilities, command history, UI config)
  - `utils/`: Utility functions (string, math, path, coordinate helpers)
  - `styles/`: Professional styling system (spacing, typography, animations)
  - `io/`: File I/O operations (preferences, image export, VTK loading)
  - `tools/`: Analysis tools (comparison, field sampling)
  - `solver/`: Solver integration and thread management
  - `models/`: Data models (application state)
- `third_party/`: External dependencies (Dear ImGui, GLFW, stb_image)
- `tools/`: LaTeX preview renderer for the GUI (Python)
- `build/`: CMake build outputs (generated)

For detailed file-level documentation, see `PROJECT_DOCS.md`. For an overview of
the module boundaries, see `ARCHITECTURE.md`.

## Implementation Status

- **Completed Phases**: See `COMPLETED_PHASES.md` for a consolidated summary of all completed work (Phase 1 and Phase 2 GUI improvements).
- **Ongoing Work**: See `PENDING_UPDATES.md` for in-progress refactoring and partially-completed features.
- **Proposed Work**: See `PROPOSED_PHASES.md` for all proposed tasks, organized for parallel agent assignment.
- **High-Impact Extensions Plan**: See `HIGH_IMPACT_PLAN.md` for a detailed, phased roadmap to expand academic and personal project value.

## BUILDING INSTRUCTIONS - FOLLOW CAREFULLY

Navigate to be in the pde_solver directory and run the following in terminal
If the build fails, open a new issue in the public repository and/or send me an email at 
`nathanielasun@gmail.com`

```bash
cmake -S . -B build #new build
cmake --build build
```

## GUI (OpenGL / ImGui) (to build the GUI exclusively)

```bash
cmake --build build --target pde_gui
./build/pde_gui
```

OpenGL GUI requirements:
- OpenGL 3.3-capable GPU/driver (GL 3.3 path preserved for broad coverage).

The viewer overlays axis tick labels when "Show domain grid" is enabled.

UI layout/theme is driven by `gui_gl/config/default_ui.json` with optional
`ui_config.json` overrides. The UI Configuration panel supports setting
`theme.font_path` to a Unicode-capable font file (TTF/OTF/TTC) for Greek/math symbols.
Local fonts live under `gui_gl/styles/fonts/`, and relative font paths resolve against that directory.
Boundary templates initialize all six faces; in 2D, front/back default to Dirichlet `0` and the UI
clamps invalid boundary kinds to a safe default.

## CLI (pde_sim)

### Usage

```bash
./build/pde_sim \
  --pde "<latex>" \
  --domain xmin,xmax,ymin,ymax[,zmin,zmax] \
  --grid nx,ny[,nz] \
  --bc "left:dirichlet:0;right:neumann:0;bottom:robin:alpha=1,beta=1,gamma=0;top:dirichlet:0;front:dirichlet:0;back:dirichlet:0" \
  --out output.vtk
```

Options:
- `--pde` LaTeX PDE string (see formats below).
- `--domain` `xmin,xmax,ymin,ymax[,zmin,zmax]`.
- `--grid` `nx,ny[,nz]` (grid resolution).
- `--bc` boundary specification string; unspecified sides default to `dirichlet:0`.
- `--shape` or `--domain-shape` implicit domain function `f(x,y[,z]) <= 0`.
- `--shape-file` load implicit domain expression from a file.
- `--shape-mask` load a VTK mask for implicit domains (sampled scalar field).
- `--shape-mask-threshold` mask threshold (inside if value <= threshold).
- `--shape-mask-invert` invert mask inside/outside.
- `--shape-offset` `x,y[,z]` and `--shape-scale` `x,y[,z]` (transform for shape/mask alignment).
- `--mesh` load an unstructured mesh (`.vtk` legacy UNSTRUCTURED_GRID or ASCII `.msh`).
- `--mesh-format` override mesh format (`vtk` or `msh`).
- `--mesh-discretization` unstructured solver stub (`fe` or `fv`).
- `--mesh-solve` attempt the unstructured solver stub (currently returns a stub error).
- `--out` output file (`.vtk` or `.vti`).
- `--out-dir` directory for random output name (supports naming tokens; see Batch Runs). Defaults
  to `outputs/` next to the executable if neither `--out` nor `--out-dir` is set.
- `--in-dir` load and summarize `.vtk` files from a directory.
- `--config` load a JSON run configuration file.
- `--export-config` write a JSON run configuration file for the current run.
- `--batch` run a batch JSON spec for sweeps.
- `--backend` `auto|cpu|cuda|metal|tpu`.
- `--method` `jacobi|gs|sor|cg|bicgstab|gmres|mg` (solver method, default: jacobi).
- `--omega` SOR relaxation parameter (default: 1.5).
- `--gmres-restart` GMRES restart parameter (default: 30).
- `--residual-interval` N (print residual every N iterations, 0 = off).
- `--self-test` run the built-in regression suite of analytic test cases.
- `--mms` override RHS/BCs with a manufactured solution and report error norms (steady-state, constant coefficients).
- `--convergence` grid sweep for error norms (uses MMS, no VTK output). Use a list like `17,33,65` (square/cubic grids) or explicit grids separated by `;` such as `33,33;65,65`.
- `--convergence-out` `<file>` CSV output path (default: `out_dir/convergence.csv`).
- `--convergence-plot` `[file]` gnuplot script path (default: `convergence.gp` next to the CSV).
- `--convergence-json` `[file]` JSON output path (default: `convergence.json` next to the CSV).
- `--max-iter`, `--tol`, `--threads` (`0` = auto).
- `--time` `t_start,t_end,dt,frames` (enable time-dependent solve).
- `--checkpoint` `<file>` (write checkpoint after each frame).
- `--restart` `<file>` (resume from checkpoint).
- `--format` `vtk|vti` (output format, default: vtk).
- `--validate` parse inputs and exit without solving.
- `--dump-operator[=json]` print parsed operator summary (text by default, JSON when `=json`, includes `schema_version`).
- `--dump-metadata[=<file>]` print run metadata JSON (or write to file).
- `--dump-summary[=<file>]` print run summary JSON (or write to file).
- `--dataset-index <dir>` build a dataset index and summary stats.
- `--dataset-index-out <file>` override dataset index output path (use `-` for stdout).
- `--dataset-cleanup <dir>` remove orphaned summary/metadata sidecars.
- `--dataset-cleanup-dry-run` show what cleanup would remove without deleting.
- `--dataset-cleanup-empty-dirs` remove empty directories after cleanup.
- `--metal-reduce-interval` N, `--metal-threadgroup` X,Y (Metal-specific tuning).
- Variable coefficient terms (e.g., `\sin(x) u_{xx}`) are supported on CPU Jacobi/Gauss-Seidel/SOR and CPU BiCGStab/GMRES. CG, Multigrid, and GPU backends require constant coefficients.

Checkpoint files are versioned and written with full double precision; v2 includes
velocity data for `u_tt` time stepping to enable bitwise CPU restarts.

When `--mesh` is used, the CLI runs in mesh mode: it loads and summarizes the unstructured
mesh, and writes a legacy VTK unstructured grid preview to `--out`/`--out-dir` (forced to `.vtk`).
Only ASCII VTK legacy and ASCII Gmsh `.msh` are supported; unstructured solves remain a
separate stub pipeline for now. Output naming tokens are not expanded for mesh previews.

### Run Config (JSON)

Use `--export-config` to write a JSON run specification for the current CLI arguments, or save from the GUI via File > Save Run Config. Load the same file with `--config` or File > Load Run Config; the spec stores PDE, domain/grid/BCs, solver settings, backend selection, time settings, and output options (`schema_version: 1`).

Each solve also writes metadata (`.meta.json`) and summary (`.summary.json`) sidecars next to outputs. Metadata includes the run spec, build info, platform/backends, and timestamps; summaries include residual history, convergence stats, and timing. Use `--dump-metadata` or `--dump-summary` to print or export these JSON logs.

For mesh-only runs, use `domain.mesh` in the config (with optional `domain.mesh_format`
and `domain.mesh_discretization`) and omit `domain.bounds`/`domain.grid`.

### Batch Runs (JSON)

Use `--batch <file>` to execute multiple run configs (sweeps). Each run entry can be a config
file path or an inline run spec. Relative paths inside the batch file are resolved relative to
the batch file's directory. A manifest JSON/CSV is written next to the batch file by default
(`*_manifest.json` and `*_manifest.csv`).

```json
{
  "schema_version": 1,
  "output": {
    "dir": "outputs/{name}",
    "format": "vti"
  },
  "runs": [
    { "name": "poisson_64", "config": "configs/poisson_64.json" },
    { "name": "poisson_128", "config": "configs/poisson_128.json" }
  ],
  "manifest": {
    "json": "outputs/batch_manifest.json",
    "csv": "outputs/batch_manifest.csv"
  }
}
```

Output naming patterns (for `--out-dir`, batch `output.dir`, or batch `output.path`) support:
`{index}` (1-based), `{index0}`, `{name}`, `{grid}`, `{nx}`, `{ny}`, `{nz}`, `{backend}`,
`{method}`, `{format}`, `{tag}`, `{timestamp}`. Collisions append `_2`, `_3`, etc.

### Solver Methods

The solver supports multiple iterative methods:

- **Jacobi**: Basic point-wise iteration (default, available on all backends).
- **Gauss-Seidel (gs)**: Sequential updates using latest values (CPU, CUDA, Metal).
- **SOR (sor)**: Successive Over-Relaxation with configurable omega (CPU, CUDA, Metal).
- **CG (cg)**: Conjugate Gradient for symmetric positive-definite systems (CPU, CUDA, Metal).
- **BiCGStab (bicgstab)**: Bi-Conjugate Gradient Stabilized for general systems (CPU, CUDA, Metal).
- **GMRES (gmres)**: Generalized Minimal Residual with restart (CPU, CUDA, Metal).
- **Multigrid V-cycle (mg)**: Multigrid V-cycle for fast convergence (CPU, CUDA, Metal).

**Note**: Krylov methods (CG, BiCGStab, GMRES) and Multigrid require linear PDEs with Dirichlet-only boundary conditions and no nonlinear terms. GPU implementations are limited to 2D domains for Krylov/MG methods.
Mixed and higher-order spatial derivatives (u_xy/u_xz/u_yz, u_xxx/u_yyyy/u_zzzz, etc.) are supported on CPU Jacobi/Gauss-Seidel/SOR and CPU BiCGStab/GMRES in Cartesian coordinates only. CG and Multigrid require symmetric second-order operators.
Higher-order derivatives use 5-point stencils, so affected dimensions must have at least 5 grid points.

### Examples

#### Steady-State Examples

Laplace / Poisson:
```bash
./build/pde_sim \
  --pde "-u_{xx} - u_{yy} = 1" \
  --domain 0,1,0,1 \
  --grid 64,64 \
  --bc "left:dirichlet:0;right:dirichlet:0;bottom:dirichlet:0;top:dirichlet:0" \
  --method cg \
  --tol 1e-6 \
  --out outputs/poisson.vtk
```

Manufactured solution (MMS) check:
```bash
./build/pde_sim \
  --mms \
  --pde "-u_{xx} - u_{yy}" \
  --domain 0,1,0,1 \
  --grid 33,33 \
  --method cg \
  --tol 1e-10 \
  --out outputs/mms.vtk
```

#### Unstructured Mesh Preview

Convert a Gmsh mesh to legacy VTK unstructured grid for inspection:
```bash
./build/pde_sim \
  --mesh meshes/example.msh \
  --out outputs/example_mesh.vtk
```

Convergence study (MMS):
```bash
./build/pde_sim \
  --convergence 17,33,65 \
  --pde "-u_{xx} - u_{yy}" \
  --domain 0,1,0,1 \
  --grid 17,17 \
  --method cg \
  --tol 1e-10 \
  --out-dir outputs
```

Convection-diffusion with spatial RHS:
```bash
./build/pde_sim \
  --pde "2 u_{xx} + 2 u_{yy} - 0.5 u_x + 3 u = \sin(\pi x) \cos(\pi y)" \
  --domain -1,1,-1,1 \
  --grid 80,80 \
  --bc "left:dirichlet:1;right:dirichlet:0;bottom:neumann:0;top:neumann:0" \
  --method sor \
  --omega 1.8 \
  --out outputs/convdiff.vtk
```

3D (Cartesian) with Multigrid:
```bash
./build/pde_sim \
  --pde "-u_{xx} - u_{yy} - u_{zz} + u = 0" \
  --domain 0,1,0,1,0,1 \
  --grid 32,32,32 \
  --bc "left:dirichlet:0;right:dirichlet:0;bottom:dirichlet:0;top:dirichlet:0;front:dirichlet:0;back:dirichlet:0" \
  --method mg \
  --out outputs/box3d.vtk
```

Implicit domain shape (disk) with embedded boundary:
```bash
./build/pde_sim \
  --pde "u_{xx} + u_{yy} = 0" \
  --domain -1,1,-1,1 \
  --grid 128,128 \
  --shape "x^2 + y^2 - 1" \
  --bc "left:dirichlet:0;right:dirichlet:0;bottom:dirichlet:0;top:dirichlet:0" \
  --out outputs/disk.vtk
```
Note: Implicit shapes use subcell sampling and cut-cell ghost values to better capture curved boundaries.

Implicit domain from VTK mask (sampled scalar field):
```bash
./build/pde_sim \
  --pde "u_{xx} + u_{yy} = 0" \
  --domain -1,1,-1,1 \
  --grid 128,128 \
  --shape-mask path/to/mask.vtk \
  --shape-mask-threshold 0.0 \
  --shape-offset 0,0 \
  --shape-scale 1,1 \
  --bc "left:dirichlet:0;right:dirichlet:0;bottom:dirichlet:0;top:dirichlet:0" \
  --out outputs/mask_domain.vtk
```
Note: Mask values are treated as an implicit function; use `--shape-mask-invert` for binary masks where inside = 1.

**Note**: Coordinate system selection (polar, cylindrical, spherical, etc.) is currently available in the GUI only. The CLI uses Cartesian coordinates by default. For non-Cartesian coordinate systems, use the GUI.

Spatial boundary conditions:
```bash
./build/pde_sim \
  --pde "-u_{xx} - u_{yy} = 0" \
  --domain 0,1,0,1 \
  --grid 64,64 \
  --bc "left:dirichlet:0;right:dirichlet:1+0.5*y;bottom:dirichlet:0;top:dirichlet:\sin(\pi x)" \
  --out outputs/spatial_bc.vtk
```

Piecewise boundary conditions:
```bash
./build/pde_sim \
  --pde "-u_{xx} - u_{yy} = 0" \
  --domain 0,1,0,1 \
  --grid 64,64 \
  --bc "left:dirichlet:0;right:piecewise:y<0.5:dirichlet:1;y>=0.5:dirichlet:0;bottom:dirichlet:0;top:dirichlet:0" \
  --out outputs/piecewise_bc.vtk
```

Global integral term (CPU fallback):
```bash
./build/pde_sim \
  --pde "u_{xx} + u_{yy} + 0.25 \int u \, dx dy = 0" \
  --domain 0,1,0,1 \
  --grid 64,64 \
  --bc "left:dirichlet:0;right:dirichlet:0;bottom:dirichlet:0;top:dirichlet:0" \
  --out outputs/integral.vtk
```

Nonlinear reaction term (CPU fallback):
```bash
./build/pde_sim \
  --pde "u_{xx} + u_{yy} + u^2 = 0" \
  --domain 0,1,0,1 \
  --grid 64,64 \
  --bc "left:dirichlet:0;right:dirichlet:0;bottom:dirichlet:0;top:dirichlet:0" \
  --out outputs/nonlinear.vtk
```

#### Time-Dependent Examples

Heat equation (first-order in time):
```bash
./build/pde_sim \
  --pde "u_t - u_{xx} - u_{yy} = 0" \
  --domain 0,1,0,1 \
  --grid 64,64 \
  --bc "left:dirichlet:0;right:dirichlet:0;bottom:dirichlet:0;top:dirichlet:0" \
  --time 0,1,0.001,100 \
  --out outputs/heat_series \
  --format vti \
  --checkpoint outputs/heat_checkpoint.txt
```

Wave equation (second-order in time):
```bash
./build/pde_sim \
  --pde "u_{tt} - u_{xx} - u_{yy} = 0" \
  --domain 0,1,0,1 \
  --grid 128,128 \
  --bc "left:dirichlet:0;right:dirichlet:0;bottom:dirichlet:0;top:dirichlet:0" \
  --time 0,2,0.0005,400 \
  --out outputs/wave_series \
  --format vti
```

Time-dependent with spatial RHS:
```bash
./build/pde_sim \
  --pde "u_t - u_{xx} - u_{yy} = \sin(\pi x) \cos(\pi y) \exp(-t)" \
  --domain 0,1,0,1 \
  --grid 64,64 \
  --bc "left:dirichlet:0;right:dirichlet:0;bottom:dirichlet:0;top:dirichlet:0" \
  --time 0,1,0.01,100 \
  --out outputs/timedep_rhs_series \
  --format vti
```

Restart from checkpoint:
```bash
./build/pde_sim \
  --pde "u_t - u_{xx} - u_{yy} = 0" \
  --domain 0,1,0,1 \
  --grid 64,64 \
  --bc "left:dirichlet:0;right:dirichlet:0;bottom:dirichlet:0;top:dirichlet:0" \
  --time 0,2,0.001,200 \
  --restart outputs/heat_checkpoint.txt \
  --out outputs/heat_restarted \
  --format vti
```

### Boundary Conditions

Format for each side:
- `dirichlet:<expr>` - Dirichlet boundary condition: `u = <expr>`
- `neumann:<expr>` - Neumann boundary condition: `∂u/∂n = <expr>`
- `robin:alpha=<expr>,beta=<expr>,gamma=<expr>` - Robin boundary condition: `alpha*u + beta*∂u/∂n = gamma`
- `piecewise:<condition1>:<bc1>;<condition2>:<bc2>;...;default:<bc_default>` - Piecewise boundary conditions

Expressions can be:
- **Constants**: `0`, `1`, `-2.5`
- **Linear in coordinates**: `1+0.5*x-2*y+z`
- **General expressions**: `\sin(\pi x)`, `\cos(2*y)`, `x^2 + y^2`, `\exp(-x)`
- **Time-dependent** (for time-dependent PDEs): `\sin(t)`, `t^2`, `\exp(-t)`

Examples:
- `dirichlet:1+0.5*x` - Linear variation along boundary
- `neumann:\sin(\pi y)` - Sinusoidal flux
- `robin:alpha=1,beta=0.5,gamma=\cos(x)` - Robin with spatial variation
- `piecewise:x<0.5:dirichlet:1;x>=0.5:dirichlet:0` - Piecewise Dirichlet

### Output Formats

**VTK Legacy (`.vtk`)**: ASCII format, compatible with older tools.
```bash
./build/pde_sim ... --out output.vtk
```

**VTK XML ImageData (`.vti`)**: Binary format with compression support, includes derived fields.
```bash
./build/pde_sim ... --out output.vti --format vti
```

VTK XML output includes derived fields:
- `gradient_x`, `gradient_y`, `gradient_z` - Gradient components
- `laplacian` - Laplacian of the solution
- `flux_x`, `flux_y`, `flux_z` - Diffusion flux components
- `energy_norm` - Energy norm (u²)

## LaTeX PDE Support

The parser accepts linear PDEs with constant coefficients plus optional global integral terms and simple nonlinear reaction terms. Numeric coefficients can multiply any supported term. Supported term forms include:

### Spatial Derivatives
- Second derivatives: `u_{xx}`, `u_{yy}`, `u_{zz}`, `\partial_{xx}u`, `\partial_x\partial_x u`, `\frac{\partial^2 u}{\partial x^2}`
- First derivatives: `u_x`, `u_y`, `u_z`, `\partial_x u`, `\frac{\partial u}{\partial x}`
- Laplacian: `\nabla^2 u`, `\Delta u`, `\triangle u`

### Time Derivatives
- First-order: `u_t`, `\partial_t u`, `\frac{\partial u}{\partial t}`, `\dot{u}`
- Second-order: `u_{tt}`, `\partial_{tt}u`, `\frac{\partial^2 u}{\partial t^2}`, `\ddot{u}`

### Right-Hand Side (RHS)
- Constant: `1`, `-2.5`, `0`
- General expression: `\sin(\pi x) \cos(\pi y)`, `x^2 + y^2`, `\exp(-x)`, `\sin(\pi x) \exp(-t)` (for time-dependent)

### Special Terms
- Global integral term: `K(x,y) \int u \, dx dy` or `K(x,y) \iint u \, dx dy`
- Nonlinear reaction terms: `u^p` (integer `p >= 2`), `\sin(u)`, `\cos(u)`, `\exp(u)`, `|u|`, `\abs(u)`

### Not Supported
- Mixed derivatives (`u_{xy}`) or nonlinear derivative terms (`u u_x`)
- Kernels inside the integral (`\int K(x,y)u \, dx dy`)
- Time derivatives higher than second order

## Coordinate Systems

The solver supports multiple coordinate systems with proper metric terms:

### 2D Coordinate Systems
- **Cartesian**: `(x, y)` - Standard rectangular coordinates
- **Polar**: `(r, theta)` - Radial coordinates with metric terms `(1/r) ∂/∂r (r ∂u/∂r) + (1/r²) ∂²u/∂θ²`
- **Axisymmetric**: `(r, z)` - Cylindrical symmetry with metric terms
- **SphericalSurface**: `(theta, phi)` - Surface of a sphere with metric terms

### 3D Coordinate Systems
- **Cartesian**: `(x, y, z)` - Standard rectangular coordinates
- **Cylindrical**: `(r, theta, z)` - Cylindrical coordinates with metric terms
- **SphericalVolume**: `(r, theta, phi)` - Spherical coordinates with metric terms
- **ToroidalSurface**: `(theta, phi)` - Surface of a torus
- **ToroidalVolume**: `(r, theta, phi)` - Toroidal coordinates

**Metric Terms**: All non-Cartesian coordinate systems now include proper metric coefficients for the Laplacian and advection operators. Singularities (e.g., r=0) are handled using L'Hôpital's rule. Periodic boundaries are automatically detected for angular coordinates.

### Coordinate System Examples

**Polar coordinates (2D)**:
```bash
./build/pde_sim \
  --pde "-\nabla^2 u = 1" \
  --domain 0,1,0,6.28318 \
  --grid 64,64 \
  --coord polar \
  --bc "left:dirichlet:0;right:dirichlet:0;bottom:dirichlet:0;top:dirichlet:0" \
  --out outputs/polar.vtk
```

**Note**: Coordinate system selection (cylindrical, spherical, etc.) is available in the GUI. In the GUI, select the coordinate system from the coordinate mode dropdown, and the solver will automatically apply the correct metric terms.

## GUI (OpenGL, pde_gui)

Build and run:
```bash
cmake -S . -B build
cmake --build build --target pde_gui
./build/pde_gui
```

Features:
- LaTeX PDE input with live preview
- Boundary condition inputs and previews per side (including piecewise BCs)
- Domain configuration (bounding box or implicit shape, including file/mask import with offset/scale alignment)
- Coordinate system selection with metric term support
- Solver method selection (Jacobi, GS, SOR, CG, BiCGStab, GMRES, Multigrid)
- Residual convergence plotting (L2 and L-infinity norms)
- GPU OpenGL viewer with grid overlay, point scaling, and fit-to-view
- Viewer tools: slice plane and isosurface band filtering
- Time controls for `u_t` or `u_tt` PDEs: play/pause, step, frame slider, and frame input
- Status/log panel with solver and file output messages
- VTK I/O: structured points output (`.vtk` or `.vti`) and point-cloud import
- Derived fields visualization (gradient, Laplacian, flux, energy norm)

Time series outputs:
- Each frame is written as `*_frame_000.vtk`, `*_frame_001.vtk`, etc. (or `.vti` if selected)
- A `.pvd` manifest (`*_series.pvd`) is written next to the frames for easy loading in ParaView
- Frames are stored under a unique subdirectory within the output directory
- Derived fields are computed and written for each frame

## Visualization

Use ParaView or other VTK-compatible tools:
```bash
# ParaView can directly open .vtk, .vti, and .pvd files
paraview outputs/heat_series_000.vti
paraview outputs/heat_series_series.pvd  # Opens entire time series
```

## Backends

### CPU Backend
- **Always available**
- Supports all solver methods (Jacobi, GS, SOR, CG, BiCGStab, GMRES, Multigrid)
- Supports all features: 2D/3D, time-dependent, metric-corrected coordinates, embedded boundaries, spatial RHS, nonlinear terms, integral terms, piecewise BCs
- Multi-threaded for parallel computation

### CUDA Backend
- **Requires**: `USE_CUDA=ON` at build time and a compatible NVIDIA GPU
- Supports: Jacobi, Red-Black GS, SOR, CG, BiCGStab, GMRES, Multigrid
- Supports: 3D domains, spatial RHS, basic nonlinear terms
- Limitations: Krylov/MG methods limited to 2D, Dirichlet-only BCs, linear PDEs

### Metal Backend
- **Enabled on Apple platforms by default**
- Supports: Jacobi, Red-Black GS, SOR, CG, BiCGStab, GMRES, Multigrid
- Supports: 2D domains, spatial RHS (when applicable)
- Limitations: Krylov/MG methods limited to 2D, Dirichlet-only BCs, linear PDEs, no 3D for Krylov/MG

### TPU Backend
- **Compile-time stub** (not yet implemented)

### Backend Selection Notes
- **CPU fallback** occurs for: global integral terms (always CPU-only)
- **GPU support** has been expanded: 3D domains, spatial RHS, and basic nonlinear terms are now supported on CUDA/Metal
- **Solver method restrictions**: Krylov methods (CG, BiCGStab, GMRES) and Multigrid require linear PDEs with Dirichlet-only boundary conditions and no nonlinear terms on GPU

## Convergence and Residuals

The solver computes and reports residual norms:
- **L2 norm**: `||Au - b||_2` - Root mean square residual
- **L-infinity norm**: `||Au - b||_∞` - Maximum absolute residual

Residual reporting:
- CLI: Use `--residual-interval N` to print residuals every N iterations
- GUI: Residual history is plotted in real-time during solves
- Output: Final residual norms are printed after solve completion

Example with residual reporting:
```bash
./build/pde_sim \
  --pde "-u_{xx} - u_{yy} = 1" \
  --domain 0,1,0,1 \
  --grid 64,64 \
  --bc "left:dirichlet:0;right:dirichlet:0;bottom:dirichlet:0;top:dirichlet:0" \
  --method cg \
  --residual-interval 10 \
  --tol 1e-6 \
  --out outputs/poisson.vtk
```

## Time-Series Monitoring

Time-dependent runs track conserved-quantity proxies per frame:
- Mass: `sum(u) * dx * dy` (and `* dz` for 3D).
- Energy: `sum(u^2) * dx * dy` (and `* dz` for 3D).
- Max norm `||u||_inf` for blow-up detection.

Warnings are logged if mass or energy drift exceeds 1% relative to the initial frame (when the initial value is non-zero), or if `||u||_inf` grows rapidly for consecutive frames. Monitor values are recorded in run summary sidecars under the `monitors` section.

## Dataset Index and Cleanup

Build a lightweight index of completed runs (one entry per summary sidecar, skipping per-frame summaries). By default the index is written to `<dir>/dataset_index.json` and a short summary is printed:
```bash
./build/pde_sim --dataset-index outputs
```

Override the output path or print JSON to stdout:
```bash
./build/pde_sim --dataset-index outputs --dataset-index-out outputs/my_index.json
./build/pde_sim --dataset-index outputs --dataset-index-out -
```

Remove orphaned summary/metadata sidecars (and optionally empty directories):
```bash
./build/pde_sim --dataset-cleanup outputs --dataset-cleanup-dry-run
./build/pde_sim --dataset-cleanup outputs --dataset-cleanup-empty-dirs
```

## Validation and Testing

Built-in self-test for solver validation:
```bash
./build/pde_sim --self-test
```

The self-test runs regression cases with analytic golden outputs (Poisson, Helmholtz, anisotropic diffusion, mixed derivatives, integral terms, and nonlinear terms) plus solver-method coverage. It reports error norms (L2 and L-infinity) against known exact solutions.

CI-style regression runner:
```bash
./build/regression_suite
```

## Checkpoint and Restart

For long-running time-dependent simulations, checkpoint/restart is available:

**Writing checkpoints**:
```bash
./build/pde_sim \
  --pde "u_t - u_{xx} - u_{yy} = 0" \
  --domain 0,1,0,1 \
  --grid 64,64 \
  --bc "left:dirichlet:0;right:dirichlet:0;bottom:dirichlet:0;top:dirichlet:0" \
  --time 0,10,0.001,10000 \
  --checkpoint outputs/checkpoint.txt \
  --out outputs/series
```

**Restarting from checkpoint**:
```bash
./build/pde_sim \
  --pde "u_t - u_{xx} - u_{yy} = 0" \
  --domain 0,1,0,1 \
  --grid 64,64 \
  --bc "left:dirichlet:0;right:dirichlet:0;bottom:dirichlet:0;top:dirichlet:0" \
  --time 0,20,0.001,20000 \
  --restart outputs/checkpoint.txt \
  --out outputs/series_continued
```

Checkpoint files contain: domain, grid state, current time, frame index, PDE coefficients, and boundary conditions.
