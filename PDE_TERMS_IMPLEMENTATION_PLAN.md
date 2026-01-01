# PDE Terms Implementation Plan

This document provides a detailed multistep implementation plan for each recommendation in `PDE_TERMS_RECOMMENDATIONS.md`, organized by priority tier.
It now also includes consolidated expansion roadmaps (Options 1-3) so the
term-level work is connected to the broader PDE expansion strategy.

--------------------------------------------------------------------------------

## Option 1: Extended Finite-Difference Operator Engine (Linear + Variable Coefficients)

### Goals
- Support a broader linear PDE class: mixed derivatives, variable coefficients,
  higher-order linear derivatives, and non-Cartesian metric-aware forms.
- Keep CLI automation straightforward and deterministic.
- Preserve current fast paths while expanding capability.

### Phases
**Phase 0: Scope and Compatibility Matrix**
- Define expanded grammar (allowed derivative orders, coefficient forms).
- Backend capability matrix (CPU full, GPU limited with clear gating).
- CLI help updates and explicit validation outcomes.

**Phase 1: PDE AST + Parsing Extensions**
- Introduce `PDEOperator`/`PDETerm` structures (derivative multi-index, coeff expr).
- Extend LaTeX parsing to handle mixed derivatives and coefficient expressions.
- Normalize terms and provide a canonical representation.
- Add CLI `--validate` and `--dump-operator` paths (no solve).

**Phase 2: Coefficient Evaluation**
- Add a coefficient evaluator for LaTeX expressions and caching.
- Precompute coefficient fields for static coefficients.
- Optional coefficient field input via file mapping.

**Phase 3: Discretization and Stencil Builder**
- Generalized stencils for 1st/2nd/mixed derivatives.
- Boundary integration for Dirichlet/Neumann/Robin with spatial BCs.
- Coordinate system metric integration.

**Phase 4: Linear Operator Runtime**
- Matrix-free `LinearOperator` interface + optional CSR assembly.
- CPU solvers for full operator set, GPU matrix-free for restricted subset.

**Phase 5: CLI and Config Extensions**
- Operator form flags (divergence vs non-divergence).
- Structured run summaries (JSON) and deterministic exit codes.

**Phase 6: UI Alignment (Optional)**
- GUI warnings for backend incompatibility and expanded PDE forms.

**Phase 7: Validation and Testing**
- Manufactured solutions for mixed derivatives and variable coefficients.
- Regression tests for existing PDEs and performance baselines.

**Milestones**
1) Parser + operator model + validate-only CLI path.
2) CPU operator engine with mixed derivatives + variable coefficients.
3) Full CLI workflow with expanded linear PDEs and summaries.

--------------------------------------------------------------------------------

## Option 2: Nonlinear PDE + Conservation-Law Pipeline (Extended Plan)

**Phase A: Nonlinear PDE Representation**
- Residual-form `R(u)=0` for nonlinear terms and products.

**Phase B: FV Discretization**
- Flux functions, Riemann solvers, and limiters.

**Phase C: Nonlinear Solvers + Time Integration**
- Newton-Krylov, IMEX, adaptive dt + CFL monitoring.

**Phase D: Backend Gating**
- CPU-first; GPU explicit FV kernels where valid.

**Phase E: CLI Automation**
- Structured output + status codes + dry-run compatibility checks.

**Phase F: Validation**
- Burgers, reaction-diffusion, shock tube benchmarks.

--------------------------------------------------------------------------------

## Option 3: Weak-Form / FEM Track (Extended Plan)

**Phase A: Mesh Ingestion**
- Gmsh/VTK unstructured input, topology utilities.

**Phase B: Weak-Form DSL + Assembly**
- Basis functions, quadrature, global assembly.

**Phase C: Solver Integration**
- PETSc/Trilinos integration, preconditioners, AMR hooks.

**Phase D: BCs and Multiphysics**
- Strong/weak BCs, mixed fields for coupled PDEs.

**Phase E: CLI + Output**
- Mesh-driven CLI, VTK outputs with FE metadata.

**Phase F: Validation**
- Poisson/heat/Stokes/elasticity benchmark suites.

--------------------------------------------------------------------------------

The term-level tiers below map directly into Option 1 Phases 1-3 (parsing,
coefficient evaluation, stencil generation) and provide the concrete, ordered
implementation tasks.

---

## Tier 1: Highest Impact, Moderate Complexity

### 1. Mixed Derivatives (`u_{xy}`, `u_{xz}`, `u_{yz}`)

**Goal**: Enable cross-derivatives for heat/wave equations, Navier-Stokes, and elasticity problems.

#### Step 1.1: Extend Data Structures
- **File**: `include/pde_types.h`
- **Action**: Add mixed derivative coefficients to `PDECoefficients`:
  ```cpp
  double ab = 0.0;  // u_xy (and u_yx, symmetric)
  double ac = 0.0;  // u_xz (and u_zx)
  double bc = 0.0;  // u_yz (and u_zy)
  ```

#### Step 1.2: Update LaTeX Parser
- **File**: `src/latex_parser.cpp`
- **Actions**:
  - Remove or modify `DetectMixedDerivative()` to accept mixed derivatives instead of rejecting them
  - Add pattern matching arrays: `kDXYPatterns[]`, `kDXZPatterns[]`, `kDYZPatterns[]`
  - Support patterns: `u_{xy}`, `u_{yx}`, `\partial_{xy}u`, `\frac{\partial^2 u}{\partial x \partial y}`
  - Add coordinate variants: `u_{rtheta}`, `u_{rphi}`, `u_{thetaphi}` for non-Cartesian systems
  - Update `ParseTerm()` to handle mixed derivative patterns
  - Update equation parsing to subtract mixed terms from RHS correctly

#### Step 1.3: Implement Finite Difference Stencils
- **File**: `src/backends/cpu/solver.cpp`
- **Actions**:
  - Add helper function `ComputeMixedDerivatives()` to compute `u_xy`, `u_xz`, `u_yz` using centered differences:
    - `u_xy ≈ (u(i+1,j+1) - u(i+1,j-1) - u(i-1,j+1) + u(i-1,j-1)) / (4*dx*dy)`
    - Similar for `u_xz` and `u_yz`
  - Update 2D solver loops to include mixed derivative contributions
  - Update 3D solver loops to include all three mixed derivatives
  - Handle boundary cases (near domain edges) with one-sided differences or ghost points

#### Step 1.4: Update Coordinate System Support
- **File**: `src/coordinate_metrics.cpp`
- **Actions**:
  - Extend `ComputeMetricDerivatives2D()` to compute mixed derivatives in polar/cylindrical coordinates
  - Add `ComputeMetricDerivatives3D()` for 3D coordinate systems
  - Account for metric factors in mixed derivative computations

#### Step 1.5: Update GPU Backends
- **Files**: `src/backends/cuda/...`, `src/backends/metal/...`
- **Actions**:
  - Add mixed derivative computation kernels
  - Update solver kernels to include mixed derivative terms
  - Test performance impact

#### Step 1.6: Testing
- Create test cases:
  - Simple 2D: `u_{xx} + 2u_{xy} + u_{yy} = 0` (known analytical solution)
  - 3D mixed derivatives
  - Coordinate system variants

---

### 2. Variable Coefficient Terms (Spatial Dependence)

**Goal**: Support spatially-varying coefficients like `a(x,y) u_{xx}`, `\sin(x) u_{yy}`.

#### Step 2.1: Extend Data Structures
- **File**: `include/pde_types.h`
- **Action**: Add LaTeX expression strings for variable coefficients:
  ```cpp
  struct PDECoefficients {
    // ... existing constant coefficients ...
    std::string a_latex;  // coefficient for u_xx (spatial)
    std::string b_latex;  // coefficient for u_yy (spatial)
    std::string az_latex; // coefficient for u_zz (spatial)
    std::string c_latex;  // coefficient for u_x (spatial)
    std::string d_latex;  // coefficient for u_y (spatial)
    std::string dz_latex; // coefficient for u_z (spatial)
    std::string e_latex;  // coefficient for u (spatial)
    // If latex string is empty, use the constant double value
  };
  ```

#### Step 2.2: Update LaTeX Parser
- **File**: `src/latex_parser.cpp`
- **Actions**:
  - Modify `MatchTerm()` to detect coefficient expressions before the derivative
  - Parse patterns like `a(x,y) u_{xx}`, `\sin(x) u_{yy}`, `(1+x^2) u_x`
  - Extract coefficient LaTeX expression and store in appropriate `*_latex` field
  - Handle multiplication: `coeff * derivative_term`

#### Step 2.3: Create Coefficient Evaluator
- **File**: `src/coefficient_evaluator.cpp` (new file)
- **Actions**:
  - Create `EvaluateCoefficient()` function similar to `eval_rhs()` in solver
  - Parse LaTeX coefficient expression and evaluate at grid point `(x,y,z,t)`
  - Reuse existing expression parser infrastructure if available
  - Cache parsed expressions for performance

#### Step 2.4: Update Solver Loops
- **File**: `src/backends/cpu/solver.cpp`
- **Actions**:
  - In each solver loop, before computing derivatives:
    - Evaluate variable coefficients at current grid point using `EvaluateCoefficient()`
    - Use evaluated coefficients instead of constant values
  - Update 2D, 3D, and time-stepping loops
  - Handle coordinate system transformations correctly

#### Step 2.5: Update GPU Backends
- **Files**: `src/backends/cuda/...`, `src/backends/metal/...`
- **Actions**:
  - Add coefficient evaluation kernels
  - Pass coefficient LaTeX strings to GPU (or pre-evaluate on CPU)
  - Update solver kernels to use variable coefficients

#### Step 2.6: Testing
- Test cases:
  - `\sin(x) u_{xx} + u_{yy} = 0` with known solution
  - `(1+x^2) u_x + u_y = 0`
  - Verify coefficient evaluation at boundaries

---

### 3. Nonlinear Derivative Terms (`u u_x`, `|\nabla u|^2`)

**Goal**: Enable Burgers' equation, KdV, and nonlinear diffusion.

#### Step 3.1: Extend Data Structures
- **File**: `include/pde_types.h`
- **Action**: Add new enum and structure:
  ```cpp
  enum class NonlinearDerivativeKind {
    UUx,        // u * u_x
    UUy,        // u * u_y
    UUz,        // u * u_z
    UxUx,       // u_x * u_x
    UyUy,       // u_y * u_y
    UzUz,       // u_z * u_z
    GradSquared, // |∇u|² = u_x² + u_y² + u_z²
  };
  
  struct NonlinearDerivativeTerm {
    NonlinearDerivativeKind kind;
    double coeff = 0.0;
  };
  ```
- Add to `SolveInput`: `std::vector<NonlinearDerivativeTerm> nonlinear_derivatives;`

#### Step 3.2: Update LaTeX Parser
- **File**: `src/latex_parser.cpp`
- **Actions**:
  - Add `ParseNonlinearDerivativeTerm()` function
  - Detect patterns: `u u_x`, `u \partial_x u`, `u_x^2`, `|\nabla u|^2`, `(\nabla u)^2`
  - Parse and extract coefficient
  - Return `NonlinearDerivativeTerm` in `LatexParseResult`

#### Step 3.3: Implement Evaluation Function
- **File**: `src/backends/cpu/solver.cpp` or new helper file
- **Actions**:
  - Create `EvalNonlinearDerivative()` function
  - Input: `NonlinearDerivativeTerm`, grid values, derivatives `u_x`, `u_y`, `u_z`
  - Compute and return the nonlinear derivative term value
  - Handle `GradSquared` as sum of squared derivatives

#### Step 3.4: Update Solver Loops
- **File**: `src/backends/cpu/solver.cpp`
- **Actions**:
  - In each solver iteration:
    - Compute first derivatives `u_x`, `u_y`, `u_z` (already computed)
    - For each `NonlinearDerivativeTerm`, evaluate using `EvalNonlinearDerivative()`
    - Add to RHS or appropriate location in discretization
  - For implicit methods, may need to linearize (Newton iteration)

#### Step 3.5: Handle Implicit Time-Stepping
- **File**: `src/backends/cpu/solver.cpp`
- **Actions**:
  - For nonlinear derivative terms in time-dependent problems:
    - Use explicit time-stepping (evaluate at current time)
    - Or implement Newton linearization for implicit schemes
    - Consider stability constraints (CFL condition for `u u_x`)

#### Step 3.6: Update GPU Backends
- **Files**: `src/backends/cuda/...`, `src/backends/metal/...`
- **Actions**:
  - Add kernels for nonlinear derivative evaluation
  - Update solver kernels

#### Step 3.7: Testing
- Test cases:
  - Burgers' equation: `u_t + u u_x = \nu u_{xx}` (known traveling wave solution)
  - KdV: `u_t + u u_x + u_{xxx} = 0` (requires higher-order derivatives too)
  - Gradient squared: `u_t = \nabla^2 u + |\nabla u|^2`

---

## Tier 2: High Impact, Higher Complexity

### 4. Higher-Order Derivatives (`u_{xxx}`, `u_{xxxx}`)

**Goal**: Support beam equations, KdV, Kuramoto-Sivashinsky.

#### Step 4.1: Extend Data Structures
- **File**: `include/pde_types.h`
- **Action**: Add to `PDECoefficients`:
  ```cpp
  double a3 = 0.0;  // u_xxx
  double b3 = 0.0;  // u_yyy
  double az3 = 0.0; // u_zzz
  double a4 = 0.0;  // u_xxxx
  double b4 = 0.0;  // u_yyyy
  double az4 = 0.0; // u_zzzz
  ```

#### Step 4.2: Update LaTeX Parser
- **File**: `src/latex_parser.cpp`
- **Actions**:
  - Add pattern arrays: `kD3XPatterns[]`, `kD4XPatterns[]`, etc.
  - Support: `u_{xxx}`, `u_{xxxx}`, `\partial_{xxx}u`, `\frac{\partial^3 u}{\partial x^3}`
  - Update `ParseTerm()` to match these patterns

#### Step 4.3: Implement Finite Difference Stencils
- **File**: `src/backends/cpu/solver.cpp`
- **Actions**:
  - Third derivative: Use 5-point stencil (or wider for accuracy)
    - `u_{xxx} ≈ (-u(i+2) + 2*u(i+1) - 2*u(i-1) + u(i-2)) / (2*dx^3)`
  - Fourth derivative: Use 5-point stencil
    - `u_{xxxx} ≈ (u(i+2) - 4*u(i+1) + 6*u(i) - 4*u(i-1) + u(i-2)) / dx^4`
  - Handle boundary conditions carefully (one-sided stencils or ghost points)
  - Add helper functions: `ComputeThirdDerivative()`, `ComputeFourthDerivative()`

#### Step 4.4: Update Solver Loops
- **File**: `src/backends/cpu/solver.cpp`
- **Actions**:
  - Add higher-order derivative contributions to discretization
  - Update matrix assembly for implicit methods
  - Consider stability: higher-order derivatives may require smaller time steps

#### Step 4.5: Update GPU Backends
- **Files**: `src/backends/cuda/...`, `src/backends/metal/...`
- **Actions**:
  - Implement higher-order derivative kernels
  - Update solver kernels

#### Step 4.6: Testing
- Test cases:
  - Euler-Bernoulli beam: `u_{xxxx} = f(x)` with known solution
  - KdV: `u_t + u u_x + u_{xxx} = 0` (soliton solution)

---

### 5. Biharmonic Operator (`\nabla^4 u`)

**Goal**: Support plate equations, Stokes flow, thin film equations.

#### Step 5.1: Update LaTeX Parser
- **File**: `src/latex_parser.cpp`
- **Actions**:
  - Add patterns: `\nabla^4 u`, `\Delta^2 u`, `\triangle^2 u`, `(\nabla^2)^2 u`
  - Parse and recognize as `\nabla^4 u = u_{xxxx} + 2u_{xxyy} + u_{yyyy}` in 2D
  - In 2D: set `a4`, `b4`, and mixed derivative `ab` appropriately
  - In 3D: include all cross terms

#### Step 5.2: Implement Biharmonic Discretization
- **File**: `src/backends/cpu/solver.cpp`
- **Actions**:
  - For 2D: `\nabla^4 u = u_{xxxx} + 2u_{xxyy} + u_{yyyy}`
    - Use fourth derivatives and mixed fourth derivatives
  - For 3D: Include all terms
  - Use appropriate finite difference stencils
  - May require wider stencils (9-point or more)

#### Step 5.3: Update Solver
- **File**: `src/backends/cpu/solver.cpp`
- **Actions**:
  - Add biharmonic contribution to discretization
  - Update matrix assembly
  - Consider boundary conditions (clamped, simply supported plates)

#### Step 5.4: Testing
- Test cases:
  - Plate equation: `\nabla^4 u = 1` on unit square with clamped BCs
  - Compare with known analytical solutions

---

### 6. Additional Nonlinear Functions (`\tanh`, `\ln`, `\max`)

**Goal**: Expand nonlinear function support for reaction-diffusion and physics applications.

#### Step 6.1: Extend Data Structures
- **File**: `include/pde_types.h`
- **Action**: Extend `NonlinearKind` enum:
  ```cpp
  enum class NonlinearKind {
    Power, Sin, Cos, Exp, Abs,  // existing
    Tanh, Sinh, Cosh,           // hyperbolic
    Ln, Log,                    // logarithmic
    Max, Min, ReLU,             // piecewise
  };
  ```

#### Step 6.2: Update LaTeX Parser
- **File**: `src/latex_parser.cpp`
- **Actions**:
  - Add pattern matching for: `\tanh(u)`, `\sinh(u)`, `\cosh(u)`, `\ln(u)`, `\log(u)`, `\max(u,0)`, `\min(u,0)`, `\text{ReLU}(u)`
  - Update `ParseNonlinearTerm()` to recognize these patterns
  - For `Max`/`Min`/`ReLU`, may need to parse threshold parameter

#### Step 6.3: Implement Evaluation
- **File**: `src/backends/cpu/solver.cpp` (or existing `EvalNonlinear()`)
- **Actions**:
  - Extend `EvalNonlinear()` function to handle new `NonlinearKind` values
  - Implement: `tanh`, `sinh`, `cosh`, `ln` (with domain checks), `max`, `min`, `ReLU`
  - Handle domain restrictions (e.g., `ln(u)` requires `u > 0`)

#### Step 6.4: Update Solver
- **File**: `src/backends/cpu/solver.cpp`
- **Actions**:
  - No changes needed if `EvalNonlinear()` is already called
  - Ensure proper handling of domain restrictions (warnings/errors)

#### Step 6.5: Testing
- Test cases:
  - `u_t = \nabla^2 u + \tanh(u)` (reaction-diffusion)
  - `u_t = \nabla^2 u + \ln(1+u)` (with positivity check)

---

## Tier 3: Specialized Applications

### 7. Integro-Differential Terms (Kernels Inside Integrals)

**Goal**: Support nonlocal diffusion, memory effects, Fredholm equations.

#### Step 7.1: Extend Data Structures
- **File**: `include/pde_types.h`
- **Action**: Extend `IntegralTerm`:
  ```cpp
  struct IntegroDifferentialTerm {
    std::string kernel_latex;      // K(x,y) or K(x-y) or K(t-s)
    std::string integration_vars;  // "dx dy", "dy", "ds"
    bool is_convolution;           // true if K(x-y) form
    double coeff = 1.0;
  };
  ```

#### Step 7.2: Update LaTeX Parser
- **File**: `src/latex_parser.cpp`
- **Actions**:
  - Modify `ParseIntegralTerm()` to detect kernels inside integrals
  - Parse: `\int K(x,y) u(y) \, dy`, `\int_0^t K(t-s) u(s) \, ds`
  - Extract kernel expression and integration variables

#### Step 7.3: Implement Integral Evaluation
- **File**: `src/integral_evaluator.cpp` (new or extend existing)
- **Actions**:
  - Create `EvaluateIntegroDifferential()` function
  - For spatial integrals: evaluate kernel and solution at all grid points, compute integral
  - For temporal integrals: maintain solution history, evaluate convolution
  - Use numerical quadrature (trapezoidal, Simpson's rule)

#### Step 7.4: Update Solver
- **File**: `src/backends/cpu/solver.cpp`
- **Actions**:
  - In solver loops, evaluate integro-differential terms
  - For temporal integrals, store solution history
  - Add to RHS or discretization

#### Step 7.5: Performance Optimization
- **Actions**:
  - Cache kernel evaluations if possible
  - Use FFT for convolution integrals if kernel is translation-invariant
  - Consider parallelization

#### Step 7.6: Testing
- Test cases:
  - Nonlocal diffusion: `u_t = \int K(x-y) u(y) dy - u(x)`
  - Known analytical solutions if available

---

### 8. Vector/System PDEs (Multiple Unknowns)

**Goal**: Enable Navier-Stokes, Maxwell equations, elasticity.

#### Step 8.1: Design Architecture
- **Files**: `include/pde_types.h`, design document
- **Actions**:
  - Design `VectorPDECoefficients` structure:
    ```cpp
    struct VectorPDECoefficients {
      std::vector<PDECoefficients> components;  // one per unknown
      std::vector<std::vector<double>> coupling;  // cross-component terms
    };
    ```
  - Design vector-valued grid storage
  - Plan solver modifications

#### Step 8.2: Extend Data Structures
- **File**: `include/pde_types.h`
- **Actions**:
  - Add `VectorPDECoefficients` structure
  - Add `num_unknowns` to `SolveInput`
  - Modify grid storage to support vector fields

#### Step 8.3: Update LaTeX Parser
- **File**: `src/latex_parser.cpp`
- **Actions**:
  - Parse vector notation: `\begin{pmatrix} u \\ v \end{pmatrix}_t = ...`
  - Parse system of equations
  - Extract component equations and coupling terms

#### Step 8.4: Implement Vector Solver
- **File**: `src/backends/cpu/solver.cpp` (or new `vector_solver.cpp`)
- **Actions**:
  - Modify solver to handle vector-valued grids
  - Update discretization for each component
  - Add coupling terms between components
  - Update matrix assembly for block systems

#### Step 8.5: Update GPU Backends
- **Files**: `src/backends/cuda/...`, `src/backends/metal/...`
- **Actions**:
  - Extend kernels for vector fields
  - Update memory layout

#### Step 8.6: Testing
- Test cases:
  - Simple 2x2 system with known solution
  - Gradually increase complexity

---

### 9. Boundary Integral Terms (Surface Integrals)

**Goal**: Support boundary element methods, Robin BCs with integrals.

#### Step 9.1: Extend Data Structures
- **File**: `include/pde_types.h`
- **Action**: Add to `BoundaryCondition`:
  ```cpp
  struct BoundaryIntegralTerm {
    std::string kernel_latex;  // K(x,y) for boundary integral
    double coeff = 0.0;
  };
  // Add to BoundaryCondition:
  std::vector<BoundaryIntegralTerm> boundary_integrals;
  ```

#### Step 9.2: Update LaTeX Parser
- **File**: `src/latex_parser.cpp`
- **Actions**:
  - Parse: `\int_{\partial \Omega} u \, dS`, `\oint u \, ds`
  - Extract boundary integral terms from BC expressions

#### Step 9.3: Implement Boundary Mesh Representation
- **File**: `src/boundary_mesh.cpp` (new)
- **Actions**:
  - Create boundary mesh representation
  - Compute surface integrals along boundaries
  - Handle 2D (curves) and 3D (surfaces)

#### Step 9.4: Update Boundary Condition Application
- **File**: `src/backends/cpu/solver.cpp`
- **Actions**:
  - Evaluate boundary integrals
  - Incorporate into BC application logic

#### Step 9.5: Testing
- Test cases:
  - Simple 2D domain with boundary integral BC
  - Compare with known solutions

---

## Tier 4: Advanced/Research

### 10. Fractional Derivatives

**Goal**: Support anomalous diffusion, fractional Laplacian.

#### Step 10.1: Research and Design
- **Actions**:
  - Research numerical methods (L1 scheme, Grünwald-Letnikov, etc.)
  - Design data structures for fractional order `α`
  - Plan implementation approach

#### Step 10.2: Implement Fractional Derivative Computation
- **File**: `src/fractional_derivatives.cpp` (new)
- **Actions**:
  - Implement Caputo or Riemann-Liouville derivative
  - Use appropriate numerical scheme
  - Handle memory requirements (fractional derivatives are non-local)

#### Step 10.3: Update Parser and Solver
- **Files**: `src/latex_parser.cpp`, `src/backends/cpu/solver.cpp`
- **Actions**:
  - Parse: `D_t^\alpha u`, `\partial_t^\alpha u`, `(-\Delta)^s u`
  - Integrate fractional derivative computation into solver

#### Step 10.4: Testing
- Test cases:
  - Fractional diffusion: `D_t^\alpha u = \nabla^2 u` with known solutions

---

### 11. Stochastic Terms (Noise)

**Goal**: Support stochastic PDEs, Langevin equations.

#### Step 11.1: Design Random Number Generation
- **Actions**:
  - Choose RNG library (or use standard library)
  - Design noise generation interface

#### Step 11.2: Extend Data Structures
- **File**: `include/pde_types.h`
- **Action**: Add:
  ```cpp
  struct StochasticTerm {
    std::string noise_latex;  // \sigma(x) dW_t, \xi(x,t)
    double intensity = 1.0;
  };
  ```

#### Step 11.3: Implement Noise Generation
- **File**: `src/stochastic.cpp` (new)
- **Actions**:
  - Generate white noise `dW`
  - Generate colored noise `\xi(x,t)`
  - Handle spatial and temporal correlation

#### Step 11.4: Update Solver
- **File**: `src/backends/cpu/solver.cpp`
- **Actions**:
  - Add noise terms to discretization
  - Use appropriate time-stepping (Euler-Maruyama, etc.)
  - Support ensemble methods (multiple realizations)

#### Step 11.5: Testing
- Test cases:
  - Stochastic heat equation: `du = \nabla^2 u dt + \sigma dW`
  - Compare statistics with theory

---

### 12. Delay Terms (Time Delays)

**Goal**: Support delay-diffusion, neural field models.

#### Step 12.1: Extend Data Structures
- **File**: `include/pde_types.h`
- **Action**: Add:
  ```cpp
  struct DelayTerm {
    double delay = 0.0;  // \tau
    std::string expression_latex;  // f(u(x,t-\tau))
  };
  ```

#### Step 12.2: Implement Solution History Storage
- **File**: `src/solution_history.cpp` (new)
- **Actions**:
  - Store solution at previous time steps
  - Interpolate for non-integer delay times
  - Manage memory efficiently

#### Step 12.3: Update Parser and Solver
- **Files**: `src/latex_parser.cpp`, `src/backends/cpu/solver.cpp`
- **Actions**:
  - Parse: `u(x,t-\tau)`, `u(t-\Delta t)`
  - Evaluate delayed terms using solution history
  - Update solver to maintain history

#### Step 12.4: Testing
- Test cases:
  - Delay-diffusion: `u_t = \nabla^2 u + f(u(x,t-\tau))`
  - Known solutions if available

---

### 13. Time-Fractional and Distributed-Order Derivatives

**Goal**: Support subdiffusion, superdiffusion, distributed-order equations.

#### Step 13.1: Implement Time-Fractional Derivatives
- **File**: `src/fractional_derivatives.cpp` (extend)
- **Actions**:
  - Implement Caputo time-fractional derivative
  - Use L1 scheme or similar
  - Handle distributed-order (integral over α)

#### Step 13.2: Update Parser and Solver
- **Files**: `src/latex_parser.cpp`, `src/backends/cpu/solver.cpp`
- **Actions**:
  - Parse: `D_t^\alpha u` for time derivatives
  - Parse distributed-order: `\int_0^1 w(\alpha) D_t^\alpha u \, d\alpha`
  - Integrate into time-stepping

#### Step 13.3: Testing
- Test cases:
  - Subdiffusion: `D_t^\alpha u = \nabla^2 u` (0 < α < 1)
  - Compare with analytical solutions

---

## Implementation Order Recommendation

1. **Phase 1 (Tier 1)**: Mixed derivatives → Variable coefficients → Nonlinear derivative terms
2. **Phase 2 (Tier 2)**: Higher-order derivatives → Biharmonic → Additional nonlinear functions
3. **Phase 3 (Tier 3)**: Integro-differential → Vector PDEs → Boundary integrals
4. **Phase 4 (Tier 4)**: Fractional → Stochastic → Delay → Time-fractional

## General Implementation Notes

- **Testing Strategy**: For each feature, create test cases with known analytical solutions
- **Backward Compatibility**: Ensure existing PDEs continue to work
- **Performance**: Profile and optimize critical paths
- **Documentation**: Update user documentation as features are added
- **Coordinate Systems**: Ensure all features work with non-Cartesian coordinates
- **GPU Support**: Extend GPU backends for performance-critical features
