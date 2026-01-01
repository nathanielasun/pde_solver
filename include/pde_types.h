#ifndef PDE_TYPES_H
#define PDE_TYPES_H

#include <atomic>
#include <cmath>
#include <map>
#include <string>
#include <vector>

struct PDECoefficients {
  double a = 0.0;  // u_xx
  double b = 0.0;  // u_yy
  double az = 0.0; // u_zz
  double c = 0.0;  // u_x
  double d = 0.0;  // u_y
  double dz = 0.0; // u_z
  double e = 0.0;  // u
  double f = 0.0;  // constant (legacy, use rhs_latex for general expressions)
  double ut = 0.0; // u_t
  double utt = 0.0; // u_tt
  // Mixed derivatives (symmetric: u_xy = u_yx, etc.)
  double ab = 0.0;  // u_xy (and u_yx)
  double ac = 0.0;  // u_xz (and u_zx)
  double bc = 0.0;  // u_yz (and u_zy)
  
  // Third-order derivatives
  double a3 = 0.0;  // u_xxx
  double b3 = 0.0;  // u_yyy
  double az3 = 0.0; // u_zzz
  
  // Fourth-order derivatives
  double a4 = 0.0;  // u_xxxx
  double b4 = 0.0;  // u_yyyy
  double az4 = 0.0; // u_zzzz
  
  std::string rhs_latex;  // general RHS expression f(x,y,z,t) in LaTeX (empty = use f constant)
  
  // Variable coefficients (spatial dependence): if non-empty, use these instead of constant values
  std::string a_latex;   // coefficient for u_xx (spatial: a(x,y,z))
  std::string b_latex;   // coefficient for u_yy (spatial: b(x,y,z))
  std::string az_latex;  // coefficient for u_zz (spatial: az(x,y,z))
  std::string c_latex;   // coefficient for u_x (spatial: c(x,y,z))
  std::string d_latex;   // coefficient for u_y (spatial: d(x,y,z))
  std::string dz_latex; // coefficient for u_z (spatial: dz(x,y,z))
  std::string e_latex;  // coefficient for u (spatial: e(x,y,z))
  std::string ab_latex; // coefficient for u_xy (spatial: ab(x,y,z))
  std::string ac_latex; // coefficient for u_xz (spatial: ac(x,y,z))
  std::string bc_latex; // coefficient for u_yz (spatial: bc(x,y,z))
  
  // Variable coefficients for higher-order derivatives
  std::string a3_latex;  // coefficient for u_xxx (spatial: a3(x,y,z))
  std::string b3_latex;  // coefficient for u_yyy (spatial: b3(x,y,z))
  std::string az3_latex; // coefficient for u_zzz (spatial: az3(x,y,z))
  std::string a4_latex;  // coefficient for u_xxxx (spatial: a4(x,y,z))
  std::string b4_latex;  // coefficient for u_yyyy (spatial: b4(x,y,z))
  std::string az4_latex; // coefficient for u_zzzz (spatial: az4(x,y,z))
  // If latex string is empty, use the corresponding constant double value
};

inline bool HasVariableCoefficients(const PDECoefficients& pde) {
  return !pde.a_latex.empty() || !pde.b_latex.empty() || !pde.az_latex.empty() ||
         !pde.c_latex.empty() || !pde.d_latex.empty() || !pde.dz_latex.empty() ||
         !pde.e_latex.empty() || !pde.ab_latex.empty() || !pde.ac_latex.empty() ||
         !pde.bc_latex.empty() || !pde.a3_latex.empty() || !pde.b3_latex.empty() ||
         !pde.az3_latex.empty() || !pde.a4_latex.empty() || !pde.b4_latex.empty() ||
         !pde.az4_latex.empty();
}

inline bool HasMixedDerivatives(const PDECoefficients& pde) {
  return std::abs(pde.ab) > 1e-12 || std::abs(pde.ac) > 1e-12 ||
         std::abs(pde.bc) > 1e-12 || !pde.ab_latex.empty() ||
         !pde.ac_latex.empty() || !pde.bc_latex.empty();
}

inline bool HasHigherOrderDerivatives(const PDECoefficients& pde) {
  return std::abs(pde.a3) > 1e-12 || std::abs(pde.b3) > 1e-12 ||
         std::abs(pde.az3) > 1e-12 || std::abs(pde.a4) > 1e-12 ||
         std::abs(pde.b4) > 1e-12 || std::abs(pde.az4) > 1e-12 ||
         !pde.a3_latex.empty() || !pde.b3_latex.empty() || !pde.az3_latex.empty() ||
         !pde.a4_latex.empty() || !pde.b4_latex.empty() || !pde.az4_latex.empty();
}

struct IntegralTerm {
  double coeff = 0.0;        // coefficient for the global integral of u
  std::string kernel_latex;  // optional spatial kernel K(x,y) in LaTeX
};

enum class NonlinearKind {
  Power,
  Sin,
  Cos,
  Exp,
  Abs,
};

struct NonlinearTerm {
  NonlinearKind kind = NonlinearKind::Power;
  double coeff = 0.0;
  int power = 2;
};

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
  NonlinearDerivativeKind kind = NonlinearDerivativeKind::UUx;
  double coeff = 0.0;
};

// Linear operator term for PDEs (derivatives of u with optional coefficient).
struct PDETerm {
  int dx = 0;
  int dy = 0;
  int dz = 0;
  int dt = 0;
  double coeff = 0.0;
  std::string coeff_latex;  // Optional variable coefficient expression.
};

// Parsed operator model (scaffolding for Option 1 expansion).
struct PDEOperator {
  std::vector<PDETerm> lhs_terms;
  double rhs_constant = 0.0;
  std::string rhs_latex;
  std::vector<IntegralTerm> integrals;
  std::vector<NonlinearTerm> nonlinear;
  std::vector<NonlinearDerivativeTerm> nonlinear_derivatives;
};

enum class CoordinateSystem {
  Cartesian,           // (x, y) or (x, y, z)
  Polar,              // (r, theta) - 2D
  Axisymmetric,        // (r, z) - 2D cylindrical symmetry
  Cylindrical,         // (r, theta, z) - 3D
  SphericalSurface,    // (theta, phi) - 2D surface
  SphericalVolume,     // (r, theta, phi) - 3D
  ToroidalSurface,     // (theta, phi) - 2D surface
  ToroidalVolume,      // (r, theta, phi) - 3D
};

struct Domain {
  double xmin = 0.0;
  double xmax = 1.0;
  double ymin = 0.0;
  double ymax = 1.0;
  double zmin = 0.0;
  double zmax = 1.0;
  int nx = 64;
  int ny = 64;
  int nz = 1;
  CoordinateSystem coord_system = CoordinateSystem::Cartesian;
};

struct PointSample {
  double x = 0.0;
  double y = 0.0;
  double z = 0.0;
  double value = 0.0;
};

struct ShapeTransform {
  double offset_x = 0.0;
  double offset_y = 0.0;
  double offset_z = 0.0;
  double scale_x = 1.0;
  double scale_y = 1.0;
  double scale_z = 1.0;
};

struct ShapeMask {
  Domain domain;
  std::vector<double> values;
  std::vector<PointSample> points;
};

enum class BCKind {
  Dirichlet,
  Neumann,
  Robin,
};

struct BoundaryCondition {
  BCKind kind = BCKind::Dirichlet;
  struct Expression {
    double constant = 0.0;
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
    std::string latex;  // general expression in LaTeX (empty = use linear form above)
  };

  Expression value;
  Expression alpha;
  Expression beta;
  Expression gamma;
  std::string condition_latex;  // optional condition expression (e.g., "x < 0.5"); if empty, always applies
};

struct PiecewiseBoundaryCondition {
  std::vector<std::pair<std::string, BoundaryCondition>> segments;  // (condition_latex, bc) pairs
  BoundaryCondition default_bc;  // used if no condition matches
};

struct BoundarySet {
  BoundaryCondition left;
  BoundaryCondition right;
  BoundaryCondition bottom;
  BoundaryCondition top;
  BoundaryCondition front;
  BoundaryCondition back;
  // Optional piecewise BCs (if provided, override the single BC above)
  std::vector<PiecewiseBoundaryCondition> left_piecewise;
  std::vector<PiecewiseBoundaryCondition> right_piecewise;
  std::vector<PiecewiseBoundaryCondition> bottom_piecewise;
  std::vector<PiecewiseBoundaryCondition> top_piecewise;
  std::vector<PiecewiseBoundaryCondition> front_piecewise;
  std::vector<PiecewiseBoundaryCondition> back_piecewise;
};

enum class SolveMethod {
  Jacobi,
  GaussSeidel,
  SOR,
  CG,
  BiCGStab,
  GMRES,
  MultigridVcycle,
};

// Advection discretization schemes for first-derivative terms
enum class AdvectionScheme {
  Centered,      // Standard centered differences (second-order, oscillatory)
  Upwind,        // First-order upwind (diffusive but stable)
  LaxWendroff,   // Second-order Lax-Wendroff
  BeamWarming,   // Second-order Beam-Warming
  Fromm,         // Second-order Fromm
  MinMod,        // TVD with minmod limiter
  Superbee,      // TVD with superbee limiter
  VanLeer,       // TVD with van Leer limiter
  MC,            // TVD with MC limiter
};

struct SolverConfig {
  SolveMethod method = SolveMethod::Jacobi;
  int max_iter = 10000;
  double tol = 1e-6;
  int thread_count = 0;
  int metal_reduce_interval = 10;
  int metal_tg_x = 0;
  int metal_tg_y = 0;
  int residual_interval = 0;  // 0 = disabled. If >0, report ||Au-b|| every N iterations.

  // SOR settings (used when method == SOR).
  double sor_omega = 1.5;

  // Krylov settings (used when method == CG/BiCGStab/GMRES).
  int gmres_restart = 30;

  // Multigrid V-cycle settings.
  int mg_pre_smooth = 2;
  int mg_post_smooth = 2;
  int mg_coarse_iters = 50;
  int mg_max_levels = 10;

  // Advection settings (for first-derivative terms c*u_x, d*u_y, dz*u_z).
  AdvectionScheme advection_scheme = AdvectionScheme::Centered;
  bool advection_cfl_check = false;  // Warn if CFL > 1 for explicit time stepping
};

// Time integration methods
enum class TimeIntegrator {
  ForwardEuler,    // First-order explicit Euler
  RK2,             // Second-order Runge-Kutta (Heun's method)
  RK4,             // Fourth-order classical Runge-Kutta
  SSPRK2,          // Second-order Strong Stability Preserving RK
  SSPRK3,          // Third-order SSP RK (Shu-Osher)
  BackwardEuler,   // First-order implicit
  CrankNicolson,   // Second-order implicit (trapezoidal)
  IMEX,            // Implicit-Explicit for advection-diffusion-reaction
};

struct TimeConfig {
  bool enabled = false;
  double t_start = 0.0;
  double t_end = 1.0;
  double dt = 0.01;
  int frames = 1;

  // Time integrator selection
  TimeIntegrator integrator = TimeIntegrator::ForwardEuler;

  // CFL-based time stepping
  bool use_cfl = false;
  double cfl_target = 0.5;

  // Adaptive time stepping
  bool adaptive_dt = false;
  double dt_min = 1e-12;
  double dt_max = 1.0;
  double error_tol = 1e-5;

  // Memory buffering for GPU time-stepping
  // Frames are computed in batches that fit within this memory limit,
  // reducing I/O overhead by decoupling GPU compute from file writes.
  int buffer_mb = 256;  // Default 256MB buffer
};

// Multi-field support for coupled PDE systems
struct FieldDefinition {
  std::string name;                     // Field name (e.g., "u", "v", "temperature")
  PDECoefficients pde;                  // PDE coefficients for this field
  BoundarySet bc;                       // Boundary conditions for this field
  std::vector<double> initial_grid;     // Initial state (checkpoint restart)
  std::vector<double> initial_velocity; // Velocity for u_tt stepping
};

struct FieldOutput {
  std::string name;
  std::vector<double> grid;
  double residual_l2 = 0.0;
  double residual_linf = 0.0;
};

// Forward declarations for coupling (full definitions below)
enum class CouplingStrategy {
  Explicit,    // Operator splitting: solve each field sequentially using previous values
  Picard,      // Block Gauss-Seidel / Picard iteration: iterate until coupled system converges
};

struct CouplingConfig {
  CouplingStrategy strategy = CouplingStrategy::Explicit;
  int max_coupling_iters = 100;        // Maximum Picard iterations (ignored for Explicit)
  double coupling_tol = 1e-6;          // Convergence tolerance for coupling loop
  bool use_relaxation = false;         // Under-relaxation for stability
  double relaxation_factor = 0.7;      // Relaxation factor (0 < factor <= 1)
};

struct CouplingDiagnostics {
  int coupling_iters = 0;                                // Number of coupling iterations used
  bool converged = true;                                 // Whether coupling loop converged
  std::vector<double> coupling_residual_history;         // L2 norm of field changes per iteration
  std::map<std::string, std::vector<double>> per_field_history;  // Per-field residual history
  std::string warning;                                   // Warning message if any issues
};

struct SolveInput {
  PDECoefficients pde;
  Domain domain;
  BoundarySet bc;
  SolverConfig solver;
  std::string domain_shape;
  ShapeTransform shape_transform;
  ShapeMask shape_mask;
  double shape_mask_threshold = 0.0;
  bool shape_mask_invert = false;
  BoundaryCondition embedded_bc;  // BC to apply on embedded boundary (if domain_shape is used)
  TimeConfig time;
  std::vector<IntegralTerm> integrals;
  std::vector<NonlinearTerm> nonlinear;
  std::vector<NonlinearDerivativeTerm> nonlinear_derivatives;  // e.g., u*u_x, |∇u|²
  std::vector<double> initial_grid;  // Optional initial state (e.g., checkpoint restart).
  std::vector<double> initial_velocity;  // Optional velocity for u_tt time stepping.
  std::atomic<bool>* cancel = nullptr;

  // Multi-field mode: if non-empty, overrides pde/bc for coupled systems
  std::vector<FieldDefinition> fields;

  // Coupling configuration for multi-field systems
  CouplingConfig coupling;
};

struct SolveOutput {
  std::string error;
  std::vector<double> grid;
  double residual_l2 = 0.0;
  double residual_linf = 0.0;
  std::vector<int> residual_iters;
  std::vector<double> residual_l2_history;
  std::vector<double> residual_linf_history;

  // Multi-field mode: per-field solutions and residuals
  std::vector<FieldOutput> field_outputs;

  // Coupling diagnostics for multi-field systems
  CouplingDiagnostics coupling_diagnostics;
};

// Multi-field helper functions
inline bool IsMultiField(const SolveInput& input) {
  return !input.fields.empty();
}

inline size_t FieldCount(const SolveInput& input) {
  return input.fields.empty() ? 1 : input.fields.size();
}

inline bool IsMultiFieldOutput(const SolveOutput& output) {
  return !output.field_outputs.empty();
}

inline bool HasShapeMask(const ShapeMask& mask) {
  return !mask.values.empty() || !mask.points.empty();
}

inline bool HasImplicitShape(const SolveInput& input) {
  return !input.domain_shape.empty() || HasShapeMask(input.shape_mask);
}

// Cross-field coupling support for coupled PDE systems
// Stores coefficients from one field appearing in another field's equation
struct CrossFieldCoefficients {
  std::string source_field;  // Field variable this term refers to (e.g., "v" in v_xx)
  PDECoefficients coeffs;    // Coefficients for derivatives of source_field
};

// Equation coefficients for a single field, including coupling to other fields
struct FieldEquationCoefficients {
  std::string field_name;                        // Field being solved (e.g., "u")
  PDECoefficients self_coeffs;                   // Coefficients for own-field terms (u_xx, u_t, etc.)
  std::vector<CrossFieldCoefficients> coupled;   // Coupling terms from other fields (v_xx, w_y, etc.)
};

// Multi-field equation system with cross-field coupling
struct MultiFieldEquation {
  std::vector<FieldEquationCoefficients> equations;  // One per field being solved

  // Helper to get all field names
  std::vector<std::string> GetFieldNames() const {
    std::vector<std::string> names;
    for (const auto& eq : equations) {
      names.push_back(eq.field_name);
    }
    return names;
  }

  // Check if any equation has coupling terms
  bool HasCoupling() const {
    for (const auto& eq : equations) {
      if (!eq.coupled.empty()) {
        return true;
      }
    }
    return false;
  }
};

// Coupling pattern classification for backend validation
enum class CouplingPattern {
  SingleField,        // No coupling (single PDE)
  ExplicitCoupling,   // One-way coupling (e.g., u_t = v_xx, v_t = v_yy)
  SymmetricCoupling,  // Two-way coupling (e.g., u_t = v_xx, v_t = u_yy)
  NonlinearCoupling,  // Coupling with nonlinear terms (u*v, etc.)
};

// Analysis result for coupling patterns
struct CouplingAnalysis {
  CouplingPattern pattern = CouplingPattern::SingleField;
  std::vector<std::string> fields;
  std::map<std::string, std::vector<std::string>> dependencies;  // field -> fields it depends on
  bool has_circular_dependency = false;
};

#endif
