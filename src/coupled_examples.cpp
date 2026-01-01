#include "coupled_examples.h"

#include <cmath>
#include <sstream>

namespace {

// Helper to create Dirichlet BC with constant value
BoundaryCondition MakeDirichletBC(double value) {
  BoundaryCondition bc;
  bc.kind = BCKind::Dirichlet;
  bc.value.constant = value;
  return bc;
}

// Helper to create Neumann BC with constant flux
BoundaryCondition MakeNeumannBC(double flux) {
  BoundaryCondition bc;
  bc.kind = BCKind::Neumann;
  bc.value.constant = flux;
  return bc;
}

// Helper to create initial grid with Gaussian perturbation
std::vector<double> MakeGaussianPerturbation(int nx, int ny, double base_value,
                                              double cx, double cy, double sigma,
                                              double amplitude) {
  std::vector<double> grid(static_cast<size_t>(nx * ny), base_value);
  for (int j = 0; j < ny; ++j) {
    double y = static_cast<double>(j) / static_cast<double>(ny - 1);
    for (int i = 0; i < nx; ++i) {
      double x = static_cast<double>(i) / static_cast<double>(nx - 1);
      double dx = x - cx;
      double dy = y - cy;
      double r2 = dx * dx + dy * dy;
      grid[static_cast<size_t>(j * nx + i)] += amplitude * std::exp(-r2 / (2.0 * sigma * sigma));
    }
  }
  return grid;
}

// Helper to create initial grid with random perturbation
std::vector<double> MakeRandomPerturbation(int nx, int ny, double base_value,
                                            double amplitude, unsigned int seed) {
  std::vector<double> grid(static_cast<size_t>(nx * ny));
  // Simple LCG random number generator for reproducibility
  unsigned int state = seed;
  for (size_t i = 0; i < grid.size(); ++i) {
    state = state * 1103515245u + 12345u;
    double r = static_cast<double>((state >> 16) & 0x7fff) / 32767.0;
    grid[i] = base_value + amplitude * (2.0 * r - 1.0);
  }
  return grid;
}

// Helper to create initial grid with central square perturbation
std::vector<double> MakeCentralSquarePerturbation(int nx, int ny,
                                                   double base_value, double center_value,
                                                   double size_fraction) {
  std::vector<double> grid(static_cast<size_t>(nx * ny), base_value);
  int cx = nx / 2;
  int cy = ny / 2;
  int half_size = static_cast<int>(size_fraction * std::min(nx, ny) / 2);

  for (int j = cy - half_size; j <= cy + half_size; ++j) {
    if (j < 0 || j >= ny) continue;
    for (int i = cx - half_size; i <= cx + half_size; ++i) {
      if (i < 0 || i >= nx) continue;
      grid[static_cast<size_t>(j * nx + i)] = center_value;
    }
  }
  return grid;
}

}  // namespace

CoupledPDEExample CreateGrayScottExample(double D_u, double D_v, double F, double k,
                                          int nx, int ny) {
  CoupledPDEExample example;
  example.name = "Gray-Scott";

  std::ostringstream desc;
  desc << "Gray-Scott reaction-diffusion system with D_u=" << D_u
       << ", D_v=" << D_v << ", F=" << F << ", k=" << k;
  example.description = desc.str();

  // Domain: unit square
  example.domain.xmin = 0.0;
  example.domain.xmax = 1.0;
  example.domain.ymin = 0.0;
  example.domain.ymax = 1.0;
  example.domain.nx = nx;
  example.domain.ny = ny;
  example.domain.nz = 1;
  example.domain.coord_system = CoordinateSystem::Cartesian;

  // Field u: chemical U
  // u_t = D_u * (u_xx + u_yy) - u*v² + F*(1-u)
  // Note: The nonlinear terms u*v² are handled via nonlinear term infrastructure
  // For now, we set up the linear diffusion part; full reaction requires time stepping
  FieldDefinition field_u;
  field_u.name = "u";
  field_u.pde.ut = 1.0;  // u_t
  field_u.pde.a = D_u;   // D_u * u_xx
  field_u.pde.b = D_u;   // D_u * u_yy
  field_u.pde.e = -F;    // -F*u term (part of F*(1-u) = F - F*u)
  field_u.pde.f = F;     // +F constant (part of F*(1-u))
  // Neumann (no-flux) boundaries
  field_u.bc.left = MakeNeumannBC(0.0);
  field_u.bc.right = MakeNeumannBC(0.0);
  field_u.bc.bottom = MakeNeumannBC(0.0);
  field_u.bc.top = MakeNeumannBC(0.0);
  // Initial condition: u = 1 everywhere except center
  field_u.initial_grid = MakeCentralSquarePerturbation(nx, ny, 1.0, 0.5, 0.1);

  // Field v: chemical V
  // v_t = D_v * (v_xx + v_yy) + u*v² - (F+k)*v
  FieldDefinition field_v;
  field_v.name = "v";
  field_v.pde.ut = 1.0;      // v_t
  field_v.pde.a = D_v;       // D_v * v_xx
  field_v.pde.b = D_v;       // D_v * v_yy
  field_v.pde.e = -(F + k);  // -(F+k)*v
  // Neumann (no-flux) boundaries
  field_v.bc.left = MakeNeumannBC(0.0);
  field_v.bc.right = MakeNeumannBC(0.0);
  field_v.bc.bottom = MakeNeumannBC(0.0);
  field_v.bc.top = MakeNeumannBC(0.0);
  // Initial condition: v = 0 everywhere except center
  field_v.initial_grid = MakeCentralSquarePerturbation(nx, ny, 0.0, 0.25, 0.1);
  // Add small random perturbation for pattern formation
  for (size_t i = 0; i < field_v.initial_grid.size(); ++i) {
    if (field_v.initial_grid[i] > 0.0) {
      // Add 1% random noise to seeded region
      unsigned int state = static_cast<unsigned int>(i) * 1103515245u + 12345u;
      double r = static_cast<double>((state >> 16) & 0x7fff) / 32767.0;
      field_v.initial_grid[i] += 0.01 * (2.0 * r - 1.0);
    }
  }

  example.fields.push_back(field_u);
  example.fields.push_back(field_v);

  // Coupling: Use Picard iteration for tighter coupling
  example.coupling.strategy = CouplingStrategy::Picard;
  example.coupling.max_coupling_iters = 10;
  example.coupling.coupling_tol = 1e-6;
  example.coupling.use_relaxation = true;
  example.coupling.relaxation_factor = 0.8;

  // Time settings for pattern evolution
  example.time.enabled = true;
  example.time.t_start = 0.0;
  example.time.t_end = 1000.0;
  example.time.dt = 1.0;
  example.time.frames = 100;

  // Solver settings
  example.solver.method = SolveMethod::CG;
  example.solver.max_iter = 1000;
  example.solver.tol = 1e-8;

  example.notes = "Gray-Scott reaction-diffusion system produces complex patterns "
                  "like spots, stripes, and spirals. The nonlinear coupling terms "
                  "(u*v^2) require time stepping. Parameters F and k control pattern type: "
                  "F=0.04, k=0.06 produces mitosis-like spots.";

  return example;
}

CoupledPDEExample CreateHeatDiffusionExample(double k_T, double D_C,
                                              double alpha, double beta,
                                              int nx, int ny) {
  CoupledPDEExample example;
  example.name = "Heat-Diffusion Coupling";

  std::ostringstream desc;
  desc << "Coupled heat-diffusion system with k_T=" << k_T
       << ", D_C=" << D_C << ", alpha=" << alpha << ", beta=" << beta;
  example.description = desc.str();

  // Domain: unit square
  example.domain.xmin = 0.0;
  example.domain.xmax = 1.0;
  example.domain.ymin = 0.0;
  example.domain.ymax = 1.0;
  example.domain.nx = nx;
  example.domain.ny = ny;
  example.domain.nz = 1;
  example.domain.coord_system = CoordinateSystem::Cartesian;

  // Field T: Temperature
  // T_t = k_T * (T_xx + T_yy) + alpha*C
  // Note: The alpha*C coupling is handled implicitly via coupling loop
  FieldDefinition field_T;
  field_T.name = "T";
  field_T.pde.ut = 1.0;   // T_t
  field_T.pde.a = k_T;    // k_T * T_xx
  field_T.pde.b = k_T;    // k_T * T_yy
  // Dirichlet boundaries: hot left, cold right
  field_T.bc.left = MakeDirichletBC(1.0);
  field_T.bc.right = MakeDirichletBC(0.0);
  field_T.bc.bottom = MakeNeumannBC(0.0);  // Insulated
  field_T.bc.top = MakeNeumannBC(0.0);     // Insulated
  // Initial condition: linear gradient
  field_T.initial_grid.resize(static_cast<size_t>(nx * ny));
  for (int j = 0; j < ny; ++j) {
    for (int i = 0; i < nx; ++i) {
      double x = static_cast<double>(i) / static_cast<double>(nx - 1);
      field_T.initial_grid[static_cast<size_t>(j * nx + i)] = 1.0 - x;
    }
  }

  // Field C: Concentration
  // C_t = D_C * (C_xx + C_yy) - beta*T
  FieldDefinition field_C;
  field_C.name = "C";
  field_C.pde.ut = 1.0;   // C_t
  field_C.pde.a = D_C;    // D_C * C_xx
  field_C.pde.b = D_C;    // D_C * C_yy
  // Dirichlet boundaries: high concentration at bottom, low at top
  field_C.bc.left = MakeNeumannBC(0.0);    // No flux
  field_C.bc.right = MakeNeumannBC(0.0);   // No flux
  field_C.bc.bottom = MakeDirichletBC(1.0);
  field_C.bc.top = MakeDirichletBC(0.0);
  // Initial condition: Gaussian blob
  field_C.initial_grid = MakeGaussianPerturbation(nx, ny, 0.0, 0.5, 0.25, 0.1, 1.0);

  example.fields.push_back(field_T);
  example.fields.push_back(field_C);

  // Coupling: Explicit for simplicity, Picard for accuracy
  example.coupling.strategy = CouplingStrategy::Explicit;
  example.coupling.max_coupling_iters = 50;
  example.coupling.coupling_tol = 1e-6;

  // Time settings
  example.time.enabled = true;
  example.time.t_start = 0.0;
  example.time.t_end = 10.0;
  example.time.dt = 0.01;
  example.time.frames = 100;

  // Solver settings
  example.solver.method = SolveMethod::CG;
  example.solver.max_iter = 1000;
  example.solver.tol = 1e-8;

  example.notes = "Simplified multi-physics example coupling temperature and "
                  "concentration fields. Temperature affects concentration decay, "
                  "concentration affects heat generation. Demonstrates explicit "
                  "coupling strategy.";

  return example;
}

CoupledPDEExample CreateBrusselatorExample(double D_u, double D_v,
                                            double A, double B,
                                            int nx, int ny) {
  CoupledPDEExample example;
  example.name = "Brusselator";

  std::ostringstream desc;
  desc << "Brusselator reaction-diffusion system with D_u=" << D_u
       << ", D_v=" << D_v << ", A=" << A << ", B=" << B;
  example.description = desc.str();

  // Domain: larger square for pattern formation
  example.domain.xmin = 0.0;
  example.domain.xmax = 64.0;
  example.domain.ymin = 0.0;
  example.domain.ymax = 64.0;
  example.domain.nx = nx;
  example.domain.ny = ny;
  example.domain.nz = 1;
  example.domain.coord_system = CoordinateSystem::Cartesian;

  // Field u:
  // u_t = D_u * (u_xx + u_yy) + A - (B+1)*u + u²*v
  FieldDefinition field_u;
  field_u.name = "u";
  field_u.pde.ut = 1.0;       // u_t
  field_u.pde.a = D_u;        // D_u * u_xx
  field_u.pde.b = D_u;        // D_u * u_yy
  field_u.pde.e = -(B + 1.0); // -(B+1)*u
  field_u.pde.f = A;          // +A constant
  // Neumann boundaries
  field_u.bc.left = MakeNeumannBC(0.0);
  field_u.bc.right = MakeNeumannBC(0.0);
  field_u.bc.bottom = MakeNeumannBC(0.0);
  field_u.bc.top = MakeNeumannBC(0.0);
  // Initial condition: steady state u = A with small perturbation
  field_u.initial_grid = MakeRandomPerturbation(nx, ny, A, 0.1, 42);

  // Field v:
  // v_t = D_v * (v_xx + v_yy) + B*u - u²*v
  FieldDefinition field_v;
  field_v.name = "v";
  field_v.pde.ut = 1.0;  // v_t
  field_v.pde.a = D_v;   // D_v * v_xx
  field_v.pde.b = D_v;   // D_v * v_yy
  // Note: B*u term is cross-field coupling, needs to be handled
  // Neumann boundaries
  field_v.bc.left = MakeNeumannBC(0.0);
  field_v.bc.right = MakeNeumannBC(0.0);
  field_v.bc.bottom = MakeNeumannBC(0.0);
  field_v.bc.top = MakeNeumannBC(0.0);
  // Initial condition: steady state v = B/A with small perturbation
  field_v.initial_grid = MakeRandomPerturbation(nx, ny, B / A, 0.1, 123);

  example.fields.push_back(field_u);
  example.fields.push_back(field_v);

  // Coupling
  example.coupling.strategy = CouplingStrategy::Picard;
  example.coupling.max_coupling_iters = 20;
  example.coupling.coupling_tol = 1e-6;
  example.coupling.use_relaxation = true;
  example.coupling.relaxation_factor = 0.7;

  // Time settings
  example.time.enabled = true;
  example.time.t_start = 0.0;
  example.time.t_end = 100.0;
  example.time.dt = 0.1;
  example.time.frames = 100;

  // Solver
  example.solver.method = SolveMethod::CG;
  example.solver.max_iter = 1000;
  example.solver.tol = 1e-8;

  example.notes = "Brusselator reaction-diffusion system, a classic model for "
                  "oscillating chemical reactions. When B > 1 + A², the system "
                  "becomes unstable and forms Turing patterns.";

  return example;
}

CoupledPDEExample CreatePredatorPreyExample(double D_u, double D_v,
                                             double alpha, double beta,
                                             double gamma, double delta,
                                             int nx, int ny) {
  CoupledPDEExample example;
  example.name = "Predator-Prey";

  std::ostringstream desc;
  desc << "Lotka-Volterra predator-prey with diffusion: "
       << "alpha=" << alpha << ", beta=" << beta
       << ", gamma=" << gamma << ", delta=" << delta;
  example.description = desc.str();

  // Domain
  example.domain.xmin = 0.0;
  example.domain.xmax = 50.0;
  example.domain.ymin = 0.0;
  example.domain.ymax = 50.0;
  example.domain.nx = nx;
  example.domain.ny = ny;
  example.domain.nz = 1;
  example.domain.coord_system = CoordinateSystem::Cartesian;

  // Field u: prey population
  // u_t = D_u * (u_xx + u_yy) + alpha*u - beta*u*v
  FieldDefinition field_u;
  field_u.name = "prey";
  field_u.pde.ut = 1.0;     // u_t
  field_u.pde.a = D_u;      // diffusion
  field_u.pde.b = D_u;
  field_u.pde.e = alpha;    // growth term (linearized)
  // No-flux boundaries (closed ecosystem)
  field_u.bc.left = MakeNeumannBC(0.0);
  field_u.bc.right = MakeNeumannBC(0.0);
  field_u.bc.bottom = MakeNeumannBC(0.0);
  field_u.bc.top = MakeNeumannBC(0.0);
  // Initial: prey distributed with local clusters
  field_u.initial_grid.resize(static_cast<size_t>(nx * ny));
  for (int j = 0; j < ny; ++j) {
    double y = static_cast<double>(j) / static_cast<double>(ny - 1);
    for (int i = 0; i < nx; ++i) {
      double x = static_cast<double>(i) / static_cast<double>(nx - 1);
      // Base population with spatial variation
      field_u.initial_grid[static_cast<size_t>(j * nx + i)] =
          1.0 + 0.5 * std::sin(4.0 * M_PI * x) * std::sin(4.0 * M_PI * y);
    }
  }

  // Field v: predator population
  // v_t = D_v * (v_xx + v_yy) - gamma*v + delta*u*v
  FieldDefinition field_v;
  field_v.name = "predator";
  field_v.pde.ut = 1.0;
  field_v.pde.a = D_v;
  field_v.pde.b = D_v;
  field_v.pde.e = -gamma;  // death term
  // No-flux boundaries
  field_v.bc.left = MakeNeumannBC(0.0);
  field_v.bc.right = MakeNeumannBC(0.0);
  field_v.bc.bottom = MakeNeumannBC(0.0);
  field_v.bc.top = MakeNeumannBC(0.0);
  // Initial: predators in different locations
  field_v.initial_grid.resize(static_cast<size_t>(nx * ny));
  for (int j = 0; j < ny; ++j) {
    double y = static_cast<double>(j) / static_cast<double>(ny - 1);
    for (int i = 0; i < nx; ++i) {
      double x = static_cast<double>(i) / static_cast<double>(nx - 1);
      field_v.initial_grid[static_cast<size_t>(j * nx + i)] =
          0.5 + 0.3 * std::cos(4.0 * M_PI * x) * std::cos(4.0 * M_PI * y);
    }
  }

  example.fields.push_back(field_u);
  example.fields.push_back(field_v);

  // Coupling
  example.coupling.strategy = CouplingStrategy::Explicit;
  example.coupling.max_coupling_iters = 10;
  example.coupling.coupling_tol = 1e-5;

  // Time
  example.time.enabled = true;
  example.time.t_start = 0.0;
  example.time.t_end = 50.0;
  example.time.dt = 0.1;
  example.time.frames = 100;

  // Solver
  example.solver.method = SolveMethod::CG;
  example.solver.max_iter = 1000;
  example.solver.tol = 1e-8;

  example.notes = "Lotka-Volterra predator-prey model with spatial diffusion. "
                  "Produces traveling waves and spiral patterns in 2D. "
                  "Nonlinear interaction terms (u*v) require time stepping.";

  return example;
}

std::vector<CoupledPDEExample> GetCoupledPDEExamples() {
  std::vector<CoupledPDEExample> examples;

  // Standard Gray-Scott patterns
  examples.push_back(CreateGrayScottExample(0.16, 0.08, 0.04, 0.06, 128, 128));

  // Gray-Scott with different parameters (coral/maze patterns)
  examples.push_back(CreateGrayScottExample(0.16, 0.08, 0.035, 0.065, 128, 128));

  // Heat-diffusion multi-physics
  examples.push_back(CreateHeatDiffusionExample(0.1, 0.05, 0.01, 0.01, 64, 64));

  // Brusselator
  examples.push_back(CreateBrusselatorExample(1.0, 8.0, 4.5, 8.0, 64, 64));

  // Predator-prey
  examples.push_back(CreatePredatorPreyExample(0.1, 0.1, 1.0, 0.1, 1.0, 0.1, 64, 64));

  return examples;
}

CoupledPDEExample GetCoupledPDEExample(const std::string& name) {
  auto examples = GetCoupledPDEExamples();
  for (const auto& ex : examples) {
    if (ex.name == name) {
      return ex;
    }
  }
  // Return first example as default
  return examples.empty() ? CoupledPDEExample{} : examples[0];
}

SolveInput BuildSolveInputFromExample(const CoupledPDEExample& example) {
  SolveInput input;

  input.domain = example.domain;
  input.solver = example.solver;
  input.time = example.time;
  input.coupling = example.coupling;
  input.fields = example.fields;

  // Set up primary PDE from first field (for single-field fallback)
  if (!example.fields.empty()) {
    input.pde = example.fields[0].pde;
    input.bc = example.fields[0].bc;
    if (!example.fields[0].initial_grid.empty()) {
      input.initial_grid = example.fields[0].initial_grid;
    }
    if (!example.fields[0].initial_velocity.empty()) {
      input.initial_velocity = example.fields[0].initial_velocity;
    }
  }

  return input;
}
