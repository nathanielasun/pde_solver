#ifndef COUPLED_EXAMPLES_H
#define COUPLED_EXAMPLES_H

#include <string>
#include <vector>

#include "pde_types.h"

// Multi-field coupled PDE example template
struct CoupledPDEExample {
  std::string name;
  std::string description;
  std::vector<FieldDefinition> fields;
  Domain domain;
  CouplingConfig coupling;
  TimeConfig time;
  SolverConfig solver;
  std::string notes;
};

// Get all available coupled PDE examples
std::vector<CoupledPDEExample> GetCoupledPDEExamples();

// Get a specific example by name
CoupledPDEExample GetCoupledPDEExample(const std::string& name);

// Build a SolveInput from a coupled example (ready to run)
SolveInput BuildSolveInputFromExample(const CoupledPDEExample& example);

// Gray-Scott reaction-diffusion system
// u_t = D_u * ∇²u - u*v² + F*(1-u)
// v_t = D_v * ∇²v + u*v² - (F+k)*v
CoupledPDEExample CreateGrayScottExample(
    double D_u = 0.16,   // Diffusion coefficient for u
    double D_v = 0.08,   // Diffusion coefficient for v
    double F = 0.04,     // Feed rate
    double k = 0.06,     // Kill rate
    int nx = 128,
    int ny = 128);

// Heat-diffusion multi-physics example
// T_t = k_T * ∇²T + alpha*C  (temperature with concentration coupling)
// C_t = D_C * ∇²C - beta*T   (concentration with temperature coupling)
CoupledPDEExample CreateHeatDiffusionExample(
    double k_T = 0.1,    // Thermal diffusivity
    double D_C = 0.05,   // Mass diffusivity
    double alpha = 0.01, // Coupling: C affects T
    double beta = 0.01,  // Coupling: T affects C
    int nx = 64,
    int ny = 64);

// Brusselator reaction-diffusion system
// u_t = D_u * ∇²u + A - (B+1)*u + u²*v
// v_t = D_v * ∇²v + B*u - u²*v
CoupledPDEExample CreateBrusselatorExample(
    double D_u = 1.0,
    double D_v = 8.0,
    double A = 4.5,
    double B = 8.0,
    int nx = 64,
    int ny = 64);

// Predator-prey (Lotka-Volterra) with diffusion
// u_t = D_u * ∇²u + alpha*u - beta*u*v   (prey)
// v_t = D_v * ∇²v - gamma*v + delta*u*v  (predator)
CoupledPDEExample CreatePredatorPreyExample(
    double D_u = 0.1,
    double D_v = 0.1,
    double alpha = 1.0,  // Prey growth rate
    double beta = 0.1,   // Predation rate
    double gamma = 1.0,  // Predator death rate
    double delta = 0.1,  // Predator growth from prey
    int nx = 64,
    int ny = 64);

#endif  // COUPLED_EXAMPLES_H
