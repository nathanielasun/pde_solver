#ifndef FV_DISCRETIZATION_H
#define FV_DISCRETIZATION_H

#include <functional>
#include <vector>

#include "fv/flux.h"
#include "pde_types.h"

// Semidiscrete RHS: dU/dt[i] = -(F_{i+1/2} - F_{i-1/2}) / dx (+ diffusion if present)
void ComputeFVSemidiscreteRHS1D(const std::vector<double>& u,
                                int nx,
                                double dx,
                                const FluxEvaluator& flux,
                                const ConservationLawConfig& config,
                                const std::function<double(int)>& left_bc,
                                const std::function<double(int)>& right_bc,
                                std::vector<double>* dudt);

void ComputeFVSemidiscreteRHS1D(const std::vector<double>& u,
                                int nx,
                                double dx,
                                const FluxEvaluator& flux,
                                const ConservationLawConfig& config,
                                const BoundaryCondition& left_bc,
                                const BoundaryCondition& right_bc,
                                std::vector<double>* dudt);

void ComputeFVSemidiscreteRHS2D(const std::vector<double>& u,
                                int nx, int ny,
                                double dx, double dy,
                                const FluxEvaluator& flux,
                                const ConservationLawConfig& config,
                                double diffusion_coeff,
                                std::vector<double>* dudt);

void ComputeFVSemidiscreteRHS2D(const std::vector<double>& u,
                                int nx, int ny,
                                double dx, double dy,
                                const FluxEvaluator& flux,
                                const ConservationLawConfig& config,
                                double diffusion_coeff,
                                const BoundarySet& bc,
                                std::vector<double>* dudt);

#endif  // FV_DISCRETIZATION_H
