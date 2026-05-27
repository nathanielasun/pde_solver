#ifndef FV_RIEMANN_H
#define FV_RIEMANN_H

#include "fv/flux.h"
#include "pde_types.h"

double RiemannFlux(double u_left, double u_right, const FluxEvaluator& flux,
                   ConservationLawConfig::RiemannSolver solver);

#endif  // FV_RIEMANN_H
