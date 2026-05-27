#ifndef NONLINEAR_DERIVATIVES_H
#define NONLINEAR_DERIVATIVES_H

#include <vector>

#include "pde_types.h"

// Centered first derivatives at interior cell (i,j) on row-major 2D grid.
double CenteredDerivativeX(const std::vector<double>& grid, int i, int j, int nx, double dx);
double CenteredDerivativeY(const std::vector<double>& grid, int i, int j, int nx, int ny, double dy);
double CenteredDerivativeZ(const std::vector<double>& grid, int i, int j, int k, int nx, int ny, int nz,
                           double dz);

// Evaluate a single nonlinear derivative term at a grid point.
double EvalNonlinearDerivative(const NonlinearDerivativeTerm& term,
                               const std::vector<double>& grid,
                               int i, int j, int nx, int ny,
                               double dx, double dy);

double EvalNonlinearDerivative3D(const NonlinearDerivativeTerm& term,
                                 const std::vector<double>& grid,
                                 int i, int j, int k, int nx, int ny, int nz,
                                 double dx, double dy, double dz);

// Sum all nonlinear derivative contributions at a cell.
double AccumulateNonlinearDerivatives(const std::vector<NonlinearDerivativeTerm>& terms,
                                      const std::vector<double>& grid,
                                      int i, int j, int nx, int ny,
                                      double dx, double dy);

double AccumulateNonlinearDerivatives3D(const std::vector<NonlinearDerivativeTerm>& terms,
                                        const std::vector<double>& grid,
                                        int i, int j, int k, int nx, int ny, int nz,
                                        double dx, double dy, double dz);

// Estimate max wave speed |df/du| for Burgers-like u*u_x terms (|u|) on active interior.
double EstimateNonlinearAdvectionSpeed(const std::vector<NonlinearDerivativeTerm>& terms,
                                       const std::vector<double>& grid,
                                       int nx, int ny,
                                       const std::vector<unsigned char>* active);

#endif  // NONLINEAR_DERIVATIVES_H
