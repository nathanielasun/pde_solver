#ifndef CPU_SOLVERS_KRYLOV_H
#define CPU_SOLVERS_KRYLOV_H

#include <vector>

#include "../cpu_utils.h"
#include "progress.h"

// Krylov solvers for 2D steady-state problems.
bool CgSolve(const LinearOperator2D& op,
             const std::vector<double>& b,
             std::vector<double>* x,
             int max_iter,
             double tol,
             int residual_interval,
             const ProgressCallback& progress);

bool BiCGStabSolve(const LinearOperator2D& op,
                   const std::vector<double>& b,
                   std::vector<double>* x,
                   int max_iter,
                   double tol,
                   int residual_interval,
                   const ProgressCallback& progress);

bool GmresSolve(const LinearOperator2D& op,
                const std::vector<double>& b,
                std::vector<double>* x,
                int max_iter,
                int restart,
                double tol,
                int residual_interval,
                const ProgressCallback& progress);

#endif  // CPU_SOLVERS_KRYLOV_H
