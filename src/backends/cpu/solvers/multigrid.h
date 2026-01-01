#ifndef CPU_SOLVERS_MULTIGRID_H
#define CPU_SOLVERS_MULTIGRID_H

#include <vector>

#include "pde_types.h"

bool CanCoarsenOdd(int n);
void RestrictFullWeighting(const std::vector<double>& fine, int nx_f, int ny_f,
                           std::vector<double>* coarse, int nx_c, int ny_c);
void ProlongBilinearAdd(const std::vector<double>& coarse, int nx_c, int ny_c,
                        std::vector<double>* fine, int nx_f, int ny_f);
void SmoothGsPoisson(const Domain& d,
                     double ax, double by, double center,
                     const std::vector<double>& b,
                     std::vector<double>* u,
                     int iterations);
void ResidualPoisson(const Domain& d,
                     double ax, double by, double center,
                     const std::vector<double>& b,
                     const std::vector<double>& u,
                     std::vector<double>* r_out);
bool MultigridVcyclePoisson(const Domain& fine_d,
                            double a, double bcoef, double e,
                            const std::vector<double>& b_fine,
                            std::vector<double>* u_fine,
                            int pre_smooth, int post_smooth,
                            int coarse_iters,
                            int max_levels);

#endif  // CPU_SOLVERS_MULTIGRID_H

