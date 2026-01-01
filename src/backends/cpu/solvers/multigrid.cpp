#include "multigrid.h"

#include <algorithm>
#include <cmath>
#include <vector>

#include "../cpu_utils.h"
#include "finite_differences.h"

bool CanCoarsenOdd(int n) {
  return n >= 5 && (n % 2 == 1);
}

void RestrictFullWeighting(const std::vector<double>& fine, int nx_f, int ny_f,
                           std::vector<double>* coarse, int nx_c, int ny_c) {
  if (!coarse) {
    return;
  }
  coarse->assign(static_cast<size_t>(nx_c * ny_c), 0.0);
  for (int jc = 1; jc < ny_c - 1; ++jc) {
    const int jf = 2 * jc;
    for (int ic = 1; ic < nx_c - 1; ++ic) {
      const int ifx = 2 * ic;
      const double c = fine[static_cast<size_t>(Index(ifx, jf, nx_f))] * 0.25;
      const double e = (fine[static_cast<size_t>(Index(ifx - 1, jf, nx_f))] +
                        fine[static_cast<size_t>(Index(ifx + 1, jf, nx_f))] +
                        fine[static_cast<size_t>(Index(ifx, jf - 1, nx_f))] +
                        fine[static_cast<size_t>(Index(ifx, jf + 1, nx_f))]) * 0.125;
      const double d = (fine[static_cast<size_t>(Index(ifx - 1, jf - 1, nx_f))] +
                        fine[static_cast<size_t>(Index(ifx + 1, jf - 1, nx_f))] +
                        fine[static_cast<size_t>(Index(ifx - 1, jf + 1, nx_f))] +
                        fine[static_cast<size_t>(Index(ifx + 1, jf + 1, nx_f))]) * 0.0625;
      (*coarse)[static_cast<size_t>(Index(ic, jc, nx_c))] = c + e + d;
    }
  }
}

void ProlongBilinearAdd(const std::vector<double>& coarse, int nx_c, int ny_c,
                        std::vector<double>* fine, int nx_f, int ny_f) {
  if (!fine) {
    return;
  }
  for (int jc = 0; jc < ny_c; ++jc) {
    for (int ic = 0; ic < nx_c; ++ic) {
      const double vc = coarse[static_cast<size_t>(Index(ic, jc, nx_c))];
      const int ifx = 2 * ic;
      const int jf = 2 * jc;
      if (ifx < nx_f && jf < ny_f) {
        (*fine)[static_cast<size_t>(Index(ifx, jf, nx_f))] += vc;
      }
      if (ic + 1 < nx_c && ifx + 1 < nx_f && jf < ny_f) {
        const double vr = coarse[static_cast<size_t>(Index(ic + 1, jc, nx_c))];
        (*fine)[static_cast<size_t>(Index(ifx + 1, jf, nx_f))] += 0.5 * (vc + vr);
      }
      if (jc + 1 < ny_c && ifx < nx_f && jf + 1 < ny_f) {
        const double vu = coarse[static_cast<size_t>(Index(ic, jc + 1, nx_c))];
        (*fine)[static_cast<size_t>(Index(ifx, jf + 1, nx_f))] += 0.5 * (vc + vu);
      }
      if (ic + 1 < nx_c && jc + 1 < ny_c && ifx + 1 < nx_f && jf + 1 < ny_f) {
        const double vru = coarse[static_cast<size_t>(Index(ic + 1, jc + 1, nx_c))];
        const double vr = coarse[static_cast<size_t>(Index(ic + 1, jc, nx_c))];
        const double vu = coarse[static_cast<size_t>(Index(ic, jc + 1, nx_c))];
        (*fine)[static_cast<size_t>(Index(ifx + 1, jf + 1, nx_f))] += 0.25 * (vc + vr + vu + vru);
      }
    }
  }
}

void SmoothGsPoisson(const Domain& d,
                     double ax, double by, double center,
                     const std::vector<double>& b,
                     std::vector<double>* u,
                     int iterations) {
  if (!u) {
    return;
  }
  const int nx = d.nx;
  const int ny = d.ny;
  for (int iter = 0; iter < iterations; ++iter) {
    for (int j = 1; j < ny - 1; ++j) {
      for (int i = 1; i < nx - 1; ++i) {
        const int idx = Index(i, j, nx);
        const double u_left = (*u)[static_cast<size_t>(Index(i - 1, j, nx))];
        const double u_right = (*u)[static_cast<size_t>(Index(i + 1, j, nx))];
        const double u_down = (*u)[static_cast<size_t>(Index(i, j - 1, nx))];
        const double u_up = (*u)[static_cast<size_t>(Index(i, j + 1, nx))];
        const double rhs = b[static_cast<size_t>(idx)]
          - ax * (u_left + u_right)
          - by * (u_down + u_up);
        (*u)[static_cast<size_t>(idx)] = rhs / center;
      }
    }
  }
}

void ResidualPoisson(const Domain& d,
                     double ax, double by, double center,
                     const std::vector<double>& b,
                     const std::vector<double>& u,
                     std::vector<double>* r_out) {
  if (!r_out) {
    return;
  }
  const int nx = d.nx;
  const int ny = d.ny;
  r_out->assign(static_cast<size_t>(nx * ny), 0.0);
  for (int j = 1; j < ny - 1; ++j) {
    for (int i = 1; i < nx - 1; ++i) {
      const int idx = Index(i, j, nx);
      const double u_c = u[static_cast<size_t>(idx)];
      const double u_left = u[static_cast<size_t>(Index(i - 1, j, nx))];
      const double u_right = u[static_cast<size_t>(Index(i + 1, j, nx))];
      const double u_down = u[static_cast<size_t>(Index(i, j - 1, nx))];
      const double u_up = u[static_cast<size_t>(Index(i, j + 1, nx))];
      const double Au = center * u_c + ax * (u_left + u_right) + by * (u_down + u_up);
      (*r_out)[static_cast<size_t>(idx)] = b[static_cast<size_t>(idx)] - Au;
    }
  }
}

bool MultigridVcyclePoisson(const Domain& fine_d,
                            double a, double bcoef, double e,
                            const std::vector<double>& b_fine,
                            std::vector<double>* u_fine,
                            int pre_smooth, int post_smooth,
                            int coarse_iters,
                            int max_levels) {
  if (!u_fine) {
    return false;
  }
  if (!CanCoarsenOdd(fine_d.nx) || !CanCoarsenOdd(fine_d.ny)) {
    return false;
  }

  struct Level {
    Domain d;
    std::vector<double> u;
    std::vector<double> b;
    std::vector<double> r;
  };
  std::vector<Level> levels;
  levels.reserve(static_cast<size_t>(std::max(1, max_levels)));

  Level finest;
  finest.d = fine_d;
  finest.u = *u_fine;
  finest.b = b_fine;
  levels.push_back(std::move(finest));

  while (static_cast<int>(levels.size()) < max_levels) {
    const Level& prev = levels.back();
    if (!CanCoarsenOdd(prev.d.nx) || !CanCoarsenOdd(prev.d.ny)) {
      break;
    }
    const int nx_c = (prev.d.nx + 1) / 2;
    const int ny_c = (prev.d.ny + 1) / 2;
    if (nx_c < 3 || ny_c < 3) {
      break;
    }
    Level coarse;
    coarse.d = prev.d;
    coarse.d.nx = nx_c;
    coarse.d.ny = ny_c;
    coarse.u.assign(static_cast<size_t>(nx_c * ny_c), 0.0);
    coarse.b.assign(static_cast<size_t>(nx_c * ny_c), 0.0);
    levels.push_back(std::move(coarse));
    if (nx_c <= 5 || ny_c <= 5) {
      break;
    }
  }

  auto coeffs_for = [&](const Domain& d, double* ax, double* by, double* center) {
    const double dx = (d.xmax - d.xmin) / static_cast<double>(d.nx - 1);
    const double dy = (d.ymax - d.ymin) / static_cast<double>(d.ny - 1);
    *ax = a / (dx * dx);
    *by = bcoef / (dy * dy);
    *center = -2.0 * (*ax) - 2.0 * (*by) + e;
  };

  const int L = static_cast<int>(levels.size());
  for (int ell = 0; ell < L - 1; ++ell) {
    double ax = 0.0, by = 0.0, center = 0.0;
    coeffs_for(levels[ell].d, &ax, &by, &center);
    if (std::abs(center) < 1e-12) {
      return false;
    }
    SmoothGsPoisson(levels[ell].d, ax, by, center, levels[ell].b, &levels[ell].u,
                    std::max(0, pre_smooth));
    ResidualPoisson(levels[ell].d, ax, by, center, levels[ell].b, levels[ell].u, &levels[ell].r);
    RestrictFullWeighting(levels[ell].r, levels[ell].d.nx, levels[ell].d.ny,
                          &levels[ell + 1].b, levels[ell + 1].d.nx, levels[ell + 1].d.ny);
    std::fill(levels[ell + 1].u.begin(), levels[ell + 1].u.end(), 0.0);
  }

  {
    Level& coarse = levels.back();
    double ax = 0.0, by = 0.0, center = 0.0;
    coeffs_for(coarse.d, &ax, &by, &center);
    if (std::abs(center) < 1e-12) {
      return false;
    }
    SmoothGsPoisson(coarse.d, ax, by, center, coarse.b, &coarse.u,
                    std::max(1, coarse_iters));
  }

  for (int ell = L - 2; ell >= 0; --ell) {
    Level& fine = levels[ell];
    Level& coarse = levels[ell + 1];
    ProlongBilinearAdd(coarse.u, coarse.d.nx, coarse.d.ny,
                       &fine.u, fine.d.nx, fine.d.ny);
    double ax = 0.0, by = 0.0, center = 0.0;
    coeffs_for(fine.d, &ax, &by, &center);
    if (std::abs(center) < 1e-12) {
      return false;
    }
    SmoothGsPoisson(fine.d, ax, by, center, fine.b, &fine.u,
                    std::max(0, post_smooth));
  }

  *u_fine = levels.front().u;
  return true;
}

