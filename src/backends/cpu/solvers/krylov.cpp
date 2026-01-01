#include "krylov.h"

#include <algorithm>
#include <cmath>
#include <mutex>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "../cpu_utils.h"

bool CgSolve(const LinearOperator2D& op,
             const std::vector<double>& b,
             std::vector<double>* x,
             int max_iter,
             double tol,
             int residual_interval,
             const ProgressCallback& progress) {
  if (!x) {
    return false;
  }
  const size_t n = b.size();
  x->assign(n, 0.0);

  std::vector<double> r(n, 0.0), p(n, 0.0), Ap(n, 0.0);
  op.Apply(*x, &Ap);
  for (size_t i = 0; i < n; ++i) {
    r[i] = b[i] - Ap[i];
    p[i] = r[i];
  }

  double rr = Dot(r, r);
  if (progress && residual_interval > 0) {
    progress("residual_l2", std::sqrt(rr));
    progress("residual_linf", NormInf(r));
  }
  if (std::sqrt(rr) < tol) {
    return true;
  }

  for (int iter = 0; iter < max_iter; ++iter) {
    op.Apply(p, &Ap);
    const double pAp = Dot(p, Ap);
    if (std::abs(pAp) < 1e-30) {
      return false;
    }
    const double alpha = rr / pAp;
#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (size_t i = 0; i < n; ++i) {
      (*x)[i] += alpha * p[i];
      r[i] -= alpha * Ap[i];
    }

    const double rr_new = Dot(r, r);
    if (progress) {
      progress("solve_total", static_cast<double>(iter + 1) / static_cast<double>(std::max(1, max_iter)));
      if (residual_interval > 0 && (iter % residual_interval) == 0) {
        progress("residual_l2", std::sqrt(rr_new));
        progress("residual_linf", NormInf(r));
      }
    }
    if (std::sqrt(rr_new) < tol) {
      return true;
    }

    const double beta = rr_new / rr;
#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (size_t i = 0; i < n; ++i) {
      p[i] = r[i] + beta * p[i];
    }
    rr = rr_new;
  }
  return false;
}

bool BiCGStabSolve(const LinearOperator2D& op,
                   const std::vector<double>& b,
                   std::vector<double>* x,
                   int max_iter,
                   double tol,
                   int residual_interval,
                   const ProgressCallback& progress) {
  if (!x) {
    return false;
  }
  const size_t n = b.size();
  x->assign(n, 0.0);

  std::vector<double> r(n, 0.0), rhat(n, 0.0), p(n, 0.0), v(n, 0.0), s(n, 0.0), t(n, 0.0);
  std::vector<double> Ax(n, 0.0);
  op.Apply(*x, &Ax);
  for (size_t i = 0; i < n; ++i) {
    r[i] = b[i] - Ax[i];
    rhat[i] = r[i];
  }
  double rho_prev = 1.0;
  double alpha = 1.0;
  double omega = 1.0;
  double resid0 = Norm2(r);
  if (progress && residual_interval > 0) {
    progress("residual_l2", resid0);
    progress("residual_linf", NormInf(r));
  }
  if (resid0 < tol) {
    return true;
  }

  for (int iter = 0; iter < max_iter; ++iter) {
    const double rho = Dot(rhat, r);
    if (std::abs(rho) < 1e-30) {
      return false;
    }
    if (iter == 0) {
      p = r;
    } else {
      const double beta = (rho / rho_prev) * (alpha / omega);
#ifdef _OPENMP
      #pragma omp parallel for schedule(static)
#endif
      for (size_t i = 0; i < n; ++i) {
        p[i] = r[i] + beta * (p[i] - omega * v[i]);
      }
    }
    op.Apply(p, &v);
    const double denom = Dot(rhat, v);
    if (std::abs(denom) < 1e-30) {
      return false;
    }
    alpha = rho / denom;
#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (size_t i = 0; i < n; ++i) {
      s[i] = r[i] - alpha * v[i];
    }
    const double s_norm = Norm2(s);
    if (s_norm < tol) {
      for (size_t i = 0; i < n; ++i) {
        (*x)[i] += alpha * p[i];
      }
      return true;
    }
    op.Apply(s, &t);
    const double tt = Dot(t, t);
    if (std::abs(tt) < 1e-30) {
      return false;
    }
    omega = Dot(t, s) / tt;
#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (size_t i = 0; i < n; ++i) {
      (*x)[i] += alpha * p[i] + omega * s[i];
    }
    op.Apply(*x, &Ax);
#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (size_t i = 0; i < n; ++i) {
      r[i] = b[i] - Ax[i];
    }
    const double resid = Norm2(r);
    if (progress) {
      progress("solve_total", static_cast<double>(iter + 1) / static_cast<double>(std::max(1, max_iter)));
      if (residual_interval > 0 && (iter % residual_interval) == 0) {
        progress("residual_l2", resid);
        progress("residual_linf", NormInf(r));
      }
    }
    if (resid < tol) {
      return true;
    }
    if (std::abs(omega) < 1e-30) {
      return false;
    }
    rho_prev = rho;
  }
  return false;
}

bool GmresSolve(const LinearOperator2D& op,
                const std::vector<double>& b,
                std::vector<double>* x,
                int max_iter,
                int restart,
                double tol,
                int residual_interval,
                const ProgressCallback& progress) {
  if (!x) {
    return false;
  }
  const size_t n = b.size();
  x->assign(n, 0.0);
  restart = std::max(1, restart);
  max_iter = std::max(1, max_iter);

  std::vector<double> Ax(n, 0.0);
  std::vector<double> r(n, 0.0);
  std::vector<double> w(n, 0.0);

  auto apply_A = [&](const std::vector<double>& in, std::vector<double>* out) {
    op.Apply(in, out);
  };

  apply_A(*x, &Ax);
  for (size_t i = 0; i < n; ++i) {
    r[i] = b[i] - Ax[i];
  }
  double beta = Norm2(r);
  if (progress && residual_interval > 0) {
    progress("residual_l2", beta);
    progress("residual_linf", NormInf(r));
  }
  if (beta < tol) {
    return true;
  }

  int iter_total = 0;
  while (iter_total < max_iter) {
    const int m = std::min(restart, max_iter - iter_total);
    std::vector<std::vector<double>> V(static_cast<size_t>(m + 1),
                                       std::vector<double>(n, 0.0));
    std::vector<std::vector<double>> H(static_cast<size_t>(m + 1),
                                       std::vector<double>(static_cast<size_t>(m), 0.0));
    std::vector<double> cs(static_cast<size_t>(m), 0.0);
    std::vector<double> sn(static_cast<size_t>(m), 0.0);
    std::vector<double> g(static_cast<size_t>(m + 1), 0.0);
    g[0] = beta;

    for (size_t i = 0; i < n; ++i) {
      V[0][i] = r[i] / beta;
    }

    auto apply_givens = [&](double v1, double v2, size_t k, double* out1, double* out2) {
      const double cs_k = cs[k];
      const double sn_k = sn[k];
      *out1 = cs_k * v1 - sn_k * v2;
      *out2 = sn_k * v1 + cs_k * v2;
    };

    for (int j = 0; j < m; ++j) {
      apply_A(V[j], &w);
      for (int i = 0; i <= j; ++i) {
        H[static_cast<size_t>(i)][static_cast<size_t>(j)] = Dot(w, V[static_cast<size_t>(i)]);
        for (size_t k = 0; k < n; ++k) {
          w[k] -= H[static_cast<size_t>(i)][static_cast<size_t>(j)] * V[static_cast<size_t>(i)][k];
        }
      }
      H[static_cast<size_t>(j + 1)][static_cast<size_t>(j)] = Norm2(w);
      if (H[static_cast<size_t>(j + 1)][static_cast<size_t>(j)] > 0.0) {
        for (size_t k = 0; k < n; ++k) {
          V[static_cast<size_t>(j + 1)][k] = w[k] / H[static_cast<size_t>(j + 1)][static_cast<size_t>(j)];
        }
      }

      for (int i = 0; i < j; ++i) {
        double temp1 = 0.0, temp2 = 0.0;
        apply_givens(H[static_cast<size_t>(i)][static_cast<size_t>(j)],
                     H[static_cast<size_t>(i + 1)][static_cast<size_t>(j)],
                     static_cast<size_t>(i), &temp1, &temp2);
        H[static_cast<size_t>(i)][static_cast<size_t>(j)] = temp1;
        H[static_cast<size_t>(i + 1)][static_cast<size_t>(j)] = temp2;
      }

      const double h_j1j = H[static_cast<size_t>(j + 1)][static_cast<size_t>(j)];
      const double h_jj = H[static_cast<size_t>(j)][static_cast<size_t>(j)];
      const double denom = std::sqrt(h_jj * h_jj + h_j1j * h_j1j);
      if (denom < 1e-30) {
        return false;
      }
      cs[static_cast<size_t>(j)] = h_jj / denom;
      sn[static_cast<size_t>(j)] = -h_j1j / denom;
      H[static_cast<size_t>(j)][static_cast<size_t>(j)] = cs[static_cast<size_t>(j)] * h_jj - sn[static_cast<size_t>(j)] * h_j1j;
      H[static_cast<size_t>(j + 1)][static_cast<size_t>(j)] = 0.0;

      double temp1 = 0.0, temp2 = 0.0;
      apply_givens(g[static_cast<size_t>(j)], g[static_cast<size_t>(j + 1)], static_cast<size_t>(j), &temp1, &temp2);
      g[static_cast<size_t>(j)] = temp1;
      g[static_cast<size_t>(j + 1)] = temp2;

      if (progress && residual_interval > 0 && (iter_total % std::max(1, residual_interval)) == 0) {
        progress("residual_l2", std::abs(g[static_cast<size_t>(j + 1)]));
      }

      ++iter_total;
      if (std::abs(g[static_cast<size_t>(j + 1)]) < tol) {
        break;
      }
    }

    int y_dim = std::min(m, iter_total);
    std::vector<double> y(static_cast<size_t>(y_dim), 0.0);
    for (int i = y_dim - 1; i >= 0; --i) {
      double sum = g[static_cast<size_t>(i)];
      for (int k = i + 1; k < y_dim; ++k) {
        sum -= H[static_cast<size_t>(i)][static_cast<size_t>(k)] * y[static_cast<size_t>(k)];
      }
      if (std::abs(H[static_cast<size_t>(i)][static_cast<size_t>(i)]) < 1e-30) {
        return false;
      }
      y[static_cast<size_t>(i)] = sum / H[static_cast<size_t>(i)][static_cast<size_t>(i)];
    }

    for (int j = 0; j < y_dim; ++j) {
      for (size_t k = 0; k < n; ++k) {
        (*x)[k] += y[static_cast<size_t>(j)] * V[static_cast<size_t>(j)][k];
      }
    }

    apply_A(*x, &Ax);
    for (size_t i = 0; i < n; ++i) {
      r[i] = b[i] - Ax[i];
    }
    beta = Norm2(r);
    if (progress && residual_interval > 0) {
      progress("residual_l2", beta);
      progress("residual_linf", NormInf(r));
    }
    if (beta < tol) {
      return true;
    }
    if (iter_total >= max_iter) {
      break;
    }
  }
  return false;
}
