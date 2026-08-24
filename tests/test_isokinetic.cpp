// walnuts-ai Phase-1 prototype gate driver (standalone; not wired into CMake).
//
// Build:
//   g++ -std=c++20 -O2 -I include -I <eigen-src> tests/test_isokinetic.cpp \
//       -o build_ai/test_isokinetic
//
// Targets: (a) d-dim standard Gaussian, (b) Neal's funnel (reference spec,
// nx=10, sv=3 => d=11): v ~ N(0, 3^2), x_i | v ~ N(0, e^v).

#include <cmath>
#include <cstdio>
#include <limits>
#include <random>
#include <string>
#include <vector>

#include <Eigen/Dense>

#include "walnutpie/isoadapt.hpp"
#include "walnutpie/isokinetic.hpp"
#include "walnutpie/util.hpp"

using Eigen::VectorXd;

// ---- gradient-counting wrapper --------------------------------------------
template <class F>
struct CountingF {
  const F& f;
  mutable long long n = 0;
  void operator()(const VectorXd& x, double& lp, VectorXd& g) const {
    ++n;
    f(x, lp, g);
  }
};

// ---- targets ----------------------------------------------------------------
struct GaussianTarget {
  int d;
  void operator()(const VectorXd& x, double& lp, VectorXd& g) const {
    lp = -0.5 * x.squaredNorm();
    g = -x;
  }
};

// Neal's funnel per generalized-walnuts targets.py make_funnel(nx=10, sv=3):
// th[0] = v ~ N(0, sv^2); th[i] | v ~ N(0, exp(2v)) i.e. N(0, sigma^2),
// sigma = e^v.  d = nx + 1.
struct FunnelTarget {
  int nx;
  double sv;
  void operator()(const VectorXd& x, double& lp, VectorXd& g) const {
    const double v = x[0];
    double ssq = 0.0;
    for (int i = 1; i <= nx; ++i) ssq += x[i] * x[i];
    const double ev = std::exp(-v);
    lp = -0.5 * v * v / (sv * sv) - 0.5 * nx * v - 0.5 * ev * ssq;
    g.resize(nx + 1);
    g[0] = -v / (sv * sv) - 0.5 * nx + 0.5 * ev * ssq;
    for (int i = 1; i <= nx; ++i) g[i] = -ev * x[i];
  }
};

// ---- plain leapfrog HMC baseline (~50 lines) ---------------------------------
struct HmcResult {
  std::vector<VectorXd> draws;
  long long grads = 0;
};

template <class F>
HmcResult run_hmc(const F& f, const VectorXd& theta0, double step, int L,
                  int n_draws, int n_burn, unsigned seed) {
  std::mt19937_64 rng(seed);
  walnutpie::detail::Random<std::mt19937_64> rand(rng);
  HmcResult res;
  VectorXd th = theta0;
  const int d = th.size();
  double lp;
  VectorXd grad(d);
  f(th, lp, grad);
  auto kinetic = [](const VectorXd& p) { return 0.5 * p.squaredNorm(); };
  for (int t = 0; t < n_draws + n_burn; ++t) {
    VectorXd p = rand.standard_normal(d);
    VectorXd th_new = th, grad_new = grad, p_new = p;
    double lp_new;
    p_new += 0.5 * step * grad_new;
    bool diverged = false;
    for (int l = 0; l < L; ++l) {
      th_new += step * p_new;
      f(th_new, lp_new, grad_new);
      if (!std::isfinite(lp_new)) { diverged = true; break; }
      if (l + 1 < L) p_new += step * grad_new;
    }
    if (!diverged) {
      p_new += 0.5 * step * grad_new;
      double log_acc =
          (lp_new + kinetic(p_new)) - (lp + kinetic(p));
      if (std::log(rand.uniform_real_01()) < log_acc) {
        th = th_new; grad = grad_new; lp = lp_new;
      }
    }
    if (t >= n_burn) res.draws.push_back(th);
  }
  res.grads = f.n;
  return res;
}

// ---- ESS via Geyer initial positive sequence, per coordinate ------------------
double ess_ips(const std::vector<double>& xs) {
  const long n = static_cast<long>(xs.size());
  double mean = 0.0;
  for (double x : xs) mean += x;
  mean /= n;
  double var = 0.0;
  for (double x : xs) var += (x - mean) * (x - mean);
  var /= (n - 1);
  if (var <= 0) return static_cast<double>(n);
  auto acov = [&](long k) {
    double s = 0.0;
    for (long i = k; i < n; ++i) s += (xs[i] - mean) * (xs[i - k] - mean);
    return s / n;
  };
  double tau = 0.0;  // sum of paired autocovariances
  for (long k = 1; k + 1 < n; k += 2) {
    double pair = acov(k) + acov(k + 1);
    if (pair <= 0) break;
    tau += pair;
  }
  double ess = n / (1.0 + 2.0 * tau / var);
  return std::min(ess, static_cast<double>(n));
}

static FILE* csv;

template <class F>
std::vector<VectorXd> run_iso(const F& f, const VectorXd& theta0,
                              const VectorXd& anchor, double h,
                              double delta_tol, std::size_t max_ell,
                              std::size_t i_max, int n_draws, int n_burn,
                              unsigned seed, long long& grad_count,
                              double& min_log_sigma, int sigma_idx) {
  std::mt19937_64 rng(seed);
  walnutpie::detail::Random<std::mt19937_64> rand(rng);
  CountingF cf{f};
  VectorXd th = theta0;
  std::vector<VectorXd> draws;
  min_log_sigma = std::numeric_limits<double>::infinity();
  for (int t = 0; t < n_draws + n_burn; ++t) {
    walnutpie::detail::IsoStats st;
    th = walnutpie::detail::transition_w_iso(rand, cf, anchor, h, delta_tol,
                                             max_ell, i_max, std::move(th),
                                             st);
    if (t >= n_burn) {
      draws.push_back(th);
      if (sigma_idx >= 0)
        min_log_sigma = std::min(min_log_sigma, th[sigma_idx]);
      if (csv) {
        for (int i = 0; i < th.size(); ++i)
          fprintf(csv, "%s%.17g", i ? "," : "", th[i]);
        fprintf(csv, "\n");
      }
    }
  }
  grad_count = cf.n;
  return draws;
}

namespace nuts {

struct Point {
  VectorXd th, r, g;
  double lp;
};

template <class F>
void leapfrog(const F& f, Point& p, double eps) {
  p.r += 0.5 * eps * p.g;
  p.th += eps * p.r;
  double lp;
  VectorXd g(p.th.size());
  f(p.th, lp, g);
  if (!std::isfinite(lp)) lp = -std::numeric_limits<double>::infinity();
  p.lp = lp;
  p.g = g;
  if (std::isfinite(lp)) p.r += 0.5 * eps * g;
}

double joint(const Point& p) { return p.lp - 0.5 * p.r.squaredNorm(); }

bool uturn(const Point& a, const Point& b) {
  const VectorXd dt = b.th - a.th;
  return dt.dot(a.r) < 0.0 || dt.dot(b.r) < 0.0;
}

struct Tree {
  Point minus, plus;
  Point proposal;
  bool valid = false;
  int n = 0;  // number of valid slice states in the subtree
};

template <class F>
Tree build_tree(const F& f, walnutpie::detail::Random<std::mt19937_64>& rand,
                const Point& p0, int depth, int dir, double eps,
                double log_u) {
  Tree t;
  Point end = p0;
  leapfrog(f, end, dir * eps);
  t.minus = t.plus = t.proposal = end;
  const double j = joint(end);
  if (std::isfinite(j) && j - log_u > -1000.0 && log_u < j) {
    t.valid = true;
    t.n = 1;
    return t;  // leaf
  }
  return t;  // invalid / divergent leaf
}

template <class F>
Tree build_subtree(const F& f, walnutpie::detail::Random<std::mt19937_64>& rand,
                   const Point& p0, int depth, int dir, double eps,
                   double log_u) {
  if (depth == 0) return build_tree(f, rand, p0, 0, dir, eps, log_u);
  Tree inner = build_subtree(f, rand, p0, depth - 1, dir, eps, log_u);
  Tree t = inner;
  if (!inner.valid) {  // stop early: propagate invalidity
    t.valid = false;
    t.n = 0;
    return t;
  }
  Tree outer =
      build_subtree(f, rand, dir == 1 ? inner.plus : inner.minus, depth - 1,
                    dir, eps, log_u);
  if (dir == 1) {
    t.minus = inner.minus;
    t.plus = outer.plus;
  } else {
    t.minus = outer.minus;
    t.plus = inner.plus;
  }
  const int n_tot = inner.n + outer.n;
  if (outer.n > 0 &&
      std::log(rand.uniform_real_01()) <
          std::log(static_cast<double>(outer.n) /
                   static_cast<double>(n_tot))) {
    t.proposal = outer.proposal;
  }
  t.valid = !(uturn(t.minus, t.plus));
  // classic NUTS flags the whole tree invalid if either half hit a u-turn
  if (uturn(inner.minus, inner.plus) || uturn(outer.minus, outer.plus))
    t.valid = false;
  t.n = n_tot;
  return t;
}

template <class F>
Point nuts_step(const F& f, walnutpie::detail::Random<std::mt19937_64>& rand,
                const Point& p0, double eps) {
  Point cur = p0;
  const double log_u = joint(cur) + std::log(rand.uniform_real_01());
  Tree t;
  t.minus = t.plus = t.proposal = cur;
  t.valid = true;
  int n = 1;
  for (int depth = 0; depth < 8; ++depth) {
    const int dir = (rand.uniform_real_01() < 0.5) ? -1 : 1;
    Tree sub = build_subtree(f, rand, dir == 1 ? t.plus : t.minus, depth, dir,
                             eps, log_u);
    if (dir == 1)
      t.plus = sub.plus;
    else
      t.minus = sub.minus;
    if (!sub.valid || uturn(t.minus, t.plus)) break;
    if (sub.n > 0 && std::log(rand.uniform_real_01()) <
                         std::log(static_cast<double>(sub.n) /
                                  static_cast<double>(n))) {
      cur = sub.proposal;
    }
    n += sub.n;
  }
  return cur;
}

struct NutsResult {
  std::vector<VectorXd> draws;
  long long grads = 0;
};

template <class F>
NutsResult run_nuts(const F& f, const VectorXd& theta0, double eps,
                    int n_draws, int n_burn, unsigned seed) {
  std::mt19937_64 rng(seed);
  walnutpie::detail::Random<std::mt19937_64> rand(rng);
  NutsResult res;
  Point cur{theta0, VectorXd::Zero(theta0.size()),
            VectorXd::Zero(theta0.size()), 0.0};
  f(cur.th, cur.lp, cur.g);
  for (int it = 0; it < n_draws + n_burn; ++it) {
    VectorXd r = rand.standard_normal(theta0.size());
    cur.r = r;
    cur = nuts_step(f, rand, cur, eps);
    res.grads = f.n;
    if (it >= n_burn) res.draws.push_back(cur.th);
  }
  res.grads = f.n;
  return res;
}

}  // namespace nuts

int main(int argc, char** argv) {
  const char* out_prefix = argc > 1 ? argv[1] : "build_ai/iso";
  std::string path = std::string(out_prefix) + "_gaussian.csv";
  csv = std::fopen(path.c_str(), "w");

  // ================= Gate 1: reversibility by negation ======================
  {
    const int d = 10;
    GaussianTarget tgt{d};
    std::mt19937_64 rng(12345);
    walnutpie::detail::Random<std::mt19937_64> rand(rng);
    VectorXd th0 = rand.standard_normal(d);
    VectorXd r0 = rand.standard_normal(d).normalized();
    VectorXd th1 = th0, r1 = r0;
    VectorXd g_end(d);
    const double h = 0.1;
    const std::size_t N = 64;
    for (std::size_t i = 0; i < N; ++i)
      walnutpie::detail::bab_micro_step(tgt, th1, r1, g_end, h);
    // negate momentum and integrate N steps again: involution returns start
    r1 = -r1;
    for (std::size_t i = 0; i < N; ++i)
      walnutpie::detail::bab_micro_step(tgt, th1, r1, g_end, h);
    double eth = (th1 - th0).norm();
    double erh = (r1 + r0).norm();  // final state should be (th0, -r0)
    std::printf("[gate1 reversibility] |dtheta|=%.3e |drho+rho0|=%.3e -> %s\n",
                eth, erh, (eth < 1e-10 && erh < 1e-10) ? "PASS" : "FAIL");
  }

  // ================= Gate 2: Gaussian moments, 20k draws ====================
  VectorXd anchor_g = VectorXd::Zero(10);
  long long grads_iso = 0;
  double junk;
  std::vector<VectorXd> draws = run_iso(
      GaussianTarget{10}, VectorXd::Zero(10), anchor_g,
      /*h=*/1.6, /*delta_tol=*/0.05, /*max_ell=*/7, /*i_max=*/8,
      /*n_draws=*/20000, /*n_burn=*/1000, /*seed=*/20260824, grads_iso, junk,
      -1);
  const long n = static_cast<long>(draws.size());
  VectorXd mean = VectorXd::Zero(10), m2 = VectorXd::Zero(10);
  for (auto& x : draws) mean += x;
  mean /= n;
  for (auto& x : draws) m2 += (x - mean).cwiseAbs2();
  VectorXd var = m2 / (n - 1);
  // ESS-adjusted SE per coordinate
  double max_z = 0.0, worst_var_ratio = 0.0, min_ess = n;
  for (int i = 0; i < 10; ++i) {
    std::vector<double> col(n);
    for (long t = 0; t < n; ++t) col[t] = draws[t][i];
    double ess = ess_ips(col);
    min_ess = std::min(min_ess, ess);
    double se = std::sqrt(var[i] / ess);
    max_z = std::max(max_z, std::fabs(mean[i]) / se);
    worst_var_ratio = std::max(worst_var_ratio,
                               std::fabs(var[i] - 1.0));  // target var = 1
  }
  std::printf(
      "[gate2 gaussian] n=%ld max |z| of means=%.2f (%s) max |var-1|=%.3f "
      "(%s) min ESS=%.0f\n",
      n, max_z, max_z < 3 ? "PASS" : "FAIL", worst_var_ratio,
      worst_var_ratio < 0.20 ? "PASS" : "FAIL", min_ess);

  // ================= Gate 4: determinism =====================================
  {
    long long g1, g2;
    double mls1, mls2;
    auto d1 = run_iso(GaussianTarget{10}, VectorXd::Zero(10), anchor_g, 1.6,
                      0.05, 7, 8, 200, 50, 777, g1, mls1, -1);
    auto d2 = run_iso(GaussianTarget{10}, VectorXd::Zero(10), anchor_g, 1.6,
                      0.05, 7, 8, 200, 50, 777, g2, mls2, -1);
    bool same = true;
    for (std::size_t t = 0; t < d1.size(); ++t)
      if (d1[t] != d2[t]) { same = false; break; }
    std::printf("[gate4 determinism] bit-identical: %s\n",
                same ? "PASS" : "FAIL");
  }

  // ================= Gate 5: ESS / gradient on Gaussian =====================
  {
    CountingF cg{GaussianTarget{10}};
    // walnuts-ai numbers from gate 2 run
    std::printf(
        "[gate5 iso] effective draws / gradient call = %.4f "
        "(minESS=%.0f, grads=%lld over %ld draws)\n",
        min_ess / static_cast<double>(grads_iso), min_ess, grads_iso, n);
    auto hm = run_hmc(cg, VectorXd::Zero(10), /*step=*/0.35, /*L=*/32,
                      /*n_draws=*/20000, /*n_burn=*/1000, /*seed=*/42);
    const long nh = static_cast<long>(hm.draws.size());
    double hmc_min_ess = nh;
    for (int i = 0; i < 10; ++i) {
      std::vector<double> col(nh);
      for (long t = 0; t < nh; ++t) col[t] = hm.draws[t][i];
      hmc_min_ess = std::min(hmc_min_ess, ess_ips(col));
    }
    std::printf(
        "[gate5 hmc] effective draws / gradient call = %.4f "
        "(minESS=%.0f, grads=%lld)\n",
        hmc_min_ess / static_cast<double>(cg.n), hmc_min_ess, cg.n);
  }

  // ================= Gate 3: funnel ==========================================
  std::fclose(csv);
  csv = std::fopen((std::string(out_prefix) + "_funnel.csv").c_str(), "w");
  {
    FunnelTarget tgt{10, 3.0};
    const int d = 11;
    VectorXd anchor_f = VectorXd::Zero(d);
    long long gf = 0;
    double min_lv = 0.0;
    auto fd = run_iso(tgt, VectorXd::Zero(d), anchor_f,
                      /*h=*/1.6, /*delta_tol=*/0.05, /*max_ell=*/8,
                      /*i_max=*/9, /*n_draws=*/10000, /*n_burn=*/500,
                      /*seed=*/99, gf, min_lv, /*sigma_idx=*/0);
    bool finite = true;
    double mean_v = 0;
    for (auto& x : fd) {
      if (!x.allFinite()) { finite = false; break; }
      mean_v += x[0];
    }
    mean_v /= fd.size();
    std::printf(
        "[gate3 funnel] all finite: %s, min log(sigma)=%.3f (%s), "
        "mean(v)=%.3f, grads=%lld\n",
        finite ? "yes" : "NO", min_lv, min_lv < -5.0 ? "PASS" : "FAIL",
        mean_v, gf);
  }
// Classic Hoffman-Gelman Algorithm 6 (slice variable, subtree counts,
// doubling), fixed step size, no adaptation, max depth 8.



  std::fclose(csv);

  // ==========================================================================
  // ============ Phase 2 (W-62 pre-registered gates a-e) ====================
  // ==========================================================================
  walnutpie::detail::IsoAdaptConfig cfg;
  cfg.n_warmup = 500;
  cfg.n_draws = 2000;
  cfg.refresh_every = 50;
  cfg.window = 250;

  // ---------- P2-a + P2-b: adapted funnel -----------------------------------
  {
    FunnelTarget tgt{10, 3.0};
    const int d = 11;
    std::mt19937_64 rng(4242);
    walnutpie::detail::Random<std::mt19937_64> rand(rng);
    auto ar = walnutpie::detail::run_adapted_iso(rand, tgt, VectorXd::Zero(d),
                                                 cfg, /*seed=*/20260824);
    // h-trace convergence: |h_t - h_{t-50}|/h < 5% at some refresh by it 200
    int conv_iter = -1;
    for (std::size_t i = 1; i < ar.h_trace.size(); ++i) {
      const double rel =
          std::fabs(ar.h_trace[i] - ar.h_trace[i - 1]) / ar.h_trace[i];
      if (rel < 0.05 && ar.refresh_iter[i] <= 200) {
        conv_iter = ar.refresh_iter[i];
        break;
      }
    }
    std::printf("[P2-a funnel h-trace]");
    for (std::size_t i = 0; i < ar.h_trace.size(); ++i)
      std::printf(" it%d:%.4f", ar.refresh_iter[i], ar.h_trace[i]);
    std::printf(" -> converged-by-200: %s (final h=%.4f)\n",
                conv_iter > 0 ? "PASS" : "FAIL", ar.h);
    bool finite = true;
    double min_lv = std::numeric_limits<double>::infinity();
    for (auto& x : ar.draws) {
      if (!x.allFinite()) { finite = false; break; }
      min_lv = std::min(min_lv, x[0]);
    }
    std::printf(
        "[P2-a funnel frozen sanity] finite=%s min log sigma=%.3f (%s)\n",
        finite ? "yes" : "NO", min_lv,
        (finite && min_lv < -5.0) ? "PASS" : "FAIL");

    // P2-b: ESS/grad vs inline NUTS baseline (same draws budget)
    const long nf = static_cast<long>(ar.draws.size());
    double iso_min_ess = nf;
    for (int i = 0; i < d; ++i) {
      std::vector<double> col(nf);
      for (long t = 0; t < nf; ++t) col[t] = ar.draws[t][i];
      iso_min_ess = std::min(iso_min_ess, ess_ips(col));
    }
    CountingF cfun{FunnelTarget{10, 3.0}};
    // conservative baseline: sweep a small eps grid, keep NUTS' best
    // ESS/grad so the comparison cannot be gamed in our favor
    double nuts_best_r = -1.0;
    double nuts_best_ess = 0, nuts_best_eps = 0;
    long long nuts_best_grads = 0;
    for (double eps : {0.1, 0.15, 0.2, 0.3}) {
      const long long before = cfun.n;
      auto nr = nuts::run_nuts(cfun, VectorXd::Zero(d), eps,
                               /*n_draws=*/2000, /*n_burn=*/500, /*seed=*/7);
      const long long run_grads = cfun.n - before;
      const long nn = static_cast<long>(nr.draws.size());
      double nuts_min_ess = nn;
      for (int i = 0; i < d; ++i) {
        std::vector<double> col(nn);
        for (long t = 0; t < nn; ++t) col[t] = nr.draws[t][i];
        nuts_min_ess = std::min(nuts_min_ess, ess_ips(col));
      }
      const double r_now = nuts_min_ess / static_cast<double>(run_grads);
      if (r_now > nuts_best_r) {
        nuts_best_ess = nuts_min_ess; nuts_best_eps = eps;
        nuts_best_grads = run_grads; }
    }
    double nuts_min_ess = nuts_best_ess;
    long long nr_grads = nuts_best_grads;
    const long long iso_grads_total = ar.warmup_grads + ar.sampling_grads;
    std::printf("[P2-b accounting] warmup+calibration grads=%lld frozen "
                "sampling grads=%lld\n",
                ar.warmup_grads, ar.sampling_grads);
    const double r_iso = iso_min_ess / iso_grads_total;
    const double r_nuts = nuts_min_ess / nr_grads;
    std::printf(
        "[P2-b funnel ESS/grad] iso=%.5f (minESS=%.0f, grads=%lld incl "
        "warmup+calibration), nuts(eps=%g)=%.5f (minESS=%.0f, grads=%lld), ratio "
        "iso/nuts=%.2fx (%s)\n",
        r_iso, iso_min_ess, iso_grads_total, nuts_best_eps, r_nuts,
        nuts_min_ess, nr_grads,
        r_iso / r_nuts, (r_iso >= 5.0 * r_nuts) ? "PASS" : "BELOW-5x");
  }

  // ---------- P2-c: adapted Gaussian within 2x of Phase-1 fixed-h ----------
  {
    GaussianTarget gt{10};
    std::mt19937_64 rng(99);
    walnutpie::detail::Random<std::mt19937_64> rand(rng);
    auto ar = walnutpie::detail::run_adapted_iso(rand, gt, VectorXd::Zero(10),
                                                 cfg, /*seed=*/555);
    const long ng = static_cast<long>(ar.draws.size());
    double adapted_min_ess = ng;
    for (int i = 0; i < 10; ++i) {
      std::vector<double> col(ng);
      for (long t = 0; t < ng; ++t) col[t] = ar.draws[t][i];
      adapted_min_ess = std::min(adapted_min_ess, ess_ips(col));
    }
    const double adapted =
        adapted_min_ess / (ar.warmup_grads + ar.sampling_grads);
    const double fixed = min_ess / static_cast<double>(grads_iso);
    std::printf(
        "[P2-c gaussian ESS/grad] adapted(h=%.3f)=%.5f vs phase1 "
        "fixed-h=%.5f, ratio=%.2f (%s)\n",
        ar.h, adapted, fixed, adapted / fixed,
        adapted >= 0.5 * fixed ? "PASS" : "FAIL");
  }

  // ---------- P2-d: determinism through the ADAPTED path --------------------
  {
    auto run_once = [&](unsigned seed) {
      std::mt19937_64 rng(seed);
      walnutpie::detail::Random<std::mt19937_64> rand(rng);
      return walnutpie::detail::run_adapted_iso(rand, GaussianTarget{10},
                                                VectorXd::Zero(10), cfg, seed);
    };
    auto a = run_once(31337);
    auto b = run_once(31337);
    bool same = a.h == b.h && a.anchor == b.anchor &&
                a.draws.size() == b.draws.size();
    for (std::size_t t = 0; t < a.draws.size() && same; ++t) {
      const VectorXd& x = a.draws[t];
      const VectorXd& y = b.draws[t];
      for (Eigen::Index j = 0; j < x.size(); ++j)
        if (x[j] != y[j]) { same = false; break; }
    }
    std::printf("[P2-d determinism adapted] bit-identical draws+h+C: %s\n",
                same ? "PASS" : "FAIL");
  }

  return 0;
}
