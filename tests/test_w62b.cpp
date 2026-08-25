// W-62b pre-registered diagnostic driver (standalone; not wired into CMake).
//
// Build:
//   clang++ -std=c++20 -O2 -I include -I <eigen-src> tests/test_w62b.cpp \
//       -o build_ai/test_w62b
//
// Implements stan/WORKLOG.md entry "W-62b PRE-REGISTRATION":
//   Q1 decompose iso gradient calls on Neal's funnel (d=11 reference spec):
//      warmup transitions / h-calibration probes / frozen sampling, and
//      within frozen: level-search attempt grads vs accepted-leaf grads,
//      mean+max accepted level ell, grads per draw.
//   Q2 FAIR baseline: walnutpie's own walnuts-h (detail::transition_w,
//      identity diagonal mass -- see NOTE below) with find_reasonable_step
//      + minimal dual averaging (target accept 0.8), 500 warmup iters,
//      frozen step/mass, 2000 frozen draws (same budget as iso arm).
//   Q3 grid: max_ell in {8,6,4} x delta_tol in {0.05,0.20}, fresh h
//      recalibration per cell, fixed seeds, funnel only.
//
// Metric (both arms): SAMPLING-PHASE-ONLY min-over-coords rank-normalized
// ESS_bulk (single-chain rank transform z = Phi^{-1}((r-3/8)/(n+1/4)), then
// Geyer initial-positive-sequence ESS) divided by sampling-phase gradient
// calls.
//
// NOTE (baseline metric choice): the inverse mass is held at the identity.
// A variance-based diagonal mass is ill-defined on Neal's funnel because
// x_i has no finite marginal second moment along the neck; the identity
// matches the inline classic-NUTS reference used in P2-b, so both baselines
// face iso under the same metric.

#include <algorithm>
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
#include "walnutpie/walnuts.hpp"
#include "walnutpie/warmup_heuristics.hpp"

using Eigen::VectorXd;
using Random = walnutpie::detail::Random<std::mt19937_64>;
using IsoStats = walnutpie::detail::IsoStats;

// ---- target ------------------------------------------------------------------
// Neal's funnel per generalized-walnuts targets.py make_funnel(nx=10, sv=3):
// th[0] = v ~ N(0, sv^2); th[i] | v ~ N(0, sigma^2), sigma = e^v. d = nx + 1.
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

// ---- phase-attributed gradient counter ----------------------------------------
enum Phase { PH_WARMUP = 0, PH_CALIB = 1, PH_SAMPLE = 2 };

inline Phase& cur_phase_slot() {
  static thread_local Phase p = PH_WARMUP;
  return p;
}

struct IsoCountingF {
  const FunnelTarget& f;
  mutable long long by_phase[3] = {0, 0, 0};
  mutable long long samp_attempt = 0;   // forward level-search grads (frozen)
  mutable long long samp_accepted = 0;  // p-micro/reverse/entry grads (frozen)
  void operator()(const VectorXd& x, double& lp, VectorXd& g) const {
    ++by_phase[cur_phase_slot()];
    if (cur_phase_slot() == PH_SAMPLE) {
      if (walnutpie::detail::g_iso_probe_mode == 1) ++samp_attempt;
      else ++samp_accepted;
    }
    f(x, lp, g);
  }
};

struct PlainCountingF {
  const FunnelTarget& f;
  mutable long long n = 0;
  void operator()(const VectorXd& x, double& lp, VectorXd& g) const {
    ++n;
    f(x, lp, g);
  }
};

// dual-averaging hook for the baseline arm: fires on each min-attempt stat
template <class Upd>
struct DaHandler {
  double target;
  Upd update;
  double step_size() const { return std::numeric_limits<double>::quiet_NaN(); }
  void operator()(double accept_prob) const { update(accept_prob); }
};

// ---- rank-normalized ESS (single chain, bulk, Geyer IPS) ----------------------
double inv_phi(double p) {  // Acklam's inverse normal CDF
  static const double a[6] = {-3.969683028665376e+01, 2.209460984245205e+02,
                              -2.759285104469687e+02, 1.383577518672690e+02,
                              -3.066479806614716e+01, 2.506628277459239e+00};
  static const double b[5] = {-5.447609879822406e+01, 1.615858368580409e+02,
                              -1.556989798598866e+02, 6.680131188771972e+01,
                              -1.328068155288572e+01};
  static const double c[6] = {-7.784894002430293e-03, -3.223964580411365e-01,
                              -2.400758277161838e+00, -2.549732539343734e+00,
                              4.374664141464968e+00, 2.938163982698783e+00};
  static const double dd[4] = {7.784695709041462e-03, 3.224671290700398e-01,
                               2.445134137142996e+00, 3.754408661907416e+00};
  const double plow = 0.02425, phigh = 1 - plow;
  double q, r;
  if (p < plow) {
    q = std::sqrt(-2 * std::log(p));
    return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) /
           ((((dd[0]*q+dd[1])*q+dd[2])*q+dd[3])*q+1);
  }
  if (p > phigh) {
    q = std::sqrt(-2 * std::log(1 - p));
    return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) /
            ((((dd[0]*q+dd[1])*q+dd[2])*q+dd[3])*q+1);
  }
  q = p - 0.5;
  r = q * q;
  return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q /
         (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1);
}

double ess_ips(const std::vector<double>& xs) {
  const long n = static_cast<long>(xs.size());
  double mean = 0.0;
  for (double x : xs) mean += x;
  mean /= n;
  double var = 0.0;
  for (double x : xs) var += (x - mean) * (x - mean);
  var /= (n - 1);
  if (!(var > 0)) return 0.0;  // degenerate (pinned) chain scores zero
  auto acov = [&](long k) {
    double s = 0.0;
    for (long i = k; i < n; ++i) s += (xs[i] - mean) * (xs[i - k] - mean);
    return s / n;
  };
  double tau = 0.0;
  for (long k = 1; k + 1 < n; k += 2) {
    double pair = acov(k) + acov(k + 1);
    if (pair <= 0) break;
    tau += pair;
  }
  double ess = n / (1.0 + 2.0 * tau / var);
  return std::min(ess, static_cast<double>(n));
}

double rn_ess_bulk(const std::vector<VectorXd>& draws) {
  const long n = static_cast<long>(draws.size());
  const int d = static_cast<int>(draws[0].size());
  std::vector<std::pair<double, int>> order(n);
  double best = std::numeric_limits<double>::infinity();
  for (int j = 0; j < d; ++j) {
    for (long t = 0; t < n; ++t)
      order[t] = {draws[t][j], static_cast<int>(t)};
    std::sort(order.begin(), order.end());
    std::vector<double> z(n);
    long t = 0;
    while (t < n) {
      long u = t;
      while (u + 1 < n && order[u + 1].first == order[t].first) ++u;
      const double ravg = 0.5 * (t + u) + 1.0;  // average rank, 1-based
      const double zz = inv_phi((ravg - 0.375) / (n + 0.25));
      for (long k = t; k <= u; ++k) z[order[k].second] = zz;
      t = u + 1;
    }
    best = std::min(best, ess_ips(z));
  }
  return best;
}

// ---- adapted-iso driver with phase attribution ---------------------------------
// Replicates detail::run_adapted_iso (isoadapt.hpp) iteration-for-iteration;
// only the counting differs.
struct IsoRun {
  VectorXd anchor;
  double h = 0.0;
  std::vector<VectorXd> draws;
  long long warmup_grads = 0, calib_grads = 0, sample_grads = 0;
  long long sample_attempt_grads = 0, sample_accepted_grads = 0;
  walnutpie::detail::IsoLeafProbe probe;         // cumulative, all phases
  walnutpie::detail::IsoLeafProbe probe_frozen;  // snapshot before freezing
  int stop_codes[5] = {0, 0, 0, 0, 0};
  long long leaves_frozen = 0, failed_leaves_frozen = 0;
  double min_log_sigma = std::numeric_limits<double>::infinity();
  long long total_grads() const {
    return warmup_grads + calib_grads + sample_grads;
  }
};

IsoRun run_adapted_iso_w62b(Random& rand, const FunnelTarget& tgt,
                            VectorXd theta0,
                            const walnutpie::detail::IsoAdaptConfig& cfg,
                            unsigned long long seed, int sigma_idx) {
  IsoCountingF cf{tgt};
  IsoRun out;
  VectorXd anchor = VectorXd::Zero(theta0.size());
  double h = cfg.h0;
  std::vector<VectorXd> win;
  win.reserve(cfg.window);
  walnutpie::detail::g_iso_probe = &out.probe;

  for (int t = 0; t < cfg.n_warmup; ++t) {
    cur_phase_slot() = PH_WARMUP;
    IsoStats st;
    theta0 = walnutpie::detail::transition_w_iso(rand, cf, anchor, h,
                                                 cfg.delta_tol, cfg.max_ell,
                                                 cfg.i_max, std::move(theta0),
                                                 st);
    win.push_back(theta0);
    if (win.size() > cfg.window) win.erase(win.begin());
    if ((t + 1) % static_cast<int>(cfg.refresh_every) == 0 &&
        win.size() >= cfg.refresh_every) {
      cur_phase_slot() = PH_CALIB;
      anchor = walnutpie::detail::coord_median(win);
      h = walnutpie::detail::tune_h_iso(cf, win, cfg.gamma, cfg.delta_tol,
                                        seed + 900000);
    }
  }
  out.warmup_grads = cf.by_phase[PH_WARMUP];
  out.calib_grads = cf.by_phase[PH_CALIB];
  out.anchor = anchor;
  out.h = h;

  out.probe_frozen = out.probe;
  cur_phase_slot() = PH_SAMPLE;
  for (int t = 0; t < cfg.n_draws; ++t) {
    IsoStats st;
    theta0 = walnutpie::detail::transition_w_iso(rand, cf, out.anchor, out.h,
                                                 cfg.delta_tol, cfg.max_ell,
                                                 cfg.i_max, std::move(theta0),
                                                 st);
    ++out.stop_codes[std::min(st.stop_code, 4)];
    if (sigma_idx >= 0)
      out.min_log_sigma = std::min(out.min_log_sigma, theta0[sigma_idx]);
    out.draws.push_back(theta0);
  }
  walnutpie::detail::g_iso_probe = nullptr;
  cur_phase_slot() = PH_WARMUP;
  out.sample_grads = cf.by_phase[PH_SAMPLE];
  out.sample_attempt_grads = cf.samp_attempt;
  out.sample_accepted_grads = cf.samp_accepted;
  out.leaves_frozen = out.probe.leaves - out.probe_frozen.leaves;
  out.failed_leaves_frozen =
      out.probe.failed_leaves - out.probe_frozen.failed_leaves;
  return out;
}

int main() {
  FunnelTarget tgt{10, 3.0};
  const int d = 11;
  const unsigned long long seed = 20260824;
  walnutpie::detail::IsoAdaptConfig base_cfg;
  base_cfg.n_warmup = 500;
  base_cfg.n_draws = 2000;
  base_cfg.refresh_every = 50;
  base_cfg.window = 250;
  base_cfg.max_ell = 8;
  base_cfg.i_max = 9;

  // ================= Q1 + iso arm of Q2: reference cell ======================
  IsoRun iso;
  {
    std::mt19937_64 rng(seed);
    Random rand(rng);
    iso = run_adapted_iso_w62b(rand, tgt, VectorXd::Zero(d), base_cfg, seed, 0);
  }
  const long n_iso = static_cast<long>(iso.draws.size());
  const double iso_ess = rn_ess_bulk(iso.draws);
  const long long iso_samp_grads = iso.sample_grads;
  const double iso_r = iso_ess / static_cast<double>(iso_samp_grads);

  std::printf("=== Q1: iso gradient decomposition (funnel d=11, max_ell=%zu "
              "delta_tol=%.2f, seed=%llu) ===\n",
              base_cfg.max_ell, base_cfg.delta_tol, seed);
  std::printf(
      "  warmup-transition grads : %lld (%.1f%%)\n"
      "  h-calibration grads     : %lld (%.1f%%)\n"
      "  frozen-sampling grads   : %lld (%.1f%%)\n",
      iso.warmup_grads, 100.0 * iso.warmup_grads / iso.total_grads(),
      iso.calib_grads, 100.0 * iso.calib_grads / iso.total_grads(),
      iso.sample_grads, 100.0 * iso.sample_grads / iso.total_grads());
  std::printf(
      "  frozen breakdown: level-search attempt=%lld (%.1f%%), "
      "accepted-leaf(p-micro+reverse+entry)=%lld (%.1f%%)\n",
      iso.sample_attempt_grads,
      100.0 * iso.sample_attempt_grads / iso.sample_grads,
      iso.sample_accepted_grads,
      100.0 * iso.sample_accepted_grads / iso.sample_grads);
  const long long frozen_leaves =
      iso.probe.leaves - iso.probe_frozen.leaves;
  const double leaves_per_draw = static_cast<double>(frozen_leaves) / n_iso;
  const double ell_mean =
      static_cast<double>(iso.probe.ell_sum - iso.probe_frozen.ell_sum) /
      frozen_leaves;
  std::printf(
      "  frozen leaves/draw=%.2f (failed %.2f/draw), accepted ell* "
      "mean=%.2f max=%zu\n",
      leaves_per_draw,
      static_cast<double>(iso.failed_leaves_frozen) / n_iso, ell_mean,
      iso.probe.ell_max_seen);
  std::printf("  stop codes [1..4]: %d %d %d %d (transitions==draws==%ld)\n",
              iso.stop_codes[1], iso.stop_codes[2], iso.stop_codes[3],
              iso.stop_codes[4], n_iso);
  std::printf(
      "  frozen grads/draw=%.1f; final h=%.4f; min log sigma=%.3f\n",
      static_cast<double>(iso_samp_grads) / n_iso, iso.h, iso.min_log_sigma);
  std::printf(
      "  iso SAMPLING-ONLY: rnESS_bulk(min coord)=%.0f grads=%lld -> "
      "ESS/grad=%.5f\n\n",
      iso_ess, iso_samp_grads, iso_r);

  // ================= Q2: walnutpie's own walnuts-h baseline ==================
  // Design (documented): identity diagonal inverse mass (see NOTE above),
  // detail::find_reasonable_step init + minimal Hoffman-Gelman dual averaging
  // (target accept 0.8, gamma=0.05, t0=10, kappa=0.75) over 500 warmup
  // transitions of transition_w (max_depth 8, halvings 5, min_micro 1,
  // max_error 0.5), then frozen step/mass for 2000 draws (same budget as the
  // iso arm). Endpoint gradient caching mirrors WalnutsSampler (W-23).
  long long wh_warmup_grads = 0, wh_sample_grads = 0;
  std::vector<VectorXd> wh_draws;
  double eps_final = 0.0;
  double wh_min_sigma = std::numeric_limits<double>::infinity();
  {
    PlainCountingF cf{tgt};
    std::mt19937_64 rng(seed + 1);
    Random rand(rng);
    const VectorXd inv_mass = VectorXd::Ones(d);
    const VectorXd chol_mass = VectorXd::Ones(d);
    VectorXd theta = VectorXd::Zero(d);
    walnutpie::detail::NoOpStepSizeAdapter noop;

    double eps0 = walnutpie::detail::find_reasonable_step(rand, tgt, theta,
                                                          inv_mass, 1.0);
    const double mu = std::log(10.0 * eps0);
    const double gamma_da = 0.05, t0 = 10.0, kappa = 0.75, delta_t = 0.8;
    double h_bar = 0.0, log_eps_bar = 0.0, eps = eps0;
    VectorXd grad_cached;
    double logp_cached = -std::numeric_limits<double>::infinity();
    std::size_t depth;
    for (int t = 1; t <= 500; ++t) {
      const double eta = 1.0 / (t + t0);
      auto update = [&](double a) {
        h_bar = (1 - eta) * h_bar + eta * (delta_t - a);
        double x_k =
            mu - std::sqrt(static_cast<double>(t)) / gamma_da * h_bar;
        // Clamp: on the funnel neck the gradient vanishes, alpha -> 1 for ANY
        // step size, and unclamped dual averaging diverges (observed eps ->
        // 9e8, frozen chain pinned). [1e-3, 10] brackets sane walnuts-h
        // macro steps for this target.
        x_k = std::clamp(x_k, std::log(1e-3), std::log(10.0));
        eps = std::exp(x_k);
        const double w = std::pow(static_cast<double>(t), -kappa);
        log_eps_bar = w * x_k + (1 - w) * log_eps_bar;
      };
      DaHandler<decltype(update)> handler{delta_t, update};
      theta = walnutpie::detail::transition_w(rand, cf, inv_mass, chol_mass,
                                              eps, /*max_depth=*/8,
                                              /*max_step_halvings=*/5,
                                              /*min_micro_steps=*/1,
                                              /*max_error=*/0.5,
                                              std::move(theta), depth,
                                              grad_cached, logp_cached,
                                              handler, grad_cached,
                                              logp_cached);
    }
    eps_final = std::clamp(std::exp(log_eps_bar), 1e-3, 10.0);
    wh_warmup_grads = cf.n;
    for (int t = 0; t < 2000; ++t) {
      theta = walnutpie::detail::transition_w(rand, cf, inv_mass, chol_mass,
                                              eps_final, /*max_depth=*/8,
                                              /*max_step_halvings=*/5,
                                              /*min_micro_steps=*/1,
                                              /*max_error=*/0.5,
                                              std::move(theta), depth,
                                              grad_cached, logp_cached, noop,
                                              grad_cached, logp_cached);
      wh_min_sigma = std::min(wh_min_sigma, theta[0]);
      wh_draws.push_back(theta);
    }
    wh_sample_grads = cf.n - wh_warmup_grads;
  }
  const long n_wh = static_cast<long>(wh_draws.size());
  const double wh_ess = rn_ess_bulk(wh_draws);
  const double wh_r = wh_ess / static_cast<double>(wh_sample_grads);
  std::printf("=== Q2: walnutpie-h baseline (identity diag mass, DA warmup "
              "500 -> eps=%.4f, seed=%llu) ===\n", eps_final, seed + 1);
  std::printf(
      "  warmup grads=%lld (not charged), frozen grads=%lld "
      "(%.1f/draw), min log sigma=%.3f\n",
      wh_warmup_grads, wh_sample_grads,
      static_cast<double>(wh_sample_grads) / n_wh, wh_min_sigma);
  std::printf(
      "  walnutpie-h SAMPLING-ONLY: rnESS_bulk(min coord)=%.0f -> "
      "ESS/grad=%.5f\n", wh_ess, wh_r);
  std::printf(
      "  RATIO iso / walnutpie-h = %.3fx   (iso %.5f vs wh %.5f)\n\n",
      iso_r / wh_r, iso_r, wh_r);

  // ================= Q3: pre-specified sensitivity grid =======================
  std::printf("=== Q3: grid max_ell x delta_tol (fresh recalibration per "
              "cell, seed=%llu, 2000 frozen draws) ===\n", seed);
  std::printf("  %-8s %-9s %-9s %-10s %-12s %-12s %-10s %-10s\n", "max_ell",
              "delta_tol", "final_h", "minLogSig", "sampGrads", "rnESSbulk",
              "ESS/grad", "vs wh");
  for (std::size_t mell : {std::size_t{8}, std::size_t{6}, std::size_t{4}}) {
    for (double dtol : {0.05, 0.20}) {
      walnutpie::detail::IsoAdaptConfig cfg = base_cfg;
      cfg.max_ell = mell;
      cfg.delta_tol = dtol;
      std::mt19937_64 rng(seed);
      Random rand(rng);
      IsoRun cell = run_adapted_iso_w62b(rand, tgt, VectorXd::Zero(d), cfg,
                                         seed, 0);
      const double ess = rn_ess_bulk(cell.draws);
      const double r = ess / static_cast<double>(cell.sample_grads);
      std::printf("  %-8zu %-9.2f %-9.4f %-10.3f %-12lld %-12.0f %-10.5f "
                  "%.2fx\n",
                  mell, dtol, cell.h, cell.min_log_sigma, cell.sample_grads,
                  ess, r, r / wh_r);
    }
  }

  std::printf("\nDECISION RULE: iso>=1x walnutpie-h? %s | grid cells vs "
              "wh-baseline in table above (>=3x?)\n",
              iso_r >= wh_r ? "YES" : "NO");
  return 0;
}
