#pragma once

// walnuts-ai Phase 2: warmup adaptation for the isokinetic-anchored WALNUTS
// variant. Port of github.com/nawafbourabee/generalized-walnuts tuning.py:
//
//   * h calibration (tuning.py::make_nohalve_probe / expand_gamma_bracket /
//     tune_macro_step): the macro step h solves
//         P(micro(theta, rho, h, delta_tol) = 0) = gamma
//     on the states of a calibration chain, where "micro = 0" means a single
//     level-0 macro step keeps the effective-Hamiltonian range within
//     delta_tol. We chose this Gamma-bisection route over any adapter-
//     statistic shortcut because it is the reference scheme, it reuses the
//     exact Phase-1 micro integrator (bab_micro_step at s = h), and it is
//     trivially deterministic: every probe evaluation draws its momenta from
//     its own mt19937_64 seeded deterministically from the caller's seed, so
//     bisection is a pure function of (calibration states, seed).
//
//   * anchor C estimation (tuning.py::estimate_section_center): the
//     coordinate-wise median of calibration-chain positions. Online choice
//     documented below (windowed refresh rather than one-shot): during
//     warmup we keep the most recent `window` positions and recompute the
//     coordinate-wise median (std::nth_element on per-coordinate buffers)
//     every `refresh_every` iterations together with an h re-bisection.
//     The anchor is passed by const ref into transition_w_iso, so within a
//     transition forward and backward growth see bit-identical C by
//     construction; C only ever changes between transitions.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <random>
#include <vector>

#include <Eigen/Dense>

#include "walnutpie/isokinetic.hpp"
#include "walnutpie/util.hpp"

namespace walnutpie::detail {

/**
 * @brief Fraction of calibration states where one level-0 isokinetic macro
 *        step of duration `h` stays admissible ("no-halving" probability,
 *        tuning.py::frac_no_halve, FLOW_ISO branch).
 *
 * Each state gets a fresh uniform unit momentum drawn from a private
 * mt19937_64 seeded with `seed`, making the estimate deterministic given
 * (states, seed). Non-finite states count as failures.
 */
template <LogpGrad F>
double frac_nohalve_iso(const F& logp_grad,
                        const std::vector<Eigen::VectorXd>& states, double h,
                        double delta_tol, unsigned long long seed,
                        std::size_t probe_max = 120) {
  if (states.empty()) return 0.0;
  // Deterministic cost cap: probe only the most recent `probe_max` states
  // (the window is already the trailing segment of the chain, so this keeps
  // the freshest calibration information while bounding each probe to
  // O(probe_max) gradient pairs).
  const std::size_t begin =
      states.size() > probe_max ? states.size() - probe_max : 0;
  std::mt19937_64 rng(seed);
  Random<std::mt19937_64> rand(rng);
  long long cnt = 0;
  Eigen::VectorXd g_end;
  for (std::size_t si = begin; si < states.size(); ++si) {
    const auto& th0 = states[si];
    if (!th0.allFinite()) continue;
    Eigen::VectorXd z =
        rand.standard_normal(static_cast<Eigen::Index>(th0.size()));
    Eigen::VectorXd rho = z.normalized();
    Eigen::VectorXd th = th0;
    double lp0;
    {
      Eigen::VectorXd g(th.size());
      logp_grad(th, lp0, g);
      if (!std::isfinite(lp0)) continue;
    }
    double H0 = -lp0, Hmax = H0, Hmin = H0;
    double logJ = bab_micro_step(logp_grad, th, rho, g_end, h);
    double H = -l0_at(logp_grad, th) - logJ;
    if (H > Hmax) Hmax = H;
    if (H < Hmin) Hmin = H;
    if (Hmax - Hmin <= delta_tol) ++cnt;
  }
  return static_cast<double>(cnt) /
         static_cast<double>(states.size() - begin);
}

/**
 * @brief Geometric-bisection solve of P(micro = 0) = gamma
 *        (tuning.py::expand_gamma_bracket + tune_macro_step body).
 *
 * Bracket [h_lo, h_hi] is expanded geometrically until f(lo) >= gamma >=
 * f(hi); each iteration replaces one endpoint with the geometric mean
 * sqrt(lo*hi). Deterministic: probe seeds are seed_base + fixed offsets (one
 * per probe evaluation), so the bisection is a pure function of
 * (states, seed_base).
 */
template <LogpGrad F>
double tune_h_iso(const F& logp_grad,
                  const std::vector<Eigen::VectorXd>& states, double gamma,
                  double delta_tol, unsigned long long seed_base,
                  int bisection_iters = 14) {
  constexpr int kMaxExpand = 16;
  double lo = 1e-4, hi = 2.0;
  double f_lo = frac_nohalve_iso(logp_grad, states, lo, delta_tol, seed_base);
  double f_hi =
      frac_nohalve_iso(logp_grad, states, hi, delta_tol, seed_base + 1);
  int n_expand = 0;
  while (f_lo < gamma && n_expand < kMaxExpand) {
    hi = lo;
    f_hi = f_lo;
    lo *= 0.5;
    f_lo = frac_nohalve_iso(logp_grad, states, lo, delta_tol,
                            seed_base + 10 + n_expand);
    ++n_expand;
  }
  while (f_hi > gamma && n_expand < kMaxExpand) {
    lo = hi;
    f_lo = f_hi;
    hi *= 2.0;
    f_hi = frac_nohalve_iso(logp_grad, states, hi, delta_tol,
                            seed_base + 100 + n_expand);
    ++n_expand;
  }
  for (int it = 0; it < bisection_iters; ++it) {
    const double mid = std::sqrt(lo * hi);
    const double frac = frac_nohalve_iso(logp_grad, states, mid, delta_tol,
                                         seed_base + 20000 + it);
    if (frac > gamma)
      lo = mid;
    else
      hi = mid;
  }
  return std::sqrt(lo * hi);
}

/**
 * @brief Coordinate-wise median of a set of positions
 *        (tuning.py::estimate_section_center, windowed form).
 */
inline Eigen::VectorXd coord_median(
    const std::vector<Eigen::VectorXd>& xs) {
  const Eigen::Index d = xs.front().size();
  Eigen::VectorXd med(d);
  std::vector<double> buf;
  buf.reserve(xs.size());
  for (Eigen::Index j = 0; j < d; ++j) {
    buf.clear();
    for (const auto& x : xs)
      if (x.allFinite()) buf.push_back(x[j]);
    if (buf.empty()) {
      med[j] = 0.0;
      continue;
    }
    const std::size_t mid = buf.size() / 2;
    std::nth_element(buf.begin(), buf.begin() + mid, buf.end());
    med[j] = buf[mid];
    if (buf.size() % 2 == 0) {
      const double upper = buf[mid];
      const double lower = *std::max_element(buf.begin(), buf.begin() + mid);
      med[j] = 0.5 * (lower + upper);
    }
  }
  return med;
}

/** @brief Configuration and result record for warmup adaptation. */
struct IsoAdaptConfig {
  int n_warmup = 500;        /**< warmup iterations with adaptation ON */
  int n_draws = 1000;        /**< frozen sampling-phase draws */
  std::size_t refresh_every = 50;  /**< refresh (C, h) cadence */
  std::size_t window = 250;  /**< max positions kept for calibration */
  double gamma = 0.80;       /**< target no-halving probability */
  double h0 = 0.5;           /**< initial macro step */
  double delta_tol = 0.05;
  std::size_t max_ell = 8;
  std::size_t i_max = 9;
};

struct IsoAdaptResult {
  Eigen::VectorXd anchor;       /**< frozen anchor C */
  double h = 0.0;               /**< frozen macro step */
  std::vector<int> refresh_iter;/**< warmup iteration of each refresh */
  std::vector<double> h_trace;  /**< h after each refresh */
  std::vector<Eigen::VectorXd> draws;  /**< frozen-phase draws */
  long long sampling_grads = 0; /**< gradient calls in the frozen phase */
  long long warmup_grads = 0;   /**< gradient calls during warmup */
};

/** @brief Gradient-call counter wrapper (for ESS/grad accounting). */
template <class F>
struct GradCounter {
  const F& f;
  long long* n;
  void operator()(const Eigen::VectorXd& x, double& lp,
                  Eigen::VectorXd& g) const {
    ++(*n);
    f(x, lp, g);
  }
};

/**
 * @brief Warmup-with-adaptation driver followed by a frozen sampling phase.
 *
 * Warmup: every `refresh_every` iterations (once at least `refresh_every`
 * positions are buffered), recompute the coordinate-median anchor over the
 * trailing window and re-bisect h to P(micro = 0) = gamma on the same
 * window. Sampling: freeze both and run plain transition_w_iso.
 *
 * Deterministic given (seed): a single Random stream drives the chain and
 * every probe uses its own seed-derived stream (see frac_nohalve_iso).
 */
template <LogpGrad F, std::uniform_random_bit_generator RNG>
IsoAdaptResult run_adapted_iso(Random<RNG>& rand, const F& logp_grad,
                               Eigen::VectorXd theta0,
                               const IsoAdaptConfig& cfg,
                               unsigned long long seed) {
  IsoAdaptResult res;
  long long n_grads = 0;
  GradCounter<F> cf{logp_grad, &n_grads};
  Eigen::VectorXd anchor = Eigen::VectorXd::Zero(theta0.size());
  double h = cfg.h0;
  std::vector<Eigen::VectorXd> win;
  win.reserve(cfg.window);

  for (int t = 0; t < cfg.n_warmup; ++t) {
    IsoStats st;
    theta0 = transition_w_iso(rand, cf, anchor, h, cfg.delta_tol, cfg.max_ell,
                              cfg.i_max, std::move(theta0), st);
    win.push_back(theta0);
    if (win.size() > cfg.window) win.erase(win.begin());
    if ((t + 1) % static_cast<int>(cfg.refresh_every) == 0 &&
        win.size() >= cfg.refresh_every) {
      // order matters for reproducibility: anchor first (pure function),
      // then the h bisection on the pre-refresh chain positions
      anchor = coord_median(win);
      h = tune_h_iso(cf, win, cfg.gamma, cfg.delta_tol, seed + 900000);
      res.refresh_iter.push_back(t + 1);
      res.h_trace.push_back(h);
    }
  }
  res.anchor = anchor;
  res.h = h;


  res.warmup_grads = n_grads;
  n_grads = 0;  // gate accounting: frozen-phase gradients only
  for (int t = 0; t < cfg.n_draws; ++t) {
    IsoStats st;
    theta0 = transition_w_iso(rand, cf, res.anchor, res.h, cfg.delta_tol,
                              cfg.max_ell, cfg.i_max, std::move(theta0), st);
    res.draws.push_back(theta0);
  }
  res.sampling_grads = n_grads;
  return res;
}

}  // namespace walnutpie::detail
