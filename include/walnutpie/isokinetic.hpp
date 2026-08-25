#pragma once

// walnuts-ai: opt-in isokinetic-anchored WALNUTS variant ("generalized
// WALNUTS", Bou-Rabee / Carpenter / Kleppe). Phase-1 prototype.
//
// Implements the isokinetic flow on R^d x S^{d-1} with BAB micro steps,
// randomized micro levels inside one macro leaf (Listing build-leaf-rand),
// and the first-return anchored radial-max stopping rule (Listing
// cross-radial-max / WALNUTS-STEP). Faithful port of
// github.com/nawafbourabee/generalized-walnuts (engine.py), commit noted
// in its module docstring.
//
// Deviation from the classical walnuts.hpp tree structure: the first-return
// driver grows a linear chain of leaves per direction per doubling round and
// merges them with biased progressive selection toward the fresh extension;
// there is no recursive build_span binary tree and no U-turn test. SpanW /
// combine<Barker/Metropolis> are therefore NOT reused here because the
// generalized sampler's weighting (per-leaf Hastings-corrected weights,
// biased progressive merge, non-retained crossing leaves) does not factor
// through the span algebra. The pin_trace hooks are used where semantically
// apt; the backward ladder counters stay at zero on this path.

#include <cmath>
#include <cstddef>
#include <limits>
#include <optional>
#include <utility>

#include <Eigen/Dense>

#include "walnutpie/concepts.hpp"
#include "walnutpie/pin_trace.hpp"
#include "walnutpie/util.hpp"

namespace walnutpie::detail {

/**
 * @brief Closed-form isokinetic half-B kick (Listing BAB, Proposition b-step).
 *
 * Rotates the unit momentum `rho` toward the normalized gradient `e = g/||g||`
 * over half the micro step (`delta = s_macro * ||g|| / (2 (d-1))`) and returns
 * the rotated momentum together with the log Jacobian contribution
 * `dlogJ = -(d-1) (delta + log pre)`, where
 * `pre = ((1 + gamma) + (1 - gamma) exp(-2 delta)) / 2` and
 * `gamma = rho . e`. Degenerate gradients (`||g|| ~ 0`) act as identity.
 *
 * Exact formula verified against engine.py::b_step_update:
 * ```
 * coef_pre = ((1 + gamma) - (1 - gamma) exp(-2 delta))/2 - gamma exp(-delta)
 * rho_new  = normalize(rho * exp(-delta)/pre + (coef_pre/pre) * e)
 * ```
 */
inline std::pair<Eigen::VectorXd, double> b_step_update(
    const Eigen::VectorXd& g, const Eigen::VectorXd& rho, double s_macro) {
  const double xi = g.norm();
  const double d = static_cast<double>(rho.size());
  if (xi < 1e-300) {
    return {rho, 0.0};
  }
  const Eigen::VectorXd e = g / xi;
  const double gamma = rho.dot(e);
  const double delta = s_macro * xi / (2.0 * (d - 1.0));
  const double e_neg2 = std::exp(-2.0 * delta);
  const double e_neg = std::exp(-delta);
  const double pre = 0.5 * ((1.0 + gamma) + (1.0 - gamma) * e_neg2);
  if (!(pre > 0.0)) {
    return {rho, 0.0};
  }
  const double coef_pre =
      0.5 * ((1.0 + gamma) - (1.0 - gamma) * e_neg2) - gamma * e_neg;
  Eigen::VectorXd rho_new =
      rho * (e_neg / pre) + (coef_pre / pre) * e;
  const double nm = rho_new.norm();
  if (nm > 0) {
    rho_new /= nm;
  }
  return {std::move(rho_new), -(d - 1.0) * (delta + std::log(pre))};
}

/**
 * @brief One full isokinetic BAB micro step of duration `s` (signed).
 *
 * Half-B, position update `theta += s * rho` (unit speed; inv_mass is the
 * identity by construction of the isokinetic flow), half-B. Negative `s`
 * (backward integration) is realized by negating the momentum, integrating
 * forward for `-s`, and negating again, exactly as engine.py does.
 *
 * @param[in] logp_grad Target log density/gradient.
 * @param[in,out] theta Position; set to the new position.
 * @param[in,out] rho Unit momentum; set to the new unit momentum.
 * @param[out] g_end Gradient at the new position.
 * @return Accumulated log Jacobian of the full micro step.
 */
// helper: fresh logp evaluation at a position (gradient discarded)
template <LogpGrad F>
double l0_at(const F& logp_grad, const Eigen::VectorXd& th) {
  double lp;
  Eigen::VectorXd g(th.size());
  logp_grad(th, lp, g);
  return lp;
}

template <LogpGrad F>
double bab_micro_step(const F& logp_grad, Eigen::VectorXd& theta,
                      Eigen::VectorXd& rho, Eigen::VectorXd& g_end,
                      double s) {
  double log_jac = 0.0;
  const bool backward = s < 0.0;
  if (backward) {
    rho = -rho;
    s = -s;
  }
  // first half-B: gradient at the current position
  {
    double logp;
    Eigen::VectorXd g;
    logp_grad(theta, logp, g);
    auto [q_new, dj] = b_step_update(g, rho, s);
    log_jac += dj;
    rho = std::move(q_new);
  }
  // drift of duration s (unit speed; inv_mass = identity)
  theta += s * rho;
  // second half-B: gradient at the new position
  {
    double logp;
    Eigen::VectorXd g;
    logp_grad(theta, logp, g);
    auto [q_new, dj] = b_step_update(g, rho, s);
    log_jac += dj;
    rho = std::move(q_new);
    g_end = std::move(g);
  }
  if (backward) {
    rho = -rho;  // engine.py: q = -qtmp after forward-integrating -q
  }
  return log_jac;
}

/** @brief Oriented anchored radial-max crossing predicate (Listing
 *         cross-radial-max). Fires when the radial derivative
 *         phi(z) = (theta - C)' rho changes from + to - along forward
 *         physical time; `dir_` == -1 exchanges the roles. */
inline bool cross_radial_max(const Eigen::VectorXd& t_prev,
                             const Eigen::VectorXd& r_prev,
                             const Eigen::VectorXd& t_new,
                             const Eigen::VectorXd& r_new,
                             const Eigen::VectorXd& C, int dir_) {
  double vp = (t_prev - C).dot(r_prev);
  double vn = (t_new - C).dot(r_new);
  if (dir_ == 1) {
    return vp > 0.0 && vn < 0.0;
  }
  return vp < 0.0 && vn > 0.0;
}

/** @brief Result of one randomized-level macro leaf (Listing build-leaf-rand).
 */
struct IsoLeaf {
  Eigen::VectorXd theta;   /**< new endpoint position */
  Eigen::VectorXd rho;     /**< new endpoint unit momentum */
  Eigen::VectorXd grad;    /**< target gradient at the new endpoint */
  double log_weight;       /**< accumulated weight (-inf => inadmissible) */
};

// ---- W-62b opt-in gradient-cost probe (default OFF; no arithmetic change) --
// When g_iso_probe is non-null, build_leaf_iso attributes its work between
// (a) the forward level search (attempted levels until ell_star is found)
// and (b) the accepted-leaf integration (p-micro draw + capped reverse
// search), and records the accepted ell_star. Purely observational.
struct IsoLeafProbe {
  long long attempt_grads = 0;   /**< grads in forward level-search loop */
  long long accepted_grads = 0;  /**< grads in p-micro + reverse search */
  long long leaves = 0;          /**< leaves built */
  long long failed_leaves = 0;   /**< leaves with no admissible level */
  long long ell_sum = 0;         /**< sum of accepted ell_star */
  std::size_t ell_max_seen = 0;  /**< max accepted ell_star */
};
inline IsoLeafProbe* g_iso_probe = nullptr;
// 0 = outside leaf, 1 = forward level search, 2 = accepted-leaf integration
inline int g_iso_probe_mode = 0;

/**
 * @brief One macro leaf with randomized micro level (Listing build-leaf-rand,
 * isokinetic BAB branch).
 *
 * Forward search: fixed macro time `h`; level `ell` integrates `n = 2^ell`
 * BAB micro steps of size `s = dir * h / n`. The smallest admissible level
 * `ell*` (effective-Hamiltonian range `-logp - logJ` staying within
 * `delta_tol`) is selected; `ell_p` is drawn from p-micro ({ell*, ell*+1} with
 * probabilities 2/3, 1/3) and, when different from `ell*`, the integration is
 * repeated at level `ell_p`. A capped reverse search from the new endpoint
 * determines `ell_plus` and supplies the Hastings correction
 * `log p(ell_p | ell_plus) - log p(ell_p | ell*)`. If no level up to
 * `max_ell` is admissible the weight is `-inf`.
 *
 * @tparam F The type of the log density/gradient function.
 * @tparam RNG The base random number generator type.
 *
 * @param[in,out] rand Source of randomness (p-micro draw).
 * @param[in] logp_grad The target log density/gradient function.
 * @param[in] theta Start position.
 * @param[in] rho Start unit momentum.
 * @param[in] logw_start Weight accumulated upstream.
 * @param[in] dir +1 forward, -1 backward (temporal direction).
 * @param[in] h Macro time.
 * @param[in] delta_tol Max effective-Hamiltonian range per leaf.
 * @param[in] max_ell Maximum level searched.
 */
template <LogpGrad F, std::uniform_random_bit_generator RNG>
IsoLeaf build_leaf_iso(Random<RNG>& rand, const F& logp_grad,
                       const Eigen::VectorXd& theta,
                       const Eigen::VectorXd& rho, double logw_start, int dir,
                       double h, double delta_tol, std::size_t max_ell) {
  if (g_iso_probe) ++g_iso_probe->leaves;
  const double l0 = [&] {
    double lp;
    Eigen::VectorXd g;
    logp_grad(theta, lp, g);
    return lp;
  }();
  const Eigen::Index d = theta.size();
  g_iso_probe_mode = g_iso_probe ? 1 : 0;

  // ---- forward level search (Listing micro, fused with integration) ----
  std::size_t ell_star = max_ell + 1;
  bool fwd = false;
  Eigen::VectorXd th_s = theta, q_s = rho, g_s = Eigen::VectorXd::Zero(d);
  double lJ_s = 0.0, sw_s = 0.0;
  for (std::size_t ell = 0; ell <= max_ell; ++ell) {
    const std::size_t n = std::size_t(1) << ell;
    const double s = static_cast<double>(dir) * h / static_cast<double>(n);
    Eigen::VectorXd th = theta, q = rho, g_end = Eigen::VectorXd::Zero(d);
    double logJ = 0.0;
    double H = -l0;
    double Hmax = H, Hmin = H;
    bool ok = true;
    for (std::size_t i = 0; i < n; ++i) {
      logJ += bab_micro_step(logp_grad, th, q, g_end, s);
      H = -l0_at(logp_grad, th) - logJ;
      if (H > Hmax) Hmax = H;
      if (H < Hmin) Hmin = H;
      if (Hmax - Hmin > delta_tol) {
        ok = false;
        break;
      }
    }
    pin_trace::observe_attempt(std::fabs(Hmax - Hmin), n);
    if (ok) {
      ell_star = ell;
      fwd = true;
      th_s = th;
      q_s = q;
      lJ_s = logJ;
      sw_s = Hmax - Hmin;
      g_s = g_end;
      break;
    }
  }
  if (g_iso_probe) {
    g_iso_probe_mode = 2;
    if (!fwd) ++g_iso_probe->failed_leaves;
    else {
      g_iso_probe->ell_sum += static_cast<long long>(ell_star);  // full value
    }
    const std::size_t ell_now = fwd ? ell_star : std::size_t{0};
    if (ell_now > g_iso_probe->ell_max_seen)
      g_iso_probe->ell_max_seen = ell_now;
  }

  // ---- p-micro randomization ----
  const std::size_t ell_p =
      ell_star + (rand.uniform_real_01() < 2.0 / 3.0 ? 0u : 1u);
  Eigen::VectorXd th1, q_int, g1;
  double lJ_leaf;
  double sw_leaf;
  if (fwd && ell_p == ell_star) {
    th1 = th_s;
    q_int = q_s;
    lJ_leaf = lJ_s;
    sw_leaf = sw_s;
    g1 = g_s;
  } else {
    const std::size_t n = std::size_t(1) << ell_p;
    const double s = static_cast<double>(dir) * h / static_cast<double>(n);
    th1 = theta;
    q_int = rho;
    lJ_leaf = 0.0;
    double Hmax = -l0, Hmin = -l0;
    for (std::size_t i = 0; i < n; ++i) {
      lJ_leaf += bab_micro_step(logp_grad, th1, q_int, g1, s);
      const double H = -l0_at(logp_grad, th1) - lJ_leaf;
      if (H > Hmax) Hmax = H;
      if (H < Hmin) Hmin = H;
    }
    sw_leaf = Hmax - Hmin;
  }

  // ---- capped reverse search (micro from the new leaf) ----
  const double D = l0_at(logp_grad, th1) + lJ_leaf - l0;
  int ell_plus = -1;
  const std::size_t rev_cap = std::min(ell_p, max_ell);
  for (std::size_t ell = 0; ell <= rev_cap; ++ell) {
    bool ok;
    if (ell == ell_p) {
      ok = sw_leaf <= delta_tol;  // time symmetry: same range in reverse
    } else {
      const std::size_t n = std::size_t(1) << ell;
      const double s =
          static_cast<double>(-dir) * h / static_cast<double>(n);
      Eigen::VectorXd th = th1, q = q_int, g_end = Eigen::VectorXd::Zero(d);
      double logJ = 0.0;
      // engine.py initializes H = -logp(th1) (NOT including lJ_leaf; the
      // accumulated leaf Jacobian must not shift the reverse range)
      double H = -(D + l0 - lJ_leaf);
      double Hmax = H, Hmin = H;
      ok = true;
      for (std::size_t i = 0; i < n; ++i) {
        logJ += bab_micro_step(logp_grad, th, q, g_end, s);
        H = -(l0_at(logp_grad, th)) - logJ;
        if (H > Hmax) Hmax = H;
        if (H < Hmin) Hmin = H;
        if (Hmax - Hmin > delta_tol) {
          ok = false;
          break;
        }
      }
    }
    if (ok) {
      ell_plus = static_cast<int>(ell);
      break;
    }
  }

  // ---- Hastings weight: p(ell_p | ell_plus) / p(ell_p | ell*) ----
  auto p_micro_pmf = [](std::size_t j, std::size_t i) {
    if (j == i) return 2.0 / 3.0;
    if (j == i + 1) return 1.0 / 3.0;
    return 0.0;
  };
  const std::size_t ell_plus_sz =
      ell_plus < 0 ? max_ell + 1 : static_cast<std::size_t>(ell_plus);
  const double pn = p_micro_pmf(ell_p, ell_plus_sz);
  const double pd = p_micro_pmf(ell_p, ell_star);
  IsoLeaf out;
  out.log_weight =
      (pn > 0.0 && pd > 0.0)
          ? logw_start + D + std::log(pn) - std::log(pd)
          : -std::numeric_limits<double>::infinity();
  out.theta = std::move(th1);
  out.rho = std::move(q_int);
  out.grad = std::move(g1);
  g_iso_probe_mode = g_iso_probe ? 0 : g_iso_probe_mode;
  return out;
}

/** @brief Per-transition diagnostics for the isokinetic sampler. */
struct IsoStats {
  std::size_t depth = 0;          /**< doubling rounds attempted */
  std::size_t leaves_built = 0;   /**< leaves built (incl. discarded) */
  std::size_t selectable = 0;     /**< retained (positive-weight) leaves */
  int stop_code = 0;              /**< 1 reject/fail, 3 both returned,
                                       4 size cap */
};

/**
 * @brief One walnuts-ai transition with anchored radial-max first-return
 * stopping (Listings WALNUTS-STEP / extend-orbit, STOP_RADIAL_MAX branch).
 *
 * Draws a fresh uniform unit momentum, then alternates random directions,
 * extending each side by chains of randomized-level leaves until the anchored
 * section phi(z) = (theta - C)' rho is crossed outward-in (first-return
 * convention: the crossing leaf terminates growth but is not selectable),
 * both sides have returned, a leaf is inadmissible (zero weight), or the
 * orbit reaches 2^i_max leaves. Selection is biased progressive toward the
 * fresh extension.
 *
 * @tparam F The type of the log density/gradient function.
 * @tparam RNG The base random number generator type.
 *
 * @param[in,out] rand Source of randomness.
 * @param[in] logp_grad The target log density/gradient function.
 * @param[in] anchor Radial anchor C (e.g. coordinate-wise median estimate).
 * @param[in] h Macro time.
 * @param[in] delta_tol Effective-Hamiltonian range tolerance per leaf.
 * @param[in] max_ell Maximum micro level searched per leaf.
 * @param[in] i_max Maximum number of doubling rounds.
 * @param[in,out] theta Current state; replaced by the next draw.
 * @param[out] stats Diagnostics for this transition.
 * @return The next position of the Markov chain.
 */
template <LogpGrad F, std::uniform_random_bit_generator RNG>
Eigen::VectorXd transition_w_iso(Random<RNG>& rand, const F& logp_grad,
                                 const Eigen::VectorXd& anchor, double h,
                                 double delta_tol, std::size_t max_ell,
                                 std::size_t i_max, Eigen::VectorXd&& theta,
                                 IsoStats& stats) {
  pin_trace::begin_transition();
  const Eigen::Index d = theta.size();
  // refresh: auxiliary variable uniform on S^{d-1}
  Eigen::VectorXd z = rand.standard_normal(d);
  Eigen::VectorXd rho = z.normalized();
  pin_trace::observe_step(h);
  pin_trace::observe_z(rho.norm());

  double lp0;
  {
    Eigen::VectorXd g0(d);
    logp_grad(theta, lp0, g0);
  }
  const double lw0 = lp0;

  bool crossed_left = false, crossed_right = false, failed_any = false;
  Eigen::VectorXd theta_tilde = theta;
  double gW = lw0;
  double gEf = lw0, gEb = lw0;
  Eigen::VectorXd gr_t = theta, gr_r = rho, gl_t = theta, gl_r = rho;
  std::size_t npos = 1;

  for (std::size_t depth = 0; depth < i_max; ++depth) {
    const int dir = (rand.uniform_real_01() < 0.5) ? 1 : -1;
    if (dir == 1 && crossed_right) {
      if (crossed_left && crossed_right) break;
      continue;
    }
    if (dir == -1 && crossed_left) {
      if (crossed_left && crossed_right) break;
      continue;
    }
    Eigen::VectorXd th = (dir == 1) ? gr_t : gl_t;
    Eigen::VectorXd rh = (dir == 1) ? gr_r : gl_r;
    double lw = (dir == 1) ? gEf : gEb;
    double eW = -std::numeric_limits<double>::infinity();
    Eigen::VectorXd cand_th = th;
    Eigen::VectorXd end_th = th, end_rh = rh;
    bool crossed = false, failed = false;

    const std::size_t chain = std::size_t(1) << depth;
    for (std::size_t k = 0; k < chain; ++k) {
      IsoLeaf leaf = build_leaf_iso(rand, logp_grad, th, rh, lw, dir, h,
                                    delta_tol, max_ell);
      if (!std::isfinite(leaf.log_weight) ||
          leaf.log_weight == -std::numeric_limits<double>::infinity()) {
        ++stats.leaves_built;
        failed = true;
        failed_any = true;
        break;
      }
      if (cross_radial_max(th, rh, leaf.theta, leaf.rho, anchor, dir)) {
        // first-return convention: crossing leaf stops growth, not retained
        ++stats.leaves_built;
        crossed = true;
        break;
      }
      ++stats.leaves_built;
      const double newW =
          walnutpie::detail::log_sum_exp(eW, leaf.log_weight);
      if (eW == -std::numeric_limits<double>::infinity() ||
          std::log(rand.uniform_real_01()) <=
              leaf.log_weight - newW) {
        cand_th = leaf.theta;
      }
      eW = newW;
      th = leaf.theta;
      rh = leaf.rho;
      lw = leaf.log_weight;
      end_th = th;
      end_rh = rh;
      ++npos;
    }
    if (eW > -std::numeric_limits<double>::infinity()) {
      if (std::log(rand.uniform_real_01()) <= eW - gW) {
        theta_tilde = cand_th;
      }
      gW = walnutpie::detail::log_sum_exp(gW, eW);
    }
    if (dir == 1) {
      gr_t = end_th;
      gr_r = end_rh;
      gEf = lw;
      crossed_right = crossed_right || crossed || failed;
    } else {
      gl_t = end_th;
      gl_r = end_rh;
      gEb = lw;
      crossed_left = crossed_left || crossed || failed;
    }
    stats.depth = depth + 1;
    if (crossed_left && crossed_right) {
      stats.stop_code = failed_any ? 1 : 3;
      break;
    }
  }
  if (stats.stop_code == 0) {
    stats.stop_code = failed_any ? 1 : 4;
  }
  stats.selectable = npos;
  return theta_tilde;
}

}  // namespace walnutpie::detail
