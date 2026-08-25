#pragma once

// W-78: pair-Barker emission (W-76 theory doc, route (a) with k=2).
//
// During SAMPLING-phase transitions only (the caller decides — see
// WalnutsSampler::operator(); warmup adapters never opt in), after the
// selected state is drawn, a second leaf index j is sampled from the
// accepted span's Barker leaf law p_B(j|S) = e^{-H_j} / sum_i e^{-H_i}
// (Lemma 1 of scratch/w61/w76_antithetic_theory.md) and emitted as an
// additional draw. By Proposition 1 (ibid.) each emitted draw is
// marginally pi-distributed; no HT weights are needed.
//
// Mechanism: while a transition runs with emission requested, build_leaf
// appends every successfully created leaf's (log joint density, log
// density, position) to a thread-local collector. Discarded subtrees
// (internal U-turn / ladder rejection) are removed by truncation at the
// transition-driver level in walnuts.hpp, so at the end of expansion the
// collector holds exactly the leaves of the final accepted span S with
// their stored LSE weights w_j = e^{L_j}. Sampling j is then a single
// O(N) LSE + cumulative walk — computable from quantities walnutpie
// already computes (W-76 §2.3).
//
// Env gate: WALNUTPIE_PAIR_EMISSION={off,pair_barker}; default off. With
// the gate off, no extra RNG draws are consumed and all arithmetic is
// untouched (off-mode bit-identity). Warmup is unaffected because the
// warmup adapters never request emission; adaptation statistics are
// therefore identical between modes for a given seed.

#include <Eigen/Dense>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <mutex>
#include <random>
#include <string>
#include <vector>

#include "walnutpie/util.hpp"

namespace walnutpie::detail {
namespace pair_emission {

/** Emission mode selected once from the environment. */
enum class Mode { Off, PairBarker };

inline Mode mode() {
  static const Mode m = []() {
    if (const char* v = std::getenv("WALNUTPIE_PAIR_EMISSION")) {
      std::string s(v);
      if (s == "pair_barker") {
        return Mode::PairBarker;
      }
    }
    return Mode::Off;
  }();
  return m;
}

inline bool enabled() { return mode() != Mode::Off; }

/**
 * Per-transition leaf accumulator. Active only around transitions whose
 * driver requested pair emission; holds one entry per accepted leaf.
 */
struct Collector {
  bool active = false;
  std::vector<double> logw;      // joint log density L_j (leaf weight)
  std::vector<double> logp_pos;  // marginal log density of the position
  std::vector<Eigen::VectorXd> theta;
  std::size_t mark = 0;

  void start() {
    active = true;
    logw.clear();
    logp_pos.clear();
    theta.clear();
    mark = 0;
  }

  /** Truncate back to `m` leaves: discards a failed expansion subtree. */
  void truncate(std::size_t m) {
    if (!active || logw.size() <= m) {
      return;
    }
    logw.resize(m);
    logp_pos.resize(m);
    theta.resize(m);
  }
};

inline Collector& collector() {
  thread_local Collector c;
  return c;
}

/** Run-lifetime degeneracy diagnostics (min weight mass / dominance). */
struct Diag {
  std::atomic<unsigned long long> transitions{0};
  std::atomic<unsigned long long> leaves_total{0};
  std::atomic<double> max_mass_max{0.0};   // max_t max_j p_B(j|S_t)
  std::atomic<double> min_mass_min{1.0};   // min_t min_j p_B(j|S_t)
};
inline Diag& diag() {
  static Diag d;
  return d;
}

/**
 * Sample one leaf index from p_B(.|S) via LSE + cumulative inverse-CDF,
 * record degeneracy diagnostics, and copy the leaf position out.
 *
 * @param[in,out] rng The transition's random source (one uniform draw).
 * @param[in] c The collector holding the accepted span's leaves.
 * @return The position of the sampled leaf.
 */
template <std::uniform_random_bit_generator RNG>
Eigen::VectorXd sample_leaf(Random<RNG>& rng, const Collector& c,
                            double* logp_out = nullptr) {
  const std::size_t n = c.logw.size();
  Eigen::VectorXd lw(n);
  for (std::size_t i = 0; i < n; ++i) {
    lw[i] = c.logw[i];
  }
  const double lse = log_sum_exp(lw);
  double u = rng.uniform_real_01();
  // cumulative walk over exp(L_j - LSE); guard the tail against rounding
  std::size_t j = n - 1;
  double acc = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    acc += std::exp(c.logw[i] - lse);
    if (u < acc) {
      j = i;
      break;
    }
  }
  // diagnostics: max/min leaf mass of this span
  double max_p = 0.0;
  double min_p = 1.0;
  for (std::size_t i = 0; i < n; ++i) {
    double p = std::exp(c.logw[i] - lse);
    max_p = std::max(max_p, p);
    min_p = std::min(min_p, p);
  }
  auto& d = diag();
  d.transitions.fetch_add(1, std::memory_order_relaxed);
  d.leaves_total.fetch_add(n, std::memory_order_relaxed);
  {
    double cur = d.max_mass_max.load(std::memory_order_relaxed);
    while (max_p > cur &&
           !d.max_mass_max.compare_exchange_weak(cur, max_p,
                                                 std::memory_order_relaxed)) {
    }
    cur = d.min_mass_min.load(std::memory_order_relaxed);
    while (min_p < cur &&
           !d.min_mass_min.compare_exchange_weak(cur, min_p,
                                                 std::memory_order_relaxed)) {
    }
  }
  if (logp_out != nullptr) {
    *logp_out = c.logp_pos[j];
  }
  return c.theta[j];
}

/** End-of-run one-line diagnostic printout (pair mode only). */
inline void print_diag() {
  auto& d = diag();
  const unsigned long long t = d.transitions.load(std::memory_order_relaxed);
  if (t == 0) {
    return;
  }
  const unsigned long long leaves =
      d.leaves_total.load(std::memory_order_relaxed);
  std::cout << "W-78 pair-emission diag: transitions=" << t
            << " mean_leaves=" << static_cast<double>(leaves) / t
            << " max_over_runs_of_max_leaf_mass="
            << d.max_mass_max.load(std::memory_order_relaxed)
            << " min_leaf_mass="
            << d.min_mass_min.load(std::memory_order_relaxed) << std::endl;
}

}  // namespace pair_emission
}  // namespace walnutpie::detail
