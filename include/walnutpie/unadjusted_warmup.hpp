#pragma once

// W-73 (two-phase warmup, preregistered in WORKLOG.md 2026-08-25): during
// the first f*num_warmup warmup iterations, transitions run UNADJUSTED —
// the backward reversibility certificate is bypassed (`reversible()` and
// `reversible_lr()` return true immediately). Activated by the environment
// variable WALNUTPIE_UNADJUSTED_WARMUP_FRAC (a fraction in (0,1); empty,
// unset, or outside (0,1) = disabled => EXACT current behavior).
//
// Design note (documented decision): rather than threading a bool through
// the macro_step/build_leaf/build_span template signatures (which would
// touch every call site of both the diagonal and low-rank paths), this
// follows the pin_trace.hpp precedent of a config namespace with inline
// state. The flag `active` is toggled once per warmup iteration by
// AdaptiveWalnuts::operator() BEFORE any transition runs, so it applies to
// BOTH the diagonal (transition_w) and low-rank (transition_w_lr) paths.
// The sampling phase never enters operator() and `active` is cleared at
// the last warmup iteration by construction (boundary <= num_warmup), so
// sampling-phase behavior is untouched. Optional variant: when
// WALNUTPIE_PHASE_RESET=1, the MassEstimator accumulators are reset to
// their seeds at the phase boundary (first post-phase iteration), so the
// metric re-learns from post-boundary draws only.

#include <cstddef>
#include <cstdlib>

namespace walnutpie::detail::unadjusted_warmup {

/**
 * @brief Return the configured unadjusted-warmup fraction, reading
 * WALNUTPIE_UNADJUSTED_WARMUP_FRAC from the environment exactly once.
 * Returns 0 when disabled (unset/empty/invalid/outside (0,1)).
 */
inline double configured_frac() {
  static const double frac = [] {
    const char* env = std::getenv("WALNUTPIE_UNADJUSTED_WARMUP_FRAC");
    if (env == nullptr || env[0] == '\0') {
      return 0.0;
    }
    const double v = std::strtod(env, nullptr);
    return (v > 0.0 && v < 1.0) ? v : 0.0;
  }();
  return frac;
}

/**
 * @brief Return whether WALNUTPIE_PHASE_RESET=1 (read exactly once).
 */
inline bool phase_reset() {
  static const bool flag = [] {
    const char* env = std::getenv("WALNUTPIE_PHASE_RESET");
    return env != nullptr && env[0] == '1' && env[1] == '\0';
  }();
  return flag;
}

/**
 * @brief Whether the CURRENT warmup iteration runs unadjusted transitions.
 * Set/cleared by AdaptiveWalnuts::operator() each warmup iteration; false
 * throughout the sampling phase.
 */
inline bool active = false;

}  // namespace walnutpie::detail::unadjusted_warmup
