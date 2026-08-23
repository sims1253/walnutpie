#pragma once

#include <cmath>
#include <cstddef>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

#include "walnutpie/adapt.hpp"
#include "walnutpie/adaptive_walnuts.hpp"
#include "walnutpie/concepts.hpp"
#include "walnutpie/config.hpp"
#include "walnutpie/sampler.hpp"
#include "walnutpie/walnuts.hpp"

namespace walnutpie {

/**
 * Return the chain records from running Walnuts with the specified
 * seed, sampling event handlers, and configuration.
 *
 * @tparam Handler The type of the event handlers.
 * @param[in] seed The seed for the pseudo-random number generator.
 * @param[in] chain_handlers The collection of chain-specific handlers, which
 * are called back.
 * @param[in] global_handler The handler for global cross-chain events.
 * @param[in] interrupt_callback The callback for stopping.
 * @param[in] log_p_grad The log density and gradient function, called back.
 * @param[in] config The configuration for Walnutpie.
 * @throws std::invalid_argument If the number of handlers doesn't match
 * the initialization configuration's number of chains.
 */
template <std::uniform_random_bit_generator RNG, ChainHandler H,
          GlobalHandler GH, InterruptCallback IC, LogpGrad F>
inline void walnuts(std::size_t seed, std::vector<H>& chain_handlers,
                    GH& global_handler, const IC& interrupt_callback,
                    const F& log_p_grad, const WalnutsConfig& config) {
  using AdaptiveSampler = AdaptiveWalnuts<F, RNG, H>;
  using Sampler = WalnutsSampler<F, RNG, H>;

  if (chain_handlers.size() != config.init().num_chains()) {
    throw std::invalid_argument(
        "chain_handlers.size() must be equal to config.init().num_chains()");
  }

  std::vector<RNG> rngs(0);
  rngs.reserve(config.init().num_chains());
  for (std::size_t m = 0; m < config.init().num_chains(); ++m) {
    std::seed_seq ss{seed, m + 1u};
    rngs.emplace_back(ss);
  }
  std::vector<AdaptiveSampler> adapters;
  adapters.reserve(config.init().num_chains());
  for (std::size_t m = 0; m < config.init().num_chains(); ++m) {
    adapters.emplace_back(rngs[m], chain_handlers[m], log_p_grad,
                          config.init().init_chain_config(m), config.warmup(),
                          config.sampling());
  }
  detail::adapt(config.init(), config.warmup(), adapters, interrupt_callback);

  std::vector<Sampler> samplers;
  samplers.reserve(adapters.size());
  for (std::size_t n = 0; n < adapters.size(); ++n) {
    samplers.emplace_back(std::move(adapters[n].sampler()));
  }

  detail::sample(config.sampling(), samplers, global_handler,
                 interrupt_callback);
}

/**
 * @brief Run adaptive Walnuts with mode-aware re-initialization.
 *
 * After warmup adaptation, checks the cross-chain log-mass dispersion
 * (AdaptResult::log_mass_dispersion). If it exceeds `max_log_mass_dispersion`,
 * the chains have locked into different posterior scales; the run restarts
 * warmup with fresh initializations drawn from `reinit_positions` (cycling),
 * up to `max_reinits` times. This implements the mode-aware reinit policy
 * motivated by the scale-lock failure class (see the init-robustness
 * discussion): cheap (one O(M*D) check), and it never fires on healthy runs.
 */
template <std::uniform_random_bit_generator RNG, ChainHandler H,
          GlobalHandler GH, InterruptCallback IC, LogpGrad F>
inline void walnuts_with_reinit(
    std::size_t seed, std::vector<H>& chain_handlers, GH& global_handler,
    const IC& interrupt_callback, const F& log_p_grad,
    const WalnutsConfig& config,
    const std::vector<std::vector<Eigen::VectorXd>>& reinit_positions,
    double max_log_mass_dispersion, std::size_t max_reinits = 1) {
  using AdaptiveSampler = AdaptiveWalnuts<F, RNG, H>;
  using Sampler = WalnutsSampler<F, RNG, H>;
  const std::size_t M = config.init().num_chains();
  WalnutsConfig cfg = config;  // local copy; reinit rounds rewrite its init part

  std::vector<RNG> rngs(0);
  rngs.reserve(M);
  for (std::size_t m = 0; m < M; ++m) {
    std::seed_seq ss{seed, m + 1u};
    rngs.emplace_back(ss);
  }

  std::size_t reinits = 0;
  std::vector<Sampler> samplers;
  while (true) {
    std::vector<AdaptiveSampler> adapters;
    adapters.reserve(M);
    for (std::size_t m = 0; m < M; ++m) {
      adapters.emplace_back(rngs[m], chain_handlers[m], log_p_grad,
                            cfg.init().init_chain_config(m), cfg.warmup(),
                            cfg.sampling());
    }
    detail::AdaptResult ar = detail::adapt_with_stats(
        cfg.init(), cfg.warmup(), adapters, interrupt_callback);
    samplers.clear();
    samplers.reserve(adapters.size());
    for (std::size_t n = 0; n < adapters.size(); ++n) {
      samplers.emplace_back(std::move(adapters[n].sampler()));
    }
    if (ar.log_mass_dispersion <= max_log_mass_dispersion ||
        reinits >= max_reinits || reinit_positions.size() == 0) {
      break;
    }
    // Chains locked into different scales. Targeted policy (W-15):
    //  - attribute the dispersion to outlier chains (per-chain mean abs
    //    deviation from the cross-chain mean log-mass; outlier if more than
    //    2x the median deviation and at least 0.25 nats);
    //  - outlier chains are re-drawn from the pool (best log-density draws,
    //    distinct per chain) and seeded with the CONSENSUS mass/step so they
    //    re-enter the same scale the healthy chains agreed on;
    //  - healthy chains keep their end-of-warmup position, mass and macro
    //    time, so the second warmup only refines them.
    // If the attribution is ambiguous (no clear outliers, e.g. a 2v2 split),
    // fall back to the blanket policy: re-draw every chain.
    ++reinits;
    std::vector<double> dev(M, 0.0);
    for (std::size_t m = 0; m < M; ++m) {
      dev[m] = (ar.chain_log_mass[m].array() -
                ar.chain_log_mass[m].array().mean())
                   .abs()
                   .mean();
    }
    // median of dev
    std::vector<double> dev_sorted = dev;
    std::sort(dev_sorted.begin(), dev_sorted.end());
    double dev_med =
        dev_sorted[M % 2 == 1 ? M / 2 : M / 2 - 1 + (M > 1 ? 1 : 0)];
    std::vector<bool> outlier(M, false);
    std::size_t n_out = 0;
    for (std::size_t m = 0; m < M; ++m) {
      outlier[m] = dev[m] > 2.0 * dev_med && dev[m] > 0.25;
      if (outlier[m]) ++n_out;
    }
    const bool targeted = n_out > 0 && n_out < M;

    std::vector<Eigen::VectorXd> new_positions(M);
    std::vector<Eigen::VectorXd> new_masses(M);
    std::vector<double> new_steps(M);
    // W-41: ar.step_bar is the geometric mean of the per-chain snapshot
    // step sizes; one chain whose log step underflowed to -inf (or NaN)
    // makes it degenerate and reseeds the round with an unusable step.
    // Fall back to the geometric mean of the just-frozen per-chain macro
    // times, else the round's initial step, else the hard floor, and warn
    // on stderr.
    double step_bar = ar.step_bar;
    if (!(std::isfinite(step_bar) && step_bar > 0.0)) {
      double log_sum = 0.0;
      std::size_t n_ok = 0;
      for (std::size_t m = 0; m < M; ++m) {
        const double mt = samplers[m].macro_time();
        if (std::isfinite(mt) && mt > 0.0) {
          log_sum += std::log(mt);
          ++n_ok;
        }
      }
      const char* source = nullptr;
      double fallback = 0.0;
      if (n_ok > 0) {
        fallback = std::exp(log_sum / static_cast<double>(n_ok));
        source = "geometric mean of frozen chain steps";
      }
      if (!(std::isfinite(fallback) && fallback > 0.0)) {
        fallback = cfg.init().init_chain_config(0).step_size();
        source = "initial step size";
      }
      if (!(std::isfinite(fallback) && fallback > 0.0)) {
        fallback = 1000.0 * std::numeric_limits<double>::min();
        source = "hard floor (1000 * DBL_MIN)";
      }
      std::cerr << "WALNUTS WARNING: reinit step_bar degenerate ("
                << ar.step_bar << "); falling back to " << fallback << " ("
                << source << ")" << std::endl;
      step_bar = fallback;
    }
    for (std::size_t m = 0; m < M; ++m) {
      if (targeted && !outlier[m]) {
        new_positions[m] = samplers[m].position();
        // builder convention: masses(v) seeds mass (inv_mass = 1/v), so the
        // sampler's inverse mass must be inverted back before re-seeding.
        new_masses[m] = samplers[m].inv_mass().cwiseInverse();
        new_steps[m] = samplers[m].macro_time();
      } else {
        new_positions[m] = cfg.init().init_chain_config(m).position();
        new_masses[m] = ar.mass_bar;
        new_steps[m] = step_bar;
      }
    }
    {  // both paths: outliers (or all chains, blanket) draw from the pool
      // Score pool draws by log-density at the position; keep finite ones,
      // best first, and assign distinct draws to the re-drawn chains.
      struct ScoredDraw {
        double logp;
        std::size_t pool_idx;
        std::size_t draw_idx;
      };
      std::vector<ScoredDraw> scored;
      for (std::size_t p = 0; p < reinit_positions.size(); ++p) {
        for (std::size_t k = 0; k < reinit_positions[p].size(); ++k) {
          double lp = 0.0;
          Eigen::VectorXd g(
              reinit_positions[p][k].size());
          bool ok = true;
          try {
            log_p_grad(reinit_positions[p][k], lp, g);
          } catch (...) {
            ok = false;
          }
          if (ok && std::isfinite(lp)) {
            scored.push_back({lp, p, k});
          }
        }
      }
      std::sort(scored.begin(), scored.end(),
                [](const ScoredDraw& a, const ScoredDraw& b) {
                  return a.logp > b.logp;
                });
      std::size_t take = 0;
      for (std::size_t m = 0; m < M; ++m) {
        if (targeted && !outlier[m]) continue;
        if (take < scored.size()) {
          new_positions[m] =
              reinit_positions[scored[take].pool_idx][scored[take].draw_idx];
          ++take;
        } else {
          const auto& pool =
              reinit_positions[(m + reinits) % reinit_positions.size()];
          new_positions[m] = pool[(m + reinits) % pool.size()];
        }
      }
    }
    // Reinit rounds inherit the warmup config including any init-robustness
    // settings (mass clamp, step-size heuristic): a fresh distant draw needs
    // the same safeguards as the initial round, or the restarted warmup
    // collapses the step size before the policy can help again.
    InitConfigBuilder builder{
        M, cfg.init().init_chain_config(0).position().size()};
    builder.step_sizes(new_steps);
    builder.positions(new_positions);
    builder.masses(new_masses);
    cfg = WalnutsConfig(builder.build(), cfg.warmup(), cfg.sampling());
  }

  detail::sample(config.sampling(), samplers, global_handler,
                 interrupt_callback);
}

}  // namespace walnutpie
