#pragma once

#include <cstddef>
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
    // Chains locked into different scales: re-draw initializations from the
    // pool (per-chain, cycling with the reinit round) and re-run warmup.
    ++reinits;
    std::vector<Eigen::VectorXd> new_positions(M);
    for (std::size_t m = 0; m < M; ++m) {
      const auto& pool = reinit_positions[(m + reinits) % reinit_positions.size()];
      new_positions[m] = pool[(m + reinits) % pool.size()];
    }
    InitConfigBuilder builder{
        M, cfg.init().init_chain_config(0).position().size()};
    builder.step_sizes(cfg.init().init_chain_config(0).step_size());
    builder.positions(new_positions);
    builder.masses(log_p_grad, cfg.warmup().mass_additive_smoothing());
    cfg = WalnutsConfig(builder.build(), cfg.warmup(), cfg.sampling());
  }

  detail::sample(config.sampling(), samplers, global_handler,
                 interrupt_callback);
}

}  // namespace walnutpie
