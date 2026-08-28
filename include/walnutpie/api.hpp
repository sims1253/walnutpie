#pragma once

#include <algorithm>
#include <cmath>
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

  // Ridge guard (opt-in; sampling().ridge_guard() == 0 disables it):
  // chains that locked onto different points of a likelihood-null ridge
  // disperse far more across chains than the adapted within-chain scale
  // sqrt(inv_mass). The log density is invariant along such ridges, so
  // no log-mass statistic can detect the lock; positions can. On
  // detection, raise the frozen trajectory budget, scaled to the misfit:
  // longer trajectories traverse the ridge (the lock is trajectory-
  // length binding, not metric binding).
  const double ridge_threshold = config.sampling().ridge_guard();
  if (ridge_threshold > 0.0 && samplers.size() > 1) {
    const std::size_t chains = samplers.size();
    const std::size_t dim =
        static_cast<std::size_t>(samplers[0].position().size());
    double worst_f = 0.0;
    for (std::size_t j = 0; j < dim; ++j) {
      double mean_of_means = 0.0;
      double mean_scale = 0.0;
      for (std::size_t c = 0; c < chains; ++c) {
        mean_of_means += samplers[c].position()[j];
        mean_scale += std::sqrt(samplers[c].inverse_mass_matrix_diagonal()[j]);
      }
      mean_of_means /= static_cast<double>(chains);
      mean_scale /= static_cast<double>(chains);
      if (!(mean_scale > 0.0)) {
        continue;
      }
      double ss = 0.0;
      for (std::size_t c = 0; c < chains; ++c) {
        const double dev = samplers[c].position()[j] - mean_of_means;
        ss += dev * dev;
      }
      const double f = std::sqrt(ss / static_cast<double>(chains - 1))
                       / mean_scale;
      worst_f = std::max(worst_f, f);
    }
    if (worst_f > ridge_threshold) {
      const double scale = worst_f / ridge_threshold;
      const std::size_t cap = config.sampling().ridge_min_micro();
      const std::size_t budget = std::max<std::size_t>(
          std::min(static_cast<std::size_t>(16.0 * std::max(scale, 1.0)),
                   cap),
          16);
      std::vector<Sampler> replaced;
      replaced.reserve(chains);
      for (std::size_t c = 0; c < chains; ++c) {
        replaced.emplace_back(adapters[c].sampler_min_micro(budget));
      }
      samplers = std::move(replaced);
    }
  }

  detail::sample(config.sampling(), samplers, global_handler,
                 interrupt_callback);
}

}  // namespace walnutpie
