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

namespace walnutpie::detail {

/**
 * @brief Bound the ridge guard's replacement minimum-micro-step budget.
 *
 * The nominal floor of 16 is clipped to the positive cap. Saturate before
 * converting to size_t so very large demands, including +infinity, are safe.
 * This does not decide whether a ridge guard should fire.
 *
 * @throw std::invalid_argument If cap is zero, or scale is NaN or below one.
 */
inline std::size_t ridge_micro_budget(double scale, std::size_t cap) {
  if (cap == 0 || !(scale >= 1.0)) {
    throw std::invalid_argument("ridge budget needs cap > 0 and scale >= 1");
  }
  if (cap <= 16) {
    return cap;
  }
  const double requested = 16.0 * scale;
  if (!(requested < static_cast<double>(cap))) {
    return cap;
  }
  return std::max<std::size_t>(16, static_cast<std::size_t>(requested));
}

}  // namespace walnutpie::detail

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

  // Opt-in experimental detector: compare final chain positions (not chain
  // means) with the adapted within-chain scale sqrt(inv_mass). A large ratio
  // is not proof of a null ridge or a convergence failure. On a fire, rebuild
  // the samplers with a scaled replacement budget bounded by ridge_min_micro.
  // The nominal floor 16 is clipped to that cap, including caps below 16.
  // Replacement can lower the adapted budget; this is not always an increase.
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
      const std::size_t budget = detail::ridge_micro_budget(scale, cap);
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
