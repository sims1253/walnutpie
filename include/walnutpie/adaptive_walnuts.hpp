#pragma once

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <vector>
#include <cstddef>
#include <functional>
#include <random>
#include <utility>

#include <Eigen/Dense>

#include "walnutpie/adam.hpp"
#include "walnutpie/step_optimizers.hpp"
#include "walnutpie/concepts.hpp"
#include "walnutpie/config.hpp"
#include "walnutpie/online_moments.hpp"
#include "walnutpie/low_rank_metric.hpp"
#include "walnutpie/util.hpp"
#include "walnutpie/walnuts.hpp"
#include "walnutpie/warmup_heuristics.hpp"

namespace walnutpie::detail {

/**
 * @brief A mass matrix estimator based on exponentially discounted draws
 * and scores (gradients of log densities).
 */
class MassEstimator {
 public:
  /**
   * @brief Construct a mass matrix estimator with the specified configuration,
   * at the specified initial position and gradient of the log density at the
   * position.
   *
   * The estimator observes positions and their gradients at given iterations
   * with the function `observe()`. At each step, the discount factor for
   * discounting past draws the online moment estimators is set to
   * ```
   * discount_factor = 1 - 1 / (iter_offset + iter)
   * ```
   * where `iter_offset` is the offset specified in the configuration and `iter`
   * is the iteration number for the observation.
   *
   * The initial estimate is additively smoothed by multiplying by one
   * minus the additive smoothing and adding the additive smoothing.
   *
   * The final estimate for the inverse mass matrix is given by the
   * geometric mean of the variance of the scores (the inverse
   * variance estimator) and the inverse variance of the draws (the
   * variance estimator).
   *
   * @param[in] warmup_cfg The warmup configuration.
   * @param[in] init_cfg The initialization configuration.
   * @throw std::invalid_argument If the position and gradient are not the same
   * size.
   */
  MassEstimator(const WarmupConfig& warmup_cfg, const InitChainConfig& init_cfg)
      : warmup_cfg_(warmup_cfg),
        init_score_var_(init_cfg.mass()),
        init_draw_var_(init_cfg.mass().array().inverse().matrix()) {
    Eigen::VectorXd zero = Eigen::VectorXd::Zero(init_cfg.position().size());
    score_var_estimator_ =
        OnlineMoments(warmup_cfg.mass_init_count(), zero, init_cfg.mass());
    draw_var_estimator_ =
        OnlineMoments(warmup_cfg.mass_init_count(), zero,
                      init_cfg.mass().array().inverse().matrix());
  }

  /**
   * @brief Return the effective sample size of the draw moment estimates
   * (Kish weight of the discounted Welford accumulator).
   */
  double draw_n_eff() const { return draw_var_estimator_.weight(); }

  /**
   * @brief Update the estimate for the specified iteration with the
   * observation and gradient.
   *
   * @param[in] theta The position observed.
   * @param[in] grad The gradient of the log density at the position.
   * @param[in] iteration The iteration number.
   * @pre theta.size() = grad.size()
   * @pre iteration >= 0
   */
  void observe(const Eigen::VectorXd& theta, const Eigen::VectorXd& grad,
               std::size_t iteration) {
    double discount_factor = 1.0 - 1.0 / (warmup_cfg_.mass_init_count() +
                                          static_cast<double>(iteration));
    draw_var_estimator_.discount_observe(discount_factor, theta);
    score_var_estimator_.discount_observe(discount_factor, grad);
    if (warmup_cfg_.metric_rank() > 0) {
      const std::size_t window = warmup_cfg_.metric_window();
      if (window > 0) {
        window_draws_.push_back(theta);
        window_scores_.push_back(grad);
        if (window_draws_.size() > window) {
          window_draws_.erase(window_draws_.begin());
          window_scores_.erase(window_scores_.begin());
        }
      }
    }
    if (warmup_cfg_.metric_stall_reset() > 0) {
      // Stall detector on the raw draws (metric-independent): if the chain
      // has barely moved over the last `stall_window` observations, the mass
      // estimate is being fed by a pinned chain and cannot recover (tiny
      // metric -> tiny moves -> tiny Var_draw -> tinier metric). Break the
      // loop by resetting both accumulators to their seeds.
      if (stall_reference_.size() == 0) {
        stall_reference_ = theta;
        stall_countdown_ = warmup_cfg_.metric_stall_window();
      } else if (--stall_countdown_ == 0) {
        const double movement =
            (theta - stall_reference_).cwiseAbs().maxCoeff();
        if (movement < warmup_cfg_.metric_stall_reset()) {
          reset_to_seeds();
        }
        stall_reference_ = theta;
        stall_countdown_ = warmup_cfg_.metric_stall_window();
      }
    }
  }

  /**
   * @brief Reset both moment accumulators to their initialization seeds.
   */
  void reset_to_seeds() {
    Eigen::VectorXd zero = Eigen::VectorXd::Zero(init_draw_var_.size());
    score_var_estimator_ = OnlineMoments(
        warmup_cfg_.mass_init_count(), zero, init_score_var_);
    draw_var_estimator_ = OnlineMoments(
        warmup_cfg_.mass_init_count(), zero, init_draw_var_);
  }

  /**
   * @brief Return an estimate of the inverse mass matrix. The result
   * is the geometric average of the variance of the draws and the
   * inverse variance of the scores.
   *
   * @return The inverse mass matrix estimate.
   */
  /**
   * @brief Aggregate two variance vectors in log space (geometric mean).
   *
   * The arithmetic average of two variances is dominated by the larger one;
   * the log-space average preserves relative scale information when the two
   * estimates disagree by orders of magnitude, as happens during the early
   * drift from a distant initialization.
   */
  static Eigen::VectorXd logspace_average(const Eigen::VectorXd& a,
                                          const Eigen::VectorXd& b) {
    return ((a.array().log() + b.array().log()) * 0.5).exp().matrix();
  }

  /**
   * @brief Rank-corrected diagonal estimate: folds the low-rank correction
   * sqrt(D) U C U^T sqrt(D) into its per-coordinate marginal
   *   d_eff_i = d_i + sqrt(d_i) * (U diag(c) U^T)_ii * sqrt(d_i)
   * keeping the diagonal interface of transition_w while carrying the
   * dominant directions of the draw-score cross structure (full rank part
   * reserved for a dedicated transition variant; see low_rank_metric.hpp).
   */
  const Eigen::MatrixXd& rank_U() const { return U_; }
  const Eigen::VectorXd& rank_c() const { return c_; }

  Eigen::VectorXd rank_folded_estimate() const {
    Eigen::VectorXd diag = inv_mass_estimate();
    if (warmup_cfg_.metric_rank() == 0 ||
        draw_var_estimator_.weight() <
            2.0 * warmup_cfg_.mass_init_count()) {
      return diag;
    }
    LowRankMetricEstimator lr(draw_var_estimator_.mean().size(),
                              warmup_cfg_.metric_rank(),
                              warmup_cfg_.metric_window());
    // replay is unavailable in the streaming estimator; instead the rank
    // factors are refreshed by the explicit low_rank_update() below.
    if (U_.cols() == 0) {
      return diag;
    }
    Eigen::VectorXd sq = diag.cwiseSqrt();
    // marginal: sqrtD (U C U^T) sqrtD -> row-wise weighted sum
    Eigen::VectorXd marg =
        (U_.array().square().matrix() * c_).cwiseProduct(sq.cwiseProduct(sq));
    return diag + marg;
  }

  /**
   * @brief Refresh the low-rank factors from the accumulated window.
   *
   * Called at window boundaries (chopping); stores U, c for the folded
   * diagonal estimate. Draws/scores come from the streaming accumulators'
   * raw window, retained for exactly this purpose.
   */
  static Eigen::VectorXd row_sample_variance_pub(const Eigen::MatrixXd& M) {
    Eigen::VectorXd mu = M.rowwise().mean();
    Eigen::MatrixXd centered = M.colwise() - mu;
    return centered.cwiseProduct(centered).rowwise().sum() /
           std::max(1.0, static_cast<double>(M.cols() - 1));
  }

  /**
   * @brief Cross-structure strength of the current window: the second-to-first
   * singular value ratio of the standardized stacked [draws | scores] matrix.
   *
   * ~0: geometry is (conditionally) diagonal, rank corrections are noise.
   * O(1): strong off-diagonal structure, rank corrections carry signal.
   */
  double window_cross_ratio() const {
    // Concentration of the singular-value EXCESS above the isotropic baseline
    // (sqrt(2K)) in the top-r directions. Empirically inverted as a screening
    // signal: spread spectra (low fraction, e.g. < 0.1) mark genuinely
    // cross-correlated geometry where rank corrections help; concentrated
    // spectra (fraction -> 1) mark funnel/spike geometry where rank
    // corrections destabilize (eight_schools_centered, blr).
    if (window_draws_.size() < 4) return 1.0;
    const std::size_t dim = window_draws_[0].size();
    const std::size_t K = window_draws_.size();
    const std::size_t r = std::min<std::size_t>(
        5, std::min(dim, static_cast<std::size_t>(K / 4)));
    if (r == 0) return 1.0;
    Eigen::MatrixXd Y(dim, K), S(dim, K);
    for (std::size_t k = 0; k < K; ++k) {
      Y.col(k) = window_draws_[k];
      S.col(k) = window_scores_[k];
    }
    Eigen::VectorXd vy = row_sample_variance_pub(Y);
    Eigen::VectorXd vs = row_sample_variance_pub(S);
    Eigen::MatrixXd Ys = vy.cwiseSqrt().cwiseInverse().asDiagonal() * Y;
    Eigen::MatrixXd Ss = vs.cwiseSqrt().cwiseInverse().asDiagonal() * S;
    Eigen::MatrixXd st(dim, 2 * K);
    st << Ys, Ss;
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(st, Eigen::ComputeThinU);
    const auto& sv = svd.singularValues();
    if (sv.size() < 2) return 1.0;
    const double baseline = std::sqrt(2.0 * static_cast<double>(K));
    double excess_total = 0.0;
    for (int i = 0; i < sv.size(); ++i) {
      excess_total += std::max(0.0, sv[i] - baseline);
    }
    if (excess_total <= 1e-12) return 1.0;
    double excess_top = 0.0;
    for (std::size_t i = 0; i < r && i < static_cast<std::size_t>(sv.size()); ++i) {
      excess_top += std::max(0.0, sv[i] - baseline);
    }
    return excess_top / excess_total;
  }

  void low_rank_update() {
    if (warmup_cfg_.metric_rank() == 0 || window_draws_.size() < 4) {
      return;
    }
    const std::size_t dim = window_draws_[0].size();
    LowRankMetricEstimator lr(dim, warmup_cfg_.metric_rank(),
                              warmup_cfg_.metric_window(),
                              warmup_cfg_.metric_basis());
    for (std::size_t k = 0; k < window_draws_.size(); ++k) {
      lr.observe(window_draws_[k], window_scores_[k]);
    }
    Eigen::VectorXd diag = inv_mass_estimate();
    lr.low_rank_factors(U_, c_, diag);
  }

  Eigen::VectorXd inv_mass_estimate() const {
    Eigen::VectorXd draw_var = draw_var_estimator_.variance();
    Eigen::VectorXd score_var = score_var_estimator_.variance();
    if (warmup_cfg_.metric_drift_guard()) {
      // During the initial drift from a distant point, the two moment
      // estimators can disagree by orders of magnitude (score variance on
      // plateau gradients vs draw variance of a pinned chain). An arithmetic
      // combination then produces an unusable metric either way; aggregate in
      // log space so relative scale information survives on both sides.
      draw_var = logspace_average(draw_var, init_draw_var_);
      score_var = logspace_average(score_var, init_score_var_);
    }
    const double kappa = warmup_cfg_.mass_shrink_kappa();
    if (kappa > 0) {
      // Regularized (shrinkage) estimates in the style of Stan's
      // var_adaptation: var <- (n/(n+kappa)) var + (kappa/(n+kappa)) var_init,
      // using the discounted weight as effective sample size.
      const double n_draw = draw_var_estimator_.weight();
      const double n_score = score_var_estimator_.weight();
      const double w_draw = n_draw / (n_draw + kappa);
      const double w_score = n_score / (n_score + kappa);
      draw_var = (w_draw * draw_var.array() +
                  (1 - w_draw) * draw_var_estimator_.initial_variance().array())
                     .matrix();
      score_var = (w_score * score_var.array() +
                   (1 - w_score) * score_var_estimator_.initial_variance().array())
                      .matrix();
    }
    const double floor_v = warmup_cfg_.mass_var_floor();
    if (floor_v > 0) {
      draw_var = draw_var.cwiseMax(Eigen::VectorXd::Constant(
          draw_var.size(), floor_v));
      score_var = score_var.cwiseMax(Eigen::VectorXd::Constant(
          score_var.size(), floor_v));
    }
    // Combine the two reciprocal-scale estimates: estimate A = Var_draw
    // (posterior scale seen by the chain), estimate B = 1 / Var_score
    // (Fisher-information scale from gradients). The classical geometric
    // mean (power p -> 0) makes a collapsed A drag the metric to zero even
    // when B is healthy; a higher-power mean lets the healthy estimate
    // rescue the collapsed one (p = 1 arithmetic, p -> infinity max).
    const double pcomb = warmup_cfg_.mass_combine_power();
    Eigen::VectorXd est_a = draw_var;
    Eigen::VectorXd est_b = score_var.array().inverse().matrix();
    if (pcomb <= 0.0) {
      return (est_a.array() * est_b.array()).sqrt().matrix();  // geometric
    }
    if (pcomb >= 64.0) {  // effectively max
      return est_a.cwiseMax(est_b);
    }
    Eigen::VectorXd mp = ((est_a.array().pow(pcomb) +
                           est_b.array().pow(pcomb)) * 0.5)
                              .pow(1.0 / pcomb)
                              .matrix();
    return mp;
  }

 private:
  /** The warmup configuration for adaptive Walnutpie. */
  WarmupConfig warmup_cfg_;

  /** The online variance estimator for draws. */
  OnlineMoments draw_var_estimator_;

  /** The online inverse variance estimator for scores. */
  OnlineMoments score_var_estimator_;

  /** Initial variance seeds (regularization/blend targets). */
  Eigen::VectorXd init_score_var_;
  Eigen::VectorXd init_draw_var_;

  /** Last reference position for the stall detector. */
  Eigen::VectorXd stall_reference_;

  /** Low-rank factors of the metric (empty when metric_rank == 0). */
  Eigen::MatrixXd U_;
  Eigen::VectorXd c_;

  /** Rolling raw window of draws/scores for low-rank refresh. */
  std::vector<Eigen::VectorXd> window_draws_;
  std::vector<Eigen::VectorXd> window_scores_;

  /** Iterations until the next stall check. */
  std::size_t stall_countdown_ = 0;
};

/**
 * @brief The adaptation handler for the minimum number of micro steps
 * per macro step.
 *
 * After being constructed with a target number of macro steps, this
 * class is given observations of the number of micro steps taken and
 * adjusts the minimum number of micro steps per macro step in order
 * to achieve the target expected number of macro steps historically.
 * There is slight regularization of a single observation at depth 2,
 * but otherwise it just uses the floor of an average and thus rounds
 * down.
 */
class MinMicroStepsAdaptHandler {
 public:
  /**
   * Construct a minimum number of micro steps per macro step handler.
   *
   * @param[in] target_macro_steps Target number of expected macro steps.
   * @param[in] min_micro_steps The minimum number of micro steps to return.
   */
  MinMicroStepsAdaptHandler(double target_macro_steps,
                            std::size_t min_micro_steps)
      : target_macro_steps_(target_macro_steps),
        min_micro_steps_(min_micro_steps),
        total_macro_steps_(2.0),
        count_(1.0) {}

  /**
   * @brief Observe the specified number of macro steps in a Nuts trajectory.
   *
   * @param[in] macro_steps The number of macro steps used in a trajectory.
   */
  void observe(std::size_t macro_steps) {
    total_macro_steps_ += static_cast<double>(macro_steps);
    ++count_;
  }

  /**
   * @brief Return the estimated minimum number of micro steps.
   *
   * This estimate is designed to achieve the expected number of macro steps
   * per iteration.
   *
   * @return The minimum number of micro steps to use per macro step.
   */
  std::size_t min_micro_steps() const noexcept {
    double mean_micro = total_macro_steps_ / count_;
    double min_micro_steps = mean_micro / target_macro_steps_;
    return std::max(min_micro_steps_,
                    static_cast<std::size_t>(std::lround(min_micro_steps)));
  }

 private:
  const double target_macro_steps_;
  const std::size_t min_micro_steps_;
  double total_macro_steps_;
  double count_;
};

}  // namespace walnutpie::detail


namespace walnutpie::detail {

/**
 * @brief Traits-based factory for step size adapters.
 *
 * Specializations construct each adapter shipped with this library from the
 * chain initialization and warmup configuration. User code may add
 * specializations for custom adapters.
 */
template <class Opt>
struct StepAdapterFactory;  // primary: intentionally undefined

template <>
struct StepAdapterFactory<Adam> {
  static Adam make(const InitChainConfig& init_cfg,
                   const WarmupConfig& warmup_cfg) {
    return Adam(init_cfg.step_size(), warmup_cfg.step_accept_rate_target(),
                warmup_cfg.step_learning_rate(),
                warmup_cfg.step_gradient_decay(),
                warmup_cfg.step_sq_gradient_decay(),
                warmup_cfg.step_stabilization(),
                warmup_cfg.step_learn_rate_decay());
  }
};

template <>
struct StepAdapterFactory<DualAveraging> {
  static DualAveraging make(const InitChainConfig& init_cfg,
                            const WarmupConfig& warmup_cfg) {
    return DualAveraging(init_cfg.step_size(),
                         warmup_cfg.step_accept_rate_target(),
                         warmup_cfg.da_gamma(), warmup_cfg.da_t0(),
                         warmup_cfg.da_kappa(),
                         warmup_cfg.da_freeze_average());
  }
};

template <>
struct StepAdapterFactory<AdEMAMix> {
  static AdEMAMix make(const InitChainConfig& init_cfg,
                       const WarmupConfig& warmup_cfg) {
    return AdEMAMix(init_cfg.step_size(), warmup_cfg.step_accept_rate_target(),
                    warmup_cfg.step_learning_rate(),
                    warmup_cfg.step_gradient_decay(),
                    warmup_cfg.step_sq_gradient_decay(), 0.9999,
                    warmup_cfg.step_stabilization(),
                    warmup_cfg.step_learn_rate_decay(),
                    warmup_cfg.slow_ema_warmup());
  }
};

template <>
struct StepAdapterFactory<AdaBelief> {
  static AdaBelief make(const InitChainConfig& init_cfg,
                        const WarmupConfig& warmup_cfg) {
    return AdaBelief(init_cfg.step_size(), warmup_cfg.step_accept_rate_target(),
                     warmup_cfg.step_learning_rate(),
                     warmup_cfg.step_gradient_decay(),
                     warmup_cfg.step_sq_gradient_decay(),
                     warmup_cfg.step_stabilization(),
                     warmup_cfg.step_learn_rate_decay());
  }
};

template <StepSizeAdapter Inner>
struct StepAdapterFactory<BatchedAdapter<Inner>> {
  static BatchedAdapter<Inner> make(const InitChainConfig& init_cfg,
                                    const WarmupConfig& warmup_cfg) {
    return BatchedAdapter<Inner>(
        StepAdapterFactory<Inner>::make(init_cfg, warmup_cfg),
        warmup_cfg.step_opt_batch_stride());
  }
};

template <StepSizeAdapter Inner>
struct StepAdapterFactory<AntiWindupAdapter<Inner>> {
  static AntiWindupAdapter<Inner> make(const InitChainConfig& init_cfg,
                                       const WarmupConfig& warmup_cfg) {
    return AntiWindupAdapter<Inner>(
        StepAdapterFactory<Inner>::make(init_cfg, warmup_cfg),
        1e-12, warmup_cfg.anti_windup_pass_rate());
  }
};

template <StepSizeAdapter Inner>
struct StepAdapterFactory<ClippedAdapter<Inner>> {
  static ClippedAdapter<Inner> make(const InitChainConfig& init_cfg,
                                    const WarmupConfig& warmup_cfg) {
    return ClippedAdapter<Inner>(
        StepAdapterFactory<Inner>::make(init_cfg, warmup_cfg),
        warmup_cfg.step_grad_clip());
  }
};

template <StepSizeAdapter Inner>
struct StepAdapterFactory<ClippedAdapter<BatchedAdapter<Inner>>> {
  static ClippedAdapter<BatchedAdapter<Inner>> make(
      const InitChainConfig& init_cfg, const WarmupConfig& warmup_cfg) {
    return ClippedAdapter<BatchedAdapter<Inner>>(
        StepAdapterFactory<BatchedAdapter<Inner>>::make(init_cfg, warmup_cfg),
        warmup_cfg.step_grad_clip());
  }
};

}  // namespace walnutpie::detail

namespace walnutpie {

/**
 * @brief Construct the step adapter requested by the warmup configuration.
 *
 * Library users cannot template-dispatch as the CLI does, so configuration
 * carries the selection: anti-windup wrapping (pass_rate > 0) is applied
 * around the requested base adapter type. Default: the base adapter itself.
 */
template <detail::StepSizeAdapter Opt>
Opt make_configured_adapter(const InitChainConfig& init_cfg,
                            const WarmupConfig& warmup_cfg) {
  // The factory for AntiWindupAdapter reads the configured pass rate
  // (0 = pass-through), so library users select anti-windup purely through
  // WarmupConfig — no template dispatch needed, matching the CLI behavior.
  return detail::StepAdapterFactory<Opt>::make(init_cfg, warmup_cfg);
}

/**
 * @brief The adaptive Walnuts sampler.
 *
 * The adaptive Walnuts sampler is configured in the constructor, then
 * provides a functor method `operator()()` for returning the next
 * state in warmup. Warmup re-estimates step size and mass matrix
 * each iteration, exponentially discounting the past.
 *
 * @tparam F Type of log density/gradient function.
 * @tparam RNG Type of base random number generator.
 * @tparam Handler Type of adaptation and sampling event handler.
 */
template <LogpGrad F, std::uniform_random_bit_generator RNG, ChainHandler H,
          detail::StepSizeAdapter Opt = detail::AntiWindupAdapter<detail::Adam>>
class AdaptiveWalnuts {
 public:
  /**
   * @brief Construct an adaptive Walnuts sampler.
   *
   * The configuration objects, the base random number generator, and
   * the log density/gradient function are held by reference. The RNG
   * changes state every time a random number is generated. The
   * target depth specifies the expected Nuts tree depth, which is
   * controlled through the minimum number of micro steps per macro
   * step and adjusted with a mean estimator to achieve this average.
   *
   * @param[in] rng The base random number generator, stored by reference and
   * modifed.
   * @param[in,out] handler Event handler for adaptation and sampling, stored by
   * reference and called back.
   * @param[in] logp_grad The target log density and gradient function.
   * @param[in] init_chain_cfg The initialization configuration for a single
   * chain.
   * @param[in] warmup_cfg The warmup configuration.
   * @param[in] sampling_cfg The sampling configuration.
   */
  AdaptiveWalnuts(RNG& rng, H& handler, const F& logp_grad,
                  const InitChainConfig& init_chain_cfg,
                  const WarmupConfig& warmup_cfg,
                  const SamplingConfig& sampling_cfg)
      : warmup_cfg_(std::cref(warmup_cfg)),
        sampling_cfg_(std::cref(sampling_cfg)),
        rand_(rng),
        handler_(handler),
        logp_grad_(logp_grad, handler),
        theta_(init_chain_cfg.position()),
        // W-42: seed the endpoint cache with the init-position (grad,
        // logp) recorded by InitConfigBuilder::masses() — the same
        // (position, function) pair the first transition would otherwise
        // re-evaluate at its start. Reused doubles change no arithmetic
        // (W-23 precedent); the first warmup transition skips one
        // logp_grad call.
        cached_grad_(
            (init_chain_cfg.has_init_eval() &&
             init_chain_cfg.init_grad().size() ==
                 init_chain_cfg.position().size())
                ? init_chain_cfg.init_grad()
                : Eigen::VectorXd()),
        cached_logp_(init_chain_cfg.has_init_eval()
                         ? init_chain_cfg.init_logp()
                         : -std::numeric_limits<double>::infinity()),
        last_mass_(init_chain_cfg.mass()),
        iteration_(0),
        opt_(make_configured_adapter<Opt>(init_chain_cfg, warmup_cfg)),
        mass_estimator_(warmup_cfg, init_chain_cfg),
        min_micro_estimator_(warmup_cfg.max_macro_steps_target(),
                             sampling_cfg.min_micro_steps()),
        init_step_(init_chain_cfg.step_size()),
        last_finite_step_(init_step_) {}

  /**
   * @brief Generate the next state for adaptation and the handler.
   *
   * This method should be called a number of time equal to the number
   * of warmup iterations desired. These warmup draws are *not* drawn
   * from a Markov chain and are not valid for inference. After
   * warmup, call `sampler()` to return a sampler that fixes the
   * tuning parameters and provides a proper Markov chain.
   */
  /**
   * @brief Effective Hamiltonian-error cap for the current iteration.
   *
   * With a max-error schedule configured, interpolates geometrically (in log
   * space) from `max_error_start` down to the configured cap over the first
   * `max_error_schedule_iters` iterations; afterwards returns the configured
   * cap. Option (c) for robustness to distant initializations: the loose
   * early cap lets trajectories through while the chain drifts toward the
   * typical set, then tightens to the intended error control.
   */
  double effective_max_error() const {
    const double base = sampling_cfg_.get().max_hamiltonian_error();
    const double start = warmup_cfg_.get().max_error_start();
    const std::size_t iters = warmup_cfg_.get().max_error_schedule_iters();
    if (!(start > base) || iters == 0 || iteration_ >= iters) {
      return base;
    }
    const double frac =
        static_cast<double>(iteration_) / static_cast<double>(iters);
    return std::exp(std::log(start) +
                    frac * (std::log(base) - std::log(start)));
  }

  void operator()() {
    const bool drifting = iteration_ < warmup_cfg_.get().drift_iters();
    const std::size_t window = warmup_cfg_.get().metric_window();
    if (window > 0 && !drifting && iteration_ > 0 &&
        (iteration_ + 1) % window == 0) {
      mass_estimator_.low_rank_update();
    }
    const bool full_rank_mode =
        warmup_cfg_.get().metric_rank() > 0 && warmup_cfg_.get().metric_full();
    const bool auto_screen = warmup_cfg_.get().metric_auto() > 0;
    const bool rank_active =
        warmup_cfg_.get().metric_rank() > 0 &&
        (!auto_screen ||
         mass_estimator_.window_cross_ratio() <= warmup_cfg_.get().metric_auto());
    Eigen::VectorXd inv_mass =
        drifting ? Eigen::VectorXd::Ones(theta_.size())
                 : (rank_active
                        ? mass_estimator_.rank_folded_estimate()
                        : mass_estimator_.inv_mass_estimate());
    Eigen::VectorXd chol_mass = inv_mass.array().inverse().sqrt().matrix();
    if (full_rank_mode) {
      detail::LowRankMass lrm;
      lrm.D = mass_estimator_.inv_mass_estimate();
      lrm.U = mass_estimator_.rank_U();
      lrm.c = mass_estimator_.rank_c();
      Eigen::VectorXd grad_select;
      double logp_select;
      std::size_t depth;
      theta_ = detail::transition_w_lr(
          rand_, logp_grad_.logp_grad_, lrm, opt_.step_size(),
          sampling_cfg_.get().max_trajectory_doublings(),
          sampling_cfg_.get().max_step_halvings(),
          min_micro_estimator_.min_micro_steps(),
          drifting ? std::numeric_limits<double>::infinity()
                   : effective_max_error(),
          std::move(theta_), depth, grad_select, logp_select, opt_,
          cached_grad_, cached_logp_);
      // W-23: cache the endpoint (grad, logp) for the next transition's
      // start-position reuse (duplicate eval elimination, see W-20).
      cached_grad_ = grad_select;
      cached_logp_ = logp_select;
      if (!drifting) {
        mass_estimator_.observe(theta_, grad_select, iteration_);
      }
      // Full-rank mode integrates with the low-rank OPERATOR whose diagonal
      // is the UNFOLDED inv_mass_estimate() (see lrm.D above), not the folded
      // inv_mass this iteration also computed. The frozen sampler rebuilds
      // lrm.D from inv_mass(), so the memo must carry the unfolded diagonal
      // (mass convention) or sampling silently runs a different operator than
      // warmup tuned under — the third instance of the freeze-mismatch family.
      last_mass_ = lrm.D.cwiseInverse();
      min_micro_estimator_.observe(1 << depth);
      handler_.get().on_warmup(theta_, logp_select, step_size(), lrm.D);
      note_step_();
      ++iteration_;
      return;
    }
    Eigen::VectorXd grad_select;
    double logp_select;
    std::size_t depth;
    // During the drift phase the error cap is suspended entirely (option (b)):
    // a distant initialization cannot satisfy any tight cap, and rejecting
    // every macro step pins the chain at its starting point.
    const double max_err =
        drifting ? std::numeric_limits<double>::infinity()
                 : effective_max_error();
    // During drift the acceptance statistics are meaningless (huge energy
    // errors by construction), so the step adapter is not updated either:
    // WALNUTS' within-orbit dyadic step adaptation already selects viable
    // micro steps, and feeding the adapter saturated alphas drives the macro
    // step toward zero and freezes the chain when the drift phase ends.
    detail::NoOpStepSizeAdapter drift_noop;
    if (drifting) {
      theta_ = transition_w(rand_, logp_grad_, inv_mass, chol_mass,
                            opt_.step_size(),
                            sampling_cfg_.get().max_trajectory_doublings(),
                            sampling_cfg_.get().max_step_halvings(),
                            min_micro_estimator_.min_micro_steps(), max_err,
                            std::move(theta_), depth, grad_select,
                            logp_select, drift_noop, cached_grad_,
                            cached_logp_);
    } else {
      theta_ = transition_w(rand_, logp_grad_, inv_mass, chol_mass,
                            opt_.step_size(),
                            sampling_cfg_.get().max_trajectory_doublings(),
                            sampling_cfg_.get().max_step_halvings(),
                            min_micro_estimator_.min_micro_steps(), max_err,
                            std::move(theta_), depth, grad_select,
                            logp_select, opt_, cached_grad_, cached_logp_);
    }
    cached_grad_ = grad_select;
    cached_logp_ = logp_select;
    if (!drifting) {
      // Suspend metric estimation during drift: the draws observed while the
      // chain is pinned/throttled poison the variance estimates (the
      // self-locking failure mode documented in the init-robustness notes).
      mass_estimator_.observe(theta_, grad_select, iteration_);
      // Memoryless windows ("chopping", Fisher-HMC discipline,
      // arXiv:2603.18845): at each window boundary, discard the accumulated
      // history entirely rather than exponentially discounting it forward.
      // Stale early draws are noise, not signal; the metric is rebuilt from
      // post-drift samples only.
      const std::size_t window = warmup_cfg_.get().metric_window();
      if (window > 0 && iteration_ > 0 && (iteration_ + 1) % window == 0 &&
          iteration_ + 1 < warmup_cfg_.get().max_iter()) {
        mass_estimator_.reset_to_seeds();
      }
    }
    last_mass_ = inv_mass.cwiseInverse();
    min_micro_estimator_.observe(1 << depth);
    handler_.get().on_warmup(theta_, logp_select, step_size(), inv_mass);
    note_step_();
    ++iteration_;
  }

  /**
   * @brief Return a Walnuts sampler with the current tuning parameter
   * estimates.
   *
   * The returned sampler forms a proper Markov chain. The method passes
   * along the compound random number generator and log density function and
   * is hence not marked `const`.
   *
   * W-41 freeze clamp: the frozen macro time is `step_size()`, and a
   * degenerate warmup (e.g. log density -inf at the initialization, which
   * NaNs the acceptance statistic and with it the adapter state) can leave
   * it 0 / NaN / +inf. The WalnutsSampler constructor validates it and
   * throws, aborting the whole run at the warmup/sampling boundary. Instead
   * of aborting, fall back (in order) to (a) the last finite step size
   * observed during warmup (seeded with the initial step), (b) a
   * find_reasonable_step re-derivation at the current position, (c) a
   * documented hard floor. The fallback is computed once and cached; a loud
   * warning is written to stderr so runs remain auditable. Healthy freezes
   * are untouched: the clamp is dead code while `step_size()` is finite and
   * positive.
   *
   * @return The Walnuts sampler with current tuning parameter estimates.
   */
  WalnutsSampler<F, RNG, H> sampler() {
    double macro_time = step_size();
    if (!(std::isfinite(macro_time) && macro_time > 0.0)) {
      if (!freeze_clamped_) {
        freeze_fallback_step_ = freeze_step_fallback(macro_time);
        freeze_clamped_ = true;
      }
      macro_time = freeze_fallback_step_;
    }
    handler_.get().on_warmup_complete(macro_time, inv_mass());
    WalnutsSampler<F, RNG, H> out(
        rand_.rng(), handler_, logp_grad_.logp_grad_, theta_, inv_mass(),
        macro_time, sampling_cfg_.get().max_trajectory_doublings(),
        sampling_cfg_.get().max_step_halvings(),
        min_micro_estimator_.min_micro_steps(),
        sampling_cfg_.get().max_hamiltonian_error());
    if (warmup_cfg_.get().metric_rank() > 0 &&
        warmup_cfg_.get().metric_full()) {
      // Preserve the low-rank factors the warmup adapted with: freezing to
      // the diagonal alone would sample under a different metric than the
      // one step size and micro-step tuning were calibrated for.
      out.set_low_rank(mass_estimator_.rank_U(), mass_estimator_.rank_c());
    }
    // W-23: seed the frozen sampler's endpoint cache with the final warmup
    // transition's (grad, logp) at exactly this position, so the first
    // sampling transition skips its start-position re-evaluation too.
    out.seed_endpoint_cache(cached_grad_, cached_logp_);
    return out;
  }

  /**
   * @brief Return the diagonal of the diagonal inverse mass matrix.
   *
   * @return The diagonal of the inverse mass matrix.
   */
  Eigen::VectorXd inv_mass() const {
    // The frozen sampler must carry the SAME metric the last warmup
    // transitions used. In fold mode (metric_rank > 0, rank active per the
    // auto-screen) warmup integrates with rank_folded_estimate(); freezing
    // with the unfolded estimate silently changes the Hamiltonian at the
    // warmup/sampling boundary (step size was tuned for the folded metric).
    // The memo avoids a second staleness hazard: operator() recomputes the
    // estimate AFTER the final observe(), so recomputing here reads an
    // estimator one draw ahead of the last transition and can flip the
    // auto-screen decision at the freeze boundary.
    return last_mass_.cwiseInverse();
  }

  /**
   * @brief Return the step size.
   *
   * @return The step size.
   */
  double step_size() const { return opt_.step_size(); }

  /**
   * @brief Return the minimum number of micro steps per macro step.
   *
   * @return The minimum number of micro steps per macro step.
   */
  std::size_t min_micro_steps() const {
    return min_micro_estimator_.min_micro_steps();
  }

  /**
   * @brief Return the number of dimensions of the position.
   *
   * @return The number of dimensions.
   */
  std::size_t dim() const noexcept {
    return static_cast<std::size_t>(theta_.size());
  }

  /**
   * @brief Return the natural logarithm of the step size.
   *
   * @return The log of the step size.
   */
  double log_step_size() const noexcept { return std::log(step_size()); }

  /**
   * @brief Return the natural logarithm of the diagonal of the
   * diagonal mass matirx.
   *
   * @return The log of the diagonal of the mass matrix.
   */
  Eigen::VectorXd log_mass() const {
    // equiv. inv_mass().array().inverse().log().matrix();
    return -inv_mass().array().log().matrix();
  }

  /**
   * @brief Return the current iteration.
   *
   * @return The iteration.
   */
  std::size_t iter() const noexcept { return iteration_; }

 private:
  /** The warmup configuration. */
  std::reference_wrapper<const WarmupConfig> warmup_cfg_;

  /** The Walnuts sampler configuration. */
  std::reference_wrapper<const SamplingConfig> sampling_cfg_;

  /** The random number generator required for Nuts. */
  detail::Random<RNG> rand_;

  /** The adaptation and sampling event handler. */
  std::reference_wrapper<H> handler_;

  /** The target log density/gradient function. */
  const detail::NoExceptLogpGrad<F, H> logp_grad_;

  /** The current state. */
  Eigen::VectorXd theta_;

  /** Cached endpoint gradient at `theta_` from the last transition (W-23). */
  Eigen::VectorXd cached_grad_;

  /** Cached endpoint log density at `theta_` from the last transition. */
  double cached_logp_ = -std::numeric_limits<double>::infinity();

  /**
   * @brief Mass used by the most recent warmup transition (MASS convention,
   * like InitChainConfig::mass(); inv_mass() inverts it).
   *
   * operator() recomputes the estimate AFTER observing (the estimator window
   * advances one draw between the last transition and a sampler() call), so
   * recomputing at freeze time can flip the auto-screen decision and freeze
   * a different metric than the chain's final transitions used. Frozen
   * samplers must carry the metric they were tuned with, hence the memo.
   */
  Eigen::VectorXd last_mass_;

  /** The current iteration. */
  std::size_t iteration_;

  /** The optimizer for step size adaptation.
   */
  Opt opt_;

  /** The estimator for the mass matrix. */
  detail::MassEstimator mass_estimator_;

  /** The estimator for the minimum number of micro steps per macro step. */
  detail::MinMicroStepsAdaptHandler min_micro_estimator_;

  /**
   * @brief Remember the adapter's step size when it is usable.
   *
   * Pure observation of `opt_.step_size()` after the iteration's adapter
   * updates; it changes no warmup arithmetic (W-41 gate: bit-identical
   * draws when the freeze is healthy).
   */
  void note_step_() noexcept {
    const double s = opt_.step_size();
    if (std::isfinite(s) && s > 0.0) {
      last_finite_step_ = s;
    }
  }

  /**
   * @brief Resolve a finite positive freeze step for a degenerate
   * `step_size()`, in the spirit of the init-robustness clamps.
   *
   * Fallback order: (a) `last_finite_step_` (the last finite adapter state
   * seen during warmup, seeded with the initial step — on the W-41
   * reproductions the adapter NaNs at iteration 0, so this is the initial
   * step), (b) a find_reasonable_step probe at the current position with
   * the current metric (consumes RNG draws and log density evaluations;
   * both are already owned by this adapter, and the previously-aborting
   * path carries no bit-identity contract), (c) the documented hard floor
   * 1000 * numeric_limits<double>::min(). Emits one stderr warning naming
   * the degenerate value, the fallback and its source.
   */
  double freeze_step_fallback(double degenerate) {
    const double floor_v = 1000.0 * std::numeric_limits<double>::min();
    const char* source = nullptr;
    double fallback = last_finite_step_;
    if (std::isfinite(fallback) && fallback > 0.0) {
      source = "last finite warmup step size";
    }
    if (source == nullptr) {
      try {
        fallback = detail::find_reasonable_step(
            rand_, logp_grad_.logp_grad_, theta_, inv_mass(), init_step_);
      } catch (...) {
        fallback = 0.0;
      }
      if (std::isfinite(fallback) && fallback > 0.0) {
        source = "find_reasonable_step heuristic";
      }
    }
    if (source == nullptr) {
      fallback = floor_v;
      source = "hard floor (1000 * DBL_MIN)";
    }
    std::cerr << "WALNUTS WARNING: freeze step size degenerate (step_size()="
              << degenerate << "); falling back to " << fallback << " ("
              << source << "); warmup iterations=" << iteration_ << std::endl;
    return fallback;
  }

  /** The initial step size the adapter was seeded with. */
  double init_step_;

  /** Last finite positive step size observed during warmup (W-41). */
  double last_finite_step_;

  /** Cached freeze fallback (W-41): computed once per adapter. */
  double freeze_fallback_step_ = 0.0;

  /** Whether the degenerate-freeze fallback has been computed (W-41). */
  bool freeze_clamped_ = false;
};

}  // namespace walnutpie
