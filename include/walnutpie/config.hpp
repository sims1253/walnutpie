#pragma once

#include <cstddef>
#include <ostream>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Dense>

#include "walnutpie/concepts.hpp"
#include "walnutpie/util.hpp"
#include "walnutpie/validate.hpp"

namespace walnutpie {

/**
 * @brief The initialization configuration for a single Markov chain.
 *
 * The initialization configuration specifies a step size, initial position,
 * and initial mass matrix.
 */
class InitChainConfig {
 public:
  /**
   * @brief Construct an initialization configuration.
   *
   * @param[in] step_size The initial step size.
   * @param[in] position The initial position.
   * @param[in] mass The initial mass matrix (diagonal).
   */
  InitChainConfig(double step_size, const Eigen::VectorXd& position,
                  const Eigen::VectorXd& mass)
      : step_size_(step_size), position_(position), mass_(mass) {}

  /**
   * @brief Construct an initialization configuration carrying the log
   * density and gradient evaluated at the initial position (W-42).
   *
   * The pair comes from the evaluation `InitConfigBuilder::masses()`
   * already performs at each chain's position (whose log density was
   * previously discarded). Callers use it to (a) refuse non-finite
   * initial log densities before warmup starts and (b) seed the first
   * transition's start-position cache so the duplicate re-evaluation is
   * skipped.
   *
   * @param[in] step_size The initial step size.
   * @param[in] position The initial position.
   * @param[in] mass The initial mass matrix (diagonal).
   * @param[in] init_grad The gradient of the log density at `position`.
   * @param[in] init_logp The log density at `position`.
   */
  InitChainConfig(double step_size, const Eigen::VectorXd& position,
                  const Eigen::VectorXd& mass, const Eigen::VectorXd& init_grad,
                  double init_logp)
      : step_size_(step_size),
        position_(position),
        mass_(mass),
        init_grad_(init_grad),
        init_logp_(init_logp),
        has_init_eval_(true) {}

  /**
   * @brief Return the initial step size.
   *
   * @return The step size.
   */
  double step_size() const noexcept { return step_size_; }

  /**
   * @brief Return the initial position.
   *
   * @return The position.
   */
  const Eigen::VectorXd& position() const noexcept { return position_; }

  /**
   * @brief Return the initial mass matrix.
   *
   * @return The mass matrix.
   */
  const Eigen::VectorXd& mass() const noexcept { return mass_; }

  /**
   * @brief Return whether a (log density, gradient) evaluation at the
   * initial position is recorded in this configuration (W-42).
   *
   * @return True if `init_logp()`/`init_grad()` are meaningful.
   */
  bool has_init_eval() const noexcept { return has_init_eval_; }

  /**
   * @brief Return the recorded log density at the initial position.
   *
   * Only meaningful when `has_init_eval()` is true.
   *
   * @return The log density.
   */
  double init_logp() const noexcept { return init_logp_; }

  /**
   * @brief Return the recorded gradient at the initial position.
   *
   * Only meaningful when `has_init_eval()` is true; empty otherwise.
   *
   * @return The gradient.
   */
  const Eigen::VectorXd& init_grad() const noexcept { return init_grad_; }

 private:
  double step_size_;
  Eigen::VectorXd position_;
  Eigen::VectorXd mass_;

  /** Gradient at the initial position from the mass-seeding eval (W-42). */
  Eigen::VectorXd init_grad_;

  /** Log density at the initial position from the mass-seeding eval. */
  double init_logp_ = 0.0;

  /** Whether the init evaluation above was recorded. */
  bool has_init_eval_ = false;
};

/**
 * @brief The initialization configuration for multiple Markov chains.
 *
 * Rather than a public constructor, it is built using an
 * `InitConfigBuilder` instance.
 *
 * The initialization configuration specifies a step size, initial
 * position, and initial mass matrix.
 */
class InitConfig {
 public:
  /**
   * @brief Return the number of chains.
   *
   * @return The number of chains.
   */
  std::size_t num_chains() const noexcept { return step_sizes_.size(); }

  /**
   * @brief Return the dimensionality of the positions.
   *
   * If the initialization is empty, 0 is returned.
   *
   * @return The dimensionality.
   */
  std::size_t dims() const noexcept {
    return positions_.empty()
               ? 0u
               : static_cast<std::size_t>(positions_.front().size());
  }

  /**
   * @brief Return the initial step sizes.
   *
   * @return The step sizes.
   */
  const std::vector<double>& step_sizes() const noexcept { return step_sizes_; }

  /**
   * @brief Return the initial step size for the specified chain.
   *
   * @param[in] n The chain identifier.
   * @return The step size.
   */
  double step_size(std::size_t n) const noexcept { return step_sizes_[n]; }

  /**
   * @brief Return the initial positions for all chains.
   *
   * @return The positions.
   */
  const std::vector<Eigen::VectorXd>& positions() const noexcept {
    return positions_;
  }

  /**
   * @brief Return the initial position for the specified chain.
   *
   * @param[in] n The chain index.
   * @return The positions.
   */
  const Eigen::VectorXd& position(std::size_t n) const noexcept {
    return positions_[n];
  }

  /**
   * @brief Return the initial diagonal mass matrices for all chains.
   *
   * @return The mass matrix diagonals.
   */
  const std::vector<Eigen::VectorXd>& masses() const noexcept {
    return masses_;
  }

  /**
   * @brief Return the mass matrix for the specified chain.
   *
   * @param[in] n The chain index.
   * @return The mass matrix.
   */
  const Eigen::VectorXd& mass(std::size_t n) const noexcept {
    return masses_[n];
  }

  /**
   * @brief Return the log densities at the initial positions (W-42).
   *
   * One entry per chain when the configuration was built through
   * `InitConfigBuilder::masses(logp_grad, ...)` (which evaluates each
   * chain's position); empty otherwise (no evaluation exists to
   * report). Callers use this to refuse starting a chain at a
   * non-finite log density.
   *
   * @return The initial log densities (empty if none were recorded).
   */
  const std::vector<double>& init_logps() const noexcept {
    return init_logps_;
  }

  /**
   * @brief Return the initialization configuration for the specified chain.
   *
   * @param[in] n The chain index.
   * @return The indexed chain's initialization configuration.
   */
  InitChainConfig init_chain_config(std::size_t n) const {
    if (n < init_logps_.size() && n < init_grads_.size()) {
      return InitChainConfig(step_size(n), position(n), mass(n),
                             init_grads_[n], init_logps_[n]);
    }
    return InitChainConfig(step_size(n), position(n), mass(n));
  }

 private:
  friend class InitConfigBuilder;

  /**
   * @brief Construct an initialization configuration.
   *
   * This constructor does not validate arguments because it is only
   * called internally. It only implements rvalue moves because that
   * is the only way it is called.
   *
   * @param[in] step_sizes The step sizes.
   * @param[in] positions The positions.
   * @param[in] masses The diagonals of the diagonal mass matrixes.
   * @param[in] init_logps Log densities at the positions (W-42; may be
   * empty when no mass-seeding evaluation was performed).
   * @param[in] init_grads Gradients at the positions (W-42; same
   * emptiness rule as `init_logps`).
   */
  InitConfig(std::vector<double>&& step_sizes,
             std::vector<Eigen::VectorXd>&& positions,
             std::vector<Eigen::VectorXd>&& masses,
             std::vector<double>&& init_logps = {},
             std::vector<Eigen::VectorXd>&& init_grads = {})
      : step_sizes_(std::move(step_sizes)),
        positions_(std::move(positions)),
        masses_(std::move(masses)),
        init_logps_(std::move(init_logps)),
        init_grads_(std::move(init_grads)) {}

  InitConfig() = default;

  std::vector<double> step_sizes_;
  std::vector<Eigen::VectorXd> positions_;
  std::vector<Eigen::VectorXd> masses_;

  /** Log densities at the initial positions (empty if not evaluated). */
  std::vector<double> init_logps_;

  /** Gradients at the initial positions (empty if not evaluated). */
  std::vector<Eigen::VectorXd> init_grads_;
};

/**
 * @brief The builder for initialization configurations.
 *
 * The usage to return an `InitConfig` is
 * `InitConfigBuilder(4, 20).step_sizes(0.5).build();`
 * with any number of config methods
 * chained between the construction and call to build.
 */
class InitConfigBuilder {
 public:
  /**
   * @brief Construct an initialization builder of the given sizes.
   *
   * @param[in] num_chains The number of Markov chains.
   * @param[in] dims The dimensionality of each chain.
   */
  InitConfigBuilder(std::size_t num_chains, std::size_t dims)
      : num_chains_(num_chains),
        dims_(dims),
        step_sizes_(std::vector<double>(num_chains, 0.1)),
        positions_(std::vector<Eigen::VectorXd>(
            num_chains,
            Eigen::VectorXd::Zero(static_cast<Eigen::Index>(dims)))),
        masses_(std::vector<Eigen::VectorXd>(
            num_chains,
            Eigen::VectorXd::Ones(static_cast<Eigen::Index>(dims)))) {}

  /**
   * @brief Set the step sizes to all be the specified value.
   *
   * @param[in] v The step size.
   * @return A reference to this builder for chaining.
   * @throw std::invalid_argument If the step size is not finite and positive.
   */
  InitConfigBuilder& step_sizes(double v) {
    detail::validate_finite_positive(v, "step size");
    step_sizes_ = std::vector<double>(num_chains_, v);
    return *this;
  }

  /**
   * @brief Set the step sizes to all be the specified values.
   *
   * @param[in] v The step sizes.
   * @return A reference to this builder for chaining.
   * @throw std::invalid_argument If any of the step sizes are not finite
   * positive.
   * @throw std::invalid_argument If the number of chains doesn't match the
   * number specified in the constructor.
   */
  InitConfigBuilder& step_sizes(const std::vector<double>& v) {
    detail::validate_size(v, num_chains_, "step_sizes", "num_chains");
    detail::validate_finite_positive(v, "step_size");
    step_sizes_ = v;
    return *this;
  }

  /**
   * @brief Randomly initialization the positions.
   *
   * Initialization is independent in each dimension with values drawn
   * from a zero-centered normal distribution with the specified
   * scale.
   *
   * @tparam RNG The type of the base random number generator.
   * @param[in,out] rng The base random number generator.
   * @param[in] init_scale The scale of the normal initial values.
   * @return A reference to this builder for chaining.
   * @throw std::invalid_argument If the initial scale is not finite and
   * positive.
   */
  template <std::uniform_random_bit_generator RNG>
  InitConfigBuilder& positions(RNG& rng, double init_scale) {
    detail::validate_finite_positive(init_scale, "init_scale");
    invalidate_init_evals_();
    detail::Random<RNG> rand(rng);
    positions_.resize(num_chains_);
    for (std::size_t c = 0; c < num_chains_; ++c) {
      rand.standard_normal(static_cast<Eigen::Index>(dims_), positions_[c]);
      positions_[c] *= init_scale;
    }
    return *this;
  }

  /**
   * @brief Initialize the positions all to the same value.
   *
   * @param[in] v The initial position.
   * @return A reference to this builder for chaining.
   * @throw std::invalid_argument If the dimensionality doesn't match
   * that specified during construction.
   * @throw std::invalid_argument If any of the initial positions contains
   * non-finite values.
   */
  InitConfigBuilder& positions(const Eigen::VectorXd& v) {
    detail::validate_size(v, dims_, "position", "dims");
    detail::validate_finite(v, "position");
    invalidate_init_evals_();
    positions_ = std::vector<Eigen::VectorXd>(num_chains_, v);
    return *this;
  }

  /**
   * @brief Initialize the positions to the specified values.
   *
   * @param[in] vs The initial positions.
   * @return A reference to this builder for chaining.
   * @throw std::invalid_argument If the number of initial positions doesn't
   * match the number of chains specified in the constructor.
   * @throw std::invalid_argument If any of the initial positions contains
   * non-finite values.
   * @throw std::invalid_argumet If any of the initial positions has a
   * dimensionality that does not match the dimensionality specified in the
   * constructor.
   */
  InitConfigBuilder& positions(const std::vector<Eigen::VectorXd>& vs) {
    detail::validate_size(vs, num_chains_, "positions", "num_chains");
    detail::validate_finite(vs, "positions");
    for (const auto& v : vs) {
      detail::validate_size(v, dims_, "position", "dims");
    }
    invalidate_init_evals_();
    positions_ = vs;
    return *this;
  }

  /**
   * @brief Initialize the positions to the specified values via move.
   *
   * @param[in] vs The initial positions.
   * @return A reference to this builder for chaining.
   * @throw std::invalid_argument If the number of initial positions doesn't
   * match the number of chains specified in the constructor.
   * @throw std::invalid_argument If any of the initial positions contains
   * non-finite values.
   * @throw std::invalid_argumet If any of the initial positions has a
   * dimensionality that does not match the dimensionality specified in the
   * constructor.
   */
  InitConfigBuilder& positions(std::vector<Eigen::VectorXd>&& vs) {
    detail::validate_size(vs, num_chains_, "positions", "num_chains");
    detail::validate_finite(vs, "positions");
    for (const auto& v : vs) {
      detail::validate_size(v, dims_, "position", "dims");
    }
    invalidate_init_evals_();
    positions_ = std::move(vs);
    return *this;
  }

  /**
   * @brief Initialize the masses using the Nutpie outer product strategy.
   *
   * Following Nutpie, the initialization uses a smoothed negative
   * outer product of gradient, which is the absolute value of the
   * outer product of gradients linearly interpolated with a unit
   * matrix with weight `mass_smoothing` on the unit matrix and `1 -
   * mass_smoothing` on the regularized outer product.
   *
   * If the flag `average_masses` is `true`, then each chain's mass
   * matrix is set to the geometric average of the per-chain mass
   * matrixes.
   *
   * See: Seyboldt, Adrian and Carlson, Eliot and Carpenter,
   * Bob. 2026. [Preconditioning Hamiltonian Monte Carlo by
   * minimizing Fisher
   * divergence](https://arxiv.org/abs/2603.18845v1). arXiv
   * 2603.18845.
   *
   * @tparam LPG The type of the log density and gradient function.
   * @param[in] logp_grad The log density and gradient function, called back.
   * @param[in] mass_smoothing The additive smoothing for mass matrices.
   * @param[in] average_masses Set to `true` to geometrically average mass
   * matrices.
   * @throw std::invalid_argumet If the mass smoothing is not in (0, 1).
   * @return A reference to this builder for chaining.
   */
  template <LogpGrad F>
  InitConfigBuilder& masses(const F& logp_grad, double mass_smoothing,
                            bool average_masses = false,
                            double clamp = 0.0) {
    detail::validate_probability(mass_smoothing, "mass_smoothing");
    Eigen::VectorXd grad;
    masses_.resize(num_chains_);
    init_logps_.resize(num_chains_);
    init_grads_.resize(num_chains_);
    for (std::size_t c = 0; c < num_chains_; ++c) {
      double lp;
      logp_grad(positions_[c], lp, grad);
      // W-42: record the mass-seeding evaluation (the log density was
      // previously discarded) — callers use it to refuse non-finite
      // initial log densities before warmup and to seed the first
      // transition's start-position cache.
      init_logps_[c] = lp;
      init_grads_[c] = grad;
      masses_[c] = (1 - mass_smoothing) * grad.array().abs() + mass_smoothing;
      if (clamp > 0) {
        // Guard against far-from-typical-set initializations where the
        // gradient magnitude (and hence the seeded mass) is enormous; a
        // degenerate seed throttles all early movement (inv_mass ~ 1/|grad|).
        masses_[c] = masses_[c].cwiseMin(clamp).cwiseMax(1.0 / clamp);
      }
    }
    if (average_masses) {
      Eigen::Index D = masses_[0].size();
      Eigen::VectorXd sum_log_mass = Eigen::VectorXd::Zero(D);
      for (const auto& mass : masses_) {
        sum_log_mass += mass.array().log().matrix();
      }
      auto avg_log_mass = sum_log_mass / num_chains_;
      auto geom_mean_mass = avg_log_mass.array().exp().matrix();
      masses_ = std::vector<Eigen::VectorXd>(num_chains_, geom_mean_mass);
    }
    return *this;
  }

  /**
   * @brief Initialize the mass matrices all to the same value.
   *
   * @param[in] v The initial diagonal mass matrix.
   * @return A reference to this builder for chaining.
   * @throw std::invalid_argument If the dimensionality doesn't match
   * that specified during construction.
   * @throw std::invalid_argument If any of the initial mass matrix
   * diagonals contains non-finite or non-positive values.
   */
  InitConfigBuilder& masses(const Eigen::VectorXd& v) {
    detail::validate_size(v, dims_, "masses", "dims");
    detail::validate_finite_positive(v, "masses");
    masses_ = std::vector<Eigen::VectorXd>(num_chains_, v);
    return *this;
  }

  /**
   * @brief Initialize the mass matrices to the specified values.
   *
   * @param[in] vs The initial mass matrices.
   * @return A reference to this builder for chaining.
   * @throw std::invalid_argument If the number of initial mass
   * matrices doesn't match the number of chains specified in the
   * constructor.
   * @throw std::invalid_argument If any of the initial mass matrices
   * contains non-finite values.
   * @throw std::invalid_argumet If any of the initial mass matrices
   * has a dimensionality that does not match the dimensionality
   * specified in the constructor.
   */
  InitConfigBuilder& masses(const std::vector<Eigen::VectorXd>& vs) {
    detail::validate_size(vs, num_chains_, "masses", "num_chains");
    detail::validate_finite_positive(vs, "masses");
    for (const auto& v : vs) {
      detail::validate_size(v, dims_, "all masses", "dims");
    }
    masses_ = vs;
    return *this;
  }

  /**
   * @brief Initialize the mass matrices to the specified values via
   * move.
   *
   * @param[in] vs The initial mass matrices.
   * @return A reference to this builder for chaining.
   * @throw std::invalid_argument If the number of initial mass
   * matrices doesn't match the number of chains specified in the
   * constructor.
   * @throw std::invalid_argument If any of the initial mass matrices
   * contains non-finite values.
   * @throw std::invalid_argumet If any of the initial mass matrices
   * has a dimensionality that does not match the dimensionality
   * specified in the constructor.
   */
  InitConfigBuilder& masses(std::vector<Eigen::VectorXd>&& vs) {
    detail::validate_size(vs, num_chains_, "masses", "num_chains");
    detail::validate_finite_positive(vs, "masses");
    for (const auto& v : vs) {
      detail::validate_size(v, dims_, "all masses", "dims");
    }
    masses_ = std::move(vs);
    return *this;
  }

  /**
   * @brief Return the initialization configuration.
   *
   * @return The initialization configuration.
   */
  InitConfig build() {
    return InitConfig{std::move(step_sizes_), std::move(positions_),
                      std::move(masses_), std::move(init_logps_),
                      std::move(init_grads_)};
  }

  /**
   * @brief Heuristically adapt the initial step sizes, then return
   * the initialization configuration.
   *
   * @tparam RNG Type of the base random number generator.
   * @tparam F Type of the log density and gradient function.
   * @param[in] rng The base random number generator.
   * @param[in] logp_grad The log density and gradient function.
   */
  template <std::uniform_random_bit_generator RNG, LogpGrad F>
  InitConfig adapt_step_build(RNG& rng, const F& logp_grad) {
    for (std::size_t c = 0; c < num_chains_; ++c) {
      step_sizes_[c] = detail::adapt_step(rng, logp_grad, positions_[c],
                                          masses_[c], step_sizes_[c], dims_);
    }
    return build();
  }

 private:
  /**
   * @brief Drop any recorded init evaluations (W-42 hygiene).
   *
   * Called whenever the positions change after a `masses(logp_grad,
   * ...)` call: the recorded (logp, grad) pairs belong to the positions
   * they were evaluated at and must not survive a position swap.
   */
  void invalidate_init_evals_() {
    init_logps_.clear();
    init_grads_.clear();
  }

  std::size_t num_chains_;
  std::size_t dims_;
  std::vector<double> step_sizes_;
  std::vector<Eigen::VectorXd> positions_;
  std::vector<Eigen::VectorXd> masses_;

  /** Log densities at the positions from the mass-seeding eval (W-42). */
  std::vector<double> init_logps_;

  /** Gradients at the positions from the mass-seeding eval (W-42). */
  std::vector<Eigen::VectorXd> init_grads_;
};

/**
 * @internal @brief Write a dump of the initial configurations to the
 * specified stream.
 *
 * @param[in,out] out Stream to which configuration is written.
 * @param[in] cfg The configuration to write.
 * @return A reference to the output stream for chaining.
 */
inline std::ostream& operator<<(std::ostream& out, const InitConfig& cfg) {
  out << "InitConfigs (by chain)\n";
  for (std::size_t n = 0; n < cfg.step_sizes().size(); ++n) {
    if (n > 0) {
      out << "\n";
    }
    out << "  chain         = " << n << "\n"
        << "    num_chains  = " << cfg.num_chains() << "\n"
        << "    step_size   = " << cfg.step_sizes()[n] << "\n"
        << "    position    = " << cfg.positions()[n].transpose() << "\n"
        << "    mass        = " << cfg.masses()[n].transpose() << "\n";
  }
  return out;
}

/**
 * @brief The warmup configuration object. The object supplies methods
 * for all of the tuning parameters for warmup.
 */
class WarmupConfig {
 public:
  /**
   * @brief Return the minimum number of warmup iterations.
   *
   * @return Minimum warmup iterations.
   */
  std::size_t min_iter() const { return min_iter_; }

  /**
   * @brief Return the maximum number of warmup iterations.
   *
   * @return Maximum warmup iterations.
   */
  std::size_t max_iter() const { return max_iter_; }

  /**
   * @brief Return a copy of this configuration with a different iteration
   * range (W-28 pilot-gate resume: resumed warmup phases run against a
   * shorter per-phase budget while the total budget is enforced by the
   * caller).
   *
   * @param[in] min_iter The new minimum warmup iterations.
   * @param[in] max_iter The new maximum warmup iterations.
   * @throw std::invalid_argument If `min_iter > max_iter`.
   * @return The configuration copy with the new iteration range.
   */
  WarmupConfig with_min_max_iter(std::size_t min_iter,
                                 std::size_t max_iter) const {
    if (min_iter > max_iter) {
      throw std::invalid_argument(
          "min_iter cannot be greater than max_iter");
    }
    WarmupConfig out = *this;
    out.min_iter_ = min_iter;
    out.max_iter_ = max_iter;
    return out;
  }

  /**
   * @brief Return the step-size convergence tolerance.
   *
   * @return The step-size convergence tolerance.
   */
  double step_size_converge_tol() const { return step_size_converge_tol_; }

  /**
   * @brief Return the mass matrix convergence tolerance. The
   * tolerance is for the L2-norm of the diagonal.
   *
   * @return The mass matrix  convergence tolerance.
   */
  double mass_converge_tol() const { return mass_converge_tol_; }

  /**
   * @brief Return the initial count for the mass matrix estimator.
   *
   * @return The initial count for the mass matrix estimator.
   */
  double mass_init_count() const { return mass_init_count_; }

  /**
   * @brief Return the additive smoothing for the mass matrix estimator.
   *
   * @return The additive smoothing for the mass matrix estimator.
   */
  double mass_additive_smoothing() const { return mass_additive_smoothing_; }

  /**
   * @brief Return the target number of macro steps.
   *
   * @return The target number of macro steps.
   */
  double max_macro_steps_target() const { return max_macro_steps_target_; }

  /**
   * @brief Return the target acceptance rate.
   *
   * @return The target acceptance rate.
   */
  double step_accept_rate_target() const { return step_accept_rate_target_; }

  /**
   * @brief Return the step-size learning rate for the Adam optimizer.
   *
   * @return The step-size learning rate for the Adam optimizer.
   */
  double step_learning_rate() const { return step_learning_rate_; }

  /**
   * @brief Return the gradient decay rate for the Adam optimizer.
   *
   * @return The gradient decay rate for the Adam optimizer.
   */
  double step_gradient_decay() const { return step_gradient_decay_; }

  /**
   * @brief Return the squared gradient decay rate for the Adam optimizer.
   *
   * @return The squared gradient decay rate for the Adam optimizer.
   */
  double step_sq_gradient_decay() const { return step_sq_gradient_decay_; }

  /**
   * @brief Return the step-size stabilization.
   *
   * @return The step-size stabilization.
   */
  double step_stabilization() const { return step_stabilization_; }

  /**
   * @brief Return the step-size learning rate decay factor.
   *
   * @return The step-size learning rate decay factor.
   */
  double step_learn_rate_decay() const { return step_learn_rate_decay_; }

  double da_gamma() const { return da_gamma_; }

  double da_t0() const { return da_t0_; }

  double da_kappa() const { return da_kappa_; }

  std::size_t slow_ema_warmup() const { return slow_ema_warmup_; }

  std::size_t step_opt_batch_stride() const { return step_opt_batch_stride_; }

  double step_grad_clip() const { return step_grad_clip_; }

  bool da_freeze_average() const { return da_freeze_average_; }

  double mass_shrink_kappa() const { return mass_shrink_kappa_; }

  double mass_var_floor() const { return mass_var_floor_; }

  bool metric_drift_guard() const { return metric_drift_guard_; }

  double mass_combine_power() const { return mass_combine_power_; }

  double metric_collapse_reset() const { return metric_collapse_reset_; }

  double metric_stall_reset() const { return metric_stall_reset_; }

  std::size_t metric_stall_window() const { return metric_stall_window_; }

  double mass_init_clamp() const { return mass_init_clamp_; }

  /**
   * @brief Return the init-buffer length for mass adaptation (W-54 arm A).
   *
   * When positive, the first N warmup iterations run with the IDENTITY
   * inverse mass and do not feed the mass estimator at all (no
   * observe(), no window chopping, no low-rank refresh); continuous
   * adaptation begins at iteration N from identity-seeded accumulators
   * (no metric discontinuity at the boundary). This is the Stan-style
   * init buffer: tail geometry at a distant initialization cannot
   * contaminate the metric while the chain drifts toward the typical
   * set. 0 (the DEFAULT) = current behavior (continuous adaptation
   * from iteration 0, gradient-seeded).
   *
   * @return The number of leading warmup iterations to buffer (0 = off).
   */
  std::size_t mass_init_buffer() const { return mass_init_buffer_; }

  /**
   * @brief Return the soft gradient-clipping scale for the adapter's
   * score stream (W-54 arm B).
   *
   * When positive, the gradient fed to the mass estimator's score
   * moments during warmup iterations below `grad_clip_iters()` is
   * replaced elementwise by the soft clip g' = c * asinh(g / c)
   * (identity below ~c/100, logarithmic beyond; smooth and
   * sign-preserving). This bounds the score-variance scale tail
   * gradients can imprint on the metric WITHOUT touching the
   * trajectory integrator's gradient (which would change the
   * Hamiltonian being sampled — a different target). The step adapter
   * consumes only the scalar acceptance statistic and is unaffected
   * by construction. 0.0 (the DEFAULT) = current behavior (raw
   * scores).
   *
   * @return The clipping scale c in gradient units (0 = off).
   */
  double grad_clip_scale() const { return grad_clip_scale_; }

  /**
   * @brief Return the number of leading warmup iterations during which
   * the soft gradient clip is applied (W-54 arm B; only meaningful
   * when `grad_clip_scale()` is positive).
   *
   * @return The clipping window length in warmup iterations.
   */
  std::size_t grad_clip_iters() const { return grad_clip_iters_; }

  /**
   * @brief Basis-extraction rule for the low-rank metric factors.
   *
   * 0 = windowed thin SVD of the standardized stacked draw/score matrix
   *     (default; Algorithm 1 of the low-rank Fisher metric).
   * 1 = streaming orthogonal (power) iteration with a persistent basis.
   * 2 = Muon-style Newton-Schulz polar orthogonalization of the stacked
   *     matrix (column-selection by pre-orthonormalization leverage).
   * 3 = as 2 but with row RMS equilibration before orthogonalization
   *     (MuonEq-style), the diagonal-adjacent ablation.
   */
  std::size_t metric_basis() const { return metric_basis_; }

  std::size_t anti_windup_pass_rate() const { return anti_windup_pass_rate_; }

  std::size_t drift_iters() const { return drift_iters_; }

  std::size_t metric_window() const { return metric_window_; }

  std::size_t metric_rank() const { return metric_rank_; }

  bool metric_full() const { return metric_full_; }

  double metric_auto() const { return metric_auto_; }

  double max_error_start() const { return max_error_start_; }

  std::size_t max_error_schedule_iters() const {
    return max_error_schedule_iters_;
  }

  /**
   * @brief Return the temporal step-size drift tolerance for early exit.
   *
   * When positive, the multi-chain controller additionally requires every
   * chain's step size to be temporally stable — relative drift below this
   * tolerance across the last full `temporal_window()` iterations ending
   * at or after `temporal_min_iter()` — before stopping warmup early on
   * the cross-chain criteria. This guards against the failure mode where
   * all chains agree with each other while their step sizes are still
   * marching toward equilibrium (W-22: +170% late-warmup step growth with
   * stable mass degraded post-warmup quality on the marginal model class).
   *
   * @return The temporal step drift tolerance (0 = gate off).
   */
  double temporal_step_drift_tol() const { return temporal_step_drift_tol_; }

  /**
   * @brief Return the window length for the temporal step-drift gate.
   *
   * @return The temporal window in warmup iterations.
   */
  std::size_t temporal_window() const { return temporal_window_; }

  /**
   * @brief Return the minimum iteration for the temporal step-drift gate.
   *
   * @return The minimum warmup iterations before temporal early exit.
   */
  std::size_t temporal_min_iter() const { return temporal_min_iter_; }

  /**
   * @brief Return whether the multi-chain controller may stop warmup
   * before the budget (W-31).
   *
   * When false (the DEFAULT), the controller never stops warmup early:
   * the only stop is the `max_iter` budget, and `AdaptResult` reports
   * `exit_iter == max_iter`, `early_exit == false`. This is the safe
   * default: the cross-chain criteria with their default tolerances
   * (mass 1.0 / step 0.1, temporal gate off) can hold at iteration
   * 50-80 with good inits while warmup would still materially improve
   * the frozen sampler (measured: hier_2pl bulk-ESS-min 519 -> 61, and
   * even the temporal 2-window gate at tol 0.05 degraded it 519 -> 126;
   * no cheap tolerance-based gate preserved quality in the W-25/W-28
   * grids). Embedders who want the controller's convergence-based early
   * exit must opt in explicitly via `WarmupConfigBuilder::allow_early_exit`,
   * taking responsibility for the tolerances they set.
   *
   * @return Whether convergence-based early exit is enabled.
   */
  bool allow_early_exit() const { return allow_early_exit_; }

  /**
   * @brief Return the stride for publishing updates for convergence monitoring.
   *
   * @return The stride for publishing updates.
   */
  std::size_t publish_stride() const { return publish_stride_; }

  /**
   * @brief Return the period at which threads for chains yield.
   *
   * @return The the period at which threads for chains yield.
   */
  std::size_t yield_period() const { return yield_period_; }

 private:
  friend class WarmupConfigBuilder;

  WarmupConfig() = default;

  std::size_t min_iter_ = 50;
  std::size_t max_iter_ = 1000;
  double step_size_converge_tol_ = 0.1;
  double mass_converge_tol_ = 1.0;
  double mass_init_count_ = 4.0;
  double mass_additive_smoothing_ = 1e-5;
  double max_macro_steps_target_ = 15.0;
  double step_accept_rate_target_ = 0.8;
  double step_learning_rate_ = 0.05;
  double step_gradient_decay_ = 0.8;
  double step_sq_gradient_decay_ = 0.9;
  double step_stabilization_ = 1e-4;
  double step_learn_rate_decay_ = 0.5;
  double da_gamma_ = 0.05;
  double da_t0_ = 10.0;
  double da_kappa_ = 0.75;
  std::size_t slow_ema_warmup_ = 100;
  std::size_t step_opt_batch_stride_ = 1;
  double step_grad_clip_ = 0.0;
  bool da_freeze_average_ = false;
  double mass_shrink_kappa_ = 0.0;
  double mass_var_floor_ = 0.0;
  bool metric_drift_guard_ = false;
  double mass_combine_power_ = 0.0;
  double metric_collapse_reset_ = 0.0;
  double metric_stall_reset_ = 0.0;
  std::size_t metric_stall_window_ = 100;
  double mass_init_clamp_ = 0.0;
  std::size_t mass_init_buffer_ = 0;  // W-54 arm A: 0 = off
  double grad_clip_scale_ = 0.0;      // W-54 arm B: 0 = off
  std::size_t grad_clip_iters_ = 200;
  std::size_t metric_basis_ = 0;  // 0=svd 1=power 2=muon 3=muoneq
  std::size_t anti_windup_pass_rate_ = 0;
  std::size_t drift_iters_ = 0;
  std::size_t metric_window_ = 0;
  std::size_t metric_rank_ = 0;
  bool metric_full_ = false;
  double metric_auto_ = 0.0;
  double max_error_start_ = 0.0;
  std::size_t max_error_schedule_iters_ = 0;
  std::size_t publish_stride_ = 5;
  std::size_t yield_period_ = 32;
  double temporal_step_drift_tol_ = 0.0;  // 0 = temporal gate off
  std::size_t temporal_window_ = 50;
  std::size_t temporal_min_iter_ = 200;
  bool allow_early_exit_ = false;  // W-31: early exit is opt-in
};

/**
 * @brief The builder for `WarmupConfig` objects.
 */
class WarmupConfigBuilder {
 public:
  /**
   * @brief Set the minimum and maximum number of warmup iterations.
   *
   * @param[in] min_iter The minimum number of warmup iterations.
   * @param[in] max_iter The maximum number of warmup iterations.
   * @return This builder for chaining.
   * @throw std::invalid_argument If `min_iter` > `max_iter`.
   */
  WarmupConfigBuilder& min_max_iter(std::size_t min_iter,
                                    std::size_t max_iter) {
    if (min_iter > max_iter) {
      throw std::invalid_argument(
          "min_iter cannot be greater than than max_iter");
    }
    cfg_.min_iter_ = min_iter;
    cfg_.max_iter_ = max_iter;
    return *this;
  }

  /**
   * @brief Set the step size convergence tolerance.
   *
   * @param[in] v The step size convergence tolerance.
   * @return This builder for chaining.
   * @throw std::invalid_argument If the tolerance is not finite and positive.
   */
  WarmupConfigBuilder& step_size_converge_tol(double v) {
    detail::validate_finite_positive(v, "step_size_converge_tol");
    cfg_.step_size_converge_tol_ = v;
    return *this;
  }

  /**
   * @brief Set the mass matrix L2-norm convergence tolerance.
   *
   * @param[in] v The mass matrix L2-norm convergence tolerance.
   * @return This builder for chaining.
   * @throw std::invalid_argument If the tolerance is not finite and positive.
   */
  WarmupConfigBuilder& mass_converge_tol(double v) {
    detail::validate_finite_positive(v, "mass_converge_tol");
    cfg_.mass_converge_tol_ = v;
    return *this;
  }

  /**
   * @brief Set the mass matrix estimator initial count.
   *
   * @param[in] v The mass matrix estimator initial count.
   * @return This builder for chaining.
   * @throw std::invalid_argument If the initial count is not finite and
   * positive.
   */
  WarmupConfigBuilder& mass_init_count(double v) {
    detail::validate_finite_positive(v, "mass_init_count");
    cfg_.mass_init_count_ = v;
    return *this;
  }

  /**
   * @brief Set the mass matrix estimator additive smoothing.
   *
   * @param[in] v The mass matrix estimator additive smoothing.
   * @return This builder for chaining.
   * @throw std::invalid_argument If the smoothing is not finite and positive.
   */
  WarmupConfigBuilder& mass_additive_smoothing(double v) {
    detail::validate_finite_positive(v, "mass_additive_smoothing");
    cfg_.mass_additive_smoothing_ = v;
    return *this;
  }

  /**
   * @brief Set the target number of macro steps.
   *
   * @param[in] v The target number of macro steps.
   * @return This builder for chaining.
   * @throw std::invalid_argument If the target is not finite and positive.
   */
  WarmupConfigBuilder& max_macro_steps_target(double v) {
    detail::validate_finite_positive(v, "max_macro_steps_target");
    cfg_.max_macro_steps_target_ = v;
    return *this;
  }

  /**
   * @brief Set the accept-rate target for step-size estimation.
   *
   * @param[in] v The accept-rate target.
   * @return This builder for chaining.
   * @throw std::invalid_argument If the accept rate is not in (0, 1).
   */
  WarmupConfigBuilder& step_accept_rate_target(double v) {
    detail::validate_probability(v, "step_accept_rate_target");
    cfg_.step_accept_rate_target_ = v;
    return *this;
  }

  /**
   * @brief Set the step size learning rate.
   *
   * @param[in] v The step size learning rate.
   * @return This builder for chaining.
   * @throw std::invalid_argument If the learning rate is not finite and
   * positive.
   */
  WarmupConfigBuilder& step_learning_rate(double v) {
    detail::validate_finite_positive(v, "step_learning_rate");
    cfg_.step_learning_rate_ = v;
    return *this;
  }

  /**
   * @brief Set the gradient decay for the step-size estimator.
   *
   * @param[in] v The gradient decay for the step-size estimator.
   * @return This builder for chaining.
   * @throw std::invalid_argument If the decay is not in (0, 1).
   */
  WarmupConfigBuilder& step_gradient_decay(double v) {
    detail::validate_probability(v, "step_gradient_decay");
    cfg_.step_gradient_decay_ = v;
    return *this;
  }

  /**
   * @brief Set the squared gradient decay for the step-size estimator.
   *
   * @param[in] v The squared gradient decay for the step-size estimator.
   * @return This builder for chaining.
   * @throw std::invalid_argument If the decay is not in (0, 1).
   */
  WarmupConfigBuilder& step_sq_gradient_decay(double v) {
    detail::validate_probability(v, "step_sq_gradient_decay");
    cfg_.step_sq_gradient_decay_ = v;
    return *this;
  }

  /**
   * @brief Set the step-size estimator stabilization term.
   *
   * @param[in] v The step-size estimator stabilization term.
   * @return This builder for chaining.
   * @throw std::invalid_argument If the stabilization is not finite and
   * positive.
   */
  WarmupConfigBuilder& step_stabilization(double v) {
    detail::validate_finite_positive(v, "step_stabilization");
    cfg_.step_stabilization_ = v;
    return *this;
  }

  /**
   * @brief Set the learning rate decay exponent for step size.
   *
   * @param[in] v The learning rate decay exponent.
   * @return This builder for chaining.
   * @throw std::invalid_argument If the decay exponent is not in (0, 1).
   */
  WarmupConfigBuilder& da_gamma(double v) {
    detail::validate_finite_positive(v, "da_gamma");
    cfg_.da_gamma_ = v;
    return *this;
  }

  WarmupConfigBuilder& da_t0(double v) {
    detail::validate_finite_positive(v, "da_t0");
    cfg_.da_t0_ = v;
    return *this;
  }

  WarmupConfigBuilder& da_kappa(double v) {
    detail::validate_probability(v, "da_kappa");
    cfg_.da_kappa_ = v;
    return *this;
  }

  WarmupConfigBuilder& slow_ema_warmup(std::size_t v) {
    cfg_.slow_ema_warmup_ = v;
    return *this;
  }

  WarmupConfigBuilder& step_grad_clip(double v) {
    if (v < 0) {
      throw std::invalid_argument("step_grad_clip must be >= 0 (0 = off)");
    }
    cfg_.step_grad_clip_ = v;
    return *this;
  }

  WarmupConfigBuilder& mass_shrink_kappa(double v) {
    if (v < 0) {
      throw std::invalid_argument("mass_shrink_kappa must be >= 0 (0 = off)");
    }
    cfg_.mass_shrink_kappa_ = v;
    return *this;
  }

  WarmupConfigBuilder& metric_auto(double v) {
    if (v < 0 || v > 1) {
      throw std::invalid_argument(
          "metric_auto threshold must be in [0, 1] (0 = off)");
    }
    cfg_.metric_auto_ = v;
    return *this;
  }

  WarmupConfigBuilder& metric_full(bool v) {
    cfg_.metric_full_ = v;
    return *this;
  }

  WarmupConfigBuilder& metric_rank(std::size_t v) {
    cfg_.metric_rank_ = v;  // 0 = off
    return *this;
  }

  WarmupConfigBuilder& metric_window(std::size_t v) {
    cfg_.metric_window_ = v;  // 0 = off (exponential discounting only)
    return *this;
  }

  WarmupConfigBuilder& drift_iters(std::size_t v) {
    cfg_.drift_iters_ = v;
    return *this;
  }

  WarmupConfigBuilder& max_error_schedule(double start, std::size_t iters) {
    if (start < 0) {
      throw std::invalid_argument("max_error_schedule start must be >= 0");
    }
    cfg_.max_error_start_ = start;
    cfg_.max_error_schedule_iters_ = iters;
    return *this;
  }

  WarmupConfigBuilder& anti_windup_pass_rate(std::size_t v) {
    cfg_.anti_windup_pass_rate_ = v;  // 0 = off
    return *this;
  }

  WarmupConfigBuilder& metric_basis(std::size_t v) {
    if (v > 3) {
      throw std::invalid_argument("metric_basis must be 0..3");
    }
    cfg_.metric_basis_ = v;
    return *this;
  }

  WarmupConfigBuilder& mass_init_clamp(double v) {
    if (v < 0) {
      throw std::invalid_argument("mass_init_clamp must be >= 0");
    }
    cfg_.mass_init_clamp_ = v;
    return *this;
  }

  /**
   * Set the init-buffer length for mass adaptation (W-54 arm A).
   *
   * @param[in] v Number of leading warmup iterations to hold the mass
   * at identity and skip estimator feeding (0 = off, the default).
   * @return This builder for chaining.
   */
  WarmupConfigBuilder& mass_init_buffer(std::size_t v) {
    cfg_.mass_init_buffer_ = v;
    return *this;
  }

  /**
   * Set the soft gradient-clipping scale for the adapter's score
   * stream (W-54 arm B). The clip is g' = c * asinh(g / c), applied to
   * the mass estimator's score observations only.
   *
   * @param[in] v The clipping scale c in gradient units (0 = off, the
   * default; thread values 1e10 / 1e8).
   * @return This builder for chaining.
   */
  WarmupConfigBuilder& grad_clip_scale(double v) {
    if (!(v >= 0.0)) {
      throw std::invalid_argument("grad_clip_scale must be >= 0");
    }
    cfg_.grad_clip_scale_ = v;
    return *this;
  }

  /**
   * Set the warmup-only window length for the soft gradient clip
   * (W-54 arm B; default 200).
   *
   * @param[in] v The clipping window in warmup iterations.
   * @return This builder for chaining.
   */
  WarmupConfigBuilder& grad_clip_iters(std::size_t v) {
    cfg_.grad_clip_iters_ = v;
    return *this;
  }

  WarmupConfigBuilder& metric_stall_reset(double v,
                                          std::size_t window = 100) {
    if (v < 0) {
      throw std::invalid_argument(
          "metric_stall_reset must be >= 0 (0 = off; e.g. 1e-3)");
    }
    cfg_.metric_stall_reset_ = v;
    cfg_.metric_stall_window_ = window;
    return *this;
  }

  WarmupConfigBuilder& metric_collapse_reset(double v) {
    if (v < 0 || v >= 1) {
      throw std::invalid_argument(
          "metric_collapse_reset must be in [0, 1) (0 = off; e.g. 0.01)");
    }
    cfg_.metric_collapse_reset_ = v;
    return *this;
  }

  WarmupConfigBuilder& mass_combine_power(double v) {
    if (v < 0 || v > 1e3) {
      throw std::invalid_argument(
          "mass_combine_power must be in [0, 1000] (0 = geometric mean)");
    }
    cfg_.mass_combine_power_ = v;
    return *this;
  }

  WarmupConfigBuilder& metric_drift_guard(bool v) {
    cfg_.metric_drift_guard_ = v;
    return *this;
  }

  WarmupConfigBuilder& mass_var_floor(double v) {
    if (v < 0) {
      throw std::invalid_argument("mass_var_floor must be >= 0 (0 = off)");
    }
    cfg_.mass_var_floor_ = v;
    return *this;
  }

  WarmupConfigBuilder& da_freeze_average(bool v) {
    cfg_.da_freeze_average_ = v;
    return *this;
  }

  WarmupConfigBuilder& step_opt_batch_stride(std::size_t v) {
    if (v == 0) {
      throw std::invalid_argument("step_opt_batch_stride must be >= 1");
    }
    cfg_.step_opt_batch_stride_ = v;
    return *this;
  }

  WarmupConfigBuilder& step_learn_rate_decay(double v) {
    detail::validate_probability(v, "step_learn_rate_decay");
    cfg_.step_learn_rate_decay_ = v;
    return *this;
  }

  /**
   * @brief Set the stride between publishing statistics for convergence
   * monitoring.
   *
   * @param[in] v The stride for publishing statistics for convergence
   * monitoring.
   * @return This builder for chaining.
   * @throw std::invalid_argument If the stride is not positive.
   */
  WarmupConfigBuilder& publish_stride(std::size_t v) {
    detail::validate_positive(v, "publish_stride");
    cfg_.publish_stride_ = v;
    return *this;
  }

  /**
   * @brief Set the iteration period between chain threads yielding.
   *
   * @param[in] v The iteration period between chain threads yielding.
   * @return This builder for chaining.
   * @throw std::invalid_argument If the yield period is not positive.
   */
  WarmupConfigBuilder& yield_period(std::size_t v) {
    detail::validate_positive(v, "yield_period");
    cfg_.yield_period_ = v;
    return *this;
  }

  /**
   * @brief Set the temporal step-size drift tolerance for early exit.
   *
   * @param[in] v The tolerance (0 = gate off; e.g. 0.05).
   * @return This builder for chaining.
   * @throw std::invalid_argument If the tolerance is negative.
   */
  WarmupConfigBuilder& temporal_step_drift_tol(double v) {
    if (v < 0) {
      throw std::invalid_argument(
          "temporal_step_drift_tol must be >= 0 (0 = off)");
    }
    cfg_.temporal_step_drift_tol_ = v;
    return *this;
  }

  /**
   * @brief Set the window length for the temporal step-drift gate.
   *
   * @param[in] v The window in warmup iterations.
   * @return This builder for chaining.
   * @throw std::invalid_argument If the window is not positive.
   */
  WarmupConfigBuilder& temporal_window(std::size_t v) {
    detail::validate_positive(v, "temporal_window");
    cfg_.temporal_window_ = v;
    return *this;
  }

  /**
   * @brief Set the minimum iteration for the temporal step-drift gate.
   *
   * @param[in] v The minimum warmup iterations before temporal early exit.
   * @return This builder for chaining.
   */
  WarmupConfigBuilder& temporal_min_iter(std::size_t v) {
    cfg_.temporal_min_iter_ = v;
    return *this;
  }

  /**
   * @brief Allow the multi-chain controller to stop warmup before the
   * budget on its cross-chain criteria (W-31; opt-in).
   *
   * OFF by default: the default cross-chain tolerances (mass 1.0 /
   * step 0.1, temporal gate off) stop warmup at iteration 50-80 with
   * good initializations and destroy post-warmup quality on the
   * marginal model class, and no tolerance-based gate tested preserved
   * quality (W-25/W-28). With this flag false the controller runs
   * warmup to the full `max_iter` budget; the convergence criteria
   * (and the temporal gate) are only consulted — and may only stop
   * warmup — when it is true. See `WarmupConfig::allow_early_exit`.
   *
   * @param[in] v Whether convergence-based early exit is allowed.
   * @return This builder for chaining.
   */
  WarmupConfigBuilder& allow_early_exit(bool v) {
    cfg_.allow_early_exit_ = v;
    return *this;
  }

  /**
   * @brief Return the warmup configuration.
   *
   * @return The warmup configuration.
   */
  WarmupConfig build() { return cfg_; }

 private:
  WarmupConfig cfg_;
};

/**
 * @internal @brief Print the tuning parameters specified by the
 * warmup configuration to the output stream.
 *
 * @param[in,out] out Output stream to which configuration is printed.
 * @param[in] cfg The sampling configuration.
 * @return The output stream for chained calls.
 */
inline std::ostream& operator<<(std::ostream& out, const WarmupConfig& cfg) {
  out << "WarmupConfig\n"
      << "  min_iter                 = " << cfg.min_iter() << "\n"
      << "  max_iter                 = " << cfg.max_iter() << "\n"
      << "  step_size_converge_tol   = " << cfg.step_size_converge_tol() << "\n"
      << "  mass_converge_tol        = " << cfg.mass_converge_tol() << "\n"
      << "  mass_init_count          = " << cfg.mass_init_count() << "\n"
      << "  mass_additive_smoothing  = " << cfg.mass_additive_smoothing()
      << "\n"
      << "  max_macro_steps_target   = " << cfg.max_macro_steps_target() << "\n"
      << "  step_accept_rate_target  = " << cfg.step_accept_rate_target()
      << "\n"
      << "  step_learning_rate       = " << cfg.step_learning_rate() << "\n"
      << "  step_gradient_decay      = " << cfg.step_gradient_decay() << "\n"
      << "  step_sq_gradient_decay   = " << cfg.step_sq_gradient_decay() << "\n"
      << "  step_stabilization       = " << cfg.step_stabilization() << "\n"
      << "  step_learn_rate_decay    = " << cfg.step_learn_rate_decay() << "\n"
      << "  publish_stride           = " << cfg.publish_stride() << "\n"
      << "  yield_period             = " << cfg.yield_period() << "\n";
  return out;
}

/**
 * @brief A class to hold the configuration for the Walnuts sampler.
 */
class SamplingConfig {
 public:
  /**
   * @brief Return the minimum number of sampling iterations.
   *
   * @return The minimum number of sampling iterations.
   */
  std::size_t min_iter() const noexcept { return min_iter_; }

  /**
   * @brief Return the maximum number of sampling iterations.
   *
   * @return The maximum number of sampling iterations.
   */
  std::size_t max_iter() const noexcept { return max_iter_; }

  /**
   * @brief Return the maximum number of trajectory doublings
   * for Nuts.
   *
   * @return The maximum number of trajectory doublings.
   */
  std::size_t max_trajectory_doublings() const noexcept {
    return max_trajectory_doublings_;
  }

  /**
   * @brief Return the maximum number of stepsize halvings
   * for Nuts.
   *
   * @return The maximum number of trajectory doublings.
   */
  std::size_t max_step_halvings() const noexcept { return max_step_halvings_; }

  /**
   * @brief Return the maximum error in the Hamiltonian allowed for Walnutpie.
   *
   * @return The maximum error in the Hamiltonian allowed for Walnutpie.
   */
  double max_hamiltonian_error() const noexcept {
    return max_hamiltonian_error_;
  }

  /**
   * @brief Return the minimum number of micro steps per macro step.
   *
   * @return The minimum number of micro steps per macro step.
   */
  std::size_t min_micro_steps() const noexcept { return min_micro_steps_; }

  /**
   * @brief Return the convergence tolerance for the R-hat statistic.
   *
   * @return The convergence tolerance for the R-hat statistic.
   */
  double rhat_converge_tol() const noexcept { return rhat_converge_tol_; }

 private:
  friend class SamplingConfigBuilder;

  SamplingConfig() = default;

  std::size_t min_iter_ = 50;
  std::size_t max_iter_ = 1000;
  std::size_t max_trajectory_doublings_ = 5;
  std::size_t max_step_halvings_ = 5;
  double max_hamiltonian_error_ = 0.5;
  std::size_t min_micro_steps_ = 1;
  double rhat_converge_tol_ = 1.01;
};

/**
 * @brief The builder for sampling configurations.
 *
 * An example use would be:
 * @code
 * SampleConfigBuilder(50u, 100u)
 *     .max_step_halvings(4u)
 *     .min_micro_steps(2u)
 *     .build();
 * @endcode
 */
class SamplingConfigBuilder {
 public:
  /**
   * @brief Set the minimum and maximum number of iterations.
   *
   * @param[in] min_iter The minimum number of iterations.
   * @param[in] max_iter The maximum number of iterations.
   * @return A reference to this builder for chaining.
   * @throw std::invalid_argument If the minimum number of iterations
   * is greater than the maximum number of iterations.
   */
  SamplingConfigBuilder& min_max_iter(std::size_t min_iter,
                                      std::size_t max_iter) {
    if (min_iter > max_iter) {
      throw std::invalid_argument("min_iter must be <= max_iter");
    }
    cfg_.min_iter_ = min_iter;
    cfg_.max_iter_ = max_iter;
    return *this;
  }

  /**
   * @brief Set the maximum number of trajectory doublings.
   *
   * @param[in] v The maximum number of trajectory doublings.
   * @return A reference to this builder for chaining.
   */
  SamplingConfigBuilder& max_trajectory_doublings(std::size_t v) noexcept {
    cfg_.max_trajectory_doublings_ = v;
    return *this;
  }

  /**
   * @brief Set the maximum number of step size halvings.
   *
   * @param[in] v The maximum number of step size halvings.
   * @return A reference to this builder for chaining.
   */
  SamplingConfigBuilder& max_step_halvings(std::size_t v) noexcept {
    cfg_.max_step_halvings_ = v;
    return *this;
  }

  /**
   * @brief Set the maximum error in the Hamiltonian for Walnutpie.
   *
   * @param[in] v The maximum error in the Hamiltonian for Walnutpie.
   * @return A reference to this builder for chaining.
   * @throw std::invalid_argument If the error is not finite and positive.
   */
  SamplingConfigBuilder& max_hamiltonian_error(double v) {
    detail::validate_finite_positive(v, "max_hamiltonian_error");
    cfg_.max_hamiltonian_error_ = v;
    return *this;
  }

  /**
   * @brief Set the minimum number of micro steps per macro step.
   *
   * @param[in] v The minimum number of micro steps per macro step.
   * @return A reference to this builder for chaining.
   * @throw std::invalid_argument If the minimum number of steps is not
   * positive.
   */
  SamplingConfigBuilder& min_micro_steps(std::size_t v) {
    detail::validate_positive(v, "min_micro_steps");
    cfg_.min_micro_steps_ = v;
    return *this;
  }

  /**
   * @brief Set the R-hat convergence tolerance.
   *
   * @param[in] v The R-hat convergence tolerance.
   * @return A reference to this builder for chaining.
   * @throw std::invalid_argument If the tolerance is not finite and > 1.
   */
  SamplingConfigBuilder& rhat_converge_tol(double v) {
    detail::validate_finite_gt1(v, "rhat_convergence_tol");
    cfg_.rhat_converge_tol_ = v;
    return *this;
  }

  /**
   * @brief Return the sampling configuration.
   *
   * @return The sampling configuration.
   */
  SamplingConfig build() { return cfg_; }

 private:
  SamplingConfig cfg_;
};

/**
 * @internal @brief Print the tuning parameters specified by the sampling
 * configuration to the output stream.
 *
 * @param[in,out] out Output stream to which configuration is printed.
 * @param[in] cfg The sampling configuration.
 * @return The output stream for chained calls.
 */
inline std::ostream& operator<<(std::ostream& out, const SamplingConfig& cfg) {
  out << "SamplingConfig\n"
      << "  min_iter                   = " << cfg.min_iter() << "\n"
      << "  max_iter                   = " << cfg.max_iter() << "\n"
      << "  max_trajectory_doublings   = " << cfg.max_trajectory_doublings()
      << "\n"
      << "  max_step_halvings          = " << cfg.max_step_halvings() << "\n"
      << "  max_hamiltonian_error      = " << cfg.max_hamiltonian_error()
      << "\n"
      << "  min_micro_steps            = " << cfg.min_micro_steps() << "\n"
      << "  rhat_converge_tol          = " << cfg.rhat_converge_tol() << "\n";
  return out;
}

/**
 * @brief Encapsulated configuration for Walnutpie.
 *
 * Walnuts configurations include initialization, warmup, and sampling
 * configurations.
 */
class WalnutsConfig {
 public:
  /**
   * @brief Construct a Walnuts configuration given the component
   * configurations.
   *
   * The arguments will be moved if they are rvalues and copied if lvalues.
   *
   * @param[in] init The initialization configuration.
   * @param[in] warmup The warmup configuration.
   * @param[in] sampling The sampling configuration.
   */
  WalnutsConfig(InitConfig init, WarmupConfig warmup, SamplingConfig sampling)
      : init_(std::move(init)),
        warmup_(std::move(warmup)),
        sampling_(std::move(sampling)) {};

  /**
   * @brief Return the initialization configuration.
   *
   * @return The initialization configuration.
   */
  const InitConfig& init() const noexcept { return init_; }

  /**
   * @brief Return the warmup configuration.
   *
   * @return The warmup configuration.
   */
  const WarmupConfig& warmup() const noexcept { return warmup_; }

  /**
   * @brief Return the sampling configuration.
   *
   * @return The sampling configuration.
   */
  const SamplingConfig& sampling() const noexcept { return sampling_; }

 private:
  /** The initialization configuration for all chains. */
  InitConfig init_;

  /** The warmup configuration shared by all chains. */
  WarmupConfig warmup_;

  /** The sampling configuration shared by all chains for warmup and sampling.
   */
  SamplingConfig sampling_;
};

/**
 * @brief Print the Walnuts configuration.
 *
 * This just delegates to printing the initialization, warmup, and
 * sampling configurations separated by newlines.
 *
 * @param[in,out] out Output stream to which configuration is printed.
 * @param[in] cfg The Walnuts configuration.
 * @return The output stream for chained calls.
 */
inline std::ostream& operator<<(std::ostream& out, const WalnutsConfig& cfg) {
  out << cfg.init() << "\n" << cfg.warmup() << "\n" << cfg.sampling() << "\n";
  return out;
}

}  // namespace walnutpie
