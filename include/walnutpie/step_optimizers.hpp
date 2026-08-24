#pragma once

#include <cmath>
#include <cstddef>
#include <vector>

#include "walnutpie/concepts.hpp"
#include "walnutpie/config.hpp"

namespace walnutpie::detail {

/**
 * @brief Nesterov-style dual averaging for step size adaptation,
 * as used in Stan (Hoffman & Gelman 2014, Algorithm 5/6).
 *
 * Adapts log step size to a target acceptance statistic. This variant
 * observes the same per-micro-step acceptance statistics as the Adam
 * adapter and is compatible with the `StepSizeAdapter` concept.
 */
class DualAveraging {
 public:
  DualAveraging(double step_size_init, double accept_rate_target,
                double gamma, double t0, double kappa,
                bool use_average = false)
      : use_average_(use_average),
        mu_(std::log(10.0 * step_size_init)),
        x_(0.0),
        x_bar_(0.0),
        h_bar_(0.0),
        t_(0),
        accept_rate_target_(accept_rate_target),
        gamma_(gamma),
        t0_(t0),
        kappa_(kappa) {}

  void operator()(double alpha) noexcept {
    ++t_;
    const double eta = 1.0 / (static_cast<double>(t_) + t0_);
    h_bar_ = (1.0 - eta) * h_bar_ + eta * (accept_rate_target_ - alpha);
    x_ = mu_ - std::sqrt(static_cast<double>(t_)) / gamma_ * h_bar_;
    const double w = std::pow(static_cast<double>(t_), -kappa_);
    x_bar_ = w * x_ + (1.0 - w) * x_bar_;
  }

  double step_size() const noexcept {
    return std::exp(use_average_ ? x_bar_ : x_);
  }

  double step_size_bar() const noexcept { return std::exp(x_bar_); }

 private:
  bool use_average_;
  double mu_;
  double x_;
  double x_bar_;
  double h_bar_;
  std::size_t t_;
  const double accept_rate_target_;
  const double gamma_;
  const double t0_;
  const double kappa_;
};

/**
 * @brief Scalar AdEMAMix (Pagliardini et al. 2024, arXiv:2409.03137)
 * step-size adapter: dual exponential moving averages (fast + slow)
 * over noisy acceptance-statistic gradients.
 */
class AdEMAMix {
 public:
  AdEMAMix(double step_size_init, double accept_rate_target,
           double learning_rate, double gradient_decay,
           double sq_gradient_decay, double slow_gradient_decay,
           double stabilization, double learn_rate_decay,
           std::size_t slow_warmup_iters)
      : theta_(std::log(step_size_init)),
        m_fast_(0),
        m_slow_(0),
        v_(0),
        t_(0),
        gradient_decay_pow_(1),
        sq_gradient_decay_pow_(1),
        target_accept_rate_(accept_rate_target),
        learn_rate_(learning_rate),
        gradient_decay_(gradient_decay),
        sq_gradient_decay_(sq_gradient_decay),
        slow_gradient_decay_(slow_gradient_decay),
        stabilization_(stabilization),
        learn_rate_decay_(learn_rate_decay),
        slow_warmup_iters_(slow_warmup_iters) {}

  void operator()(double alpha) noexcept {
    ++t_;
    gradient_decay_pow_ *= gradient_decay_;
    sq_gradient_decay_pow_ *= sq_gradient_decay_;

    const double grad = target_accept_rate_ - alpha;

    m_fast_ = gradient_decay_ * m_fast_ + (1.0 - gradient_decay_) * grad;
    v_ = sq_gradient_decay_ * v_ + (1.0 - sq_gradient_decay_) * grad * grad;

    // slow EMA with warmup scaling (alpha_t ramp over slow_warmup_iters_)
    double alpha_t = 1.0;
    if (t_ < slow_warmup_iters_) {
      alpha_t = static_cast<double>(t_) / static_cast<double>(slow_warmup_iters_);
    }
    m_slow_ = m_slow_ * (1.0 - alpha_t * (1.0 - slow_gradient_decay_)) +
              alpha_t * (1.0 - slow_gradient_decay_) * grad;

    const double m_hat_fast =
        m_fast_ / (1.0 - gradient_decay_pow_);
    const double v_hat = v_ / (1.0 - sq_gradient_decay_pow_);

    // normalized slow term scaled to match fast magnitude
    const double slow_scale = (1.0 - gradient_decay_) / (1.0 - slow_gradient_decay_);
    const double m_mix = m_hat_fast + slow_scale * m_slow_;

    const double decayed_learn_rate =
        learn_rate_ / std::pow(static_cast<double>(t_), learn_rate_decay_);
    theta_ -= decayed_learn_rate * m_mix / (std::sqrt(v_hat) + stabilization_);
  }

  double step_size() const noexcept { return std::exp(theta_); }

 private:
  double theta_;
  double m_fast_;
  double m_slow_;
  double v_;
  std::size_t t_;
  double gradient_decay_pow_;
  double sq_gradient_decay_pow_;
  const double target_accept_rate_;
  const double learn_rate_;
  const double gradient_decay_;
  const double sq_gradient_decay_;
  const double slow_gradient_decay_;
  const double stabilization_;
  const double learn_rate_decay_;
  const std::size_t slow_warmup_iters_;
};

/**
 * @brief Scalar AdaBelief (Zhuang et al. 2020, arXiv:2010.07468) adapter:
 * the second moment tracks the deviation of the gradient from the mean,
 * giving larger effective steps when gradients are inconsistent.
 */
class AdaBelief {
 public:
  AdaBelief(double step_size_init, double accept_rate_target,
            double learning_rate, double gradient_decay,
            double sq_gradient_decay, double stabilization,
            double learn_rate_decay)
      : theta_(std::log(step_size_init)),
        m_(0),
        v_(0),
        t_(0),
        gradient_decay_pow_(1),
        sq_gradient_decay_pow_(1),
        target_accept_rate_(accept_rate_target),
        learn_rate_(learning_rate),
        gradient_decay_(gradient_decay),
        sq_gradient_decay_(sq_gradient_decay),
        stabilization_(stabilization),
        learn_rate_decay_(learn_rate_decay) {}

  void operator()(double alpha) noexcept {
    ++t_;
    gradient_decay_pow_ *= gradient_decay_;
    sq_gradient_decay_pow_ *= sq_gradient_decay_;

    const double grad = target_accept_rate_ - alpha;

    m_ = gradient_decay_ * m_ + (1.0 - gradient_decay_) * grad;
    const double grad_dev = grad - m_;
    v_ = sq_gradient_decay_ * v_ +
         (1.0 - sq_gradient_decay_) * grad_dev * grad_dev;

    const double m_hat = m_ / (1.0 - gradient_decay_pow_);
    const double v_hat = v_ / (1.0 - sq_gradient_decay_pow_);

    const double decayed_learn_rate =
        learn_rate_ / std::pow(static_cast<double>(t_), learn_rate_decay_);
    theta_ -= decayed_learn_rate * m_hat / (std::sqrt(v_hat) + stabilization_);
  }

  double step_size() const noexcept { return std::exp(theta_); }

 private:
  double theta_;
  double m_;
  double v_;
  std::size_t t_;
  double gradient_decay_pow_;
  double sq_gradient_decay_pow_;
  const double target_accept_rate_;
  const double learn_rate_;
  const double gradient_decay_;
  const double sq_gradient_decay_;
  const double stabilization_;
  const double learn_rate_decay_;
};

/**
 * @brief Wrap any step size adapter, batching observations.
 *
 * The inner adapter is updated once per `stride` observations with the mean
 * acceptance statistic, which slows its internal iteration count. This is
 * useful because the base adapters in this library observe per-micro-step
 * statistics (hundreds per warmup iteration), while classic schemes such as
 * dual averaging were calibrated for one observation per iteration.
 */
template <StepSizeAdapter Inner>
class BatchedAdapter {
 public:
  BatchedAdapter(Inner&& inner, std::size_t stride)
      : inner_(std::move(inner)), stride_(stride), count_(0), alpha_sum_(0.0) {}

  void operator()(double alpha) noexcept {
    alpha_sum_ += alpha;
    if (++count_ >= stride_) {
      inner_(alpha_sum_ / static_cast<double>(count_));
      count_ = 0;
      alpha_sum_ = 0.0;
    }
  }

  double step_size() const noexcept { return inner_.step_size(); }

 private:
  Inner inner_;
  const std::size_t stride_;
  std::size_t count_;
  double alpha_sum_;
};


/**
 * @brief Wrap a step size adapter, clipping the acceptance-statistic
 * gradient signal.
 *
 * A single near-zero acceptance statistic produces a large gradient
 * (target - alpha up to the target itself) that can crash the log step
 * size; clipping the observed statistic to [1 - c, 1] bounds that impulse.
 * (Equivalent to clipping the gradient to [-c, c].)
 */
template <StepSizeAdapter Inner>
class ClippedAdapter {
 public:
  ClippedAdapter(Inner&& inner, double clip)
      : inner_(std::move(inner)), clip_(clip) {}

  void operator()(double alpha) noexcept {
    const double floored = std::max(alpha, 1.0 - clip_);
    inner_(floored);
  }

  double step_size() const noexcept { return inner_.step_size(); }

 private:
  Inner inner_;
  const double clip_;
};


/**
 * @brief Wrap a step size adapter with anti-windup (conditional integration).
 *
 * Classical ODE step-size control (Gustafsson-Lundh-Soderlind) freezes the
 * integral action while the controller is saturated. Here: when the observed
 * acceptance statistic is exactly zero (a divergent macro step) or below
 * `floor_alpha`, the observation is passed to the inner adapter but the
 * adapter's accumulated state is protected by limiting how many such
 * observations update it: only 1 in `pass_rate` saturated observations is
 * forwarded (the rest are dropped). This bounds the windup of the error
 * integral (Adam's m) during a divergence burst, preventing the log step
 * size from crashing afterwards.
 */
template <StepSizeAdapter Inner>
class AntiWindupAdapter {
 public:
  AntiWindupAdapter(Inner&& inner, double floor_alpha = 1e-12,
                    std::size_t pass_rate = 8)
      : inner_(std::move(inner)),
        floor_alpha_(floor_alpha),
        pass_rate_(pass_rate),
        saturated_seen_(0) {}

  void operator()(double alpha) noexcept {
    // !(alpha >= floor) treats NaN as saturated: a NaN acceptance statistic
    // must never reach the inner adapter (it would poison its state).
    if (pass_rate_ > 0 && !(alpha >= floor_alpha_)) {
      ++saturated_seen_;
      if (saturated_seen_ % pass_rate_ != 1) {
        return;  // drop this saturated observation
      }
    }
    inner_(alpha);
  }

  double step_size() const noexcept { return inner_.step_size(); }

 private:
  Inner inner_;
  const double floor_alpha_;
  const std::size_t pass_rate_;
  std::size_t saturated_seen_;
};

}  // namespace walnutpie::detail
