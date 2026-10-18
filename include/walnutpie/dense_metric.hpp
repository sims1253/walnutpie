#pragma once

#include <cmath>
#include <cstddef>
#include <vector>

#include <Eigen/Dense>

#include "walnutpie/validate.hpp"

namespace walnutpie::detail {

/**
 * @brief Online estimator of a target's second-moment (covariance) matrix.
 *
 * A diagonal metric is the right choice for a target whose scales differ but
 * whose coordinates are independent: for a Gaussian whose variances run from
 * 1 to D, a diagonal metric recovers 97% of the effective sample size of an
 * isotropic target. A diagonal metric cannot represent correlation at all,
 * though, and for a strongly correlated target the loss is severe. For an
 * AR(1) process with correlation 0.99 in 100 dimensions the effective sample
 * size per draw falls to about 1.3% of the isotropic value, while
 * preconditioning by the exact covariance recovers it completely.
 *
 * This estimator maintains the full matrix so that the covariance can be
 * factored at the end of warmup. Updates use the standard rank-one form of
 * Welford's algorithm, so a single observation costs O(dim^2) and the running
 * state costs O(dim^2) memory.
 *
 * @tparam D The type of the observed position.
 */
class DenseMetricEstimator {
 public:
  /**
   * @brief Construct an estimator for the specified dimensionality.
   *
   * @param[in] dim The number of coordinates.
   * @param[in] init_count The number of initial observations to ignore, so
   * that the transient before the chain has moved does not dominate.
   * @throw std::invalid_argument If `dim` is zero.
   */
  explicit DenseMetricEstimator(Eigen::Index dim, std::size_t init_count = 0)
      : dim_(dim),
        init_count_(init_count),
        mean_(Eigen::VectorXd::Zero(dim)),
        M2_(Eigen::MatrixXd::Zero(dim, dim)),
        n_(0) {
    detail::validate_positive(dim, "dim");
  }

  /**
   * @brief Record one observation.
   *
   * @param[in] x The observed position.
   * @pre `x.size()` is the dimensionality this estimator was built for.
   */
  void observe(const Eigen::VectorXd& x) {
    ++n_;
    if (n_ <= init_count_) {
      return;
    }
    const double inv_n = 1.0 / static_cast<double>(n_ - init_count_);
    const Eigen::VectorXd delta = x - mean_;
    mean_ += inv_n * delta;
    // Welford rank-one update of the scatter about the running mean.
    M2_.noalias() += delta * (x - mean_).transpose();
  }

  /**
   * @brief Return the number of observations retained.
   */
  std::size_t count() const { return n_ > init_count_ ? n_ - init_count_ : 0; }

  /**
   * @brief Return the unbiased sample covariance of the retained
   * observations.
   *
   * @return The sample covariance, or a zero matrix if too few
   * observations have been retained.
   */
  Eigen::MatrixXd covariance() const {
    const std::size_t k = count();
    if (k < 2) {
      return Eigen::MatrixXd::Zero(dim_, dim_);
    }
    return M2_ / static_cast<double>(k - 1);
  }

  /**
   * @brief Return the mean of the retained observations.
   */
  const Eigen::VectorXd& mean() const { return mean_; }

  /**
   * @brief Return how strongly correlated the coordinates are.
   *
   * This is the largest absolute off-diagonal correlation in the sample
   * covariance, which is the quantity a diagonal metric is unable to
   * represent. It is one for coordinates that are perfectly correlated and
   * zero for independent coordinates.
   *
   * @return The largest absolute off-diagonal correlation, or zero if the
   * sample is too small to estimate one.
   */
  double max_abs_correlation() const {
    const std::size_t k = count();
    if (k < 2) {
      return 0.0;
    }
    const Eigen::MatrixXd cov = covariance();
    Eigen::VectorXd s(dim_);
    for (Eigen::Index i = 0; i < dim_; ++i) {
      s(i) = std::sqrt(std::max(cov(i, i), 0.0));
    }
    double worst = 0.0;
    for (Eigen::Index i = 0; i < dim_; ++i) {
      for (Eigen::Index j = i + 1; j < dim_; ++j) {
        if (s(i) <= 0.0 || s(j) <= 0.0) {
          continue;
        }
        worst = std::max(worst, std::fabs(cov(i, j) / (s(i) * s(j))));
      }
    }
    return worst;
  }

  /**
   * @brief Return the Cholesky factor of a usable covariance.
   *
   * The sample covariance of a chain that has barely moved is close to
   * singular, so the estimate is shrunk towards its diagonal before being
   * factored. Shrinkage interpolates between the sample covariance `S` and its
   * diagonal, `S_shrunk = (1 - a) S + a diag(S)`, for the smallest `a` in
   * `[0, 1]` that makes the factorization succeed. The diagonal is always
   * positive by construction, so some `a` always works.
   *
   * @param[in] max_shrink The largest shrinkage to allow. A value of one
   * returns the diagonal, that is, the metric a diagonal estimator would
   * have produced.
   * @param[out] ok Set to `false` if the factorization could not be made to
   * succeed within `max_shrink`, in which case the caller should fall back to
   * a diagonal metric.
   * @return The lower Cholesky factor `L` with `L L'` equal to the shrunk
   * covariance, or an identity matrix on failure.
   */
  Eigen::MatrixXd cholesky(double max_shrink = 1.0, bool* ok = nullptr) const {
    if (ok) {
      *ok = false;
    }
    const std::size_t k = count();
    if (k < 2) {
      return Eigen::MatrixXd::Identity(dim_, dim_);
    }
    Eigen::MatrixXd cov = covariance();
    // Guard against a degenerate diagonal from a frozen chain.
    for (Eigen::Index i = 0; i < dim_; ++i) {
      const double floor = 1e-12 * std::max(cov(i, i), 1.0);
      if (!(cov(i, i) > floor)) {
        cov(i, i) = floor;
      }
    }
    const Eigen::MatrixXd diag = Eigen::MatrixXd(cov.diagonal().asDiagonal());
    for (int step = 0; step <= 64; ++step) {
      const double a = max_shrink * (static_cast<double>(step) / 64.0);
      Eigen::MatrixXd shrunk = (1.0 - a) * cov + a * diag;
      Eigen::LLT<Eigen::MatrixXd> llt(shrunk);
      if (llt.info() == Eigen::Success) {
        if (ok) {
          *ok = true;
        }
        return Eigen::MatrixXd(llt.matrixL());
      }
    }
    return Eigen::MatrixXd::Identity(dim_, dim_);
  }

 private:
  /** The number of coordinates. */
  Eigen::Index dim_;

  /** The number of leading observations to discard. */
  std::size_t init_count_;

  /** The running mean of the retained observations. */
  Eigen::VectorXd mean_;

  /** The scatter about the running mean. */
  Eigen::MatrixXd M2_;

  /** The number of observations made, including discarded ones. */
  std::size_t n_;
};

/**
 * @brief The linear reparametrization used to sample under a full-rank
 * metric.
 *
 * Sampling in the coordinates `z` defined by `theta = L z`, where `L` is the
 * Cholesky factor of the estimated covariance, turns the target into an
 * isotropic one and leaves the sampler's own diagonal metric and its cost per
 * leapfrog step unchanged. The cost is the two triangular solves this class
 * performs per log density evaluation, which is why a full-rank metric is only
 * worth its price when the gradient is expensive relative to `dim^2`.
 */
class DenseMetricTransform {
 public:
  /**
   * @brief Construct an identity transform, which leaves positions unchanged.
   */
  DenseMetricTransform() : dim_(0) {}

  /**
   * @brief Construct a transform from a Cholesky factor.
   *
   * @param[in] L The lower Cholesky factor of the covariance, `dim` by `dim`.
   * @throw std::invalid_argument If `L` is empty or not square.
   */
  explicit DenseMetricTransform(const Eigen::MatrixXd& L)
      : dim_(L.rows()), L_(L), log_det_(0.0) {
    if (L.rows() == 0 || L.rows() != L.cols()) {
      throw std::invalid_argument("L must be a non-empty square matrix.");
    }
    // log|L| from the Cholesky diagonal, used to correct the log density.
    for (Eigen::Index i = 0; i < dim_; ++i) {
      log_det_ += std::log(std::max(L(i, i), 1e-300));
    }
  }

  /** Return `true` if this transform actually changes positions. */
  bool active() const { return dim_ > 0; }

  /** Return the dimensionality of the transform. */
  Eigen::Index dim() const { return dim_; }

  /** Return the log determinant of the transform, i.e. `log|L|`. */
  double log_determinant() const { return log_det_; }

  /**
   * @brief Map a position in sampling coordinates to target coordinates.
   *
   * An inactive transform is the identity, so it copies `z`.
   */
  void forward(const Eigen::VectorXd& z, Eigen::VectorXd& theta) const {
    if (dim_ == 0) {
      theta = z;
      return;
    }
    theta.noalias() = L_ * z;
  }

  /**
   * @brief Map a position in target coordinates to sampling coordinates.
   *
   * An inactive transform is the identity, so it copies `theta`.
   */
  void inverse(const Eigen::VectorXd& theta, Eigen::VectorXd& z) const {
    if (dim_ == 0) {
      z = theta;
      return;
    }
    z = L_.triangularView<Eigen::Lower>().solve(theta);
  }

  /**
   * @brief Map a gradient with respect to target coordinates to a gradient
   * with respect to sampling coordinates.
   *
   * This is the adjoint of `forward`, so the product of the transform and its
   * adjoint is the identity and no factor of the determinant appears. An
   * inactive transform copies `g`.
   *
   * @param[in] g The gradient with respect to `theta`.
   * @param[out] gz The gradient with respect to `z`.
   */
  void backward(const Eigen::VectorXd& g, Eigen::VectorXd& gz) const {
    if (dim_ == 0) {
      gz = g;
      return;
    }
    gz.noalias() = L_.transpose() * g;
  }

 private:
  /** The dimensionality, or zero when the transform is inactive. */
  Eigen::Index dim_;

  /** The Cholesky factor. */
  Eigen::MatrixXd L_;

  /** The log determinant of `L`. */
  double log_det_;
};

}  // namespace walnutpie::detail