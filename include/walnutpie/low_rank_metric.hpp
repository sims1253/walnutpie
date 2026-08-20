#pragma once

#include <Eigen/Dense>
#include <cmath>
#include <vector>

#include "walnutpie/online_moments.hpp"
#include "walnutpie/config.hpp"

namespace walnutpie::detail {

/**
 * @brief Low-rank + diagonal Fisher metric estimator.
 *
 * Implements the Fisher-divergence preconditioning metric of Seyboldt,
 * Carlson & Carpenter (arXiv:2603.18845, Algorithm 1): the inverse mass is
 * diag * (I + U C U^T) with U an orthonormal basis of the jointly-spanned
 * subspace of standardized draws and standardized scores, and C a small
 * rank-r correction. The diagonal part is the closed-form Fisher-optimal
 * diagonal (their Thm 2.2); the low-rank part captures the cross
 * (draw-score) structure a diagonal metric cannot represent (empirically
 * large: per-coordinate draw-score corr ~ -0.5 vs ideal -1).
 *
 * The estimator accumulates a window of (draw, score) pairs (memoryless,
 * 'chopping' discipline) and recomputes U, C at window boundaries:
 * zero-staleness preconditioning within a window is not required here
 * because the metric is fixed per window by construction (the frozen
 * sampler uses the final window's estimate).
 */
class LowRankMetricEstimator {
 public:
  LowRankMetricEstimator(std::size_t dim, std::size_t rank,
                         std::size_t window,
                         double var_floor = 1e-8)
      : dim_(dim), rank_(std::min(rank, dim)), window_(window),
        var_floor_(var_floor) {
    draws_.reserve(window_);
    scores_.reserve(window_);
  }

  void observe(const Eigen::VectorXd& theta, const Eigen::VectorXd& grad) {
    draws_.push_back(theta);
    scores_.push_back(grad);
    if (draws_.size() > window_) {
      draws_.erase(draws_.begin());
      scores_.erase(scores_.begin());
    }
    ++n_obs_;
  }

  std::size_t iterations() const { return n_obs_; }

  /**
   * @brief Compute the diagonal Fisher estimate (Thm 2.2 geometric mean).
   */
  Eigen::VectorXd diagonal_estimate() const {
    if (draws_.size() < 2) {
      return Eigen::VectorXd::Ones(dim_);
    }
    Eigen::MatrixXd Y(dim_, draws_.size());
    Eigen::MatrixXd S(dim_, scores_.size());
    for (std::size_t k = 0; k < draws_.size(); ++k) {
      Y.col(k) = draws_[k];
      S.col(k) = scores_[k];
    }
    Eigen::VectorXd var_y = row_sample_variance(Y);
    Eigen::VectorXd var_s = row_sample_variance(S);
    var_y = var_y.cwiseMax(Eigen::VectorXd::Constant(dim_, var_floor_));
    var_s = var_s.cwiseMax(Eigen::VectorXd::Constant(dim_, var_floor_));
    // inv_mass_diag = sqrt(var_draw / var_score) = geometric mean of
    // var_draw and 1/var_score.
    return (var_y.array() / var_s.array()).sqrt().matrix();
  }

  /**
   * @brief Compute the low-rank factors U (dim x r, orthonormal) and
   * diag_correction c (r) such that
   *   inv_mass = diag * (I + U diag(c) U^T)
   * using the standardized stacked-matrix SVD of Algorithm 1.
   */
  void low_rank_factors(Eigen::MatrixXd& U, Eigen::VectorXd& c,
                        Eigen::VectorXd& diag) const {
    diag = diagonal_estimate();
    if (draws_.size() < 2 || rank_ == 0) {
      U = Eigen::MatrixXd::Zero(dim_, 0);
      c = Eigen::VectorXd::Zero(0);
      return;
    }
    const std::size_t K = draws_.size();
    Eigen::MatrixXd Y(dim_, K), S(dim_, K);
    for (std::size_t k = 0; k < K; ++k) {
      Y.col(k) = draws_[k];
      S.col(k) = scores_[k];
    }
    // Standardize by the diagonal estimate.
    Eigen::VectorXd inv_diag = diag.cwiseInverse();
    Eigen::MatrixXd Ys = inv_diag.asDiagonal() * Y;
    Eigen::MatrixXd Ss = diag.asDiagonal() * S;
    Eigen::MatrixXd stacked(dim_, 2 * K);
    stacked << Ys, Ss;
    // Thin SVD; take top-r right? left singular vectors (dim x r).
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(
        stacked, Eigen::ComputeThinU | Eigen::ComputeThinV);
    std::size_t r = std::min(rank_, static_cast<std::size_t>(svd.rank()));
    U = svd.matrixU().leftCols(r);
    // The singular-value excess above the isotropic baseline (sqrt(2K))
    // gives the correction magnitude per direction.
    Eigen::VectorXd sv = svd.singularValues().head(r);
    const double baseline = std::sqrt(2.0 * static_cast<double>(K));
    c = (sv.array() - baseline).cwiseMax(0.0);
    // Normalize correction to a modest relative magnitude (damping).
    const double denom = baseline + c.sum();
    c = c / std::max(denom, 1e-12);
  }

 private:
  static Eigen::VectorXd row_sample_variance(const Eigen::MatrixXd& M) {
    Eigen::VectorXd mu = M.rowwise().mean();
    Eigen::MatrixXd centered = M.colwise() - mu;
    return centered.cwiseProduct(centered).rowwise().sum() /
           std::max(1.0, static_cast<double>(M.cols() - 1));
  }

  std::size_t dim_;
  std::size_t rank_;
  std::size_t window_;
  double var_floor_;
  std::vector<Eigen::VectorXd> draws_;
  std::vector<Eigen::VectorXd> scores_;
  std::size_t n_obs_ = 0;
};

}  // namespace walnutpie::detail
