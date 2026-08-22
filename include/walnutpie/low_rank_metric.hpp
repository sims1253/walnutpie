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
                         std::size_t window, std::size_t basis = 0,
                         double var_floor = 1e-8)
      : dim_(dim), rank_(std::min(rank, dim)), window_(window),
        var_floor_(var_floor), basis_(basis) {
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
    if (basis_ == 0) {
      // Windowed thin SVD (default; Algorithm 1).
      Eigen::JacobiSVD<Eigen::MatrixXd> svd(
          stacked, Eigen::ComputeThinU | Eigen::ComputeThinV);
      std::size_t r = std::min(rank_, static_cast<std::size_t>(svd.rank()));
      U = svd.matrixU().leftCols(r);
      Eigen::VectorXd sv = svd.singularValues().head(r);
      const double baseline = std::sqrt(2.0 * static_cast<double>(K));
      c = (sv.array() - baseline).cwiseMax(0.0);
      const double denom = baseline + c.sum();
      c = c / std::max(denom, 1e-12);
      prev_U_ = U;
      return;
    }
    if (basis_ == 1) {
      // Streaming orthogonal (power) iteration on C = stacked stacked^T via
      // matrix-free products, warm-started from the previous window's basis.
      std::size_t r = std::min(rank_, dim_);
      Eigen::MatrixXd V;
      if (prev_U_.cols() == r && prev_U_.rows() == dim_) {
        V = prev_U_;
      } else {
        V = Eigen::MatrixXd::Zero(dim_, r);
        // deterministic start: standardized stacked columns' top-leverage
        Eigen::VectorXd cn = stacked.colwise().norm();
        for (std::size_t j = 0; j < r; ++j) {
          std::ptrdiff_t best;
          cn.maxCoeff(&best);
          cn(best) = -1.0;
          V.col(j) = stacked.col(best);  // D-dim column as start direction
        }
      }
      const int iters = 4;  // per window; persistence does the streaming
      for (int it = 0; it < iters; ++it) {
        Eigen::MatrixXd W = stacked * (stacked.transpose() * V);  // C V
        Eigen::HouseholderQR<Eigen::MatrixXd> qr(W);
        V = qr.householderQ() * Eigen::MatrixXd::Identity(
                                    W.rows(), std::min(W.cols(), W.rows()));
      }
      // Weights: Rayleigh-style excess of the directional variance.
      Eigen::VectorXd ray(V.cols());
      for (Eigen::Index j = 0; j < V.cols(); ++j) {
        Eigen::VectorXd Cv = stacked * (stacked.transpose() * V.col(j));
        ray(j) = std::sqrt(std::max(0.0, V.col(j).dot(Cv)));
      }
      const double baseline = std::sqrt(2.0 * static_cast<double>(K));
      U = V;
      c = (ray.array() - baseline).cwiseMax(0.0);
      const double denom = baseline + c.sum();
      c = c / std::max(denom, 1e-12);
      prev_U_ = U;
      return;
    }
    // basis_ 2/3: Muon-style Newton-Schulz column orthogonalization of the
    // stacked matrix (and MuonEq-style row equilibration first for 3).
    Eigen::MatrixXd M = stacked;
    Eigen::VectorXd row_scale = Eigen::VectorXd::Ones(dim_);
    if (basis_ == 3) {
      row_scale = M.cwiseAbs2().rowwise().mean().cwiseSqrt().cwiseMax(1e-12);
      M = row_scale.cwiseInverse().asDiagonal() * M;
    }
    // Column norms of the (equilibrated) matrix are the leverage scores
    // used for rank selection.
    Eigen::VectorXd cn = M.colwise().norm();
    std::size_t r = std::min(rank_, static_cast<std::size_t>(M.cols()));
    // Newton-Schulz quintic iteration on the small Gram side:
    // X <- X (aI + bA + cA^2), A = X^T X (2K x 2K), coefficients per
    // modded-nanogpt Muon (Jordan); converges to column-orthonormal.
    M /= std::max(M.norm(), 1e-12);
    const double a = 3.4445, b = -4.7750, cc = 2.0315;
    for (int it = 0; it < 5; ++it) {
      Eigen::MatrixXd A = M.transpose() * M;
      M *= (a * Eigen::MatrixXd::Identity(A.rows(), A.cols()) + b * A +
            cc * A * A);
      M /= std::max(M.norm(), 1e-12);
    }
    // Select r columns by pre-orthogonalization leverage; the NS iterate's
    // columns are near-equal norm, so selection uses the original scores.
    Eigen::VectorXi sel(r);
    Eigen::VectorXd cn_work = cn;
    for (std::size_t j = 0; j < r; ++j) {
      std::ptrdiff_t best;
      cn_work.maxCoeff(&best);
      cn_work(best) = -1.0;
      sel(j) = static_cast<int>(best);
    }
    Eigen::MatrixXd Usel(M.rows(), sel.size());
    for (Eigen::Index j = 0; j < sel.size(); ++j) {
      Usel.col(j) = M.col(sel(j));
    }
    Eigen::HouseholderQR<Eigen::MatrixXd> qr(Usel);
    U = qr.householderQ() * Eigen::MatrixXd::Identity(
                                Usel.rows(), std::min(Usel.cols(), Usel.rows()));
    // Weights from the selected leverage scores, damped like the SVD path.
    Eigen::VectorXd lev(r);
    for (std::size_t j = 0; j < r; ++j) {
      lev(j) = cn(sel(j));
    }
    const double baseline = std::sqrt(2.0 * static_cast<double>(K));
    c = (lev.array() - baseline).cwiseMax(0.0);
    const double denom = baseline + c.sum();
    c = c / std::max(denom, 1e-12);
    prev_U_ = U;
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
  std::size_t basis_ = 0;

  /**
   * @brief Persistent basis for the streaming (power) mode; carries the
   * previous window's basis forward so refresh cadence is measurable.
   */
  mutable Eigen::MatrixXd prev_U_{Eigen::MatrixXd::Zero(0, 0)};
};

}  // namespace walnutpie::detail
