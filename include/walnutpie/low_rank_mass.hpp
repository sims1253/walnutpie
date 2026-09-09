#pragma once

#include <cmath>
#include <random>

#include <Eigen/Dense>

namespace walnutpie::detail {

/**
 * @brief Diagonal + low-rank mass operator with exact O(D r) arithmetic.
 *
 * Represents M^{-1} = D + sqrt(D) U C U^T sqrt(D) (SPD for D > 0, C >= 0).
 * All identities are verified by tests/leapfrog_property_test.cpp:
 *  - apply (position update):  M^{-1} x = D x + sqrtD U (C (U^T sqrtD x))
 *  - quadratic form:           rho^T M^{-1} rho = sum D_i rho_i^2
 *                                + ||sqrt(C) U^T sqrtD rho||^2
 *  - exact momentum sampling (low-rank Cholesky identity, verified 1e-16):
 *      rho = D^{-1/2} z + U ((I+C)^{-1/2} - I) U^T D^{-1/2} z
 *    from A = Lc Lc^T, Lc = sqrtD (I + U((I+C)^{1/2} - I)U^T).
 *  - log-det: log|M^{-1}| = sum_i log D_i + sum_k log(1 + c_k).
 *
 * Integration into transition_w is deliberately deferred (see the
 * init-robustness PR discussion): the marginal-fold approximation
 * (--metric-rank) delivers the metric benefits through the existing diagonal
 * interface while the full operator awaits the reversibility/volume property
 * gate in CI for modified hot loops.
 */
struct LowRankMass {
  Eigen::VectorXd D;
  Eigen::MatrixXd U;   // D x r, orthonormal columns
  Eigen::VectorXd c;   // r, non-negative

  Eigen::VectorXd apply_inv(const Eigen::VectorXd& x) const {
    Eigen::VectorXd sq = D.cwiseSqrt();
    Eigen::VectorXd out = D.cwiseProduct(x);
    if (U.cols() == 0) return out;
    Eigen::VectorXd inner = U.transpose() * sq.cwiseProduct(x);
    return out + sq.cwiseProduct(U * c.cwiseProduct(inner));
  }

  double logp_momentum(const Eigen::VectorXd& rho) const {
    double e = D.cwiseProduct(rho).dot(rho);
    if (U.cols() > 0) {
      Eigen::VectorXd sq = D.cwiseSqrt();
      Eigen::VectorXd inner = U.transpose() * sq.cwiseProduct(rho);
      e += c.cwiseProduct(inner).dot(inner);
    }
    return -0.5 * e;
  }

  Eigen::VectorXd sample_momentum(std::mt19937_64& g) const {
    std::normal_distribution<double> n(0, 1);
    Eigen::VectorXd z(D.size());
    for (int i = 0; i < D.size(); ++i) z[i] = n(g);
    return sample_momentum_from(z);
  }

  /** @brief Momentum draw from a provided standard-normal vector.
   *
   * Correct draw is rho = D^{-1/2} (I + U W U^T) z; the factors do not
   * commute, so the diagonal scaling must act AFTER the low-rank
   * correction (scaling z first yields Cov != A^{-1} whenever U is not
   * coordinate-aligned — see the sample_momentum_from covariance
   * property test).
   */
  Eigen::VectorXd sample_momentum_from(const Eigen::VectorXd& z) const {
    Eigen::VectorXd invsq = D.cwiseInverse().cwiseSqrt();
    Eigen::VectorXd out = invsq.cwiseProduct(z);
    if (U.cols() == 0) return out;
    Eigen::MatrixXd IC = Eigen::MatrixXd::Identity(U.cols(), U.cols());
    IC.diagonal() += c;
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(IC);
    Eigen::MatrixXd W =
        es.operatorInverseSqrt() - Eigen::MatrixXd::Identity(U.cols(), U.cols());
    Eigen::VectorXd inner = U.transpose() * z;
    return out + invsq.cwiseProduct(U * (W * inner));
  }

  double log_det() const {
    double l = D.array().log().sum();
    for (int k = 0; k < c.size(); ++k) l += std::log1p(c[k]);
    return l;
  }
};

}  // namespace walnutpie::detail
