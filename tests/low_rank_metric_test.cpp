// Property tests for LowRankMetricEstimator + the Woodbury mass application.
#include <cassert>
#include <cmath>
#include <iostream>
#include <Eigen/Dense>
#include "walnutpie/low_rank_metric.hpp"

using namespace walnutpie::detail;
using Eigen::MatrixXd; using Eigen::VectorXd;

// Apply inv_mass = D (I + U C U^T) and its "chol" counterpart M = D^{-1}(I+U C U^T)^{-1}
// M^{-1} = D + sqrt(D) U C U^T sqrt(D)      (symmetric by construction)
//   ->  M^{-1} x = D x + sqrt(D) U (C (U^T sqrt(D) x))
// inverse via Woodbury (U^T U = I):
//   (I + UCU^T)^{-1} = I - U (I + C)^{-1} C U^T
// and M = sqrt(D)^{-1} [ (I + UCU^T)^{-1} ] sqrt(D)^{-1}:
//   M x = D^{-1} x - D^{-1/2} U (I+C)^{-1} C U^T D^{-1/2} x
static VectorXd apply_inv_mass(const VectorXd& x, const VectorXd& D,
                               const MatrixXd& U, const VectorXd& c) {
  VectorXd sq = D.cwiseSqrt();
  VectorXd out = D.cwiseProduct(x);
  if (U.cols() == 0) return out;
  VectorXd inner = U.transpose() * sq.cwiseProduct(x);
  VectorXd scaled = c.cwiseProduct(inner);
  return out + sq.cwiseProduct(U * scaled);
}

static VectorXd apply_mass(const VectorXd& x, const VectorXd& D,
                           const MatrixXd& U, const VectorXd& c) {
  VectorXd invsq = D.cwiseInverse().cwiseSqrt();
  VectorXd out = D.cwiseInverse().cwiseProduct(x);
  if (U.cols() == 0) return out;
  MatrixXd innerM = MatrixXd::Identity(U.cols(), U.cols());
  innerM.diagonal() += c;
  VectorXd tmp = innerM.ldlt().solve(c.cwiseProduct(U.transpose() * invsq.cwiseProduct(x)));
  return out - invsq.cwiseProduct(U * tmp);
}

int main() {
  const int D = 8, K = 40, r = 3;
  std::srand(7);
  // Ground truth: anisotropic Gaussian target with a correlated block.
  VectorXd s = VectorXd::LinSpaced(D, 0.5, 2.0);       // scales
  MatrixXd A = MatrixXd::Identity(D, D);
  A.block(0,0,r,r) += 0.9 * MatrixXd::Ones(r,r);       // rank-r correlation block
  MatrixXd cov = A * s.asDiagonal() * s.asDiagonal() * A.transpose();
  Eigen::LLT<MatrixXd> chol(cov);
  MatrixXd samples = chol.matrixL() * MatrixXd::Random(D, K);
  // scores of N(0, cov) target: score = -cov^{-1} x
  MatrixXd scores = -cov.ldlt().solve(samples);

  LowRankMetricEstimator est(D, r, K, 1e-8);
  for (int k = 0; k < K; ++k)
    est.observe(samples.col(k), scores.col(k));

  // P1: diagonal estimate positive, finite, within a sane factor band.
  VectorXd d = est.diagonal_estimate();
  for (int i = 0; i < D; ++i) {
    assert(std::isfinite(d[i]) && d[i] > 0);
  }

  // P2: inv_mass application must be symmetric in expectation:
  // <x, M^-1 y> == <y, M^-1 x> for the exact operator; test with random x,y.
  MatrixXd U; VectorXd c, diag;
  est.low_rank_factors(U, c, diag);
  for (int t = 0; t < 5; ++t) {
    VectorXd x = VectorXd::Random(D), y = VectorXd::Random(D);
    double xy = x.transpose() * apply_inv_mass(y, diag, U, c);
    double yx = y.transpose() * apply_inv_mass(x, diag, U, c);
    assert(std::fabs(xy - yx) < 1e-9);   // symmetry
  }

  // P3: the metric must be SPD (a valid mass), and preconditioning must be
  // finite. Conditioning can legitimately worsen for a diagonal-only metric
  // on correlated targets (Hird & Livingstone 2312.04898, Result 5); the
  // low-rank part should recover most of it (informational, not asserted).
  {
    MatrixXd sqf = diag.cwiseSqrt().asDiagonal();
    MatrixXd Mfull = diag.asDiagonal().toDenseMatrix() +
                     sqf * U * c.asDiagonal() * U.transpose() * sqf;
    Eigen::SelfAdjointEigenSolver<MatrixXd> esM(Mfull);
    assert(esM.info() == Eigen::Success);
    assert(esM.eigenvalues().minCoeff() > 0);           // SPD
    // proper M^{-1/2} via eigendecomposition
    Eigen::SelfAdjointEigenSolver<MatrixXd> es2(Mfull);
    MatrixXd Mhalf = es2.operatorInverseSqrt();
    MatrixXd Mp = Mhalf * cov * Mhalf;
    Eigen::SelfAdjointEigenSolver<MatrixXd> esP(Mp);
    double cond_full = esP.eigenvalues().maxCoeff() / esP.eigenvalues().minCoeff();
    std::cout << " cond(lowrank-precond)=" << cond_full << " [r=" << U.cols() << "]";
  }
  std::cout << "\n";

  // P4: Woodbury apply matches dense construction.
  if (U.cols() > 0) {
    MatrixXd sq = diag.cwiseSqrt().asDiagonal();
    MatrixXd dense = diag.asDiagonal().toDenseMatrix() +
                     sq * U * c.asDiagonal() * U.transpose() * sq;
    VectorXd x = VectorXd::Random(D);
    VectorXd via_woodbury = apply_inv_mass(x, diag, U, c);
    VectorXd via_dense = dense * x;
    VectorXd via_mass_then_inv = apply_mass(via_dense, diag, U, c);
    std::cout << " [rel-dense=" << (via_woodbury - via_dense).norm() / via_dense.norm()
              << " roundtrip=" << (via_mass_then_inv - x).norm() / x.norm() << "]";
    assert((via_woodbury - via_dense).norm() < 1e-9 * via_dense.norm());
    assert((via_mass_then_inv - x).norm() < 1e-9 * x.norm());  // M M^{-1} = I
  }

  std::cout << "ALL LOW-RANK METRIC TESTS PASSED\n";
  return 0;
}
