#include <gtest/gtest.h>

#include <cmath>
#include <random>

#include <Eigen/Dense>

#include <walnutpie.hpp>

#include "test_util.hpp"

// Dense metric estimation **************************************************

namespace {

using walnutpie::detail::DenseMetricEstimator;
using walnutpie::detail::DenseMetricTransform;

/** Covariance of an AR(1) target with unit marginal variance. */
Eigen::MatrixXd ar1_covariance(Eigen::Index dim, double rho) {
  Eigen::MatrixXd cov(dim, dim);
  for (Eigen::Index i = 0; i < dim; ++i) {
    for (Eigen::Index j = 0; j < dim; ++j) {
      cov(i, j) = std::pow(rho, std::abs(i - j));
    }
  }
  return cov;
}

/** Draw `n` samples from `N(0, cov)`. */
std::vector<Eigen::VectorXd> draw_gauss(const Eigen::MatrixXd& cov,
                                        std::size_t n, unsigned seed) {
  const Eigen::LLT<Eigen::MatrixXd> llt(cov);
  std::mt19937_64 rng(seed);
  std::normal_distribution<double> normal(0.0, 1.0);
  std::vector<Eigen::VectorXd> out;
  out.reserve(n);
  for (std::size_t t = 0; t < n; ++t) {
    Eigen::VectorXd z(cov.rows());
    for (Eigen::Index i = 0; i < cov.rows(); ++i) {
      z(i) = normal(rng);
    }
    out.emplace_back(llt.matrixL() * z);
  }
  return out;
}

/** Score of `N(0, cov)`, i.e. the gradient of its log density. */
Eigen::VectorXd score_of(const Eigen::MatrixXd& cov, const Eigen::VectorXd& x) {
  const Eigen::LLT<Eigen::MatrixXd> llt(cov);
  return Eigen::VectorXd(llt.matrixU().solve(llt.matrixL().solve(x)));
}

}  // namespace

TEST(DenseMetric, RecoversCovarianceOfCorrelatedTarget) {
  constexpr Eigen::Index kDim = 8;
  constexpr std::size_t kDraws = 20000;
  const Eigen::MatrixXd cov = ar1_covariance(kDim, 0.85);
  DenseMetricEstimator est(kDim, 50);
  for (const auto& x : draw_gauss(cov, kDraws, 20240613)) {
    est.observe(x, score_of(cov, x));
  }
  EXPECT_EQ(est.count(), kDraws - 50);
  EXPECT_LT((est.covariance() - cov).cwiseAbs().maxCoeff(), 0.05);
  EXPECT_GT(est.max_abs_correlation(), 0.8);
}

TEST(DenseMetric, IndependentTargetHasNoCorrelationToExploit) {
  constexpr Eigen::Index kDim = 8;
  DenseMetricEstimator est(kDim, 50);
  for (const auto& x :
       draw_gauss(Eigen::MatrixXd::Identity(kDim, kDim), 8000, 7)) {
    est.observe(x);
  }
  EXPECT_LT(est.max_abs_correlation(), 0.2);
}

TEST(DenseMetric, ExploredChainPassesTheAgreementCheck) {
  // The draw covariance and the independently estimated inverse score
  // covariance must agree when the chain has actually explored.
  constexpr Eigen::Index kDim = 8;
  const Eigen::MatrixXd cov = ar1_covariance(kDim, 0.9);
  DenseMetricEstimator est(kDim, 50);
  for (const auto& x : draw_gauss(cov, 20000, 99)) {
    est.observe(x, score_of(cov, x));
  }
  EXPECT_LT(est.agreement(), 0.5);
}

TEST(DenseMetric, UnexploredChainFailsTheAgreementCheck) {
  // A chain that never moves yields a rank-one sample covariance however many
  // draws are recorded, so the two estimates cannot agree.
  constexpr Eigen::Index kDim = 8;
  DenseMetricEstimator est(kDim, 0);
  const Eigen::VectorXd stuck = Eigen::VectorXd::Constant(kDim, 2.0);
  for (std::size_t t = 0; t < 5000; ++t) {
    est.observe(stuck, score_of(ar1_covariance(kDim, 0.9), stuck));
  }
  EXPECT_GE(est.count(), 4 * static_cast<std::size_t>(kDim));
  EXPECT_GT(est.agreement(), 0.5);
}

TEST(DenseMetric, TooFewObservationsYieldsNoUsableFactor) {
  constexpr Eigen::Index kDim = 8;
  DenseMetricEstimator est(kDim, 0);
  est.observe(Eigen::VectorXd::Ones(kDim));
  bool ok = true;
  const Eigen::MatrixXd L = est.cholesky(1.0, &ok);
  EXPECT_FALSE(ok);
  EXPECT_EQ(L, Eigen::MatrixXd::Identity(kDim, kDim));
}

TEST(DenseMetric, FrozenChainStillYieldsAFactor) {
  constexpr Eigen::Index kDim = 8;
  DenseMetricEstimator est(kDim, 0);
  const Eigen::VectorXd c = Eigen::VectorXd::Constant(kDim, 3.0);
  for (std::size_t t = 0; t < 50; ++t) {
    est.observe(c);
  }
  bool ok = false;
  const Eigen::MatrixXd L = est.cholesky(1.0, &ok);
  EXPECT_TRUE(ok);
  EXPECT_TRUE(L.allFinite());
  EXPECT_EQ(L.rows(), kDim);
}

TEST(DenseMetricTransform, IsInactiveByDefaultAndCopies) {
  DenseMetricTransform t;
  ASSERT_FALSE(t.active());
  const Eigen::VectorXd x = Eigen::VectorXd::LinSpaced(4, 1.0, 4.0);
  Eigen::VectorXd y;
  t.forward(x, y);
  EXPECT_EQ(y, x);
  t.inverse(y, y);
  EXPECT_EQ(y, x);
  const Eigen::VectorXd g = Eigen::VectorXd::LinSpaced(4, 5.0, 8.0);
  Eigen::VectorXd gz;
  t.backward(g, gz);
  EXPECT_EQ(gz, g);
}

TEST(DenseMetricTransform, RoundTripsAndReportsLogDeterminant) {
  constexpr Eigen::Index kDim = 6;
  const Eigen::MatrixXd cov = ar1_covariance(kDim, 0.7);
  const Eigen::LLT<Eigen::MatrixXd> llt(cov);
  DenseMetricTransform t(Eigen::MatrixXd(llt.matrixL()));
  ASSERT_TRUE(t.active());
  EXPECT_EQ(t.dim(), kDim);

  const Eigen::VectorXd z = Eigen::VectorXd::Random(kDim).normalized();
  Eigen::VectorXd theta, back;
  t.forward(z, theta);
  t.inverse(theta, back);
  EXPECT_LT((z - back).cwiseAbs().maxCoeff(), 1e-9);

  double expected = 0.0;
  for (Eigen::Index i = 0; i < kDim; ++i) {
    expected += std::log(llt.matrixL()(i, i));
  }
  EXPECT_NEAR(t.log_determinant(), expected, 1e-12);
}

TEST(DenseMetricTransform, BackwardIsTheAdjointOfForward) {
  // For the map theta = L z, the pullback of a gradient is L' g. That identity
  // is what makes a reparametrized chain correct, so check it directly.
  constexpr Eigen::Index kDim = 5;
  const Eigen::MatrixXd cov = ar1_covariance(kDim, 0.6);
  const Eigen::LLT<Eigen::MatrixXd> llt(cov);
  const Eigen::MatrixXd L(llt.matrixL());
  DenseMetricTransform t(L);
  const Eigen::VectorXd g = Eigen::VectorXd::Random(kDim);
  Eigen::VectorXd gz;
  t.backward(g, gz);
  EXPECT_LT((gz - L.transpose() * g).cwiseAbs().maxCoeff(), 1e-12);
}

TEST(DenseMetricTransform, ReparametrizedDensityHasTheRightGradient) {
  // For theta = L z the pullback of a standard normal is
  // -0.5 z'L'Lz + log|L| with gradient -L'Lz.  Composing the transform the
  // way the log density wrapper does must produce exactly that.
  constexpr Eigen::Index kDim = 4;
  const Eigen::MatrixXd cov = ar1_covariance(kDim, 0.8);
  const Eigen::LLT<Eigen::MatrixXd> llt(cov);
  const Eigen::MatrixXd L(llt.matrixL());
  double logdet = 0.0;
  for (Eigen::Index i = 0; i < kDim; ++i) {
    logdet += std::log(L(i, i));
  }

  const Eigen::VectorXd z = Eigen::VectorXd::Random(kDim);
  Eigen::VectorXd theta, grad;
  DenseMetricTransform t(L);
  t.forward(z, theta);
  // Gradient of the original standard normal at theta, then pulled back.
  t.backward(Eigen::VectorXd(-theta), grad);
  const Eigen::VectorXd expected = -L.transpose() * (L * z);
  EXPECT_LT((grad - expected).cwiseAbs().maxCoeff(), 1e-9);

  // The value of the pullback density, including the Jacobian correction.
  const double logp = -0.5 * theta.squaredNorm() + logdet;
  EXPECT_NEAR(logp, -0.5 * z.dot(L.transpose() * (L * z)) + logdet, 1e-9);
}

TEST(DenseMetricTransform, RejectsNonSquareFactor) {
  EXPECT_THROW(DenseMetricTransform(Eigen::MatrixXd(3, 4)),
               std::invalid_argument);
  EXPECT_THROW(DenseMetricTransform(Eigen::MatrixXd(0, 0)),
               std::invalid_argument);
}

TEST(WarmupConfig, DenseMetricIsOffByDefault) {
  const auto cfg = walnutpie::WarmupConfigBuilder().build();
  EXPECT_FALSE(cfg.metric_dense());
  EXPECT_DOUBLE_EQ(cfg.metric_dense_min_corr(), 0.3);
  EXPECT_EQ(cfg.metric_dense_min_obs(), 4u);
  EXPECT_DOUBLE_EQ(cfg.metric_dense_max_agreement(), 0.5);
}

TEST(WarmupConfig, DenseMetricScreeningIsConfigurable) {
  const auto cfg = walnutpie::WarmupConfigBuilder()
                       .metric_dense(true)
                       .metric_dense_min_corr(0.8)
                       .metric_dense_min_obs(10u)
                       .metric_dense_max_agreement(0.25)
                       .build();
  EXPECT_TRUE(cfg.metric_dense());
  EXPECT_DOUBLE_EQ(cfg.metric_dense_min_corr(), 0.8);
  EXPECT_EQ(cfg.metric_dense_min_obs(), 10u);
  EXPECT_DOUBLE_EQ(cfg.metric_dense_max_agreement(), 0.25);
  EXPECT_THROW(
      walnutpie::WarmupConfigBuilder().metric_dense_max_agreement(-1.0),
      std::invalid_argument);
}
