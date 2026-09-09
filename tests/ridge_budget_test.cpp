#include <gtest/gtest.h>
#include <atomic>
#include <bit>
#include <cstdint>
#include <limits>
#include <random>
#include <walnutpie/api.hpp>

namespace {
using walnutpie::detail::ridge_micro_budget;
TEST(RidgeBudget, PositiveCapsBelowFloorArePreserved) {
  for (std::size_t cap : {1, 2, 15, 16, 17, 128}) {
    EXPECT_EQ(walnutpie::SamplingConfigBuilder()
                  .ridge_min_micro(cap)
                  .build()
                  .ridge_min_micro(),
              cap);
  }
  EXPECT_THROW(walnutpie::SamplingConfigBuilder().ridge_min_micro(0),
               std::invalid_argument);
  for (std::size_t cap : {1, 2, 15, 16}) {
    EXPECT_EQ(ridge_micro_budget(1.0, cap), cap);
    EXPECT_EQ(ridge_micro_budget(1000.0, cap), cap);
  }
}
TEST(RidgeBudget, KeepsTruncationAndSaturatesAtBoundary) {
  EXPECT_EQ(ridge_micro_budget(1.0, 17), 16);
  EXPECT_EQ(ridge_micro_budget(std::nextafter(17.0 / 16.0, 0.0), 17), 16);
  EXPECT_EQ(ridge_micro_budget(17.0 / 16.0, 17), 17);
  EXPECT_EQ(ridge_micro_budget(std::nextafter(17.0 / 16.0, 2.0), 17), 17);
  EXPECT_EQ(ridge_micro_budget(1.5, 128), 24);
  EXPECT_EQ(ridge_micro_budget(2.0, 128), 32);
  EXPECT_EQ(ridge_micro_budget(4.0, 128), 64);
  EXPECT_EQ(ridge_micro_budget(std::nextafter(8.0, 0.0), 128), 127);
  EXPECT_EQ(ridge_micro_budget(8.0, 128), 128);
}
TEST(RidgeBudget, HugeDemandIsClippedBeforeIntegerConversion) {
  const auto cap = std::numeric_limits<std::size_t>::max();
  for (double scale : {std::numeric_limits<double>::max(),
                       std::numeric_limits<double>::infinity()}) {
    EXPECT_EQ(ridge_micro_budget(scale, 128), 128);
    EXPECT_EQ(ridge_micro_budget(scale, cap), cap);
  }
  const double boundary = static_cast<double>(cap) / 16.0;
  EXPECT_EQ(ridge_micro_budget(boundary, cap), cap);
  EXPECT_LE(ridge_micro_budget(std::nextafter(boundary, 0.0), cap), cap);
}
TEST(RidgeBudget, PreconditionsAreExplicit) {
  EXPECT_THROW(ridge_micro_budget(2.0, 0), std::invalid_argument);
  for (double scale : {-1.0, 0.0, .99, std::numeric_limits<double>::quiet_NaN(),
                       -std::numeric_limits<double>::infinity()}) {
    EXPECT_THROW(ridge_micro_budget(scale, 128), std::invalid_argument);
  }
}
TEST(RidgeBudget, BoundedMatrixAndOldFiniteBehavior) {
  for (std::size_t cap : {1, 2, 15, 16, 17, 31, 32, 127, 128, 1024}) {
    for (double scale : {1.0, 1.01, 1.5, 2.0, 7.999, 8.0, 1000.0}) {
      auto got = ridge_micro_budget(scale, cap);
      EXPECT_GE(got, 1);
      EXPECT_LE(got, cap);
      if (cap >= 16) {
        auto old = std::max<std::size_t>(
            std::min(static_cast<std::size_t>(16 * scale), cap), 16);
        EXPECT_EQ(got, old);
      }
    }
  }
}
// Synthetic constant-density target: tests plumbing, not a valid posterior.
struct Flat {
  std::atomic<std::size_t>* calls;
  void operator()(const Eigen::VectorXd& x, double& lp,
                  Eigen::VectorXd& grad) const {
    ++*calls;
    lp = 0.0;
    grad = Eigen::VectorXd::Zero(x.size());
  }
};
struct Handler {
  std::atomic<std::size_t>* calls;
  int freezes = 0, draws = 0;
  std::vector<std::uint64_t> trace;
  void add(double x) { trace.push_back(std::bit_cast<std::uint64_t>(x)); }
  void on_warmup(const Eigen::VectorXd& x, double lp, double step,
                 const Eigen::VectorXd& inv) {
    add(x[0]);
    add(lp);
    add(step);
    add(inv[0]);
  }
  void on_warmup_complete(double step, const Eigen::VectorXd& inv) {
    ++freezes;
    *calls = 0;
    add(step);
    add(inv[0]);
  }
  void on_sample(const Eigen::VectorXd& x, double lp) {
    ++draws;
    add(x[0]);
    add(lp);
  }
  void on_logp_exception(const Eigen::VectorXd&,
                         const std::exception&) noexcept {}
};
struct Global {
  void on_r_hat(double) {}
};
struct Interrupt {
  void throw_if_interrupted() const {}
};
struct Result {
  std::size_t sample_calls;
  std::vector<int> freezes;
  std::vector<std::vector<std::uint64_t>> traces;
};
Result run(double threshold, std::size_t cap) {
  using namespace walnutpie;
  std::atomic<std::size_t> calls = 0;
  Flat flat{&calls};
  std::vector<Handler> handlers(2);
  for (auto& h : handlers) {
    h.calls = &calls;
  }
  std::vector<Eigen::VectorXd> initial{Eigen::VectorXd::Constant(1, -1e6),
                                       Eigen::VectorXd::Constant(1, 1e6)};
  auto init =
      InitConfigBuilder(2, 1).positions(initial).step_sizes(1e-6).build();
  auto warm =
      WarmupConfigBuilder().min_max_iter(4, 4).publish_stride(1).build();
  auto sampling = SamplingConfigBuilder()
                      .min_max_iter(1, 1)
                      .max_trajectory_doublings(1)
                      .max_step_halvings(1)
                      .min_micro_steps(32)
                      .ridge_guard(threshold)
                      .ridge_min_micro(cap)
                      .build();
  Global global;
  walnuts<std::mt19937>(42, handlers, global, Interrupt{}, flat,
                        WalnutsConfig(init, warm, sampling));
  Result r{calls.load(), {}, {}};
  for (auto& h : handlers) {
    EXPECT_EQ(h.draws, 1);
    r.freezes.push_back(h.freezes);
    r.traces.push_back(h.trace);
  }
  return r;
}
TEST(RidgeBudget, FiredReplacementHonorsSmallCapEvenWhenBudgetDecreases) {
  for (std::size_t cap : {1, 2, 15}) {
    auto result = run(.1, cap);
    EXPECT_EQ(result.freezes, (std::vector<int>{2, 2}));
    // One initial evaluation + exactly cap micro steps, in each chain.
    EXPECT_EQ(result.sample_calls, 2 * (1 + cap));
  }
}
TEST(RidgeBudget, DisabledAndUnfiredPathsRemainIdentical) {
  auto disabled = run(0, 1);
  auto unfired = run(1e300, 1);
  EXPECT_EQ(disabled.freezes, (std::vector<int>{1, 1}));
  EXPECT_EQ(disabled.sample_calls, 66);
  EXPECT_EQ(disabled.sample_calls, unfired.sample_calls);
  EXPECT_EQ(disabled.traces, unfired.traces);
}
}  // namespace
