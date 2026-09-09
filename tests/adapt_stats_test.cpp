#include <gtest/gtest.h>
#include <bit>
#include <cstdint>
#include <limits>
#include <random>
#include <walnutpie/adapt.hpp>
#include <walnutpie/adaptive_walnuts.hpp>

namespace {
using namespace walnutpie;
using namespace walnutpie::detail;
struct Interrupt {
  void throw_if_interrupted() const {}
};
std::vector<AdaptSnapshot> snapshots(
    const std::vector<std::vector<double>>& x) {
  std::vector<AdaptSnapshot> result;
  for (const auto& row : x) {
    AdaptSnapshot s(row.size());
    for (std::size_t j = 0; j < row.size(); ++j) {
      s.log_mass[j] = row[j];
    }
    s.mass = s.log_mass.array().exp();
    s.log_step = 0.0;
    s.iter = 8;
    result.push_back(s);
  }
  return result;
}
// Independent pairwise identity: sample variance = sum_{i<j}(xi-xj)^2/M/(M-1).
double scalar_reference(const std::vector<AdaptSnapshot>& s) {
  long double total = 0;
  for (Eigen::Index d = 0; d < s[0].log_mass.size(); ++d) {
    for (std::size_t i = 0; i < s.size(); ++i) {
      for (std::size_t j = i + 1; j < s.size(); ++j) {
        long double delta =
            static_cast<long double>(s[i].log_mass[d]) - s[j].log_mass[d];
        total += delta * delta;
      }
    }
  }
  return static_cast<double>(total / s.size() / (s.size() - 1) /
                             s[0].log_mass.size());
}
TEST(AdaptStats, SingleFiniteChainIsZero) {
  EXPECT_DOUBLE_EQ(log_mass_dispersion(snapshots({{2, -3, 7}})), 0.0);
}
TEST(AdaptStats, TwoChainsAndCoordinatesHaveSampleVariance) {
  EXPECT_DOUBLE_EQ(log_mass_dispersion(snapshots({{0, 0}, {2, 4}})), 5.0);
}
TEST(AdaptStats, MultipleChainsMatchIndependentScalarReference) {
  for (auto s : {snapshots({{1, 4, -3}, {7, -2, 11}, {5, 8, 6}}),
                 snapshots({{1e12 + 1}, {1e12 + 2}, {1e12 + 4}, {1e12 + 8}})}) {
    EXPECT_NEAR(log_mass_dispersion(s), scalar_reference(s), 1e-12);
  }
  EXPECT_DOUBLE_EQ(log_mass_dispersion(snapshots({{3, 7}, {3, 7}, {3, 7}})),
                   0.0);
}
TEST(AdaptStats, InvalidShapesThrow) {
  EXPECT_THROW(log_mass_dispersion({}), std::invalid_argument);
  EXPECT_THROW(log_mass_dispersion(snapshots({{}})), std::invalid_argument);
  EXPECT_THROW(log_mass_dispersion(snapshots({{1}, {1, 2}})),
               std::invalid_argument);
}
TEST(AdaptStats, NonfiniteAndOverflowAreUnavailable) {
  for (double x : {std::numeric_limits<double>::quiet_NaN(),
                   std::numeric_limits<double>::infinity()}) {
    EXPECT_TRUE(std::isnan(log_mass_dispersion(snapshots({{x}}))));
    EXPECT_TRUE(std::isnan(log_mass_dispersion(snapshots({{1}, {x}}))));
  }
  EXPECT_TRUE(std::isnan(log_mass_dispersion(snapshots({{-1e308}, {1e308}}))));
}
TEST(AdaptStats, ControllerExitValuesUnchanged) {
  auto init = InitConfigBuilder(3, 2).build();
  auto warm = WarmupConfigBuilder().min_max_iter(8, 8).build();
  auto s = snapshots({{0, 1}, {2, 3}, {4, 5}});
  auto buffers = construct_buffers(3, 2);
  for (std::size_t m = 0; m < s.size(); ++m) {
    s[m].log_step = static_cast<double>(m) - 1;
    buffers[m].write_buffer() = s[m];
    buffers[m].publish();
  }
  auto legacy = controller_loop<Interrupt>(buffers, Interrupt{}, init, warm);
  auto observed =
      controller_loop<Interrupt, true>(buffers, Interrupt{}, init, warm);
  EXPECT_EQ(legacy.mass_bar, observed.mass_bar);
  EXPECT_EQ(legacy.step_bar, observed.step_bar);
  EXPECT_TRUE(std::isnan(legacy.log_mass_dispersion));
  EXPECT_DOUBLE_EQ(observed.log_mass_dispersion, 4.0);
}
TEST(AdaptStats, ConvergedExitDoesNotWaitForMaximumBudget) {
  auto init = InitConfigBuilder(2, 2).build();
  auto warm = WarmupConfigBuilder().min_max_iter(8, 64).build();
  auto s = snapshots({{0, 1}, {0, 1}});
  auto buffers = construct_buffers(2, 2);
  for (std::size_t m = 0; m < s.size(); ++m) {
    buffers[m].write_buffer() = s[m];
    buffers[m].publish();
  }
  struct MustNotPoll {
    void throw_if_interrupted() const {
      throw std::runtime_error("unexpected poll");
    }
  };
  auto legacy = controller_loop(buffers, MustNotPoll{}, init, warm);
  auto observed =
      controller_loop<MustNotPoll, true>(buffers, MustNotPoll{}, init, warm);
  EXPECT_EQ(legacy.mass_bar, observed.mass_bar);
  EXPECT_EQ(legacy.step_bar, observed.step_bar);
  EXPECT_DOUBLE_EQ(observed.log_mass_dispersion, 0.0);
}

TEST(AdaptStats, ControllerRejectsBadSnapshotDimensions) {
  auto init = InitConfigBuilder(1, 2).build();
  auto warm = WarmupConfigBuilder().min_max_iter(8, 8).build();
  auto buffers = construct_buffers(1, 1);
  EXPECT_THROW(
      (controller_loop<Interrupt, true>(buffers, Interrupt{}, init, warm)),
      std::invalid_argument);
  auto empty = construct_buffers(0, 0);
  auto empty_init = InitConfigBuilder(0, 0).build();
  EXPECT_THROW(
      (controller_loop<Interrupt, true>(empty, Interrupt{}, empty_init, warm)),
      std::invalid_argument);
}
struct Logp {
  void operator()(const Eigen::VectorXd& x, double& lp,
                  Eigen::VectorXd& grad) const {
    lp = -.5 * x.squaredNorm();
    grad = -x;
  }
};
struct Trace {
  std::vector<std::uint64_t> values;
  void add(double x) { values.push_back(std::bit_cast<std::uint64_t>(x)); }
  void add(const Eigen::VectorXd& x) {
    for (double v : x) {
      add(v);
    }
  }
  void on_sample(const Eigen::VectorXd& x, double lp) {
    add(x);
    add(lp);
  }
  void on_warmup(const Eigen::VectorXd& x, double lp, double step,
                 const Eigen::VectorXd& mass) {
    add(x);
    add(lp);
    add(step);
    add(mass);
  }
  void on_warmup_complete(double step, const Eigen::VectorXd& mass) {
    add(step);
    add(mass);
  }
  void on_logp_exception(const Eigen::VectorXd&,
                         const std::exception&) noexcept {}
};
using Adapter = AdaptiveWalnuts<Logp, std::mt19937, Trace>;
std::vector<std::vector<std::uint64_t>> run(bool observe, std::size_t chains) {
  Logp f;
  std::vector<Trace> handlers(chains);
  std::vector<std::mt19937> rngs;
  rngs.reserve(chains);
  for (std::size_t m = 0; m < chains; ++m) {
    rngs.emplace_back(42 + m);
  }
  auto init = InitConfigBuilder(chains, 2).build();
  auto warm =
      WarmupConfigBuilder().min_max_iter(16, 16).publish_stride(1).build();
  auto sample = SamplingConfigBuilder().build();
  std::vector<Adapter> adapters;
  adapters.reserve(chains);
  for (std::size_t m = 0; m < chains; ++m) {
    adapters.emplace_back(rngs[m], handlers[m], f, init.init_chain_config(m),
                          warm, sample);
  }
  if (observe) {
    auto stats = adapt_with_stats(init, warm, adapters, Interrupt{});
    std::vector<AdaptSnapshot> final;
    for (const auto& a : adapters) {
      AdaptSnapshot s(2);
      s.log_mass = a.log_mass();
      final.push_back(s);
    }
    EXPECT_DOUBLE_EQ(stats.log_mass_dispersion, log_mass_dispersion(final));
  } else {
    adapt(init, warm, adapters, Interrupt{});
  }
  std::vector<std::vector<std::uint64_t>> traces;
  for (std::size_t m = 0; m < chains; ++m) {
    EXPECT_EQ(adapters[m].iter(), 16);
    auto frozen = adapters[m].sampler();
    for (int n = 0; n < 12; ++n) {
      frozen();
    }
    traces.push_back(handlers[m].values);
  }
  return traces;
}
TEST(AdaptStats, FixedBudgetWarmupAndSamplingBitsUnchanged) {
  for (std::size_t chains : {1, 2, 4}) {
    EXPECT_EQ(run(false, chains), run(true, chains));
  }
}
TEST(AdaptStats, NewEntryRejectsEmptyOrMismatchedAdapters) {
  std::vector<Adapter> none;
  auto warm = WarmupConfigBuilder().build();
  EXPECT_THROW(adapt_with_stats(InitConfigBuilder(0, 0).build(), warm, none,
                                Interrupt{}),
               std::invalid_argument);
  EXPECT_THROW(adapt_with_stats(InitConfigBuilder(1, 2).build(), warm, none,
                                Interrupt{}),
               std::invalid_argument);
  Logp f;
  Trace h;
  std::mt19937 rng(17);
  auto sample = SamplingConfigBuilder().build();
  auto one = InitConfigBuilder(1, 1).build();
  none.emplace_back(rng, h, f, one.init_chain_config(0), warm, sample);
  EXPECT_THROW(adapt_with_stats(InitConfigBuilder(1, 2).build(), warm, none,
                                Interrupt{}),
               std::invalid_argument);
  EXPECT_THROW(adapt_with_stats(InitConfigBuilder(1, 0).build(), warm, none,
                                Interrupt{}),
               std::invalid_argument);
}
}  // namespace
