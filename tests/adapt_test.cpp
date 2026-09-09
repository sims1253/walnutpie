#include <gtest/gtest.h>

#include <Eigen/Dense>
#include <cmath>
#include <cstddef>
#include <deque>
#include <random>
#include <stdexcept>
#include <vector>

#include <walnutpie.hpp>

namespace {

// Interrupts after a fixed number of polls so a controller that never
// converges can be observed without hanging the test.
class InterruptAfter {
 public:
  explicit InterruptAfter(std::size_t polls) : polls_(polls) {}

  void throw_if_interrupted() const {
    if (++polls_done_ >= polls_) {
      throw std::runtime_error("poll limit reached");
    }
  }

  std::size_t polls_done() const noexcept { return polls_done_; }

 private:
  std::size_t polls_;
  mutable std::size_t polls_done_ = 0;
};

class NeverInterrupt {
 public:
  void throw_if_interrupted() const {}
};

walnutpie::detail::AdaptSnapshot converged_snapshot(double log_step,
                                                    std::size_t iter) {
  walnutpie::detail::AdaptSnapshot snap(1);
  snap.iter = iter;
  snap.log_step = log_step;
  snap.log_mass = Eigen::VectorXd::Constant(1, 0.0);
  snap.mass = Eigen::VectorXd::Constant(1, 1.0);
  return snap;
}

// Two chains with identical, mutually converged snapshots well past
// min_iter and well below max_iter.
std::deque<walnutpie::detail::SpscBuffer<walnutpie::detail::AdaptSnapshot>>
converged_buffers() {
  std::deque<walnutpie::detail::SpscBuffer<walnutpie::detail::AdaptSnapshot>>
      buffers;
  buffers.emplace_back(converged_snapshot(0.0, 100));
  buffers.emplace_back(converged_snapshot(0.0, 100));
  return buffers;
}

walnutpie::InitConfig two_chain_init() {
  std::mt19937 rng(1);
  return walnutpie::InitConfigBuilder(2, 1)
      .step_sizes(0.1)
      .positions(rng, 1.0)
      .masses(Eigen::VectorXd::Ones(1))
      .build();
}

}  // namespace

TEST(WarmupConfig, EarlyExitDisabledByDefault) {
  walnutpie::WarmupConfig cfg = walnutpie::WarmupConfigBuilder().build();
  EXPECT_FALSE(cfg.allow_early_exit());
}

TEST(WarmupConfig, EarlyExitOptIn) {
  walnutpie::WarmupConfig cfg =
      walnutpie::WarmupConfigBuilder().allow_early_exit(true).build();
  EXPECT_TRUE(cfg.allow_early_exit());
}

// The cross-chain criteria hold in both tests; only the opt-in differs.
TEST(ControllerLoop, ConvergenceDoesNotStopWarmupByDefault) {
  walnutpie::InitConfig init_cfg = two_chain_init();
  walnutpie::WarmupConfig warmup_cfg =
      walnutpie::WarmupConfigBuilder().min_max_iter(50, 1000).build();
  auto buffers = converged_buffers();

  InterruptAfter interrupt(3);
  EXPECT_THROW(walnutpie::detail::controller_loop(buffers, interrupt, init_cfg,
                                                  warmup_cfg),
               std::runtime_error);
  EXPECT_GE(interrupt.polls_done(), 3);
}

TEST(ControllerLoop, OptInStopsOnConvergence) {
  walnutpie::InitConfig init_cfg = two_chain_init();
  walnutpie::WarmupConfig warmup_cfg = walnutpie::WarmupConfigBuilder()
                                           .min_max_iter(50, 1000)
                                           .allow_early_exit(true)
                                           .build();
  auto buffers = converged_buffers();

  NeverInterrupt interrupt;
  walnutpie::detail::AdaptResult result = walnutpie::detail::controller_loop(
      buffers, interrupt, init_cfg, warmup_cfg);
  EXPECT_DOUBLE_EQ(result.step_bar, 1.0);
  ASSERT_EQ(result.mass_bar.size(), Eigen::Index{1});
  EXPECT_DOUBLE_EQ(result.mass_bar(0), 1.0);
}
