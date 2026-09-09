#define main walnutpie_stan_cli_main_for_test
#include "../examples/stan_cli.cpp"
#undef main
#include <gtest/gtest.h>
#include <memory>

struct FixtureModel {
  bool pinned = false;
  mutable std::vector<std::shared_ptr<std::mt19937_64>> gq_rngs;
  std::size_t unconstrained_dimensions() const { return 2; }
  std::size_t constrained_dimensions() const { return 5; }
  auto make_rng(unsigned int seed) const {
    auto rng = std::make_shared<std::mt19937_64>(seed);
    gq_rngs.push_back(rng);
    return rng;
  }
  std::vector<std::string> param_names() const {
    return {"fixed", "x", "y", "gq1", "gq2"};
  }
  void logp_grad(const Eigen::VectorXd& x, double& lp, Eigen::VectorXd& grad) {
    grad = -x;
    lp = pinned && x.squaredNorm() != 0
             ? -std::numeric_limits<double>::infinity()
             : -0.5 * x.squaredNorm();
  }
  template <typename Out, typename RNG>
  void constrain_draw(const Eigen::VectorXd& x, Out&& out, RNG& rng) {
    out[0] = 1;
    out[1] = x[0];
    out[2] = x[1];
    out[3] = static_cast<double>((*rng)());
    out[4] = static_cast<double>((*rng)());
  }
};

static auto fixture_run(FixtureModel& model, walnutpie::SamplingConfig& cfg,
                        const MicroGuardSpec& guard, MicroGuardResult* result,
                        bool save_warmup = false,
                        std::mt19937_64* final_rng = nullptr) {
  auto init = walnutpie::InitConfigBuilder(1, 2)
                  .positions(Eigen::VectorXd::Zero(2))
                  .step_sizes(0.1);
  auto warmup = walnutpie::WarmupConfigBuilder().build();
  return run_walnuts(model, 123, init, 3, 8, save_warmup, warmup, cfg, guard,
                     result, final_rng);
}
TEST(MicroGuard, FullPositionAndNonfinite) {
  MicroGuardProbe probe({true, 3, 2});
  Eigen::VectorXd p = Eigen::VectorXd::Zero(3);
  probe.observe(p);
  p[2] = 1;
  probe.observe(p);
  probe.observe(p);
  EXPECT_EQ(probe.result().unique, 2);
  EXPECT_FALSE(probe.result().pinned);
  MicroGuardProbe pin({true, 2, 2});
  p.setZero();
  pin.observe(p);
  pin.observe(p);
  EXPECT_TRUE(pin.result().pinned);
  MicroGuardProbe bad({true, 2, 2});
  p[0] = NAN;
  EXPECT_THROW(bad.observe(p), std::runtime_error);
  EXPECT_THROW(bad.observe(Eigen::VectorXd()), std::runtime_error);
  MicroGuardProbe inert({});
  EXPECT_NO_THROW(inert.observe(p));
}
TEST(MicroGuard, Validation) {
  EXPECT_THROW(validate_micro_guard({true, 0, 1}, 8), std::invalid_argument);
  EXPECT_THROW(validate_micro_guard({true, 3, 0}, 8), std::invalid_argument);
  EXPECT_THROW(validate_micro_guard({true, 3, 4}, 8), std::invalid_argument);
  EXPECT_THROW(validate_micro_guard({true, 9, 2}, 8), std::invalid_argument);
  EXPECT_NO_THROW(validate_micro_guard({false, 9, 2}, 8));
}
TEST(MicroGuard, SilentGuardIsExactIncludingGQ) {
  FixtureModel model;
  auto mm2 = walnutpie::SamplingConfigBuilder().min_micro_steps(2).build();
  for (bool save : {false, true}) {
    MicroGuardResult result;
    std::mt19937_64 baseline_rng, guarded_rng;
    auto baseline = fixture_run(model, mm2, {}, nullptr, save, &baseline_rng);
    auto baseline_gq = *model.gq_rngs.back();
    auto guarded =
        fixture_run(model, mm2, {true, 3, 1}, &result, save, &guarded_rng);
    EXPECT_EQ(baseline_rng, guarded_rng);
    EXPECT_EQ(baseline_gq, *model.gq_rngs.back());
    EXPECT_FALSE(result.pinned);
    EXPECT_EQ(baseline.draws(), guarded.draws());
    EXPECT_EQ(baseline.size(), save ? 11 : 8);
  }
}
TEST(MicroGuard, ForcedRestartEqualsMM1AndDoesNotRetryPinnedFallback) {
  FixtureModel model;
  model.pinned = true;
  auto mm1 = walnutpie::SamplingConfigBuilder().min_micro_steps(1).build();
  auto mm2 = walnutpie::SamplingConfigBuilder().min_micro_steps(2).build();
  for (bool save : {false, true}) {
    std::mt19937_64 baseline_rng, retry_rng;
    auto baseline = fixture_run(model, mm1, {}, nullptr, save, &baseline_rng);
    auto baseline_gq = *model.gq_rngs.back();
    int calls = 0, triggers = 0;
    auto run = [&](auto& cfg, const auto& guard, MicroGuardResult* result) {
      ++calls;
      if (calls == 2) {
        EXPECT_FALSE(guard.armed);
        EXPECT_EQ(cfg.min_micro_steps(), 1);
      }
      return fixture_run(model, cfg, guard, result, save, &retry_rng);
    };
    auto recovered =
        run_with_micro_guard(run, mm2, mm1, {true, 3, 2}, [&](auto result) {
          ++triggers;
          EXPECT_EQ(result.unique, 1);
          EXPECT_EQ(result.probe, 3);
        });
    EXPECT_EQ(calls, 2);
    EXPECT_EQ(triggers, 1);
    EXPECT_EQ(baseline_rng, retry_rng);
    EXPECT_EQ(baseline_gq, *model.gq_rngs.back());
    EXPECT_EQ(recovered.size(), save ? 11 : 8);
    EXPECT_EQ(recovered.draws(), baseline.draws());
  }
}
TEST(MicroGuard, ForcedDecisionOnMovingSamplerResetsTrajectory) {
  FixtureModel model;
  auto mm1 = walnutpie::SamplingConfigBuilder().min_micro_steps(1).build();
  auto mm2 = walnutpie::SamplingConfigBuilder().min_micro_steps(2).build();
  auto baseline = fixture_run(model, mm1, {}, nullptr, true);
  int calls = 0;
  auto run = [&](auto& cfg, const auto& guard, MicroGuardResult* result) {
    ++calls;
    auto draws = fixture_run(model, cfg, guard, result, true);
    if (calls == 1) {
      result->pinned = true;  // test-only decision injection, no CLI flag
    }
    return draws;
  };
  auto recovered =
      run_with_micro_guard(run, mm2, mm1, {true, 3, 1}, [](auto) {});
  EXPECT_EQ(calls, 2);
  EXPECT_EQ(recovered.draws(), baseline.draws());
}

TEST(MicroGuard, IncompleteAttemptCannotBeWritten) {
  FixtureModel model;
  StanHandler<FixtureModel> storage(model, 123, 3, 8, false);
  storage.on_sample(Eigen::VectorXd::Zero(2), 0);
  std::string output = "micro_guard_incomplete_test.csv";
  EXPECT_THROW(storage.summarize(), std::logic_error);
  EXPECT_THROW(storage.write_csv(output), std::logic_error);
}
