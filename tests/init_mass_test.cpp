#include <gtest/gtest.h>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <walnutpie/config.hpp>

namespace {
using walnutpie::InitConfigBuilder;
const double inf = std::numeric_limits<double>::infinity();
const double nan = std::numeric_limits<double>::quiet_NaN();
struct Failure : std::runtime_error {
  using std::runtime_error::runtime_error;
};
void expect_unchanged(InitConfigBuilder& b) {
  auto c = b.build();
  ASSERT_EQ(c.masses().size(), 2);
  for (const auto& m : c.masses()) {
    ASSERT_EQ(m.size(), 2);
    EXPECT_EQ(m[0], 7);
    EXPECT_EQ(m[1], 7);
  }
  EXPECT_EQ(c.step_sizes()[0], .25);
  EXPECT_EQ(c.position(0)[0], 1);
}
InitConfigBuilder builder() {
  auto b = InitConfigBuilder(2, 2);
  b.positions(Eigen::VectorXd::Ones(2))
      .masses(Eigen::VectorXd::Constant(2, 7))
      .step_sizes(.25);
  return b;
}
TEST(InitMassValidation, NonfiniteLogDensityLeavesAllMassesUnchanged) {
  for (bool average : {false, true}) {
    for (double bad : {nan, inf, -inf}) {
      auto b = builder();
      int calls = 0;
      auto f = [&](const Eigen::VectorXd&, double& lp, Eigen::VectorXd& g) {
        lp = ++calls == 2 ? bad : 0;
        g = Eigen::VectorXd::Constant(2, 3);
      };
      try {
        b.masses(f, .1, average);
        FAIL() << "expected rejection";
      } catch (const std::invalid_argument& e) {
        EXPECT_EQ(std::string(e.what()),
                  "initial evaluation for chain 1 log density must be finite");
      }
      EXPECT_EQ(calls, 2);
      expect_unchanged(b);
    }
  }
}
TEST(InitMassValidation, MissingLogDensityLeavesMassesUnchanged) {
  for (bool average : {false, true}) {
    auto b = builder();
    int calls = 0;
    auto f = [&](const Eigen::VectorXd&, double& lp, Eigen::VectorXd& g) {
      if (++calls == 1) {
        lp = 0;
      }
      g = Eigen::VectorXd::Constant(2, 3);
    };
    try {
      b.masses(f, .1, average);
      FAIL() << "expected rejection";
    } catch (const std::invalid_argument& e) {
      EXPECT_EQ(std::string(e.what()),
                "initial evaluation for chain 1 log density must be finite");
    }
    EXPECT_EQ(calls, 2);
    expect_unchanged(b);
  }
}

TEST(InitMassValidation, NonfiniteGradientReportsChainAndCoordinate) {
  for (bool average : {false, true}) {
    for (double bad : {nan, inf, -inf}) {
      for (int index : {0, 1}) {
        auto b = builder();
        int calls = 0;
        auto f = [&](const Eigen::VectorXd&, double& lp, Eigen::VectorXd& g) {
          lp = 0;
          g = Eigen::VectorXd::Constant(2, 3);
          if (++calls == 2) {
            g[index] = bad;
          }
        };
        try {
          b.masses(f, .1, average);
          FAIL() << "expected rejection";
        } catch (const std::invalid_argument& e) {
          EXPECT_EQ(std::string(e.what()),
                    "initial evaluation for chain 1 gradient[" +
                        std::to_string(index) + "] must be finite");
        }
        EXPECT_EQ(calls, 2);
        expect_unchanged(b);
      }
    }
  }
}
TEST(InitMassValidation, WrongGradientSizeLeavesMassesUnchanged) {
  for (bool average : {false, true}) {
    for (int size : {0, 1, 3}) {
      auto b = builder();
      int calls = 0;
      auto f = [&](const Eigen::VectorXd&, double& lp, Eigen::VectorXd& g) {
        lp = 0;
        g = Eigen::VectorXd::Ones(++calls == 2 ? size : 2);
      };
      try {
        b.masses(f, .1, average);
        FAIL() << "expected rejection";
      } catch (const std::invalid_argument& e) {
        EXPECT_EQ(
            std::string(e.what()),
            "initial evaluation for chain 1 gradient size must match dims");
      }
      EXPECT_EQ(calls, 2);
      expect_unchanged(b);
    }
  }
}
TEST(InitMassValidation, CallbackExceptionPropagatesAndBuilderCanRetry) {
  for (bool average : {false, true}) {
    auto b = builder();
    int calls = 0;
    auto f = [&](const Eigen::VectorXd&, double& lp, Eigen::VectorXd& g) {
      if (++calls == 2) {
        throw Failure("sentinel");
      }
      lp = 0;
      g = Eigen::VectorXd::Constant(2, 3);
    };
    try {
      b.masses(f, .1, average);
      FAIL() << "expected exception";
    } catch (const Failure& e) {
      EXPECT_EQ(std::string(e.what()), "sentinel");
    }
    EXPECT_EQ(calls, 2);
    // Copy to inspect preserved state without consuming the retryable builder.
    auto copy = b;
    expect_unchanged(copy);
    auto good = [](const Eigen::VectorXd&, double& lp, Eigen::VectorXd& g) {
      lp = 0;
      g = Eigen::VectorXd::Zero(2);
    };
    auto c = b.masses(good, .1, average).build();
    for (const auto& m : c.masses()) {
      for (int i = 0; i < 2; ++i) {
        EXPECT_DOUBLE_EQ(m[i], average ? std::exp(std::log(.1)) : .1);
      }
    }
  }
}
TEST(InitMassValidation, HealthyArithmeticAndZeroGradientsAreUnchanged) {
  for (bool average : {false, true}) {
    for (double lp_value : {-100., 0., 100.}) {
      auto b = builder();
      int calls = 0;
      auto f = [&](const Eigen::VectorXd&, double& lp, Eigen::VectorXd& g) {
        lp = lp_value;
        g.resize(2);
        g << 0, ++calls == 1 ? -3 : 5;
      };
      auto c = b.masses(f, .1, average).build();
      Eigen::VectorXd a(2), z(2);
      a << 0, -3;
      z << 0, 5;
      Eigen::VectorXd ma = .9 * a.array().abs() + .1,
                      mz = .9 * z.array().abs() + .1;
      if (average) {
        Eigen::VectorXd sum = Eigen::VectorXd::Zero(2);
        sum += ma.array().log().matrix();
        sum += mz.array().log().matrix();
        ma = (sum / 2).array().exp().matrix();
        mz = ma;
      }
      EXPECT_TRUE((c.masses()[0].array() == ma.array()).all());
      EXPECT_TRUE((c.masses()[1].array() == mz.array()).all());
      EXPECT_EQ(calls, 2);
    }
  }
}
TEST(InitMassValidation, InvalidSmoothingDoesNotCallTargetOrMutateBuilder) {
  for (double s : {0., 1., -1., nan, inf}) {
    auto b = builder();
    int calls = 0;
    auto f = [&](const Eigen::VectorXd&, double& lp, Eigen::VectorXd& g) {
      ++calls;
      lp = 0;
      g = Eigen::VectorXd::Zero(2);
    };
    EXPECT_THROW(b.masses(f, s), std::invalid_argument);
    EXPECT_EQ(calls, 0);
    expect_unchanged(b);
  }
}
TEST(InitMassValidation, ExplicitMassAndDefaultPathsRemainAvailable) {
  auto defaults = InitConfigBuilder(1, 2).build();
  EXPECT_TRUE(defaults.masses()[0].isOnes());
  auto c =
      InitConfigBuilder(1, 2).masses(Eigen::VectorXd::Constant(2, 2)).build();
  EXPECT_EQ(c.masses()[0][0], 2);
}
}  // namespace
