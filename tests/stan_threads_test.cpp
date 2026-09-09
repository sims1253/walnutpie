#include <gtest/gtest.h>
#include <optional>
#include "stan_thread_test_util.hpp"
using walnutpie::internal::stan_threads_from_model_info;

TEST(StanThreads, RecognizesCompleteDefines) {
  EXPECT_EQ(stan_threads_from_model_info("STAN_THREADS=true"), true);
  EXPECT_EQ(stan_threads_from_model_info("STAN_THREADS=false"), false);
  EXPECT_EQ(stan_threads_from_model_info(
                "BridgeStan version: 2.9.0\nStan version: 2.39.0\nStan C++ "
                "Defines:\n\tSTAN_THREADS=true\n\tSTAN_MPI=false\nStan "
                "Compiler Details:\n\tstanc_version = stanc3 v2.39.0\n"),
            true);
  EXPECT_EQ(stan_threads_from_model_info(" \tSTAN_THREADS=false \r\n"), false);
  EXPECT_EQ(stan_threads_from_model_info(
                "OTHER_STAN_THREADS=false\nSTAN_THREADS=true\n"),
            true);
}
TEST(StanThreads, UnknownIsNotTrue) {
  EXPECT_EQ(stan_threads_from_model_info(nullptr), std::nullopt);
  for (const char* info :
       {"", "STAN_MPI=true", "OTHER_STAN_THREADS=true",
        "STAN_THREADS_extra=true", "STAN_THREADS=trueish", "STAN_THREADS=TRUE",
        "STAN_THREADS=1", "STAN_THREADS = true", "STAN_THREADS",
        "STAN_THREADS=true # comment", "STAN_THREADS=true\nSTAN_THREADS=false",
        "STAN_THREADS=false\nSTAN_THREADS=true",
        "STAN_THREADS=true\nSTAN_THREADS=true",
        "STAN_THREADS=false\nSTAN_THREADS=false",
        "STAN_THREADS=garbage\nSTAN_THREADS=true"}) {
    EXPECT_EQ(stan_threads_from_model_info(info), std::nullopt) << info;
  }
}
TEST(StanThreads, LoaderQueryDoesNotRestrictSerialUse) {
  ThreadStub stub(THREAD_STUB_PATH);
  for (const char* info :
       {"STAN_THREADS=true", "STAN_THREADS=false", "<null>", "malformed"}) {
    walnutpie::DynamicStanModel model(THREAD_STUB_PATH, info, 17);
    EXPECT_EQ(model.stan_threads(), stan_threads_from_model_info(info));
    Eigen::VectorXd x = Eigen::VectorXd::Constant(1, 0.2), g;
    double lp;
    model.logp_grad(x, lp, g);
    EXPECT_DOUBLE_EQ(lp, -0.5 * 0.2 * 0.2);
    EXPECT_DOUBLE_EQ(g[0], -0.2);
  }
}
TEST(StanThreads, MissingOptionalSymbolDoesNotBreakSerialLoader) {
  walnutpie::DynamicStanModel model(THREAD_MISSING_PATH, "{}", 17);
  EXPECT_EQ(model.stan_threads(), std::nullopt);
  Eigen::VectorXd x = Eigen::VectorXd::Zero(1), g;
  double lp;
  model.logp_grad(x, lp, g);
  EXPECT_DOUBLE_EQ(lp, 0.0);
}
