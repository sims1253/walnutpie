#include <gtest/gtest.h>
#include <algorithm>
#include <array>
#include <atomic>
#include <vector>
#include "../python/src/walnutpie/errors.hpp"
#include "stan_thread_test_util.hpp"
using PRINT_CALLBACK = void (*)(const char*, std::size_t, bool);
extern "C" int walnutpie_sample_bridgestan(
    const char* bs_dll, const char* json_data, STREAM_CALLBACK callback,
    unsigned int model_seed, const char* inits, size_t num_chains,
    unsigned int seed, unsigned int id, double init_radius,
    const double* init_inv_metric, int min_warmup_iter, int max_warmup_iter,
    int min_sampling_iter, int max_sampling_iter, int max_trajectory_doublings,
    int max_step_halvings, int min_micro_steps, double max_hamiltonian_error,
    double step_size_converge_tol, double mass_converge_tol,
    double rhat_converge_tol, double mass_init_count,
    double mass_additive_smoothing, double max_macro_steps_target,
    double step_size_init, double step_accept_rate_target,
    double step_learning_rate, double step_gradient_decay,
    double step_sq_gradient_decay, double step_stabilization,
    double step_learn_rate_decay, bool save_warmup, double* out,
    size_t out_size, int* final_lengths, double* stepsize_out,
    double* inv_metric_out, int refresh, PRINT_CALLBACK print,
    WalnutpyError** err);
extern "C" void walnutpie_destroy_error(WalnutpyError*);
namespace {
std::atomic<int> print_calls{0};
void print_callback(const char*, std::size_t, bool) { ++print_calls; }
struct Result {
  std::vector<double> draws;
  std::vector<int> lengths;
  int code;
  WalnutpyError* error = nullptr;
  explicit Result(std::size_t chains)
      : draws(chains * 4, 12345), lengths(chains * 2, -999) {}
  ~Result() {
    if (error) {
      walnutpie_destroy_error(error);
    }
  }
};
void run(const char* path, const char* info, std::size_t chains,
         Result& result) {
  std::vector<double> mass(chains, 1.0);
  result.code = walnutpie_sample_bridgestan(
      path, info, nullptr, 17, nullptr, chains, 17, 1, 1.0, mass.data(), 1, 1,
      4, 4, 2, 2, 1, 1.0, 0.05, 0.05, 1.05, 10.0, 0.1, 8.0, 0.1, 0.8, 0.05, 0.8,
      0.9, 1e-4, 0.5, false, result.draws.data(), result.draws.size(),
      result.lengths.data(), nullptr, nullptr, 0, print_callback,
      &result.error);
}
void expect_rejection(const char* path, const char* info, std::size_t chains) {
  ThreadStub stub(path);
  stub.reset();
  print_calls = 0;
  Result result(chains);
  run(path, info, chains, result);
  ASSERT_EQ(result.code, -1);
  ASSERT_NE(result.error, nullptr);
  EXPECT_EQ(result.error->type, WalnutpyErrorType::config);
  EXPECT_NE(result.error->msg.find("STAN_THREADS=true"), std::string::npos);
  EXPECT_NE(result.error->msg.find(path), std::string::npos);
  EXPECT_EQ(stub.calls("stub_gradient_calls"), 0);
  EXPECT_EQ(stub.calls("stub_init_calls"), 0);
  EXPECT_EQ(stub.calls("stub_constrain_calls"), 0);
  EXPECT_EQ(print_calls.load(), 0);
  EXPECT_TRUE(std::all_of(result.draws.begin(), result.draws.end(),
                          [](double v) { return v == 12345; }));
  EXPECT_TRUE(std::all_of(result.lengths.begin(), result.lengths.end(),
                          [](int v) { return v == -999; }));
}
TEST(StanBindingThreads, RejectsFalseAndUnknownBeforeWork) {
  for (std::size_t chains : {1u, 2u}) {
    for (const char* info :
         {"STAN_THREADS=false", "<null>", "", "malformed",
          "STAN_THREADS=trueish", "STAN_THREADS=true\nSTAN_THREADS=false"}) {
      expect_rejection(THREAD_STUB_PATH, info, chains);
    }
    expect_rejection(THREAD_MISSING_PATH, "{}", chains);
  }
}
TEST(StanBindingThreads, AcceptsKnownTrue) {
  ThreadStub stub(THREAD_STUB_PATH);
  stub.reset();
  Result result(2);
  run(THREAD_STUB_PATH, "Stan C++ Defines:\n\tSTAN_THREADS=true\n", 2, result);
  ASSERT_EQ(result.code, 0);
  EXPECT_EQ(result.error, nullptr);
  EXPECT_GT(stub.calls("stub_gradient_calls"), 0);
  EXPECT_EQ(stub.calls("stub_init_calls"), 2);
  EXPECT_EQ(stub.calls("stub_constrain_calls"), 8);
  for (double v : result.draws) {
    EXPECT_TRUE(std::isfinite(v));
  }
  EXPECT_EQ(result.lengths, (std::vector<int>{0, 0, 4, 4}));
}
}  // namespace
