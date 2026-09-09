#include "../examples/warmup_trace.hpp"
#include <gtest/gtest.h>
#if defined(NDEBUG) || defined(EIGEN_NO_DEBUG)
#error "warmup_trace_test requires active Eigen allocation assertions"
#endif

#include <chrono>
#include <iterator>
#include <walnutpie/adaptive_walnuts.hpp>

namespace {
namespace fs = std::filesystem;
struct Temp {
  fs::path root;
  Temp() {
    static unsigned count = 0;
    root = fs::temp_directory_path() /
           ("walnutpie-trace-test-" +
            std::to_string(
                std::chrono::steady_clock::now().time_since_epoch().count()) +
            "-" + std::to_string(count++));
    if (!fs::create_directory(root)) {
      throw std::runtime_error("test temp exists");
    }
  }
  ~Temp() {
    std::error_code ec;
    fs::remove_all(root, ec);
  }
};
std::string read(const fs::path& p) {
  std::ifstream in(p, std::ios::binary);
  return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}
static_assert(!std::is_copy_constructible_v<warmup_trace::Writer>);
static_assert(!std::is_copy_assignable_v<warmup_trace::Writer>);
static_assert(!std::is_move_constructible_v<warmup_trace::Writer>);
static_assert(!std::is_move_assignable_v<warmup_trace::Writer>);

TEST(WarmupTrace, DisabledSinkHasNoVectorStorageOrEigenAllocation) {
  static_assert(sizeof(warmup_trace::Sink) == sizeof(warmup_trace::Writer*));
  warmup_trace::Sink sink;
  Eigen::VectorXd large =
      Eigen::VectorXd::Constant(4096, std::numeric_limits<double>::quiet_NaN());
  Eigen::VectorXd empty;
  Eigen::internal::set_is_malloc_allowed(false);
  sink.record(large, empty, std::numeric_limits<double>::infinity(), 0, empty,
              0);
  Eigen::internal::set_is_malloc_allowed(true);
  EXPECT_EQ(sink.writer, nullptr);
}

TEST(WarmupTrace, EigenAllocationGuardIsActuallyArmed) {
#if GTEST_HAS_DEATH_TEST
  EXPECT_DEATH(
      {
        Eigen::internal::set_is_malloc_allowed(false);
        Eigen::VectorXd forbidden(4096);
        forbidden.setZero();
      },
      "heap allocation is forbidden");
#else
  GTEST_SKIP()
      << "death test unsupported; compile-time assertion checks remain active";
#endif
}

TEST(WarmupTrace, LittleEndianGoldenBytes) {
  std::ostringstream out;
  warmup_trace::write_u64(out, 0x0102030405060708ULL);
  EXPECT_EQ(out.str(), std::string("\x08\x07\x06\x05\x04\x03\x02\x01", 8));
  for (auto bits : {0ULL, 0x3ff0000000000000ULL, 0x8000000000000000ULL,
                    0x7ff0000000000000ULL, 0xfff0000000000000ULL,
                    0x7ff8000000000042ULL, 0xffffffffffffffffULL}) {
    std::ostringstream bytes;
    warmup_trace::write_f64(bytes, std::bit_cast<double>(std::uint64_t(bits)));
    auto encoded = bytes.str();
    ASSERT_EQ(encoded.size(), 8);
    for (unsigned i = 0; i < 8; ++i) {
      EXPECT_EQ(static_cast<unsigned char>(encoded[i]),
                (bits >> (8 * i)) & 255);
    }
  }
}
TEST(WarmupTrace, JsonEscapesUtf8AndRejectsInvalidText) {
  EXPECT_EQ(warmup_trace::quote("\"\\\n\t\x01"),
            "\"\\\"\\\\\\u000a\\u0009\\u0001\"");
  EXPECT_EQ(warmup_trace::quote("caf\xc3\xa9"), "\"caf\xc3\xa9\"");
  for (auto s : {std::string("\xc0\x80"), std::string("\xed\xa0\x80"),
                 std::string("\xf4\x90\x80\x80"), std::string("\xc3")}) {
    EXPECT_THROW(warmup_trace::quote(s), std::invalid_argument);
  }
}
TEST(WarmupTrace, ShapeCountsAndExclusiveOwnership) {
  Temp t;
  auto dir = t.root / "trace";
  Eigen::VectorXd x = Eigen::VectorXd::Ones(3);
  warmup_trace::Writer writer({dir, "model", {}}, x, x, 2, 123);
  EXPECT_THROW((warmup_trace::Writer({dir, "model", {}}, x, x, 2, 123)),
               std::runtime_error);
  EXPECT_THROW(writer.record(x, Eigen::VectorXd::Zero(2), 0, .1, x, 1),
               std::invalid_argument);
  EXPECT_EQ(writer.rows(), 0);
  EXPECT_THROW(writer.finish(), std::logic_error);
  EXPECT_FALSE(fs::exists(dir / "meta.json"));
  writer.record(x, -x, -1.5, .1, x, 1);
  writer.record(2 * x, -2 * x, -6, .2, x, 2);
  EXPECT_THROW(writer.record(x, -x, 0, .1, x, 1), std::invalid_argument);
  writer.finish();
  EXPECT_EQ(fs::file_size(dir / "theta.f64"), 48);
  EXPECT_EQ(fs::file_size(dir / "grad.f64"), 48);
  EXPECT_EQ(fs::file_size(dir / "invmass.f64"), 48);
  EXPECT_EQ(fs::file_size(dir / "step.f64"), 16);
  EXPECT_EQ(fs::file_size(dir / "lp.f64"), 16);
  EXPECT_EQ(fs::file_size(dir / "depth.u64"), 16);
  EXPECT_NE(read(dir / "meta.json").find("\"warmup_iters_recorded\":2"),
            std::string::npos);
  EXPECT_FALSE(fs::exists(dir / "meta.json.tmp"));
  EXPECT_THROW(writer.finish(), std::logic_error);
}
TEST(WarmupTrace, ZeroWarmupAndInitialDimensionValidation) {
  Temp t;
  auto x = Eigen::VectorXd::Zero(2).eval();
  EXPECT_THROW((warmup_trace::Writer({t.root / "bad", "m", {}}, x,
                                     Eigen::VectorXd::Zero(1), 0, 1)),
               std::invalid_argument);
  EXPECT_FALSE(fs::exists(t.root / "bad"));
  warmup_trace::Writer writer({t.root / "zero", "m", {}}, x, x, 0, 1);
  writer.finish();
  EXPECT_EQ(fs::file_size(t.root / "zero/theta.f64"), 0);
  EXPECT_EQ(fs::file_size(t.root / "zero/depth.u64"), 0);
}
TEST(WarmupTrace, FailureDoesNotPublishManifestOrOverwriteExistingDirectory) {
  Temp t;
  auto x = Eigen::VectorXd::Ones(1).eval();
  {
    std::ofstream(t.root / "user.txt") << "keep";
  }
  EXPECT_THROW((warmup_trace::Writer({t.root, "m", {}}, x, x, 0, 1)),
               std::runtime_error);
  EXPECT_EQ(read(t.root / "user.txt"), "keep");
  warmup_trace::Writer data({t.root / "data", "m", {}}, x, x, 0, 1);
  fs::create_directory(t.root / "data/theta.f64");
  EXPECT_THROW(data.finish(), std::runtime_error);
  EXPECT_FALSE(fs::exists(t.root / "data/meta.json"));
  warmup_trace::Writer meta({t.root / "meta", "m", {}}, x, x, 0, 1);
  fs::create_directory(t.root / "meta/meta.json.tmp");
  EXPECT_THROW(meta.finish(), std::runtime_error);
  EXPECT_FALSE(fs::exists(t.root / "meta/meta.json"));
  EXPECT_THROW(
      (warmup_trace::Writer({t.root / "missing/child", "m", {}}, x, x, 0, 1)),
      fs::filesystem_error);
  // Linux /dev/full supplies a real late buffered write/close failure.
  if (fs::exists("/dev/full")) {
    warmup_trace::Writer close({t.root / "close", "m", {}}, x, x, 0, 1);
    fs::create_symlink("/dev/full", t.root / "close/meta.json.tmp");
    EXPECT_THROW(close.finish(), std::runtime_error);
    EXPECT_FALSE(fs::exists(t.root / "close/meta.json"));
  }
}
struct Comma : std::numpunct<char> {
  char do_decimal_point() const override { return ','; }
};
TEST(WarmupTrace, MetadataUsesClassicLocaleAndNullForNonfinite) {
  Temp t;
  auto x =
      Eigen::VectorXd::Constant(1, std::numeric_limits<double>::infinity());
  auto previous = std::locale();
  std::locale::global(std::locale(previous, new Comma));
  try {
    warmup_trace::Writer writer(
        {t.root / "trace",
         "quote\"\n",
         {{"finite", 1.25}, {"bad", std::numeric_limits<double>::quiet_NaN()}}},
        x, x, 0, 9);
    writer.finish();
    auto json = read(t.root / "trace/meta.json");
    EXPECT_NE(json.find("\"finite\":1.25"), std::string::npos);
    EXPECT_NE(json.find("\"bad\":null"), std::string::npos);
    EXPECT_NE(json.find("\"initial_position\":[null]"), std::string::npos);
  } catch (...) {
    std::locale::global(previous);
    throw;
  }
  std::locale::global(previous);
}
struct Gaussian {
  int* calls;
  void operator()(const Eigen::VectorXd& x, double& lp,
                  Eigen::VectorXd& g) const {
    ++*calls;
    lp = -.5 * x.squaredNorm();
    g = -x;
  }
};
struct Ordinary {
  int warmups = 0;
  const Eigen::VectorXd* theta_ref = nullptr;
  const Eigen::VectorXd* mass_ref = nullptr;
  Eigen::VectorXd last;
  double lp = 0, step = 0;
  void on_logp_exception(const Eigen::VectorXd&,
                         const std::exception&) noexcept {}
  void on_warmup(const Eigen::VectorXd& x, double l, double s,
                 const Eigen::VectorXd& m) {
    ++warmups;
    theta_ref = &x;
    mass_ref = &m;
    last = x;
    lp = l;
    step = s;
  }
  void on_warmup_complete(double, const Eigen::VectorXd&) {}
  void on_sample(const Eigen::VectorXd& x, double l) {
    last = x;
    lp = l;
  }
};
struct Observing : Ordinary {
  int notifications = 0;
  void on_warmup_trace(const Eigen::VectorXd& x, const Eigen::VectorXd& g,
                       double l, double s, const Eigen::VectorXd& m,
                       std::size_t depth) {
    ++notifications;
    EXPECT_EQ(&x, theta_ref);
    EXPECT_EQ(&m, mass_ref);
    EXPECT_TRUE((g.array() == -x.array()).all());
    EXPECT_EQ(l, lp);
    EXPECT_EQ(s, step);
    EXPECT_LE(depth, 10);
  }
};
TEST(WarmupTrace,
     OptionalNotificationUsesLiveLocalsWithoutChangingOrdinaryCallbacks) {
  int ca = 0, cb = 0;
  Gaussian fa{&ca}, fb{&cb};
  Ordinary ha;
  Observing hb;
  std::mt19937 ra(12), rb(12);
  auto warm = walnutpie::WarmupConfigBuilder().build();
  auto sampling = walnutpie::SamplingConfigBuilder().build();
  walnutpie::InitChainConfig init(.1, Eigen::VectorXd::Constant(2, .2),
                                  Eigen::VectorXd::Ones(2));
  walnutpie::AdaptiveWalnuts a(ra, ha, fa, init, warm, sampling);
  walnutpie::AdaptiveWalnuts b(rb, hb, fb, init, warm, sampling);
  for (int i = 0; i < 30; ++i) {
    a();
    b();
    EXPECT_TRUE((ha.last.array() == hb.last.array()).all());
    EXPECT_EQ(a.step_size(), b.step_size());
    EXPECT_EQ(ra, rb);
  }
  EXPECT_EQ(ha.warmups, 30);
  EXPECT_EQ(hb.warmups, 30);
  EXPECT_EQ(hb.notifications, 30);
  EXPECT_EQ(ca, cb);
}
struct MutableOnly : Ordinary {
  int mutable_calls = 0;
  void on_warmup_trace(Eigen::VectorXd&, Eigen::VectorXd&, double, double,
                       Eigen::VectorXd&, std::size_t) {
    ++mutable_calls;
  }
};
struct Overloaded : Observing {
  using Observing::on_warmup_trace;
  int mutable_calls = 0;
  void on_warmup_trace(Eigen::VectorXd&, Eigen::VectorXd&, double, double,
                       Eigen::VectorXd&, std::size_t) {
    ++mutable_calls;
  }
};
TEST(WarmupTrace, OptionalObserverNeverSelectsMutableVectorOverload) {
  int calls = 0;
  Gaussian target{&calls};
  MutableOnly mutable_only;
  Overloaded overloaded;
  std::mt19937 ra(12), rb(12);
  auto warm = walnutpie::WarmupConfigBuilder().build();
  auto sampling = walnutpie::SamplingConfigBuilder().build();
  walnutpie::InitChainConfig init(.1, Eigen::VectorXd::Constant(2, .2),
                                  Eigen::VectorXd::Ones(2));
  walnutpie::AdaptiveWalnuts a(ra, mutable_only, target, init, warm, sampling);
  walnutpie::AdaptiveWalnuts b(rb, overloaded, target, init, warm, sampling);
  for (int i = 0; i < 5; ++i) {
    a();
    b();
  }
  EXPECT_EQ(mutable_only.mutable_calls, 0);
  EXPECT_EQ(overloaded.mutable_calls, 0);
  EXPECT_EQ(overloaded.notifications, 5);
  EXPECT_TRUE((mutable_only.last.array() == overloaded.last.array()).all());
  EXPECT_EQ(ra, rb);
}

struct MutableScalars : Ordinary {
  int mutable_calls = 0;
  void on_warmup_trace(const Eigen::VectorXd&, const Eigen::VectorXd&,
                       double& lp, double step, const Eigen::VectorXd&,
                       std::size_t& depth) {
    ++mutable_calls;
    lp = 123;
    (void)step;
    depth = 789;
  }
};
struct ScalarOverloaded : Observing {
  using Observing::on_warmup_trace;
  int mutable_calls = 0;
  void on_warmup_trace(const Eigen::VectorXd&, const Eigen::VectorXd&,
                       double& lp, double step, const Eigen::VectorXd&,
                       std::size_t& depth) {
    ++mutable_calls;
    lp = 123;
    (void)step;
    depth = 789;
  }
};
TEST(WarmupTrace, OptionalObserverNeverSelectsMutableScalarOverload) {
  int calls = 0;
  Gaussian target{&calls};
  MutableScalars mutable_only;
  ScalarOverloaded overloaded;
  std::mt19937 ra(12), rb(12);
  auto warm = walnutpie::WarmupConfigBuilder().build();
  auto sampling = walnutpie::SamplingConfigBuilder().build();
  walnutpie::InitChainConfig init(.1, Eigen::VectorXd::Constant(2, .2),
                                  Eigen::VectorXd::Ones(2));
  walnutpie::AdaptiveWalnuts a(ra, mutable_only, target, init, warm, sampling);
  walnutpie::AdaptiveWalnuts b(rb, overloaded, target, init, warm, sampling);
  for (int i = 0; i < 5; ++i) {
    a();
    b();
  }
  EXPECT_EQ(mutable_only.mutable_calls, 0);
  EXPECT_EQ(overloaded.mutable_calls, 0);
  EXPECT_EQ(overloaded.notifications, 5);
  EXPECT_TRUE((mutable_only.last.array() == overloaded.last.array()).all());
  EXPECT_EQ(a.step_size(), b.step_size());
  EXPECT_EQ(ra, rb);
}

}  // namespace
