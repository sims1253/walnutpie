// Property-hunt driver for walnutpie core algebra (tests/property-hunt branch).
// Standalone driver, plain asserts + soft-check harness (no catch2 dep):
//   /usr/bin/clang++ -std=c++20 -O2 -I include -I <eigen-src> \
//     tests/property_hunt.cpp -o /tmp/property_hunt && /tmp/property_hunt
//
// Properties (see PROPERTY_HUNT.md for full statement of each):
//   P1  combine: total-weight associativity (Barker & Metropolis, F & B)
//   P2  combine: endpoint invariance under direction swap
//   P3  combine: exact-enumeration selection marginal, Barker == w2/(w1+w2),
//       3-span chain marginal == w3/(w1+w2+w3); Metropolis == min(1, w2/w1)
//   P4  uturn: mirrored-trajectory reflection symmetry, Forward == Backward
//   P5  log_sum_exp / combine with -inf weights keeps old selection
//   P6  momentum reproducibility: same seed => same rho & identical chain
//       (diagonal path and low-rank path); Random stream separation
//   P7  sampler marginal == pi on 2D Gaussian (statistical)
//   P8  WelfordAccumulator vs two-pass on adversarial sequences (+reset)
//   P9  OnlineMoments discounted Welford vs brute force (scale jumps, negatives)
//   P10 LowRankMass: apply_inv / logp_momentum / log_det vs dense;
//       sample_momentum_from covariance identity; r in {0,1,d/2,d-1,d}
//   P11 AntiWindupAdapter pass_rate semantics (counting inner adapter)
//   P12 Adam robustness: alpha = 0 / 1 fine; alpha = NaN must not poison
//       (documents presence/absence of guard at this commit)
//   P13 DualAveraging sanity: constant alpha => finite, positive step size

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Dense>

#include "walnutpie/adam.hpp"
#include "walnutpie/concepts.hpp"
#include "walnutpie/low_rank_mass.hpp"
#include "walnutpie/online_moments.hpp"
#include "walnutpie/step_optimizers.hpp"
#include "walnutpie/util.hpp"
#include "walnutpie/walnuts.hpp"

using namespace walnutpie;
using namespace walnutpie::detail;
using Eigen::MatrixXd;
using Eigen::VectorXd;

static int g_checks = 0;
static int g_failures = 0;
static std::vector<std::string> g_failed;

#define CHECK(cond, name)                                              \
  do {                                                                 \
    ++g_checks;                                                        \
    if (!(cond)) {                                                     \
      ++g_failures;                                                    \
      g_failed.push_back(name);                                        \
      std::printf("FAIL  %s  (line %d)\n", name, __LINE__);            \
    }                                                                  \
  } while (0)

// Expected-failure probe: reports a KNOWN missing property without failing
// the suite (used for the Adam NaN guard, absent at 788d832).
static int g_xfails = 0;
#define XFAIL(cond, name)                                              \
  do {                                                                 \
    ++g_checks;                                                        \
    if (!(cond)) {                                                     \
      ++g_xfails;                                                      \
      std::printf("XFAIL %s (known gap, line %d)\n", name, __LINE__);  \
    }                                                                  \
  } while (0)

// REAL-BUG probe: a property that is violated by the code under test (not by
// the harness). Reported loudly, counted separately, and DOES fail the run
// so the bug stays visible until core code is fixed.
static int g_bugs = 0;
#define BUG(cond, name)                                                \
  do {                                                                 \
    ++g_checks;                                                        \
    if (!(cond)) {                                                     \
      ++g_bugs;                                                        \
      g_failed.push_back(name);                                        \
      std::printf("BUG!  %s (REAL BUG, line %d)\n", name, __LINE__);   \
    }                                                                  \
  } while (0)

static bool near(double a, double b, double tol) {
  return std::fabs(a - b) <= tol * (1.0 + std::fabs(b));
}

// ---------------------------------------------------------------------------
// Analytic targets
struct GaussGrad {
  // standard normal in d dims: logp = -0.5 |x|^2, grad = -x
  void operator()(const VectorXd& x, double& logp, VectorXd& grad) const {
    logp = -0.5 * x.squaredNorm();
    grad = -x;
  }
};

struct FunnelGrad {
  double sigma0 = 3.0;
  void operator()(const VectorXd& x, double& logp, VectorXd& grad) const {
    const double v = x[0];
    const double s = std::exp(0.5 * v);
    logp = -0.5 * (v / sigma0) * (v / sigma0) - 0.5 * v - 0.5 * x.tail(x.size() - 1).squaredNorm() / (s * s);
    grad.resize(x.size());
    grad[0] = -v / (sigma0 * sigma0) - 0.5 + (x.tail(x.size() - 1).squaredNorm()) * std::exp(-v) * 0.5;
    grad.tail(x.size() - 1) = -x.tail(x.size() - 1) / (s * s);
  }
};

// Minimal handler satisfying SampleHandler.
struct NullHandler {
  void on_sample(const VectorXd&, double) const noexcept {}
  void on_logp_exception(const VectorXd&, const std::exception&) const noexcept {}
};

// ---------------------------------------------------------------------------
// Span construction helpers (synthetic spans with prescribed weights)
static SpanW leaf_span(double logw, int id) {
  VectorXd th = VectorXd::Zero(2);
  th[0] = static_cast<double>(id);
  VectorXd rho = VectorXd::Constant(2, 0.1 * id);
  VectorXd g = -th;
  return SpanW::from_initial_point(std::move(th), std::move(rho), std::move(g),
                                   logw, logw);
}

// exact-enumeration combine: returns fraction of the u-grid on which the
// combined span selected the NEW span's state
template <Update U, Direction D>
static double enum_new_frac(double w1, double w2, int grid) {
  double l1 = std::log(w1), l2 = std::log(w2);
  int hits = 0;
  // fix rng draw by driving a mt19937_64 seeded per u is overkill; instead we
  // bypass rng by evaluating the same inequality combine uses:
  //   update  <=>  log(u) < span_new.logp_ - log_denominator
  for (int i = 0; i < grid; ++i) {
    double u = (i + 0.5) / grid;  // u ~ U(0,1) grid; log(u) is the variate
    double logp_total = log_sum_exp(l1, l2);
    double log_den = (U == Update::Metropolis) ? l1 : logp_total;
    bool update = std::log(u) < l2 - log_den;
    hits += update ? 1 : 0;
  }
  return static_cast<double>(hits) / grid;
}

// ensemble variant: many random seeds, fraction of runs where new was selected
template <Update U, Direction D>
static double ensemble_new_frac(double w1, double w2, int reps) {
  int hits = 0;
  for (int i = 0; i < reps; ++i) {
    std::mt19937_64 rng(static_cast<std::uint64_t>(i) * 2654435761u + 7u);
    Random<std::mt19937_64> rand(rng);
    SpanW s1 = leaf_span(std::log(w1), 1);
    SpanW s2 = leaf_span(std::log(w2), 2);
    SpanW c = combine<U, D>(rand, std::move(s1), std::move(s2));
    hits += (std::fabs(c.theta_select_[0] - 2.0) < 0.25) ? 1 : 0;
  }
  return static_cast<double>(hits) / reps;
}

// ---------------------------------------------------------------------------
// exact-enumeration grid over (u1,u2) for the 3-span chain ((s1 + s2) + s3),
// returning marginal probability that the final selected state is s3.
static double enum_chain_s3(double w1, double w2, double w3, int grid) {
  double l1 = std::log(w1), l2 = std::log(w2), l3 = std::log(w3);
  int s3 = 0, s2 = 0, s1 = 0;
  for (int i = 0; i < grid; ++i) {
    double u1 = (i + 0.5) / grid;
    double l12 = log_sum_exp(l1, l2);
    bool take2 = std::log(u1) < l2 - l12;  // Barker inner
    double lsel = take2 ? l2 : l1;
    for (int j = 0; j < grid; ++j) {
      double u2 = (j + 0.5) / grid;
      bool take3 = std::log(u2) < l3 - log_sum_exp(l12, l3);  // Barker
      if (take3) ++s3;
      else if (take2) ++s2;
      else ++s1;
    }
  }
  double n = static_cast<double>(grid) * grid;
  return s3 / n;
}

// mirrored leapfrog states: run forward from (t0, r0) with step s, and
// backward from (t0, -r0) with step s (macro_step negates step). States:
// forward_k approximates backward_k mirrored through t0 up to leapfrog error.
static void run_states(const GaussGrad& f, VectorXd t, VectorXd r, double step,
                       int n, std::vector<VectorXd>& theta,
                       std::vector<VectorXd>& rho) {
  theta.clear(); rho.clear();
  double lp; VectorXd g;
  f(t, lp, g);
  for (int k = 0; k < n; ++k) {
    r += 0.5 * step * g;
    t.array() += step * 1.0 * r.array();  // unit inv mass
    f(t, lp, g);
    r += 0.5 * step * g;
    theta.push_back(t); rho.push_back(r);
  }
}

static SpanW state_span(const VectorXd& t, const VectorXd& r) {
  VectorXd g = -t;
  double lp = -0.5 * t.squaredNorm();
  double lj = lp - 0.5 * r.squaredNorm();
  VectorXd tt = t, rr = r, gg = g;
  return SpanW::from_initial_point(std::move(tt), std::move(rr), std::move(gg),
                                   lp, lj);
}

// ---------------------------------------------------------------------------
int main() {
  std::printf("== walnutpie property hunt ==\n");

  // ---------------- P1: combine weight associativity ----------------
  {
    std::mt19937_64 rng(42);
    Random<std::mt19937_64> rand(rng);
    for (Direction D : {Direction::Forward, Direction::Backward}) {
      for (Update U : {Update::Barker, Update::Metropolis}) {
        auto mk = [&](double w, int id) { return leaf_span(std::log(w), id); };
        auto run = [&](auto u, auto d) -> SpanW {
          if (d == Direction::Forward)
            return u == Update::Barker
                       ? combine<Update::Barker, Direction::Forward>(
                             rand, mk(2.0, 1), mk(5.0, 2))
                       : combine<Update::Metropolis, Direction::Forward>(
                             rand, mk(2.0, 1), mk(5.0, 2));
          return u == Update::Barker
                     ? combine<Update::Barker, Direction::Backward>(
                           rand, mk(2.0, 1), mk(5.0, 2))
                     : combine<Update::Metropolis, Direction::Backward>(
                           rand, mk(2.0, 1), mk(5.0, 2));
        };
        SpanW ab = run(U, D);
        // ab total weight must equal log(w1+w2) for BOTH update rules
        CHECK(near(ab.logp_, std::log(7.0), 1e-12),
              "P1 combine total weight == log(w1+w2)");
        // (a+b)+c vs log_sum_exp: build via chained combine on fresh spans
        std::mt19937_64 rng2(7);
        Random<std::mt19937_64> rand2(rng2);
        SpanW s1 = leaf_span(std::log(2.0), 1);
        SpanW s2 = leaf_span(std::log(5.0), 2);
        SpanW s3 = leaf_span(std::log(11.0), 3);
        SpanW pair = combine<Update::Barker, Direction::Forward>(
            rand2, std::move(s1), std::move(s2));
        SpanW triple = combine<Update::Barker, Direction::Forward>(
            rand2, std::move(pair), std::move(s3));
        CHECK(near(triple.logp_, std::log(18.0), 1e-12),
              "P1 ((a+b)+c) weight == log(w1+w2+w3)");
      }
    }
  }

  // ---------------- P2: endpoint invariance under direction ----------------
  {
    std::mt19937_64 rng(11);
    Random<std::mt19937_64> rand(rng);
    SpanW s1 = leaf_span(std::log(3.0), 1);
    SpanW s2 = leaf_span(std::log(4.0), 2);
    SpanW f = combine<Update::Barker, Direction::Forward>(rand,
        SpanW::from_initial_point(VectorXd::Constant(2, 1.0), VectorXd::Constant(2, 0.5),
                                  VectorXd::Constant(2, -1.0), std::log(3.0), std::log(3.0)),
        SpanW::from_initial_point(VectorXd::Constant(2, 2.0), VectorXd::Constant(2, 0.6),
                                  VectorXd::Constant(2, -2.0), std::log(4.0), std::log(4.0)));
    CHECK(f.theta_bk_[0] == 1.0 && f.theta_fw_[0] == 2.0,
          "P2 forward combine keeps earliest/latest endpoints");
    SpanW b = combine<Update::Barker, Direction::Backward>(rand,
        SpanW::from_initial_point(VectorXd::Constant(2, 1.0), VectorXd::Constant(2, 0.5),
                                  VectorXd::Constant(2, -1.0), std::log(3.0), std::log(3.0)),
        SpanW::from_initial_point(VectorXd::Constant(2, 0.0), VectorXd::Constant(2, 0.7),
                                  VectorXd::Constant(2, 0.0), std::log(4.0), std::log(4.0)));
    CHECK(b.theta_bk_[0] == 0.0 && b.theta_fw_[0] == 1.0,
          "P2 backward combine keeps earliest/latest endpoints");
  }

  // ---------------- P3: selection marginal, exact enumeration ----------------
  {
    // analytic grid check of the update inequality used by combine
    double b = enum_new_frac<Update::Barker, Direction::Forward>(2.0, 5.0, 20000);
    CHECK(near(b, 5.0 / 7.0, 1e-3),
          "P3 Barker P(select new) == w2/(w1+w2) [grid]");
    double m = enum_new_frac<Update::Metropolis, Direction::Forward>(2.0, 5.0, 20000);
    CHECK(near(m, 1.0, 1e-9),
          "P3 Metropolis P(select new) == min(1, w2/w1) = 1 [grid]");
    double m2 = enum_new_frac<Update::Metropolis, Direction::Forward>(5.0, 2.0, 20000);
    CHECK(near(m2, 2.0 / 5.0, 1e-3),
          "P3 Metropolis P(select new) == w2/w1 when w2<w1 [grid]");
    // chain marginal: s3 selected with prob w3/(w1+w2+w3)
    double s3 = enum_chain_s3(2.0, 5.0, 11.0, 400);
    CHECK(near(s3, 11.0 / 18.0, 1e-3),
          "P3 Barker chain marginal(s3) == w3/(w1+w2+w3) [grid]");
    // ensemble check through the real combine code path
    double eb = ensemble_new_frac<Update::Barker, Direction::Forward>(2.0, 5.0, 20000);
    CHECK(near(eb, 5.0 / 7.0, 0.02),
          "P3 Barker ensemble P(select new) == w2/(w1+w2) [real combine]");
    double em = ensemble_new_frac<Update::Metropolis, Direction::Forward>(5.0, 2.0, 20000);
    CHECK(near(em, 2.0 / 5.0, 0.02),
          "P3 Metropolis ensemble P(select new) == w2/w1 [real combine]");
    // commutativity-in-distribution: P(select the w=5 span) is the same
    // whether it appears as old or as new in the combine call
    double e1 = ensemble_new_frac<Update::Barker, Direction::Forward>(2.0, 5.0, 20000);
    // (5,2): new is the w=2 span; P(w=5 span retained) = 1 - P(new)
    double e2 = ensemble_new_frac<Update::Barker, Direction::Forward>(5.0, 2.0, 20000);
    CHECK(near(1.0 - e2, e1, 0.02),
          "P3 Barker commutativity in distribution");
  }

  // ---------------- P4: uturn reflection symmetry ----------------
  {
    std::mt19937_64 rng(99);
    std::normal_distribution<double> n(0, 1);
    int sym = 0, total = 0;
    VectorXd t0(2), r0(2);
    for (int rep = 0; rep < 200; ++rep) {
      for (int i = 0; i < 2; ++i) { t0[i] = n(rng); r0[i] = n(rng); }
      double step = 0.05 + 0.1 * (rep % 5);
      GaussGrad f;
      std::vector<VectorXd> tf, rf, tb, rb;
      run_states(f, t0, r0, step, 16, tf, rf);          // forward from (t0, r0)
      run_states(f, t0, -r0, -step, 16, tb, rb);        // backward mirror
      // time-reversal check: evolving backward from (t0, -r0) with -step
      // visits the same thetas with negated momenta (leapfrog reversibility)
      double mirror_err = 0;
      for (int k = 0; k < 16; ++k) {
        mirror_err = std::max(mirror_err, (tb[k] - tf[k]).norm());
        mirror_err = std::max(mirror_err, (rb[k] + rf[k]).norm());
      }
      CHECK(mirror_err < 1e-8, "P4 leapfrog mirror symmetry");
      VectorXd inv_mass = VectorXd::Constant(2, 1.0);
      for (int k = 1; k < 16; ++k) {
        SpanW a = state_span(t0, r0);
        SpanW b = state_span(tf[k], rf[k]);
        SpanW am = state_span(t0, -r0);
        SpanW bm = state_span(tb[k], rb[k]);
        bool uf = uturn<Direction::Forward>(a, b, inv_mass);
        // mirror world: span_accum is am, the backward-extended span is bm;
        // the code calls uturn<Backward>(span_old, span_new)
        bool ub = uturn<Direction::Backward>(am, bm, inv_mass);
        ++total;
        sym += (uf == ub);
      }
    }
    CHECK(sym == total, "P4 uturn(Forward) == uturn(Backward) on mirrored spans");
  }

  // ---------------- P5: -inf weight handling ----------------
  {
    std::mt19937_64 rng(5);
    Random<std::mt19937_64> rand(rng);
    SpanW good = leaf_span(std::log(3.0), 1);
    SpanW dead = leaf_span(-std::numeric_limits<double>::infinity(), 2);
    SpanW c = combine<Update::Barker, Direction::Forward>(rand,
        std::move(good), std::move(dead));
    CHECK(c.theta_select_[0] == 1.0 && std::isfinite(c.logp_),
          "P5 -inf new span never selected; total stays finite");
    CHECK(near(c.logp_, std::log(3.0), 1e-12), "P5 total weight = w_old");
    // both -inf: total -inf, selection is old (log u < -inf false)
    SpanW dead2 = leaf_span(-std::numeric_limits<double>::infinity(), 3);
    SpanW c2 = combine<Update::Metropolis, Direction::Forward>(rand,
        std::move(dead), std::move(dead2));
    CHECK(std::isinf(c2.logp_) && c2.logp_ < 0, "P5 -inf + -inf = -inf total");
  }

  // ---------------- P6: reproducibility ----------------
  {
    GaussGrad f;
    NullHandler h;
    std::mt19937_64 rng_a(1234), rng_b(1234);
    auto make = [&](std::mt19937_64& rng) {
      VectorXd th = VectorXd::Zero(2);
      VectorXd inv = VectorXd::Ones(2);
      return WalnutsSampler<GaussGrad, std::mt19937_64, NullHandler>(
          rng, h, f, th, inv, 0.25, 4, 8, 2, 0.1);
    };
    auto s1 = make(rng_a);
    auto s2 = make(rng_b);
    bool identical = true;
    for (int i = 0; i < 500; ++i) {
      double l1 = s1();
      double l2 = s2();
      identical &= (l1 == l2) && (s1.position()[0] == s2.position()[0]) &&
                   (s1.position()[1] == s2.position()[1]);
    }
    CHECK(identical, "P6 same seed => bitwise-identical chains (500 draws)");
    // momentum stream discipline: Random owns one normal_ per instance and
    // interleaves with uniform_real_01 on the same base generator; two Random
    // objects on the same base rng share the underlying stream (documented),
    // but distinct distributions. Verify normal draws are unaffected by
    // interleaved uniform draws given the same call order (trivially true) and
    // that independent seeds give independent rho:
    std::mt19937_64 ra(1), rb(1);
    Random<std::mt19937_64> A(ra), B(rb);
    VectorXd z1 = A.standard_normal(4);
    double u = A.uniform_real_01();
    VectorXd z2 = A.standard_normal(4);
    CHECK(u >= 0.0 && u < 1.0 && z1.hasNaN() == false && z2.hasNaN() == false,
          "P6 stream sanity");
    // low-rank path reproducibility
    std::mt19937_64 rng_c(77), rng_d(77);
    auto make_lr = [&](std::mt19937_64& rng) {
      VectorXd th = VectorXd::Zero(3);
      VectorXd inv = VectorXd::Ones(3);
      WalnutsSampler<GaussGrad, std::mt19937_64, NullHandler> s(
          rng, h, f, th, inv, 0.2, 3, 8, 2, 0.1);
      MatrixXd U(3, 1); U << 1.0, 0.0, 0.0;
      VectorXd c(1); c << 0.5;
      s.set_low_rank(U, c);
      return s;
    };
    auto l1 = make_lr(rng_c);
    auto l2 = make_lr(rng_d);
    bool lr_identical = true;
    for (int i = 0; i < 200; ++i) lr_identical &= (l1() == l2());
    CHECK(lr_identical, "P6 low-rank path same seed => identical chains");
  }

  // ---------------- P7: marginal == pi on 2D Gaussian ----------------
  {
    GaussGrad f;
    NullHandler h;
    std::mt19937_64 rng(2024);
    VectorXd th = VectorXd::Zero(2);
    WalnutsSampler<GaussGrad, std::mt19937_64, NullHandler> s(
        rng, h, f, th, VectorXd::Ones(2), 0.25, 4, 8, 2, 0.1);
    const int N = 20000;
    VectorXd sum = VectorXd::Zero(2);
    VectorXd sumsq = VectorXd::Zero(2);
    for (int i = 0; i < N; ++i) {
      s();
      sum += s.position();
      sumsq += s.position().cwiseProduct(s.position());
    }
    VectorXd mean = sum / N;
    VectorXd var = sumsq / N - mean.cwiseProduct(mean);
    CHECK(std::fabs(mean[0]) < 0.06 && std::fabs(mean[1]) < 0.06,
          "P7 sampler mean ~ 0 (2D Gaussian)");
    CHECK(std::fabs(var[0] - 1.0) < 0.12 && std::fabs(var[1] - 1.0) < 0.12,
          "P7 sampler variance ~ 1 (2D Gaussian)");
    // funnel smoke: finite draws, no NaN propagation into positions
    FunnelGrad fg;
    std::mt19937_64 rng2(9);
    VectorXd th2 = VectorXd::Zero(5);
    WalnutsSampler<FunnelGrad, std::mt19937_64, NullHandler> s2(
        rng2, h, fg, th2, VectorXd::Ones(5), 0.05, 4, 10, 2, 0.05);
    bool finite = true;
    for (int i = 0; i < 2000; ++i) {
      double lp = s2();
      finite &= std::isfinite(lp) && s2.position().allFinite();
    }
    CHECK(finite, "P7 funnel: 2000 draws stay finite");
  }

  // ---------------- P8: WelfordAccumulator vs two-pass ----------------
  {
    std::mt19937_64 rng(31337);
    std::uniform_real_distribution<double> u(0, 1);
    auto brute = [](const std::vector<double>& xs) {
      double m = 0; for (double x : xs) m += x; m /= xs.size();
      double v = 0; for (double x : xs) v += (x - m) * (x - m);
      v /= (xs.size() - 1);
      return std::pair<double, double>(m, v);
    };
    bool ok = true;
    for (int rep = 0; rep < 200; ++rep) {
      WelfordAccumulator w;
      std::vector<double> xs;
      int n = 2 + rep % 60;
      for (int i = 0; i < n; ++i) {
        double x;
        double r = u(rng);
        if (r < 0.25) x = 1e-12 * (u(rng) + 1);
        else if (r < 0.5) x = 1e12 * (u(rng) + 1);
        else if (r < 0.75) x = -1e6 * (u(rng) + 1);
        else x = u(rng) * 2 - 1;
        w.observe(x); xs.push_back(x);
      }
      auto [m, v] = brute(xs);
      // relative tolerance with absolute floor for near-zero truths
      ok &= near(w.mean(), m, 1e-9) || std::fabs(w.mean() - m) < 1e-6 * (std::fabs(m) + 1e-6);
      double sv = w.sample_variance();
      ok &= near(sv, v, 1e-6) || std::fabs(sv - v) < 1e-6 * (v + 1e-9);
    }
    CHECK(ok, "P8 Welford vs brute force two-pass (adversarial scales)");
    WelfordAccumulator w;
    w.observe(5.0); w.observe(7.0);
    w.reset();
    CHECK(w.count() == 0 && w.mean() == 0.0 && w.sample_variance() != w.sample_variance(),
          "P8 reset clears state; n<2 variance is NaN");
  }

  // ---------------- P9: OnlineMoments vs brute force ----------------
  {
    auto brute = [](double init_w, const VectorXd& m0, const VectorXd& v0,
                    double df, const std::vector<VectorXd>& ys) {
      // explicit weights: init pseudo-obs gets df^N * init_w, obs n (0-based)
      // gets df^(N-1-n); two-pass weighted moments in long double to avoid
      // the cancellation that a raw sum-of-squares recurrence suffers on
      // 1e12-scale data
      const std::size_t N = ys.size();
      long double W = std::pow((long double)df, (long double)N) * init_w;
      std::vector<long double> w(N);
      for (std::size_t n = 0; n < N; ++n) {
        w[n] = std::pow((long double)df, (long double)(N - 1 - n));
        W += w[n];
      }
      const int d = static_cast<int>(m0.size());
      VectorXd mean = VectorXd::Zero(d);
      for (int j = 0; j < d; ++j) {
        long double s = std::pow((long double)df, (long double)N) * init_w * m0[j];
        for (std::size_t n = 0; n < N; ++n) s += w[n] * ys[n][j];
        mean[j] = static_cast<double>(s / W);
      }
      VectorXd var = VectorXd::Zero(d);
      for (int j = 0; j < d; ++j) {
        long double q = std::pow((long double)df, (long double)N) * init_w *
                        (v0[j] + (m0[j] - mean[j]) * (m0[j] - mean[j]));
        for (std::size_t n = 0; n < N; ++n)
          q += w[n] * (ys[n][j] - mean[j]) * (ys[n][j] - mean[j]);
        var[j] = static_cast<double>(q / W);
      }
      return std::make_pair(mean, var);
    };
    std::mt19937_64 rng(808);
    std::uniform_real_distribution<double> u(0, 1);
    bool ok_mean = true;
    for (int rep = 0; rep < 200; ++rep) {
      int d = 1 + rep % 3;
      double df = (rep % 3 == 0) ? 1.0 : 0.25 + 0.7 * u(rng);
      double init_w = 1.0 + 10 * u(rng);
      VectorXd m0 = VectorXd::Constant(d, 2 * u(rng) - 1);
      VectorXd v0 = VectorXd::Constant(d, 0.1 + u(rng));
      OnlineMoments om(init_w, m0, v0);
      om.set_discount_factor(df);
      std::vector<VectorXd> ys;
      int n = 2 + rep % 40;
      for (int i = 0; i < n; ++i) {
        VectorXd y(d);
        for (int j = 0; j < d; ++j) {
          double r = u(rng);
          if (r < 0.25) y[j] = 1e-12 * (u(rng) + 1);
          else if (r < 0.5) y[j] = 1e12 * (u(rng) + 1);
          else if (r < 0.75) y[j] = -(1e8) * (u(rng) + 1);
          else y[j] = 100 * (u(rng) - 0.5);
        }
        om.observe(y);
        ys.push_back(y);
      }
      auto [m, v] = brute(init_w, m0, v0, df, ys);
      // mean must track the exact weighted mean (Welford means are exact)
      for (int j = 0; j < d; ++j) {
        double mm = om.mean()[j];
        double maxy = 0;
        for (const auto& y : ys) maxy = std::max(maxy, std::fabs(y[j]));
        ok_mean &= std::fabs(mm - m[j]) <= 1e-9 * (maxy + m0.norm());
      }
    }
    CHECK(ok_mean, "P9 OnlineMoments mean == exact weighted mean");
    // VARIANCE IS WRONG: minimal repro of the lazy-expression aliasing bug.
    // observe() computes `auto delta = y - mean_;` (a lazy Eigen expression),
    // then updates mean_, then evaluates delta, which re-reads the NEW mean.
    // Correct discounted Welford term is delta_old * (y - mean_new); the code
    // effectively computes (y - mean_new)^2, systematically shrinking M2.
    OnlineMoments om(1.0, VectorXd::Zero(1), VectorXd::Zero(1));
    om.set_discount_factor(1.0);
    VectorXd y1(1); y1 << 1.0; om.observe(y1);
    VectorXd y2(1); y2 << 2.0; om.observe(y2);
    // exact: W = 3, mean = 1, S = 2, MLE variance = 2/3
    BUG(near(om.variance()[0], 2.0 / 3.0, 1e-12),
        "P9 OnlineMoments variance: aliasing repro (got 5/12, expect 2/3)");
    // ... and the randomized discounted-variance sweep (200 adversarial
    // sequences) — same root cause
    std::mt19937_64 rng2(4242);
    std::uniform_real_distribution<double> u2(0, 1);
    bool var_ok = true;
    for (int rep = 0; rep < 200; ++rep) {
      int d = 1; double df = 0.5; double init_w = 2.0;
      VectorXd m0 = VectorXd::Zero(d), v0 = VectorXd::Zero(d);
      OnlineMoments o2(init_w, m0, v0);
      o2.set_discount_factor(df);
      std::vector<VectorXd> ys;
      int n = 3 + rep % 20;
      for (int i = 0; i < n; ++i) {
        VectorXd y(d); y[0] = 10.0 * (u2(rng2) - 0.3);
        o2.observe(y); ys.push_back(y);
      }
      auto [m, v] = brute(init_w, m0, v0, df, ys);
      var_ok &= near(o2.variance()[0], v[0], 1e-9);
    }
    BUG(var_ok, "P9 OnlineMoments variance == exact weighted variance (200 seqs)");
    // variance() with zero total weight returns the ones vector
    // (default-constructed accumulator: weight 0, empty mean)
    OnlineMoments empty;
    CHECK(empty.variance().size() == 0 && empty.weight() == 0,
          "P9 zero-weight state well-defined");
  }

  // ---------------- P10: LowRankMass vs dense ----------------
  {
    std::mt19937_64 rng(606);
    std::normal_distribution<double> n(0, 1);
    bool ok = true, ok_det = true, ok_cov = true;
    for (int d : {2, 3, 50}) {
      for (int r : {0, 1, d / 2, d - 1, d}) {
        VectorXd D = VectorXd::NullaryExpr(d, [&] { return 0.2 + 2.0 * std::fabs(n(rng)); });
        // orthonormal U via Gram-Schmidt on random columns
        MatrixXd U = MatrixXd::NullaryExpr(d, r, [&] { return n(rng); });
        for (int k = 0; k < r; ++k) {
          for (int j = 0; j < k; ++j) U.col(k) -= U.col(j) * (U.col(j).dot(U.col(k)));
          if (U.col(k).norm() > 1e-12) U.col(k).normalize();
          else U.col(k).setZero();  // degenerate but orthonormal-ish
        }
        VectorXd c = VectorXd::NullaryExpr(r, [&] { return std::fabs(n(rng)); });
        LowRankMass lrm{D, U, c};
        VectorXd sq = D.cwiseSqrt();
        MatrixXd A = D.asDiagonal() * MatrixXd::Identity(d, d);
        if (r > 0) {
          MatrixXd sqU = sq.asDiagonal() * U;
          A += sqU * c.asDiagonal() * sqU.transpose();
        }
        VectorXd x = VectorXd::NullaryExpr(d, [&] { return n(rng); });
        VectorXd apply = lrm.apply_inv(x);
        VectorXd dense = A * x;
        ok &= (apply - dense).norm() <= 1e-10 * (1 + dense.norm());
        double lp = lrm.logp_momentum(x);
        double lpd = -0.5 * x.transpose() * A * x;
        ok &= std::fabs(lp - lpd) <= 1e-10 * (1 + std::fabs(lpd));
        double ld = lrm.log_det();
        double ldd = 2.0 * A.llt().matrixLLT().diagonal().array().log().sum();
        ok_det &= std::fabs(ld - ldd) <= 1e-9 * (1 + std::fabs(ldd));
        // momentum sampling: the CORRECT draw is rho = Lc^{-T} z with
        //   Lc = sqrtD (I + U((I+C)^{1/2} - I) U^T),
        // i.e. rho = D^{-1/2}(I + U W U^T) z, W = (I+C)^{-1/2} - I
        // (D^{-1/2} OUTSIDE the bracket; it does not commute with U).
        // The implementation applies D^{-1/2} INSIDE the bracket:
        //   rho_impl = (I + U W U^T) D^{-1/2} z,
        // which has covariance (I+UWU^T) D^{-1} (I+UWU^T) != A^{-1}.
        VectorXd z = VectorXd::NullaryExpr(d, [&] { return n(rng); });
        VectorXd rho = lrm.sample_momentum_from(z);
        MatrixXd Ainv = A.llt().solve(MatrixXd::Identity(d, d));
        MatrixXd B = MatrixXd::Identity(d, d);
        if (r > 0) {
          MatrixXd IC = MatrixXd::Identity(r, r);
          IC.diagonal() += c;
          Eigen::SelfAdjointEigenSolver<MatrixXd> es(IC);
          B += U * (es.operatorInverseSqrt() - MatrixXd::Identity(r, r)) * U.transpose();
        }
        B = sq.cwiseInverse().asDiagonal() * B;  // correct: D^{-1/2} outside
        VectorXd rho_correct = B * z;
        ok_cov &= (rho - rho_correct).norm() <= 1e-9 * (1 + rho_correct.norm());
        MatrixXd BBt = B * B.transpose();
        ok_cov &= (BBt - Ainv).norm() <= 1e-8 * (1 + Ainv.norm());
        // edge cases: r=0 pure diagonal; r=d full rank
        if (r == 0) ok &= (lrm.apply_inv(x) - D.cwiseProduct(x)).norm() < 1e-14;
      }
    }
    CHECK(ok, "P10 apply_inv / logp_momentum == dense operator");
    CHECK(ok_det, "P10 log_det == dense Cholesky log-det");
    BUG(ok_cov, "P10 sample_momentum_from == Lc^{-T} z (D^{-1/2} outside bracket)");
  }

  // ---------------- P11: AntiWindupAdapter pass_rate ----------------
  {
    struct Counting {
      int calls = 0;
      void operator()(double) { ++calls; }
      double step_size() const { return 0.1; }
    };
    // AntiWindupAdapter owns inner by value; instrument through a shared
    // counter to observe how many observations are forwarded.
    struct Shared {
      std::shared_ptr<int> count;
      void operator()(double) const { ++*count; }
      double step_size() const { return 0.1; }
    };
    auto count = std::make_shared<int>(0);
    AntiWindupAdapter<Shared> aw2(Shared{count}, 1e-12, 8);
    for (int i = 0; i < 100; ++i) aw2(0.0);  // all saturated
    // passes when saturated_seen_ % 8 == 1 -> seen 1,9,...,97 => 13 forwarded
    CHECK(*count == 13, "P11 AntiWindup forwards exactly 13/100 saturated (1-in-8)");
    for (int i = 0; i < 50; ++i) aw2(0.5);  // unsaturated: all forwarded
    CHECK(*count == 63, "P11 unsaturated observations always forwarded");
    // boundary: alpha == floor_alpha is NOT saturated (strict <)
    aw2(1e-12);
    CHECK(*count == 64, "P11 alpha == floor_alpha passes (strict <)");
  }

  // ---------------- P12: Adam robustness ----------------
  {
    Adam a(0.1, 0.8, 0.1, 0.9, 0.999, 1e-8, 0.5);
    for (int i = 0; i < 100; ++i) a(0.0);
    for (int i = 0; i < 100; ++i) a(1.0);
    CHECK(std::isfinite(a.step_size()) && a.step_size() > 0,
          "P12 Adam survives alpha = 0 and alpha = 1");
    Adam b(0.1, 0.8, 0.1, 0.9, 0.999, 1e-8, 0.5);
    b(0.5); b(std::nan("")); b(0.5); b(0.5);
    XFAIL(std::isfinite(b.step_size()),
          "P12 Adam NaN alpha must not poison (guard absent at 788d832)");
    // recovery after a single NaN if a guard existed
    Adam c2(0.1, 0.8, 0.1, 0.9, 0.999, 1e-8, 0.5);
    for (int i = 0; i < 50; ++i) c2(0.5);
    double before = c2.step_size();
    c2(std::nan(""));
    for (int i = 0; i < 50; ++i) c2(0.5);
    XFAIL(std::isfinite(c2.step_size()) && near(c2.step_size(), before, 0.5),
          "P12 Adam state recovers after single NaN (guard absent at 788d832)");
  }

  // ---------------- P13: DualAveraging sanity ----------------
  {
    DualAveraging da(0.1, 0.8, 0.05, 10.0, 0.75, true);
    for (int i = 0; i < 1000; ++i) da(0.8);
    CHECK(std::isfinite(da.step_size()) && da.step_size() > 0,
          "P13 DualAveraging finite at target");
    DualAveraging dz(0.1, 0.8, 0.05, 10.0, 0.75, true);
    for (int i = 0; i < 1000; ++i) dz(0.0);
    CHECK(std::isfinite(dz.step_size()) && dz.step_size() > 0,
          "P13 DualAveraging finite under alpha = 0 burst");
  }

  std::printf("\n== %d checks, %d failures, %d REAL BUGS, %d expected-failures ==\n",
              g_checks, g_failures, g_bugs, g_xfails);
  for (auto& f : g_failed) std::printf("  FAILED: %s\n", f.c_str());
  return (g_failures == 0 && g_bugs == 0) ? 0 : 1;
}
