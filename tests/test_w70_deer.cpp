// W-70 DEER/Picard within-trajectory parallelism feasibility study.
// Pre-registration: /home/m0hawk/Documents/apin/stan/WORKLOG.md, W-70 entry
// (2026-08-25). Measurement only — NO sampler changes anywhere.
//
// Phase A per model: 100 warmup iterations (documented scheme below), then
// record the FULL micro-step trajectory of 200 sampling transitions
// (positions + gradients + step + inv_mass). Then an OFFLINE Picard (DEER)
// replay per transition, as pure post-processing on the recorded data:
//   iterate round r: evaluate gradients at ALL current iterate positions
//   t=1..S (the parallelizable batch; g_0 is exact), then rebuild the
//   trajectory from x_0 with those cached gradients via the leapfrog
//   position/momentum updates.
//   Round 0 initialized by linear momentum extrapolation from x_0
//   (WALNUTS' extrapolation trick; does not peek at the sequential truth).
//
// Warmup scheme (fixed, documented; plain leapfrog + DualAveraging — no
// walnutpie adaptation machinery touched):
//   1) init position from BridgeStan param_initialize (radius 2),
//      identity diagonal metric;
//   2) find_reasonable_step (walnutpie::detail, W-43-corrected);
//   3) phase 1: iters 1..50, transitions of S micro steps, DualAveraging
//      (Stan defaults: target 0.8, gamma .05, t0 10, kappa .75) observing
//      mean over micro steps of min(1, exp(-dH_step));
//   4) at iter 51: diagonal metric = inverse regularized variance of the
//      positions of iters 26..50 (Stan-style shrinkage); find_reasonable_step
//      again under the new metric;
//   5) phase 2: iters 51..100 with a fresh DualAveraging;
//   6) freeze step = step_size_bar() and inv_mass for all sampling.

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <optional>
#include <random>
#include <map>
#include <string>
#include <vector>

#include "load_stan.hpp"
#include "step_optimizers.hpp"
#include "util.hpp"
#include "warmup_heuristics.hpp"

using walnutpie::DynamicStanModel;

namespace {

constexpr std::size_t kWarmupIters = 100;
constexpr std::size_t kMetricStartIdx = 25;  // 0-based: collect iters 26..50
constexpr std::size_t kSamplingIters = 200;
constexpr std::size_t kMaxPicardRounds = 300;
constexpr double kTolStrict = 1e-12;  // abs inf-norm vs sequential truth
constexpr double kTolRelaxed = 1e-10; // relative fallback
constexpr double kMaxError = 0.5;  // walnutpie default max_hamiltonian_error

struct Transition {
  double step;
  Eigen::VectorXd inv_mass;
  Eigen::VectorXd rho0;
  std::vector<Eigen::VectorXd> positions;  // [0]=x_0 .. [S]
  std::vector<Eigen::VectorXd> grads;      // grads[t] = grad AT positions[t]
};

double kinetic(const Eigen::VectorXd& rho, const Eigen::VectorXd& inv_mass) {
  return 0.5 * rho.cwiseProduct(inv_mass).dot(rho);
}

struct PicardResult {
  bool converged_strict = false;
  bool converged_relaxed = false;
  bool diverged = false;
  std::size_t rounds = 0;
  std::size_t grad_evals = 0;  // batch evals (t=1..S per round)
  double final_abs_err = 0.0;
};

// Picard/DEER replay against recorded sequential trajectory `seq`.
template <typename F>
PicardResult picard_replay(F& logp_grad, const Transition& seq) {
  const std::size_t S = seq.positions.size() - 1;
  const Eigen::Index d = seq.positions[0].size();
  static int* replay_dbg = new int(0);
  const int replay_trans = (*replay_dbg)++;
  const double scale =
      std::max(1.0, [&] {
        double m = 0.0;
        for (const auto& p : seq.positions) m = std::max(m, p.cwiseAbs().maxCoeff());
        return m;
      }());

  // Round-0 iterate: linear momentum extrapolation from endpoints
  // x_t^(0) = x_0 + t * eps * inv_mass .* rho_0  (WALNUTS extrapolation trick).
  std::vector<Eigen::VectorXd> it_new(S + 1), it_old(S + 1);
  for (std::size_t t = 0; t <= S; ++t) {
    it_old[t] = seq.positions[0] +
                static_cast<double>(t) * seq.step *
                    seq.inv_mass.cwiseProduct(seq.rho0);
  }

  PicardResult res;
  Eigen::VectorXd g0;
  double lp_dummy;
  logp_grad(seq.positions[0], lp_dummy, g0);  // exact boundary gradient

  if (getenv("W70_DEBUG") && *replay_dbg < 3) {
    double e0 = 0.0;
    for (std::size_t t = 1; t <= S; ++t)
      e0 = std::max(e0, (it_old[t] - seq.positions[t]).cwiseAbs().maxCoeff());
    std::cerr << "# replay t" << replay_trans << " r=0 init_err=" << e0
              << "\n";
  }
  for (std::size_t r = 1; r <= kMaxPicardRounds; ++r) {
    // Parallelizable batch: gradients at all current iterate positions.
    std::vector<Eigen::VectorXd> g(S + 1);
    g[0] = g0;
    for (std::size_t t = 1; t <= S; ++t) {
      double lp;
      logp_grad(it_old[t], lp, g[t]);
      ++res.grad_evals;
      if (!g[t].allFinite()) {
        res.diverged = true;
        res.rounds = r;
        return res;
      }
    }
    // Sequential rebuild from x_0 using cached gradient field {g}.
    const double half = 0.5 * seq.step;
    Eigen::VectorXd rho = seq.rho0;
    it_new[0] = seq.positions[0];
    for (std::size_t t = 1; t <= S; ++t) {
      rho += half * g[t - 1];
      it_new[t] = it_new[t - 1] + seq.step * seq.inv_mass.cwiseProduct(rho);
      rho += half * g[t];
    }
    // Convergence check vs sequential truth.
    double max_abs = 0.0;
    for (std::size_t t = 1; t <= S; ++t) {
      max_abs = std::max(max_abs,
                         (it_new[t] - seq.positions[t]).cwiseAbs().maxCoeff());
    }
    it_old.swap(it_new);
    res.rounds = r;
    if (getenv("W70_DEBUG") && *replay_dbg < 3) {
      std::cerr << "# replay t" << replay_trans << " r=" << r
                << " max_abs=" << max_abs << "\n";
      if (r == 1)
        std::cerr << "#   scale=" << scale << " step=" << seq.step
                  << " |rho0|=" << seq.rho0.norm() << "\n";
    }
    if (max_abs < kTolStrict) {
      res.converged_strict = true;
      res.converged_relaxed = true;
      res.final_abs_err = max_abs;
      return res;
    }
    if (!it_old[S].allFinite()) {
      res.diverged = true;
      return res;
    }
    res.final_abs_err = max_abs;
    if (r == kMaxPicardRounds && max_abs <= kTolRelaxed * scale) {
      res.converged_relaxed = true;
    }
  }
  return res;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 3) {
    std::cerr << "usage: test_w70_deer <model.so> <data.json> [seed] [S] "
                 "[dump.bin]\n";
    return 2;
  }
  const std::string model_path = argv[1];
  const std::string data_path = argv[2];
  unsigned int seed = argc > 3 ? static_cast<unsigned int>(std::atoi(argv[3])) : 70u;
  std::size_t S = argc > 4 ? std::strtoull(argv[4], nullptr, 10) : 32;
  std::string dump_path = argc > 5 ? argv[5] : "";

  DynamicStanModel model(model_path.c_str(), data_path.c_str(), seed);
  const Eigen::Index d = static_cast<Eigen::Index>(model.unconstrained_dimensions());
  std::mt19937_64 mt(seed * 9176 + 13);
  walnutpie::detail::Random<std::mt19937_64> rand(mt);

  auto logp_grad = [&](const Eigen::VectorXd& x, double& lp, Eigen::VectorXd& g) {
    model.logp_grad(x, lp, g);
  };

  Eigen::VectorXd inv_mass = Eigen::VectorXd::Ones(d);
  auto init_rng = model.make_rng(seed);
  Eigen::VectorXd theta = model.initialize(nullptr, init_rng, 2.0);
  double eps = walnutpie::detail::find_reasonable_step(rand, logp_grad, theta,
                                                       inv_mass, 1.0);
  std::cout << std::setprecision(16) << "# init theta=" << theta.transpose()
            << "\n# find_reasonable_step eps=" << eps << "\n";
  double lp;
  Eigen::VectorXd grad;
  logp_grad(theta, lp, grad);

  Transition tr_buf;
  Transition* tr_out = &tr_buf;
  std::optional<walnutpie::detail::DualAveraging> da;
  da.emplace(eps, 0.8, 0.05, 10.0, 0.75);
  Eigen::VectorXd mean = Eigen::VectorXd::Zero(d), m2 = Eigen::VectorXd::Zero(d);
  std::size_t n_metric = 0;

  // One transition of S micro steps. Robustness policy (documented):
  //  - any non-finite position/momentum/logp/gradient aborts the transition,
  //    which is REJECTED (state untouched, alpha 0 fed to DualAveraging) and
  //    retried with a fresh momentum, max 20 attempts;
  //  - the adaptation statistic is the WORST-CASE over micro steps,
  //    min_t min(1, exp(-dH_t)) (walnutpie's own min_accept notion), so dual
  //    averaging bounds cumulative end-to-end drift rather than its average;
  //    without this, fixed-length raw leapfrog drifts into overflow.
  // One transition of S leapfrog micro steps. Robustness policy (documented):
  //  - walnutpie-macro_step-style error control: if any micro step produces
  //    a non-finite state/logp/gradient, or the end-to-end |dH| exceeds
  //    kMaxError, the WHOLE trajectory is retried at eps/2 (up to 8 halvings);
  //    the recorded step size is the EFFECTIVE (possibly halved) one;
  //  - then a standard Metropolis accept/reject on the end-to-end dH;
  //  - adaptation statistic fed to DualAveraging: worst case over micro
  //    steps min_t min(1, exp(-dH_t)) (walnutpie's min_accept notion);
  //  - if all halvings fail the transition is rejected (alpha 0).
  bool last_mh_reject = false;
  auto transition = [&](Transition* rec) {
    last_mh_reject = false;
    Transition cand;
    Transition* out = rec ? &cand : nullptr;
    double alpha_min = 1.0, dh_total = 0.0, h_start = 0.0;
    Eigen::VectorXd x_end = theta, g_end = grad;
    double l_end = lp;
    double eff = eps;
    bool ok = false;
    for (int halvings = 0; halvings < 8 && !ok; ++halvings, eff *= 0.5) {
      Eigen::VectorXd z = rand.standard_normal(d);
      Eigen::VectorXd rho =
          z.cwiseProduct(inv_mass.cwiseSqrt().cwiseInverse()).matrix();
      Eigen::VectorXd x = theta, g = grad;
      double l = lp;
      ok = true;
      alpha_min = 1.0;
      if (rec) {
        cand.step = eff;
        cand.inv_mass = inv_mass;
        cand.rho0 = rho;
        cand.positions.assign(1, x);
        cand.grads.assign(1, g);
      }
      h_start = -l + kinetic(rho, inv_mass);
      const double half = 0.5 * eff;
      for (std::size_t t = 0; t < S; ++t) {
        const double h_old = -l + kinetic(rho, inv_mass);
        rho += half * g;
        x.array() += eff * inv_mass.array() * rho.array();
        double l_new;
        if (!x.allFinite() || !rho.allFinite()) { ok = false; break; }
        logp_grad(x, l_new, g);
        if (!std::isfinite(l_new) || !g.allFinite()) { ok = false; break; }
        rho += half * g;
        if (!rho.allFinite()) { ok = false; break; }
        const double dh = (-l_new + kinetic(rho, inv_mass)) - h_old;
        alpha_min = std::min(alpha_min, dh);
        l = l_new;
        if (rec) {
          cand.positions.push_back(x);
          cand.grads.push_back(g);
        }
      }
      if (ok) {
        dh_total = (-l + kinetic(rho, inv_mass)) - h_start;
        if (!std::isfinite(dh_total) || std::fabs(dh_total) > kMaxError)
          ok = false;
      }
      if (ok) {
        // walnutpie macro_step min_accept notion: end-to-end exp(-|dlogp|).
        alpha_min = std::exp(-std::fabs(dh_total));
        x_end = x;
        l_end = l;
        g_end = g;
      }
    }
    if (!ok) {
      static std::size_t dbg2 = 0;
      if (dbg2 < 6)
        std::cerr << "# stab-fail eff=" << eff << " lp_start=" << lp
                  << " |g_start|=" << grad.norm() << "\n";
      ++dbg2;
      return 0.0;
    }
    const double log_u = std::log(rand.uniform_real_01());
    if (!(log_u < -dh_total)) { last_mh_reject = true; return alpha_min; }
    theta = x_end;
    lp = l_end;
    grad = g_end;
    if (rec) *rec = cand;
    return alpha_min;
  };

  for (std::size_t it = 0; it < kWarmupIters; ++it) {
    if (it == 50 && !getenv("W70_IDENTITY_METRIC")) {
      Eigen::VectorXd var = m2 / static_cast<double>(n_metric - 1);
      var.array() *= static_cast<double>(n_metric) /
                     (static_cast<double>(n_metric) + 5.0);
      var.array() += 1e-3 * (5.0 / (static_cast<double>(n_metric) + 5.0));
      inv_mass = var.cwiseInverse();
      eps = walnutpie::detail::find_reasonable_step(rand, logp_grad, theta,
                                                    inv_mass, eps);
      da.emplace(eps, 0.8, 0.05, 10.0, 0.75);
      mean.setZero();
      m2.setZero();
    }
    const double alpha = transition(nullptr);
    if (it < 8 || it % 20 == 0)
      std::cout << std::setprecision(6) << "# warmup it=" << it << " alpha=" << alpha
                << " eps=" << eps << " lp=" << lp << " theta0=" << theta[0] << "\n";
    (*da)(alpha);
    eps = da->step_size();
    if (it >= kMetricStartIdx && it < 50) {
      ++n_metric;
      const Eigen::VectorXd delta = theta - mean;
      mean += delta / static_cast<double>(n_metric);
      m2 += delta.cwiseProduct(delta - delta / static_cast<double>(n_metric));
    }
  }
  eps = da->step_size_bar();  // frozen sampling step size
  std::cout << std::setprecision(6) << "# warmup done: eps=" << eps
            << " d=" << d << " S=" << S << "\n";

  // Sampling + Picard replay + optional binary dump.
  std::ofstream dump;
  if (!dump_path.empty()) {
    dump.open(dump_path, std::ios::binary);
    dump.write("W70D", 4);
    std::uint64_t hdr[3] = {seed, S, static_cast<std::uint64_t>(d)};
    dump.write(reinterpret_cast<const char*>(hdr), sizeof(hdr));
  }

  std::vector<std::size_t> rounds_vec;
  std::vector<std::size_t> evals_vec;
  std::vector<double> err_vec;
  std::size_t n_strict = 0, n_relaxed_only = 0, n_fail = 0;
  std::size_t total_seq_evals = 0, total_picard_evals = 0;

  std::cerr.precision(17);
  std::cerr << "# sampling starts: eps=" << eps << " lp=" << lp << " theta=" << theta.transpose() << " |g|=" << grad.norm() << "\n";
  std::size_t n_discarded = 0;
  for (std::size_t k = 0; k < kSamplingIters; ++k) {
    Transition tr;
    while (true) {
      transition(&tr);
      if (last_mh_reject) continue;  // ordinary MH rejection: fresh momentum
      if (tr.positions.size() == S + 1) break;
      ++n_discarded;
      if (n_discarded > 10000) {
        std::cerr << "cannot draw a non-divergent transition\n";
        return 1;
      }
    }
    const PicardResult pr = picard_replay(logp_grad, tr);
    rounds_vec.push_back(pr.rounds);
    evals_vec.push_back(pr.grad_evals);
    err_vec.push_back(pr.final_abs_err);
    total_picard_evals += pr.grad_evals;
    total_seq_evals += S + 1;
    if (pr.converged_strict) {
      ++n_strict;
    } else if (pr.converged_relaxed) {
      ++n_relaxed_only;
    } else {
      ++n_fail;
      std::cout << "# FAIL transition " << k << " rounds=" << pr.rounds
                << (pr.diverged ? " DIVERGED" : "") << " err="
                << std::setprecision(3) << pr.final_abs_err << "\n";
    }
    if (dump) {
      for (const auto& v : tr.positions)
        dump.write(reinterpret_cast<const char*>(v.data()),
                   static_cast<std::streamsize>(d * sizeof(double)));
      for (const auto& v : tr.grads)
        dump.write(reinterpret_cast<const char*>(v.data()),
                   static_cast<std::streamsize>(d * sizeof(double)));
      dump.write(reinterpret_cast<const char*>(tr.rho0.data()),
                 static_cast<std::streamsize>(d * sizeof(double)));
      dump.write(reinterpret_cast<const char*>(&tr.step), sizeof(double));
      dump.write(reinterpret_cast<const char*>(tr.inv_mass.data()),
                 static_cast<std::streamsize>(d * sizeof(double)));
    }
  }

  // Rounds statistics are computed over CONVERGED transitions only;
  // diverged/non-converged ones are reported separately as failures.
  std::vector<std::size_t> conv_rounds;
  for (std::size_t i = 0; i < rounds_vec.size(); ++i)
    if (err_vec[i] < kTolStrict) conv_rounds.push_back(rounds_vec[i]);
  auto pct = [&](double q) {
    if (conv_rounds.empty()) return static_cast<std::size_t>(0);
    std::vector<std::size_t> v = conv_rounds;
    std::sort(v.begin(), v.end());
    auto idx = static_cast<std::size_t>(
        std::min(static_cast<double>(v.size() - 1), std::floor(q * (v.size() - 1))));
    return v[idx];
  };
  const double frac_le3 =
      static_cast<double>(std::count_if(rounds_vec.begin(), rounds_vec.end(),
                                        [](std::size_t r) { return r <= 3; })) /
      static_cast<double>(rounds_vec.size());

  { std::vector<std::size_t> hist(rounds_vec.begin(), rounds_vec.end());
    std::sort(hist.begin(), hist.end());
    std::vector<double> ev(err_vec); std::sort(ev.begin(), ev.end());
    std::cout << std::setprecision(3) << "# rounds_hist:";
    std::map<std::size_t,std::size_t> cnt;
    for (auto r : rounds_vec) ++cnt[r <= 5 ? r : 99];
    for (auto& kv : cnt) std::cout << " r" << (kv.first==99?std::string(">5"):std::to_string(kv.first)) << "=" << kv.second;
    std::cout << "\n# err_abs med=" << ev[ev.size()/2] << " p90=" << ev[(ev.size()*9)/10]
              << " max=" << ev.back() << "\n";
  }
  std::cout << std::setprecision(6) << "RESULT model=" << model_path
            << " median_rounds=" << pct(0.5) << " p90_rounds=" << pct(0.9)
            << " frac_le3=" << frac_le3 << " strict=" << n_strict
            << " relaxed_only=" << n_relaxed_only << " fail=" << n_fail
            << " seq_evals=" << total_seq_evals
            << " picard_evals=" << total_picard_evals
            << " overhead="
            << static_cast<double>(total_picard_evals) /
                   static_cast<double>(total_seq_evals)
            << "\n";
  return 0;
}
