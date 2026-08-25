// W-64 pre-registered BridgeStan pilot driver (standalone; not wired into
// CMake). Implements the WORKLOG.md entry "W-64 PRE-REGISTRATION" for the
// three in-process arms; the fourth arm (WH-ADAPT) is the stock w54-lineage
// stan_cli binary run separately by scratch/w61/runs_w64/run_w64.sh.
//
// Build:
//   clang++ -std=c++20 -O2 -I include -I <eigen-src> tests/test_w64_pilot.cpp \
//       -o build_ai/test_w64_pilot
//
// Usage:
//   test_w64_pilot <model.so> <data.json> <init.txt|-> <seed> <outdir>
//                  [model_label]
//
// Arms run per invocation (one (model, rep, chain) cell), 1000 warmup +
// 1000 frozen draws each:
//   iso005 : isokinetic path, delta_tol=0.05, max_ell=8 -- h by Gamma
//            bisection (P(micro=0)=0.80) refreshed every 50 warmup iters on
//            the trailing 250-state window (detail::run_adapted_iso),
//            anchor = online coordinate median, identity mass.
//   iso020 : same with delta_tol=0.20.
//   whid   : walnutpie's own transition_w, IDENTITY inv_mass, minimal dual
//            averaging (target accept 0.8, eps clamp [1e-3,10]) -- the
//            W-62b/c baseline construction verbatim.
//
// Metric: SAMPLING-PHASE-ONLY gradient calls; rank-normalized min-coord
// ESS_bulk (Vehtari-2021 single-chain form as W-62b/c) over CONSTRAINED
// draws (bs_param_constrain) -- same choice as the WH-ADAPT arm whose
// --output CSV is constrained; degenerate/pinned chains score ESS=0.
// Unconstrained draws are additionally logged per arm as
// <outdir>/<arm>_unc.csv (pre-registered logging requirement).

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <limits>
#include <random>
#include <string>
#include <vector>

#include <Eigen/Dense>

#include "walnutpie/isoadapt.hpp"
#include "walnutpie/isokinetic.hpp"
#include "walnutpie/load_stan.hpp"
#include "walnutpie/util.hpp"
#include "walnutpie/walnuts.hpp"
#include "walnutpie/warmup_heuristics.hpp"

using Eigen::VectorXd;
using Random = walnutpie::detail::Random<std::mt19937_64>;
using walnutpie::DynamicStanModel;

// ---- counting wrapper ---------------------------------------------------------
struct CountF {
  const DynamicStanModel& m;
  mutable long long n = 0;
  void operator()(const VectorXd& x, double& lp, VectorXd& g) const {
    ++n;
    m.logp_grad(x, lp, g);
  }
};

// dual-averaging hook (W-62b/c baseline construction)
template <class Upd>
struct DaHandler {
  double target;
  Upd update;
  double step_size() const { return std::numeric_limits<double>::quiet_NaN(); }
  void operator()(double accept_prob) const { update(accept_prob); }
};

// ---- rank-normalized ESS (identical to test_w62c.cpp) --------------------------
double inv_phi(double p) {
  static const double a[6] = {-3.969683028665376e+01, 2.209460984245205e+02,
                              -2.759285104469687e+02, 1.383577518672690e+02,
                              -3.066479806614716e+01, 2.506628277459239e+00};
  static const double b[5] = {-5.447609879822406e+01, 1.615858368580409e+02,
                              -1.556989798598866e+02, 6.680131188771972e+01,
                              -1.328068155288572e+01};
  static const double c[6] = {-7.784894002430293e-03, -3.223964580411365e-01,
                              -2.400758277161838e+00, -2.549732539343734e+00,
                              4.374664141464968e+00, 2.938163982698783e+00};
  static const double dd[4] = {7.784695709041462e-03, 3.224671290700398e-01,
                               2.445134137142996e+00, 3.754408661907416e+00};
  const double plow = 0.02425, phigh = 1 - plow;
  double q, r;
  if (p < plow) {
    q = std::sqrt(-2 * std::log(p));
    return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) /
           ((((dd[0]*q+dd[1])*q+dd[2])*q+dd[3])*q+1);
  }
  if (p > phigh) {
    q = std::sqrt(-2 * std::log(1 - p));
    return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) /
            ((((dd[0]*q+dd[1])*q+dd[2])*q+dd[3])*q+1);
  }
  q = p - 0.5;
  r = q * q;
  return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q /
         (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1);
}

double ess_ips(const std::vector<double>& xs) {
  const long n = static_cast<long>(xs.size());
  double mean = 0.0;
  for (double x : xs) mean += x;
  mean /= n;
  double var = 0.0;
  bool finite = true;
  for (double x : xs) {
    if (!std::isfinite(x)) finite = false;
    var += (x - mean) * (x - mean);
  }
  var /= (n - 1);
  if (!finite || !(var > 0)) return 0.0;  // degenerate/pinned chain -> zero
  auto acov = [&](long k) {
    double s = 0.0;
    for (long i = k; i < n; ++i) s += (xs[i] - mean) * (xs[i - k] - mean);
    return s / n;
  };
  double tau = 0.0;
  for (long k = 1; k + 1 < n; k += 2) {
    double pair = acov(k) + acov(k + 1);
    if (pair <= 0) break;
    tau += pair;
  }
  double ess = n / (1.0 + 2.0 * tau / var);
  return std::min(ess, static_cast<double>(n));
}

double rn_ess_bulk(const std::vector<VectorXd>& draws) {
  const long n = static_cast<long>(draws.size());
  const int d = static_cast<int>(draws[0].size());
  std::vector<std::pair<double, int>> order(n);
  double best = std::numeric_limits<double>::infinity();
  for (int j = 0; j < d; ++j) {
    for (long t = 0; t < n; ++t)
      order[t] = {draws[t][j], static_cast<int>(t)};
    std::sort(order.begin(), order.end());
    std::vector<double> z(n);
    long t = 0;
    while (t < n) {
      long u = t;
      while (u + 1 < n && order[u + 1].first == order[t].first) ++u;
      const double ravg = 0.5 * (t + u) + 1.0;
      const double zz = inv_phi((ravg - 0.375) / (n + 0.25));
      for (long k = t; k <= u; ++k) z[order[k].second] = zz;
      t = u + 1;
    }
    best = std::min(best, ess_ips(z));
  }
  return best;
}

// ---- helpers -------------------------------------------------------------------
static VectorXd read_init_file(const char* path, std::size_t d_expected) {
  std::ifstream in(path);
  if (!in) throw std::invalid_argument(std::string("cannot open init: ") + path);
  std::vector<double> vals;
  double v;
  while (in >> v) vals.push_back(v);
  if (vals.size() != d_expected)
    throw std::invalid_argument(std::string("init dim mismatch: ") + path);
  return VectorXd(VectorXd::Map(vals.data(), static_cast<Eigen::Index>(vals.size())));
}

static void write_csv(const std::string& filename,
                      const std::vector<VectorXd>& draws) {
  std::ofstream out(filename);
  for (const auto& dr : draws) {
    for (Eigen::Index j = 0; j < dr.size(); ++j)
      out << (j ? "," : "") << std::setprecision(12) << dr[j];
    out << "\n";
  }
}

struct ArmResult {
  double ess = 0.0;
  long long samp_grads = 0;
  double tuning_detail = 0.0;  // frozen h (iso) or eps (whid)
};

int main(int argc, char** argv) {
  if (argc < 6) {
    std::fprintf(stderr,
                 "usage: %s <model.so> <data.json> <init.txt|-> <seed> "
                 "<outdir> [label]\n",
                 argv[0]);
    return 2;
  }
  const std::string so_path = argv[1];
  const std::string data_path = argv[2];
  const std::string init_path = argv[3];
  const unsigned long long seed = std::strtoull(argv[4], nullptr, 10);
  const std::string outdir = argv[5];
  const std::string label = argc > 6 ? argv[6] : "m";

  const int n_warmup = 1000;
  const int n_draws = 1000;

  DynamicStanModel model(so_path.c_str(),
                         data_path.empty() ? nullptr : data_path.c_str(),
                         static_cast<unsigned int>(seed));
  const std::size_t d_unc = model.unconstrained_dimensions();
  const std::size_t d_con = model.constrained_dimensions();
  // "-" = no pf-init available (low_dim_gauss_mix): draw ONE random init
  // (-init 2.0 equivalent, deterministic from the cell seed), persist it to
  // <outdir>/init.txt so the WH-ADAPT arm consumes the identical file.
  const VectorXd theta0 = [&]() {
    if (init_path != "-") return read_init_file(init_path.c_str(), d_unc);
    auto init_rng = model.make_rng(static_cast<unsigned int>(seed + 31337));
    VectorXd p = model.initialize(nullptr, init_rng, 2.0);
    std::ofstream f(outdir + "/init.txt");
    for (Eigen::Index j = 0; j < p.size(); ++j)
      f << std::setprecision(17) << p[j] << "\n";
    return p;
  }();

  // constrained-draw RNG: separate stream, never touches chain dynamics
  auto con_rng = model.make_rng(static_cast<unsigned int>(seed + 777));

  auto constrain_all = [&](const std::vector<VectorXd>& unc) {
    std::vector<VectorXd> out;
    out.reserve(unc.size());
    for (const auto& u : unc) {
      VectorXd c(d_con);
      model.constrain_draw(u, c, con_rng);
      out.push_back(c);
    }
    return out;
  };

  walnutpie::detail::IsoAdaptConfig base_cfg;
  base_cfg.n_warmup = n_warmup;
  base_cfg.n_draws = n_draws;
  base_cfg.refresh_every = 50;
  base_cfg.window = 250;
  base_cfg.max_ell = 8;
  base_cfg.i_max = 9;

  struct Rec { std::string arm; ArmResult r; };
  std::vector<Rec> recs;

  // ---- ISO arms ---------------------------------------------------------------
  for (double dtol : {0.05, 0.20}) {
    walnutpie::detail::IsoAdaptConfig cfg = base_cfg;
    cfg.delta_tol = dtol;
    CountF cf{model};
    std::mt19937_64 rng(seed);
    Random rand(rng);
    auto res = walnutpie::detail::run_adapted_iso(rand, cf, theta0, cfg, seed);
    ArmResult r;
    r.samp_grads = res.sampling_grads;
    r.tuning_detail = res.h;
    r.ess = rn_ess_bulk(constrain_all(res.draws));
    write_csv(outdir + "/iso" + (dtol == 0.05 ? "005" : "020") + "_unc.csv",
              res.draws);
    recs.push_back({dtol == 0.05 ? "iso005" : "iso020", r});
  }

  // ---- WH-ID arm: W-62b/c baseline construction, identity mass -----------------
  {
    CountF cf{model};
    std::mt19937_64 rng(seed + 1);  // seed convention as W-62b/c
    Random rand(rng);
    const VectorXd inv_mass = VectorXd::Ones(d_unc);
    const VectorXd chol_mass = VectorXd::Ones(d_unc);
    VectorXd theta = theta0;
    walnutpie::detail::NoOpStepSizeAdapter noop;

    double eps0 = walnutpie::detail::find_reasonable_step(rand, cf, theta,
                                                          inv_mass, 1.0);
    const double mu = std::log(10.0 * eps0);
    const double gamma_da = 0.05, t0 = 10.0, kappa = 0.75, delta_t = 0.8;
    double h_bar = 0.0, log_eps_bar = 0.0, eps = eps0;
    VectorXd grad_cached;
    double logp_cached = -std::numeric_limits<double>::infinity();
    std::size_t depth;
    for (int t = 1; t <= n_warmup; ++t) {
      const double eta = 1.0 / (t + t0);
      auto update = [&](double a) {
        h_bar = (1 - eta) * h_bar + eta * (delta_t - a);
        double x_k = mu - std::sqrt(static_cast<double>(t)) / gamma_da * h_bar;
        x_k = std::clamp(x_k, std::log(1e-3), std::log(10.0));
        eps = std::exp(x_k);
        const double w = std::pow(static_cast<double>(t), -kappa);
        log_eps_bar = w * x_k + (1 - w) * log_eps_bar;
      };
      DaHandler<decltype(update)> handler{delta_t, update};
      theta = walnutpie::detail::transition_w(rand, cf, inv_mass, chol_mass,
                                              eps, 8, 5, 1, 0.5,
                                              std::move(theta), depth,
                                              grad_cached, logp_cached,
                                              handler, grad_cached,
                                              logp_cached);
    }
    const double eps_final = std::clamp(std::exp(log_eps_bar), 1e-3, 10.0);
    const long long warmup_grads = cf.n;
    std::vector<VectorXd> draws;
    draws.reserve(n_draws);
    for (int t = 0; t < n_draws; ++t) {
      theta = walnutpie::detail::transition_w(rand, cf, inv_mass, chol_mass,
                                              eps_final, 8, 5, 1, 0.5,
                                              std::move(theta), depth,
                                              grad_cached, logp_cached,
                                              noop, grad_cached,
                                              logp_cached);
      draws.push_back(theta);
    }
    ArmResult r;
    r.samp_grads = cf.n - warmup_grads;
    r.tuning_detail = eps_final;
    r.ess = rn_ess_bulk(constrain_all(draws));
    write_csv(outdir + "/whid_unc.csv", draws);
    recs.push_back({"whid", r});
  }

  for (const auto& rec : recs) {
    std::printf(
        "W64SUM\t%s\t%s\tn_warmup=%d\tn_draws=%d\trness_min=%.4f\t"
        "samp_grads=%lld\tgrads_per_draw=%.4f\tess_per_grad=%.3e\tdetail=%.6f\n",
        label.c_str(), rec.arm.c_str(), n_warmup, n_draws, rec.r.ess,
        rec.r.samp_grads,
        static_cast<double>(rec.r.samp_grads) / n_draws,
        rec.r.ess > 0 ? rec.r.ess / static_cast<double>(rec.r.samp_grads) : 0.0,
        rec.r.tuning_detail);
  }
  return 0;
}
