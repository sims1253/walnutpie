#include <walnutpie.hpp>
#include <walnutpie/load_stan.hpp>
#include <walnutpie/warmup_heuristics.hpp>

#include <CLI/CLI.hpp>
#include <algorithm>
#include <fstream>
#include <memory>
#include <utility>
#include <vector>
#include <Eigen/Dense>

#include <chrono>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <vector>

using walnutpie::DynamicStanModel;
using walnutpie::unique_bs_rng;

static void summarize(const std::vector<std::string>& names,
                      const Eigen::MatrixXd& draws) {
  auto N = draws.cols();
  auto D = draws.rows();
  for (auto d = 0; d < D; ++d) {
    if (d > 3 && d < D - 3) {
      if (d == 4) {
        std::cout << "... elided " << (D - 6) << " rows ..." << std::endl;
      }
      continue;
    }
    auto mean = draws.row(d).mean();
    auto var = (draws.row(d).array() - mean).square().sum() / (N - 1);
    auto stddev = std::sqrt(var);
    std::cout << names[static_cast<std::size_t>(d)] << ": mean = " << mean
              << ", stddev = " << stddev << "\n";
  }
}

static void write_draws(const std::string& filename,
                        const std::vector<std::string>& names,
                        const Eigen::MatrixXd& draws) {
  if (filename.empty()) {
    return;
  }
  std::ofstream out(filename);
  if (!out) {
    std::cerr << "Failed to open output file: " << filename << std::endl;
    return;
  }

  for (size_t i = 0; i < names.size(); ++i) {
    if (i > 0) {
      out << ",";
    }
    out << names[i];
  }
  out << "\n";

  auto EigenCommaFormat =
      Eigen::IOFormat(12, Eigen::DontAlignCols, ",", "\n", "", "", "", "");

  out << draws.transpose().format(EigenCommaFormat);
  out.close();
}

class StanHandler {
 public:
  StanHandler(DynamicStanModel& model, unsigned int seed,
              std::size_t num_warmup, std::size_t num_draws, bool save_warmup)
      : model_(model),
        rng_(model.make_rng(seed + 1)),
        draws_(model.constrained_dimensions(),
               num_draws + static_cast<std::size_t>(save_warmup) * num_warmup),
        save_warmup_(save_warmup) {}

  void on_sample(const Eigen::VectorXd& position, double lp) {
    model_.constrain_draw(position, draws_.col(n_), rng_);
    n_++;
  }

  void on_warmup(const Eigen::VectorXd& position, double lp, double step_size,
                 const Eigen::VectorXd& diag_inv_mass) {
    if (const char* dbg = std::getenv("WALNUTPIE_DEBUG_WARMUP")) {
      if (warmup_dbg_it_ % std::max(1, atoi(dbg)) == 0) {
        std::cout << "[warmup it " << warmup_dbg_it_ << "] lp=" << lp
                  << " step=" << step_size
                  << " invm[0]=" << (diag_inv_mass.size() ? diag_inv_mass[0] : -1)
                  << " pos[0]=" << (position.size() ? position[0] : 0)
                  << " pos[last]=" << (position.size() ? position[position.size()-1] : 0)
                  << std::endl;
      }
    }
    ++warmup_dbg_it_;
    if (!save_warmup_) {
      return;
    }
    model_.constrain_draw(position, draws_.col(n_), rng_);
    n_++;
  }

  void on_warmup_complete(double step_size,
                          const Eigen::VectorXd& diag_inv_mass) {}

  void on_logp_exception(const Eigen::VectorXd& position,
                         const std::exception& exn) const noexcept {
    std::cout << "Logp failed with exception " << exn.what() << " at "
              << position.transpose() << "\n";
  }

  void summarize() {
    auto names = model_.param_names();
    ::summarize(names, draws_);
  }

  void write_csv(std::string& output_file) {
    auto names = model_.param_names();
    ::write_draws(output_file, names, draws_);
  }

 private:
  DynamicStanModel& model_;
  unique_bs_rng rng_;
  Eigen::MatrixXd draws_;
  bool save_warmup_;
  Eigen::Index n_ = 0;
  std::size_t warmup_dbg_it_ = 0;
};

template <typename Opt>
StanHandler run_walnuts(DynamicStanModel& model, unsigned int seed,
                        walnutpie::InitConfigBuilder& init_builder,
                        std::size_t num_warmup, std::size_t num_draws,
                        bool save_warmup, walnutpie::WarmupConfig& warmup_cfg,
                        walnutpie::SamplingConfig& sample_cfg,
                        double mass_init_clamp = 0.0,
                        bool step_init_heuristic = false,
                        double early_exit_tol = 0.0) {
  using Clock = std::chrono::high_resolution_clock;
  auto elapsed_seconds = [](auto t) {
    return std::chrono::duration<double>(Clock::now() - t).count();
  };
  double logp_time = 0.0;
  std::size_t logp_count = 0;
  auto global_start = Clock::now();

  auto end_timing = [&]() {
    auto global_total_time = elapsed_seconds(global_start);
    std::cout << "    total time: " << global_total_time << "s" << std::endl;
    std::cout << "logp_grad time: " << logp_time << "s" << std::endl;
    std::cout << "logp_grad fraction: " << logp_time / global_total_time
              << std::endl;
    std::cout << "        logp_grad calls: " << logp_count << std::endl;
    std::cout << "        time per call: " << logp_time / logp_count << "s"
              << std::endl;
    std::cout << std::endl;
  };

  StanHandler storage(model, seed, num_warmup, num_draws, save_warmup);

  auto logp = [&](auto&&... args) {
    auto start = Clock::now();
    model.logp_grad(args...);
    logp_time += elapsed_seconds(start);
    ++logp_count;
  };

  auto init_cfg = init_builder.masses(logp, warmup_cfg.mass_additive_smoothing(),
                                    false, mass_init_clamp)
                      .build();
  auto inits = init_cfg.init_chain_config(0);
  std::mt19937_64 rng{seed};
  if (step_init_heuristic) {
    const auto inv_mass = inits.mass().array().inverse().matrix().eval();
    walnutpie::detail::Random heur_rand(rng);
    double eps = walnutpie::detail::find_reasonable_step(
        heur_rand, logp, inits.position(), inv_mass, inits.step_size());
    inits = walnutpie::InitChainConfig(eps, inits.position(), inits.mass());
    std::cout << "Heuristic initial step size: " << eps << std::endl;
  }

  walnutpie::AdaptiveWalnuts<decltype(logp), decltype(rng), StanHandler, Opt>
      walnuts(
      rng, storage, logp, inits, warmup_cfg, sample_cfg);
  // W-21: temporal-stabilization early exit (single-chain analogue of the
  // multi-chain controller's convergence stop). Every 50 iterations — the
  // metric-window length, so successive snapshots are INDEPENDENT window
  // estimates — compare inv_mass() to the previous snapshot; exit when the
  // l2 rel-diff of the mass and the rel-diff of the step size have both
  // stabilized. Floors: 200 iters minimum (4 full metric windows).
  if (early_exit_tol > 0.0) {
    const std::size_t period = 50;
    const std::size_t min_iters = 200;
    Eigen::VectorXd prev_mass;
    double prev_step = -1.0;
    for (std::size_t w = 0; w < num_warmup; ++w) {
      walnuts();
      if ((w + 1) >= min_iters && (w + 1) % period == 0) {
        Eigen::VectorXd mass = walnuts.inv_mass();
        double step = walnuts.step_size();
        if (prev_mass.size() == mass.size()) {
          double mass_diff = (mass - prev_mass).norm() /
                             std::max(prev_mass.norm(), 1e-12);
          double step_diff = std::abs(step - prev_step) /
                             std::max(std::abs(prev_step), 1e-12);
          if (mass_diff < early_exit_tol && step_diff < 0.1) {
            std::cout << "Early warmup exit at iteration " << (w + 1)
                      << " (mass_diff=" << mass_diff
                      << ", step_diff=" << step_diff << ")" << std::endl;
            break;
          }
        }
        prev_mass = mass;
        prev_step = step;
      }
    }
  } else {
    for (std::size_t w = 0; w < num_warmup; ++w) {
      walnuts();
    }
  }
  end_timing();

  // N post-warmup draws
  auto sampler = walnuts.sampler();  // freeze tuning
  std::cout << "Adaptation completed." << std::endl;
  std::cout << "Note: multi-chain mode (walnutpie::adapt) reports a "
               "log-mass cross-chain dispersion diagnostic for mode-aware "
               "reinit policies; single-chain CLI does not." << std::endl;
  std::cout << "Macro time = " << sampler.macro_time() << std::endl;
  std::cout << "Mass matrix diagonal = ["
            << sampler.inverse_mass_matrix_diagonal() << "]" << std::endl;

  logp_time = 0.0;
  logp_count = 0;
  global_start = Clock::now();
  for (std::size_t n = 0; n < num_draws; ++n) {
    sampler();
  }
  end_timing();

  return storage;
}

// W-25: multi-chain mode driving the LIBRARY warmup controller
// (walnutpie::detail::adapt_with_stats), which supports cross-chain early
// exit plus the temporal step-drift gate. One BridgeStan model instance,
// one mt19937_64 stream, and one StanHandler per chain (BridgeStan model
// instances are not thread-safe; a shared std::normal_distribution across
// streams is not reproducible). Seeding replicates the single-chain CLI
// invoked once per chain with --seed (seed + c), so a full-warmup
// multi-chain run consumes the same per-chain RNG streams.
struct NullInterrupt {
  void throw_if_interrupted() const {}
};

// W-28: recording-only handler for pilot-burst draws. Pilot draws are
// diagnostics only — they never reach the output draws (the real handlers
// are untouched, so their GQ rng streams stay bit-identical to a no-pilot
// run at the same exit point).
struct PilotSampleHandler {
  std::vector<double> lp;
  void on_sample(const Eigen::VectorXd& /*position*/, double logp) {
    lp.push_back(logp);
  }
  void on_logp_exception(const Eigen::VectorXd& position,
                         const std::exception& exn) const noexcept {
    std::cout << "Pilot logp failed with exception " << exn.what() << " at "
              << position.transpose() << "\n";
  }
};

// W-28 pre-registered gate statistics on the per-chain pilot lp draws:
// 1) per-chain lag-1 autocorrelation with the biased (ML) autocovariance
//    estimator: mean = (1/P) sum lp; c0 = (1/P) sum (lp-mean)^2;
//    c1 = (1/P) sum_{n=0..P-2} (lp_n-mean)(lp_{n+1}-mean); rho1 = c1/c0
//    (c0 <= 0 -> rho1 := 1, i.e. a frozen chain fails).
// 2) split-half cross-chain R-hat proxy (NOT rank-normalized; 50 draws is
//    too few): each chain's P draws split into first/last P/2 -> J = 2M
//    half-chains of n_h = P/2; W = mean half-chain sample variance (ddof=1);
//    B = n_h/(J-1) * sum (mean_j - grand)^2;
//    var_plus = (n_h-1)/n_h * W + B/n_h; Rhat = sqrt(var_plus/W)
//    (W <= 0 -> Rhat := +inf, fail).
struct PilotStats {
  double rho1_max;
  double rhat_lp;
  bool pass;
};

static PilotStats pilot_gate_stats(
    const std::vector<std::vector<double>>& chain_lps, double rho1_tol,
    double rhat_tol) {
  double rho1_max = -std::numeric_limits<double>::infinity();
  for (const auto& lp : chain_lps) {
    const std::size_t p = lp.size();
    if (p < 2) {
      rho1_max = std::numeric_limits<double>::infinity();
      continue;
    }
    double mean = 0.0;
    for (double x : lp) {
      mean += x;
    }
    mean /= static_cast<double>(p);
    double c0 = 0.0;
    double c1 = 0.0;
    for (std::size_t n = 0; n < p; ++n) {
      c0 += (lp[n] - mean) * (lp[n] - mean);
      if (n + 1 < p) {
        c1 += (lp[n] - mean) * (lp[n + 1] - mean);
      }
    }
    c0 /= static_cast<double>(p);
    c1 /= static_cast<double>(p);
    const double rho1 = c0 > 0.0 ? c1 / c0 : 1.0;
    rho1_max = std::max(rho1_max, rho1);
  }
  // Split-half R-hat over J half-chains of length n_h.
  const std::size_t n_h = chain_lps[0].size() / 2;
  double w_sum = 0.0;
  double mean_sum = 0.0;
  std::size_t j = 0;
  for (const auto& lp : chain_lps) {
    for (std::size_t half = 0; half < 2; ++half) {
      double m = 0.0;
      for (std::size_t n = 0; n < n_h; ++n) {
        m += lp[half * n_h + n];
      }
      m /= static_cast<double>(n_h);
      double s2 = 0.0;
      for (std::size_t n = 0; n < n_h; ++n) {
        s2 += (lp[half * n_h + n] - m) * (lp[half * n_h + n] - m);
      }
      s2 /= static_cast<double>(n_h - 1);
      w_sum += s2;
      mean_sum += m;
      ++j;
    }
  }
  const double w = w_sum / static_cast<double>(j);
  const double grand = mean_sum / static_cast<double>(j);
  double b_sum = 0.0;
  for (const auto& lp : chain_lps) {
    for (std::size_t half = 0; half < 2; ++half) {
      double m = 0.0;
      for (std::size_t n = 0; n < n_h; ++n) {
        m += lp[half * n_h + n];
      }
      m /= static_cast<double>(n_h);
      b_sum += (m - grand) * (m - grand);
    }
  }
  const double b =
      static_cast<double>(n_h) / static_cast<double>(j - 1) * b_sum;
  const double var_plus =
      static_cast<double>(n_h - 1) / static_cast<double>(n_h) * w +
      b / static_cast<double>(n_h);
  const double rhat =
      w > 0.0 ? std::sqrt(var_plus / w)
              : std::numeric_limits<double>::infinity();
  return {rho1_max, rhat, rho1_max <= rho1_tol && rhat < rhat_tol};
}

struct ChainTiming {
  double logp_time = 0.0;
  std::size_t logp_count = 0;
};

static auto make_timed_logp(DynamicStanModel& model, ChainTiming& t) {
  using Clock = std::chrono::high_resolution_clock;
  return [&model, &t](auto&&... args) {
    auto start = Clock::now();
    model.logp_grad(args...);
    t.logp_time +=
        std::chrono::duration<double>(Clock::now() - start).count();
    ++t.logp_count;
  };
}

template <typename Opt>
void run_walnuts_multi(
    const std::string& lib, const std::string& data, unsigned int seed,
    std::size_t chains, std::size_t num_warmup, std::size_t num_draws,
    bool save_warmup, const walnutpie::WarmupConfig& warmup_cfg,
    const walnutpie::SamplingConfig& sample_cfg, double init_radius,
    double step_size_init, const std::string& init_pattern,
    const std::string& out_pattern, std::size_t pilot_burst,
    double pilot_rho1_max, double pilot_rhat_max, bool serial_exec) {
  using Clock = std::chrono::high_resolution_clock;
  using LogpT = decltype(make_timed_logp(std::declval<DynamicStanModel&>(),
                                         std::declval<ChainTiming&>()));
  using RNG = std::mt19937_64;
  using Adapter =
      walnutpie::AdaptiveWalnuts<LogpT, RNG, StanHandler, Opt>;
  using Sampler = walnutpie::WalnutsSampler<LogpT, RNG, StanHandler>;

  auto elapsed = [](auto t) {
    return std::chrono::duration<double>(Clock::now() - t).count();
  };
  auto print_stanza = [&](std::size_t c, double total, ChainTiming& t) {
    std::cout << "chain " << c << "    total time: " << total << "s"
              << std::endl;
    std::cout << "chain " << c << " logp_grad time: " << t.logp_time << "s"
              << std::endl;
    std::cout << "chain " << c << " logp_grad fraction: "
              << t.logp_time / total << std::endl;
    std::cout << "chain " << c << "     logp_grad calls: " << t.logp_count
              << std::endl;
    std::cout << "chain " << c << "     time per call: "
              << t.logp_time / t.logp_count << "s" << std::endl;
    std::cout << std::endl;
  };

  auto global_start = Clock::now();

  // One model + timing + timed logp per chain (models are heap-stable).
  std::vector<std::unique_ptr<DynamicStanModel>> models;
  std::vector<ChainTiming> timing;
  std::vector<LogpT> logps;
  models.reserve(chains);
  timing.reserve(chains);
  logps.reserve(chains);
  for (std::size_t c = 0; c < chains; ++c) {
    models.emplace_back(
        std::make_unique<DynamicStanModel>(lib.c_str(), data.c_str(),
                                           seed + static_cast<unsigned>(c)));
    timing.emplace_back();
    logps.push_back(
        make_timed_logp(*models.back(), timing.back()));
  }

  // Initial positions per chain, replicating the single-chain CLI
  // (--init-file wins; else model.initialize with the model's rng).
  std::vector<Eigen::VectorXd> positions;
  positions.reserve(chains);
  for (std::size_t c = 0; c < chains; ++c) {
    if (!init_pattern.empty()) {
      std::string f = init_pattern;
      f.replace(f.find("{c}"), 3, std::to_string(c));
      std::ifstream in(f);
      if (!in) {
        throw std::invalid_argument("cannot open init file: " + f);
      }
      std::vector<double> vals;
      double v;
      while (in >> v) {
        vals.push_back(v);
      }
      if (vals.size() !=
          static_cast<std::size_t>(models[c]->unconstrained_dimensions())) {
        throw std::invalid_argument("init file dimension mismatch: " + f);
      }
      positions.emplace_back(
          Eigen::VectorXd::Map(vals.data(), vals.size()));
    } else {
      auto mrng = models[c]->make_rng(seed + static_cast<unsigned>(c));
      positions.push_back(
          models[c]->initialize(nullptr, mrng, init_radius));
    }
  }

  // Mass seeding via any chain's logp (deterministic; identical values).
  auto init_cfg =
      walnutpie::InitConfigBuilder{
          chains, static_cast<std::size_t>(positions[0].size())}
          .step_sizes(step_size_init)
          .positions(std::move(positions))
          .masses(logps[0], warmup_cfg.mass_additive_smoothing(), false, 0.0)
          .build();

  std::vector<RNG> rngs;
  rngs.reserve(chains);
  for (std::size_t c = 0; c < chains; ++c) {
    rngs.emplace_back(seed + static_cast<unsigned>(c));
  }
  std::vector<StanHandler> handlers;
  handlers.reserve(chains);
  for (std::size_t c = 0; c < chains; ++c) {
    handlers.emplace_back(*models[c], seed + static_cast<unsigned>(c),
                          num_warmup, num_draws, save_warmup);
  }
  std::vector<Adapter> adapters;
  adapters.reserve(chains);
  for (std::size_t c = 0; c < chains; ++c) {
    adapters.emplace_back(rngs[c], handlers[c], logps[c],
                          init_cfg.init_chain_config(c), warmup_cfg,
                          sample_cfg);
  }

  NullInterrupt interrupt;
  const walnutpie::detail::ChainExec exec =
      serial_exec ? walnutpie::detail::ChainExec::Serial
                  : walnutpie::detail::ChainExec::Threads;
  std::cout << "chain exec: " << (serial_exec ? "serial" : "threads")
            << std::endl;
  // W-28 pilot-burst gate: at each candidate early exit, run pilot_burst
  // draws per chain on the would-be-frozen sampler (separate RNG streams,
  // recording-only handlers — saved draws and the chains' sampling RNG
  // streams are untouched) and approve the exit only when the pilot's lp
  // mixing statistics pass. A veto resumes warmup from the preserved
  // adapter state (adapt_with_pilot).
  std::size_t pilot_checks = 0;
  auto pilot_gate_fn = [&](std::vector<Adapter>& adapters,
                           std::size_t cand_iter) -> bool {
    std::vector<std::vector<double>> pilot_lps(chains);
    auto run_pilot_chain = [&](std::size_t c) {
      walnutpie::detail::interactive_qos();
      // Read-only snapshot of the frozen tuning + position (sampler()
      // does not mutate the adapter or advance any rng). The pilot runs
      // on its own rng stream: 7919*(c+1) offsets cannot collide with
      // the warmup streams (seed + c, c < chains << 7919). Diagonal
      // metric only (the CLI study path; --metric-rank --metric-full is
      // not piloted).
      auto src = adapters[c].sampler();
      RNG prng(seed + 7919u * static_cast<unsigned int>(c + 1));
      PilotSampleHandler ph;
      walnutpie::WalnutsSampler<LogpT, RNG, PilotSampleHandler> pilot(
          prng, ph, logps[c], src.position(), src.inv_mass(),
          src.macro_time(), sample_cfg.max_trajectory_doublings(),
          sample_cfg.max_step_halvings(), adapters[c].min_micro_steps(),
          src.max_error());
      for (std::size_t n = 0; n < pilot_burst; ++n) {
        pilot();
      }
      pilot_lps[c] = std::move(ph.lp);
    };
    if (serial_exec) {
      for (std::size_t c = 0; c < chains; ++c) {
        run_pilot_chain(c);
      }
    } else {
      std::vector<std::jthread> ts;
      ts.reserve(chains);
      for (std::size_t c = 0; c < chains; ++c) {
        ts.emplace_back([&, c]() { run_pilot_chain(c); });
      }
    }
    ++pilot_checks;
    const PilotStats st =
        pilot_gate_stats(pilot_lps, pilot_rho1_max, pilot_rhat_max);
    std::cout << "pilot check " << pilot_checks << " at iter " << cand_iter
              << ": rho1_max=" << st.rho1_max << " rhat_lp=" << st.rhat_lp
              << " -> " << (st.pass ? "approve" : "resume") << std::endl;
    return st.pass;
  };
  walnutpie::detail::AdaptResult ar =
      pilot_burst > 0
          ? walnutpie::detail::adapt_with_pilot(init_cfg, warmup_cfg,
                                                adapters, interrupt,
                                                pilot_gate_fn, exec)
          : walnutpie::detail::adapt_with_stats(init_cfg, warmup_cfg,
                                                adapters, interrupt, exec);
  if (pilot_burst > 0) {
    std::cout << "pilot checks total=" << pilot_checks << std::endl;
  }
  double warm_wall = elapsed(global_start);
  std::cout << "controller exit_iter=" << ar.exit_iter
            << " early_exit=" << (ar.early_exit ? 1 : 0)
            << " (max_iter=" << warmup_cfg.max_iter() << ")" << std::endl;
  for (std::size_t c = 0; c < chains; ++c) {
    print_stanza(c, warm_wall, timing[c]);
  }

  std::vector<Sampler> samplers;
  samplers.reserve(chains);
  for (std::size_t c = 0; c < chains; ++c) {
    samplers.emplace_back(adapters[c].sampler());
  }

  for (auto& t : timing) {
    t = ChainTiming{};
  }
  global_start = Clock::now();
  if (serial_exec) {
    // W-30 serial topology: chains run one at a time on the calling
    // thread (identical draws to the threaded sampling below).
    for (std::size_t c = 0; c < chains; ++c) {
      for (std::size_t n = 0; n < num_draws; ++n) {
        samplers[c]();
      }
    }
  } else {
    std::vector<std::jthread> ts;
    ts.reserve(chains);
    for (std::size_t c = 0; c < chains; ++c) {
      ts.emplace_back([&samplers, c, num_draws]() {
        walnutpie::detail::interactive_qos();
        for (std::size_t n = 0; n < num_draws; ++n) {
          samplers[c]();
        }
      });
    }
  }
  double sample_wall = elapsed(global_start);
  for (std::size_t c = 0; c < chains; ++c) {
    print_stanza(c, sample_wall, timing[c]);
  }

  if (!out_pattern.empty()) {
    for (std::size_t c = 0; c < chains; ++c) {
      std::string f = out_pattern;
      f.replace(f.find("{c}"), 3, std::to_string(c));
      handlers[c].write_csv(f);
    }
  }
}

int main(int argc, char** argv) {
  auto clock_count =
      std::chrono::system_clock::now().time_since_epoch().count();
  auto clock_seed = static_cast<unsigned int>(clock_count);
  srand(clock_seed);

  auto seed = static_cast<unsigned long int>(rand());
  std::size_t num_warmup = 128;
  std::size_t num_draws = 128;
  bool save_warmup = false;

  walnutpie::WarmupConfig default_warmup =
      walnutpie::WarmupConfigBuilder().build();
  double mass_init_count = default_warmup.mass_init_count();
  double mass_additive_smoothing = default_warmup.mass_additive_smoothing();
  double max_macro_steps_target = default_warmup.max_macro_steps_target();
  double step_accept_rate_target = default_warmup.step_accept_rate_target();
  double step_learning_rate = default_warmup.step_learning_rate();
  double step_gradient_decay = default_warmup.step_gradient_decay();
  double step_sq_gradient_decay = default_warmup.step_sq_gradient_decay();
  double step_stabilization = default_warmup.step_stabilization();
  double step_learn_rate_decay = default_warmup.step_learn_rate_decay();

  walnutpie::SamplingConfig default_sampling =
      walnutpie::SamplingConfigBuilder().build();

  std::string step_optimizer = "adam";
  std::size_t step_opt_batch_stride = 1;
  double step_grad_clip = 0.0;
  bool da_freeze_average = false;
  double mass_shrink_kappa = 0.0;
  double mass_var_floor = 0.0;
  double mass_init_clamp = 0.0;
  bool step_init_heuristic = false;
  bool metric_drift_guard = false;
  double mass_combine_power = 0.0;
  double metric_collapse_reset = 0.0;
  double metric_stall_reset = 0.0;
  std::size_t anti_windup = 0;
  std::size_t drift_iters = 0;
  std::size_t metric_window = 0;
  std::size_t metric_rank = 0;
  std::size_t metric_basis = 0;
  double early_exit_tol = 0.0;  // 0 = fixed warmup (default)
  std::size_t chains = 1;  // 1 = legacy single-chain loop; >1 = library
                           // multi-chain controller (adapt_with_stats)
  std::string chain_exec = "threads";  // W-30: multi-chain topology
  bool fixed_warmup = false;  // W-30: pin controller min_iter to budget
  bool early_exit = false;  // W-31: opt in to controller early exit
  double temporal_step_tol = 0.0;  // 0 = temporal gate off (multi-chain)
  std::size_t temporal_window = 50;
  std::size_t temporal_min_iter = 200;
  std::size_t pilot_burst = 0;  // W-28: 0 = pilot gate off (multi-chain)
  double pilot_rho1_max = 0.5;
  double pilot_rhat_max = 1.1;
  bool metric_full = false;
  double metric_auto = 0.0;
  double max_error_start = 0.0;
  std::size_t max_error_schedule_iters = 0;
  double da_gamma = default_warmup.da_gamma();
  double da_t0 = default_warmup.da_t0();
  double da_kappa = default_warmup.da_kappa();

  std::size_t max_trajectory_doublings =
      default_sampling.max_trajectory_doublings();
  std::size_t max_step_halvings = default_sampling.max_step_halvings();
  double max_hamiltonian_error = default_sampling.max_hamiltonian_error();
  std::size_t min_micro_steps = default_sampling.min_micro_steps();

  double init = 2.0;
  double step_size_init = 1.0;
  std::string init_file = "";

  std::string lib;
  std::string data;
  std::string output_file;

  // parse from command line with CLI11
  {
    CLI::App app{"Run WALNUTs on a Stan model"};

    app.add_option("--seed", seed, "Random seed (default randomize with clock)")
        ->default_val(seed);

    app.add_option("--warmup", num_warmup, "Number of warmup iterations")
        ->default_val(num_warmup)
        ->check(CLI::NonNegativeNumber);

    app.add_option("--samples", num_draws, "Number of samples to draw")
        ->default_val(num_draws)
        ->check(CLI::PositiveNumber);

    app.add_flag("--save-warmup", save_warmup,
                 "Pass this flag to save the warmup iterations as well.")
        ->default_val(save_warmup);

    app.add_option("--max-trajectory-doublings", max_trajectory_doublings,
                   "Maximum depth for Nuts trajectory doublings")
        ->default_val(max_trajectory_doublings)
        ->check(CLI::PositiveNumber);

    app.add_option("--max-step-halvings", max_step_halvings,
                   "Maximum depth for the step size adaptation")
        ->default_val(max_step_halvings)
        ->check(CLI::PositiveNumber);

    app.add_option("--min-micro-steps", min_micro_steps,
                   "Minimum micro steps per macro step")
        ->default_val(min_micro_steps)
        ->check(CLI::PositiveNumber);

    app.add_option("--max-hamiltonian-error", max_hamiltonian_error,
                   "Maximum error allowed in joint densities")
        ->default_val(max_hamiltonian_error)
        ->check(CLI::PositiveNumber);

    app.add_option("--init-file", init_file,
                   "Text file with the unconstrained initial position, one "
                   "coordinate per line (e.g. a Pathfinder draw)")
        ->default_val(init_file);

    app.add_option("--init", init,
                   "Range [-init,init] for uniform parameter initial values")
        ->default_val(init)
        ->check(CLI::NonNegativeNumber);

    app.add_option("--mass-init-count", mass_init_count,
                   "Initial count for the mass matrix adaptation")
        ->default_val(mass_init_count)
        ->check(CLI::Range(1.0, (std::numeric_limits<double>::max)()));

    app.add_option("--mass-additive-smoothing", mass_additive_smoothing,
                   "Additive smoothing for the mass matrix adaptation")
        ->default_val(mass_additive_smoothing)
        ->check(CLI::PositiveNumber);

    app.add_option("--max-macro-steps-target", max_macro_steps_target,
                   "Target number of macro steps")
        ->default_val(max_macro_steps_target)
        ->check(CLI::PositiveNumber);

    app.add_option("--step-size-init", step_size_init,
                   "Initial step size for the step size adaptation")
        ->default_val(step_size_init)
        ->check(CLI::PositiveNumber);

    app.add_option("--step-accept-rate-target", step_accept_rate_target,
                   "Target acceptance rate for the step size adaptation")
        ->default_val(step_accept_rate_target)
        ->check(CLI::Range((std::numeric_limits<double>::min)(), 1.0));

    app.add_option("--step-optimizer", step_optimizer,
                   "Step size adaptation optimizer: adam | da (dual averaging) "
                   "| dem (AdEMAMix) | belief (AdaBelief)")
        ->default_val(step_optimizer)
        ->check(CLI::IsMember({"adam", "da", "dem", "belief"}));

    app.add_option("--step-opt-batch-stride", step_opt_batch_stride,
                   "Update the step optimizer once per N acceptance "
                   "observations, using their mean")
        ->default_val(step_opt_batch_stride)
        ->check(CLI::PositiveNumber);

    app.add_option("--da-gamma", da_gamma, "Dual averaging shrinkage gamma")
        ->default_val(da_gamma);

    app.add_option("--da-t0", da_t0, "Dual averaging iteration offset t0")
        ->default_val(da_t0);

    app.add_option("--da-kappa", da_kappa,
                   "Dual averaging averaging exponent kappa")
        ->default_val(da_kappa);

    app.add_option("--step-grad-clip", step_grad_clip,
                   "Clip acceptance-statistic gradient impulse (e.g. 0.3; 0 "
                   "= off)")
        ->default_val(step_grad_clip);

    app.add_flag("--da-freeze-average", da_freeze_average,
                 "Dual averaging: use Polyak-Ruppert averaged log step size");

    app.add_option("--mass-shrink-kappa", mass_shrink_kappa,
                   "Shrink mass-matrix variance toward init with weight "
                   "n/(n+kappa) (Stan uses 5; 0 = off)")
        ->default_val(mass_shrink_kappa);

    app.add_option("--mass-init-clamp", mass_init_clamp,
                   "Clamp gradient-seeded initial masses to [1/clamp, clamp] "
                   "(e.g. 100; 0 = off)")
        ->default_val(mass_init_clamp);

    app.add_option("--metric-auto", metric_auto,
                   "Auto-select the rank-corrected metric per window: apply "
                   "when the singular-excess concentration in the top "
                   "directions is at most this threshold (spread spectra = "
                   "cross-correlated geometry; 0 = off; e.g. 0.5)")
        ->default_val(metric_auto);

    app.add_flag("--metric-full", metric_full,
                 "Use the exact low-rank mass operator in the hot loop "
                 "(requires --metric-rank; otherwise the rank correction is "
                 "folded into the diagonal)");

    app.add_option("--metric-rank", metric_rank,
                   "Low-rank correction rank folded into the diagonal metric; "
                   "0 = off; e.g. 5-20")
        ->default_val(metric_rank);

    app.add_option("--metric-basis", metric_basis,
                   "Basis rule for the low-rank metric: 0=windowed SVD "
                   "(default), 1=streaming power iteration, 2=Muon-style "
                   "Newton-Schulz polar, 3=MuonEq-style equilibrated polar")
        ->default_val(metric_basis)
        ->check(CLI::Range(0, 3));

    app.add_option("--early-exit-warmup", early_exit_tol,
                   "Temporal stabilization early-exit for single-chain "
                   "warmup: exit when successive 50-iter window mass "
                   "estimates agree within this l2 rel-diff (and step "
                   "within 0.3), after 200 iters; 0 = fixed warmup")
        ->default_val(early_exit_tol)
        ->check(CLI::NonNegativeNumber);

    app.add_option("--chains", chains,
                   "Number of chains; >1 runs the library multi-chain "
                   "warmup controller (fixed-budget warmup by default; "
                   "cross-chain convergence early exit only with "
                   "--early-exit or --temporal-step-tol > 0) in one "
                   "process")
        ->default_val(chains)
        ->check(CLI::PositiveNumber);

    app.add_option("--chain-exec", chain_exec,
                   "Multi-chain execution topology (W-30): threads = one "
                   "worker thread per chain with the event-driven "
                   "controller; serial = all chains round-robin on the "
                   "calling thread (deterministic observation points, "
                   "identical per-chain draws)")
        ->default_val(chain_exec)
        ->check(CLI::IsMember({"threads", "serial"}));

    app.add_flag(
        "--fixed-warmup", fixed_warmup,
        "Multi-chain: pin the controller's minimum warmup to the full "
        "--warmup budget, so the cross-chain criteria can only stop at the "
        "budget (deterministic warmup length; redundant while early exit "
        "is off — the W-31 default — but still meaningful with "
        "--early-exit)");

    app.add_flag(
        "--early-exit", early_exit,
        "Multi-chain: re-enable the controller's cross-chain convergence "
        "early exit (the pre-W-31 default). UNSAFE with the default "
        "tolerances (mass 1.0 / step 0.1): with good inits warmup stops "
        "at iteration 50-80 and post-warmup quality collapses on hard "
        "models (W-25: hier_2pl bulk-ESS 519 -> 61). Pass only for "
        "reproducing those experiments or with tolerances you have "
        "validated. Passing --temporal-step-tol > 0 also opts in.");

    app.add_option("--temporal-step-tol", temporal_step_tol,
                   "Multi-chain controller temporal step-drift gate: require "
                   "every chain's step size to drift less than this relative "
                   "amount across the last --temporal-window iters before "
                   "early exit (W-22; 0 = off; e.g. 0.05)")
        ->default_val(temporal_step_tol)
        ->check(CLI::NonNegativeNumber);

    app.add_option("--temporal-window", temporal_window,
                   "Window length (warmup iters) for the temporal step-drift "
                   "gate")
        ->default_val(temporal_window)
        ->check(CLI::PositiveNumber);

    app.add_option("--temporal-min-iter", temporal_min_iter,
                   "Minimum warmup iters before the temporal step-drift gate "
                   "may trigger early exit")
        ->default_val(temporal_min_iter);

    app.add_option("--pilot-burst", pilot_burst,
                   "W-28 pilot sampling-burst gate (multi-chain only): at "
                   "each candidate early exit, draw N pilot draws per chain "
                   "on the would-be-frozen sampler (separate rng streams, "
                   "discarded) and approve the exit only if the pilot's lp "
                   "lag-1 autocorrelation and split-half R-hat pass "
                   "--pilot-rho1-max / --pilot-rhat-max; otherwise warmup "
                   "resumes (0 = off; N must be even, e.g. 50)")
        ->default_val(pilot_burst)
        ->check(CLI::NonNegativeNumber);

    app.add_option("--pilot-rho1-max", pilot_rho1_max,
                   "Pilot gate: max per-chain lag-1 autocorrelation of lp "
                   "to approve an early exit")
        ->default_val(pilot_rho1_max);

    app.add_option("--pilot-rhat-max", pilot_rhat_max,
                   "Pilot gate: split-half cross-chain R-hat of lp below "
                   "this to approve an early exit")
        ->default_val(pilot_rhat_max);

    app.add_option("--metric-window", metric_window,
                   "Memoryless metric windows: reset the draw/score moment "
                   "accumulators every N warmup iterations (0 = off; "
                   "Fisher-HMC 'chopping', arXiv:2603.18845)")
        ->default_val(metric_window);

    app.add_option("--drift-iters", drift_iters,
                   "Drift-phase warmup: for the first N iterations, suspend "
                   "the Hamiltonian-error cap and the mass estimation "
                   "(identity metric) while the chain moves toward the "
                   "typical set (0 = off)")
        ->default_val(drift_iters);

    app.add_option("--max-error-start", max_error_start,
                   "Max-error schedule start: begin at this cap and decay to "
                   "--max-error over --max-error-iters (0 = off; e.g. 100)")
        ->default_val(max_error_start);

    app.add_option("--max-error-iters", max_error_schedule_iters,
                   "Iterations for the max-error schedule decay (0 = off)")
        ->default_val(max_error_schedule_iters);

    app.add_option("--anti-windup", anti_windup,
                   "During acceptance-statistic saturation (alpha ~ 0), pass "
                   "only 1 in N observations to the step optimizer (0 = off; "
                   "e.g. 8)")
        ->default_val(anti_windup);

    app.add_option("--metric-stall-reset", metric_stall_reset,
                   "Reset mass estimators to seeds when max coordinate "
                   "movement over 100 warmup iterations is below this "
                   "(0 = off; e.g. 1e-3)")
        ->default_val(metric_stall_reset);

    app.add_option("--metric-collapse-reset", metric_collapse_reset,
                   "Reset mass estimators to seeds when observed draw "
                   "variance falls below this fraction of the metric-implied "
                   "variance (0 = off; e.g. 0.01)")
        ->default_val(metric_collapse_reset);

    app.add_option("--mass-combine-power", mass_combine_power,
                   "Power-mean order for combining Var_draw and 1/Var_score "
                   "into the mass estimate (0 = geometric [default], 1 = "
                   "arithmetic, >=64 = max; higher resists collapse)")
        ->default_val(mass_combine_power);

    app.add_flag("--metric-drift-guard", metric_drift_guard,
                 "Aggregate draw/score variances with their seeds in log "
                 "space (guards metric during early drift)");

    app.add_flag("--step-init-heuristic", step_init_heuristic,
                 "Find initial step size with a Stan-style doubling/halving "
                 "probe instead of a fixed value");

    app.add_option("--mass-var-floor", mass_var_floor,
                   "Elementwise floor for draw/score variances (e.g. 1e-3; "
                   "0 = off)")
        ->default_val(mass_var_floor);

    app.add_option("--step-learning-rate", step_learning_rate,
                   "Learning rates for step adaptation")
        ->default_val(step_learning_rate)
        ->check(CLI::PositiveNumber);

    app.add_option("--step-gradient-decay", step_gradient_decay,
                   "Decay rate of gradient moving average for step adaptation")
        ->default_val(step_gradient_decay)
        ->check(CLI::Range((std::numeric_limits<double>::min)(), 1.0));

    app.add_option(
           "--step-sq-gradient-decay", step_sq_gradient_decay,
           "Decay rate of squared gradient moving average for step adaptation")
        ->default_val(step_sq_gradient_decay)
        ->check(CLI::Range((std::numeric_limits<double>::min)(), 1.0));

    app.add_option("--step-stabilization", step_stabilization,
                   "Update stabilization term for step size adaptation")
        ->default_val(step_stabilization)
        ->check(CLI::PositiveNumber);

    app.add_option("--step-learn-rate-decay", step_learn_rate_decay,
                   "Decay rate of exponent for step adaptation")
        ->default_val(step_learn_rate_decay)
        ->check(CLI::Range((std::numeric_limits<double>::min)(), 1.0));

    app.add_option("model", lib,
                   "Path to the Stan model library (.so from BridgeStan)")
        ->required()
        ->check(CLI::ExistingFile);

    app.add_option("data", data,
                   "Path to the Stan model data (.json, optional)")
        ->check(CLI::ExistingFile);

    app.add_option("--output", output_file, "Output file for the draws")
        ->check(CLI::NonexistentPath);

    CLI11_PARSE(app, argc, argv);
  }

  DynamicStanModel model(lib.c_str(), data.c_str(), seed);

  walnutpie::WarmupConfigBuilder warmup_builder =
      walnutpie::WarmupConfigBuilder()
          .mass_init_count(mass_init_count)
          .mass_additive_smoothing(mass_additive_smoothing)
          .max_macro_steps_target(max_macro_steps_target)
          .step_accept_rate_target(step_accept_rate_target)
          .step_learning_rate(step_learning_rate)
          .step_gradient_decay(step_gradient_decay)
          .step_sq_gradient_decay(step_sq_gradient_decay)
          .step_stabilization(step_stabilization)
          .step_learn_rate_decay(step_learn_rate_decay)
          .da_gamma(da_gamma)
          .da_t0(da_t0)
          .da_kappa(da_kappa)
          .step_opt_batch_stride(step_opt_batch_stride)
          .step_grad_clip(step_grad_clip)
          .da_freeze_average(da_freeze_average)
          .mass_shrink_kappa(mass_shrink_kappa)
          .mass_var_floor(mass_var_floor)
          .mass_init_clamp(mass_init_clamp)
          .metric_drift_guard(metric_drift_guard)
          .mass_combine_power(mass_combine_power)
          .metric_collapse_reset(metric_collapse_reset)
          .metric_stall_reset(metric_stall_reset)
          .anti_windup_pass_rate(anti_windup)
          .drift_iters(drift_iters)
          .max_error_schedule(max_error_start, max_error_schedule_iters)
          .metric_window(metric_window)
          .metric_rank(metric_rank)
          .metric_basis(metric_basis)
          .metric_full(metric_full)
          .metric_auto(metric_auto);
  if (chains > 1) {
    // Multi-chain: the controller bounds warmup (min 50, max num_warmup).
    // --fixed-warmup pins min to the budget so the cross-chain criteria
    // can only stop at max_iter (deterministic length for the W-30 gates).
    warmup_builder.min_max_iter(
        fixed_warmup ? num_warmup : std::min<std::size_t>(50, num_warmup),
        num_warmup);
    // W-31: convergence-based early exit is opt-in — an explicit
    // --early-exit, or a positive temporal tolerance (which is itself an
    // explicit request for gated early exit; keeps the W-25/W-28 arm
    // command lines reproducible verbatim).
    warmup_builder.allow_early_exit(early_exit || temporal_step_tol > 0.0);
  }
  walnutpie::WarmupConfig warmup_cfg =
      warmup_builder.temporal_step_drift_tol(temporal_step_tol)
          .temporal_window(temporal_window)
          .temporal_min_iter(temporal_min_iter)
          .build();

  walnutpie::SamplingConfig sample_cfg =
      walnutpie::SamplingConfigBuilder()
          .max_trajectory_doublings(max_trajectory_doublings)
          .max_step_halvings(max_step_halvings)
          .max_hamiltonian_error(max_hamiltonian_error)
          .min_micro_steps(min_micro_steps)
          .build();

  unique_bs_rng rng = model.make_rng(seed);

  if (pilot_burst > 0 && chains <= 1) {
    throw std::invalid_argument("--pilot-burst requires --chains > 1");
  }
  if (chains <= 1 && (fixed_warmup || chain_exec != "threads")) {
    // Fail loudly rather than silently no-op (the CLI dispatch lesson).
    throw std::invalid_argument(
        "--chain-exec and --fixed-warmup require --chains > 1");
  }

  if (chains <= 1 && (early_exit || temporal_step_tol > 0.0)) {
    // Fail loudly rather than silently no-op: --early-exit and
    // --temporal-step-tol configure the multi-chain controller only (the
    // single-chain analogue is --early-exit-warmup).
    throw std::invalid_argument(
        "--early-exit, --temporal-step-tol, --temporal-window and "
        "--temporal-min-iter require --chains > 1");
  }

  if (chains > 1) {
    // W-25 multi-chain path: library controller, per-chain models.
    if (early_exit_tol > 0.0 || step_init_heuristic || mass_init_clamp > 0.0) {
      throw std::invalid_argument(
          "--early-exit-warmup, --step-init-heuristic and --mass-init-clamp "
          "are single-chain-only flags");
    }
    if (pilot_burst > 0 && (pilot_burst < 2 || pilot_burst % 2 != 0)) {
      throw std::invalid_argument(
          "--pilot-burst must be 0 (off) or an even number >= 2 (the gate "
          "splits each chain's pilot draws in half for the R-hat proxy)");
    }
    if (pilot_burst > 0 && !(early_exit || temporal_step_tol > 0.0)) {
      // Fail loudly rather than silently no-op (the CLI dispatch
      // lesson): with the W-31 safe default there are no candidate
      // early exits for the pilot gate to inspect.
      throw std::invalid_argument(
          "--pilot-burst requires an early-exit enabler: --early-exit or "
          "--temporal-step-tol > 0 (with early exit off, warmup always "
          "runs to the --warmup budget and pilots can never fire)");
    }
    if (!init_file.empty() && init_file.find("{c}") == std::string::npos) {
      throw std::invalid_argument(
          "--init-file in multi-chain mode must contain {c}");
    }
    if (!output_file.empty() && output_file.find("{c}") ==
                                    std::string::npos) {
      throw std::invalid_argument(
          "--output in multi-chain mode must contain {c}");
    }
    auto run_multi = [&](auto opt_tag) {
      using Opt = typename decltype(opt_tag)::type;
      run_walnuts_multi<Opt>(
          lib, data, static_cast<unsigned int>(seed), chains, num_warmup,
          num_draws, save_warmup, warmup_cfg, sample_cfg, init,
          step_size_init, init_file, output_file, pilot_burst,
          pilot_rho1_max, pilot_rhat_max, chain_exec == "serial");
    };
    if (step_optimizer == "adam") {
      run_multi(std::type_identity<walnutpie::detail::Adam>{});
    } else if (step_optimizer == "da") {
      run_multi(std::type_identity<walnutpie::detail::DualAveraging>{});
    } else if (step_optimizer == "dem") {
      run_multi(std::type_identity<walnutpie::detail::AdEMAMix>{});
    } else {
      run_multi(std::type_identity<walnutpie::detail::AdaBelief>{});
    }
    return 0;
  }

  auto init_positions = [&]() {
    if (!init_file.empty()) {
      // plain text: one unconstrained coordinate per line
      std::ifstream in(init_file);
      if (!in) {
        throw std::invalid_argument("cannot open --init-file: " + init_file);
      }
      std::vector<double> vals;
      double v;
      while (in >> v) {
        vals.push_back(v);
      }
      if (vals.size() != model.unconstrained_dimensions()) {
        throw std::invalid_argument(
            "--init-file dimension mismatch: file has " +
            std::to_string(vals.size()) + ", model has " +
            std::to_string(model.unconstrained_dimensions()));
      }
      return Eigen::VectorXd(Eigen::VectorXd::Map(vals.data(), vals.size()));
    }
    return model.initialize(nullptr, rng, init);
  }();

  auto init_cfg =
      walnutpie::InitConfigBuilder{1, model.unconstrained_dimensions()}
          .step_sizes(step_size_init)
          .positions(init_positions);

  auto res = [&]() {
    using walnutpie::detail::Adam;
    using walnutpie::detail::AntiWindupAdapter;
    using walnutpie::detail::AdaBelief;
    using walnutpie::detail::AdEMAMix;
    using walnutpie::detail::BatchedAdapter;
    using walnutpie::detail::ClippedAdapter;
    using walnutpie::detail::DualAveraging;
    // dispatch: base optimizer, optional batching, optional clipping
    auto run_base = [&](auto opt_tag) -> StanHandler {
      using Opt = typename decltype(opt_tag)::type;
      auto extra = std::make_pair(mass_init_clamp, step_init_heuristic);
      // --anti-windup selects the AntiWindupAdapter wrapper around whatever
      // optimizer/composition the other flags chose; the wrapper's pass rate
      // comes from warmup_cfg (StepAdapterFactory<AntiWindupAdapter<Inner>>).
      if (anti_windup > 0) {
        if (step_opt_batch_stride > 1 && step_grad_clip > 0.0) {
          return run_walnuts<
              AntiWindupAdapter<ClippedAdapter<BatchedAdapter<Opt>>>>(
              model, seed, init_cfg, num_warmup, num_draws, save_warmup,
              warmup_cfg, sample_cfg, extra.first, extra.second, early_exit_tol);
        } else if (step_opt_batch_stride > 1) {
          return run_walnuts<AntiWindupAdapter<BatchedAdapter<Opt>>>(
              model, seed, init_cfg, num_warmup, num_draws, save_warmup,
              warmup_cfg, sample_cfg, extra.first, extra.second, early_exit_tol);
        } else if (step_grad_clip > 0.0) {
          return run_walnuts<AntiWindupAdapter<ClippedAdapter<Opt>>>(
              model, seed, init_cfg, num_warmup, num_draws, save_warmup,
              warmup_cfg, sample_cfg, extra.first, extra.second, early_exit_tol);
        }
        return run_walnuts<AntiWindupAdapter<Opt>>(
            model, seed, init_cfg, num_warmup, num_draws, save_warmup,
            warmup_cfg, sample_cfg, extra.first, extra.second, early_exit_tol);
      }
      if (step_opt_batch_stride > 1 && step_grad_clip > 0.0) {
        return run_walnuts<ClippedAdapter<BatchedAdapter<Opt>>>(
            model, seed, init_cfg, num_warmup, num_draws, save_warmup,
            warmup_cfg, sample_cfg, extra.first, extra.second, early_exit_tol);
      } else if (step_opt_batch_stride > 1) {
        return run_walnuts<BatchedAdapter<Opt>>(
            model, seed, init_cfg, num_warmup, num_draws, save_warmup,
            warmup_cfg, sample_cfg, extra.first, extra.second, early_exit_tol);
      } else if (step_grad_clip > 0.0) {
        return run_walnuts<ClippedAdapter<Opt>>(
            model, seed, init_cfg, num_warmup, num_draws, save_warmup,
            warmup_cfg, sample_cfg, extra.first, extra.second, early_exit_tol);
      }
      return run_walnuts<Opt>(model, seed, init_cfg, num_warmup, num_draws,
                              save_warmup, warmup_cfg, sample_cfg, extra.first,
                              extra.second, early_exit_tol);
    };
    if (step_optimizer == "adam") {
      return run_base(std::type_identity<Adam>{});
    } else if (step_optimizer == "da") {
      return run_base(std::type_identity<DualAveraging>{});
    } else if (step_optimizer == "dem") {
      return run_base(std::type_identity<AdEMAMix>{});
    } else {
      return run_base(std::type_identity<AdaBelief>{});
    }
  }();

  res.summarize();
  res.write_csv(output_file);

  return 0;
}
