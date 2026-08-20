#include <walnutpie.hpp>
#include <walnutpie/load_stan.hpp>
#include <walnutpie/warmup_heuristics.hpp>

#include <CLI/CLI.hpp>
#include <fstream>
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
    static std::size_t it = 0;
    if (const char* dbg = std::getenv("WALNUTPIE_DEBUG_WARMUP")) {
      if (it % std::max(1, atoi(dbg)) == 0) {
        std::cout << "[warmup it " << it << "] lp=" << lp
                  << " step=" << step_size
                  << " invm[0]=" << (diag_inv_mass.size() ? diag_inv_mass[0] : -1)
                  << " pos[0]=" << (position.size() ? position[0] : 0)
                  << " pos[last]=" << (position.size() ? position[position.size()-1] : 0)
                  << std::endl;
      }
    }
    ++it;
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
};

template <typename Opt>
StanHandler run_walnuts(DynamicStanModel& model, unsigned int seed,
                        walnutpie::InitConfigBuilder& init_builder,
                        std::size_t num_warmup, std::size_t num_draws,
                        bool save_warmup, walnutpie::WarmupConfig& warmup_cfg,
                        walnutpie::SamplingConfig& sample_cfg,
                        double mass_init_clamp = 0.0,
                        bool step_init_heuristic = false) {
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
  if (step_init_heuristic) {
    const auto inv_mass = inits.mass().array().inverse().matrix().eval();
    double eps = walnutpie::detail::find_reasonable_step(
        logp, inits.position(), inv_mass, inits.step_size());
    inits = walnutpie::InitChainConfig(eps, inits.position(), inits.mass());
    std::cout << "Heuristic initial step size: " << eps << std::endl;
  }

  std::mt19937_64 rng{seed};
  walnutpie::AdaptiveWalnuts<decltype(logp), decltype(rng), StanHandler, Opt>
      walnuts(
      rng, storage, logp, inits, warmup_cfg, sample_cfg);
  for (std::size_t w = 0; w < num_warmup; ++w) {
    walnuts();
  }
  end_timing();

  // N post-warmup draws
  auto sampler = walnuts.sampler();  // freeze tuning
  std::cout << "Adaptation completed." << std::endl;
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
        ->default_val(step_accept_rate_target);

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

  walnutpie::WarmupConfig warmup_cfg =
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
          .metric_drift_guard(metric_drift_guard)
          .mass_combine_power(mass_combine_power)
          .metric_collapse_reset(metric_collapse_reset)
          .metric_stall_reset(metric_stall_reset)
          .anti_windup_pass_rate(anti_windup)
          .drift_iters(drift_iters)
          .max_error_schedule(max_error_start, max_error_schedule_iters)
          .metric_window(metric_window)
          .build();

  walnutpie::SamplingConfig sample_cfg =
      walnutpie::SamplingConfigBuilder()
          .max_trajectory_doublings(max_trajectory_doublings)
          .max_step_halvings(max_step_halvings)
          .max_hamiltonian_error(max_hamiltonian_error)
          .min_micro_steps(min_micro_steps)
          .build();

  unique_bs_rng rng = model.make_rng(seed);

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
    using walnutpie::detail::AdaBelief;
    using walnutpie::detail::AdEMAMix;
    using walnutpie::detail::BatchedAdapter;
    using walnutpie::detail::ClippedAdapter;
    using walnutpie::detail::DualAveraging;
    // dispatch: base optimizer, optional batching, optional clipping
    auto run_base = [&](auto opt_tag) -> StanHandler {
      using Opt = typename decltype(opt_tag)::type;
      auto extra = std::make_pair(mass_init_clamp, step_init_heuristic);
      if (anti_windup > 0) {
        using AW = walnutpie::detail::ClippedAdapter<
            walnutpie::detail::AntiWindupAdapter<Opt>>;
        return run_walnuts<AW>(model, seed, init_cfg, num_warmup, num_draws,
                               save_warmup, warmup_cfg, sample_cfg,
                               mass_init_clamp, step_init_heuristic);
      }
      if (step_opt_batch_stride > 1 && step_grad_clip > 0.0) {
        return run_walnuts<ClippedAdapter<BatchedAdapter<Opt>>>(
            model, seed, init_cfg, num_warmup, num_draws, save_warmup,
            warmup_cfg, sample_cfg, extra.first, extra.second);
      } else if (step_opt_batch_stride > 1) {
        return run_walnuts<BatchedAdapter<Opt>>(
            model, seed, init_cfg, num_warmup, num_draws, save_warmup,
            warmup_cfg, sample_cfg, extra.first, extra.second);
      } else if (step_grad_clip > 0.0) {
        return run_walnuts<ClippedAdapter<Opt>>(
            model, seed, init_cfg, num_warmup, num_draws, save_warmup,
            warmup_cfg, sample_cfg, extra.first, extra.second);
      }
      return run_walnuts<Opt>(model, seed, init_cfg, num_warmup, num_draws,
                              save_warmup, warmup_cfg, sample_cfg, extra.first,
                              extra.second);
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
