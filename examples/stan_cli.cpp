#include <walnutpie.hpp>
#include <walnutpie/load_stan.hpp>
#include "micro_guard.hpp"

#include <CLI/CLI.hpp>
#include <Eigen/Dense>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <utility>
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

template <typename Model>
class StanHandler {
 public:
  StanHandler(Model& model, unsigned int seed, std::size_t num_warmup,
              std::size_t num_draws, bool save_warmup,
              const MicroGuardSpec& guard = {})
      : model_(model),
        rng_(model.make_rng(seed + 1)),
        draws_(model.constrained_dimensions(),
               num_draws + static_cast<std::size_t>(save_warmup) * num_warmup),
        save_warmup_(save_warmup),
        probe_(guard) {}

  void on_sample(const Eigen::VectorXd& position, double lp) {
    probe_.observe(position);
    model_.constrain_draw(position, draws_.col(n_), rng_);
    n_++;
  }

  void on_warmup(const Eigen::VectorXd& position, double lp, double step_size,
                 const Eigen::VectorXd& diag_inv_mass) {
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

  MicroGuardResult guard_result() const { return probe_.result(); }
  const Eigen::MatrixXd& draws() const { return draws_; }
  std::size_t size() const { return static_cast<std::size_t>(n_); }

  void summarize() {
    auto names = model_.param_names();
    if (n_ != draws_.cols()) {
      throw std::logic_error("incomplete attempt output");
    }
    ::summarize(names, draws_);
  }

  void write_csv(std::string& output_file) {
    auto names = model_.param_names();
    if (n_ != draws_.cols()) {
      throw std::logic_error("incomplete attempt output");
    }
    ::write_draws(output_file, names, draws_);
  }

 private:
  Model& model_;
  decltype(std::declval<Model&>().make_rng(0)) rng_;
  Eigen::MatrixXd draws_;
  bool save_warmup_;
  Eigen::Index n_ = 0;
  MicroGuardProbe probe_;
};

template <typename Model>
StanHandler<Model> run_walnuts(
    Model& model, unsigned int seed, walnutpie::InitConfigBuilder& init_builder,
    std::size_t num_warmup, std::size_t num_draws, bool save_warmup,
    walnutpie::WarmupConfig& warmup_cfg, walnutpie::SamplingConfig& sample_cfg,
    const MicroGuardSpec& micro_guard = MicroGuardSpec{},
    MicroGuardResult* micro_guard_result = nullptr,
    std::mt19937_64* final_rng = nullptr, bool collect_diagnostics = false) {
  collect_diagnostics = collect_diagnostics || micro_guard.armed;
  using Clock = std::chrono::high_resolution_clock;
  auto elapsed_seconds = [](auto t) {
    return std::chrono::duration<double>(Clock::now() - t).count();
  };
  double logp_time = 0.0;
  std::size_t logp_count = 0;
  std::size_t thrown_calls = 0, nonfinite_returns = 0;
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
    if (collect_diagnostics) {
      // Completed calls include errors converted internally by BridgeStan.
      // Nonfinite returns are not necessarily BridgeStan errors.
      std::cout << "evaluation categories: completed=" << logp_count
                << " nonfinite_returns=" << nonfinite_returns
                << " throws=" << thrown_calls << "\n";
    }
    std::cout << std::endl;
  };

  StanHandler<Model> storage(model, seed, num_warmup, num_draws, save_warmup,
                             micro_guard);

  auto logp = [&](const Eigen::VectorXd& x, double& lp, Eigen::VectorXd& grad) {
    auto start = Clock::now();
    if (!collect_diagnostics) {
      model.logp_grad(x, lp, grad);
      logp_time += elapsed_seconds(start);
      ++logp_count;
      return;
    }
    try {
      model.logp_grad(x, lp, grad);
    } catch (...) {
      ++thrown_calls;
      logp_time += elapsed_seconds(start);
      throw;
    }
    logp_time += elapsed_seconds(start);
    ++logp_count;
    if (!std::isfinite(lp) || !grad.allFinite()) {
      ++nonfinite_returns;
    }
  };

  auto init_cfg =
      init_builder.masses(logp, warmup_cfg.mass_additive_smoothing()).build();
  auto inits = init_cfg.init_chain_config(0);

  std::mt19937_64 rng{seed};
  walnutpie::AdaptiveWalnuts walnuts(rng, storage, logp, inits, warmup_cfg,
                                     sample_cfg);
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
  thrown_calls = 0;
  nonfinite_returns = 0;
  global_start = Clock::now();
  // Probe only post-warmup unconstrained positions. The caller discards a
  // triggered attempt and explicitly disables this check on the one retry.
  const std::size_t guard_probe_n = micro_guard.armed ? micro_guard.probe : 0;
  for (std::size_t n = 0; n < num_draws; ++n) {
    sampler();
    if (guard_probe_n > 0 && n + 1 == guard_probe_n) {
      const std::size_t unique = storage.guard_result().unique;
      if (micro_guard_result != nullptr) {
        micro_guard_result->probe = guard_probe_n;
        micro_guard_result->unique = unique;
        micro_guard_result->pinned = unique < micro_guard.min_unique;
      }
      if (unique < micro_guard.min_unique) {
        break;
      }
    }
  }
  end_timing();

  if (final_rng) {
    *final_rng = rng;
  }
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

  std::size_t max_trajectory_doublings =
      default_sampling.max_trajectory_doublings();
  std::size_t max_step_halvings = default_sampling.max_step_halvings();
  double max_hamiltonian_error = default_sampling.max_hamiltonian_error();
  std::size_t min_micro_steps = default_sampling.min_micro_steps();
  bool min_micro_guard = false;  // W-82: only meaningful with min-micro > 1
  std::size_t min_micro_guard_probe = 50;
  std::size_t min_micro_guard_min_unique = 25;

  double init = 2.0;
  double step_size_init = 1.0;

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

    app.add_flag("--min-micro-guard", min_micro_guard,
                 "Guard against pinned chains when --min-micro-steps > 1: "
                 "after the first --min-micro-guard-probe sampling draws, if "
                 "fewer than --min-micro-guard-min-unique unique unconstrained "
                 "rows were drawn, discard the attempt and re-run the whole "
                 "chain from the same init and seed with min-micro-steps "
                 "forced to 1 (the final output is the re-run's)")
        ->default_val(min_micro_guard);

    app.add_option("--min-micro-guard-probe", min_micro_guard_probe,
                   "Number of initial sampling draws examined by "
                   "--min-micro-guard")
        ->default_val(min_micro_guard_probe)
        ->check(CLI::PositiveNumber);

    app.add_option("--min-micro-guard-min-unique", min_micro_guard_min_unique,
                   "Unique unconstrained-position threshold that trips "
                   "--min-micro-guard")
        ->default_val(min_micro_guard_min_unique)
        ->check(CLI::PositiveNumber);

    app.add_option("--max-hamiltonian-error", max_hamiltonian_error,
                   "Maximum error allowed in joint densities")
        ->default_val(max_hamiltonian_error)
        ->check(CLI::PositiveNumber);

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
    if (!min_micro_guard && (app.count("--min-micro-guard-probe") ||
                             app.count("--min-micro-guard-min-unique"))) {
      std::cerr << "guard probe/threshold options require --min-micro-guard\n";
      return 2;
    }
    try {
      validate_micro_guard({min_micro_guard && min_micro_steps > 1,
                            min_micro_guard_probe, min_micro_guard_min_unique},
                           num_draws);
    } catch (const std::invalid_argument& error) {
      std::cerr << error.what() << "\n";
      return 2;
    }
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
          .build();

  walnutpie::SamplingConfig sample_cfg =
      walnutpie::SamplingConfigBuilder()
          .max_trajectory_doublings(max_trajectory_doublings)
          .max_step_halvings(max_step_halvings)
          .max_hamiltonian_error(max_hamiltonian_error)
          .min_micro_steps(min_micro_steps)
          .build();

  // W-82 (min-micro guard): armed only when explicitly requested AND
  // min-micro-steps > 1 — otherwise the run must be semantically
  // identical to the unguarded CLI (W-54 draw-neutrality).
  MicroGuardSpec micro_guard;
  micro_guard.armed = min_micro_guard && min_micro_steps > 1;
  micro_guard.probe = min_micro_guard_probe;
  micro_guard.min_unique = min_micro_guard_min_unique;

  // Keep one exact initial vector. Rebuild consumed builders, sampler RNG and
  // GQ RNG for every attempt. This assumes a deterministic model density.
  const auto overall_start = std::chrono::steady_clock::now();
  std::optional<Eigen::VectorXd> initial_position;
  if (micro_guard.armed) {
    unique_bs_rng init_rng = model.make_rng(seed);
    initial_position = model.initialize(nullptr, init_rng, init);
  }
  std::size_t attempt = 0;
  auto run_chain =
      [&](walnutpie::SamplingConfig& cfg, const MicroGuardSpec& guard,
          MicroGuardResult* result) -> StanHandler<DynamicStanModel> {
    if (micro_guard.armed) {
      std::cout << "[min-micro-guard] attempt=" << attempt++
                << " min_micro_steps=" << cfg.min_micro_steps() << "\n";
    }
    auto init_cfg =
        walnutpie::InitConfigBuilder{1, model.unconstrained_dimensions()}
            .step_sizes(step_size_init);
    if (initial_position) {
      init_cfg.positions(*initial_position);
    } else {
      unique_bs_rng init_rng = model.make_rng(seed);
      init_cfg.positions(model.initialize(nullptr, init_rng, init));
    }
    return run_walnuts(model, seed, init_cfg, num_warmup, num_draws,
                       save_warmup, warmup_cfg, cfg, guard, result, nullptr,
                       micro_guard.armed);
  };
  auto mm1_cfg = walnutpie::SamplingConfigBuilder()
                     .max_trajectory_doublings(max_trajectory_doublings)
                     .max_step_halvings(max_step_halvings)
                     .max_hamiltonian_error(max_hamiltonian_error)
                     .min_micro_steps(1)
                     .build();
  auto res = run_with_micro_guard(
      run_chain, sample_cfg, mm1_cfg, micro_guard,
      [&](const MicroGuardResult& result) {
        std::cout << "[min-micro-guard] pin signature: " << result.unique << "/"
                  << result.probe << " unique unconstrained positions; "
                  << "discarded_warmup=" << num_warmup
                  << " discarded_samples=" << result.probe
                  << "; one MM1 restart (heuristic, not convergence proof)\n";
      });
  if (micro_guard.armed) {
    const double elapsed = std::chrono::duration<double>(
                               std::chrono::steady_clock::now() - overall_start)
                               .count();
    std::cout << "[min-micro-guard] restart_count=" << (attempt - 1)
              << " retained_samples=" << num_draws
              << " total_attempt_wall_s=" << elapsed
              << " (includes initialization and discarded work)\n";
  }

  res.summarize();
  res.write_csv(output_file);

  return 0;
}
