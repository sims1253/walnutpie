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
#include <sstream>
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

// W-42 init guard: refuse to start a chain at a non-finite-logp position
// (file init fails immediately; random init retries up to --init-tries
// draws). W-78 extends the same guard points to the EVAL-THROWS class:
// a model evaluation that fails at the init position (the kronecker_gp
// LKJ-Cholesky-boundary dead inits) is treated identically to a
// non-finite logp — file init aborts loudly naming the exception,
// random init counts it as a rejected draw.

static std::string format_double(double v) {
  std::ostringstream oss;
  oss << v;
  return oss.str();
}

// W-42: fail before warmup on a non-finite init logp. Such a start pins
// the chain (NaN acceptance statistic, NaN adapter) and wastes the whole
// budget; the error message spells out the pathology.
[[noreturn]] static void throw_nonfinite_init(std::size_t chain,
                                              const std::string& source,
                                              double lp) {
  std::string msg = "initial position has non-finite log probability: chain " +
                    std::to_string(chain) + ", init source: " + source +
                    ", logp at init: " + format_double(lp);
  std::cerr << "WALNUTS ERROR (init guard): " << msg << "\n"
            << "  A chain started at a non-finite-logp position cannot adapt:\n"
               "  the within-orbit acceptance statistic is NaN, the step\n"
               "  adapter NaNs at its first update and the chain stays pinned\n"
               "  for the whole warmup budget (W-36/W-41 pathology). Warmup\n"
               "  was refused BEFORE starting; no budget was consumed.\n"
               "  Provide an init draw with finite logp (or use random init,\n"
               "  which retries: see --init-tries)."
            << std::endl;
  throw std::invalid_argument(msg);
}

// W-78: fail before warmup when the model EVALUATION itself fails at the
// init position (throws / returns an error state), not merely a non-finite
// logp value. Root cause class: the kronecker_gp dead init on the
// LKJ-Cholesky constraint boundary — every logp_grad eval throws (log rate
// NaN inside poisson_log_lpmf; eigenvectors_sym rejects the degenerate
// kernel). BridgeStan maps the throw to logp = -inf, so the W-42 value
// guard sees it, but without naming the exception; and a poisoned
// gradient with finite logp slips past a logp-only check entirely. Either
// way the chain "zombies" (the per-eval "Error in logp_grad" loop, pinned
// or garbage draws through the whole budget). Treated exactly like the
// non-finite case: loud abort naming the failure, warmup refused.
[[noreturn]] static void throw_eval_failed_init(std::size_t chain,
                                                const std::string& source,
                                                const std::string& what) {
  std::string msg = "model evaluation failed at initial position: chain " +
                    std::to_string(chain) + ", init source: " + source +
                    ", failure: " + what;
  std::cerr << "WALNUTS ERROR (init guard): " << msg << "\n"
            << "  The Stan model threw / returned an error when evaluated at\n"
               "  the initial position (e.g. the kronecker_gp LKJ-Cholesky\n"
               "  boundary class: every eval fails, so the chain can only\n"
               "  zombie on the error loop instead of sampling). Warmup was\n"
               "  refused BEFORE starting; no budget was consumed.\n"
               "  Provide an init draw the model can evaluate (or use random\n"
               "  init, which retries: see --init-tries)."
            << std::endl;
  throw std::invalid_argument(msg);
}

// W-42: random-init rejection loop, Stan convention (draw, check logp
// finite, retry). BridgeStan's param_initialize implements the same
// protocol internally; it is called with max_tries=1 (one draw, throws
// instead of redrawing) so this loop owns the budget, the audit lines,
// and the failure. Each attempt consumes exactly one draw from the
// chain's init stream, in order and before any warmup consumption
// (warmup uses the separate std::mt19937_64 stream), so the accepted
// position matches the stock binary's.
// W-78: a draw whose EVALUATION fails counts as rejected too — the model
// throwing at the draw (load_stan maps it to logp = -inf and reports the
// BridgeStan error text) or returning a poisoned gradient with a finite
// logp (Stan's own init-validity rule: finite logp AND finite gradient).
// Same failure family as the kronecker_gp LKJ-Cholesky-boundary class.
static Eigen::VectorXd initialize_finite(DynamicStanModel& model,
                                         unique_bs_rng& rng, double init_radius,
                                         unsigned int seed, std::size_t chain,
                                         std::size_t tries) {
  std::string last_err;
  for (std::size_t attempt = 1; attempt <= tries; ++attempt) {
    Eigen::VectorXd pos;
    double lp = -std::numeric_limits<double>::infinity();
    Eigen::VectorXd grad;
    std::string eval_err;
    bool rejected = false;
    try {
      // one draw per call: the inner protocol throws instead of redrawing
      pos = model.initialize(nullptr, rng, init_radius, 1);
      // Re-check the draw the inner layer accepted: a model error maps to
      // lp = -inf (load_stan), and this call stays outside the timing
      // stanzas.
      model.logp_grad(pos, lp, grad, &eval_err);
      rejected = !std::isfinite(lp) || !eval_err.empty() ||
                 (grad.size() > 0 && !grad.allFinite());
    } catch (const std::exception& e) {
      // The inner layer's rejection ("Initialization failed"), a bridgestan
      // boundary throw, or any other per-draw failure counts as one
      // rejected draw.
      rejected = true;
      last_err = e.what();
    }
    if (!rejected) {
      return pos;
    }
    // W-78: name the failure when the draw's evaluation errored or the
    // gradient is poisoned; a plain non-finite logp keeps the W-42 audit
    // line format.
    std::string detail;
    if (!eval_err.empty()) {
      detail = "; eval failed: " + eval_err;
    } else if (grad.size() > 0 && !grad.allFinite()) {
      detail = "; gradient non-finite at init";
    }
    std::cerr << "WALNUTS WARNING (init guard): random init draw rejected "
              << "(chain " << chain << ", seed " << seed << ", attempt "
              << attempt << "/" << tries
              << ", logp=" << format_double(lp) << detail
              << (last_err.empty() ? std::string()
                                   : "; inner: " + last_err)
              << "); redrawing" << std::endl;
    last_err.clear();
  }
  std::string msg =
      "random initialization failed: all " + std::to_string(tries) +
      " draws have non-finite log probability or failed to evaluate "
      "(chain " +
      std::to_string(chain) + ", seed " + std::to_string(seed) + ")";
  std::cerr << "WALNUTS ERROR (init guard): " << msg << "\n"
            << "  Increase --init-tries, adjust --init, or provide "
               "--init-file draws from the typical set."
            << std::endl;
  throw std::invalid_argument(msg);
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
                        bool step_init_heuristic = false,
                        double early_exit_tol = 0.0,
                        const std::string& init_file = "") {
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

  // W-78: error text from the last model evaluation (empty on success).
  // Filled by every logp_grad below via load_stan's eval-error out-param;
  // at the init guard point it names the exception behind a failed init
  // eval (the kronecker_gp LKJ-Cholesky-boundary class), which otherwise
  // surfaces only as the mapped logp = -inf.
  std::string init_eval_error;
  auto logp = [&](const Eigen::VectorXd& x, double& lp, Eigen::VectorXd& grad) {
    auto start = Clock::now();
    model.logp_grad(x, lp, grad, &init_eval_error);
    logp_time += elapsed_seconds(start);
    ++logp_count;
  };

  // W-42: masses() already evaluated the logp at each init position;
  // refuse a non-finite one before the step heuristic and the adapter
  // exist (no warmup consumption). W-78: an exception ESCAPING the
  // mass-seeding eval (load_stan throws only when bridgestan returns an
  // error with no message) is converted to the same loud init-guard
  // abort instead of an uncaught terminate.
  const std::string init_source =
      init_file.empty() ? "random init (seed " + std::to_string(seed) + ")"
                        : init_file;
  auto init_cfg = [&]() {
    try {
      return init_builder.masses(logp, warmup_cfg.mass_additive_smoothing(),
                                 false, mass_init_clamp)
          .build();
    } catch (const std::exception& e) {
      throw_eval_failed_init(0, init_source, e.what());
    }
  }();
  {
    const std::vector<double>& init_logps = init_cfg.init_logps();
    for (std::size_t c = 0; c < init_logps.size(); ++c) {
      // W-78: the init eval THREW / returned an error state. load_stan
      // maps the failure to logp = -inf, so W-42's check below would also
      // fire — this branch fires first to NAME the exception (kronecker_gp
      // LKJ-Cholesky-boundary root cause: the model errors at every eval
      // from this init). The CLI runs a single chain, so the last
      // evaluation's error text is exactly chain c's mass-seeding eval.
      if (!init_eval_error.empty()) {
        throw_eval_failed_init(c, init_source, init_eval_error);
      }
      if (!std::isfinite(init_logps[c])) {
        throw_nonfinite_init(c, init_source, init_logps[c]);
      }
      // W-78: finite logp but a poisoned gradient (the "logp stays
      // finite" miss variant). masses() seeds the mass from |grad|, so a
      // non-finite gradient yields a non-finite mass and the pinned-chain
      // pathology — reject it like a failed evaluation.
      if (auto inits_c = init_cfg.init_chain_config(c);
          inits_c.has_init_eval() && !inits_c.init_grad().allFinite()) {
        throw_eval_failed_init(
            c, init_source,
            "logp finite but gradient non-finite at init (Stan init "
            "validity requires a finite gradient)");
      }
    }
  }
  auto inits = init_cfg.init_chain_config(0);
  std::mt19937_64 rng{seed};
  if (step_init_heuristic) {
    const auto inv_mass = inits.mass().array().inverse().matrix().eval();
    walnutpie::detail::Random heur_rand(rng);
    double eps = walnutpie::detail::find_reasonable_step(
        heur_rand, logp, inits.position(), inv_mass, inits.step_size());
    // W-42: keep the recorded init evaluation (the probe does not move the
    // position) so the first transition's cache seed survives.
    inits = inits.has_init_eval()
                ? walnutpie::InitChainConfig(eps, inits.position(),
                                             inits.mass(), inits.init_grad(),
                                             inits.init_logp())
                : walnutpie::InitChainConfig(eps, inits.position(),
                                             inits.mass());
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
  std::size_t init_tries = 100;  // W-42: random-init rejection-loop budget

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

    app.add_option("--init-tries", init_tries,
                   "Random init (W-42): maximum draws rejecting non-finite "
                   "log probability before failing (Stan convention: 100). "
                   "W-78: draws whose evaluation fails or whose gradient is "
                   "non-finite count as rejected too. Rejections draw again "
                   "from the same init rng stream before any warmup "
                   "consumption; a file init never resamples and instead "
                   "fails immediately on non-finite logp or a failed "
                   "evaluation")
        ->default_val(init_tries)
        ->check(CLI::PositiveNumber);

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
          .metric_auto(metric_auto)
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
    // W-42: random init retries on non-finite logp (see initialize_finite)
    return initialize_finite(model, rng, init, static_cast<unsigned int>(seed),
                             0, init_tries);
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
              warmup_cfg, sample_cfg, extra.first, extra.second, early_exit_tol,
              init_file);
        } else if (step_opt_batch_stride > 1) {
          return run_walnuts<AntiWindupAdapter<BatchedAdapter<Opt>>>(
              model, seed, init_cfg, num_warmup, num_draws, save_warmup,
              warmup_cfg, sample_cfg, extra.first, extra.second, early_exit_tol,
              init_file);
        } else if (step_grad_clip > 0.0) {
          return run_walnuts<AntiWindupAdapter<ClippedAdapter<Opt>>>(
              model, seed, init_cfg, num_warmup, num_draws, save_warmup,
              warmup_cfg, sample_cfg, extra.first, extra.second, early_exit_tol,
              init_file);
        }
        return run_walnuts<AntiWindupAdapter<Opt>>(
            model, seed, init_cfg, num_warmup, num_draws, save_warmup,
            warmup_cfg, sample_cfg, extra.first, extra.second, early_exit_tol,
            init_file);
      }
      if (step_opt_batch_stride > 1 && step_grad_clip > 0.0) {
        return run_walnuts<ClippedAdapter<BatchedAdapter<Opt>>>(
            model, seed, init_cfg, num_warmup, num_draws, save_warmup,
            warmup_cfg, sample_cfg, extra.first, extra.second, early_exit_tol,
            init_file);
      } else if (step_opt_batch_stride > 1) {
        return run_walnuts<BatchedAdapter<Opt>>(
            model, seed, init_cfg, num_warmup, num_draws, save_warmup,
            warmup_cfg, sample_cfg, extra.first, extra.second, early_exit_tol,
            init_file);
      } else if (step_grad_clip > 0.0) {
        return run_walnuts<ClippedAdapter<Opt>>(
            model, seed, init_cfg, num_warmup, num_draws, save_warmup,
            warmup_cfg, sample_cfg, extra.first, extra.second, early_exit_tol,
            init_file);
      }
      return run_walnuts<Opt>(model, seed, init_cfg, num_warmup, num_draws,
                              save_warmup, warmup_cfg, sample_cfg, extra.first,
                              extra.second, early_exit_tol, init_file);
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
