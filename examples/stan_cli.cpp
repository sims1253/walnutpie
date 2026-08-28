#include <walnutpie.hpp>
#include <walnutpie/load_stan.hpp>

#include <CLI/CLI.hpp>
#include <Eigen/Dense>

#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <string>
#include <vector>

using walnutpie::DynamicStanModel;
using walnutpie::unique_bs_rng;

// --warmup-trace-dir plumbing: file-scope so run_walnuts can record the
// warmup without threading a new parameter through its signature. Set
// once in main(); an empty dir disables tracing. model_name/meta_extras
// carry main()-side values (library basename, flag values as a JSON
// fragment) that run_walnuts cannot see.
struct WarmupTraceSettings {
  std::string dir;
  std::string model_name;
  std::string meta_extras;
};
static WarmupTraceSettings g_warmup_trace;

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
    // Warmup-tracer capture (read back after each iteration by run_walnuts).
    warmup_pos_ = position;
    warmup_lp_ = lp;
    warmup_step_ = step_size;
    warmup_invmass_ = diag_inv_mass;
    if (!save_warmup_) {
      return;
    }
    model_.constrain_draw(position, draws_.col(n_), rng_);
    n_++;
  }

  void on_warmup_complete(double step_size,
                          const Eigen::VectorXd& diag_inv_mass) {}

  // Last on_warmup arguments, for the --warmup-trace-dir driver-loop
  // tracer (read right after each iteration).
  const Eigen::VectorXd& last_warmup_position() const {
    return warmup_pos_;
  }
  const Eigen::VectorXd& last_warmup_inv_mass() const {
    return warmup_invmass_;
  }
  double last_warmup_lp() const { return warmup_lp_; }
  double last_warmup_step() const { return warmup_step_; }

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

  // Last on_warmup arguments (warmup tracer capture).
  Eigen::VectorXd warmup_pos_;
  Eigen::VectorXd warmup_invmass_;
  double warmup_lp_ = 0.0;
  double warmup_step_ = 0.0;
};

// Opt-in binary warmup tracer backing --warmup-trace-dir: one row per
// warmup iteration so an offline analysis can replay mass-matrix
// adaptation exactly. theta/grad/invmass are little-endian float64 in
// C-order [iteration x D]; step/lp are float64 columns and depth a u64
// column. Rows accumulate in memory and flush once at the end of warmup
// (simple over performant). grad/depth come from AdaptiveWalnuts'
// last_grad()/last_depth(); position/lp/step/invmass mirror the arguments
// StanHandler::on_warmup just received.
class WarmupTracer {
 public:
  WarmupTracer(const Eigen::VectorXd& initial_position,
               const Eigen::VectorXd& initial_mass)
      : initial_position_(initial_position),
        initial_mass_(initial_mass) {}

  bool enabled() const { return !g_warmup_trace.dir.empty(); }

  void record(const Eigen::VectorXd& theta, const Eigen::VectorXd& grad,
              const Eigen::VectorXd& inv_mass, double step, double lp,
              std::size_t depth) {
    append_row(theta_, theta);
    append_row(grads_, grad);
    append_row(invmass_, inv_mass);
    steps_.push_back(step);
    lps_.push_back(lp);
    depths_.push_back(static_cast<std::uint64_t>(depth));
  }

  // Write the accumulated columns and the meta.json manifest. num_warmup
  // is the requested budget; one row is recorded per completed warmup
  // iteration.
  void flush(std::size_t dim, std::size_t num_warmup,
             unsigned long int seed) {
    const std::string& dir = g_warmup_trace.dir;
    write_f64(dir + "/theta.f64", theta_);
    write_f64(dir + "/grad.f64", grads_);
    write_f64(dir + "/invmass.f64", invmass_);
    write_f64(dir + "/step.f64", steps_);
    write_f64(dir + "/lp.f64", lps_);
    write_u64(dir + "/depth.u64", depths_);
    std::ofstream meta(dir + "/meta.json");
    if (!meta) {
      std::cerr << "Failed to open warmup trace file: " << dir
                << "/meta.json" << std::endl;
      return;
    }
    meta << std::setprecision(17);
    meta << "{\n";
    meta << "  \"model\": \"" << g_warmup_trace.model_name << "\",\n";
    meta << "  \"dim\": " << dim << ",\n";
    meta << "  \"num_warmup\": " << num_warmup << ",\n";
    meta << "  \"seed\": " << seed << ",\n";
    meta << "  \"warmup_iters_recorded\": " << depths_.size() << ",\n";
    meta << "  \"flags\": " << g_warmup_trace.meta_extras << ",\n";
    write_vector_json(meta, "initial_position", initial_position_);
    meta << ",\n";
    write_vector_json(meta, "initial_mass", initial_mass_);
    meta << "\n}\n";
  }

 private:
  static void append_row(std::vector<double>& out,
                         const Eigen::VectorXd& v) {
    out.insert(out.end(), v.data(), v.data() + v.size());
  }

  static void write_f64(const std::string& path,
                        const std::vector<double>& v) {
    std::ofstream out(path, std::ios::binary);
    if (!out) {
      std::cerr << "Failed to open warmup trace file: " << path << std::endl;
      return;
    }
    if (!v.empty()) {
      out.write(reinterpret_cast<const char*>(v.data()),
                static_cast<std::streamsize>(v.size() * sizeof(double)));
    }
  }

  static void write_u64(const std::string& path,
                        const std::vector<std::uint64_t>& v) {
    std::ofstream out(path, std::ios::binary);
    if (!out) {
      std::cerr << "Failed to open warmup trace file: " << path << std::endl;
      return;
    }
    if (!v.empty()) {
      out.write(reinterpret_cast<const char*>(v.data()),
                static_cast<std::streamsize>(v.size() *
                                             sizeof(std::uint64_t)));
    }
  }

  static void write_vector_json(std::ostream& os, const char* key,
                                const Eigen::VectorXd& v) {
    os << "  \"" << key << "\": [";
    for (Eigen::Index i = 0; i < v.size(); ++i) {
      if (i > 0) {
        os << ", ";
      }
      os << v[i];
    }
    os << "]";
  }

  std::vector<double> theta_;
  std::vector<double> grads_;
  std::vector<double> invmass_;
  std::vector<double> steps_;
  std::vector<double> lps_;
  std::vector<std::uint64_t> depths_;
  Eigen::VectorXd initial_position_;
  Eigen::VectorXd initial_mass_;
};

StanHandler run_walnuts(DynamicStanModel& model, unsigned int seed,
                        walnutpie::InitConfigBuilder& init_builder,
                        std::size_t num_warmup, std::size_t num_draws,
                        bool save_warmup, walnutpie::WarmupConfig& warmup_cfg,
                        walnutpie::SamplingConfig& sample_cfg) {
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

  auto init_cfg =
      init_builder.masses(logp, warmup_cfg.mass_additive_smoothing()).build();
  auto inits = init_cfg.init_chain_config(0);

  std::mt19937_64 rng{seed};
  walnutpie::AdaptiveWalnuts walnuts(rng, storage, logp, inits, warmup_cfg,
                                     sample_cfg);
  // --warmup-trace-dir: record every warmup transition; the sampler
  // advances iteration-by-iteration on the calling thread. inits carries
  // the position and the mass seed InitChainConfig actually built
  // (masses() output).
  WarmupTracer trace(inits.position(), inits.mass());
  auto trace_record = [&]() {
    if (!trace.enabled()) {
      return;
    }
    trace.record(storage.last_warmup_position(), walnuts.last_grad(),
                 storage.last_warmup_inv_mass(), storage.last_warmup_step(),
                 storage.last_warmup_lp(), walnuts.last_depth());
  };
  for (std::size_t w = 0; w < num_warmup; ++w) {
    walnuts();
    trace_record();
  }
  if (trace.enabled()) {
    trace.flush(model.unconstrained_dimensions(), num_warmup, seed);
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

  std::size_t max_trajectory_doublings =
      default_sampling.max_trajectory_doublings();
  std::size_t max_step_halvings = default_sampling.max_step_halvings();
  double max_hamiltonian_error = default_sampling.max_hamiltonian_error();
  std::size_t min_micro_steps = default_sampling.min_micro_steps();

  double init = 2.0;
  double step_size_init = 1.0;

  std::string lib;
  std::string data;
  std::string output_file;
  std::string warmup_trace_dir;  // empty = warmup tracing off

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

    app.add_option("--warmup-trace-dir", warmup_trace_dir,
                   "Opt-in warmup tracer directory: per-iteration "
                   "theta/grad/invmass as little-endian f64, step/lp as f64, "
                   "depth as u64, plus meta.json, for offline replay of "
                   "mass-matrix adaptation (empty = off)")
        ->default_val(warmup_trace_dir);

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
          .build();

  walnutpie::SamplingConfig sample_cfg =
      walnutpie::SamplingConfigBuilder()
          .max_trajectory_doublings(max_trajectory_doublings)
          .max_step_halvings(max_step_halvings)
          .max_hamiltonian_error(max_hamiltonian_error)
          .min_micro_steps(min_micro_steps)
          .build();

  unique_bs_rng rng = model.make_rng(seed);

  auto init_cfg =
      walnutpie::InitConfigBuilder{1, model.unconstrained_dimensions()}
          .step_sizes(step_size_init)
          .positions(model.initialize(nullptr, rng, init));

  if (!warmup_trace_dir.empty()) {
    std::filesystem::create_directories(warmup_trace_dir);
    g_warmup_trace.dir = warmup_trace_dir;
    const auto slash = lib.find_last_of('/');
    g_warmup_trace.model_name =
        slash == std::string::npos ? lib : lib.substr(slash + 1);
    // Flag values in effect, rendered once here (main scope; run_walnuts
    // cannot see them) and spliced into meta.json by the tracer.
    std::string flags = "{";
    auto jkv = [&](const std::string& k, const std::string& v) {
      if (flags.size() > 1) {
        flags += ", ";
      }
      flags += "\"" + k + "\": " + v;
    };
    auto jstr = [&](const char* k, const std::string& v) {
      jkv(k, "\"" + v + "\"");
    };
    auto jbool = [&](const char* k, bool v) {
      jkv(k, v ? "true" : "false");
    };
    auto jint = [&](const char* k, std::size_t v) {
      jkv(k, std::to_string(v));
    };
    auto jdouble = [&](const char* k, double v) {
      std::ostringstream vs;
      vs << std::setprecision(17) << v;
      jkv(k, vs.str());
    };
    jint("warmup", num_warmup);
    jint("samples", num_draws);
    jbool("save-warmup", save_warmup);
    jint("max-trajectory-doublings", max_trajectory_doublings);
    jint("max-step-halvings", max_step_halvings);
    jint("min-micro-steps", min_micro_steps);
    jdouble("max-hamiltonian-error", max_hamiltonian_error);
    jdouble("init", init);
    jdouble("step-size-init", step_size_init);
    jdouble("mass-init-count", mass_init_count);
    jdouble("mass-additive-smoothing", mass_additive_smoothing);
    jdouble("max-macro-steps-target", max_macro_steps_target);
    jdouble("step-accept-rate-target", step_accept_rate_target);
    jdouble("step-learning-rate", step_learning_rate);
    jdouble("step-gradient-decay", step_gradient_decay);
    jdouble("step-sq-gradient-decay", step_sq_gradient_decay);
    jdouble("step-stabilization", step_stabilization);
    jdouble("step-learn-rate-decay", step_learn_rate_decay);
    flags += "}";
    g_warmup_trace.meta_extras = std::move(flags);
  }

  auto res = run_walnuts(model, seed, init_cfg, num_warmup, num_draws,
                         save_warmup, warmup_cfg, sample_cfg);

  res.summarize();
  res.write_csv(output_file);

  return 0;
}
