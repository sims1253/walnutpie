#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdlib>
#include <cstdint>
#include <iostream>
#include <cstddef>
#include <deque>
#include <functional>
#include <latch>
#include <limits>
#include <mutex>
#include <optional>
#include <stop_token>
#include <thread>
#include <utility>
#include <vector>

#include <Eigen/Dense>

#include "walnutpie/concepts.hpp"
#include "walnutpie/config.hpp"
#include "walnutpie/spsc_buffer.hpp"
#include "walnutpie/util.hpp"

namespace walnutpie::detail {

/**
 * @brief A struct to represent a snapshot of the adaptation process
 * in a single chain.
 */
struct alignas(FALSE_SHARING_GUARD_SIZE) AdaptSnapshot {
  /**
   * @brief Construct an adaptation snapshot of size 0.
   */
  AdaptSnapshot() : AdaptSnapshot(0) {}

  /**
   * @brief Construct an adaptation snapshot of the given dimensionality.
   *
   * @param[in] dim The number of dimensions in the positions.
   */
  explicit AdaptSnapshot(Eigen::Index dim)
      : log_mass(Eigen::VectorXd::Constant(
            dim, std::numeric_limits<double>::quiet_NaN())),
        mass(Eigen::VectorXd::Constant(
            dim, std::numeric_limits<double>::quiet_NaN())) {}

  /** The number of iterations carried out in the chain. */
  std::size_t iter = 0;

  /** The currently adapted log step size. */
  double log_step = std::numeric_limits<double>::quiet_NaN();

  /** The currently adapted log mass matirx. */
  Eigen::VectorXd log_mass;

  /** The currently adapted mass matrix. */
  Eigen::VectorXd mass;
};

/**
 * @brief Return a deque of buffers of the given sizes.
 *
 * @param[in] num_chains The number of Markov chains.
 * @param[in] dim The number of dimensions.
 * @return The buffer container.
 */
inline std::deque<SpscBuffer<AdaptSnapshot>> construct_buffers(
    std::size_t num_chains, std::size_t dim) {
  std::deque<SpscBuffer<AdaptSnapshot>> buffers;
  auto snapshot = AdaptSnapshot(static_cast<Eigen::Index>(dim));
  for (std::size_t m = 0; m < num_chains; ++m) {
    buffers.emplace_back(snapshot);
  }
  return buffers;
}

/**
 * @brief The execution topology of the multi-chain adaptation (W-30).
 */
enum class ChainExec {
  /**
   * One worker thread per chain (a `std::jthread` running `AdaptWorker`);
   * the controller monitors snapshot buffers, woken by an `AdaptMonitor`
   * notification on every publish. The default.
   */
  Threads,
  /**
   * All chains run round-robin on the calling thread in blocks of
   * `publish_stride` iterations, with the same stop evaluation at every
   * block boundary. Deterministic observation points (every chain is seen
   * at the same iteration); no worker threads are spawned. Per-chain draw
   * content is identical to `Threads` — chain state is chain-local and the
   * RNG streams never interleave — only the schedule differs.
   */
  Serial
};

/**
 * @brief The notification channel between adaptation workers and the
 * controller thread (W-30).
 *
 * Replaces the controller's former busy-poll loop, which burned one full
 * core re-reading snapshot buffers that change only every `publish_stride`
 * iterations per chain. Workers call `notify_published()` after every
 * snapshot publish; the controller blocks in `wait_for_change()` until some
 * chain published again. The version counter is read and updated under the
 * mutex, so a publish that races the controller's observation cannot be
 * missed (the predicate stays true until observed).
 */
class AdaptMonitor {
 public:
  /**
   * @brief Record a snapshot publication and wake the controller.
   *
   * Called by the producing side (worker thread or serial driver) after
   * `SpscBuffer::publish()`.
   */
  void notify_published() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      ++version_;
    }
    cv_.notify_one();
  }

  /**
   * @brief Return the current publication version.
   */
  std::uint64_t version() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return version_;
  }

  /**
   * @brief Block until the publication version differs from `seen`.
   *
   * Updates `seen` to the observed version before returning. Returns true
   * when woken by a publication, false when the timeout elapsed without
   * one. The timeout bounds the controller's reaction to a missed
   * notification (none is possible by construction) and keeps the
   * interrupt-callback contract bounded: the caller re-checks interrupts
   * at most `timeout` apart.
   *
   * @param[in,out] seen The version at the caller's last observation.
   * @param[in] timeout Maximum time to block.
   * @return Whether a new publication was observed.
   */
  bool wait_for_change(std::uint64_t& seen,
                       std::chrono::milliseconds timeout) {
    std::unique_lock<std::mutex> lock(mutex_);
    const bool changed =
        cv_.wait_for(lock, timeout, [&] { return version_ != seen; });
    seen = version_;
    return changed;
  }

 private:
  mutable std::mutex mutex_;
  std::condition_variable cv_;
  std::uint64_t version_ = 0;
};

/**
 * @brief Publish an adapter's current tuning into its snapshot buffer.
 *
 * Factored out of `AdaptWorker` so the serial controller (W-30) writes
 * the same snapshot contents at the same iteration grid.
 *
 * @tparam A The adapter type.
 * @param[in] adapter The chain's adaptive sampler.
 * @param[out] buffer The chain's snapshot buffer.
 * @param[in] iter The iteration the snapshot is taken at.
 */
template <AdaptiveSampler A>
inline void publish_adapter_snapshot(A& adapter,
                                     SpscBuffer<AdaptSnapshot>& buffer,
                                     std::size_t iter) {
  AdaptSnapshot& snap = buffer.write_buffer();
  snap.iter = iter;
  snap.log_step = adapter.log_step_size();
  const auto lm = adapter.log_mass();
  snap.log_mass = lm;
  snap.mass = lm.array().exp().matrix();
  buffer.publish();
}

/**
 * @brief A class that encapsulates the work done in a Markov chain for
 * embedding in a thread.
 *
 * @tparam AdaptiveSampler The base sampler being adapted.
 */
template <AdaptiveSampler A>
class AdaptWorker {
 public:
  /**
   * @brief Construct an adaptation worker with the specified configuration.
   *
   * @param[in] warmup_cfg The configuration for warmup.
   * @param[in] adapter The base sampler, methods of which are later called.
   * @param[out] buffer The buffer to hold the adaptation states.
   * @param[inout] start_gate A latch to gate work to start synchronously across
   * workers.
   * @param[inout] monitor The publication monitor waking the controller.
   */
  AdaptWorker(const WarmupConfig& warmup_cfg, A& adapter,
              SpscBuffer<AdaptSnapshot>& buffer, std::latch& start_gate,
              AdaptMonitor& monitor)
      : warmup_config_(warmup_cfg),
        adapter_(adapter),
        buffer_(buffer),
        start_gate_(start_gate),
        monitor_(monitor) {}

  /**
   * @brief The functor called by the thread to do the adaptation.
   *
   * A call to this function first waits for the latch, then iterates
   * for a number of iterations between the specified min and max.
   * Adaptation terminates when the maximum number of iterations is
   * hit or a stop is requested through the stop token. The thread
   * will yield based on the yield period specified in the warmup
   * configuration. Every snapshot publication notifies the controller
   * through the monitor (W-30) instead of relying on the controller
   * re-reading the buffers in a spin loop.
   *
   * @param[in] st The stop token for stopping the worker thread.
   */
  void operator()(const std::stop_token st) {
    interactive_qos();  // Apple silicon top priority; o.w. no-op
    start_gate_.get().arrive_and_wait();
    publish_snapshot(0);
    // from 1 so modulo ops don't trigger on first iteration
    std::size_t iter = 1;
    for (; iter <= warmup_config_.get().max_iter(); ++iter) {
      if (st.stop_requested()) {
        break;
      }
      if (iter % warmup_config_.get().yield_period() == 0) {
        std::this_thread::yield();
      }
      adapter_.get()();  // do the sampling
      if (iter % warmup_config_.get().publish_stride() == 0) {
        publish_snapshot(iter);
      }
    }
    publish_snapshot(iter - 1);
  }

 private:
  void publish_snapshot(std::size_t iter) {
    publish_adapter_snapshot(adapter_.get(), buffer_.get(), iter);
    monitor_.get().notify_published();
  }

  std::reference_wrapper<const WarmupConfig> warmup_config_;
  std::reference_wrapper<A> adapter_;
  std::reference_wrapper<SpscBuffer<AdaptSnapshot>> buffer_;
  std::reference_wrapper<std::latch> start_gate_;
  std::reference_wrapper<AdaptMonitor> monitor_;
};

/**
 * @brief A struct to hold the diagonal of the mass matrix and a step size.
 */
struct AdaptResult {
  /**
   * The average mass matrix.
   */
  Eigen::VectorXd mass_bar;

  /**
   * The average step size.
   */
  double step_bar;

  /**
   * @brief Per-chain dispersion of log mass diagonals at the end of
   * adaptation, a cheap cross-chain disagreement diagnostic.
   *
   * Values near 0 indicate the chains converged to the same scale estimate;
   * large values indicate chains locked into different posterior scales (the
   * multimodality/scale-lock signature). Intended as the trigger for
   * mode-aware re-initialization policies in embedding code.
   */
  double log_mass_dispersion = 0.0;

  /**
   * @brief Per-chain log-mass diagonals at the end of adaptation.
   *
   * Size M; lets embedding code attribute the cross-chain dispersion to
   * individual chains (the outlier chains in a scale-lock run) instead of
   * only observing the aggregate.
   */
  std::vector<Eigen::VectorXd> chain_log_mass{};

  /**
   * @brief The smallest per-chain iteration count at controller exit.
   *
   * Equal to `max_iter` when the controller ran warmup to completion;
   * smaller when the controller stopped warmup early (cross-chain
   * convergence, subject to the temporal step-drift gate when enabled).
   */
  std::size_t exit_iter = 0;

  /**
   * @brief Whether the controller stopped warmup before the maximum.
   */
  bool early_exit = false;
};

/**
 * @brief The per-controller state of the temporal step-drift gate (W-25)
 * plus the debug trace counter.
 *
 * Carried across stop evaluations so drift is measured window-over-window;
 * reset when a controller (or a resumed phase) starts.
 */
struct ControllerGateState {
  /**
   * @brief Construct the gate state for the given chain count.
   *
   * @param[in] num_chains The number of chains.
   */
  explicit ControllerGateState(std::size_t num_chains)
      : window_step(num_chains, std::numeric_limits<double>::quiet_NaN()),
        window_step2(num_chains, std::numeric_limits<double>::quiet_NaN()),
        window_mass(num_chains),
        window_mass2(num_chains),
        window_iter(num_chains, std::numeric_limits<std::size_t>::max()),
        window_step_drift(num_chains,
                          std::numeric_limits<double>::quiet_NaN()),
        window_mass_drift(num_chains,
                          std::numeric_limits<double>::quiet_NaN()) {}

  /** Step size at the last window boundary (per chain). */
  std::vector<double> window_step;
  /** Step size at the boundary two windows back (per chain). */
  std::vector<double> window_step2;
  /** Mass diagonal at the last window boundary (per chain). */
  std::vector<Eigen::VectorXd> window_mass;
  /** Mass diagonal two windows back (per chain). */
  std::vector<Eigen::VectorXd> window_mass2;
  /** Iteration of the last window boundary (per chain). */
  std::vector<std::size_t> window_iter;
  /** Step drift across the last two windows (per chain). */
  std::vector<double> window_step_drift;
  /** Mass drift across the last two windows (per chain). */
  std::vector<double> window_mass_drift;
  /** Debug trace counter (WALNUTPIE_DEBUG_CTRL). */
  std::size_t dbg_n = 0;
};

/**
 * @brief One stop evaluation of the control monitor (W-30 factoring).
 *
 * Reads the latest snapshot of every chain and evaluates the controller's
 * stop decision exactly as the (former) spin loop did per pass: no chain
 * below `min_iter`, then cross-chain agreement and — when the temporal
 * drift tolerance is positive — per-chain two-window step/mass drift.
 * Returns the adaptation statistics when the controller should stop, and
 * `std::nullopt` when warmup should continue. Called by both the threaded
 * controller (woken by the monitor) and the serial controller (at every
 * round-robin block boundary); shared so the two schedules cannot drift
 * apart in semantics.
 *
 * @param[inout] buffers The adaptation state of all the chains.
 * @param[in] init_cfg The initialization configuration.
 * @param[in] warmup_cfg The warmup configuration.
 * @param[inout] gate The temporal gate state carried across evaluations.
 * @return The stop decision and statistics, or `std::nullopt` to continue.
 */
inline std::optional<AdaptResult> poll_controller(
    std::deque<SpscBuffer<AdaptSnapshot>>& buffers, const InitConfig& init_cfg,
    const WarmupConfig& warmup_cfg, ControllerGateState& gate) {
  const std::size_t M = init_cfg.num_chains();
  const std::size_t D = init_cfg.dims();

  std::vector<AdaptSnapshot> latest(init_cfg.num_chains());

  // Temporal step-drift gate (W-25): see ControllerGateState.
  const double drift_tol = warmup_cfg.temporal_step_drift_tol();
  const double mass_drift_tol = warmup_cfg.mass_converge_tol();
  const std::size_t window = warmup_cfg.temporal_window();
  const std::size_t temporal_min = warmup_cfg.temporal_min_iter();
  auto& window_step = gate.window_step;
  auto& window_step2 = gate.window_step2;
  auto& window_mass = gate.window_mass;
  auto& window_mass2 = gate.window_mass2;
  auto& window_iter = gate.window_iter;
  auto& window_step_drift = gate.window_step_drift;
  auto& window_mass_drift = gate.window_mass_drift;

  std::size_t max_draws = M * warmup_cfg.max_iter();
  bool achieved_min_draws = true;
  std::size_t num_draws = 0;

  Eigen::VectorXd mean_log_mass(D);
  mean_log_mass.setZero();
  double mean_log_step = 0.0;

  for (std::size_t m = 0; m < M && achieved_min_draws; ++m) {
    latest[m] = buffers[m].read_latest();
    if (latest[m].iter < warmup_cfg.min_iter()) {
      achieved_min_draws = false;
    }
    num_draws += latest[m].iter;

    mean_log_step += latest[m].log_step;  // means after division
    mean_log_mass += latest[m].log_mass;
  }
  if (!achieved_min_draws) {
    return std::nullopt;
  }
  mean_log_step /= static_cast<double>(M);
  mean_log_mass /= static_cast<double>(M);
  Eigen::VectorXd geom_mean_mass = mean_log_mass.array().exp().matrix();

  double max_rel_diff_mass = 0.0;
  double max_rel_diff_step = 0.0;
  double geom_mean_step = std::exp(mean_log_step);
  for (std::size_t m = 0; m < M; ++m) {
    double rel_diff_mass = l2_rel_diff(latest[m].mass, geom_mean_mass);
    max_rel_diff_mass = std::fmax(max_rel_diff_mass, rel_diff_mass);
    double chain_m_step = std::exp(latest[m].log_step);
    double rel_diff_step = (chain_m_step - geom_mean_step) / geom_mean_step;
    max_rel_diff_step = std::fmax(max_rel_diff_step, rel_diff_step);
  }

  bool converged;
  if (drift_tol > 0.0) {
    // Temporal gate mode (W-22/W-25): cross-chain agreement can hold
    // while every chain's step size is still marching toward its
    // equilibrium; exiting then degraded post-warmup quality on the
    // marginal model class. Early exit requires: cross-chain step
    // agreement (the existing step tolerance), per-chain step drift
    // below the temporal tolerance across the last full window, and
    // per-chain mass drift below the mass tolerance across the same
    // window (replacing the cross-chain mass comparison, which the
    // noise of windowed estimates keeps from ever converging).
    bool temporal_ok = true;
    for (std::size_t m = 0; m < M; ++m) {
      const std::size_t it = latest[m].iter;
      if (it >= temporal_min &&
          (window_iter[m] == std::numeric_limits<std::size_t>::max() ||
           it >= window_iter[m] + window)) {
        // Drift over the last TWO windows (boundary k vs k-2, ~2*window
        // iterations apart): a single window can pass by luck while the
        // step is still marching (measured: 1-window gate exited the
        // marginal class at ~250-300 iters and degraded ESS 5-9x).
        const double prev2_step = window_step2[m];
        const double cur_step = std::exp(latest[m].log_step);
        if (std::isfinite(prev2_step)) {
          window_step_drift[m] =
              std::abs(cur_step - prev2_step) /
              std::max(prev2_step, std::numeric_limits<double>::min());
        }
        if (window_mass2[m].size() == latest[m].mass.size() &&
            window_mass2[m].size() > 0) {
          window_mass_drift[m] =
              (latest[m].mass - window_mass2[m]).norm() /
              std::max(window_mass2[m].norm(),
                       std::numeric_limits<double>::min());
        }
        window_step2[m] = window_step[m];
        window_mass2[m] = window_mass[m];
        window_step[m] = cur_step;
        window_mass[m] = latest[m].mass;
        window_iter[m] = it;
      }
      if (!(window_iter[m] != std::numeric_limits<std::size_t>::max() &&
            window_iter[m] >= temporal_min &&
            std::isfinite(window_step_drift[m]) &&
            window_step_drift[m] <= drift_tol &&
            std::isfinite(window_mass_drift[m]) &&
            window_mass_drift[m] <= mass_drift_tol)) {
        temporal_ok = false;
      }
    }
    converged = max_rel_diff_step <= warmup_cfg.step_size_converge_tol() &&
                temporal_ok;
  } else {
    converged = max_rel_diff_mass <= warmup_cfg.mass_converge_tol() &&
                max_rel_diff_step <= warmup_cfg.step_size_converge_tol();
  }
  if (const char* dbg = [] {
        static const char* d = std::getenv("WALNUTPIE_DEBUG_CTRL");
        return d;
      }()) {
    if (gate.dbg_n++ % atoi(dbg) == 0) {
      std::cerr << "[ctrl it~" << latest[0].iter << "] mass_diff="
                << max_rel_diff_mass << " step_diff=" << max_rel_diff_step
                << " drift_gate=" << (drift_tol > 0.0 ? "on" : "off");
      for (std::size_t m = 0; m < M && m < 4; ++m) {
        std::cerr << " c" << m << ":step=" << std::exp(latest[m].log_step)
                  << ":sd=" << window_step_drift[m]
                  << ":md=" << window_mass_drift[m];
      }
      std::cerr << std::endl;
    }
  }
  bool hit_max_iter = num_draws == max_draws;
  if (!(converged || hit_max_iter)) {
    return std::nullopt;
  }
  // Cross-chain scale disagreement: mean over coordinates of the
  // variance across chains of log mass. Cheap O(M*D); recomputed here
  // rather than accumulated to keep the loop branch simple.
  double disp_sum = 0.0;
  for (std::size_t d = 0; d < D; ++d) {
    double m_log = 0.0;
    for (std::size_t m2 = 0; m2 < M; ++m2) {
      m_log += latest[m2].log_mass[d];
    }
    m_log /= static_cast<double>(M);
    double v = 0.0;
    for (std::size_t m2 = 0; m2 < M; ++m2) {
      const double dev = latest[m2].log_mass[d] - m_log;
      v += dev * dev;
    }
    disp_sum += v / static_cast<double>(M - 1);
  }
  std::vector<Eigen::VectorXd> chain_lm;
  chain_lm.reserve(M);
  for (std::size_t m2 = 0; m2 < M; ++m2) {
    chain_lm.push_back(latest[m2].log_mass);
  }
  std::size_t exit_iter = latest[0].iter;
  for (std::size_t m2 = 1; m2 < M; ++m2) {
    exit_iter = std::min(exit_iter, latest[m2].iter);
  }
  return AdaptResult{std::move(geom_mean_mass), std::exp(mean_log_step),
                     disp_sum / static_cast<double>(D), std::move(chain_lm),
                     exit_iter, exit_iter < warmup_cfg.max_iter()};
}

/**
 * @brief The implementation of the control monitor with the adaptation
 * state of each chain and configuration (W-30: event-driven).
 *
 * The controller blocks on the adaptation monitor between stop
 * evaluations instead of re-reading the snapshot buffers in a tight
 * loop: workers notify after every publish, and the 100 ms wait cap
 * bounds interrupt latency (the interrupt callback is re-checked after
 * every wait) as well as any reaction to a final publish. Stop
 * semantics are unchanged — `poll_controller` evaluates exactly the
 * former per-pass decision.
 *
 * @param[inout] buffers The adaptation state of all the chains.
 * @param[inout] monitor The publication monitor shared with the workers.
 * @param[in] interrupt_callback The interrupt callback for stopping.
 * @param[in] init_cfg The initialization configuration.
 * @param[in] warmup_cfg The warmup configuration.
 * @return Statistics for the completed adaptation process.
 */
template <InterruptCallback IC>
inline AdaptResult controller_loop(
    std::deque<SpscBuffer<AdaptSnapshot>>& buffers, AdaptMonitor& monitor,
    const IC& interrupt_callback, const InitConfig& init_cfg,
    const WarmupConfig& warmup_cfg) {
  ControllerGateState gate(init_cfg.num_chains());
  std::uint64_t seen = monitor.version();
  while (true) {
    if (std::optional<AdaptResult> result =
            poll_controller(buffers, init_cfg, warmup_cfg, gate)) {
      return std::move(*result);
    }
    monitor.wait_for_change(seen, std::chrono::milliseconds(100));
    interrupt_callback.throw_if_interrupted();
  }
}

/**
 * @brief The serial control monitor: all chains on the calling thread
 * (W-30, `ChainExec::Serial`).
 *
 * Runs every chain round-robin in blocks of `publish_stride` iterations,
 * publishing each chain's snapshot at the block boundary and evaluating
 * the shared stop decision there. Observation points are deterministic
 * (every chain is seen at the same iteration), and the publish grid is
 * the same `(chain, iteration)` set the threaded workers produce; no
 * worker threads are spawned and no yield is performed (there is nothing
 * to yield to). Per-chain draw content is identical to the threaded
 * controller — see `ChainExec::Serial`.
 *
 * @tparam A The type of the adaptive samplers.
 * @tparam IC The type of the interrupt callback.
 * @param[inout] adapters The adaptive samplers for each chain.
 * @param[inout] buffers The adaptation state of all the chains.
 * @param[in] interrupt_callback The interrupt callback for stopping.
 * @param[in] init_cfg The initialization configuration.
 * @param[in] warmup_cfg The warmup configuration.
 * @return Statistics for the completed adaptation process.
 * @throw std::runtime_error If no stop decision is possible after the
 * full budget (a misconfiguration with `min_iter > max_iter`; the
 * threaded controller would spin forever in the same situation).
 */
template <AdaptiveSampler A, InterruptCallback IC>
inline AdaptResult controller_serial(
    std::vector<A>& adapters, std::deque<SpscBuffer<AdaptSnapshot>>& buffers,
    const IC& interrupt_callback, const InitConfig& init_cfg,
    const WarmupConfig& warmup_cfg) {
  const std::size_t M = init_cfg.num_chains();
  const std::size_t stride = warmup_cfg.publish_stride();
  const std::size_t max_iter = warmup_cfg.max_iter();
  ControllerGateState gate(M);
  for (std::size_t m = 0; m < M; ++m) {
    publish_adapter_snapshot(adapters[m], buffers[m], 0);
  }
  if (std::optional<AdaptResult> result =
          poll_controller(buffers, init_cfg, warmup_cfg, gate)) {
    return std::move(*result);
  }
  for (std::size_t base = 1; base <= max_iter; base += stride) {
    const std::size_t hi = std::min(base + stride - 1, max_iter);
    for (std::size_t m = 0; m < M; ++m) {
      for (std::size_t it = base; it <= hi; ++it) {
        adapters[m]();  // do the sampling
      }
      publish_adapter_snapshot(adapters[m], buffers[m], hi);
    }
    interrupt_callback.throw_if_interrupted();
    if (std::optional<AdaptResult> result =
            poll_controller(buffers, init_cfg, warmup_cfg, gate)) {
      return std::move(*result);
    }
  }
  // Unreachable when min_iter <= max_iter: the last block's evaluation
  // sees every chain at max_iter (hit_max_iter). Guard the misconfiguration
  // explicitly rather than spin.
  throw std::runtime_error(
      "serial controller exhausted warmup without a stop decision "
      "(min_iter > max_iter?)");
}

/**
 * @brief The top-level function call for adaptation for the given configuration
 * and samplers.
 *
 * @tparam Adapter The type of adaptive sampler.
 * @tparam IC The type of the interrupt callback.
 * @param[in] init_cfg The initial configuration.
 * @param[in] warmup_cfg The warmup configuration.
 * @param[inout] adapters The adaptive samplers for each chain.
 * @param[in] interrupt_callback The interrupt callback for stopping.
 */
template <AdaptiveSampler A, InterruptCallback IC>
inline void adapt(const InitConfig& init_cfg, const WarmupConfig& warmup_cfg,
                  std::vector<A>& adapters, const IC& interrupt_callback) {
  std::deque<SpscBuffer<AdaptSnapshot>> buffers =
      construct_buffers(init_cfg.num_chains(), init_cfg.dims());

  AdaptMonitor monitor;
  std::latch start_gate(static_cast<std::ptrdiff_t>(init_cfg.num_chains() + 1));
  std::vector<std::jthread> threads;
  threads.reserve(init_cfg.num_chains());
  for (std::size_t m = 0; m < init_cfg.num_chains(); ++m) {
    threads.emplace_back(AdaptWorker<A>(warmup_cfg, adapters[m], buffers[m],
                                        start_gate, monitor));
  }
  start_gate.arrive_and_wait();

  AdaptResult result = controller_loop(buffers, monitor, interrupt_callback,
                                       init_cfg, warmup_cfg);
}

/**
 * @brief Adaptation returning the adaptation statistics.
 *
 * Identical to the void overload, but exposes the AdaptResult (including the
 * cross-chain log-mass dispersion) for mode-aware reinitialization policies.
 *
 * @tparam Adapter The type of adaptive sampler.
 * @tparam IC The type of the interrupt callback.
 * @param[in] init_cfg The initial configuration.
 * @param[in] warmup_cfg The warmup configuration.
 * @param[inout] adapters The adaptive samplers for each chain.
 * @param[in] interrupt_callback The interrupt callback for stopping.
 * @param[in] exec The execution topology (W-30): `ChainExec::Threads` (one
 * worker thread per chain, event-driven controller) or `ChainExec::Serial`
 * (all chains round-robin on the calling thread; identical per-chain draws).
 */
template <AdaptiveSampler A, InterruptCallback IC>
inline AdaptResult adapt_with_stats(const InitConfig& init_cfg,
                                    const WarmupConfig& warmup_cfg,
                                    std::vector<A>& adapters,
                                    const IC& interrupt_callback,
                                    ChainExec exec = ChainExec::Threads) {
  if (exec == ChainExec::Serial) {
    std::deque<SpscBuffer<AdaptSnapshot>> buffers =
        construct_buffers(init_cfg.num_chains(), init_cfg.dims());
    return controller_serial(adapters, buffers, interrupt_callback, init_cfg,
                             warmup_cfg);
  }
  std::deque<SpscBuffer<AdaptSnapshot>> buffers =
      construct_buffers(init_cfg.num_chains(), init_cfg.dims());

  AdaptMonitor monitor;
  std::latch start_gate(static_cast<std::ptrdiff_t>(init_cfg.num_chains() + 1));
  std::vector<std::jthread> threads;
  threads.reserve(init_cfg.num_chains());
  for (std::size_t m = 0; m < init_cfg.num_chains(); ++m) {
    threads.emplace_back(AdaptWorker<A>(warmup_cfg, adapters[m], buffers[m],
                                        start_gate, monitor));
  }
  start_gate.arrive_and_wait();

  return controller_loop(buffers, monitor, interrupt_callback, init_cfg,
                         warmup_cfg);
}

/**
 * @brief Adaptation with a pilot sampling-burst gate on early exits (W-28).
 *
 * Runs the multi-chain controller in phases under a TOTAL warmup budget of
 * `warmup_cfg.max_iter()` iterations. Whenever a phase stops warmup early
 * (cross-chain convergence, subject to the temporal drift gate when
 * enabled), the caller-supplied `pilot_gate` may inspect the would-be
 * frozen samplers (e.g. by running a short pilot burst of draws per chain)
 * and VETO the early exit. On a veto, adaptation RESUMES from the preserved
 * adapter state (nothing is discarded) with the remaining budget as the
 * next phase's max_iter; a resumed phase re-arms the controller's temporal
 * gate state from scratch, so the next candidate requires the gate
 * conditions to hold again over fresh windows. If the budget is exhausted
 * after a veto, warmup has effectively run to completion and the result is
 * reported with `early_exit == false`.
 *
 * `pilot_gate` is invoked as `bool pilot_gate(std::vector<A>& adapters,
 * std::size_t candidate_iter)` where `candidate_iter` is the cumulative
 * warmup iteration count of the candidate exit; it returns true to approve
 * the early exit, false to veto and resume. It is only called between
 * phases (no warmup threads are running), so it may freely read the
 * adapters and run sampler() on them.
 *
 * @tparam Adapter The type of adaptive sampler.
 * @tparam IC The type of the interrupt callback.
 * @tparam PilotGate The type of the pilot gate callable.
 * @param[in] init_cfg The initial configuration (chain count/dimensionality).
 * @param[in] warmup_cfg The warmup configuration; its max_iter is the TOTAL
 * budget across all phases.
 * @param[inout] adapters The adaptive samplers for each chain.
 * @param[in] interrupt_callback The interrupt callback for stopping.
 * @param[in] pilot_gate The veto callable described above.
 * @param[in] exec The execution topology forwarded to every phase (W-30).
 */
template <AdaptiveSampler A, InterruptCallback IC, typename PilotGate>
inline AdaptResult adapt_with_pilot(const InitConfig& init_cfg,
                                    const WarmupConfig& warmup_cfg,
                                    std::vector<A>& adapters,
                                    const IC& interrupt_callback,
                                    PilotGate& pilot_gate,
                                    ChainExec exec = ChainExec::Threads) {
  const std::size_t total = warmup_cfg.max_iter();
  std::size_t done = 0;
  WarmupConfig phase_cfg = warmup_cfg;
  while (true) {
    AdaptResult r = adapt_with_stats(init_cfg, phase_cfg, adapters,
                                     interrupt_callback, exec);
    done += std::min(r.exit_iter, phase_cfg.max_iter());
    if (!r.early_exit || done >= total) {
      // Ran to the phase (or total) budget: no pilot check applies.
      r.exit_iter = done;
      r.early_exit = false;
      return r;
    }
    if (pilot_gate(adapters, done)) {
      r.exit_iter = done;
      return r;  // approved early exit
    }
    // Vetoed: resume warmup with the remaining budget. Clamp the phase
    // min_iter to the remaining budget so a short final phase still runs
    // (with min == max the controller can only stop at the budget, i.e. it
    // completes warmup; the temporal gate re-arms only after
    // temporal_min_iter more iterations).
    const std::size_t remaining = total - done;
    phase_cfg = warmup_cfg.with_min_max_iter(
        std::min(warmup_cfg.min_iter(), remaining), remaining);
  }
}

}  // namespace walnutpie::detail
