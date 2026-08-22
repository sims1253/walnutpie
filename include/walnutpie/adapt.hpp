#pragma once

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <cstddef>
#include <deque>
#include <functional>
#include <latch>
#include <limits>
#include <stop_token>
#include <thread>
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
   */
  AdaptWorker(const WarmupConfig& warmup_cfg, A& adapter,
              SpscBuffer<AdaptSnapshot>& buffer, std::latch& start_gate)
      : warmup_config_(warmup_cfg),
        adapter_(adapter),
        buffer_(buffer),
        start_gate_(start_gate) {}

  /**
   * @brief The functor called by the thread to do the adaptation.
   *
   * A call to this function first waits for the latch, then iterates
   * for a number of iterations between the specified min and max.
   * Adaptation terminates when the maximum number of iterations is
   * hit or a stop is requested through the stop token. The thread
   * will yield based on the yield period specified in the warmup
   * configuration.
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
    AdaptSnapshot& snap = buffer_.get().write_buffer();
    snap.iter = iter;
    snap.log_step = adapter_.get().log_step_size();
    const auto lm = adapter_.get().log_mass();
    snap.log_mass = lm;
    snap.mass = lm.array().exp().matrix();
    buffer_.get().publish();
  }

  std::reference_wrapper<const WarmupConfig> warmup_config_;
  std::reference_wrapper<A> adapter_;
  std::reference_wrapper<SpscBuffer<AdaptSnapshot>> buffer_;
  std::reference_wrapper<std::latch> start_gate_;
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
 * @brief The implementation of the control monitor with the adaptation
 * state of each chain and configuration.
 *
 * @param[inout] buffers The adaptation state of all the chains.
 * @param[in] init_cfg The initialization configuration.
 * @param[in] warmup_cfg The warmup configuration.
 * @return Statistics for the completed adaptation process.
 */
template <InterruptCallback IC>
inline AdaptResult controller_loop(
    std::deque<SpscBuffer<AdaptSnapshot>>& buffers,
    const IC& interrupt_callback, const InitConfig& init_cfg,
    const WarmupConfig& warmup_cfg) {
  std::size_t M = init_cfg.num_chains();
  std::size_t D = init_cfg.dims();

  std::vector<AdaptSnapshot> latest(init_cfg.num_chains());
  Eigen::VectorXd mean_log_mass(D);
  Eigen::VectorXd geom_mean_mass(D);
  Eigen::VectorXd scratch_mass(D);

  // Temporal step-drift gate state (W-25): per-chain step size and mass
  // diagonal at the last window boundary, the iteration of that boundary,
  // and the drifts measured across the most recent completed window.
  // Uninitialized (npos / NaN) until the chain crosses its first window
  // boundary after min_iter. In temporal mode the cross-chain mass
  // criterion (l2 diff vs the geometric mean) is replaced by per-chain
  // temporal mass drift: windowed mass estimates are too noisy across
  // chains for the cross-chain comparison to ever fire (measured 1.4-2.8
  // vs tol 1.0 late in warmup on healthy models), while window-to-window
  // drift within a chain is the stabilization signal W-21/W-22 used.
  const double drift_tol = warmup_cfg.temporal_step_drift_tol();
  const double mass_drift_tol = warmup_cfg.mass_converge_tol();
  const std::size_t window = warmup_cfg.temporal_window();
  const std::size_t temporal_min = warmup_cfg.temporal_min_iter();
  std::vector<double> window_step(M, std::numeric_limits<double>::quiet_NaN());
  std::vector<double> window_step2(
      M, std::numeric_limits<double>::quiet_NaN());  // boundary k-2
  std::vector<Eigen::VectorXd> window_mass(M);
  std::vector<Eigen::VectorXd> window_mass2(M);  // boundary k-2
  std::vector<std::size_t> window_iter(M,
                                       std::numeric_limits<std::size_t>::max());
  std::vector<double> window_step_drift(M,
                                        std::numeric_limits<double>::quiet_NaN());
  std::vector<double> window_mass_drift(M,
                                        std::numeric_limits<double>::quiet_NaN());

  std::size_t max_draws = M * warmup_cfg.max_iter();
  while (true) {
    bool achieved_min_draws = true;
    std::size_t num_draws = 0;

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
    if (achieved_min_draws) {
      mean_log_step /= static_cast<double>(M);
      mean_log_mass /= static_cast<double>(M);
      geom_mean_mass = mean_log_mass.array().exp().matrix();

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
        converged = max_rel_diff_step <=
                        warmup_cfg.step_size_converge_tol() &&
                    temporal_ok;
      } else {
        converged = max_rel_diff_mass <= warmup_cfg.mass_converge_tol() &&
                    max_rel_diff_step <= warmup_cfg.step_size_converge_tol();
      }
      if (const char* dbg = [] {
            static const char* d = std::getenv("WALNUTPIE_DEBUG_CTRL");
            return d;
          }()) {
        static std::size_t dbg_n = 0;
        if (dbg_n++ % atoi(dbg) == 0) {
          std::cerr << "[ctrl it~" << latest[0].iter << "] mass_diff="
                    << max_rel_diff_mass << " step_diff=" << max_rel_diff_step
                    << " drift_gate="
                    << (drift_tol > 0.0 ? "on" : "off");
          for (std::size_t m = 0; m < M && m < 4; ++m) {
            std::cerr << " c" << m << ":step=" << std::exp(latest[m].log_step)
                      << ":sd=" << window_step_drift[m]
                      << ":md=" << window_mass_drift[m];
          }
          std::cerr << std::endl;
        }
      }
      bool hit_max_iter = num_draws == max_draws;
      if (converged || hit_max_iter) {
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
        return {geom_mean_mass, std::exp(mean_log_step),
                disp_sum / static_cast<double>(D), std::move(chain_lm),
                exit_iter, exit_iter < warmup_cfg.max_iter()};
      }
    }

    interrupt_callback.throw_if_interrupted();
  }
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

  std::latch start_gate(static_cast<std::ptrdiff_t>(init_cfg.num_chains() + 1));
  std::vector<std::jthread> threads;
  threads.reserve(init_cfg.num_chains());
  for (std::size_t m = 0; m < init_cfg.num_chains(); ++m) {
    threads.emplace_back(
        AdaptWorker<A>(warmup_cfg, adapters[m], buffers[m], start_gate));
  }
  start_gate.arrive_and_wait();

  AdaptResult result =
      controller_loop(buffers, interrupt_callback, init_cfg, warmup_cfg);
}

/**
 * @brief Adaptation returning the adaptation statistics.
 *
 * Identical to the void overload, but exposes the AdaptResult (including the
 * cross-chain log-mass dispersion) for mode-aware reinitialization policies.
 */
template <AdaptiveSampler A, InterruptCallback IC>
inline AdaptResult adapt_with_stats(const InitConfig& init_cfg,
                                    const WarmupConfig& warmup_cfg,
                                    std::vector<A>& adapters,
                                    const IC& interrupt_callback) {
  std::deque<SpscBuffer<AdaptSnapshot>> buffers =
      construct_buffers(init_cfg.num_chains(), init_cfg.dims());

  std::latch start_gate(static_cast<std::ptrdiff_t>(init_cfg.num_chains() + 1));
  std::vector<std::jthread> threads;
  threads.reserve(init_cfg.num_chains());
  for (std::size_t m = 0; m < init_cfg.num_chains(); ++m) {
    threads.emplace_back(
        AdaptWorker<A>(warmup_cfg, adapters[m], buffers[m], start_gate));
  }
  start_gate.arrive_and_wait();

  return controller_loop(buffers, interrupt_callback, init_cfg, warmup_cfg);
}

}  // namespace walnutpie::detail
