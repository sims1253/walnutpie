#pragma once
#include <Eigen/Dense>
#include <cstddef>
#include <stdexcept>
#include <vector>

// Example-only heuristic. Low diversity is not proof of convergence failure.
struct MicroGuardSpec {
  bool armed = false;
  std::size_t probe = 50;
  std::size_t min_unique = 25;
};
struct MicroGuardResult {
  bool pinned = false;
  std::size_t probe = 0;
  std::size_t unique = 0;
};
inline void validate_micro_guard(const MicroGuardSpec& spec,
                                 std::size_t samples) {
  if (spec.armed && (spec.probe == 0 || spec.min_unique == 0 ||
                     spec.min_unique > spec.probe || spec.probe > samples)) {
    throw std::invalid_argument(
        "guard requires 1 <= min-unique <= probe <= samples");
  }
}
class MicroGuardProbe {
 public:
  explicit MicroGuardProbe(const MicroGuardSpec& spec) : spec_(spec) {}
  void observe(const Eigen::VectorXd& position) {
    if (!spec_.armed || positions_.size() >= spec_.probe) {
      return;
    }
    if (position.size() == 0 || !position.allFinite()) {
      throw std::runtime_error("guard requires finite nonempty positions");
    }
    if (!positions_.empty() && position.size() != positions_[0].size()) {
      throw std::runtime_error("guard position dimension changed");
    }
    positions_.push_back(position);
  }
  MicroGuardResult result() const {
    MicroGuardResult result;
    if (!spec_.armed || positions_.size() != spec_.probe) {
      return result;
    }
    result.probe = positions_.size();
    for (std::size_t c = 0; c < positions_.size(); ++c) {
      bool seen = false;
      for (std::size_t p = 0; p < c; ++p) {
        if (positions_[c] == positions_[p]) {
          seen = true;
          break;
        }
      }
      if (!seen) {
        ++result.unique;
      }
    }
    result.pinned = result.unique < spec_.min_unique;
    return result;
  }

 private:
  MicroGuardSpec spec_;
  std::vector<Eigen::VectorXd> positions_;
};

// Scope destroys all discarded draws and GQ state before retry. Never recurse.
template <typename Run, typename Config, typename OnRestart>
auto run_with_micro_guard(Run&& run, Config& initial, Config& mm1,
                          const MicroGuardSpec& spec, OnRestart&& on_restart) {
  {
    MicroGuardResult result;
    auto first = run(initial, spec, &result);
    if (!spec.armed || !result.pinned) {
      return first;
    }
    on_restart(result);
  }
  return run(mm1, MicroGuardSpec{}, nullptr);
}
