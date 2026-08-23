#pragma once

#include <cmath>
#include <Eigen/Dense>

#include "walnutpie/concepts.hpp"

#include "walnutpie/util.hpp"

namespace walnutpie::detail {

/**
 * @brief Stan-style initial step size heuristic (Hoffman & Gelman 2014,
 * Algorithm 4) adapted to walnutpie's macro-step regime.
 *
 * Starting from `eps`, repeatedly double or halve until the acceptance
 * statistic of a single micro (leapfrog) step with the given diagonal
 * inverse mass crosses 0.5.
 *
 * @param logp_grad Log density and gradient functor.
 * @param theta Initial position.
 * @param inv_mass Diagonal inverse mass.
 * @param eps Initial step size guess (positive).
 * @return The adapted step size.
 */
template <LogpGrad F, std::uniform_random_bit_generator RNG>
double find_reasonable_step(detail::Random<RNG>& rand, const F& logp_grad,
                            const Eigen::VectorXd& theta,
                            const Eigen::VectorXd& inv_mass, double eps) {
  Eigen::VectorXd grad;
  double lp;
  logp_grad(theta, lp, grad);
  // One leapfrog micro step. rho comes from the CALLER'S seeded RNG:
  // Eigen::VectorXd::Random() draws from std::rand(), which nothing ties to
  // --seed, so the heuristic made fixed-seed runs irreproducible (found via
  // bit-diffing paired runs; see PR #4 follow-up 14).
  //
  // W-43: draw the momentum once, as in Hoffman & Gelman (2014)
  // Algorithm 4; one momentum holds the one-step error fixed across
  // probes, so the doubling decision compares steps, not draws. The
  // scale matches the sampler (transition_w draws rho = sqrt(mass) * z);
  // seeding from inv_mass instead would move the probe on the wrong
  // scale.
  Eigen::VectorXd rho0 = rand.standard_normal(theta.size()).matrix();
  Eigen::VectorXd p0 =
      rho0.cwiseProduct(inv_mass.cwiseSqrt().cwiseInverse());  // ~N(0, M)
  double h0 = -lp + 0.5 * p0.cwiseProduct(inv_mass).dot(p0);
  auto accept_of = [&](double e) {
    Eigen::VectorXd p1 = p0 + 0.5 * e * grad;
    Eigen::VectorXd th1 = theta + e * inv_mass.cwiseProduct(p1);
    double lp1;
    Eigen::VectorXd grad1;
    logp_grad(th1, lp1, grad1);
    p1 += 0.5 * e * grad1;
    double h1 = -lp1 + 0.5 * p1.cwiseProduct(inv_mass).dot(p1);
    // Symmetric error, mirroring the sampler's acceptance statistic
    // exp(-|dH|) and its |dH| <= max_error tolerance: the one-sided form
    // reads h1 << h0 as inf > 0.5 and doubles eps on divergent-direction
    // errors.
    return std::exp(-std::abs(h1 - h0));
  };
  double a = accept_of(eps);
  int direction = a > 0.5 ? 1 : -1;
  for (int it = 0; it < 60; ++it) {
    if ((direction == 1 && a < 0.5) || (direction == -1 && a > 0.5)) {
      break;
    }
    eps *= (direction == 1) ? 2.0 : 0.5;
    a = accept_of(eps);
  }
  return eps;
}

}  // namespace walnutpie::detail
