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
  // one leapfrog micro step. rho comes from the CALLER'S seeded RNG:
  // Eigen::VectorXd::Random() draws from std::rand(), which nothing ties to
  // --seed, so the heuristic made fixed-seed runs irreproducible (found via
  // bit-diffing paired runs; see PR #4 follow-up 14).
  auto accept_of = [&](double e) {
    Eigen::VectorXd rho = rand.standard_normal(theta.size()).matrix();
    Eigen::VectorXd p = rho.cwiseProduct(inv_mass.cwiseSqrt());  // ~N(0, M)
    double h0 = -lp + 0.5 * p.cwiseProduct(inv_mass).dot(p);
    Eigen::VectorXd p1 = p + 0.5 * e * grad;
    Eigen::VectorXd th1 = theta + e * inv_mass.cwiseProduct(p1);
    double lp1;
    Eigen::VectorXd grad1;
    logp_grad(th1, lp1, grad1);
    p1 += 0.5 * e * grad1;
    double h1 = -lp1 + 0.5 * p1.cwiseProduct(inv_mass).dot(p1);
    return std::exp(-(h1 - h0));
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
