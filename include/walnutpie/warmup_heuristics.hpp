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
  // W-43: draw the momentum ONCE, as in Hoffman & Gelman (2014) Algorithm 4.
  // A fresh draw per probe makes the one-step error's SIGN a lottery; the
  // statistic exp(-(h1-h0)) exceeds 0.5 for large NEGATIVE errors too (the
  // divergent-direction energy gain), so with fresh momenta the loop can
  // double eps on a cell whose real transitions carry |dH| = 8e6 (measured:
  // the pre-W-43 probe returned eps = 2.0 and 8.0 there).
  //
  // Momentum ~N(0, M) with M = inv_mass^-1 (per coordinate), matching the
  // sampler (transition_w draws rho = sqrt(mass) * z = z / sqrt(inv_mass)).
  // The previous line read rho.cwiseProduct(inv_mass.cwiseSqrt()), which
  // seeds ~N(0, inv_mass) — the inverted scale. Under a (near-)identity
  // mass the two coincide, but under the far-from-typical-set gradient
  // seed (mass ~1e7) the inverted probe moved ~1e-7x too little per step,
  // always "accepted", and never shrank eps (W-43).
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
    // Error MAGNITUDE, mirroring the sampler's acceptance statistic
    // exp(-|dH|) (macro_step) and its tolerance test |dH| <= max_error.
    // The pre-W-43 statistic exp(-(h1-h0)) treats divergent-direction
    // errors (h1 << h0, exp(+huge) -> inf > 0.5) as "accept and DOUBLE
    // eps" — on the W-43 pinned cell the one-step error is negative at
    // e=1 and positive at e=2, so the probe returned eps=2 (and 8)
    // where the real min-attempt |dH| was 8e6.
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
