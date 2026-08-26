# Property hunt on walnutpie core algebra

Branch `tests/property-hunt` (worktree `scratch/w61/walnutpie_w84`), based on `788d832`.
Driver: `tests/property_hunt.cpp` — standalone, plain-assert style with a soft-check
harness (no catch2 dependency).

Build & run:

```
/usr/bin/clang++ -std=c++20 -O2 -I include \
  -I /home/m0hawk/Documents/apin/stan/scratch/w61/walnutpie_instr/build_w54/_deps/eigen-src \
  tests/property_hunt.cpp -o /tmp/property_hunt && /tmp/property_hunt
```

Latest run: `244 checks, 0 failures, 3 REAL BUGS, 2 expected-failures`.

## Verdict table

| # | Property | Verdict |
|---|----------|---------|
| P1 | `combine` total weight: `((a⊕b)⊕c).logp_ == log(w1+w2+w3)`; total == `log(w1+w2)` for both Barker & Metropolis, Forward & Backward | PASS |
| P2 | `combine` endpoint invariance: result `theta_bk_`/`theta_fw_` are the temporally earliest/latest regardless of direction | PASS |
| P3 | Selection marginal: Barker `P(select new) == w2/(w1+w2)`; 3-span chain `P(s3) == w3/(w1+w2+w3)`; Metropolis `== min(1, w2/w1)`; commutativity in distribution — exact u-grid enumeration + 20k-seed ensembles through the real `combine` | PASS |
| P4 | `uturn` reflection symmetry: mirrored trajectories (same thetas, negated momenta) give `uturn<Forward>(a,b) == uturn<Backward>(am,bm)`; leapfrog mirror error < 1e-8 | PASS |
| P5 | `-inf`-weight spans: dead span never selected, total stays finite, `log_sum_exp(-inf,-inf) == -inf` | PASS |
| P6 | Reproducibility: same seed → bitwise-identical 500-draw chains (diagonal path) and 200-draw chains (low-rank path); `Random` stream sanity | PASS |
| P7 | Sampler marginal == π: 20k draws on 2D Gaussian, mean ~ 0, var ~ 1; 2k funnel draws stay finite | PASS |
| P8 | `WelfordAccumulator` vs two-pass brute force on 200 adversarial sequences (1e-12 ↔ 1e12, negatives); reset semantics; n<2 variance NaN | PASS |
| P9 | `OnlineMoments` mean == exact weighted mean (200 adversarial sequences) | PASS |
| P9' | `OnlineMoments` **variance** == exact weighted variance | **REAL BUG 1** |
| P10 | `LowRankMass::apply_inv` / `logp_momentum` == dense operator `A = D + sqrtD·U·C·Uᵀ·sqrtD` at r ∈ {0,1,d/2,d−1,d}, d ∈ {2,3,50} | PASS |
| P10' | `LowRankMass::log_det` == dense Cholesky log-det | PASS (initial FAIL was my reference missing the ×2) |
| P10'' | `LowRankMass::sample_momentum_from` == `Lc^{-T} z` (covariance `B·Bᵀ == A⁻¹`) | **REAL BUG 2** |
| P11 | `AntiWindupAdapter`: 100 saturated obs → exactly 13 forwarded (1-in-8); unsaturated always forwarded; `alpha == floor_alpha` passes (strict `<`) | PASS |
| P12 | Adam with alpha = 0 / 1 stays finite and positive | PASS |
| P12' | Adam with a single NaN alpha | **known gap** (no NaN guard at 788d832; XFAIL) |
| P13 | `DualAveraging` finite/positive under constant alpha and alpha = 0 bursts | PASS |

## REAL BUG 1 — `OnlineMoments::observe` lazy-expression aliasing corrupts the variance

`include/walnutpie/online_moments.hpp`, `observe()`:

```cpp
auto delta = y - mean_;                 // LAZY Eigen expression referencing mean_
weight_ = discount_factor_ * weight_ + 1;
mean_ += delta / weight_;               // updates mean_ (delta still lazy)
sum_sq_dev_.noalias() =
    discount_factor_ * sum_sq_dev_ + delta.cwiseProduct(y - mean_);
                                       // delta re-evaluates against the NEW mean_
```

`delta` is a `CwiseBinaryOp` holding a reference to `mean_`. When it is finally
evaluated on the last line, `mean_` has already been updated, so the code computes
`(y − mean_new)²` instead of the correct Welford term `(y − mean_old)(y − mean_new)`
(mathematically `delta² · df·W/W′`). The variance is therefore systematically
**underestimated** (the mean is unaffected). Affects every consumer of
`OnlineMoments::variance()` — i.e. mass-matrix adaptation.

Minimal repro (exact arithmetic, no randomness):

```cpp
OnlineMoments om(1.0, VectorXd::Zero(1), VectorXd::Zero(1));
om.set_discount_factor(1.0);
VectorXd y1(1); y1 << 1.0; om.observe(y1);
VectorXd y2(1); y2 << 2.0; om.observe(y2);
// exact: W = 3, mean = 1, S = 2, MLE variance = 2/3
om.variance()[0];   // returns 0.4166666667 (= 1.25/3), expect 0.6666666667
```

Fix direction (NOT applied — core code left untouched per task): materialize
`const Eigen::VectorXd delta = (y - mean_).eval();` (or reorder so the product is
formed before the mean update). This is the same class as the previously reported
"lazy-expression aliasing in OnlineMoments" incident — it is still present at
788d832.

## REAL BUG 2 — `LowRankMass::sample_momentum_from` scales on the wrong side of the low-rank bracket

`include/walnutpie/low_rank_mass.hpp`:

```cpp
Eigen::VectorXd invsq = D.cwiseInverse().cwiseSqrt();
Eigen::VectorXd out = invsq.cwiseProduct(z);
...
Eigen::VectorXd inner = U.transpose() * invsq.cwiseProduct(z);   // D^{-1/2} INSIDE
return out + U * (W * inner);
```

With `Lc = sqrtD (I + U((I+C)^{1/2} − I)Uᵀ)`, the correct momentum draw
`ρ = Lc^{-T} z` expands to

```
rho = D^{-1/2} (I + U W Uᵀ) z        with W = (I+C)^{-1/2} − I     [correct]
```

because `Lcᵀ = (I + U S Uᵀ) sqrtD` and `(I + U W Uᵀ)` does **not** commute with
`D^{-1/2}`. The implementation instead computes `(I + U W Uᵀ) D^{-1/2} z`. The two
agree only when the columns of `U` are coordinate-aligned (or `D` is a multiple of
identity) — which is why the existing `tests/leapfrog_property_test.cpp` harness
(reproducing the same formula) and the header docstring both inherit the error
silently.

Consequence: `Cov(ρ) = (I+UWUᵀ) D^{-1} (I+UWUᵀ) ≠ A^{-1}` with `A = M^{-1}`,
so the momentum draw is not from `N(0, M)` while `logp_momentum` scores it with
`A` — the invariant distribution of `transition_w_lr` is wrong whenever the
low-rank factors are engaged with a non-commuting diagonal.

Minimal repro (MC, 4·10⁵ draws): `D = diag(4,1)`, `U = (e1+e2)/√2`, `c = 1`:

```
A^{-1}     = [ 0.1875  -0.125 ; -0.125  0.75 ]
MC Cov(rho)= [ 0.2038  -0.156 ; -0.156  0.7316 ]   (≈10x above MC noise, systematic)
```

Exact-symbolism check in the driver (`rho == B z` with `B = D^{-1/2}(I+UWUᵀ)`)
fails for every random non-aligned `U` at r ≥ 1, d ∈ {2,3,50}. Fix direction:
`inner = U.transpose() * z` and `out = invsq.cwiseProduct(z + U * (W * inner))`,
i.e. move `D^{-1/2}` outside the bracket (and correct the docstring plus the
mirrored formula in `tests/leapfrog_property_test.cpp`).

## Known gap (expected-failure) — no NaN guard in `Adam`

`include/walnutpie/adam.hpp` `operator()(double alpha)` has no `std::isnan` guard:
a single `alpha = NaN` (e.g. from `min_accept = exp(-|logp − logp'|)` with a NaN
log density) makes `m_`, `v_`, and `step_size()` NaN permanently. This is the
"NaN alpha poisoning" class from the incident ledger — the guard is **absent** at
788d832. Recorded as XFAIL probes in the driver. (`AdEMAMix`, `AdaBelief`,
`DualAveraging`, `BatchedAdapter` share the same exposure; `AntiWindupAdapter`
with `floor_alpha > 0` incidentally drops exact-0 alphas but not NaN, since
`NaN < floor` is false.)

## Suspicious-but-inconclusive (noted, no failing property)

- `within_tolerance` (walnuts.hpp:237) comments "only tests one way": the
  reversibility ladder compares `|logp_next − logp|` with the *initial* joint as
  baseline; asymmetric-by-design, no violation demonstrated.
- `Random` holds its distributions per-instance but the underlying generator is
  shared by reference across all streams of a sampler; within-transition
  reproducibility holds (P6), and `LowRankMass::sample_momentum(std::mt19937_64&)`
  constructs a fresh `std::normal_distribution` per call — stateless for
  `normal_distribution`, so no ledger violation found.
- `AntiWindupAdapter` boundary semantics: an alpha exactly equal to `floor_alpha`
  is forwarded (strict `<`); documented in P11, matches the doc "below".
- `Adam::t_` is a `double` while `gradient_decay_pow_` multiplies in `double`:
  no precision issue observed for realistic iteration counts.

## Harness notes

- The initial P10 log-det failure and P4 uturn failure were **test-harness
  errors** (missing ×2 on the dense LLT log-det; wrong argument order in the
  mirrored `uturn<Backward>` call — the code's convention is
  `uturn<D>(span_old, span_new)`). Verified corrected before concluding.
- Statistical properties (P3 ensembles, P7) use 20k draws/seeds; tolerances
  ≥ 5σ. Analytic functors only; full run < 0.5 s.
