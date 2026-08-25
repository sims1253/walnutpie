> # Optimizer & adaptation research notes (pass 1): why Muon doesn't apply here, and what does

Status: research notes + empirical ablation evidence. Not a proposal to change
defaults (see the sibling PRs for the patch series and their results).

## The two optimization problems inside walnutpie warmup

1. **Scalar step-size loop** — `theta = log(macro_step)`; stochastic gradient
   `g = accept_target − alpha` where `alpha = exp(−|Δ log p_joint|)` is observed
   per trajectory (or as a batched mean; alphas ∈ [0,1], skewed toward 1 when
   healthy, exactly 0 when divergent). Noise is multiplicative, autocorrelated,
   and **nonstationary** because the mass matrix adapts simultaneously.
2. **Diagonal mass estimation** — per-coordinate discounted online variance of
   draws and of scores (gradients), combined as
   `inv_mass = sqrt(Var_draw / Var_score)`; exponential forgetting
   `1 − 1/(offset + t)`; dimension up to thousands; diagonal only.

## Why Muon (and matrix-preconditioned optimizers generally) are structurally out

The momentum-orthogonalization step in Muon (Keller Jordan's momentum + Newton–Schulz
polar factor) **requires a 2-D weight matrix**: it whitens the update by orthogonalizing
it. For our two problems:

- the polar factor of a **scalar** is its sign — Muon degenerates to sign-SGD with
  momentum (which then needs an Adam-style scale anyway to pick a magnitude);
- the polar factor of a **1×d vector / diagonal** is the vector normalized by its RMS —
  i.e. the same normalization Adam's second moment already provides, computed less
  robustly.

This is not a matter of tuning: every major training stack that ships Muon special-cases
1-D and scalar parameters back to AdamW (verified in: Keller Jordan's modded-nanogpt
records — Adam lr 0.04 on scalars/embeddings while hidden matrices use Muon;
`torch.optim.Muon` docs; Keras 3; Megatron; DeepSpeed; Moonlight; microsoft/dion).
Walnutpie's entire adaptation state is scalar + diagonal, so there is nothing for
Muon to orthogonalize.

What *is* transferable from the Muon-adjacent literature (diagonal-preconditioning
hygiene and closed-loop scalar learning rates) is covered in the notes below, and the
first two actionable items (observation batching, gradient clipping) are implemented
and measured in the sibling PRs.

## Empirical ablation summary (posteriordb CORE_SET, 21 models × 4 chains × 1000+1000 × 3 reps)

| step optimizer | models completing | R-hat > 1.01 | geo wall |
|---|---|---|---|
| Adam (default) | 21 | 17 | 1.16 s |
| **Adam + batch stride 50** | 21 | **9** | 1.20 s |
| AdaBelief | 21 | 15 | 0.87 s |
| AdEMAMix | 19 | 15 | 2.07 s |
| Dual averaging (unbatched) | 11 | 8 | 1.65 s (10 abort: step-size → 0) |
| Dual averaging + PR average | 14 | 10 | 1.65 s |
| Dual averaging + batch 50 | 21 | 16 | 0.94 s |

Key empirical conclusions:

1. **Observation frequency beats algorithm choice.** The per-micro-step Adam update
   was over-updating on hundreds of correlated statistics per iteration; batching
   (stride 50, updating on the mean) halves the adaptation-failure count. Classic
   dual averaging — calibrated for one observation per iteration — *requires* batching
   to even survive (unbatched it drives the step size to zero on 10/21 models).
2. Adam is the right *family* for the scalar loop: in 1-D, Adam's update reduces to a
   signal-to-noise-attenuated sign update (see notes: `sign(m)·|m|/sqrt(m²+σ²)`),
   i.e. it is already the robustified sign method Muon would degenerate to.
3. AdEMAMix's slow EMA is actively harmful here — it fights the nonstationarity from
   the simultaneously adapting metric.
4. Gradient clipping is redundant once batched (9 vs 10 failing models) — kept as an
   off-by-default option.

The single biggest remaining failure class is NOT optimizer-related: chains frozen at
different posterior scales (see the init-robustness PR for the diagnosis of the
freeze-from-init mechanism, and the mass-estimator notes below for the metric-collapse
story).

---

The remainder of this file is the full pass-1 research report (primary sources,
optimizer comparison table, uncertainty register, and the exact query log).

# SOTA optimizer variants for MCMC adaptation loops — web research report

**Scope:** optimizer/estimator upgrades for `walnutpie` (Flatiron Institute's C++ WALNUTS sampler,
github.com/flatironinstitute/walnutpie), with two concrete targets:

1. **(a) Scalar step-size loop** — 1-D stochastic optimization over `theta = log(stepsize)`,
   gradient `g = target_accept − observed_accept_stat` per trajectory/micro-step, currently scalar
   Adam with `1/t^0.5` learning-rate decay (repo defaults: lr 0.05, β1 0.8, β2 0.9, eps 1e-4,
   decay 0.5; the lab's experimental config reportedly uses lr 0.2, β≈(0.9, 0.95)).
2. **(b) Diagonal mass matrix** — per-coordinate online variance estimation from discounted
   Welford moments of draws and scores: `inv_mass = sqrt(Var_draws / Var_score)`, discount
   `1 − 1/(offset + t)` (walnutpie `OnlineMoments`/`MassEstimator`, d up to ~7000).

**Date of research:** 2026-08-20. All links/claims checked against primary sources on that date.
Labels: **[paper]** = peer-reviewed or arXiv preprint, **[blog]** = author/first-party blog,
**[repo]** = source code, **[forum]** = discussion thread, **[manual]** = official documentation.

Sources of truth consulted include: arXiv (API + abstract pages), GitHub (repo trees, raw files,
search), mc-stan.org reference manual + stan-dev/stan sources, Stan Discourse JSON API,
TensorFlow Probability sources, Keller Jordan's blog and modded-nanogpt records, and the
curated secondary source `Hiroki11x/awesome-muon-optimizer` (checked 2026-08-14; used as a
map, with every load-bearing claim re-verified against its primary source).

---

## TL;DR — top 5 actionable changes, ranked by (applicability × benefit) / cost

### 1. Clip the step-size gradient and batch the acceptance statistic (loop (a)) — do first
- **Change:** clip `g = delta − alpha` to `[−c, +c]` (Stan Discourse evidence uses the equivalent
  asymmetric form `min(delta − alpha, 1 − delta)`); update the adapter on the *mean* alpha over a
  stride of micro-steps (walnutpie already ships `BatchedAdapter`; make it default), not on each
  noisy micro-step statistic.
- **Expected benefit:** removes the dominant failure mode — a single near-zero-alpha trajectory
  produces |g| up to 0.8 and crashes log-stepsize, after which recovery is slow ("stepsize
  collapse"). The Stan Discourse author reports warmup iteration counts roughly halved after the
  clip fix on pathological models **[forum: topic 5995]**. Batching averages ~hundreds of
  correlated micro-step statistics → directly cuts gradient noise; DA was calibrated for 1
  observation/iteration, Adam-with-stride restores that calibration.
- **Risk:** clipping introduces bias if the target is systematically far from alpha (early
  warmup); mitigated by a generous clip (e.g. c ∈ [0.3, 0.8]) and by unclipping after the first
  fast window.
- **Cost:** trivial, header-only, ~10 lines.

### 2. Polyak–Ruppert averaging of log-stepsize + re-anchor at mass-matrix updates (loop (a))
- **Change:** maintain an averaged iterate `theta_bar` with `t^{-kappa}` weights (kappa ≈ 0.5–0.75)
  and *use* `exp(theta_bar)` when freezing the step size / ending a window; on every mass-matrix
  refresh, reset Adam state and re-anchor (Stan restarts DA with `mu = log(10 · eps_bar)` after
  each slow window **[manual]**, **[repo: stan-dev/stan]**).
- **Expected benefit:** averaging is the classical variance-reduction for stochastic
  approximation (Polyak–Ruppert; Ruppert 1988; Polyak 1990 — pre-arXiv, textbook material) and is
  exactly what Stan's DA `x_bar` provides; walnutpie's Adam adapter currently has no averaged
  output. Re-anchoring handles the *known nonstationarity* (optimal step size jumps when the
  metric changes) instead of letting stale momentum fight it.
- **Risk:** averaging over a window that spans a metric update smears two regimes — average only
  within the current window (this is what Stan's memoryless windows do).
- **Cost:** trivial (~15 lines).

### 3. Two-phase decay + closed-loop learning rate instead of a single `1/t^0.5` (loop (a))
- **Change:** keep decay exponent 0.5 while the metric is adapting, switch to ≥0.75 in the
  terminal buffer after the last metric update; optionally replace the *scheduled* magnitude with
  an AdaGrad-Norm-style scalar accumulated state (closed-loop lr).
- **Expected benefit:** classical Robbins–Monro conditions need `Σ a_t = ∞, Σ a_t² < ∞`, i.e.
  decay exponent > 0.5 for the second condition; `t^{-0.5}` never "converges" (fine while the
  target drifts, bad once it stops — the stepsize keeps bouncing with amplitude ∝ lr).
  Muon-adjacent literature now formalizes exactly this "closed-loop vs scheduled magnitude"
  distinction for orthogonalized updates: OptMuon (AdaGrad-Norm scalar on top of the direction,
  **[paper: arXiv 2606.08783]**) and AdaGO (**[paper: arXiv 2509.02981]**). For a root-finding
  problem, adaptive-Polyak/schedule-free ideas are the closest analogue
  (**[paper: arXiv 2405.15682]**, **[arXiv 2511.07767]**).
- **Risk:** adaptive-lr machinery adds a state and a failure surface; the cited papers validate
  on LLM pretraining, *not* on 1-D acceptance-statistic root finding — treat as a benchmark
  experiment, not a swap-in.
- **Cost:** small (~20 lines).

### 4. Shrink + floor the mass-matrix estimates; robustify the score variance (loop (b))
- **Change:** (i) shrink the discounted variances toward the initialization with an
  effective-sample-size-aware weight `n_eff/(n_eff + κ)` (Kish `n_eff = (Σw)²/Σw²` for the
  discounted Welford weights), mirroring Stan's regularized estimate
  `var ← (n/(n+5))·var + 1e-3·(5/(n+5))` **[repo: stan-dev/stan `var_adaptation.hpp`, verified]**;
  (ii) floor `Var_score` (currently only the initial smoothing guards the `sqrt(Var_draw/Var_score)`
  division); (iii) optionally Winsorize/trim scores before their second moments, or use a
  quantile-based scale for `Var_score`, for funnel-ish heavy tails.
- **Expected benefit:** shrinkage toward a structured target is the classical fix for noisy
  covariance/variance estimation (Ledoit–Wolf 2004; OAS **[paper: arXiv 0907.4698]**), and it is
  precisely what production Stan does at every window end; robust scatter estimation under
  elliptical heavy tails is the domain of Tyler's M-estimator (**[paper: Tyler 1987, Ann. Statist.]**;
  regularized Tyler: Sun–Babu–Palomar, IEEE TSP 2014). With d ~ 7000 and only hundreds of
  effective draws, the unregularized diagonal is statistically fragile; per-coordinate shrinkage
  is the cheapest available variance reduction.
- **Risk:** over-shrinkage slows adaptation in badly-scaled models (Stan picked κ=5 by
  convention, not theory); robust estimators can bias the metric for genuinely multi-modal tails.
- **Cost:** small, O(d), header-only.

### 5. (Speculative, higher cost) Structured metric beyond diagonal: low-rank or score-covariance preconditioner (loop (b))
- **Change:** augment the diagonal metric with a rank-m correction learned online, following
  (i) sparse-preconditioner adaptive MCMC at O(m²d) per iteration (**[paper: arXiv 2604.09286]**),
  (ii) Fisher-adaptive MALA, whose *optimal* preconditioner is provably the inverse Fisher
  covariance computed from outer products of log-target gradients — the same score stream
  walnutpie already accumulates (**[paper: arXiv 2305.14442]**), or (iii) the metric-selection
  criterion + fast variant of Bales et al. (**[paper: arXiv 1905.11916]**, implemented for Stan).
  Note walnutpie's `sqrt(Var_draws/Var_score)` geometric-mean rule is already a diagonal
  approximation of the Fisher-adaptive idea; the cited paper is the formal justification to cite
  for it.
- **Expected benefit:** diagonal metrics fail on strongly correlated posteriors; rank-10–50
  corrections capture most of the dense-metric benefit at a fraction of the O(d²) cost.
- **Risk:** new moving parts, new tuning (rank, refresh cadence), ergodicity care (adaptation
  must freeze/vanish before sampling).
- **Cost:** moderate-to-high (dense linear algebra in the header; Eigen suffices).

**Bottom line for the Muon question:** there is **no published or production use of Muon-style
orthogonalization on scalar or diagonal-vector parameters — and there is a structural reason**:
on a 1×n vector the polar factor is just the normalized vector, so "Muon on a vector" degenerates
to normalized (sign-like) momentum, which 1-D Adam already implements (see §2.1). Every major
Muon stack explicitly routes 1-D/embedding/scalar parameters to AdamW (verified: Keller Jordan's
blog and repo, `torch.optim.Muon` docs, Keras 3, Megatron-Core, DeepSpeed, Moonlight, microsoft/dion,
NVIDIA Emerging-Optimizers). Muon is therefore **not actionable** for either walnutpie loop
except as *inspiration* for diagonal preconditioning hybrids (MALT, Muon², NAMO), which are
matrix-methods; see "Rejected ideas".

---

## Optimizer/estimator table

Ratings are relative to *this use case* (1-D noisy nonstationary root finding; d-dim online
variance estimation). "1-D" = scalar stepsize loop; "vec" = vector/diagonal mass-matrix use.

| Name | Year | Core idea | Noise robustness | Nonstat. robustness | 1-D? | Vec? | C++ cost | Source |
|---|---|---|---|---|---|---|---|---|
| Dual averaging (Nesterov; NUTS use) | 2009/2014 | primal-dual averaging on log-stepsize with t^{-κ} averaged iterate | med (averages h) | med (re-anchor at windows) | **yes (standard)** | n/a | trivial | [paper] 1111.4246; Nesterov 2009 |
| Adam (scalar, on log ε) | 2014 | EMA m/EMA v with bias correction | high (self-normalized) | med (stale m,v) | **yes** | n/a | trivial | [paper] 1412.6980 |
| TFP "simple" step-size adaptation | 2017+ | multiplicative ±(1+rate) on the *sign* of accept-stat error | low-med (sign flips) | low | yes | per-dim | trivial | [repo] tfp `simple_step_size_adaptation.py` |
| Lion | 2023 | sign of interpolated momentum | low-med | med | yes (=sign SA) | per-dim sign | trivial | [paper] 2302.06675 |
| AdaBelief | 2020 | v tracks (g−m)² not g² | high (gain adapts to consistency) | med | yes (implemented in walnutpie) | yes | trivial | [paper] 2010.07468 |
| AdEMAMix | 2024 | fast + very slow (β≈0.9999) EMAs | high | **low** (slow EMA stale) | marginal | marginal | trivial | [paper] 2409.03137 |
| RAdam | 2019 | rectify variance of adaptive lr (warmup-free) | med-high | low relevance (bias transient is short here) | marginal | marginal | trivial | [paper] 1908.03265 |
| AMSGrad | 2018 | non-decreasing v̂ | med | **low** (v ratchets up after regime shift) | marginal | marginal | trivial | [paper] 1904.09237 |
| CAME | 2023 | int8 states + "confidence" modulation | med | med | no benefit at 1-D | no | trivial | [paper] 2307.02047 |
| Sophia | 2023 | diagonal-Hessian (Gauss–Newton) sketch, clipped, every-k | med | med | no (needs Hessian) | analogy only | n/a | [paper] 2305.14342 |
| Schedule-Free AdamW | 2024 | z/x iterate averaging, no lr schedule | high | med | concept yes; MCMC-validity caveat | yes | small | [paper] 2405.15682 |
| Adaptive-Polyak schedule-free | 2025 | Polyak steps replace schedule | high | med-high | concept yes | yes | small | [paper] 2511.07767 |
| AdaGrad-Norm / closed-loop scalar lr (OptMuon, AdaGO) | 2025–26 | scalar accumulated-gradient state sets magnitude | high | high | **yes (concept)** | yes | small | [paper] 2606.08783; 2509.02981 |
| Muon | 2024 | momentum → Newton–Schulz polar factor, 2-D weights | high | med | **no** (degenerates to normalization) | no | n/a here | [blog] kellerjordan.github.io/posts/muon/ |
| Moonlight/MuonClip recipe (Muon@scale) | 2025 | + weight decay, RMS-match scaling `0.2·sqrt(max(A,B))`, QK-clip | high | med | no | no | n/a | [paper] 2502.16982; 2507.20534 |
| Muon² (adaptive 2nd-moment + orthogonalize) | 2026 | Adam-style v applied to momentum before NS | high | med | no | hybrids collapse to normalized Adam at 1-D | moderate (NS) | [paper] 2604.09967 |
| NAMO / NAMO-D | 2026 | principled orthogonalized momentum × Adam-type adaptivity | high | med | no | no | moderate | [paper] 2602.17080 |
| MALT | 2026 | row/col diagonal preconditioners conjugated around NS | high | med | no | **inspiration for diag preconditioning** | moderate | [paper] 2608.05088 |
| Polar Express | 2025 | minimax-optimal iteration-varying NS polynomials | n/a (numerics) | n/a | no | no | n/a | [paper] 2505.16932 |
| Chebyshev NS / IFNSO / Turbo-Muon / Hierarchical Muon / MuD / AuON | 2025–26 | cheaper/looser orthogonalization solvers | n/a | n/a | no | no | n/a | 2506.10935; 2602.02500; 2512.04632; 2606.27216; 2603.17970; 2509.24320 |
| PowerMuon (σ→σ^p, p<1) | 2026 | partial spectral flattening | med-high | med | no | no | moderate | [paper] 2606.13867 |
| Scion (norm-constrained LMO) | 2025 | explicit norm choice → lr transfer | high | med | no | no | moderate | [paper] 2502.07529 |
| Dion / Dion2 / MuonBP / SignMuon | 2025–26 | distributed orthogonalized updates | n/a | n/a | no | no | n/a | 2504.05295; 2512.16928; 2510.16981; 2605.16311 |
| Discounted Welford (walnutpie `OnlineMoments`) | — | weighted recursive moments, forgetting λ | med | high (by design) | n/a | **yes (current)** | O(d) | [repo] walnutpie; Welford 1962 |
| Shrinkage (LW/OAS/Stan rule) | 2004/2010/2017 | regularize estimate toward structured target | high | med | n/a | **yes (recommended)** | O(d) | journals; 0907.4698; stan source |
| Tyler / regularized Tyler | 1987/2014 | distribution-free robust scatter (heavy tails) | high (tails) | med | n/a | yes (score outer products; O(d²) unless diagonalized) | high | journals |
| Fisher-adaptive MALA preconditioner | 2023 | optimal preconditioning = inv score-covariance | high | med | n/a | yes (dense) | high | [paper] 2305.14442 |

---

## 1. Muon and variants (2024–2026) — what actually applies

### 1.1 The structural fact: orthogonalization needs 2-D structure; scalars are routed to AdamW

- The canonical Muon definition **[blog]** (Keller Jordan, Dec 2024,
  https://kellerjordan.github.io/posts/muon/, no standalone paper): *"Muon is an optimizer for 2D
  parameters of neural network hidden layers … scalar and vector parameters of the network, as
  well as the input and output layers, should be optimized by a standard method such as AdamW."*
- Reference implementation **[repo]** (KellerJordan/Muon, ★2.8k): applies to `ndim ≥ 2`; ships
  `MuonWithAuxAdam` precisely for the embeddings/scalars/head group.
- `torch.optim.Muon` (PyTorch 2.9+) **[manual/docs]**: 2-D params only; docs direct biases and
  embeddings to a separate AdamW instance; exposes both update-scale conventions (`original`
  `sqrt(max(1,A/B))` and `match_rms_adamw` `0.2·sqrt(max(A,B))` — the Moonshot RMS-matching rule).
- Keras 3 `keras.optimizers.Muon` **[manual]**: built-in AdamW fallback inside one optimizer for
  all 0-D/1-D variables, embeddings, and the final layer.
- Megatron-Core **[manual]**: `muon_scalar_optimizer` (Adam or Lion) handles embeddings, biases,
  norms; NVIDIA-NeMo/Emerging-Optimizers **[repo]** asserts all params 2-D and routes 1-D to AdamW.
- DeepSpeed config docs **[manual]**: hybrid optimizer routes embeddings/norm/biases/head to Adam
  with a separate `adam_lr`.
- microsoft/dion **[repo]**: ships Dion/Dion2/Muon/NorMuon; orthonormal updates on 2-D matrices
  "with Lion or AdamW for scalars".
- modded-nanogpt speedrun record (2024-12-04_ValueEmbed, fetched from repo) **[repo]**:
  `optimizer1 = Adam([wte, vte], lr=0.6, betas=(0.8,0.95))`; `optimizer2 = Adam([lm_head], lr=0.008)`;
  `optimizer3 = Muon(matrix_params, lr=0.05, momentum=0.95)`;
  `optimizer4 = Adam(scalar_params, lr=0.04, betas=(0.8,0.95)) # note that this learning rate is neither sensitive nor tuned`.
  This is the clearest production evidence that (i) scalars stay on Adam and (ii) the scalar-group
  lr is a flat, insensitive knob.

**Why "Muon on a scalar" is vacuous (algebra, no citation needed):** for a 1×n row vector v, the
polar/orthogonalized factor UVᵀ is v/‖v‖; for n=1 it is sign(v). So a Muon-style update on a
diagonal vector is just RMS-normalized momentum, and on the scalar stepsize it is a smoothed sign
ascent — which Adam already approximates (§2.1). Nothing in the 2024–2026 literature
(~60+ Muon papers indexed by the curated list, checked 2026-08-14) reports benefits of
orthogonalization for scalar/diagonal parameters; the hybrids that add vector structure
(Muon² 2604.09967, NAMO 2602.17080, AdaMuon 2507.11005, MALT 2608.05088, NorMuon 2510.05491)
all still orthogonalize *matrices* and use diagonal statistics only as preconditioners around
the spectral step.

### 1.2 The Muon ideas that DO transfer conceptually to walnutpie
- **Diagonal preconditioning around a normalized update (MALT, 2608.05088 [paper]):** row/column
  squared-gradient statistics conjugate the momentum, then Frobenius grafting restores scale.
  Analogy for loop (b): walnutpie's `sqrt(Var_draw/Var_score)` is already a diagonal
  two-statistic conjugation; MALT's grafting step is the analogue of walnutpie's additive
  smoothing. The transferable lesson is *statistic hygiene* (floors, refresh cadence), not a new
  algorithm.
- **Adaptive second-moment before normalization (Muon² 2604.09967; NAMO 2602.17080):** at 1-D
  this collapses to Adam itself — supports the view that scalar-Adam is the correct fixed point
  of this family.
- **Closed-loop step magnitude (OptMuon 2606.08783; AdaGO 2509.02981 [paper]):** replace the
  externally scheduled lr with a scalar accumulated-gradient state (AdaGrad-Norm style) — the most
  directly transferable idea for loop (a); see TL;DR #3.
- **Partial spectral shaping (PowerMuon 2606.13867; isotropic-curvature model 2511.00674):**
  "flatten less when curvature is anisotropic-uncertain" is the same philosophy as shrinkage in
  loop (b): don't trust a flat/noisy estimate; interpolate toward a prior.
- **Update-RMS matching (Moonlight 2502.16982 [paper]):** scale updates so per-coordinate RMS
  matches a reference optimizer — the "lr transfer" trick the brief calls "momentum view". This is
  the honest, verifiable content behind that label (see §1.4).

### 1.3 Orthogonalization-solver alternatives (Newton–Schulz replacements)
Established 2025–26 options: Polar Express (minimax-optimal iteration-varying polynomials,
ICLR 2026 Oral, 2505.16932), Chebyshev-type coefficients via Remez (2506.10935), IFNSO
iteration-free single polynomial (2602.02500), Turbo-Muon preconditioned initial guess
(2512.04632), Hierarchical Muon tiled NS (2606.27216), MuD Cholesky-style whitening (2603.17970),
AuON linear-time scaling without NS (2509.24320), Gram-matrix NS (Tri Dao blog 2026). Also the
counter-evidence: exactness barely matters (How Much Orthogonalization Does Muon Need?,
2606.00371; Beyond the Ideal, 2510.19933 — NS iteration count must be co-tuned with lr).
**None of this is actionable for walnutpie** — there is no matrix to orthogonalize in either
loop. Could not identify any method named "GMS" in this literature (searched arXiv + curated
list + GitHub); the brief's parenthetical guess ("greedy momentum signal") does not match any
paper I could find — flagging as likely a mis-remembered acronym.

### 1.4 Notes on specific brief items (honest verification status)
- **"Moonshot's Muon (momentum view, better lr transfer)":** the verifiable primary source is the
  Moonlight paper (arXiv 2502.16982, MoonshotAI; code MoonshotAI/Moonlight): weight decay +
  per-parameter update-RMS matching `0.2·sqrt(max(A,B))` so AdamW learning rates transfer;
  demonstrated on a 16B MoE, 5.7T tokens; later hardened into MuonClip for Kimi K2 (1T MoE,
  15.5T tokens, zero loss spikes; arXiv 2507.20534). A separate "momentum view" blog post could
  **not** be located as of 2026-08-20: `github.com/MoonshotAI/Muon` no longer exists (checked
  via org repo listing and the curated list's "verified absences", 2026-08-14) and
  `moonshotai.github.io` now redirects to moonshot.cn; the Wayback Machine has no snapshot.
  Treat the RMS-matching paper as the canonical statement.
- **MuAdam:** the Muon+Adam hybrid family (element-wise second moments on/around the
  orthogonalized update) is realized as AdaMuon (2507.11005), Muon² (2604.09967), NAMO
  (2602.17080); "MuAdam" as a distinct published name was not found.
- **"Muon-lite":** no paper or repo by that name found (searched arXiv, GitHub, curated list).
- **AnyPrecision:** mixed-precision Adam with Kahan summation (Kalamkar et al. 2019 lineage;
  historically `apex.optimizers.AnyPrecisionOptimizer`). Verified that current NVIDIA/apex
  master (tags 25.04–25.09) no longer ships it. Irrelevant to walnutpie's fp64 C++ anyway.
- **nanotron's improvements:** current huggingface/nanotron `main` tree (fetched 2026-08-20)
  contains **no** Muon code — the improvements attributed to the HF/nanotron ecosystem actually
  live in Moonlight/Kimi (RMS matching, weight decay, QK-clip) and microsoft/dion.
- **Muon2/Muon-like successors:** no "Muon2" repo/paper by Keller Jordan exists (repo listing
  checked). The published "Muon²" is 2604.09967 (second-moment-preconditioned Muon, Apr 2026).
  The 2026 successor wave is: partial shaping (PowerMuon), curvature-aware (Mousse 2603.09697,
  MALT, Second-Order Muon 2608.09763), schedule-free Muon (Anytime Training 2605.23061, AMUSE
  2605.22432), and scaling caveats (Fantastic Pretraining Optimizers 2509.02046 — the tuning-parity
  critique; Hyperball Optimization 2606.16899).

---

## 2. Adam-family for noisy low-dimensional stochastic optimization (loop (a))

### 2.1 What scalar-Adam *is* on this problem (useful for judging variants)
With persistent mean μ and noise σ in g: `m̂ → μ`, `v̂ → μ² + σ²`, so the update direction is
`sign(μ)·|μ|/sqrt(μ²+σ²)` — an SNR-attenuated sign step of magnitude ≤ lr(t). I.e., walnutpie's
adapter is a self-normalized stochastic root finder: bounded steps, robust to gradient
magnitude, gain automatically reduced when the statistic is pure noise. This is the correct
behaviour family for a noisy 1-D root-finding problem, and it explains why sign/normalized
methods (Lion, TFP-simple, Muon-on-vectors) are all close cousins here. Framing support:
steepest descent under L∞ vs L2 norms (**[paper] Old Optimizer, New Norm, arXiv 2409.20325**;
**[paper] Geometry of Sign Gradient Descent, arXiv 2002.08056**; **[paper] Kovalev trust-region
view, arXiv 2503.12645** — normalized SGD and signSGD-with-momentum as special cases of
non-Euclidean trust regions).

### 2.2 The problem is stochastic *root finding*, not loss minimization
The target is `E[alpha(theta)] = delta`, i.e. Robbins–Monro stochastic approximation with a
sign-flipping objective gradient, not convex minimization. Consequences:
- The classical conditions `Σa_t=∞, Σa_t²<∞` (Robbins–Monro 1951) want decay exponent > 0.5;
  walnutpie's 0.5 keeps the loop *tracking* (never converging) — right during metric drift,
  wrong in the terminal buffer. Stan's DA uses kappa = 0.75 on the averaged iterate.
- Variance reduction should come from (i) averaging observations (batching), (ii) averaging
  iterates (Polyak–Ruppert; this is precisely DA's `x_bar`), (iii) clipping outliers — all
  cheaper and better-understood than exotic moment rules. Reference framing for adaptation-as-SA:
  Andrieu & Thoms (2008), "A tutorial on adaptive MCMC", Statistics and Computing (covers the
  SR recursion `log ε ← log ε + γ_t(δ − α)` and diminishing adaptation).

### 2.3 Variant-by-variant verdicts (loop (a))
- **AdaBelief (2010.07468) [paper]:** v tracks deviation from the mean → automatic gain control:
  large v when the statistic oscillates (shrink), small v when consistent (push). walnutpie
  already implements it (`step_optimizers.hpp`). No published evidence in SA/stepsize settings;
  the LLM evidence (fast convergence, GAN stability) is suggestive only. **Cheap A/B candidate.**
- **AdEMAMix (2409.03137) [paper]:** the slow EMA (β3 up to 0.9999, horizon ~5k steps) is
  fundamentally mismatched to a target that moves on the metric-update timescale; the paper
  itself needs a multi-thousand-step warmup ramp *for the slow term*. walnutpie's scalar AdEMAMix
  exists; expect it to help only with β3 close to β1, at which point it is ~Adam. **Low value.**
- **CAME (2307.02047) [paper]:** memory-efficient (int8 v) + "confidence" = normalized m²/v —
  at 1-D the memory argument evaporates and confidence ≈ AdaBelief-ish modulation. **No value
  at 1-D.**
- **Lion (2302.06675) [paper]:** 1-D Lion = sign of interpolated momentum with fixed magnitude —
  structurally identical to TFP's `SimpleStepSizeAdaptation` rule `ε ← ε(1+r) if log α > log δ
  else ε/(1+r)` (verified in TFP source). No magnitude adaptation → oscillation amplitude ~ lr;
  Lion's own paper requires careful decay. **Rejected for loop (a).** ("Lion2" does not exist;
  nearest 2026 relative: CLion 2604.x, cautious-Lion, not relevant.)
- **Sophia (2305.14342) [paper]:** diagonal Hessian every-k with clamping — inapplicable at 1-D
  (would require noisy finite differences of α w.r.t. log ε across trajectories; unpublished
  territory). Its useful pattern is *update-expensive-statistic-every-k-with-clamping*, which is
  how mass-matrix refresh already works.
- **Schedule-Free AdamW (2405.15682) [paper] + adaptive Polyak (2511.07767):** replaces the
  schedule with z/x iterate averaging. For a *bounded warmup phase* the schedule is not the
  problem; the appeal is the averaged iterate (again Polyak–Ruppert). **Adopt the averaging,
  skip the framework.** Caveat for MCMC: any adaptation that continues during sampling must be
  diminishing for ergodicity (Roberts & Rosenthal 2007) — a constant-lr schedule-free loop run
  post-warmup would violate the usual adaptive-MCMC conditions; walnutpie should keep the
  freeze-at-end-of-warmup design.
- **AMSGrad (1904.09237) [paper]:** non-decreasing v̂ ratchets up after the mass-matrix regime
  shift and then permanently shrinks steps — exactly wrong under nonstationarity (the AMSGrad
  counterexample literature concerns nonstationary/increasing gradients). **Rejected.**
- **RAdam (1908.03265) [paper]:** warmup-as-variance-rectification addresses early v̂ noise;
  here the bias-correction transient lasts only ~1/(1−β2) observations (tens of micro-steps),
  and clipping (#1) addresses the same symptom more directly. **Marginal.**
- **Closed-loop scalar lr (AdaGrad-Norm / OptMuon 2606.08783 / AdaGO 2509.02981 / DoG
  2302.12022 / D-Adaptation 2301.07733):** accumulate Σg² (or realized iterate gaps) into a
  scalar state that sets the magnitude — removes the hand-tuned `1/t^0.5` in favour of a
  data-dependent decay. Validated only on optimization problems, not root finding; but the state
  is one scalar, the failure mode is benign (bounded lr), and it composes with averaging. **Best
  "modern optimizer" bet for loop (a); benchmark it.**

### 2.4 What modern stacks actually do for 1-D/scalar params (Q5)
AdamW, with per-group learning rates, everywhere: modded-nanogpt (lr 0.6 embeddings / 0.008 head /
0.04 scalars vs Muon 0.05, with the "neither sensitive nor tuned" comment), PyTorch/Keras/
Megatron/DeepSpeed/Moonlight/dion docs all route 1-D to Adam(W) (see §1.1). No evidence exists
that anything beats Adam for 1-D noisy targets at these scales; the LLM-side "evidence" is only
that nobody bothers to change it because it is insensitive. Honest reading: **Adam is a fine
default for the scalar loop; the wins are in noise handling (clip/batch/average) and decay
policy, not in fancier moment rules.**

---

## 3. HMC/NUTS step-size adaptation beyond dual averaging

- **Baseline:** Hoffman & Gelman (1111.4246, JMLR 2014) adapt log ε by Nesterov (2009) dual
  averaging with δ=0.8, γ=0.05, t0=10, κ=0.75 (defaults per Stan reference manual). Stan
  restarts DA (`mu = log(10 ε̄)`) at each slow-window boundary and freezes after warmup
  **[manual; repo]**.
- **Known pathologies + fixes (primary-source status: forum, but concrete):**
  - *Restart overshoot:* the 10× restart after metric updates makes step size oscillate across
    windows (Stan Discourse topic 5995, 2018 — includes formulas).
  - *Gradient blow-up at α→0:* unbounded `δ − α` crashes the step size, recovery burns
    treedepth-capped iterations; fix that reportedly halves warmup on pathological models:
    clip the gradient, e.g. use `min(δ−α, 1−δ)`. Same thread.
  - *Target-statistic conservatism:* Betancourt's 2019 proposal to replace the mean-min
    acceptance proxy (Stan Discourse topic 9532) — evidence the *target* itself is a design
    choice; walnutpie's per-micro-step statistic inherits this question.
  - *Jitter:* Stan exposes post-hoc step-size jitter (default 0) to avoid resonance with fixed
    curvature regimes **[manual]**; orthogonal to the optimizer choice.
- **WALNUTS itself (2506.18746, Bou-Rabee, Carpenter, Kleppe, Liu, June 2025):** adapts the
  leapfrog step size *within* each orbit from a dyadic schedule under an energy-error threshold —
  this is why walnutpie's global-ε adaptation loop sees an unusually noisy effective acceptance
  statistic, and why robust-noise treatment (batching/clipping) matters more here than in Stan.
- **Alternatives to DA in the recent literature:**
  - **ATLAS (2410.21587):** adapts ε per iteration from a low-rank local Hessian's largest
    eigenvalue + local U-turn monitoring (local, curvature-driven — complements, not replaces,
    a global adapter).
  - **GIST (2404.15253):** Gibbs-sampling framework for tuning parameters (path length; the
    step-size analogue is "self-tuning via conditional draws") — conceptually relevant to
    within-orbit schemes like WALNUTS.
  - **PDMP practical locally-adaptive step sizes (2503.11479);** adaptive multi-stage
    integration schemes (2307.02096); entropy-based adaptive HMC (2110.14625, mass matrix via
    gradient information).
  - **TFP's two options** [repo]: dual averaging vs the "simple" multiplicative sign rule —
    useful as the two poles (averaged-gradient vs pure-sign) that walnutpie's Adam sits between.
  - **"AdaHMC":** no paper by this name exists (searched arXiv title/abstract; zero hits).
  - Published *controlled comparisons* of Adam vs DA for step-size adaptation: none found —
    walnutpie's own benchmarks are the state of the art here; the theoretical prior (§2.2) says
    they should be close once both are clipped/averaged/decayed properly.
- **Post-warmup adaptation:** adaptive-MCMC ergodicity requires vanishing adaptation (Roberts &
  Rosenthal 2007; Andrieu & Thoms 2008 tutorial). Practical stacks freeze at the end of warmup
  (Stan, TFP `num_adaptation_steps`). If walnutpie ever adapts during sampling, use a
  diminishing schedule (e.g. lr ∝ t^{-0.75+}) and re-validate.

---

## 4. Online variance estimation for the mass matrix (loop (b))

- **Current scheme is sound:** discounted Welford (`OnlineMoments`) is the standard numerically
  stable recursive moment estimator; exponentially-discounted weighted variance is the classic
  streaming estimator (West & Harrison-style discounting; the canonical modern write-up is
  Tony Finch's unpublished 2009 note "Incremental calculation of weighted mean and variance" —
  the Cambridge page is currently offline, so cite via walnutpie's own documentation of the
  same recursion). Known caveat: the naive discounted variance is biased; the bias is O(1/n_eff)
  and second-order relative to the sampling noise here.
- **Stan's practice (verified from source):** windowed, *memoryless* estimation — variance from
  the current window only, then regularized:
  `var ← (n/(n+5))·var + 1e-3·(5/(n+5))·1` (stan-dev/stan `src/stan/mcmc/var_adaptation.hpp`),
  with explicit overflow guards; windows double (25→…→term buffer 50, init buffer 75) so stale
  early estimates are discarded, not discounted **[repo; manual]**. walnutpie's discounting is a
  smoother version of the same idea; the missing piece is the **shrinkage term** (TL;DR #4).
- **Shrinkage estimators:** Ledoit–Wolf (J. Multivariate Analysis 2004; "Honey, I Shrunk the
  Sample Covariance Matrix", JPM 2004) and OAS ("Shrinkage Algorithms for MMSE Covariance
  Estimation", arXiv 0907.4698, IEEE TSP 2010) give closed-form optimal-ish shrinkage for
  Gaussian data; both reduce to scalar mixing of the sample estimate with a structured target —
  directly implementable per-coordinate at O(d) with n_eff in place of n. **No published "online
  Ledoit–Wolf" for streaming/discounted windows was found** — the honest statement is that the
  plug-in with Kish's effective sample size is the natural adaptation, untested in the
  literature.
- **Robustness under heavy tails (funnel-ish targets):** Tyler's M-estimator (Ann. Statist.
  1987) is distribution-free for elliptical heavy tails (fixed by trace normalization);
  regularized Tyler (Sun, Babu, Palomar, IEEE TSP 2014) makes it well-posed for d > n — the
  regime walnutpie lives in. Practical diagonal surrogates: Winsorize/trim the score stream
  before second moments, or replace Var_score by an interquantile-based scale. No direct
  published evaluation of robust estimators inside HMC warmup was found; mark as benchmark
  territory. Related: the non-centered-vs-centered pathology is the standard fix for funnels
  (Betancourt & Girolami 2013, arXiv 1312.0906) — an estimator cannot fully compensate for a
  bad parameterization.
- **Adaptive-metric HMC worth copying (post-2015):**
  - **Fisher-adaptive MALA (2305.14442):** proves the optimal MALA preconditioner is the inverse
    Fisher covariance E[∇log p ∇log pᵀ]; learns it online from the score stream — the formal
    justification for walnutpie's `Var_score` term (its diagonal approximation).
  - **Selecting the Metric in HMC (1905.11916):** warmup-efficient metric estimation + a
    selection criterion (Stan implementation) — the closest prior art to walnutpie's
    discounted-window design.
  - **Entropy-based adaptive HMC (2110.14625):** gradient-based mass-matrix adaptation
    maximizing an entropy criterion.
  - **Hierarchical RMHMC (2604.09832)** and RMHMC (0907.1100): position-dependent metrics;
    expensive, listed for completeness.
  - **High-dimensional adaptive MCMC with sparse preconditioner (2604.09286):** O(m²d) dense-ish
    preconditioning — the rank-m route of TL;DR #5.
  - **Dense/low-rank metric updates in Stan-land:** dense-E exists (O(d³) refresh, O(d²) per
    step); at d ~ 7000 it is infeasible, hence rank-m.

---

## Rejected ideas (explicit, with reasons)

1. **Pure Muon for the scalar step size.** No matrix exists; the polar factor of a scalar is its
   sign; the update collapses to smoothed-sign SA which Adam already implements with better gain
   control. Zero literature or production precedent; Muon's author and every framework route
   scalars to AdamW.
2. **Muon orthogonalization of the diagonal mass vector (viewing diag as 1×d).** Degenerates to
   RMS-normalization of the update/estimate — destroys the very per-coordinate scale information
   the metric is supposed to capture. (For *estimation* problems, normalization is harmful; for
   *optimization* of matrices, it is the point.)
3. **AdEMAMix for the step-size loop.** Slow EMA (β3 ≈ 0.9999) presumes a stationary target over
   ~5k steps; warmup's nonstationarity (metric updates) violates exactly that; with β3 ≈ β1 it
   is Adam. Keep the implementation for ablation only.
4. **AMSGrad-style non-decreasing v.** Ratcheting second moment is anti-tracking under regime
   shifts (the precise nonstationarity here).
5. **Sophia.** Needs a Hessian sketch; at 1-D it is noisy finite differences with no published
   SA use.
6. **AnyPrecision / mixed-precision optimizer states.** Solves bf16/fp16 accumulation problems
   walnutpie does not have (fp64 Eigen vectors); the apex implementation is not even shipped in
   current master.
7. **CAME.** Its raison d'être (int8 second moments, memory) is void for one scalar; its
   "confidence" modulation is a weaker AdaBelief.
8. **Schedule-Free framework wholesale.** The valuable part (averaged iterate) is a 15-line
   addition; the framework's constant-lr premise conflicts with diminishing-adaptation
   ergodicity requirements if adaptation ever continues post-warmup.
9. **Dense RMHMC-style position-dependent metrics.** Cost O(d²–d³) per step at d ~ 7000;
   known numerical pitfalls (2111.09995); hierarchical RMHMC (2604.09832) is a research path,
   not an upgrade.
10. **Replacing Welford with batch/periodic recomputation of variances.** Strictly worse
    numerics at equal memory; the discounted-Welford + shrinkage + floors path dominates.
11. **"GMS" as named in the brief.** Unidentifiable in the literature (see §1.3); do not
    pursue as a citation-backed option.
12. **"AdaHMC".** Does not exist as far as arXiv search can tell; do not design against it.

---

## Uncertainty register (be honest)

- All "benefit" estimates for loop-(a) changes are extrapolated from (i) one detailed forum
  report with formulas (Stan Discourse 5995) and (ii) SA theory — not from published ablations
  on walnutpie-class samplers. The team's own benchmarks remain the arbiter.
- The 2026 Muon-variant wave is largely preprints (GPT-2-scale evidence); treat "curvature-aware
  Muon" lessons as analogies, not evidence, for MCMC adaptation.
- No published head-to-head Adam-vs-DA for step-size adaptation was found; the claim that they
  converge to similar behaviour once clipped/averaged is theoretical reasoning (§2.1–2.2).
- Online LW/OAS under exponential discounting: natural construction, no published validation.
- Robust (Tyler/Winsorized) score variances inside HMC warmup: no published evaluation found.
- The "Moonshot momentum-view blog" could not be retrieved (site restructured); only the
  Moonlight paper's RMS-matching content was verifiable.

---

## Reproducibility — exact queries and fetches

Web-search note: the Serper-backed `websearch` skill had no API key in this session; DuckDuckGo
and Bing HTML endpoints were heavily rate-limited/degraded, so discovery ran through the arXiv
API, arxiv.org search UI, GitHub REST API, direct primary-source fetches, and the Stan Discourse
JSON API. Full query list:

**arXiv API (`export.arxiv.org/api/query`, search_query=...):**
- `ti:"Muon" AND cat:cs.LG` (20, submittedDate)
- `abs:"Newton-Schulz" AND cat:cs.LG` (15)
- `ti:"Polar Express"`; `ti:"Modular Duality"`; `ti:"Schedule-Free"` (8)
- `abs:"Lion optimizer" AND cat:cs.LG`; `ti:"Sophia" AND abs:optimizer`; `ti:"Came" AND abs:optimizer`
- `ti:"On the Convergence of Adam and Beyond"`; `abs:"variance of the adaptive learning rate" AND cat:cs.LG`
- `ti:"Muon" AND ti:"optimizer"` (15); `abs:"Muon" AND abs:"scalable" AND abs:"LLM training"`
- `ti:"MuClip"`; `abs:"Muon2" OR ti:"Muon2"`; `ti:"Symbolic Discovery of Optimization Algorithms"`
- `ti:"The Road Less Scheduled"`; `abs:"dual averaging" AND abs:"Monte Carlo"` (12, submittedDate)
- `abs:"step size adaptation" AND abs:"Hamiltonian Monte Carlo"`; `ti:"adaptive Hamiltonian Monte Carlo" OR ti:"adaptive HMC"`
- `abs:"AdaHMC" OR ti:"AdaHMC"`; `ti:"WALNUTS" OR all:"walnutpie"`
- `abs:"dual averaging" AND abs:"step size"`; `abs:"warmup" AND abs:"MCMC" AND abs:adaptation`
- `abs:"No-U-Turn Sampler" AND abs:"step size" AND abs:adapt`; `abs:"mass matrix" AND abs:"warmup"`
- `abs:"Riemannian" AND abs:"Hamiltonian Monte Carlo" AND abs:adaptive`; `abs:"low-rank" AND abs:"Hamiltonian Monte Carlo" AND abs:adaptation`
- `abs:"mass matrix" AND abs:"Hamiltonian Monte Carlo"`; `abs:"metric" AND abs:"precondition" AND abs:"Hamiltonian Monte Carlo"`
- `ti:"Riemannian" AND (ti:MALA OR abs:"manifold MALA")`; `abs:"online" AND abs:"variance estimation" AND abs:"exponentially"`
- `abs:"Welford"`; `abs:"Ledoit-Wolf" AND abs:"shrinkage"`; `abs:"Oracle Approximating Shrinkage"`
- `ti:"AdEMAMix"`; `ti:"AdaBelief"`; `ti:"Moonlight" AND abs:Muon`

**arXiv abstract pages fetched by ID (id_list/abs):** 2506.18746, 2502.16982, 2604.09967,
2602.17080, 2509.02981, 2606.08783, 2605.09552, 2606.13867, 2608.05088, 2507.11005, 2507.20534,
2409.03137, 2405.15682, 2010.07468, 2307.02047, 2305.14342, 2302.06675, 1908.03265, 2404.15253,
2502.02431, 2410.21587, 2305.14442, 2604.09286, 2604.09832, 1905.11916, 2110.14625, 1804.09898
(mismatch check), 0907.4698, 2503.11479.

**arxiv.org search UI queries:** `dual averaging step size Hamiltonian Monte Carlo`;
`step size adaptation Hamiltonian Monte Carlo`; `warmup adaptation Hamiltonian Monte Carlo`;
`mass matrix estimation Hamiltonian Monte Carlo`; `covariance adaptation MCMC preconditioning`;
`exponentially weighted online variance estimation`; `online covariance shrinkage estimation`;
`Riemannian Hamiltonian Monte Carlo adaptive metric`; `step size jitter Hamiltonian Monte Carlo randomized`;
`Hamiltonian Monte Carlo step size jitter`; `adaptive step size Metropolis-Hastings stochastic approximation`;
`AdaHMC`; `GMS greedy orthogonalization Newton-Schulz Muon`; `nanotron Muon optimizer improvements weight decay`;
`On the Convergence of Adam and Beyond`.

**GitHub API:** `/search/repositories?q=muon+optimizer`; `/users/KellerJordan/repos`;
`/repos/KellerJordan/modded-nanogpt/git/trees/master?recursive=1` (+ raw `train_gpt2.py`);
`/orgs/MoonshotAI/repos`; `/repos/huggingface/nanotron/git/trees/main?recursive=1`;
`/repos/NVIDIA/apex/...` (tags 25.04–25.09, contents/optimizers); `/repos/stan-dev/stan/git/trees/develop`;
raw `src/stan/mcmc/var_adaptation.hpp`; `/repos/tensorflow/probability` raw
`simple_step_size_adaptation.py`.

**Direct fetches:** kellerjordan.github.io/posts/muon/;
raw.githubusercontent.com/Hiroki11x/awesome-muon-optimizer/main/README.md (+ planned docs/research-landscape.md);
mc-stan.org/docs/reference-manual/mcmc.html (and hmc.html redirect);
discourse.mc-stan.org search.json: `dual averaging step size oscillation`, `stepsize adaptation collapse warmup`,
`step size adaptation divergent warmup target acceptance`, `adapt_delta step size`;
topics 5995 ("Issue with dual averaging") and 9532 ("Request for Volunteers to Test Adaptation Tweak");
moonshotai.github.io (redirect), kexue.fm / spaces.ac.cn (403), web.archive.org CDX (no snapshots);
people.ds.cam.ac.uk Finch note (offline).

**Local grounding (not web):** walnutpie repo at `/home/m0hawk/Documents/apin/stan/external/walnutpie`
— `include/walnutpie/{adam,step_optimizers,online_moments,adaptive_walnuts,config}.hpp`.

---

## Citation quick-list (canonical forms)

- Muon: Keller Jordan, blog 2024, kellerjordan.github.io/posts/muon/ **[blog]**; repo KellerJordan/Muon **[repo]**.
- Moonlight/Muon-at-scale: Liu, Su, Yao et al., "Muon is Scalable for LLM Training", arXiv:2502.16982 **[paper]**.
- Kimi K2 / MuonClip: Kimi Team, arXiv:2507.20534 **[paper]**.
- Polar Express: Amsel, Persson, Musco et al., arXiv:2505.16932, ICLR 2026 **[paper]**.
- Muon²: arXiv:2604.09967; NAMO: arXiv:2602.17080; AdaMuon: arXiv:2507.11005; AdaGO: arXiv:2509.02981;
  OptMuon: arXiv:2606.08783; MALT: arXiv:2608.05088; PowerMuon: arXiv:2606.13867; Mousse: arXiv:2603.09697;
  Scion: arXiv:2502.07529; Old Optimizer New Norm: arXiv:2409.20325; Modular Duality: arXiv:2410.21265;
  Kovalev trust region: arXiv:2503.12645; Fantastic Pretraining Optimizers: arXiv:2509.02046;
  How Much Orthogonalization: arXiv:2606.00371; Phases of Muon: arXiv:2605.09552 (all **[paper]**).
- Adam: arXiv:1412.6980; AMSGrad: arXiv:1904.09237 (ICLR'18); RAdam: arXiv:1908.03265; AdaBelief:
  arXiv:2010.07468 (NeurIPS'20); Lion: arXiv:2302.06675 (Symbolic Discovery); Sophia: arXiv:2305.14342;
  CAME: arXiv:2307.02047; AdEMAMix: arXiv:2409.03137 (NeurIPS'24); Schedule-Free: arXiv:2405.15682;
  Connections (SF/AdEMAMix/accelerated SGD): arXiv:2502.02431; adaptive-Polyak SF: arXiv:2511.07767
  (all **[paper]**).
- NUTS/DA: Hoffman & Gelman, arXiv:1111.4246 (JMLR'14); Nesterov, "Primal-dual subgradient methods
  for convex problems", 2009 **[paper]**; Stan reference manual (adaptation sections) **[manual]**;
  stan-dev/stan `var_adaptation.hpp` **[repo]**; Stan Discourse topics 5995, 9532 **[forum]**.
- WALNUTS: Bou-Rabee, Carpenter, Kleppe, Liu, arXiv:2506.18746 **[paper]**.
- ATLAS: arXiv:2410.21587; GIST: arXiv:2404.15253; PDMP: arXiv:2503.11479; entropy-adaptive HMC:
  arXiv:2110.14625; Fisher-adaptive MALA: arXiv:2305.14442; metric selection: arXiv:1905.11916;
  high-dim adaptive MCMC: arXiv:2604.09286; hierarchical RMHMC: arXiv:2604.09832; RMHMC: arXiv:0907.1100.
- Variance: Welford 1962 (Technometrics); Finch 2009 note **[blog/note, offline]**; Ledoit & Wolf
  2004 (JMA; JPM "Honey…"); OAS: arXiv:0907.4698 (IEEE TSP'10); Tyler 1987 (Ann. Statist.);
  Sun–Babu–Palomar 2014 (IEEE TSP, regularized Tyler); Betancourt & Girolami 2013, arXiv:1312.0906.
