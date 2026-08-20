# Pass 2: optimizer/estimator options beyond Adam & Muon — for walnutpie adaptation loops

**Scope:** same two loops as pass 1 (`research_optimizer_sota.md`, 2026-08-20):

- **(a) Scalar step-size loop:** theta = log(macro_step), g = target_accept − alpha_observed,
  alpha = exp(−|Δlog p_joint|) ∈ [0,1] (skewed toward 1 healthy, exactly 0 divergent; target 0.8).
  Noise multiplicative, autocorrelated, nonstationary. Current: scalar Adam on log-step
  (lr 0.05, β1 0.8, β2 0.9, eps 1e-4, decay t^-0.5) + optional mean-batching (stride 50);
  dual averaging / AdEMAMix / AdaBelief / clipping already shipped.
- **(b) Diagonal mass:** per-coordinate discounted online variance of draws AND scores,
  inv_mass = sqrt(Var_draw / Var_score), forgetting 1 − 1/(offset+t), d up to ~7000.
  Known failures: score-variance degeneracy during early drift; chains freezing at different
  posterior scales (multimodality).

**Date of research:** 2026-08-20. **Verification protocol:** every load-bearing arXiv ID below was
resolved through the arXiv API (`export.arxiv.org/api/query?id_list=...`) or the abstract page and
title-matched; journal/forum items were fetched directly. Three of the brief's own ID guesses were
wrong (see "ID corrections"). Labels: **[paper]**, **[blog]**, **[repo]**, **[forum]**, **[manual]**,
**[poster]**, **[thesis]**. Speculation is flagged **[SPECULATION]** inline.

Pass 1's conclusions (Muon structurally inapplicable; Adam-family tweaks marginal; clip/batch/
average/re-anchor are the wins) are NOT re-litigated — this report adds what lies outside those
families.

---

## Headline finding (changes the framing of loop (b) entirely)

**walnutpie's mass-matrix rule is now a theorem.** Seyboldt, Carlson & Carpenter,
"Preconditioning Hamiltonian Monte Carlo by minimizing Fisher Divergence" (PyMC Labs +
Flatiron Institute), **arXiv 2603.18845 [paper]**, proves that the diagonal matrix minimizing the
sample Fisher divergence of the preconditioned target to N(0, I) is — Theorem 2.2 — the
**geometric mean of diag(cov(draws)) and diag(cov(scores))^{-1}**, i.e. elementwise

  inv_mass = sqrt( Var_draw / Var_score )

with the location estimate the analogous combination. This is exactly walnutpie's estimator,
derived as the optimum of a principled objective rather than a heuristic blend. Their nutpie
implementation keeps **online Welford moments of draws and scores** and computes the combination
at each iteration — but instead of exponential tapering they **"chop off the tail end"**: discard
early draws wholesale at window boundaries rather than discount them. Benchmarked on 114
posteriordb models: diagonal Fisher estimate beats Stan/PyMC variance-based diagonals by a median
factor **1.3**; a **low-rank + diagonal** variant (their Algorithm 1: standardize by the diagonal,
thin-SVD the standardized draws and scores, take the jointly-spanned subspace) wins by a median
factor **4**. Consequences for walnutpie:

1. The formula should be cited as Fisher HMC Thm 2.2, no longer "a sqrt-ratio we invented".
2. The one published head-to-head of discounting vs chopping comes down **for chopping** (memoryless
   windows) — this is the delta to A/B against walnutpie's `1 − 1/(offset+t)` forgetting.
3. Their diagonal → low-rank+diagonal path is the concrete published recipe for the rank-m idea
   pass 1 listed as TL;DR #5 (previously only 2604.09286 / 2305.14442 supported it).
4. Their motivating example is directly about the (b) failure mode: two draws + their scores
   identify a Gaussian scale exactly (scores bypass the Cramér–Rao bound that pure-variance
   estimators face) — the score stream is most valuable precisely when draws are few and early,
   i.e., during distant-init drift. They do NOT specifically treat the Var_score-degeneracy
   regime; robustification remains open (§5 below).

---

## TL;DR — ranked top-5 NEW actionable ideas (beyond pass 1's list)

### 1. (b) Switch discounting to Fisher-HMC-style window chopping; cite Thm 2.2; A/B the low-rank+diagonal extension
- **Change:** keep the two Welford streams, but at each window boundary *reset* the moments
  (discard pre-window draws entirely) instead of carrying exponentially-discounted history;
  publish-facing docs cite 2603.18845 Thm 2.2 for the sqrt-ratio. Longer term: implement their
  Algorithm 1 (diagonal standardize → SVD of standardized draws ∪ scores → rank-m correction
  with a cutoff), O(md) storage.
- **Mechanism:** the published evidence says stale early draws are noise, not signal — chopping
  removes the exact bias that discounting merely attenuates, and the metric is rebuilt from
  post-drift samples only. Median 1.3× (diag) / 4× (low-rank) wall-time improvements on
  posteriordb are the direct evidence base, on NUTS-class samplers (nutpie), not a proxy task.
- **Evidence:** strong (peer-adjacent preprint by the walnutpie-adjacent team, 114-model benchmark,
  production implementation in nutpie **[repo: pymc-devs/nutpie]**).
- **C++ cost:** small (chopping = a reset call at window boundaries; already windowed);
  moderate (low-rank: thin SVD every window, Eigen suffices at d~7000 with rank ≤ ~50).
- **Risks:** pure memoryless windows lose information in slow-drift regimes where discounting
  tracks; low-rank adds a rank/cutoff knob and an eigenvector-staleness issue — see SOAP's
  refresh lesson (§1).

### 2. (a) Recast the scalar adapter as a PI controller with anti-windup; freeze the integral on divergence
- **Change:** the loop (a) dynamics *are* a 1-D controller on log h: control error e_t = δ − α_t;
  Adam's m is a low-pass-filtered (≈ integral) error term with self-normalized gain. Classical
  ODE-solver stepsize control (Gustafsson–Lundh–Söderlind BIT 28:270–287, 1988 **[paper]**;
  Söderlind, "Automatic Control and Adaptive Time-Stepping", LNCSE 41, 2003, + the CWI review
  "The Automatic Control of Numerical Integration" **[paper]**) runs exactly this loop in log-h
  space with PI/PID gains and proven closed-loop stability regions (their digital-filter
  controllers, e.g. H211b, shape noise vs tracking explicitly). Two concrete imports:
  (i) **conditional integration / anti-windup**: when α_t = 0 exactly (rejection/divergence) or
  the error is saturated beyond the clip, freeze the integral (m) update for that step so the
  accumulated state cannot wind up and crash log-step after the disturbance ends — a targeted
  fix for stepsize collapse that complements pass-1's clipping; (ii) optionally a derivative
  (D) term on the error for trend damping (the PID-optimizer insight: momentum overshoot is an
  integral-action pathology; PIDopt Wang et al., CVPR 2018 **[paper]**, journal IEEE TNNLS 2020).
- **Mechanism:** anti-windup is the standard remedy when a saturated actuator + integral action
  produce overshoot-and-collapse; the divergence episode is precisely a saturation event.
- **Evidence:** classical control theory (strong in ODE context), PIDLD (arXiv 2511.12603 **[paper]**,
  Nov 2025) shows PID control pays off inside Langevin sampling (inner dynamics, not outer
  adaptation). **No published PI/PID accept-stat controller for HMC/NUTS warmup was found** — this
  is a grounded-but-novel construction; treat as benchmark candidate, not a citation-backed swap.
- **C++ cost:** trivial (~10 lines on top of the existing Adam state).
- **Risks:** gain tuning in the MCMC context is unexplored; a D-term amplifies alpha's
  discreteness noise unless low-passed (PIDopt does exactly that).

### 3. (a) Amid's exponentiated-gradient step-size adaptation as the cheap "beyond Adam-on-log" A/B
- **Change:** Amid et al., "Step-size Adaptation Using Exponentiated Gradient Updates",
  **arXiv 2202.00145 [paper]** — the one published method that learns a *scalar global step-size
  scale* with EG (multiplicative) updates driven by gradient *alignment*, with per-coordinate
  gains handled separately. The 1-D walnutpie analogue: ε ← ε · exp(η·g_t · gate_t) where
  gate_t = sign agreement between the current error and a slow error EMA (Amid's
  average-gradient/current-gradient alignment collapsed to 1-D), instead of Adam's
  m/sqrt(v) normalization. Update size is *relative* (proportional in ε), matching the
  multiplicative noise structure; built-in log-space bounding; no v state.
- **Mechanism:** EG keeps steps proportional to the quantity being controlled (multiplicative
  noise → multiplicative correction); the alignment gate pauses updates when the error stream is
  sign-inconsistent (pure noise), which is Adam's self-normalization role done cheaper.
  Note the theory context: additive-in-log-space updates ARE multiplicative weights on ℝ₊
  (mirror descent under the log map — Raskutti/Wainwright/Yu lineage; "The Information Geometry
  of Mirror Descent", arXiv 1310.7780 **[paper]**); Hoffman–Gelman dual averaging is already in
  this family. So the real comparison is fixed-gain EG vs self-normalized Adam vs gain-adaptive
  AdaGrad-Norm (idea 4) — three normalizations of the same multiplicative update.
- **Evidence:** moderate — Amid's validation is image/text model training incl. distribution
  shift, not SA/MCMC; **no published head-to-head EG vs Adam-on-log for root finding exists**.
- **C++ cost:** trivial (~15 lines).
- **Risks:** unknown behavior at α ≡ 0 (gate sees consistent sign → unbounded multiplicative
  descent until clip); needs the idea-2 saturation guard.

### 4. (a) Closed-loop scalar learning rate (AdaGrad-Norm state) replacing the t^-0.5 schedule — now with 1-D structural support
- **Change:** lr_t = η / sqrt(Σ g²) on the scalar loop (optionally windowed), per
  Ward–Wu–Bottou "AdaGrad stepsizes: Sharp convergence over nonconvex landscapes",
  **arXiv 1806.01811 [paper]** (+ Faw/Khaled-line follow-ups 2209.14827, 2406.06398, 2604.10728).
  Pass 1 flagged OptMuon (**2606.08783**) and AdaGO (**2509.02981**); digging in confirms the
  closed-loop core is exactly AdaGrad-Norm on the direction — and at 1-D the orthogonalized
  direction degenerates to sign(g), so "OptMuon-for-scalars" = sign-SA with an AdaGrad-Norm
  scalar lr. **The 1-D analogue exists structurally, but its validation is minimization-only;
  root-finding use is [SPECULATION]** — the state is one scalar and the failure mode is a
  bounded lr, so the blast radius is small. Composes with pass-1's Polyak–Ruppert averaging.
  Heavy-tail caveat resolved in our favor: AdaGrad-Norm/Adam-Norm *without* clipping provably
  degrade under heavy-tailed gradient noise, *with* clipping they recover polylog high-probability
  bounds (arXiv 2406.04443 **[paper]**; normalization+clipping necessity: 2410.16561; momentum
  alone can suffice: 2607.08104) — keep the pass-1 clip when doing this.
- **Distance-based alternative:** D-Adaptation (**2301.07733**) / Prodigy (**2306.06101**) /
  DoG (**2302.12022**) estimate distance-to-solution to set scale; in warmup, log ε_init →
  log ε* is typically ≤ ~10 nat — a natural D prior for the ramp **[SPECULATION]**; no
  published SA/MCMC use.
- **C++ cost:** trivial (one accumulator).
- **Risks:** monotone Σg² never adapts *up* after a regime reset — re-anchor the accumulator at
  mass-matrix refreshes (same discipline as pass-1 re-anchoring).

### 5. (b) Robustify the score stream before its second moments (clip/Winsorize/Catoni), guided by the heavy-tail-clipping theory
- **Change:** in `OnlineMoments` for scores only: clip each score coordinate at a running
  robust scale (e.g., ±k·MAD or a k-σ Winsorization with k ≈ 3–5) before accumulating second
  moments; or accumulate Catoni-style robust second moments (Catoni 2012, ECP **[paper]**;
  online Catoni confidence sequences: arXiv 2208.03185; Catoni-gradient line: Prasad et al.
  **1802.06485**, survey Lugosi–Mendelson **1906.04280**). Var_score ≈ Fisher diagonal; the
  funnel/early-drift regime is exactly heavy tails + outliers, and the 2406.04443 / 2410.16561
  results say un-clipped adaptive scaling is the provably bad case under such noise.
- **Mechanism:** bounded-influence second moments stop a handful of giant scores (funnel neck,
  distant init) from blowing up Var_score and collapsing inv_mass = sqrt(Var_draw/Var_score) —
  the reported early-drift metric degeneration.
- **Evidence:** theory strong in SGD; **no published evaluation of robust score variance inside
  HMC warmup found** (honest gap; pass 1 said the same for Tyler). Complementary published
  support: (i) Tran, "Tuning diagonal scale matrices for HMC" (arXiv 2403.07495 **[paper]**)
  compares variance-based vs score-based (ISG) diagonals — score-based scaling is a live
  alternative objective; (ii) Hird & Livingstone ("Quantifying the Effectiveness of Linear
  Preconditioning in MCMC", JMLR, arXiv 2312.04898 **[paper]**) prove variance-only diagonals
  can *increase* the condition number (Result 5) — the Fisher/score side is not decoration,
  it is what keeps the diagonal defensible.
- **C++ cost:** trivial (per-coordinate clip; Catoni variant slightly more).
- **Risks:** clipping biases the metric for genuinely heavy-tailed posteriors (the estimator
  then targets a Winsorized Fisher diagonal); keep k generous and benchmark on funnel geometries.

**Runner-ups (deliberately below the cut):** cross-chain warmup with R-hat gating (§8 — the right
fix for "chains frozen at different scales" but an orchestration-level change: campfire
**[forum 12039, 12912]**, Zhang ACOP 2020 **[poster]**, Universal Warmup Path **2607.23788
[paper]**); Pathfinder-style LBFGS initialization to skip the distant-init drift phase entirely
(**2108.03782 [paper]**, shipped in CmdStan **[manual]**); autoMALA-style local stepsize
selection (**2310.16782 [paper]**, AISTATS 2024) — conceptually overlaps WALNUTS' within-orbit
dyadic adaptation, so low marginal value here.

---

## ID corrections (verified against the arXiv API — several brief/pass-1 guesses were wrong)

| Claimed | Actually | Correct ID |
|---|---|---|
| Schedule-free = "Defazio et al. 2404.19537" | graph-theory paper ("ε-equienergetic graphs") | "The Road Less Scheduled" = **2405.15682** (pass 1 had it right; the brief's alt ID is wrong) |
| "PID optimizer arXiv 2310.10973" | valleytronics (moiré valley filter) | PIDopt = Wang et al., CVPR 2018 (no arXiv); journal IEEE TNNLS 31(12), 2020; PIDAO = Nature Comms s41467-024-54451-3 (2024); AdaPID = ICASSP 2022; IAdaPID-ADG = **2605.21968** |
| Prodigy = 2309.07879 | Altschuler et al., "Acceleration by Stepsize Hedging I: Silver Stepsize" | Prodigy = **2306.06101** |
| Fisher-adaptive MALA author "Wallach/?" (dispatch-level) | single author | **Michalis K. Titsias**, 2305.14442 |
| "1-bit Shampoo" | no such arXiv title (0 hits) | quantized line is 4-bit: **2405.18144**, **2412.10663** |
| "RowAdagrad" | no arXiv paper found (0 hits) | exists only as community/implementation folklore (modded-nanogpt ecosystem, per-row mean-of-squared-gradients AdaGrad) — cite as [repo/community], no primary source |

---

## Method table

"1D" = scalar loop (a); "diag" = loop (b). Ratings are for *this* use case.

| Name | Year | One-line core | 1D? | diag? | Noise rob. | Nonstat. rob. | Key citation |
|---|---|---|---|---|---|---|---|
| **Fisher HMC (diag)** | 2026 | Fisher-divergence-optimal metric = geom. mean of var(draws), var(scores)^-1 | n/a | **yes (is walnutpie's rule)** | med (needs robust scores) | med (chop windows) | 2603.18845 [paper] |
| Fisher HMC (low-rank+diag) | 2026 | + joint SVD subspace of standardized draws/scores | n/a | yes (rank-m path) | med | med | 2603.18845 |
| Window chopping (memoryless Welford) | 2026 | discard early draws at windows, no taper | n/a | yes | med | **high** | 2603.18845 §"discount" |
| ISG scaling (mean sq. score) | 2024 | diagonal ∝ sqrt(E[score²]) alternative objective | n/a | yes (alternative) | med | med | 2403.07495 |
| Hird–Livingstone preconditioning theory | 2023 | variance-only diagonals can worsen κ (Result 5) | n/a | caution/justification | — | — | 2312.04898 (JMLR) |
| Universal Warmup Path | 2026 | multi-chain controller, auto diag vs low-rank selection | n/a | yes (multi-chain) | high | high | 2607.23788 |
| autoMALA | 2024 | per-iteration stepsize by doubling/halving + reversibility check | partial (local) | by-product diag var | high | **high** (local) | 2310.16782 |
| Pathfinder init | 2021 | LBFGS runs → typical set + inverse-Hessian metric | no | yes (init replacement) | high | high | 2108.03782 |
| campfire / cross-chain warmup | 2019–23 | R-hat/ESS-gated warmup, pooled metric across chains | no | yes | high (pooling) | high | [forum 12039, 12912]; ACOP 2020 [poster] |
| Amid EG step-size adaptation | 2022 | scalar global lr via exponentiated updates + alignment gate | **yes** | no | med-high | high (mult.) | 2202.00145 |
| EG± / multiplicative weights (1-D) | 1997 | theta ← theta·exp(η g) — fixed-gain multiplicative SA | yes (=fixed-gain pole) | no | low-med (fixed gain) | med | Kivinen & Warmuth 1997 [paper]; 1310.7780 |
| Dual averaging on log ε | 2014 | additive-in-log = multiplicative SA + PR average | yes (shipped) | n/a | med | med (re-anchor) | 1111.4246 |
| PI/PID stepsize control (ODE solvers) | 1988–2003 | PI in log h, proven stability, filter design | **yes (frame)** | no | **high (designed for)** | high | Gustafsson–Lundh–Söderlind BIT 1988; Söderlind 2003 |
| Anti-windup / conditional integration | classical | freeze integral when actuator saturated | **yes** | no | high at α=0 | high | Söderlind 2003 [paper] |
| PIDopt / AdaPID / PIDAO | 2018–24 | D-term on gradient diff cures momentum overshoot | weak (vector opt) | no | med | med | CVPR'18; ICASSP'22; NatComms'24 |
| PID-controlled Langevin (PIDLD) | 2025 | PID on energy-gradient feedback inside dynamics | no (inner loop) | no | high | med | 2511.12603 |
| AdaGrad-Norm closed-loop lr | 2018+ | lr = η/√Σg², hyperparameter-free decay | **yes (concept)** | n/a | high w/ clip | med (reset Σ) | 1806.01811; 2209.14827 |
| Clipping under heavy tails | 2024 | AdaGrad/Adam-Norm need clip for polylog HP bounds | **yes (keep clip)** | for scores too | **high** | high | 2406.04443; 2410.16561 |
| D-Adaptation / Prodigy / DoG | 2023 | estimate distance-to-solution for scale | speculative | no | high | med | 2301.07733; 2306.06101; 2302.12022 |
| Schedule-free / primal averaging | 2024 | z/x iterate averaging, no schedule | averaging only | n/a | high | med | 2405.15682; 2511.07767 |
| Kalman view of NG / Kalman GD | 2017–18 | EKF ≡ online natural gradient; filter the gradient | concept | no | high (filter) | high (tracking) | 1703.00209; 1810.12273 |
| Particle-filtering optimization | 2018 | PF-based stochastic optimizers, no lr schedule | concept | no | high | med | 1807.08534 |
| BOCPD | 2007 | online run-length posterior for regime shifts | concept (unused) | n/a | high | **high** | 0710.3742 |
| Median-of-means / Huber gradients | 2018–19 | robust gradient estimation under contamination | yes (batched α) | n/a | **high** | med | 1802.06485; 1906.04280 |
| Catoni robust mean/variance | 2012+ | polynomial-tail deviation control, sub-Gaussian-like | yes | for scores | **high** | med | Catoni 2012 ECP; 2208.03185; 2309.03818 |
| SPSA / Kiefer–Wolfowitz | 1952/1992 | finite-difference SA | **rejected** (direct signal exists) | n/a | low | low | Spall 1992 IEEE TAC |
| Hypergradient descent | 2017 | lr updated by gradient of loss w.r.t. lr | concept | no | med | med | 1703.04782 |
| Shampoo | 2018 | Kronecker-factored full-matrix AdaGrad | no (needs 2-D) | degenerate→AdaGrad | high | med | 1802.09568 |
| Distributed Shampoo (Google) | 2020 | grafting, one-sided preconditioners, CPU offload | no | pattern only | high | med (graft fixes scale) | 2002.09018 |
| Distributed Shampoo (Meta) | 2023 | delayed preconditioner start, merge dims | no | pattern only | high | med | 2309.06497 |
| SOAP | 2024 | Adam in Shampoo eigenbasis; refresh Q every 10 steps | 1-D falls back to Adam | **pattern for rank-m refresh** | high | **high (continuous eigenvalues)** | 2409.11321 |
| Purifying Shampoo | 2025 | decouple eigenvalue/eigenbasis updates; adaptive refresh | no | pattern (adaptive refresh) | high | high | 2506.03595 |
| New Perspective on Shampoo | 2024 | Shampoo² = one power step toward optimal Kronecker cov | no | theory for online metric view | — | — | 2406.17748 |
| DASH | 2026 | batched block preconditioning, fast inverse roots | no | no | high | med | 2602.02016 |
| 4-bit Shampoo | 2024 | quantized preconditioner states | no | no (fp64 irrelevant) | — | — | 2405.18144; 2412.10663 |
| SM3 cover-sharing | 2019 | share one scale across coordinate covers | n/a | **possible variance reduction** | high | med | 1901.11150 |
| Adafactor | 2018 | factored second moments | no | no | med | med | 1804.04235 |
| K-FAC | 2015 | Kronecker-factored natural gradient | no | no | med | med | 1503.05671 |
| EKFAC | 2018 | eigenvalue-corrected K-FAC | no | no | med | med | 1806.03884 |
| Inverse-free structured KFAC | 2023 | no matrix inversion, memory-efficient | no | only if rank-m built | med | med | 2312.05705 |
| Online inverse-Fisher (iterative) | 2023 | build F^{-1} by fixed-point updates, no inversion | no | rank-m analog | med | med | 2312.09633 |
| Fisher-adaptive MALA | 2023 | optimal preconditioner = inverse score covariance, online | n/a | dense theory for (b) | high | med | 2305.14442 |
| Position-dependent MALA | 2013/2021 | location-dependent preconditioning | no | theory | med | high (local) | 1309.2983; 2108.12662 |
| SGRLD (Riemannian SGLD) | 2013 | Riemannian Langevin on simplex | no | no | med | med | Patterson & Teh, NeurIPS 2013 [paper] |
| oLBFGS / stochastic L-BFGS | 2005/2016 | curvature from iterate/gradient diffs, damping | weak at 1-D | no | med | med | Schraudolph & Yu ICML'05; 1508.02087; Byrd et al. 2016 |
| AdaHessian (Hutchinson diag) | 2020 | Rademacher-probe Hessian diag, RMS-smoothed, blocked | n/a | alternative to Var_score | med | med | 2006.00719 |
| Rank-normalized R-hat | 2019 | convergence diagnostic foundation | n/a | gating layer | — | — | 1903.08008 |
| ASIS interweaving | 2011 | resample scales in CP then NCP within each sweep | no | explains scale lock-in | — | — | Yu & Meng, JCGS 2011 [paper] |

---

## 1. Shampoo family

- **Shampoo** (Gupta, Koren, Singer, ICML 2018; arXiv 1802.09568 **[paper]**): block-diag
  Kronecker-factored preconditioners from gradient-moment matrices L=Σggᵀ side statistics;
  inverse p-th roots.
- **Scalable/Distributed Shampoo** (Anil, Gupta, Koren, Regan, Singer, arXiv 2002.09018 **[paper]**;
  JAX impl in google-research **[repo]**): grafting (SGD/Adam/AdaGrad) to fix per-layer scale;
  **one-sided preconditioners for embedding/softmax layers** (their Lemma 1: preconditioning with
  only L or R is valid) — the closest thing in the family to a rank-1 view; block size 128;
  delayed start (`start_preconditioning_step=5` before which a **diagonal** update runs).
- **Meta Distributed Shampoo** (Shi et al., arXiv 2309.06497 **[paper]**; facebookresearch/optimizers
  **[repo]**): same algorithm, PyTorch/DTensor engineering.
- **SOAP** (Vyas et al., arXiv 2409.11321, ICLR'25, **[paper]**; nikhilvyas/SOAP **[repo]**):
  runs Adam's moment updates **in the eigenbasis** of the Shampoo preconditioner. Mechanism
  (verified from reference implementation): full `eigh` once at start; thereafter every
  `precondition_frequency=10` steps the eigenbasis is refreshed by **one power iteration + QR**
  of the accumulated statistic against the current basis, with the Adam second-moment vector
  permuted into the new ordering — continuity of the *eigenvalues* (Adam-style, updated every
  step) vs staleness of the *eigenvectors* (updated every k). Lesson for a walnutpie rank-m
  metric: keep per-direction scales fresh continuously, refresh directions rarely and cheaply
  (warm-started QR), and permute state at refresh. 1-D parameters fall back to plain Adam
  (confirmed in both the reference impl and the optax contrib PR #1692) — same routing logic as
  Muon.
- **Purifying Shampoo** (arXiv 2506.03595 **[paper]**): decomposes the preconditioner into
  eigenvalues + eigenbasis; shows grafting mainly patches stale/mis-scaled eigenvalues; proposes
  an **adaptive eigenbasis-refresh criterion** (terminate a warm-started QR when converged) — a
  principled replacement for a fixed refresh cadence, directly transferable to idea-1's rank-m
  variant.
- **A New Perspective on Shampoo's Preconditioner** (arXiv 2406.17748 **[paper]**): Shampoo's
  statistic² equals one power-iteration step toward the optimal Kronecker approximation of the
  Gauss-Newton Hessian / full-matrix-AdaGrad gradient covariance. **This is the honest content of
  "Shampoo as online metric learning":** the preconditioner IS an online estimate of gradient
  covariance (a metric), and the approximation theory now exists. Nobody phrases it as "metric
  learning for MCMC" — but that exact object is what Fisher HMC (2603.18845) estimates on the
  MCMC side. The two literatures converge on score outer products; that is the punchline.
- **Diagonal-only usage:** (i) diagonal grafting warm-start (2002.09018); (ii) `adagrad_dims` /
  `merge_small_dims` config (google-research impl) routing small/odd dims to diagonal AdaGrad;
  (iii) SOAP's 1-D→Adam fallback. Structurally, Shampoo on a 1×d or d×1 block collapses to
  row/col scalar statistics → plain AdaGrad. **No diagonal-only Shampoo variant with published
  benefits exists; the family's diagonal is AdaGrad.**
- **Quantized Shampoo:** "1-bit Shampoo" not found; the real line is 4-bit (2405.18144
  "4-bit Shampoo for Memory-Efficient Network Training"; 2412.10663 "Memory-Efficient 4-bit
  Preconditioned Stochastic Optimization"). Memory motivation — irrelevant to fp64 C++.
- **DASH** (2602.02016 **[paper]**, 2026): systems acceleration (batched blocks, Newton-DB /
  Chebyshev inverse roots). Not applicable.
- **Verdict:** nothing in the family replaces either walnutpie loop; the transferable assets are
  (a) SOAP's eigenvalue/eigenbasis freshness split and adaptive refresh (Purifying), (b) the
  2406.17748 metric-estimation theory, (c) one-sided preconditioning as a rank-1 pattern — all
  folded into idea 1 / pass-1 TL;DR #5.

## 2. Natural gradient / Fisher / K-FAC

- **K-FAC** (Martens & Grosse, arXiv 1503.05671 **[paper]**): Kronecker-factored Fisher; explicit
  that diagonal/bias parameters get no factor (routed to standard treatment). **EKFAC**
  (George, arXiv 1806.03884 **[paper]**): eigenvalue correction. **No "diagonal K-FAC"
  publication found**; at diagonal structure K-FAC degenerates to per-coordinate curvature
  scaling (= AdaGrad-like), as expected.
- **Practical 2023+ variants:** Structured Inverse-Free natural gradient / memory-efficient KFAC
  (Lin et al., **2312.05705 [paper]**) — avoids matrix inversion entirely; Godichon-Baggioni &
  Nguyen natural-gradient VB without forming/inverting F (**2312.09633 [paper]**) — iterative
  F^{-1} accumulation; PipeFisher-2 (**2211.14133**) pipelines K-FAC for LLMs. These matter to
  walnutpie only if the rank-m metric is built (idea 1); then the inverse-free fixed-point style
  is the numerically safe pattern in C++.
- **Natural gradient ≡ Kalman filtering** (Ollivier, arXiv 1703.00209 **[paper]**): EKF on a
  fixed parameter = online stochastic natural gradient on log-likelihood; gives principled
  interpretations of lr, Fisher init and damping as filter quantities. For loop (a) this is the
  theoretical license for "log-ε as a latent state tracked by a filter" — but a 1-D filter's
  gain adaptation is functionally AdaGrad-Norm (idea 4), so we adopt the simpler form.
  **Kalman Gradient Descent** (Vuckovic et al., arXiv 1810.12273 **[paper]**) filters gradient
  estimates with state-dependent noise; particle-filter optimizers (arXiv 1807.08534) drop the
  lr schedule entirely. No published Kalman-filter stepsize adapter for MCMC found.
- **Discounted online Fisher-diagonal estimation:** no standalone publication found — but
  Fisher HMC's Welford-score-stream + chopping (2603.18845) **is** the published instance of
  online Fisher-diagonal estimation for HMC metrics, and Fisher-adaptive MALA (Titsias,
  arXiv 2305.14442 **[paper]**, single author — corrects pass-1 dispatch-level uncertainty)
  proves the optimal MALA preconditioner is the inverse Fisher covariance E[ssᵀ] built online
  from the score history, with a score-*increment* trick (using gradient differences) that makes
  the online estimate cheaper/stabler — worth reading in full before implementing idea 1's
  rank-m variant. Position-dependent MALA theory: Xifara-line (1309.2983) and GLMM convergence
  (2108.12662). Riemannian SGLD: Patterson & Teh, NeurIPS 2013 (no arXiv) **[paper]**.
- **Anything combining posterior covariance AND Fisher as a metric?** Yes — exactly two places:
  Fisher HMC Thm 2.2 (the geometric mean, = walnutpie) and Tran's ISG comparison (2403.07495,
  which benchmarks pure-score against pure-variance diagonals and notes when each wins).
  No third combination rule found.

## 3. Stochastic quasi-Newton

- **oLBFGS** (Schraudolph & Yu, ICML 2005 **[paper]**, not on arXiv) and **linearly-convergent
  stochastic L-BFGS** (Moritz, Bubeck, Dick, arXiv 1508.02087 **[paper]**); **SQN** (Byrd,
  Hansen, Nocedal, Shi, Math. Prog. Comp. 2016 **[paper]**, no arXiv). All are d-dimensional
  minimizers with curvature pairs from iterate/gradient differences; at 1-D the secant
  approximation needs d(α)/d(log ε) — **no published estimator of the acceptance-statistic
  derivative w.r.t. log-stepsize exists** (searched explicitly; "no literature found").
  [SPECULATION] A crude gain model is available from classical scaling limits: α(ε) has the
  2Φ(−c·ε·d^{1/4})-type shape under unit metric (Roberts–Rosenthal-style asymptotics), whose
  derivative at the δ=0.8 operating point could seed a Newton gain or an Adam-lr prior —
  unpublished, cheap, benchmark-only.
- **Online diagonal Hessian:** AdaHessian (Yao et al., arXiv 2006.00719 **[paper]**) —
  Hutchinson probe (E[v⊙Hv]=diag H), RMS-smoothed across steps, block-diagonal averaging;
  the RMS-EMA-over-Hutchinson-noise pattern is the transferable trick if walnutpie ever
  estimates curvature directly. Recent theory: Hessian clipping under heavy tails
  (2510.10690). Relevance to (b): Var_score is already an unbiased-ish Fisher-diagonal
  surrogate computable for free — Hutchinson probing buys nothing here since scores are
  observed exactly.
- **Pathfinder** (Zhang et al., arXiv 2108.03782 **[paper]**; in CmdStan **[manual]**):
  parallel LBFGS paths, local inverse-Hessian covariances, KL-selected normal approximation;
  1–2 orders of magnitude fewer density/gradient evals than short warmup chains. The (b)
  play: run Pathfinder-style exploration first, initialize Welford moments (or the whole
  chain) from its draws — bypasses the distant-init drift where Var_score degenerates.

## 4. Schedule-free and 1-D root finding

- "The Road Less Scheduled" (Defazio et al., **2405.15682 [paper]**; the brief's 2404.19537 is a
  graph paper — see corrections) and Adaptive-Polyak schedule-free (**2511.07767 [paper]**).
  **Published schedule-free theory covers convex/nonconvex *minimization* only; no schedule-free
  or primal-averaging result for stochastic *root finding* (E[g(θ)]=0 with sign-flipping mean)
  was found** — the nearest formal bridge is minimizing (E[α]−δ)², a different, noisier
  objective. Verdict stands from pass 1: adopt the averaging (PR/DA x̄), skip the framework;
  constant-lr adaptation post-warmup would also violate diminishing-adaptation ergodicity.
- Polyak–Ruppert *tracking* of a drifting target: classical SA covers time-varying roots with
  non-summable steps (the t^{-0.5} regime); no modern 2023+ tracking analysis of primal
  averaging specific to our setting found beyond what pass 1 cited.

## 5. Control-theoretic step size, filtering, changepoints

- **The classical literature exists — in ODE solvers.** Gustafsson, Lundh & Söderlind,
  "A PI stepsize control for the numerical solution of ODEs", BIT 28:270–287, 1988;
  Gustafsson ACM TOMS 17/20 (1991/1994); Söderlind, "Automatic Control and Adaptive
  Time-Stepping" (Springer LNCSE 41, 2003) and the CWI review "The Automatic Control of
  Numerical Integration" (all **[paper]**): stepsize control **in log h** as PI/PID on
  log(error/tol), with closed-loop stability analysis, digital-filter controller designs
  (H211b family) trading noise suppression vs tracking, and anti-windup handling for rejected
  steps. This is the theoretical home of idea 2: same loop structure, one field over.
- **PID in ML proper:** PIDopt (Wang et al., CVPR 2018; IEEE TNNLS 2020 **[paper]**) — SGD-momentum
  = PI controller, D-term fixes overshoot; AdaPID (ICASSP 2022), IAdaPID-ADG (**2605.21968**),
  PIDAO (Nature Communications 2024, s41467-024-54451-3) with Lyapunov convergence analysis.
  All are parameter-vector optimizers; for 1-D root finding they reduce to momentum variants —
  the *view* is adopted (idea 2), the named optimizers are not.
- **PID in sampling:** **PIDLD** (arXiv 2511.12603 **[paper]**, Nov 2025): PID control on energy-
  gradient feedback inside Langevin dynamics for generative sampling — inner-loop dynamics,
  not accept-stat adaptation, but it establishes the control-theoretic framing is productive
  in samplers. **A published PID/PI accept-statistic stepsize controller for MCMC/HMC warmup
  does not exist as far as arXiv + web search show** — idea 2 is novel-but-grounded.
- **Kalman/filtering of log-eps as latent state:** theoretical license via Ollivier (1703.00209,
  NG=Kalman) and Kalman GD (1810.12273); no published MCMC-stepsize application. Rejected as
  redundant machinery (a 1-D adaptive-gain filter ≈ AdaGrad-Norm + EMA, which we already have).
- **BOCPD** (Adams & MacKay, arXiv 0710.3742 **[paper]**): the canonical online regime-shift
  detector; **no published application to MCMC accept-stat streams found** (adaptive-MCMC +
  changepoint literature is about sampling changepoint *models*, e.g. EJS 2018, not detecting
  warmup regime shifts). Rejected: O(T²) run-length machinery where windowed R-hat gating
  (campfire, Universal Warmup) already solves the decision problem in the published ecosystem.
- **Hypergradient lr adaptation** (Baydin et al., arXiv 1703.04782 **[paper]**) and the
  IDBD/delta-bar-delta lineage (Schraudolph 1999; Jacobs 1987, no arXiv): update lr by the
  gradient of the objective w.r.t. lr. For loop (a) the nested-noise hypergradient is dominated
  by the directly observed error signal; rejected in favor of ideas 3/4.

## 6. Robust stochastic approximation for loop (a)

- **Heavy-tailed noise theory is directly on point:** alpha ∈ [0,1] with point mass at 0 is
  bounded but effectively heavy-tailed in log terms; Gorbunov-line results: unclipped
  AdaGrad-Norm/Adam-Norm have provably bad high-probability behavior under heavy-tailed noise,
  clipping restores polylog bounds (**2406.04443 [paper]**); gradient normalization+clipping is
  *necessary* in a precise sense (2410.16561); momentum alone can suffice without clipping
  (2607.08104, 2026). Net: walnutpie's shipped clipping has 2024+ theory behind it — keep it
  in every variant tested here.
- **Median-of-means / Huberized gradients** (Prasad et al., **1802.06485 [paper]**; survey
  Lugosi & Mendelson, **1906.04280**): batch the α's (already possible via mean-batching,
  stride 50) and, in bad regimes, take the coordinate-wise median of batch means instead of the
  mean — a drop-in robustifier for the batched adapter. Cheap; validated in contamination
  settings, not MCMC.
- **Catoni estimators** (Catoni 2012 ECP **[paper]**; infinite-variance confidence sequences
  2208.03185; risk-minimization without variance 2309.03818; matrix Catoni 2506.03074):
  the principled Winsorization — deviation-optimal influence function. Online per-step use for
  the scalar α stream is plausible but unpublished in MCMC **[gap]**.
- **SPSA / Kiefer–Wolfowitz** (Kiefer–Wolfowitz 1952; Spall, IEEE TAC 1992 **[paper]**):
  finite-difference SA. Our gradient (δ−α) is *directly observed* — finite differences add
  variance and 2× cost for nothing; the O(k^{-1/3}) KW rate is strictly worse than Robbins–Monro.
  Rejected, textbook-level.
- **AdaGrad-Norm deep-dive (per brief Q6):** Ward–Wu–Bottou (1806.01811) — AdaGrad-Norm is
  robust to *all* hyperparameters, O(log N/√N) nonconvex; 2209.14827/2604.10728 sharpen it.
  OptMuon (2606.08783) and AdaGO (2509.02981) both replace the *scheduled* magnitude with a
  Σg²-normalized scalar — i.e., the closed-loop lr idea's 1-D analogue is literally
  lr_t ∝ 1/sqrt(Σ_t (δ−α)²), with re-anchoring at metric refreshes (idea 4). Published
  validation: minimization only; root-finding transfer is [SPECULATION] but structurally
  identical and cheap to benchmark.

## 7. Multiplicative / exponentiated-parameter SA (brief Q7)

- **The direct hit: Amid, Anil & Wu, "Step-size Adaptation Using Exponentiated Gradient
  Updates" (arXiv 2202.00145 [paper], 2022).** Learns a *global scalar step-size scale* with EG
  updates keyed to gradient-alignment, plus per-coordinate gains — published, at scale, exactly
  the "1-D EG" the brief asked about. This is idea 3.
- Theory context: additive updates on log θ = mirror descent with log map = multiplicative
  weights on ℝ₊ (Information Geometry of Mirror Descent, arXiv 1310.7780 **[paper]**);
  Hoffman–Gelman DA is the SA instance (averaged multiplicative updates). EG± for signed
  univariate parameters (Kivinen & Warmuth 1997 **[paper]**; Arora–Hazan–Kale survey
  **[paper]**) exists, but **no published EG±-vs-Adam-on-log comparison for stochastic root
  finding was found** — gap, flagged honestly. Design-space framing (useful for the benchmark
  matrix): fixed multiplicative gain (EG/TFP-simple), self-normalized gain (Adam-on-log,
  walnutpie current), alignment-gated gain (Amid), accumulated-closed-loop gain (AdaGrad-Norm).

## 8. Loop (b): shrinkage, interweaving, cross-chain, diagnostics, init

- **Online shrinkage under discounting:** still no published streaming/discounted LW-OAS
  scheme (re-confirmed: searches for online covariance shrinkage return portfolio/regression
  applications only). Fisher HMC's regularization is structural (penalized divergence — the
  diagonal variant penalizes Σ(scale + 1/scale), self-normalizing) rather than LW-style
  shrinkage-to-target; pass-1's Stan `n/(n+5)` rule with Kish n_eff remains the pragmatic
  answer.
- **Robust scale under funnels:** idea 5 (Winsorized/Clipped/Catoni second moments). Tyler
  remains the dense-only option (pass 1). Tran's ISG (2403.07495) is the published
  score-diagonal comparator.
- **ASIS (Yu & Meng, JCGS 20(3), 2011 [paper]; verified):** interweaving centered/noncentered
  re-draws of scale parameters within each sweep; SV-model instantiations (Kastner &
  Frühwirth-Schnatter, CSDA 2014; factor-SV shallow/deep interweaving 2016). **No published
  ASIS-style update during HMC/NUTS warmup found** (arXiv: zero hits for
  interweaving+Hamiltonian). Forum-level existence proof: the Stan Discourse "incremental and
  adaptive parallel warm-up" thread (23617) — a participant reports a custom warmup that
  "continuously adapts the centeredness of the parameterization" with large gains, explicitly
  not ready for release **[forum]**. So the brief's hypothesis (scale lock-in is CP/NCP
  discordance; interweaving during warmup would fix it) is *consistent with ASIS theory and
  one unpublished report*, and is open research — walnutpie could be first, but it is a
  research project, not a citation-backed change (also note: walnutpie is a sampler, not a
  modeling language — it sees the unconstrained space, so CP/NCP interweaving is upstream of
  it unless the metric layer emulates it via the Fisher/score streams).
- **Cross-chain warmup:** campfire (Stan Forums topics 12039, 12912 **[forum]**; R package
  [repo]) — fixed 100-draw windows, pooled-metric recompute from all chains, R-hat<1.05 +
  ESS>50 gating, reported "20× faster warmup" on friendly models and automatic diag/low-rank/
  dense metric selection (via 1905.11916's criterion); Torsten cross-chain warmup (Zhang &
  Gillespie, ACOP 2020 **[poster]**; 2021 poster) — ~15–20% wall-time gains on PK/PD models
  with ESS-gated transitions. Peer-reviewed arXiv version: **Universal Warmup Path**
  (Lao et al., arXiv 2607.23788 **[paper]**, Jul 2026) — multi-chain controller, starts
  diagonal, selects diagonal vs low-rank+diagonal at dimension-derived window endpoints with
  rank chosen under sample-support constraints. This is the published design walnutpie should
  track for the "chains frozen at different scales" failure: pooled metrics + between-chain
  R-hat gating detect scale disagreement, which per-chain adaptation cannot see.
- **Warmup diagnostics + reinit:** rank-normalized/folded/localized R-hat (Vehtari et al.,
  arXiv 1903.08008 **[paper]**) is the foundation; local R-hat exists in tooling (posterior
  package) rather than as a standalone paper. **No published systematic reinit policy on
  warmup-failure detection was found** — ecosystem practice is Pathfinder-style init or
  longer warmup. walnutpie-relevant policy: monitor per-window max-R-hat across chains;
  on failure, restart the offending chain from the pooled best draw (campfire-style) rather
  than continue adapting a frozen chain.
- **Pathfinder as init replacement:** see §3 — the cheapest published attack on the distant-init
  metric collapse; CmdStan integration is production evidence.

---

## Rejected ideas (with reasons — ruthless)

1. **Shampoo/SOAP/DASH for either loop.** No 2-D structure exists; diagonal-only Shampoo
   degenerates to AdaGrad (routing confirmed in SOAP impl + optax PR: 1-D → Adam). Transferable
   patterns only (eigenvalue/eigenbasis freshness split; adaptive refresh; one-sided factors).
2. **4-bit / quantized preconditioners (2405.18144, 2412.10663).** Memory-compression solutions
   for GPU training; walnutpie is fp64 C++ with O(d) states.
3. **Sketchy AdaGrad/Shampoo (2302.03764), SM3 as optimizer, Adafactor.** Same memory-
   motivation mismatch. (SM3's *cover-sharing* survives as a variance-reduction idea for the
   metric, noted in table, not adopted without evidence.)
4. **K-FAC / EKFAC / inverse-free KFAC as the metric.** Needs layer Kronecker structure; a
   posterior has none. Only the rank-m extension (idea 1) makes inverse-free accumulation
   (2312.05705, 2312.09633) relevant.
5. **oLBFGS / stochastic L-BFGS as the loop-(a) optimizer.** 1-D secant needs d(α)/d(log ε);
   no published estimator; cross-trajectory finite differences too noisy. (LBFGS survives
   only as Pathfinder init for (b).)
6. **SPSA / Kiefer–Wolfowitz for (a).** The gradient is directly observed; finite differences
   strictly add variance and cost.
7. **BOCPD on the accept stream.** Correct tool class, no MCMC precedent, O(T²) machinery;
   windowed R-hat gating solves the decision cheaper (published ecosystem).
8. **Kalman filter for log-eps.** Ollivier's NG=Kalman equivalence makes a 1-D filter
   ≈ adaptive-gain natural-gradient SA ≈ AdaGrad-Norm + EMA — new machinery, same behavior.
9. **Hypergradient descent / IDBD / delta-bar-delta for (a).** Hypergradient of a root-finding
   objective is nested noise; direct error dominates. Rejected for ideas 3/4.
10. **PIDopt / AdaPID / PIDAO / IAdaPID-ADG as drop-in optimizers.** Parameter-vector methods
    that reduce to momentum variants at 1-D; only the control-theoretic *view* (idea 2) and the
    D-term-low-passed pattern transfer.
11. **PIDLD (2511.12603) mechanics.** It modifies inner Langevin dynamics, not outer
    adaptation; walnutpie's analogue (within-orbit dyadic adaptation) already exists by design.
12. **EG± over a discretized stepsize grid (hedge over actions).** Unpublished, over-parameterized;
    continuous EG (Amid) dominates for a positive scalar.
13. **Schedule-free framework for loop (a).** No SA-root-finding theory; averaging already
    adopted (pass 1); constant-lr post-warmup adaptation would break diminishing adaptation.
14. **"1-bit Shampoo" and "RowAdagrad" as citable methods.** Neither exists on arXiv (0 hits);
    RowAdagrad is community folklore only — do not cite, do not design against.
15. **ASIS implementation inside walnutpie now.** No published warmup-ASIS; walnutpie operates
    on the unconstrained space where CP/NCP is invisible. Research-project tier, not an
    actionable change.

---

## "No literature found" register (explicit)

- Diagonal-only Shampoo variant with published benefit (only degenerate-to-AdaGrad routings).
- Literature phrasing Shampoo as *online metric learning for MCMC* (the object exists; the
  phrasing/bridge does not — Fisher HMC is the de-facto bridge).
- "Diagonal K-FAC".
- Estimator of d(acceptance)/d(log stepsize) for 1-D Newton root-finding on MCMC/SGLD stepsize.
- Schedule-free / primal-averaging theory for stochastic *root finding*.
- Published PI/PID accept-statistic stepsize controller for HMC/NUTS warmup (ODE-solver PID
  control is the classical neighbor; PIDLD is inner-loop).
- Kalman-filter learning-rate/stepsize tracker in MCMC.
- BOCPD applied to MCMC acceptance-probability streams.
- Online/streaming Ledoit–Wolf/OAS shrinkage under exponential discounting.
- Robust (Winsorized/Catoni/Tyler) score variance *inside HMC warmup* evaluation.
- Published head-to-head EG vs Adam-on-log-parameter stochastic approximation.
- ASIS-style interweaving during HMC warmup (forum-level only).
- Peer-reviewed systematic reinit-on-warmup-failure policy.

---

## Speculation register

- OptMuon/AdaGO closed-loop lr → 1-D root finding (idea 4): structurally identical to published
  minimization results; root-finding validation absent.
- Söderlind PI + anti-windup for the accept-stat loop (idea 2): grounded in the ODE-solver
  analogue; MCMC instantiation unpublished.
- Scaling-limit-calibrated Newton gain for log-step (2Φ-shape α(ε)): accepted asymptotics,
  unpublished as an adapter.
- Amid alignment-gate behavior at α ≡ 0 saturation: expected safe with idea-2 guard; untested.
- D-Adaptation-style distance prior on log-step range for warmup ramps: no published use.
- SM3 cover-sharing as metric variance reduction: idea only, no evaluation anywhere.

---

## Reproducibility — searches and fetches (2026-08-20)

**arXiv API (`https://export.arxiv.org/api/query`; note: http:// 301-redirects, must use https):**

- id_list verification batches (title+author matched): 1802.09568, 2409.11321, 2006.00719,
  1603.05643, 2404.19537(→mismatch found), 2405.15682, 2511.07767, 2301.07733, 2302.12022,
  2606.08783, 2509.02981, 1503.05671, 1806.03884, 2305.14442, 0907.1100, 0710.3742, 1903.08008,
  2108.03782, 1703.04782, 1805.08528(→mismatch), 1708.06228(→mismatch), 2309.07879(→mismatch),
  1304.5290(→mismatch), 2110.14149(→mismatch), 2310.10973(→mismatch), 2202.00145, 2607.23788,
  2511.12603, 1810.12273, 1508.02087, 1806.01811, 1802.06485, 2309.06497, 2602.02016,
  2406.17748, 2506.03595, 2306.06101, 1807.08534, 1505.06562, 2003.00478, 2110.11576,
  2309.03818, 2312.09633, 2312.05705, 2504.18911, 2505.07384, 2506.03074, 2310.16782,
  1309.2983, 2108.12662, 2403.07495, 2208.03185, 2312.04898, 1901.11150, 1804.04235,
  2510.25315, 2605.21968, 2310.00016, 2101.11075, 2209.14827, 2406.04443, 2410.16561,
  2607.08104, 2510.10690, 2405.18144, 2412.10663, 2211.14133, 2603.18845.
- search_query: ti:"Shampoo"; "Distributed Shampoo"; abs:"1-bit Shampoo"; all:"RowAdagrad";
  abs:"PID"+step size; PID+Langevin; feedback control+HMC; Kalman filter+learning rate;
  Kalman+stochastic approximation; exponentiated gradient+stochastic; interweaving+Hamiltonian;
  ti:"Prodigy"; AdaGrad+norm; Hessian diagonal+estimate; all:"Catoni"; median-of-means+
  heavy-tailed; ti:"PID"+optimizer; change point+MCMC+adaptive; abs:"cross-chain"; online
  covariance shrinkage; parallel tempering+temperature+stochastic approximation; ti:"Pathfinder";
  ti:"AdaGrad stepsizes"; "Robust Estimation via Robust Gradient Estimation"; EG+; mirror
  descent+Robbins-Monro; dual averaging+feedback; heavy-tailed+SGD+clipping; autoMALA;
  position dependent+MALA; Riemannian Langevin+simplex; stochastic gradient Langevin+adaptive
  step; schedule-free+stochastic approximation; Polyak-Ruppert+tracking; Shampoo+quantized;
  K-FAC+large language; local R-hat; warmup+divergent+initialization; ti:"PID optimizer";
  ti:"Adaptivity without Compromise"; abs:"step size control"+PID+math.NA.
- HTML full-text fetched: arxiv.org/abs/2603.18845 (+ full cached text mined for Thm 2.2,
  Algorithm 1, discount/chopping paragraph); ar5iv 2305.14442 (via subagent).
- **Web searches:** ASIS/interweaving (Yu & Meng JCGS; Kastner CSDA; factor-SV); distributed
  Shampoo (2002.09018 + google-research repo + Meta repo); AutoMALA; cross-chain warmup
  (Stan Discourse 12912, 12039, 23617; metrumrg ACOP 2020/2021 posters); Gustafsson/Söderlind
  PID stepsize control (Lund portal, Springer, CWI PDF, Söderlind lecture notes); PID optimizer
  (CVPR 2018 PDF, stanford.edu PDF, Nature Comms, ICME 2020); RowAdagrad (→SM3/Adafactor, no
  primary); Hird & Livingstone (arXiv 2312.04898 + JMLR PDF + UCL thesis); SOAP repo
  (nikhilvyas/SOAP, modded-nanogpt-SOAP, NoteDance README, optax PR #1692) for
  precondition_frequency/QR-refresh mechanics.
- Subagent transcripts (4 tasks: Shampoo/QN [API-timeout, recovered by direct verification],
  natural-gradient [completed], control/robust-SA [completed], EG/loop-b [completed]) at
  ~/.hermes/cache/delegation/live/deleg_fdb1762e/task-{0..3}.log; all load-bearing claims above
  re-verified by the orchestrator regardless of subagent sourcing.

**Files modified:** only this report (as instructed).
