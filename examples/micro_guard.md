# Optional micro-step restart heuristic

`--min-micro-steps 2 --min-micro-guard` checks the first50 sampling positions.
Fewer than25 distinct full unconstrained positions triggers one MM1 restart.
Use `--min-micro-guard-probe` and `--min-micro-guard-min-unique` to set these limits.
An armed guard requires `1 <= min-unique <= probe <= samples`.
The guard is off by default and inert with min-micro-steps1.

This is a pin signature, not a convergence test. It can miss later failures
or restart a slow but healthy chain. Data-dependent selection of an attempt
is a heuristic: same-seed equality does not establish an unbiased posterior.
The MM1 fallback can also remain stuck. Always check multi-chain Rhat and ESS.

A restart uses the captured initial unconstrained vector, original seed,
full configured warmup, and a fresh generated-quantity RNG. Only retry draws
(and retry warmup if requested) reach the output CSV. There is no second retry.
This requires deterministic density evaluations; external state is not reset.

All phase timing stanzas remain in the log, including discarded work.
Sum every phase for cost analysis; do not take only the final sampling stanza.
`total_attempt_wall_s` includes initialization and both attempts, not final CSV writing.
`logp_grad calls` counts completed calls, including internally handled BridgeStan
errors. Separate nonfinite-return/throw counters are not BridgeStan-error counts;
raw loader error messages remain available. A nonfinite return can be a valid
zero-density point. Throwing calls are excluded from completed calls.

The inherited acceptance policy is an explicit dependency on reviewed fork PR23
(`726d8bb56290ec27410bc1734bb3da7006ac6ff8`): nonfinite acceptance maps to rejection.
No new acceptance policy or measured recovery/ESS/s guarantee is claimed here.
