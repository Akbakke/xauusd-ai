# Risk of wrong code

Updated 2026-07-31. The filename is historical; this is the current compact
risk register.

## Highest current risks

1. **Historical input proof mistaken for active evidence.** The measured V4
   cache is schema v2; active schema v3 rejects it after the trailing-bucket
   closure repair. Zero constants/duplicates is not predictive skill.
2. **Old empirical evidence mistaken for V4 evidence.** V26/V21C used an older
   incomplete MTF surface. They cannot prove or refute the new all-eight-family
   5×111 path.
3. **Count drift.** The exact feature×timeframe surface is 555
   (`5 × 111`), not 565. Contracts and tests pin 111/40/555.
4. **Causality drift.** HTF state must be from closed bars available at the M5
   decision time. Leading unavailable state is warmup, never zero evidence.
5. **Hidden timeframe importance.** Window lengths and architecture scales
   must be recipe-owned. Live or wrapper-level fixed confluence weights are
   forbidden.
6. **Engineered composite mistaken for authority.** Cross-family formulas may
   remain evidence hypotheses, but only the learned calibrated model decides.
7. **Gate observability mistaken for influence.** Gate variation is liveness;
   raw/calibrated class-margin ablation is causal-use evidence.
8. **Overcapacity.** V21C overfit before balancing. Capacity and dropout must
   become explicit and be selected on TRAIN/VAL without touching TEST.
9. **Stale evidence lifecycle.** Pending recipes must match current source
   bytes; terminal failed historical recipes must retain and validate their
   own original binding rather than being rewritten.
10. **Source completion mistaken for empirical Exit proof.** The lifecycle
    materializer, positive same-bundle Exit loss, exact closed-M1 envelope and
    candidate-bound replay producer exist in source. No fresh combined dataset,
    trained unified candidate, train==serve proof or replay output exists.
    Entry edge alone cannot authorize capital.
11. **False live completeness.** Collector activity is not canonical-pair
    publication. No admitted live-tail event exists, and retired daemon,
    service or watchdog assumptions must never make the stack look live.
    Freshness is an exact new-Entry gate; making it a process-wide gate would
    also be wrong because it could suppress same-bundle Exit recovery.
12. **Collapsed influence proof.** Sequence and snapshot routes for the same
    513 name must each be sensitive; taking the stronger of the two hides a
    dead path.

## Required controls

- explicit immutable paths, hashes, schemas and field order;
- full-population TRAIN liveness and exact V4 cache liveness;
- no default, fill, fallback or compatibility direction path;
- TRAIN/VAL-only selection and untouched TEST;
- eight specialist, five timeframe, 40 family×timeframe and 26 fusion-group
  ablation evidence;
- all 555 feature gates finite, ordered and context-responsive;
- serve-parity v11 sensitivity for 1,723 numeric routes plus five categorical
  counterfactual routes;
- active schema-v3 split-boundary window and closed-bucket proof;
- one static launch anchor plus a newest-admission/exact-inference-pair
  recheck before every new Entry and order, without blocking Exit recovery;
- source tests described as code proof, never edge proof;
- one heavy GX1 process at a time;
- evidence-retention owner for deletion;
- launch remains `BLOCK` until all current artifacts exist and pass.
