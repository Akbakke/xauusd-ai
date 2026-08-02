# GX1 project state

Updated 2026-08-02.

## Terminal decision

Entry launch is **BLOCK**.

There is no accepted V4 model, calibrated bundle, Entry/Exit edge proof,
learned-sizing authority or launch transaction. A
failed model decision is unavailable evidence, not `FLAT`, and cannot be
replaced by a rule or older artifact.

The fresh 2026-08-02 Run13 preflight passed 30/30 checks. Its V8/V13 dataset
producer completed under the hard 14 GiB job cgroup with train 369,303 rows,
val 5,904 rows and test 6,071 rows. Full-input liveness, pretrain audit and
unified Exit lifecycle all PASS. No training or OOS edge proof exists yet;
the launch decision therefore remains BLOCK.

## What is code-proven

The active Entry path has one exact model-native direction authority:
calibrated `LONG/SHORT/FLAT` logits and their argmax. External decision
bridges, Entry-IQL, manual confluence scores, post-model trend/session/SMC
filters and compatibility fallbacks have no authority.

The source, model and runtime contracts now require:

- 513 signals = 34 base + 479 specialist fields;
- 378 mandatory outputs from twelve causal layers + 101 deterministic
  TRAIN-only ranked fields;
- 142 continuous + five categorical context fields;
- sequence length 96, 22 supervised Entry heads and one unified Exit head;
- one 26-group/96-value learned final fusion;
- exact V4 M5/M15/H1/H4/D1 inputs with 111 fields per timeframe;
- all eight specialist families on every timeframe;
- 40 family×timeframe routes and 555 feature×timeframe gates;
- exact V4 field order, cache bytes, normalization state and runtime schema;
- recipe-owned history windows with progressively coarser resolution for
  progressively older context;
- no redundant global MTF length or missing-timeframe fallback; the Dataset
  requires the exact ordered positive five-timeframe length map;
- no fixed per-timeframe live direction weight;
- serve-parity v11 raw/final influence evidence before launch: 1,723 numeric
  routes (513 sequence + 513 snapshot + 142 context + 555 MTF) and five
  categorical counterfactual routes;
- Q/V/Advantage fusion audits restricted to valid `Advantage = Q - V`
  manifold states;
- exact split-boundary decision-window coverage and fully closed trailing HTF
  buckets.

The eight V4 family widths per timeframe are
`5/11/10/2/4/5/10/64` for structure, SMC, trend, volatility, momentum,
session, geometry and candles respectively. They sum exactly to 111. The
feature×timeframe count is exactly 555, not 565.

## What is measured on real data

The current source parquet is:

`/home/andre2/GX1_DATA/runs/FASE2B_REGIME_V4_20260605/v26_6yr_rebuild_20260725_seq513_model_native_v26/cv3/xauusd_m5_CANONICAL_V3_2020_2026.parquet`

It contains 536,086 ordered unique M5 rows from
`2019-01-01T23:00:00Z` through `2026-07-24T20:55:00Z`; SHA-256:
`eca51c97ac5a1097ff1b2baae5aea8c38ca162466103d5c2f3c1c18d135848ac`.

The frozen historical V4 cache is:

`/home/andre2/GX1_DATA/runs/FASE2B_REGIME_V4_20260605/v26_6yr_rebuild_20260725_seq513_model_native_v26/MULTI_TF_V4_CACHE_20260729`

Evidence:

| Timeframe | Rows | Causal warmup | Post-warmup constants | Exact duplicates |
|---|---:|---:|---:|---:|
| M5 | 536,086 | 29 | 0 | 0 |
| M15 | 178,767 | 27 | 0 | 0 |
| H1 | 44,726 | 27 | 0 | 0 |
| H4 | 12,096 | 27 | 0 | 0 |
| D1 | 2,351 | 27 | 0 | 0 |

Cache identity:
`ff9cac78cdf6d5d4338f4d07b77df822c95efb568ed80a1e864600580a2b361a`.

Embedded V4 liveness identity:
`42b2b9a4af1870796cf9b22c9257550cb004515095e5e4d2fa31fb22fe4a4b18`.

This is real-data input liveness under cache schema v2. Active cache schema v3
adds complete trailing-resample closure, so these bytes are no longer
admissible for a new dataset or training run. They are not a direction-edge
result.

The current verification boundary is recorded once below. This remains code
evidence, not a retrained model. The live collector is healthy without a
failure latch through complete M1 `2026-07-31T09:34:00Z`, while the last
immutable native pair remains frozen on 2026-07-24. Native publication refuses
the dirty working tree until the exact producer source is committed; that
clean-source gate must not be bypassed. The compact handover reports the live
path count instead of freezing it in authority documentation.

## Latest empirical model evidence

V26 remains the latest completed historical pre-V4 experiment, and V21C is
the latest model diagnosis. Its split/audit bytes and the rejected V18 bundle
were retired and deleted on 2026-07-29. No source/split dataset is currently
admitted. Both measurements precede the complete V4 MTF contract.

V21C trained on all 369,303 TRAIN rows and was stopped after eight epochs:

- train loss fell 36%;
- validation loss roughly doubled;
- epoch 1 accuracy was 0.4048 but SHORT support was zero and admission
  correctly failed;
- balanced/admissible epochs were 4–5 points below the 0.3858 majority
  baseline;
- balanced accuracy was 0.3438;
- a plain diagnostic MLP on the same substrate reached 0.4021 with all three
  classes alive and tradable AUC 0.5833.

The honest conclusion is that the old full model overfit before it balanced.
It did not prove a direction edge and cannot be promoted. It also cannot judge
the new V4 family×timeframe architecture because it never trained on it.

## V4 cooperation design

Each timeframe computes the same semantic families from its own closed bars.
Each feature has a contextual gate at each timeframe. The resulting 5×8 token
grid is processed along both axes:

```text
5 timeframe-native 111-field tensors
        |
        +-- per-TF TRAIN-only normalization
        +-- 555 contextual feature gates
        +-- shared semantic family encoders
        +-- timeframe temporal/positional encoding
        v
5 × 8 family/timeframe token grid
        |
        +-- attention across timeframes within each family
        +-- attention across families within each timeframe
        +-- 40 learned cooperation gates
        v
learned timeframe fusion + 22 Entry evidence heads + 26-group direction fusion
        v
one calibrated LONG/SHORT/FLAT argmax
```

The same encoder also consumes the frozen Entry representation plus the exact
closed-M1 lifecycle path for its unified `HOLD/EXIT_NOW` head. It trains in the
same smoke/candidate and never enters or competes with the direction fusion.

Existing cross-family engineered fields may remain as causal evidence
hypotheses. They are not direction rules. The raw independent V4 cells run in
parallel so the model can reject, downweight or condition those composites.

## Remaining blockers, in order

1. Use the immutable source, V4 cache and exact pair manifest already bound by
   V8/V13; mutable collector bytes remain non-authoritative.
2. Use the fresh V8/V13 combined Entry/lifecycle dataset and its PASS liveness,
   audit and lifecycle evidence; do not rebuild it without a new evidence-based
   source change.
3. Use the now-explicit, hash-bound dropout/layer/window recipe inputs to run a
   bounded TRAIN/VAL-only regularization/capacity sweep.
4. Train one unified V4 smoke and require exact 8/5/40/555 gate liveness,
   non-degenerate LONG/SHORT/FLAT and HOLD/EXIT_NOW support, positive Exit loss
   and movement in every Exit component.
5. Train one same-bundle/shared-encoder Entry+Exit candidate and preserve
   untouched TEST. Exit may not be retrained or replaced afterward.
6. Fit calibration only on the declared calibration split.
7. Prove TEST direction, abstention, cost, slice and top/bottom alignment.
8. Prove train==serve parity plus raw/calibrated margin movement for all eight
   specialists, five timeframes, 40 family×timeframe routes, 26 fusion groups,
   all 1,723 numeric routes and five categorical routes for both outputs,
   including exact closed-M1 Exit envelope parity.
9. Execute two consecutive fresh canonical successor/publication events
    through the implemented owner and publish the short-lived immutable
    live-tail admission. No real event is admitted yet.
10. Execute the implemented same-candidate full-TEST unified Entry/Exit
    producer, then prove zero-order shadow, promotion and the one-time launch
    transaction. No producer output exists yet.

No acceptance threshold is changed merely to make a run pass.

## Exit

Exit remains blocked. The former separate Exit stack is deleted. The current
source model and trainer now contain the same-bundle Exit head, causal
lifecycle loader, positive loss and parameter-movement/export gates. Runtime
also retains the frozen shared Entry representation, builds the exact
source-bound closed-M1 envelope, calls the same bundle, persists the decision
in TradeState v6 and journals it idempotently. There is still no admitted
freshly trained bundle or candidate-bound train==serve/full-TEST proof. The
canonical producer exists in the sizing owner, but cannot emit launch evidence
without that candidate, so runtime and replay intentionally fail closed. See
`docs/CANONICAL_EXIT_STATUS.md`.

## Operational state

Large failed split, audit, bundle and scratch artifacts were released through
the evidence-retention owner. The fresh V8/V13 dataset, liveness, audit and
Exit-lifecycle artifacts are retained as current offline evidence; no trained
model or launch artifact is admitted.
Dead-PID training scratch is now swept before a new allocation. Direct data
deletion and overlapping heavy jobs remain forbidden.

The authoritative chronology is `DECISION_LOG.md`. Historical V2/V3 MTF
readers and V26/V21C measurements are diagnostic history only; their rejected
dataset/model bytes are absent. V4 is the active source/model/runtime contract.

The installed legacy `gx1-canonical-incremental.service` targets a removed
daemon interface. Its service and watchdog timer are disabled/inactive. The
snapshot successor/publication/admission owner now exists in source and is
routed through the public control surface, but no real admission is
launch-bound. The repository watchdog remains deleted and the dashboard now
revalidates exact launch-bound live-tail authority instead of hard-coding a
publisher state.

## Repository verification

The last complete full-suite checkpoint before the 2026-07-31 native-successor
and live-tail changes collected 1,872 tests: 1,870 passed, two were explicitly
skipped and zero failed on 2026-07-30. Targeted native, pair, runtime, control
and launch-transaction tests for the later changes are green. Final
current-tree full-suite, syntax, lint, JSON, shell, diff and handover
verification is pending; no newer count or verification timestamp is claimed
here. These checks prove source contracts only, never a V4 direction edge,
unified Entry/Exit edge or launch authority.
