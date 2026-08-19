# GX1 Engine

GX1 is an offline XAUUSD research and evidence pipeline for one learned trading
system. It is not currently an admitted or profitable trading bot.

## Architecture

```text
native OANDA M1/M5
  -> same 8 causal feature owners on separate native M5/M1 clocks
  -> shared encoder
       |-- Entry: local M5 + M15/H1/H4/D1 -> LONG / SHORT / FLAT
       |-- Exit: local M1 + M5/M15/H1/H4/D1 -> HOLD / EXIT_NOW
       `-- learned sizing/evidence heads
  -> calibration -> untouched TEST -> same-bundle replay
```

Entry consumes `MODEL_NATIVE_SEQ_LEN` local M5 bars plus continuous and
categorical context and closed M15/H1/H4/D1 lanes. **No width is restated here**
(rule 4/13) — the counts this paragraph carried were stale by 88 fields within
two days. Derive them by executing
`gx1/contracts/entry_model_native_signal_v1.py`; the runnable one-liner is in
`HANDOVER_XAU_DIRECTION_REPAIR_20260714.md`. Exit
consumes the same definitions and TRAIN normalization on 480 local M1 bars,
closed M5/M15/H1/H4/D1 context, the frozen Entry-decision token and its causal
in-trade path. The frozen value is a dedicated learned 128-wide Entry-decision
token built by a learned projection of the exact ordered six-block pre-argmax
decision source; the block names and widths are owned by
`gx1/contracts/entry_decision_token_v1.py`
(`ENTRY_DECISION_TOKEN_COMPONENTS`, `ENTRY_DECISION_TOKEN_SOURCE_DIM`) and are
not restated here (rule 13: the width restated in this paragraph was stale).
The last block is `entry_action_q_bps`, not a logit vector. It is
not a generic pre-head embedding. Higher-timeframe
OHLCV closes before feature computation; computed M1 indicators are never
resampled upward. Each MTF lane carries `htf_features.MULTI_TF_FEATURE_COUNT_V4` ordered fields;
the schema versions are printed by `bash scripts/gx1_handover.sh`.

Entry's tensors are read from one immutable, hash-bound native M5 surface and sliced by exact timestamp across all three splits. Exit uses the
corresponding native M1 surface. There is no split-local alternate feature
builder or cross-resolution value copy.

The raw volume primitives `vol_z_20`, `vol_ratio_5_20` and `vol_pct_96` use
the same owner on every closed timeframe, with tick volume summed during OHLCV
resampling. Their 95-row prefix makes the native source windows 191 rows for
the 96-row Entry slice and 575 rows for the 480-row Exit slice; warmup is never
zero-filled.

The five volatility-squeeze fields are produced by one shared state owner on
each native clock. M1/M5/M15/H1/H4/D1 each require their own immutable
TRAIN-fitted parameters inside one common lineage manifest; local and MTF
consumers reject missing, stale, cross-clock or bare payloads.

The MTF matrix, cache manifest and full-input liveness contracts bind one UTC
trading-session clock (versions printed by the handover script): H4 bins open on 22/02/06/10/14/18 UTC and D1 opens at
22:00 UTC. Retired H4 00/04/... and calendar-midnight D1 caches are not current
inputs.

Signal binds the exact causal candle geometry/relation/carry owner on local and
per-TF clocks; its width derives from
`gx1/features/entry_candle_primitives_v1.CANDLE_PRIMITIVE_FEATURE_NAMES` and is
not restated here. Its retained six-field local SMC event block emits
raw displacement, sided sweep depth, one-shot events and event age, not a
direction vote.

The accepted model's unique argmax is the only runtime decision authority, and
what it ranks is `entry_action_q_bps` — per-action expected return in basis
points, not a calibrated probability. Exact Q ties, missing paths, stale
artifacts and lineage mismatches fail closed; no post-model handwritten rule,
fallback or alternate replay selector may override it.

The objective and recipe owners
(`gx1/contracts/entry_model_native_training_objective_v1.py`,
`gx1/contracts/entry_model_native_train_recipe_v1.py`) own their schema
versions, keys and flags and **no count or version is restated here** (rule
4/13). Proven from source 2026-08-19: the sole decision loss is masked raw-bps
MSE on fitted-Q for Entry and Exit, task weights are learned by trainable
homoscedastic log-variance, and no cross-entropy holds decision authority — one
masked BCE survives on the `trendline_event` auxiliary head. Whether every
static magnitude in the trainer is gone is **not examined**.

The five handwritten regime composites, the handcrafted `tf_agreement` head
and `signed_vol_z_20` are retired while their genuine raw evidence remains for
learned interaction. Position sizing is trained only on its explicit tradable
mask using a frozen TRAIN-only selected-side path ECDF and has no direction
authority.

Each Exit output is bound to one full-input envelope containing trade identity,
side, entry quotes, Entry token/snapshot, path, exact M1 sequence timestamps and
tensor-byte hashes, and all five MTF window/cache hashes.

## Current status

The source architecture and contracts are substantially connected and tested,
but the system is not empirically finished:

- focused capped contract tests cover the current source changes; this document
  makes no aggregate whole-repository green claim after concurrent repairs;
- fresh native M1/M5 V4 sources and canonical pair generation
  `9b18e215...077232cd` (2026-08-09) are published and hash-bound;
- the historical V28 and V29J dataset chains were retired and are not valid
  training or comparison substrates;
- the current source contract is present; V31 rebuild chains have run
  repeatedly since 2026-08-18 and every one that reached a terminal event ended
  RED, so no admitted dataset exists;
- train==serve is a requirement, not a proven state: the serve-parity gate has
  never executed (zero events on disk, measured 2026-08-19) and one real
  divergence is proven in source (two different ATR formulas between the
  offline HTF owner and the serve ctx-augment block);
- no accepted Entry/Exit checkpoint or calibrated bundle;
- no untouched-TEST edge, historical PnL or win-rate proof;
- no same-candidate full-TEST unified Exit proof;
- Exit TRAIN/VAL state probes are now label-independent, and a bounded iterator
  can materialize every non-tied state in both 512-row trajectories. The epoch
  selector scores the probe population; before a candidate bundle can be
  written, its chosen checkpoint is re-evaluated on every non-tied long/short
  VAL state. Smoke runs explicitly cannot supply this admission proof.

Every dataset built on a retired feature surface is invalid as substrate for
new training. A fresh current-contract rebuild is required and has not landed;
the signal-surface schema version that a cache or dataset must match is printed
by `bash scripts/gx1_handover.sh` and is never restated here.

Registry fit payloads are bound to exact TRAIN-source provenance in the M5
cache and M1-enriched manifests. The level-registry runtime-population shadow
is a nonempty-support check through the same owner state machine, not a second
registry or a shadow/live-trading route.

Six-clock TRAIN squeeze artifacts were first fitted on 2026-08-15 and were NOT
admissible: the high-volatility state was absorbing under the runtime decoder
on all six clocks. The cause was the decoder, not the parameters — serve
replaced the accumulated posterior with a one-hot on the previous decoded
state, so a switch had to be bought with one bar's emission evidence against
the fitted persistence penalty. Fit and serve now share one causal
forward-filter decoder, the fit method identity is bumped so the old artifacts
fail closed at load, and the admission gate proves low-state reachability. The
refit `VOLATILITY_SQUEEZE_SIXCLOCK_20260818` is admissible and measured live on
real TRAIN bytes, but it has since been superseded: three non-retired six-clock
sets sit under `.../prebuilt/VOLATILITY_SQUEEZE_SIXCLOCK_*` with three different
`contract_sha256`, and the V31 chains bound `..._GEN1f9424_20260818T160532Z`.
Read the binding from a run's own V4 cache manifest, never from a document. No
chain has carried any of them to GREEN, so no model or edge claim follows.
Exit remains native closed M1. No tick-level dataset, evaluation, OOS or
trading claim exists.

## Start here

```bash
bash scripts/gx1_handover.sh --check
bash scripts/gx1_handover.sh
bash scripts/entry_next_edge_control.sh --help
```

Read `GX1_RULES.md`, `AGENTS.md`, `SYSTEM_MAP.md` and
`HANDOVER_XAU_DIRECTION_REPAIR_20260714.md`. Large immutable data lives under
`/home/andre2/GX1_DATA`; code lives in this repository. Never select artifacts
by `latest`, mtime or glob order. `requirements.txt` pins the small direct
runtime and verification surface used by the current Python 3.10.12 checkout.

## Resource safety

All heavy work runs through `scripts/gx1_capped_run.sh`, one job at a time.
Audits/tests are capped at 4G; canonical training is capped at 20G; swap is
512 MiB and CPU affinity is 0-1. Feature producers use one worker and model
DataLoaders use zero subprocess workers. Training has one deterministic FP32
path; compile, autocast, TF32 and runtime-selected fast modes are forbidden.
Partial or cap-killed output is not reusable.

Live, paper, broker, daemon, promotion and drift/adaptation work are outside
the frozen scope. See `GX1_RULES.md` for the complete binding rules.
