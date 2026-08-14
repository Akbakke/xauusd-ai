# GX1 data contract

## Source authority

GX1 consumes only complete native OANDA `XAU_USD` MBA candles in UTC. M1 and
M5 are published as one immutable generation pair. Each artifact binds request
evidence, timeframe, interval, source revision, ordered columns, row digest,
year hashes and parent/overlap identity for successors.

Physical order is:

```text
time, open, high, low, close,
bid_open, bid_high, bid_low, bid_close,
ask_open, ask_high, ask_low, ask_close, volume
```

Invalid geometry, duplicate/reversed timestamps, non-finite or non-positive
prices, ask below bid, unproven source gaps or a ×10 scale discontinuity fail.
Mid-only substitution and synthetic gap filling are forbidden.

## Shared feature-owner contract

`gx1_entry_exit_shared_feature_base_contract_v2` owns both resolutions:

- instrument `XAU_USD`;
- ordered signals: 34 fixed base fields + the mandatory causal families + 133
  TRAIN-ranked, over 16 mandatory families. **The counts are DERIVED from the
  owner tuples (`MODEL_NATIVE_SIGNAL_DIM` in
  `gx1/contracts/entry_model_native_signal_v1.py`, flattened from
  `MODEL_NATIVE_SPECIALIST_LAYER_FEATURES` in
  `gx1/features/entry_model_native_feature_layers_v1.py`) and are deliberately
  NOT restated here** — every restated count in this repository has gone stale
  within days (rule 13). Three families are produced in full and pinned only in
  part; their unpinned fields compete in the TRAIN-ranked candidate pool, so
  the mandatory count is strictly smaller than the emitted count;
- 164 continuous and 5 categorical context values;
- same eight specialist owners, formulas, taxonomy, field order and lineage;
- same dataset run ID, split boundaries and TRAIN normalization;
- Entry local M5 sequence 96 plus closed M15/H1/H4/D1 context;
- Exit local M1 sequence 480 plus closed M5/M15/H1/H4/D1 context, maximum path
  states 512;
- independent native values: no combined pre-owner source package and no M1/M5
  value copying;
- no separate feature implementation or future feature reuse.

Every MTF candle is completed before the same owner computes its features.
Resampling finished M1 indicator/feature values into M5/M15/H1/H4/D1 is
forbidden. A forming higher-timeframe candle may never enter a decision.

The M1 Exit surface must bind the exact immutable signal-manifest path/hash,
the same ordered field list at `MODEL_NATIVE_SIGNAL_DIM` width, and the
TRAIN-rank reference hash used by the M5 Entry build.
Any disagreement fails before dataset construction.

The V29 registry layers (level and trendline registries and their event
projections) consume TRAIN-fitted tolerances only. The rebuild chain fits
them once per lane on the declared TRAIN window from the explicit recipe
input `--level-tol-quantile-q` (recipe owner
`ENTRY_LEVEL_REGISTRY_TOL_QUANTILE_Q`; no default exists): the five-TF
constants freeze with provenance into the V4 cache manifest, and the Exit
M1-lane params freeze into the hash-bound M1-enriched frame manifest
(`v29_registry_m1_lane_params`). Both materializers resolve the lane-correct
frozen artifact fail-closed; cross-lane payloads and provenance-free bare
values are rejected. VAL, TEST and serve never refit.

The M5 Entry surface must additionally match the exact full M5 source timeline,
dataset run ID and pair generation. Dataset construction loads it once through
bounded memmaps and uses only contiguous timestamp views for TRAIN/VAL/TEST.
Recomputing the selected specialist fields inside a split is forbidden.

Feature bytes cannot depend on ambient flags. The active ATR regime transform
uses direct integer indexing and must be non-constant on the complete declared
population. Any dead required field fails liveness.

The single TRAIN-rank artifact is fitted from the pair generation's canonical
M5 `time/high/low/close/bid_close/ask_close` fields. The downstream M5 model
source may contain a different feature schema, but those six fields must align
exactly for every model timestamp from common-history start through TRAIN end.
Its rank sidecar therefore binds canonical bytes, while ranking and labels bind
the final M5 source bytes. This is one explicit identity proof, not a fallback
or a second feature route.

## Split and fit boundaries

TRAIN, VAL and TEST are chronological, disjoint and hash-bound. TRAIN alone
fits feature ranking and input normalization. Recipe/model selection cannot
read TEST. Calibration may use only its declared held-out non-TEST split. TEST
opens once after a candidate and all calibration bytes are frozen.

Every split manifest binds source/pair identity, builder commit, dataset run
ID, exact fields/order, row count, time range, target contract, lifecycle and
all content hashes. A training run ID must differ from the dataset run ID.

## Exit lifecycle

Lifecycle schema is `gx1_unified_exit_lifecycle_episode_envelope_v3`. Episodes
point into the immutable M1 feature artifact rather than duplicating paths.
The row clock is `consecutive_authoritative_closed_m1_source_rows` under
`oanda_complete_true_source_absence_no_synthesis_v1`.

Weekend/closure gaps therefore advance to the next observed row. Duplicates,
reversals or unexplained absence fail. The current closed M1 row is included.
Entry and Exit consume the exact same feature definitions; the causal 14-field
in-trade path is additive and never replaces the shared M1 surface.

## Targets and replay

Future outcomes are supervision only and are never model inputs. Direction's
declared horizon is 24 observed M5 bars. When higher-resolution M1 prices are
used, replay resolves the target M5 bucket and then its last observed M1 row;
it must not add 24 M1 rows.

The accepted final evidence route freezes one candidate, evaluates untouched
TEST and replays that same bundle's Exit head from exact T+5 fills. Caller-made
direction/exit rows, fixed-horizon diagnostics and filtered subsets cannot
authorize a result.

## Bundle and calibration identity

A bundle commit binds exact model bytes, source inventory, field contracts,
normalization, M1 surface, dataset/profile/populations and immutable direction
and path calibration events. Sequential calibration must retain both events.
Sizing may consume only the canonical calibrated inventory and must preserve
its provenance into the finalized bundle.

Missing files, changed hashes, stale schemas, arbitrary metadata or unknown
environment controls stop closed.

## Resource contract

Large producers and all model runs use `scripts/gx1_capped_run.sh`, one at a
time. Tests/audits are capped at 4G; the heavy dataset producers and the
canonical trainer at 20G (raised from 10G on 2026-08-09 on real batch-640
RSS measurement; see CLAUDE.md); swap at 512 MiB and CPU at 0-1. Feature production uses one worker; model DataLoaders use zero
subprocess workers. Training is deterministic FP32 without compile, autocast or
TF32. Memmap scheduling is fixed in source, not environment-tunable. Generated
evidence is deleted only after retention/reachability proof.

## Retired ancestors

A superseded native tape generation may be retired only through the
retention owner (`gx1.scripts.cleanup_gx1_evidence_v1`). After retirement,
a successor's missing parent root is admitted at consumption time ONLY when
the executed DELETE_COMPLETE retention chain (plan → approval → execution,
every artifact re-verified by content hash) attests the deleted
`MANIFEST.json` with exactly the child's recorded parent manifest sha256.
The child-side proofs (interval advance, append contract, declared overlap
sha, row/append counts, time advance) still run from the child manifest and
the recorded parent binding; only the parent-byte re-proofs are replaced by
the attestation. An absent parent without attestation, a present-but-
tampered parent, and every other mismatch fail closed exactly as before.
