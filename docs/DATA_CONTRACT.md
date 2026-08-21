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
- ordered signals: fixed base fields + mandatory causal/raw families + the
  complete code-owned candidate remainder. **All counts are DERIVED from the owner tuples
  (`MODEL_NATIVE_SIGNAL_DIM` in
  `gx1/contracts/entry_model_native_signal_v1.py`, flattened from
  `MODEL_NATIVE_SPECIALIST_LAYER_FEATURES` in
  `gx1/features/entry_model_native_feature_layers_v1.py`) and are deliberately
  NOT restated here** — every restated count in this repository has gone stale
  within days (rule 13). Every emitted active owner field is available to the
  learned model; no fixed top-k/ranker has feature-selection authority;
- the continuous and categorical context widths as the owner reports them
  (`MODEL_NATIVE_CTX_CONT_DIM` / `MODEL_NATIVE_CTX_CAT_DIM`) — restated counts
  here were stale by a factor of two within days;
- same eight specialist owners, formulas, taxonomy, field order and lineage;
- the exact schema-version identities, **named here but never restated**:
  `MODEL_NATIVE_SIGNAL_SCHEMA_VERSION`,
  `MODEL_NATIVE_MANDATORY_FULL_STACK_SCHEMA_VERSION` and
  `MODEL_NATIVE_SPLIT_MANIFEST_SCHEMA_VERSION` in the signal owner;
  `HTF_V4_MATRIX_CONTRACT`, `HTF_V4_CACHE_SCHEMA_VERSION` and
  `HTF_V4_FULL_INPUT_LIVENESS_SCHEMA_VERSION` in `gx1/features/htf_features.py`;
  `CANDLE_PRIMITIVE_FEATURE_VERSION` in
  `gx1/features/entry_candle_primitives_v1.py`; and `SCHEMA_VERSION` /
  `TRANSFORM` in `gx1/contracts/entry_model_native_input_normalization_v1.py`.
  `bash scripts/gx1_handover.sh` prints them, and an artifact whose version
  differs fails closed before a byte is read. Six of them moved on 2026-08-19
  in one uncommitted wave, which is precisely why none is written down here;
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

The immutable V4 cache persists two owner surfaces per timeframe: the ordered
per-bar MTF matrix and the compact model-native scalar matrix. Both are exact
float32 arrays with filenames, sizes and SHA-256 values in the manifest. A
consumer must preserve these bytes when slicing or rebinding a cache; it may not
recompute the scalar matrix from a shorter local parquet. Intentional identities
between an un-pooled current scalar and its per-bar history lane are declared by
`MODEL_NATIVE_MTF_SCALAR_PER_BAR_EXACT_ALIASES_V4`. The cross-surface policy
derives both local representation paths from that owner; all other active exact
duplicates remain failures.

Group-A long-memory context is unavailable until every required causal source,
including the 60-D1 liquidity lookback, exists. The attach owner represents
that absence as one whole-row NaN prefix and records it; an enriched producer
must apply `trim_causal_context_warmup_prefix` before its all-finite output gate.
Numeric filling is forbidden, and any non-finite row after the first complete
row is corruption.

The M1 Exit surface must bind the exact immutable signal-manifest path/hash,
the same ordered field list at `MODEL_NATIVE_SIGNAL_DIM` width, and the
TRAIN-rank reference hash used by the M5 Entry build.
Any disagreement fails before dataset construction.

The V29 registry layers (level and trendline registries and their event
projections) consume TRAIN-fitted tolerances only. The rebuild chain fits
them once per lane by immutable chronological inner-TRAIN competing-risk
selection over the whole empirical threshold support (no quantile or window
recipe input exists): the five-TF constants freeze with provenance into the
V4 cache manifest, and the Exit M1-lane params freeze into the hash-bound
M1-enriched frame manifest (`v29_registry_m1_lane_params`). Both
materializers resolve the lane-correct frozen artifact fail-closed;
cross-lane payloads and provenance-free bare values are rejected. VAL, TEST
and serve never refit.

The declared TRAIN population of that fit is an ordered PAIR —
`declared_train_window_start` and `declared_train_window_end`, plus the
inner boundary strictly between them. Both bounds are required arguments of
both fit owners, both are frozen into the payload and its provenance, and the
chain re-reads each published payload through its own validator and requires
exact timestamp equality against `--train-start` /
`--registry-fit-train-end` / `--registry-fit-inner-end` before the next step
consumes it. Until 2026-08-15 the owners took only the upper bound, so the
fit silently ran from the first source row (2019) and recorded the result as
a TRAIN fit — see `docs/INDICATOR_FIDELITY_AUDIT_20260813.md` §0.

Each hyperfit source provenance additionally carries
`pair_manifest_artifact` / `pair_manifest_sha256`: one hash-bound pointer to
the pair generation the fit actually read, re-verified as an immutable file
on every V4 cache load. A retention pass that reclaims that generation must
therefore fail every consumer closed. The retired names
`split_manifest_artifact` / `split_manifest_sha256` (which carried this same
binding under a false label) may never re-enter the payload.

The five-field volatility-squeeze owner is applied independently on native
closed OHLCV for M1/M5/M15/H1/H4/D1. Production consumers require one exact
immutable TRAIN-only artifact per clock through the common six-clock manifest;
the manifest binds source, tape, split, pair, clock/bar-grid, file/payload
hashes and common TRAIN lineage. Bare/default/cross-clock parameters, fitting
on VAL/TEST/live and resampling computed squeeze fields are forbidden. Fit and
serve must decode with the same causal filter, and the admission gate rejects
parameters whose low state is unreachable. Six per-clock production artifacts
are fitted and admitted, but **no artifact path or hash is named here** (rule
13): three non-retired six-clock sets sit under
`/home/andre2/GX1_DATA/data/data/prebuilt/VOLATILITY_SQUEEZE_SIXCLOCK_*` with
three different `contract_sha256`, and the V31 chains bound
`..._GEN1f9424_20260818T160532Z`, not the `..._20260818` set the documents named
until 2026-08-19. Resolve the binding from the consuming run's own V4 cache
manifest key `volatility_squeeze_artifact_set`. No downstream rebuild is
admitted. V39 completed TRAIN/VAL/TEST, compact Exit lifecycle and full-input
liveness, then ended RED at the post-build pretrain audit because the retired
selected-side `y_bad_path` diagnostic is structurally zero under the
positive-PnL direction policy. Audit v6 keeps the scalar required, finite and
exact-copy checked, and requires both side-specific sources to remain live; it
does not misrepresent the parked scalar as an active head. V39 output is
invalid (rule 7).

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

A fit window may observe a field's values but may never define its domain. The
sample-inferred binary branch was REMOVED from input normalization on
2026-08-19: it stamped `binary_mask` from `all(x in {0,1})` over the fit window,
turning a window statistic into immutable bundle state with no declared origin
(rule 2a) and no way to fail closed (rule 18). Proven from source, several
signed ternaries emit `{-1, 0, +1}`; on the then-declared TRAIN window two
`ema_stack_aligned_v2` names happened to contain only `{0, +1}`, were fitted as
binary, and the first bear stack at serve raised
`[ENTRY_INPUT_NORMALIZATION_BINARY_VALUE_INVALID]` — no Entry action at all, not
even FLAT, in a daily downtrend. A field is now scaled as a nominal category
only where a contract owner DECLARES its exact domain; everything else,
including a genuinely two-valued flag, takes the continuous branch. No branch
replaces the removed one.

Every split manifest binds source/pair identity, builder commit, dataset run
ID, exact fields/order, row count, time range, target contract, lifecycle and
all content hashes. A training run ID must differ from the dataset run ID.

`--history-start` is not free. It must precede `--train-start` by at least the
widest per-timeframe receptive field in `PRODUCTION_MTF_PER_TF_WINDOW_BARS`
(`gx1/contracts/entry_exit_production_architecture_v1.py`) — the D1 lane —
counted as real closed D1 bars on the same axis the V4 cache uses, never as
calendar days, because the row clock skips weekends and closures. The chain
enforces this at `model-source-identity` alongside the existing 96-row local M5
sequence warmup; a `--history-start` that satisfies the M5 warmup can still
leave the first TRAIN rows with an incomplete daily receptive field, and until
2026-08-19 nothing said so. The measured consequence for the current tape, and
the derivation of the declared TRAIN window, are in
`docs/TRAIN_WINDOW_WIDENING_20260819.md`.

Forward label purging at a split boundary needs no separate owner and must not
acquire one: the builder rebuilds each split with its own computation end, so
the direction/path horizon truncation, the auxiliary union-completeness mask
and the Exit lifecycle `crosses_split_end` guard all bite at every window end,
each measured in rows on that split's own clock. There is no backward embargo
and none is required — feature lookback is causal and is exactly what serving
does, so the D1 receptive field is the warmup requirement above, not a leakage
one.

## Exit lifecycle

The lifecycle schema is owned by
`UNIFIED_EXIT_LIFECYCLE_EPISODE_SCHEMA_VERSION` and is never restated here.
Episodes persist scalar pointers into the immutable M1 feature artifact, not
paths or deterministic per-state vectors. The validator reconstructs all state
indices, source rows, timestamps, decision clocks and masks from each
`m1_start_row` and the hash-bound M1 clock, then verifies the compact population
stream. Thus storage is bounded while the supervised population is unchanged.
The row clock is `consecutive_authoritative_closed_m1_source_rows` under
`oanda_complete_true_source_absence_no_synthesis_v1`.

Weekend/closure gaps therefore advance to the next observed row. Duplicates,
reversals or unexplained absence fail. The current closed M1 row is included.
Entry and Exit consume the exact same feature definitions; the causal 15-field
in-trade path is additive and never replaces the shared M1 surface.
Two state probes per side are selected by entry/side/source coordinates before
the target is inspected. Tied targets are omitted without inventing a class.
Exit supervision has no caller-selected lookahead and **no fitted horizon at
all**. Every supervised state of every episode is owned by
`gx1/contracts/unified_exit_fitted_q_v1.py`: HOLD bootstraps from a frozen
TRAIN target-network snapshot at the next causal state, EXIT_NOW terminates at
the current executable quote. There is no discount, indifference band or
lookahead window, and the pathwise hindsight optimum in
`gx1/contracts/unified_exit_optimal_stopping_v1.py` is explicitly never a
training target. Until 2026-08-19 this paragraph described a
`gx1_unified_exit_target_policy_v1` with a TRAIN-median-spread indifference
band and a 1..512 discovery-curve knee; no such owner has ever existed in
source, and `tests/test_entry_v10_outcome_targets.py` asserts its absence in
the builder.
The owner can stream every non-tied state in bounded chunks; until checkpoint
selection evaluates that full stream, probe-only validation is not acceptance
evidence. Candidate training performs that complete post-selection VAL pass;
smoke metadata records that the gate was intentionally not run and cannot be
promoted.

The persisted Exit full-input envelope binds the exact first/last M1 sequence
timestamps and signal/context tensor hashes, all five MTF tensor/cache hashes,
and the frozen little-endian float32 bytes/hash of the learned six-block
Entry-decision token projection, plus path, side, entry quotes, trade ID,
bundle and dataset/pair identities. **The projection's block names and both
widths are owned by `gx1/contracts/entry_decision_token_v1.py`
(`ENTRY_DECISION_TOKEN_SOURCE_DIM`, `ENTRY_DECISION_TOKEN_DIM`) and are not
restated here** (rule 13): this sentence said "609-to-128" until 2026-08-19,
by which date the owner reported a different source width.
The detailed in-trade path is a latest-512 rolling tail. A learned input sees
the absolute elapsed bar index, while `full_path_chain_sha256` commits every
dropped row. Neither serving nor offline replay forces EXIT_NOW at row 512.

## Targets and replay

Future outcomes are supervision only and are never model inputs. **Direction's
horizon is not a constant and is not restated here**: it is TRAIN-fitted per
build by the maximum-chord knee of the cumulative material-profit discovery
curve over candidates `1..ENTRY_DIRECTION_TARGET_POLICY_MAX_HORIZON_BARS`
(`gx1/contracts/entry_direction_target_policy_v1.py`), and each build freezes
the selected value into its own `DATASET_BUILD_PROOF.json` as
`diagnostic_outcome_horizon_bars`. This paragraph claimed a fixed 24 M5 bars
until 2026-08-19; there is no 24 in the owner. The horizon is always M5 bars —
when higher-resolution M1 prices are used, replay resolves the target M5 bucket
and then its last observed M1 row, and may never count M1 rows as M5 bars.

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
evidence is deleted only after retention/reachability proof. **That proof is
partly UNIMPLEMENTED** (CLAUDE.md rule 9): `authority_protected_paths` in
`gx1/contracts/evidence_retention_v1.py` harvests absolute path strings from
three repo-root `PROJECT_STATE_*.json` files and never opens a manifest under
`GX1_DATA`, so no successor parent pointer and no lineage binding is ever
followed. Measured 2026-08-19 by executing that function: 3 protected paths,
one of them already gone, over a 34 GB tree. Data-to-data reachability must be
proven by hand until the owner covers it.

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
