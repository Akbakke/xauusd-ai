# GX1 model-native data contract

## Scope

GX1 trades XAUUSD only. Large tapes, derived datasets and model artifacts live
under `/home/andre2/GX1_DATA`; the repository contains code and contracts.
Every run consumes explicit immutable absolute paths. Directory conventions
are organizational only and never artifact-selection authority.

## Source bars

The raw market source is complete OANDA `XAU_USD` bid/ask/mid candle data in
UTC. M1 is retained for exit and live-like fill/path reconstruction. M5 is the
Entry decision grid. M15/H1/H4/D1 state must be derived leak-safely using only
bars complete at the decision timestamp.

Required price columns are `open`, `high`, `low`, `close`, `volume`, the four
`bid_*` OHLC columns and the four `ask_*` OHLC columns. Missing bid/ask values,
non-finite prices, duplicates, non-monotonic timestamps or unexplained grid
gaps fail. Mid-only substitution is forbidden.

The current-data rebuild never edits a running collector or canonical tape.
It snapshots the exact collector parquet bytes into one fresh event, rejects
conflicting duplicate timestamps, proves finite OHLC/bid-ask geometry, builds
M5 only from either all five exact M1 minutes or the separately overlap-proven
OANDA 22:00 UTC reopen pattern (one candle at minute offset 4), and requires
bit-exact native-M5 overlap for every admitted bucket. Any other partial M1
bucket is an unsupported collector hole: it is omitted with its timestamp and
observed minute offsets in the manifest, never filled or silently admitted.
The last declared M5 bucket itself must remain admitted. The snapshot manifest
binds every source/snapshot/year hash and the explicit last complete M1/M5
cutoff. Changing live source files after snapshot cannot change event bytes.

## Entry split artifact

Each train/validation/test split has an exact immutable manifest with:

- canonical absolute data and source-tape paths;
- file sizes, content hashes and filesystem identity;
- chronological range, row count and leakage/embargo declarations;
- exact model-native signal contract and ordered field hash;
- exact 513 signal, 142 continuous-context and 5 categorical-context names;
- exact target schema and feature/specialist ownership;
- source revision and builder identity.

Splits may not overlap. Calibration and threshold-free evaluation operate only
on their declared split. Test data cannot influence field selection,
normalization, recipe selection, calibration or stopping.
All history/train/validation/test boundaries are explicit required chain
arguments and strictly ordered. There is no default model-range end date.
The finite `FULL_PLUS_CTX` surface, after every causal context warmup trim, must
begin at or before `feature_history_start_utc`; having 96 rows somewhere before
TRAIN is not a substitute. The source audit binds this boundary, and the chain
rechecks it before producing TRAIN rank state or ranking. A ranking artifact
filename timestamp must also be real and no later than contract-validation
time, so an impossible future provenance order fails before expensive work.

Full-history training preserves rare regimes but does not itself prove current
regime sensitivity. The current Entry trainer has path/side/class weighting
but no generic recency weighting. Any recent-regime phase must therefore be an
explicit immutable offline training recipe, use only TRAIN rows, keep
validation/test untouched, and beat the full-history baseline on the same OOS
contracts. Later updates are offline challengers with zero-order paired shadow
and explicit promotion; online/live weight mutation is forbidden.

## TRAIN-only state and common history

`model_native_state_contract_v4` is the only accepted rank/history contract.
It requires the exact `model_native_train_rank_reference_v4` payload, whose
NPZ and sidecar both bind the immutable Entry `--run-id`. The ID is lineage,
not an approval gate.
ATR and observed bid/ask spread are derived causally from raw prices. Their
quintile ranks use one immutable TRAIN-only ECDF whose fit begins at
`TRAIN_START` and ends exactly at `TRAIN_END`; validation, test and serving
reuse it without fitting or updating it.

The rank NPZ contains exactly seven keys: `schema_version`,
`entry_run_id`, `fit_start_ns`, `fit_end_ns`, `fit_row_count`,
`atr_bps_sorted` and `spread_bps_sorted`. Per-row timestamps, categories and
pinned ATR state are forbidden. Its sidecar binds source and artifact hashes,
declares `fit_scope=train_only`, and proves that no validation or test rows are
stored.

The deterministic feature-ranking JSON that selects the final 101 specialist
fields and the derived seq513 signal manifest are explicit immutable inputs.
The chain creates the TRAIN-rank NPZ before ranking. Ranking, manifest,
preflight, rebuild wrapper and dataset builder all revalidate their nested
lineage, Entry run ID, source hash, NPZ hash, sidecar hash and exact TRAIN
start/end against the requested build. A directory glob, mtime or lexical "latest" result is not an
artifact identity. Any mismatch invalidates the entire attempt; partial files
and non-terminal chain-status records cannot be promoted or resumed as proof.

TRAIN, validation, test and serving all compute features from the same explicit
`feature_history_start_utc`; split-local rolling-state resets are forbidden.
Group-A/dip-structure uses the same implementation offline and live. Only a
contiguous causal warmup prefix may be unavailable and trimmed. A later gap,
non-finite feature, fewer than 96 clean pre-emission history rows, or a stale/v1
contract fails closed. The trim contract also covers every REGIME_V4 source and
derived field, including long-lookback HTF availability, the 288-M5 D1 ROC and
the first observable D1 regime transition.

The decision/history frame is not itself sufficient context for Group-A. Its
M5/M15/H1/H4/D1 liquidity, pivots, volatility percentiles and dip/structure
features consume a separate complete causal M5 prefix through the decision
cutoff. Every decision timestamp and high/low/close tuple must occur identically
in that prefix. Missing/mismatched rows fail; the implementation cannot reset
the 60-closed-D1 liquidity window at `feature_history_start_utc`.

Long Group-A materialization uses one full-series causal context and disjoint
4096-row work ranges. Each schema-v2 persisted chunk is bound to exact decision
frame bytes, full causal M5-prefix bytes, five-timeframe cache arrays, ordered
output fields and the run/window key.
Resume accepts only that exact namespace and completion contract; overlapping
context chunks, partial files, changed inputs and inferred checkpoints fail.

## Input tensors

The accepted Entry tensor contract is sequence length 96 with 513 genuine
ordered signal fields, a 513-field snapshot, 142 continuous context fields and
5 categorical context fields. The five timeframes are M5/M15/H1/H4/D1.
The 479-field specialist extension is generated inline from that split's common
causal history. It consists of the exact 378 code-owned outputs from all twelve
registered causal layers plus 101 eligible fields from deterministic
TRAIN-only ranking. A separately materialized sample-parquet extension is
forbidden.

The structural auxiliary-label producer consumes only prerequisites declared
by `entry_structural_aux_label_signal_v1.py`. At least one candidate for each
of its 19 named requirements must be present in the 378-field mandatory
prefix; optional ranking may not make target construction conditional.

The pretrain polarity contract additionally binds support proximity,
resistance proximity, signed support-minus-resistance and channel position to
the mandatory prefix. The polarity proof and future-outcome target-consistency
proof are independent audit branches; failure of one never marks the other
unavailable.

The seven retired XGB/neutral bridge fields are forbidden. There is no zero-
fill, median-fill, compatibility dimension, optional context or synthetic
decision surface. Missing or non-finite values fail before inference.

## Targets

Targets cover the public `LONG/SHORT/FLAT` decision and every advertised
supporting head: MTF direction, path/MFE/tradability/bad-path/clean-edge/
survival evidence, utility, hierarchy, side validity, trendline rail,
timing/tail/volatility, TF agreement and position size. Target timestamps and
future horizons must be auditable and strictly after the input cutoff.

The position-size target is exactly `sigmoid((MFE-MAE)/(2*ATR_bps))`, where MAE
is a non-negative adverse magnitude; `FLAT` is neutral during training and
executes zero units. Any label-horizon TEST utility/exposure/drawdown result is
diagnostic only; no fresh accepted current-contract result exists. Paper/live
exposure authority additionally requires
a joint sizing-only replay with the exact adopted active Exit stack and a fresh
post-adoption broker runtime-parity event. Until both exist and pass, capital
adoption is blocked; fixed 1x is only a named historical benchmark and never a
fallback.

## Liveness and identity

Full-input liveness evaluates every 513+142+5 field on train, validation and
test. TRAIN rejects constants, insufficient generic activity and sparse-event
fields below their exact declared support floor. VAL/TEST remain untouched
chronological observations: a genuine one-state regime is recorded explicitly
instead of being relabelled or fabricated, but non-finite values, unknown
categorical values outside the TRAIN vocabulary, duplicate/reordered fields,
forbidden fields and identity mismatches still fail. ATR distribution shift is
recorded exactly as diagnostic evidence; only later untouched OOS direction,
cost and calibration gates decide whether the model handles that shift. All
five timeframe representations must be present, alive and distinct on TRAIN.
The materializer scans every `96 x 513` sequence value and requires exact
`seq[-1] == snap`; direct Arrow-buffer access is only a zero-copy performance
path and validates every nested-list offset before use. It does not sample.

Consumers revalidate bound bytes. A copied manifest or report-level `PASS`
without matching content hashes is not evidence.

No dataset build is authorized merely by this document. Rebuild requires the
exact model-native preflight contract and one validated `--run-id` shared by
the complete immutable artifact lineage.
