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

The exact physical native column order is `time`, `open`, `high`, `low`,
`close`, `bid_open`, `bid_high`, `bid_low`, `bid_close`, `ask_open`,
`ask_high`, `ask_low`, `ask_close`, `volume`. Missing bid/ask values,
non-finite prices, duplicates, non-monotonic timestamps or unexplained grid
gaps fail. Mid-only substitution is forbidden.

Canonical native M1 and M5 share one module owner,
`gx1.scripts.backfill_xauusd_m5_from_oanda`, routed as
`model-native-native-m1-source` or `model-native-native-m5-source`. Bootstrap
uses `materialize_native_xau_snapshot` and publishes source schema v3. An
advancing child uses `materialize_native_xau_successor` and must publish source
schema v4 with the exact immutable parent root and parent-manifest SHA-256.
Both modes accept an explicit immutable output root, vedtak, timeframe and
left-closed/right-open interval. Policy is owned by the contract, not the
caller: M1 uses fixed three-day chunks and M5 fixed 15-day chunks, each at most
4,320 theoretical grid slots. Requests are OANDA-only `MBA`; fixed request
sleep is absent because the shared client owns retry, `Retry-After` and
exponential backoff. Every normalized response is retained as deterministic
gzip evidence. Only source rows with literal `complete=true` enter the strict
14-column Arrow surface; source absence is closure evidence and is never
filled. Both schemas bind the exact timeframe policy, request closure,
response chunks, clean Git/source inventory, typed row digest, each year
hash/count/bounds and final absolute root. Schema v4 additionally binds the
parent and strict overlap/append envelope. Source and parquet rows are
independently rederived with a byte-identical streamed digest before a hidden
directory is fsynced and atomically published without replacement. Shallow
legacy M1 manifests, native-M5 v2 manifests, direct year-file merge and
alternate-provider repair cannot be admitted.

Raw BASE28 may carry only the 13 non-time physical M1 fields in that exact
native order. It may not carry broadcast M5 duplicates, canonical context,
session/regime fields, phase indicators or derived volume fields. The five
M1 phase indicators and four volume transforms are derived causally and
identically at training and serve time. The model bar is phase 4: the first
M1 timestamp whose decision availability can observe the just-closed M5 bar.

`atr_bucket` and `spread_bucket` are TRAIN-fit transforms, not raw pair
fields. A fresh dataset generation must bind a separate immutable rank
reference fitted only on the complete physical TRAIN population. Dataset,
bundle, replay and live must all bind and revalidate the same reference; the
mutable global `regime_bucket_edges_v1.json` has no launch authority.

The bounded event rebuild never edits a running collector or canonical tape.
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

This bounded snapshot operation is not continuous live-tail publication.
Native schema v4 adds an efficient immutable successor: exact parent-root and
manifest CAS, verified historical-chunk reuse, one bounded refetched overlap
plus the new tail, and byte-exact rejection of any completed-row rewrite.
Canonical successor publication emits its immutable PASS/BLOCK event before
the serving pointer moves. Two consecutive fresh PASS events may then produce
one short-lived admission. No real successor/admission chain is currently
published, so new paper/live Entry remains blocked; stale admission does not
remove model-native Exit authority for an already-open trade.

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

`model_native_state_contract_v6` is the only accepted rank/history contract.
It requires the exact `model_native_train_rank_reference_v5` payload, whose
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
5 categorical context fields. The five timeframes are M5/M15/H1/H4/D1. Under
the active V4 contract, each timeframe tensor has the same exact ordered
111-field surface and all eight specialist families. The combined grid is
555 feature×timeframe cells and 40 family×timeframe routes.
The 479-field specialist extension is generated inline from that split's common
causal history. It consists of the exact 378 code-owned outputs from all twelve
registered causal layers plus 101 eligible fields from deterministic
TRAIN-only ranking. A separately materialized sample-parquet extension is
forbidden.

The context identity tag is `CTX142CAT5`; the retired `CTX6CAT5` spelling is
invalid. Ranker and dataset extension builders must consume the same complete
causal M5 prefix. In particular, `vol_pct_m5_1yr` and `vol_pct_h1_1yr` must be
bit-identical under that shared history; truncated ranker history fails closed.

The structural auxiliary-label producer consumes only prerequisites declared
by `entry_structural_aux_label_signal_v1.py`. At least one candidate for each
of its 19 named requirements must be present in the 378-field mandatory
prefix; optional ranking may not make target construction conditional.

The pretrain polarity contract additionally binds support proximity,
resistance proximity, signed support-minus-resistance and channel position to
the mandatory prefix. The polarity proof and future-outcome target-consistency
proof are independent audit branches; failure of one never marks the other
unavailable.

The retired external decision-bridge fields are forbidden. There is no
zero-fill, median-fill, compatibility dimension, optional context or synthetic
decision surface. Missing or non-finite values fail before inference.

## Targets

Targets cover the public `LONG/SHORT/FLAT` decision and every advertised
supporting head: MTF direction, path/MFE/tradability/bad-path/clean-edge/
survival evidence, utility, hierarchy, side validity, trendline rail,
timing/tail/volatility, TF agreement and position size. Target timestamps and
future horizons must be auditable and strictly after the input cutoff.
The Dataset converts immutable parquet `y_direction` exactly once to the
class-index batch tensor `y`; the primary and MTF direction losses share that
same tensor. A duplicated `y_direction` batch alias is forbidden.

Counterfactual Q, expectile V and Advantage are internal evidence only.
Advantage equals `Q - V` exactly; parity and fusion ablation may not construct
an impossible state where those values disagree.

The position-size target is exactly `sigmoid((MFE-MAE)/(2*ATR_bps))`, where MAE
is a non-negative adverse magnitude. MFE is selected-side and spread-aware: it
remains signed when the path never earns back the entry spread. Path quality
is also a signed forward outcome. Validators, normalization, train loss and
validation loss must preserve both signed domains exactly; zero clipping,
absolute values and parked-zero substitution are forbidden. `FLAT` is neutral
during training and executes zero units. Any label-horizon TEST
utility/exposure/drawdown result is diagnostic only; no fresh accepted
current-contract result exists. Paper/live exposure authority additionally
requires a joint replay of the exact unified candidate bundle and a fresh
post-adoption broker runtime-parity event. Until both exist and pass, capital
adoption is blocked; fixed 1x is only a named historical benchmark and never a
fallback.

The same immutable model bundle and shared encoder must own Entry and Exit.
The lifecycle dataset records the frozen Entry snapshot plus every exact,
contiguous closed-M1 post-entry path state. Each step binds one committed
closed M1 bar to the following exact fresh quote; state PnL is recomputed from
executable bid/ask prices. The model emits finite calibrated logits ordered
`HOLD/EXIT_NOW`, and exact argmax owns the action. Missing state, cadence,
bundle identity, path hash or output evidence is terminal red, never synthetic
HOLD.

Canonical replay binds the candidate bundle directly before activation,
iterates every TEST row from exact T+5 fill to model `EXIT_NOW`, and publishes
the replay, path traces and complete transitive source inventory atomically.
Caller-supplied traces and an already-active registry cannot authorize launch.
The former separate Exit dataset, model, bridge, overlay and policy stack are
deleted and may not be restored or padded into the unified contract.

## Liveness and identity

Full-input liveness evaluates every 513+142+5 current-bar field plus all
5×111 V4 cells on train, validation and test. TRAIN rejects constants,
insufficient generic activity and sparse-event fields below their exact
declared support floor. VAL/TEST remain untouched
chronological observations: a genuine one-state regime is recorded explicitly
instead of being relabelled or fabricated, but non-finite values, unknown
categorical values outside the TRAIN vocabulary, duplicate/reordered fields,
forbidden fields and identity mismatches still fail. ATR distribution shift is
recorded exactly as diagnostic evidence; only later untouched OOS direction,
cost and calibration gates decide whether the model handles that shift. All
five timeframe representations must be present, alive and distinct on TRAIN.
The embedded V4 liveness contract additionally scans every post-warmup field
on every timeframe, rejects constants and exact within-timeframe duplicate
pairs, and binds causal warmup, field order and feature hash. A V2/V3 cache or
a cache without this complete 5×111 proof cannot authorize active Entry.
Active V4 cache schema v3 publishes only complete resample buckets; a
schema-v2 V4 cache is historical. Training must also prove the exact requested
M5/M15/H1/H4/D1 decision window at both ends of every split.
The materializer scans every `96 x 513` sequence value and requires exact
`seq[-1] == snap`; direct Arrow-buffer access is only a zero-copy performance
path and validates every nested-list offset before use. It does not sample.

Serve-parity v11 requires sampled local raw/final class-margin sensitivity for
1,723 numeric routes: 513 sequence, 513 snapshot, 142 continuous context and
555 MTF. Five categorical routes require valid-category counterfactual
movement. Route/group ablations and untouched OOS edge remain separate gates.

Consumers revalidate bound bytes. A copied manifest or report-level `PASS`
without matching content hashes is not evidence.

No dataset build is authorized merely by this document. Rebuild requires the
exact model-native preflight contract and one validated `--run-id` shared by
the complete immutable dataset-build artifact lineage. A later training run
has its own output `run_id`; its input `dataset_run_id` is derived by the
launch contract from post-rebuild and all three split manifests, never supplied
or rewritten by the operator.
