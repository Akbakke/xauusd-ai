# XAUUSD model-native direction handover

Updated 2026-07-22. This is the only GX1 handover document. Run
`bash scripts/gx1_handover.sh` once for the compact repository/launch/process
snapshot, `--check` on continuations, and `--verbose` only when the full
document must be printed again.

## Goal

Build GX1 into a gold/XAUUSD bot that learns tops, bottoms and abstention from
the full feature stack and selects `LONG`, `SHORT` or `FLAT` through one
model-native decision path. The practical precision target is intentionally
high, but no precision claim is valid until immutable OOS and live-like
evidence proves it.

The system must fail closed. There is no Entry XGB anchor, neutral bridge,
Entry-IQL fallback, hand-written live direction filter, stale artifact
selection, mutable-latest evidence or soft compatibility path.

## Current terminal status

**BLOCK.** No dataset exists for the active
`xau_seq513_model_native_direction_v2` contract, and there is no accepted model
bundle or empirical direction-edge proof. Candidate, replay, paper/demo/live
and promotion remain closed.

### 2026-07-21/22 former-v1 rebuild and foundation-contract rejection

V19 (`XAU_SEQ513_REBUILD_20260721_V19`) was the first fresh dataset lineage to
reach a `GREEN` terminal under the former v1 surface, exit zero, with reason
`stopped at smoke gate`. The source reaches the last complete M5 bar at
2026-07-21T20:00:00Z and contains 392,995 x 188 FULL_PLUS rows. All 187
numeric source fields are live, finite and free of constants/exact duplicates;
all five MTF caches bind full canonical-v3.

V19 then passed the exact TRAIN-only ranking, 513 manifest, preflight, all
three dataset splits, exhaustive liveness and pretrain audit. Split sizes are
369,081 TRAIN, 5,904 VAL and 3,934 TEST; every split binds 513 signals, 142
continuous context fields, five categorical fields, exact XAU tape bytes and
the same source/ranking identities. All 1,980 field/split liveness records
validate. Every one of the 34 auxiliary target fields is live across splits,
and target consistency has zero mismatches. June/July H4/D1 ATR shift is
explicitly recorded as `SHIFT_OBSERVED`; it is real current-regime evidence,
not a failure to be filled or normalized away.

This is a dataset breakthrough, not a model breakthrough: V19 has not yet
trained a model and proves no precision, OOS edge, cost robustness or live
authority. V18 is immutable `RED`: its chain was stopped when a host clock
rollback made the requested ranking filename appear in the future. No V18
output may be reused. V17 and earlier lineages remain diagnostic only.

The first real foundation audit then failed closed. All 57 implemented
`chart.foundation_*` fields were absent from the selected V19 signal surface
and from the split manifests, even though the downstream structure, SMC,
volatility and session specialist layers consume their derivations. Root cause:
the ranker's reflective candidate discovery never exposed
`FOUNDATION_STRUCTURE_FEATURE_NAMES`, while the 316-field mandatory registry
also omitted them. A V19 validation scan proved all 57 active and found no
exact duplicate among the prior 316 fields. The active v2 contract therefore
retains all 57 as a mandatory cross-family layer: 373 mandatory + 106
TRAIN-ranked = 479 specialist fields, still 513 total with the 34 base fields.
V19 is immutable rejected evidence and cannot be trained or smoke-tested.

The field-by-field semantic audit also found that four builders inverted the
canonical compression inputs. H1/M15 ratios are `ATR14 / ATR100` (below one is
compression), while `_v1_bb_squeeze_20_2` is
`bandwidth / mean_bandwidth - 1` (negative is squeeze). The new
`entry_volatility_semantics_v1.py` is the only transform owner; foundation,
volatility, chart geometry, chart core and deep interactions use it. Release
now means lagged compression followed by positive expansion acceleration.
Non-positive ATR ratios fail closed. All 57 corrected foundation outputs are
finite and variable on the 5,808 post-warmup V19 validation rows, but those
rows are diagnostic only and cannot authorize reuse.
The complete 57-row routing/duplicate table is in
`docs/FOUNDATION_FEATURE_ROUTING_AUDIT_20260722.md`.

The next routing audit found that smoke execution was impossible even after a
green rebuild: it required a post-rebuild artifact whose producer had been
deleted and a separate smoke-split schema no active producer could emit. The
new `model-native-post-rebuild-readiness` owner binds V19's exact green
terminal, preflight, liveness, pretrain and six canonical split artifacts. It
requires smoke and source to be the same canonical dataset; no copied or
parallel smoke dataset is admitted. Smoke manifest, readiness and
trainability now consume that one contract. The launch validator, trainer,
smoke-bundle audit and adoption-readiness all require V19's canonical split
schema; the unproducible smoke-only split schema has no remaining consumer.

The V19 source log also exposed 12 initial H1/H4 warmup rows being represented
as neutral zero in legacy HTF alignment. Those rows were outside every V19
model split, so they do not invalidate V19. Future dataset/live construction
now exposes leading warmup as `NaN`, fails closed when no completed HTF bar is
available, and uses one canonical causal path. An unreachable stateful branch
with different shift semantics and all of its unused state fields were
deleted. This closes a potential train/serve parity mismatch without changing
V19's admitted rows.

The active OANDA collector has been checked read-only through 2026-07-21:
47,086 canonical-overlap M1 rows are bit-exact across all 13 numeric fields;
all duplicate timestamps agree; all values and OHLC/bid-ask geometry are
valid. The new `materialize_current_m5_snapshot_v1` producer snapshots exact
collector bytes into the event, rejects conflicts, emits only provably complete
M5 buckets and requires exact M5 overlap before atomic publication. V13 proved
that seam but was rejected before dataset construction because its MTF cache
used trimmed model-range rather than full canonical-v3. V14 rebuilt fresh and
its source cascade is PASS through 2026-07-21T17:00Z (392,959 x 188; all 187
numeric fields live; no constants, exact duplicates, nonfinite or fallback).
V14 used TRAIN ending 2026-05-31, June VAL and July TEST through the explicit
last closed M5 cutoff. End dates are mandatory CLI inputs, never defaults.

Historical V11 incident follows for root-cause provenance.

V11 was the prior terminal attempt. It completed a fresh v2 source-cascade
audit (385,677 rows, 188 columns, all 187 numeric fields live, no constants or
exact duplicates), TRAIN-only rank reference, feature ranking, exact 513-field
manifest and green preflight. Dataset construction then failed closed twice
(initial attempt plus its one exact retry) with
`MODEL_NATIVE_COMMON_HISTORY_WARMUP_INSUFFICIENT: clean_rows_before_emit=1
required=95`.

The exhaustive checkpoint inspection proved that all 60 Group-A,
liquidity/dip and structure outputs remained NaN for 13,714 rows, through
2021-03-15T23:55Z. The code rebuilt 60 closed D1 bars from the decision slice
beginning 2021-01-05 instead of from the full causal tape. This was a
train/live history-boundary mismatch, not missing market evidence. The
documented 13,439-row/277-clean-row assumption was false.

Commit `4134ca19` repairs the owner path: attach accepts a separate full M5
prefix, verifies every decision timestamp and high/low/close exactly, binds
the full prefix into checkpoint schema v2, and dataset/live callers pass the
complete causal history through the decision cutoff. Live HTF/REGIME_V4 also
computes on full cv3 before slicing. A real Jan-5 probe over 276 rows proved
Group-A warmup=0 and finite D1 liquidity/ATR-term/dip/structure from the first
row. The earlier zero-copy commit `1a51ce42` raised complete feature throughput
to about 2,062 rows/s; a 4,096-row block measured 1.99 seconds and was
bit-identical to V10 output over 17 x 60 sampled values.

No V1-V19 ranking, manifest, preflight, checkpoint, dataset or source artifact
may be promoted or resumed. Only a fresh V20 lineage under the active v2
contract may advance through the exact post-rebuild and empirical gates.

The 2026-07-21 feature audit then closed the remaining build/serve skew before
another heavy run: the TRAIN-rank reference is created before ranking and
hash-bound through manifest/preflight/dataset; price-derived EMA inputs use
only ranked common-history `close`/`atr`; signed BOS/sweep pressure is
directionally symmetric; and the unprovable partial live MTF splice is removed
so any context gap emits no direction until full refresh. The exact specialist
partition is now 373 mandatory fields across twelve families (including all
57 foundation cross-family fields and all 11 M5 EMA50/200 fields) plus 106
TRAIN-ranked fields. These are source-contract
repairs only; launch remains BLOCK and a wholly fresh rebuild is still required.

`PROJECT_STATE_xau_direction_launch.json` is the current Entry launch state.
Every earlier Entry dataset, bundle and report is rejected by the current
contract; none can act as launch, direction or compatibility authority.

Vedtak `XAU_SEQ513_REBUILD_20260718_V1` was issued and seq513 rebuild attempts
ran on 2026-07-19. They were terminated and invalidated: the reused feature
ranking covered TRAIN `2020-11-13..2026-03-31`, while the active build contract
was TRAIN `2021-03-16..2026-03-31`, and the then-current preflight omitted that
nested window check. No rebuild process is running now. No dataset, signal
manifest, bundle or edge result from those attempts is accepted. Partial
artifacts are non-authoritative. The event-local `CHAIN_STATUS.json` is now a
terminal schema-v2 `RED` record with reason
`FEATURE_RANKING_TRAIN_WINDOW_MISMATCH` and bound vedtak/git/artifact hashes.

The preflight, wrapper and builder now require explicit feature-ranking and
signal-manifest lineage, matching vedtak/source hash and the exact requested
TRAIN window. Green source-contract tests prove wiring, not trading edge.

The smoke/candidate launch validators and trainer now bind six explicit split
artifacts: TRAIN/VAL/TEST manifest plus TRAIN/VAL/TEST parquet. Their exact
paths and hashes flow through the validated recipe environment; the trainer
does not infer VAL/TEST from a TRAIN stem or discover a split with glob/latest.

That identity now continues through the full downstream chain. Foundation
feature/target/specialist audits resolve each parquet only through an explicit
hash-bound split manifest and publish the same four path/hash fields. Smoke and
adoption gates compare those bytes to the candidate split declarations.
Selective-edge prediction, replay materialization, serve parity and learned
sizing derive their dataset inputs only from the matching immutable report;
unbound directory files have no authority and a byte mismatch fails closed.

No seq513 rebuild chain or training process is running. A new report-only
model-native abstention metadata run ended
`BLOCK_ABSTENTION_EMPIRICAL_GATE`: FLAT labels are balanced at TRAIN
`1400/4095` (`34.19%`), validation `530/1536` (`34.51%`) and TEST `516/1536`
(`33.59%`), and the active FLAT, utility and margin weights are positive. The
run read zero parquet, trained no probe and emitted no learned predictions.
Historical selection-benchmark bytes and exact learned-probe evidence are
absent, so it authorizes nothing and proves no edge.

The old empirical verifier/control route is deleted because its mandatory
historical bytes do not exist and therefore could never authorize a rebuild.
Future abstention admission begins with exact hash-bound candidate TEST rows,
an immutable proxy comparison and absolute OOT/cost/live-like gates.

## Exact Entry contract

- contract mode: `xau_seq513_model_native_direction_v2`;
- direction mode: `model_native`;
- 513 signals: 34 genuine base fields + 479 exact specialist fields;
- the 479 specialist fields contain all 373 outputs from twelve code-owned causal
  layers in registry order, plus exactly 106 deterministic TRAIN-only ranked
  fields;
- 142 continuous context fields and 5 categorical context fields;
- sequence length 96;
- M5/M15/H1/H4/D1 context;
- positional encoding, learned TF scales, MTF/cross-TF attention and FiLM;
- eight specialists: structure/swing, SMC/liquidity, trend/EMA,
  volatility/compression, momentum/flow, session/regime, chart geometry and
  price-action/candles;
- all advertised direction, hierarchy, path, utility, timing, tail,
  volatility, validity, specialist, TF-agreement and size heads supervised
  with positive objective weight;
- exactly 22 active heads feeding one ordered learned 26-group/96-value fusion
  (`LayerNorm(96) -> Linear(128) -> GELU -> Linear(3)`);
- full-counterfactual LONG/SHORT/FLAT Q targets and expectile-V at K12/K48/K96,
  with exact Q/V/Advantage export and no separate IQL policy;
- target foundation audit v2 requires all 46 aux targets in every split;
  prediction/smoke schema v2 requires learned VAL/TEST alignment for the exact
  12-value LONG/BOTTOM and SHORT/TOP timing layout plus Q reward-ranking,
  V/max-Q and `Advantage=Q-V`; finite/non-constant output alone cannot pass;
- final calibrated `argmax([LONG, SHORT, FLAT])` is the sole direction authority;
- one exact runtime evidence snapshot, validated unchanged at decision,
  `TradeState`, journal and daily-review boundaries;
- exactly 96 bars ending at the latest closed M5 row, with a fixed 90-second
  post-availability decision limit and 390-second canonical-cutoff age limit.

`entry_model_native_signal_v1` is the exact owner of the 34 base, 142
continuous-context and 5 categorical-context fields. Active Entry has zero
imports from `signal_bridge_v1` and `signal_bridge_v3`. Missing, invalid or
session-inconsistent context, including a fabricated ASIA flag, yields
`MODEL_NATIVE_ENTRY_CONTEXT_NO_DIRECTION`; no bridge or synthetic `FLAT` may
repair it.

All genuine trend/session/liquidity/volatility/momentum/structure/chart/candle
evidence remains. Only disconnected rules that could independently veto,
flip, threshold or pass a direction have been retired.

Repository cleanup is an always-on token/credit rule in `AGENTS.md`: when work
exposes apparently unused code, perform one bounded ownership/reference check
and delete it with its sole-purpose baggage when safe. Do not accumulate dead
copies or repeatedly rescan the repository; preserve unique reproducibility
evidence and active Exit ownership.

The learned `position_size` head remains mandatory. Its target is exactly
`sigmoid((MFE-MAE)/(2*ATR_bps))`, where MAE is a non-negative adverse
magnitude and `FLAT` is neutral at `0.5`; prediction is parity-checked and
journaled. Current execution admission requires explicitly learned-calibrated
sizing with hash-bound calibration, exact XAU/account constraints, immutable
TEST utility/exposure/drawdown proof and exact train/replay/serve parity.
`FLAT` executes zero units. Missing or red sizing proof yields no order; a
historical fixed-1x comparison is never a fallback or launch authority.
No fresh accepted sizing result exists for the current seq513 contract; any
label-horizon result remains diagnostic even after it is produced.

The separately retained Exit-V3 artifact remains auditable through its trainer
and thin-record reader, but a fresh Exit-V3 rebuild is
`BLOCKED_PENDING_NEW_EXACT_BUILDER`: its historical producer used retired Entry
side threshold/fallback logic and lacked a mandatory `--vedtak` write gate. Do
not restore that producer. A replacement must consume admitted model-native
LONG/SHORT/FLAT evidence and fail closed.

The M5 row is available only after its fixed five-minute close. Entry freshness
has no runtime override: an incomplete 96-row window, a row other than the
latest closed M5, latency above 90 seconds after availability, or canonical
cutoff age above 390 seconds emits no model direction. It is never converted
to `FLAT`, a cached decision or backlog execution.

## Completed source work

- Added exact seq513 signal, split-manifest, readiness, train-launch,
  objective, immutable-event and full-input-liveness contracts.
- Made full-stack retention executable rather than aspirational: the signal
  manifest producer always retains all 373 registered causal-layer outputs and
  permits deterministic TRAIN-only ranking to fill only the remaining 106
  specialist positions.
- Closed the rebuild-lineage provenance gap: the wrapper now passes one
  validated `--run-id` into both writing producers. The rank NPZ and sidecar,
  dataset build proof, model-native state contract and every split manifest
  bind that exact ID; a missing, placeholder or mismatched ID fails before the
  dataset builder writes. The current accepted state/rank artifact schemas are
  `model_native_state_contract_v4` and
  `model_native_train_rank_reference_v4`; older artifacts cannot pass.
- Closed the July-19 feature-ranking lineage gap: preflight, wrapper and builder
  require explicitly named ranking/manifest artifacts and revalidate their
  run ID, source hash and exact TRAIN start/end. A manifest ranked on a
  different split cannot be reused.
- Bound the exact eight-specialist partition and all active heads.
- Removed legacy Entry modes, anchor/neutral-bridge arguments and optional-head
  serving paths from the model, trainer, builder and strict bundle loader.
- Added exact smoke/candidate wrappers with immutable prerequisite bindings;
  no mutable discovery or generic trainer pass-through remains.
- Removed trainer split discovery/stem inference. Both wrappers forward all
  six explicit manifest/parquet identities and the trainer revalidates their
  hashes, self-paths, vedtak lineage, distinctness and shared contract before
  reading rows.
- Bound live serving gates to the exact launch-declared event path/SHA before
  revalidating immutable-event authority. Fixed-roots newest discovery cannot
  choose a different gate first.
- Made a failed live `decide()` an explicit retryable
  `MODEL_DECISION_UNAVAILABLE`, never synthetic `FLAT`/`SKIP`; downstream
  pipeline/runner tests forbid auxiliary-head direction branches or mutation
  of the model's direction/action fields.
- Moved rejected Smart520 and absent Entry-IQL records out of the artifact
  registry's `active` inventory. Only retained Exit artifacts are active.
- Added `gx1_handover.sh --check`: a deterministic authority fingerprint and
  minimal state view for continuations, avoiding repeated all-Markdown reads.
- Compacted and hardened the canonical smoke bundle audit. It strict-loads the
  bundle and proves state/meta/lock hashes, architecture, objective identity,
  exact 22-head and 26-group/96-value fusion identity, learned-component
  movement and immutable validation/test evidence. Support, confusion counts and Wilson
  lower bounds are recomputed globally, per class and per declared context
  slice.
- Extracted neutral model-native replay primitives. Candidate replay consumes
  final model direction and records logits, supporting evidence and a
  unit-normalized label-horizon outcome for offline direction diagnostics only;
  it explicitly applies no position size and has no execution authority.
  Executable learned sizing requires a separate immutable sizing OOS,
  exact adopted-Exit replay and post-adoption runtime-parity chain. Strict
  full-TEST joint-Exit and broker-shadow finalizers/validators now require
  bound raw trace/observation parquets, but the current label-horizon sizing
  proof is diagnostic only; capital authority remains `BLOCK` until fresh real
  bindings exist and live/paper can enforce `NO_ORDER` when any part is missing
  or red.
- Refactored calibration/evidence to immutable lineage and removed mutable
  report authority.
- Added `entry_model_native_runtime_evidence_v1` as the shared exact evidence
  contract for model decision, `TradeState`, journal persistence/recovery and
  daily review. It proves calibrated direction parity, hierarchy, path,
  specialist, MTF, utility and learned-size evidence and rejects retired fields.
- Split Entry freshness from Exit operational freshness. Entry now requires the
  exact latest closed M5 window and fixed 90/390-second limits; stale-data
  environment overrides cannot enable an Entry decision.
- Moved genuinely shared Exit-IQL artifact helpers into an Exit-owned module;
  deleted the old Entry-IQL/140_94/hard-safety chain without changing Exit math.
- Removed legacy warm start, manual sizing overlays, Sniper policies, old
  Monday/R5/R6/TRUTH/shadow-meta research, generic Entry-IQL, obsolete
  foundation activation, old Entry XGB/RL launchers, disconnected inference/
  router/prod modules, archived source copies and their sole-purpose tests.
- Deleted zero-reachability runtime and compatibility files including the old
  V10 live adapter, duplicate trade-log schema, detached Entry context/live
  feature modules, Entry critic trainers/runtime and manual sizing modules.
- Removed stale Sniper/XGB/Trial160/seq215 architecture documents.
- Removed the last active Entry imports from the V1/V3 signal bridges and made
  the model-native signal contract the one exact base/context owner. The
  retained V3 XGB bridge remains Exit-only: it owns real ordered 7/41-field
  validation for two Exit consumers and fails closed on import/order mismatch.
- Closed the remaining prebuilt/builder context split-brain: the Entry contract
  now owns the source/micro/swing/session groups, one strict causal micro helper
  and one confirmation-lag swing helper serve offline and live paths, and the
  builder always discards source copies before recomputation. The old
  `shift(-1/-2)` swing lookahead, optional ctx dimensions and cross-parquet
  side-load are deleted. canonical-v2 and the exact source prebuilt now have
  explicit disjoint field-owner sets; missing owner fields fail closed.
- Put every retained OANDA backfill writer behind explicit `--vedtak`
  validation before side effects. The retired Entry-IQL artifact registry
  entry is now `path=null`, status `RETIRED_ARTIFACT_ABSENT`.

Focused contract suites and full test collection have been green at each
completed boundary. Re-run the complete verification after the concurrent
source cleanup settles.

## Remaining source boundary work

1. DONE 2026-07-17: repository-wide stale-reference/call-graph/duplicate-owner/
   contract-consistency audit completed (55 adversarially verified findings);
   zero-reachability deletions executed under explicit user approval; full test
   suite green (1341 pass / 5 skip / 0 fail). See DECISION_LOG 2026-07-17.
2. DONE 2026-07-17: contract hardening — the mandatory causal-layer
   registry PREFIX ORDER is validated at every manifest consumer; the five
   launch-JSON `required_*` partition constants are enforced against code
   constants; the 90-second Entry latency limit has one numeric owner.
3. DONE 2026-07-20: fail-closed adaptation source boundary — row-recomputed
   same-bundle drift, replay-readiness v2 byte handoff, immutable
   initial/refresh/drift/challenger/shadow/promotion/rollback transitions and
   launch cross-binding are implemented. Shadow is a mandatory paired
   incumbent/challenger bid/ask comparison with absolute and lower-95%
   improvement gates. Failed refreshes invalidate older
   green events. No real lifecycle evidence was produced.
4. DONE 2026-07-21: removed the unreachable metadata-only abstention control
   route. The deleted historical benchmark is not a satisfiable pre-rebuild
   gate. Fresh candidate TEST rows must later pass the immutable proxy and
   absolute OOT/cost/live-like gates; training remains closed until an accepted
   dataset and all downstream prerequisites exist.

## Rebuild runbook

User decision 2026-07-17 (DECISION_LOG): the primary empirical admission
criterion is ABSTENTION QUALITY — the learned `FLAT` surface must match or
beat the historical selection benchmark OOT; flat-starvation (zero FLAT, the
failure mode of every July 8-16 smoke) is the central training problem.

Verified source material for a fresh seq513 rebuild (immutable, July-16 build):

- source parquet: `/home/andre2/GX1_DATA/runs/FASE2B_REGIME_V4_20260605/v10_6yr_rebuild_20260716_fresh_xau_direction_repair/FULL_PLUS_CTX_v3src.parquet`
  (sha256 93002a4b…, 2020-11-09 -> 2026-06-14)
- canonical v2 parquet: same dir, `canonical_features_v2.parquet`
- MTF cache: same dir, `MULTI_TF_V2_CACHE/`
- tape root: same dir, `cv3/`
- REJECTED and must be reproduced fresh under the new vedtak: the old
  `smart520_rank_reference_*.npz` (schema v1; the contract requires
  `model_native_train_rank_reference_v4`) and the 520-wide dataset
  (7 constant neutral-XGB bridge fields).

## 2026-07-18/19 campaign handover (vedtak XAU_SEQ513_REBUILD_20260718_V1)

CURRENT STATE: the July-19 attempts in event root
`GX1_DATA/runs/FASE2B_REGIME_V4_20260605/v10_6yr_rebuild_20260718_seq513_model_native/`
were terminated and invalidated. The feature ranking's TRAIN start was
`2020-11-13`, not the active `2021-03-16`; the old preflight incorrectly
reported GREEN because it did not compare the nested ranking window. No build
process is running, and no partial output is authoritative. In particular, a
terminal schema-v2 `CHAIN_STATUS.json=RED` now records
`FEATURE_RANKING_TRAIN_WINDOW_MISMATCH` with exact vedtak/git/ranking/manifest/
preflight hashes; it cannot be used to resume or select partial artifacts. Read
`SYSTEM_MAP.md` -> "Pipeline- og ingredienskart" before grepping anything.

### Built this campaign

1. `gx1/scripts/materialize_entry_model_native_train_feature_ranker_v1.py` —
   the previously missing ranking producer (deterministic TRAIN-only
   |Spearman| vs 24-bar forward mid return; sha+window-bound CHECKPOINT;
   parallel GROUP_A attach). Five source-contract tests included a round-trip
   through the real manifest producer. The ranking produced during the
   campaign is invalid for the active split because its TRAIN start differs;
   its scores and derived pre-EMA-mandatory manifest have no admission authority.
2. `gx1/scripts/repair_m5_tape_dec2024_from_m1_v1.py` — event-local repair of
   the Dec-2024 canonical M5 geometry defect (2375 impossible rows -> 0) from
   the clean canonical M1; convention proof (high/low EXACT vs M1 aggregation,
   open/close boundary-tick tolerance documented); 5757 bars rebuilt, 3459
   synthetic closed-market bars dropped; immutable REPAIR_MANIFEST.
3. Full source cascade rebuilt on the repaired tape: canonical_features_v2 ->
   cv3 -> cv3_modelrange (provenance sidecar) -> MULTI_TF_V2_CACHE ->
   FULL_PLUS_CTX. That campaign's 208-column surface is historical; the active
   liveness-clean source is now 188 columns after dead-source removal.
4. `scripts/run_seq513_rebuild_chain_v1.sh` — fail-closed chain driver. It now
   owns fresh TRAIN-rank reference -> fresh ranking -> fresh manifest -> fresh
   preflight -> fresh build, with
   immutable artifact targets rather than glob/lexical-latest selection or
   inferred resume. One exact checkpoint retry is allowed after a capped
   process failure. Telegram ⚙️/🔴/✅ and process-watch output are operational
   signals only; terminal authority still requires all three validated split
   manifests.
5. `attach_group_a_dip_struct_ctx_columns` factored into
   build_attach_context / compute_attach_rows / finalize_attach_columns plus
   `attach_group_a_dip_struct_ctx_columns_parallel` in the owner module —
   ONE full-series context fanned over workers (exact by construction; serial
   spot-check). The old 85.4 ms/row measurement was caused by a full float32 to
   float64 cache copy per lookup and is retired; current measured throughput is
   ~2,062 rows/s. Builder and ranker share it.

### Fixed: the 2026-07-17 hardening's requirement/supplier gaps

The hardening tightened consumers without wiring producers; nobody ran the
chain after it. Six gaps, all found by fail-closed walls, all fixed+committed:
ctx-adder raw-M5 loader now carries `volume` (REGIME_V4 requires OHLCV);
`v12_ctx_augment_live` empty-check means ROWS (zero-column output container is
legitimate); builder loads REGIME_V4 per-TF source scalars from the source
parquet (full-warmup adder values); builder HTF recompute reads FULL tape
history like serving (was: truncated frame -> D1 pctl252 NaN across first
TRAIN year); preflight `EXPECTED_MTF_BUILDER_VERSION` now IS the loader's
constant (pinned literals were mutually unsatisfiable); ranker tz bug at the
final write step (cost one 4.5 h run; checkpoint now makes late failures
cost seconds).

### Exact active window contract

Source (FULL_PLUS) first row 2021-01-04T23:55 (ctx-adder trims own warmup).
HISTORY_START=2021-01-05 < TRAIN_START=2021-03-16. The old claim of 13,439
Group-A warmup rows and 277 clean rows was disproved by V11: truncated context
actually produced 13,714 warmup rows and only one clean row. V12 then proved
the explicit full-prefix path leaves 2,207 causal REGIME_V4 warmup rows and
ample pre-TRAIN history. The active V14 split is TRAIN_END=2026-05-31 < VAL
2026-06-01..06-30 < TEST 2026-07-01 through the snapshot's explicit last
closed M5 bar.

The old preflight did not actually prove this end to end: it accepted a nested
feature ranking beginning `2020-11-13`. The hardened chain must regenerate and
bind a ranking whose TRAIN start/end exactly equal `2021-03-16` and
`2026-05-31`; a green status from the invalid attempts cannot be inherited.

### Open decisions / next work

1. No training is in flight and no dataset satisfies the active v2 contract.
   Build V20, materialize its post-rebuild readiness, run foundation
   feature/target/specialist audits, then smoke manifest/readiness/trainability
   and the exact dry-run/execute
   recipe. Zero FLAT predictions remains hard-red by definition (DECISION_LOG
   2026-07-17 abstention criterion). After an accepted candidate exists,
   adaptation still requires fresh TEST evidence, settled zero-order broker
   shadow rows and paired incumbent/challenger lifecycle proof.
2. Canonical M5 root AND live prebuilt still carry the Dec-2024 defect (only
   the event copy is repaired) — separate decision; live Exit serves on it.
3. Exit env-softeners (3 audit MEDIUMs), CI replacement, gx1/scripts sorting
   and hashing-helper consolidation remain post-smoke backlog. The former ctx
   v1/v3 builder dual-owner is resolved in source and adversarial tests; a fresh
   artifact cascade is still mandatory before it can claim empirical parity.
4. Recent-regime handling is not yet empirical. Compare full-history baseline
   with an immutable TRAIN-only recent-regime fine-tune/calibration challenger;
   June VAL selects, July TEST stays untouched until the declared final gate.

Ordered steps (each gate fail-closed; stop at first red):

1. Do **not** reuse any invalidated V1-V19 lineage. Create fresh V20 paths and
   hashes; never discover inputs through glob, mtime, symlink or mutable
   `latest` selection.
2. Run the full source/rank/manifest/split chain under
   `xau_seq513_model_native_direction_v2`, then run
   `model-native-post-rebuild-readiness` against V20's green terminal,
   preflight, liveness, pretrain and six canonical split files/manifests. It
   must prove source and smoke dataset directories are identical.
3. Run the exact foundation feature, target and specialist audits on V20.
   Accept only immutable PASS events bound to the same split bytes and seq513
   structure manifest.
4. Materialize `model-native-smoke-manifest`, then run
   `model-native-smoke-readiness` and trainability/recipe audits without
   changing dataset identity.
5. Run `model-native-smoke-train` with V20's immutable run id and `--dry-run`,
   validate the command
   contract, then run the capped `--execute` path only if every upstream gate
   is green.
6. Compare a declared full-history baseline and TRAIN-only recent-regime
   challenger. Use June validation for selection/calibration and preserve July
   TEST for the final untouched evaluation.
   Smoke acceptance ADDITIONALLY requires a non-degenerate FLAT rate on val
   and test (zero FLAT predictions is an automatic hard-red, as on
   2026-07-16) before any slice metric is even considered.
7. Candidate chain only after smoke PASS: trainability-readiness ->
   candidate-train -> calibration -> immutable prediction evidence ->
   unit-normalized replay -> the nine-item evidence list above.

Nothing in this runbook grants run authority by itself; exact evidence gates
remain authoritative and `--run-id` only prevents mixed artifact lineages.

## Required evidence before Entry can open

The deleted historical Entry-IQL selection benchmark is not a satisfiable
pre-rebuild prerequisite. Rebuild and candidate training must first produce
fresh learned OOT predictions; admission then requires a fresh immutable proxy
comparison plus absolute OOT support/confidence, cost and live-like gates.
Metadata and label counts never satisfy that empirical gate.

1. Fresh exact train/val/test split manifests and seq513 datasets.
2. Full 513+142+5 field liveness and ordered-hash proof on all splits.
3. Feature, target, specialist, leakage and pretrain audits.
4. Immutable smoke recipe and an audited bundle with every head/specialist active.
5. Honest calibration and untouched validation/test direction, class, slice,
   pocket, context, path-quality and utility metrics with adequate support.
6. Immutable candidate prediction evidence and live-like replay including costs.
7. Learned-size calibration plus untouched TEST utility/exposure/drawdown and
   exact train/replay/serve sizing proof.
8. Train==serve parity on identical bars, including journal fields, both raw
   and calibrated influence for context, all five timeframes, all eight
   specialists and all 26 fusion groups.
9. Newest terminal evidence PASS, exact bundle/hash admission and explicit
   paper/live launch decision.

If any item is missing, malformed, stale, hash-mismatched or empirically red,
the result is `BLOCK`.

## Operational takeover

```bash
cd /home/andre2/src/GX1_ENGINE
bash scripts/gx1_handover.sh
bash scripts/gx1_handover.sh --check
.venv/bin/python -m json.tool PROJECT_STATE_xau_direction_launch.json
scripts/entry_next_edge_control.sh --help
```

Use `.venv/bin/python`. Before any authorized heavy job, inspect disk, RAM and
active Python processes. Preserve persistent collectors, canonical data
builders, watchdogs and dashboard processes. Do not write or delete under
`/home/andre2/GX1_DATA` without the exact run/cleanup authority.

End every bounded change with focused tests, full collection when shared
imports changed, a stale-mode/path scan and an explicit distinction between
code proof and empirical trading proof.
