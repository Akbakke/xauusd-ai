# XAUUSD model-native direction handover

Updated 2026-07-17. This is the only GX1 handover document. Run
`bash scripts/gx1_handover.sh` to print it with repository, disk, RAM and
process state.

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

**BLOCK.** There is no accepted fresh seq513 dataset/bundle and no launch
authority. Candidate, replay, paper/demo/live and promotion remain closed.

`PROJECT_STATE_xau_direction_launch.json` is the current Entry launch state.
Every earlier Entry dataset, bundle and report is rejected by the current
contract; none can act as launch, direction or compatibility authority.

No rebuild, training or large replay has been run during this cleanup. Green
source-contract tests prove wiring, not trading edge.

## Exact Entry contract

- contract mode: `xau_seq513_model_native_direction_v1`;
- direction mode: `model_native`;
- 513 signals: 34 genuine base fields + 479 exact specialist fields;
- the 479 specialist fields contain all 305 outputs from ten code-owned causal
  layers in registry order, plus exactly 174 deterministic TRAIN-only ranked
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
- exactly 20 active heads feeding one ordered learned 23-group/75-value fusion
  (`LayerNorm(75) -> Linear(128) -> GELU -> Linear(3)`);
- final calibrated `argmax([LONG, SHORT, FLAT])` is the sole direction authority;
- one exact runtime evidence snapshot, validated unchanged at decision,
  `TradeState`, journal and daily-review boundaries;
- exactly 96 bars ending at the latest closed M5 row, with a fixed 90-second
  post-availability decision limit and 390-second canonical-cutoff age limit.

All genuine trend/session/liquidity/volatility/momentum/structure/chart/candle
evidence remains. Only disconnected rules that could independently veto,
flip, threshold or pass a direction have been retired.

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
  manifest producer always retains all 305 registered causal-layer outputs and
  permits deterministic TRAIN-only ranking to fill only the remaining 174
  specialist positions.
- Closed the rebuild-authorization provenance gap: the wrapper now passes one
  validated `--vedtak` into both writing producers. The rank NPZ and sidecar,
  dataset build proof, model-native state contract and every split manifest
  bind that exact ID; a missing, placeholder or mismatched ID fails before the
  dataset builder writes. This intentionally advances the accepted state/rank
  artifact schemas to `model_native_state_contract_v3` and
  `model_native_train_rank_reference_v3`; older artifacts cannot pass.
- Bound the exact eight-specialist partition and all active heads.
- Removed legacy Entry modes, anchor/neutral-bridge arguments and optional-head
  serving paths from the model, trainer, builder and strict bundle loader.
- Added exact smoke/candidate wrappers with immutable prerequisite bindings;
  no mutable discovery or generic trainer pass-through remains.
- Compacted and hardened the canonical smoke bundle audit. It strict-loads the
  bundle and proves state/meta/lock hashes, architecture, objective identity,
  exact 20-head and 23/75-fusion identity, learned-component movement and
  immutable validation/test evidence. Support, confusion counts and Wilson
  lower bounds are recomputed globally, per class and per declared context
  slice.
- Extracted neutral model-native replay primitives. Candidate replay consumes
  final model direction and records logits, supporting evidence and a
  unit-normalized label-horizon outcome for offline direction diagnostics only;
  it explicitly applies no position size and has no execution authority.
  Executable learned sizing requires a separate immutable sizing OOS,
  exact adopted-Exit replay and post-adoption runtime-parity chain. The current
  label-horizon sizing proof is diagnostic only; capital authority remains
  `BLOCK` until those two final bindings exist and live/paper can enforce
  `NO_ORDER` when any part is missing or red.
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

Focused contract suites and full test collection have been green at each
completed boundary. Re-run the complete verification after the concurrent
source cleanup settles.

## Remaining source boundary work

1. DONE 2026-07-17: repository-wide stale-reference/call-graph/duplicate-owner/
   contract-consistency audit completed (55 adversarially verified findings);
   zero-reachability deletions executed under explicit user approval; full test
   suite green (1341 pass / 5 skip / 0 fail). See DECISION_LOG 2026-07-17.
2. DONE 2026-07-17: contract hardening — the 305-field mandatory causal-layer
   registry PREFIX ORDER is validated at every manifest consumer; the five
   launch-JSON `required_*` partition constants are enforced against code
   constants; the 90-second Entry latency limit has one numeric owner.
3. Keep launch state `BLOCK`. Stop before rebuild/train: either requires an
   explicit `--vedtak` and the complete immutable preflight chain.

## Rebuild runbook (the next admissible work)

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
  `model_native_train_rank_reference_v3`) and the 520-wide dataset
  (7 constant neutral-XGB bridge fields).

BLOCKER discovered 2026-07-18 (verified twice, repo + HEAD): the TRAIN-only
feature-ranker producer `entry_model_native_train_feature_ranker` does NOT
exist. Its interface is fully specified by the consumer
(`materialize_entry_model_native_seq513_signal_manifest_v1`, schema
`entry_model_native_train_feature_ranking_v1`) and its test fixture, but the
producer was never built. Without it there is no ranking JSON, hence no signal
manifest, hence the rebuild wrapper aborts (correctly fail-closed). Build the
ranker FIRST: reuse the builder's one-truth feature assembly (do not duplicate
it), deterministic TRAIN-only score with `score_descending` +
`feature_name_ascending` tie-break as declared, source/target sha256 bindings,
and validate output by round-trip through the manifest producer.
User vedtak `XAU_SEQ513_REBUILD_20260718_V1` is ISSUED and covers this chain.
RANKER STATUS 2026-07-18: BUILT
(`gx1/scripts/materialize_entry_model_native_train_feature_ranker_v1.py`, five
green tests incl. round-trip through the real manifest producer). Chain
execution then hit four consecutive fail-closed stops; the first three are
FIXED (missing producer -> built; candidate pool 146<174 -> GROUP_A/DIP_STRUCT
ctx recomputed via the builder's own `attach_group_a_dip_struct_ctx_columns`;
stale MTF cache -> must be rebuilt with `prebuild_multi_tf_cache_v2` current
version). Stop four is a DATA defect and blocks everything:

**CANONICAL TAPE DEFECT (verified 2026-07-18):** the M5 canonical tape carries
2375 rows with invalid OHLC geometry (close outside [low, high]), ALL in
2024-11-30 -> 2024-12-31, ~77% on closed-market Saturdays/Sundays (synthetic
bars with no canonical-M1 backing; M1 coverage that month is also thin,
~29,885 rows). Present in BOTH the July-16 event copy AND the live
daemon-maintained prebuilt (latent since Dec-2024). The 2026-07-17
`causal_no_fallback` contract correctly refuses it. Because
`FULL_PLUS_CTX_v3src.parquet` and `canonical_features_v2.parquet` were built
FROM the defective tape, a proper repair cascades: repair Dec-2024 M5 segment
from canonical M1 (drop unbacked closed-market bars, register gaps) -> rebuild
canonical_features_v2 -> rebuild FULL_PLUS_CTX -> then ranker -> manifest ->
preflight -> dataset build. The July-16 logs show the feature-rebuild steps
take minutes each. The LIVE prebuilt repair is a separate decision (it serves
the active Exit chain).

Ordered steps (each gate fail-closed; stop at first red):

1. User issues ONE vedtak ID (format `XAU_SEQ513_REBUILD_YYYYMMDD_Vn`); it is
   bound into rank NPZ, sidecar, build proof, state contract and all split
   manifests. Wrapper-only console values do not count.
   -> `XAU_SEQ513_REBUILD_20260718_V1` issued 2026-07-18 with the window
   proposal below confirmed.
2. `scripts/entry_next_edge_control.sh model-native-rebuild-preflight` with all
   explicit inputs/windows; preflight must pass before anything writes.
3. `scripts/rebuild_entry_model_native_seq513_dataset.sh --vedtak … ` with the
   full explicit argument set (runs capped 30G; produces fresh v3 rank
   reference, seq513 signal manifest, dataset, split manifests, liveness and
   pretrain audits).
4. Split-window PROPOSAL (must be confirmed in the vedtak; chronological,
   leakage-safe): history-start 2020-11-09T00:00:00Z; train
   2020-11-09 -> 2026-03-31; val 2026-04-01 -> 2026-04-30; test
   2026-05-01 -> 2026-06-14 (untouched tail of the source).
5. `model-native-smoke-manifest` -> `model-native-smoke-readiness` ->
   `model-native-smoke-train --vedtak … --dry-run` then `--execute`.
   Smoke acceptance ADDITIONALLY requires a non-degenerate FLAT rate on val
   and test (zero FLAT predictions is an automatic hard-red, as on
   2026-07-16) before any slice metric is even considered.
6. Candidate chain only after smoke PASS: trainability-readiness ->
   candidate-train -> calibration -> immutable prediction evidence ->
   unit-normalized replay -> the nine-item evidence list above.

Nothing in this runbook grants run authority by itself; the explicit vedtak
and green preflight do.

## Required evidence before Entry can open

1. Fresh exact train/val/test split manifests and seq513 datasets.
2. Full 513+142+5 field liveness and ordered-hash proof on all splits.
3. Feature, target, specialist, leakage and pretrain audits.
4. Immutable smoke recipe and an audited bundle with every head/specialist active.
5. Honest calibration and untouched validation/test direction, class, slice,
   pocket, context, path-quality and utility metrics with adequate support.
6. Immutable candidate prediction evidence and live-like replay including costs.
7. Learned-size calibration plus untouched TEST utility/exposure/drawdown and
   exact train/replay/serve sizing proof.
8. Train==serve parity on identical bars, including calibrated logits and
   journal fields.
9. Newest terminal evidence PASS, exact bundle/hash admission and explicit
   paper/live launch decision.

If any item is missing, malformed, stale, hash-mismatched or empirically red,
the result is `BLOCK`.

## Operational takeover

```bash
cd /home/andre2/src/GX1_ENGINE
bash scripts/gx1_handover.sh
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
