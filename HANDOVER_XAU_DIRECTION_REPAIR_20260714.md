# XAUUSD model-native direction handover

Updated 2026-07-19. This is the only GX1 handover document. Run
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

**BLOCK.** There is no accepted fresh seq513 dataset/bundle and no launch
authority. Candidate, replay, paper/demo/live and promotion remain closed.

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

The empirical verifier now rejects evidence-only pass-throughs. Historical
benchmark JSON must be the exact path/SHA registered under retired Entry-IQL,
and learned TEST rows must match an exact hash-bound newest prediction report,
predictions parquet, bundle and dataset row-for-row in UTC and model direction.
This hardening does not create either missing artifact or authorize a run.

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
- exactly 22 active heads feeding one ordered learned 26-group/96-value fusion
  (`LayerNorm(96) -> Linear(128) -> GELU -> Linear(3)`);
- full-counterfactual LONG/SHORT/FLAT Q targets and expectile-V at K12/K48/K96,
  with exact Q/V/Advantage export and no separate IQL policy;
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
- Closed the July-19 feature-ranking lineage gap: preflight, wrapper and builder
  require explicitly named ranking/manifest artifacts and revalidate their
  vedtak, source hash and exact TRAIN start/end. A manifest ranked on a
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
- Removed the last active Entry imports from the V1/V3 signal bridges and made
  the model-native signal contract the one exact base/context owner. The
  retained V3 XGB bridge remains Exit-only: it owns real ordered 7/41-field
  validation for two Exit consumers and fails closed on import/order mismatch.
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
2. DONE 2026-07-17: contract hardening — the 305-field mandatory causal-layer
   registry PREFIX ORDER is validated at every manifest consumer; the five
   launch-JSON `required_*` partition constants are enforced against code
   constants; the 90-second Entry latency limit has one numeric owner.
3. Keep launch state `BLOCK`. The metadata-only abstention probe is not the
   required diagnostic. First bind immutable historical selection-benchmark
   bytes and produce exact learned-probe OOT evidence at comparable coverage.
   Only a green empirical probe may justify returning to the hardened rebuild
   chain with newly matched ranking/manifest inputs; training remains closed
   until an accepted dataset and all downstream gates exist.

## Rebuild runbook (contingent on a green abstention probe)

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
   its scores and derived 305+174 manifest have no admission authority.
2. `gx1/scripts/repair_m5_tape_dec2024_from_m1_v1.py` — event-local repair of
   the Dec-2024 canonical M5 geometry defect (2375 impossible rows -> 0) from
   the clean canonical M1; convention proof (high/low EXACT vs M1 aggregation,
   open/close boundary-tick tolerance documented); 5757 bars rebuilt, 3459
   synthetic closed-market bars dropped; immutable REPAIR_MANIFEST.
3. Full source cascade rebuilt on the repaired tape: canonical_features_v2 ->
   cv3 -> cv3_modelrange (provenance sidecar) -> MULTI_TF_V2_CACHE (current
   builder version) -> FULL_PLUS_CTX (207 cols, column-identical to July-16).
4. `scripts/run_seq513_rebuild_chain_v1.sh` — fail-closed chain driver
   (existing explicit ranking -> fresh manifest -> fresh preflight -> fresh
   build), with immutable artifact inputs rather than glob/lexical-latest
   selection or inferred resume. Telegram ⚙️/🔴/✅ and process-watch output are
   operational signals only; terminal authority still requires all three
   validated split manifests.
5. `attach_group_a_dip_struct_ctx_columns` factored into
   build_attach_context / compute_attach_rows / finalize_attach_columns plus
   `attach_group_a_dip_struct_ctx_columns_parallel` in the owner module —
   ONE full-series context fanned over 12 fork workers (exact by
   construction; serial spot-check). Serial cost measured 85.4 ms/row (~9 h);
   parallel ~1 h. Builder and ranker share it.

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
HISTORY_START=2021-01-05 < TRAIN_START=2021-03-16 (GROUP_A warmup 13,439 rows
-> 277 clean rows >= 95 required by seq_len 96) <= TRAIN_END=2026-03-31 <
VAL 2026-04-01..04-30 < TEST 2026-05-01..2026-06-14T23:55 (source's exact
last bar).

The old preflight did not actually prove this end to end: it accepted a nested
feature ranking beginning `2020-11-13`. The hardened chain must regenerate and
bind a ranking whose TRAIN start/end exactly equal `2021-03-16` and
`2026-03-31`; a green status from the invalid attempts cannot be inherited.

### Open decisions / next work

1. No rebuild or training is in flight. Before another heavy rebuild, bind the
   immutable historical selection-benchmark bytes and run the exact learned
   model-native abstention probe OOT. The current metadata-only result read no
   parquet and cannot satisfy this gate. If and only if that probe passes,
   regenerate the feature ranking and signal manifest for the exact active
   TRAIN window and rerun the hardened preflight/build chain. Smoke needs its
   own vedtak; zero FLAT predictions remains hard-red by definition
   (DECISION_LOG 2026-07-17 abstention criterion).
2. Canonical M5 root AND live prebuilt still carry the Dec-2024 defect (only
   the event copy is repaired) — separate decision; live Exit serves on it.
3. Exit env-softeners (3 audit MEDIUMs), ctx v1/v3 dual-owner in the builder
   (analyze before NEXT rebuild), CI replacement, gx1/scripts sorting,
   hashing-helper consolidation — post-smoke backlog.
4. Two commits may await push (permission prompts): check `git status -sb`.

Ordered steps (each gate fail-closed; stop at first red):

1. Do **not** reuse invalidated vedtak `XAU_SEQ513_REBUILD_20260718_V1`. Obtain
   a new explicit rebuild vedtak ID after the abstention-baseline decision; it
   must be bound into rank NPZ, sidecar, build proof, state contract and all
   split manifests. Wrapper-only console values do not count.
2. Materialize a fresh feature-ranking JSON for the exact active TRAIN window.
   Never discover it through a directory glob or lexical/mtime "latest"
   selection.
3. Invoke `scripts/run_seq513_rebuild_chain_v1.sh` with explicit `--vedtak`,
   `--event-root`, `--feature-ranking-json`, `--signal-manifest` and
   `--preflight-out-dir`. The ranking must already exist; the manifest path and
   preflight directory must be fresh. The driver creates and revalidates the
   manifest, preflight, v3 rank reference, dataset and split/audit outputs and
   never resumes inferred debris.
4. Accept the rebuild only from the driver's terminal validated split
   manifests. Console output, Telegram status, partial files and an earlier
   preflight cannot substitute for them.
5. Exact vedtak-bound window: history start `2021-01-05T00:00:00Z`; TRAIN
   `2021-03-16T00:00:00Z..2026-03-31T23:59:59Z`; validation
   `2026-04-01..2026-04-30`; TEST `2026-05-01..2026-06-14T23:55:00Z`.
6. `model-native-smoke-manifest` -> `model-native-smoke-readiness` ->
   `model-native-smoke-train --vedtak … --dry-run` then `--execute`.
   Smoke acceptance ADDITIONALLY requires a non-degenerate FLAT rate on val
   and test (zero FLAT predictions is an automatic hard-red, as on
   2026-07-16) before any slice metric is even considered.
7. Candidate chain only after smoke PASS: trainability-readiness ->
   candidate-train -> calibration -> immutable prediction evidence ->
   unit-normalized replay -> the nine-item evidence list above.

Nothing in this runbook grants run authority by itself; the explicit vedtak
and green preflight do.

## Required evidence before Entry can open

Before the rebuild/training sequence can resume, immutable historical
selection-benchmark bytes and exact learned abstention-probe OOT evidence must
pass the `BLOCK_ABSTENTION_EMPIRICAL_GATE`; metadata and label counts do not.

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
