# GX1 current system map

This is the current architecture map for XAUUSD Entry direction. Retired
rule-gated, anchored and Entry-RL flows are not documented as alternatives
because they have no Entry authority.

## Decision flow

```text
canonical XAU tapes + derived market state
        |
        v
exact split manifests and chronological leak-safe rows
        |
        +-- 34 genuine base price-state fields
        +-- 479 specialist fields
        |     +-- 316 mandatory outputs from 11 causal layers
        |     +-- 163 deterministic TRAIN-only ranked fields
        +-- 142 continuous context fields
        +-- 5 categorical context fields
        +-- M5/M15/H1/H4/D1 sequences, length 96
        |
        v
full-field liveness + field order/hash + feature/target/specialist audits
        |
        v
eight specialist encoders + temporal/MTF/cross-TF/FiLM fusion
        |
        v
22 supervised evidence heads
        |
        v
exact 26-group / 96-dimension evidence fusion -> 128 hidden -> 3 logits
        |
        v
immutable calibration on declared calibration data
        |
        v
shared exact runtime evidence contract
        |
        +-- calibrated logits [LONG, SHORT, FLAT]
        +-- hierarchy/path/utility/MTF/specialist evidence
        +-- mandatory learned position_size evidence
        |
        +-- argmax is the only direction authority
        +-- learned sizing is separately calibrated and proof-bound
        |
        v
immutable prediction evidence -> unit-normalized direction replay -> pocket/slice audits
                                      |
                                      +-- separate sizing OOS + exact active-Exit replay
                                          + post-adoption runtime parity
        |
        v
launch contract ALLOW only after newest complete evidence passes
```

Any missing or inconsistent edge in this flow stops the path. There is no
secondary Entry model, hand-written filter, default decision or stale bundle
selector.

## Input ownership

`gx1/contracts/entry_model_native_signal_v1.py` owns the exact static Entry
shape and forbids the seven retired bridge fields:

- contract mode `xau_seq513_model_native_direction_v1`;
- direction mode `model_native`;
- 34 base + 479 specialist = 513 signal fields;
- the specialist surface is 316 mandatory code-owned causal-layer outputs plus
  163 deterministic TRAIN-only ranked fields;
- sequence length 96;
- 142 continuous + 5 categorical context fields;
- no bridge source and no anchor source.

This contract is the one exact owner of the 34 base, 142 continuous-context
and 5 categorical-context identities. Active Entry has zero imports from
`signal_bridge_v1` or `signal_bridge_v3`; those modules cannot supply,
reorder or repair Entry state.

The 21-field V1 context prefix is likewise partitioned only here: source6,
micro5, swing5 and session5. `micro_structure_v1` and `swing_structure_v1`
own the causal formulas used by prebuild, dataset build and serving. During the
dataset join canonical-v2 exclusively owns base30 + context-v2-19, while the
exact source prebuilt exclusively owns context-v3-5 + regime-source15 + raw
volume. A duplicated or missing owner field is a hard failure; there is no
source/canonical preference fallback.

The complete 479-field order is manifest-owned and hash-bound. The first 316
positions must exactly equal the immutable causal-layer registry; the final
163 come from one validated deterministic TRAIN-only ranking. It cannot be
reconstructed from a mutable registry at train or serve time.

`gx1/features/entry_model_native_feature_layers_v1.py` owns the eleven mandatory
causal layers: trend/EMA, SMC/liquidity, structure/swing, momentum/flow,
session/regime, volatility/compression, chart geometry,
price-action/candles, support/resistance memory, MTF confluence and exact M5
EMA50/200 state/cross evidence.
`gx1/features/entry_specialist_feature_groups_v1.py` routes the complete
479-field surface into the exact eight-way learned specialist partition:

1. structure/swing;
2. SMC/liquidity;
3. trend/EMA;
4. volatility/compression;
5. momentum/flow;
6. session/regime;
7. chart geometry;
8. price-action/candles.

Features may inform several learned interactions, but each emitted field has
one exact ordered input identity. Genuine trend/session evidence is mandatory;
only disconnected direction filters are retired.

This means every genuine form of structure, multi-timeframe trend, liquidity,
volatility, momentum, session/regime, chart geometry and candle/price action is
retained in the eight-specialist learned path. Path quality and utility are
supervised evidence in the same fused model. No downstream component may
recreate these as an independent veto, flip, threshold or substitute direction.

"Full stack" means complete causal coverage with exact liveness and influence,
not an unlimited pile of correlated indicator aliases. A proposed trend or
session field belongs only if it is available at the decision timestamp,
non-degenerate on every declared split, uniquely identified and demonstrably
connected to the learned path. Redundant, stale or future-leaking variants make
the system less robust and are rejected rather than counted as extra evidence.

`gx1/contracts/entry_full_input_liveness_v1.py` validates every field on train,
validation and test. It rejects unexpected constants, non-finite values,
insufficient categorical support, forbidden fields, wrong order and ATR/OOD
drift outside the declared policy. There is no constant-pass-through allowlist.

## Model and objective ownership

`gx1/models/entry_v10/entry_v10_ctx_hybrid_transformer.py` owns the Entry
architecture. The active architecture consumes the exact model-native state,
uses temporal and multi-timeframe representations, learned timeframe scaling,
cross-timeframe attention, positional encoding, FiLM conditioning and eight
specialist encoders before producing the public direction surface and its
supporting heads.

`gx1/contracts/entry_model_native_readiness_v1.py` owns the exact specialist
and head declaration. `gx1/contracts/entry_model_native_training_objective_v1.py`
requires every advertised objective to have a positive loss weight. A head
cannot be retained as an unsupervised decoration.

`gx1/contracts/entry_model_native_aux_targets_v3.py` is the sole owner of the
schema-v4 46-column future-target surface and the exact 12-value turning-point
layout. For LONG, adverse-turn timing means the low before the favorable peak
(`BOTTOM`); for SHORT it means the high before the favorable trough (`TOP`).
The surface also includes nine spread-aware, full-counterfactual
LONG/SHORT/FLAT path-utility targets at K12/K48/K96.
`gx1/contracts/entry_model_native_offline_rl_v1.py` owns the schema-v2
action/horizon order, reward scaling, expectile and ranking math. Ambiguous
reward ties are excluded from ranking rather than silently assigned by action
order. There is no logged-behavior/AWR objective, Bellman backup, replay policy
or separate Entry-IQL runtime.

Direction training includes the public three-class objective, MTF direction,
tail and slice behavior, utility margins/triad, trade/side hierarchy and
validity. Supporting objectives include path quality, MFE, tradability,
bad-path, clean-edge, survival, timing, tail/volatility, TF agreement,
trendline rail and position size.

Internal contextual-bandit objectives directly regress `Q(s,a,K)` to all nine
counterfactual rewards, train `V(s,K)` toward the expectile of detached
`max_a Q`, and enforce a small reward-defined action-ranking margin.
`Adv(s,a,K)=Q(s,a,K)-V(s,K)` is derived exactly. Q, V and Advantage are learned
evidence in the final fusion; none is an independent direction selector.

The final direction layer is one exact learned fusion. Its ordered input is 26
evidence groups / 96 values, followed by `LayerNorm(96)`, a learned
`96 -> 128` projection, GELU and a learned `128 -> 3` projection. Immutable
calibration is applied after those raw `LONG/SHORT/FLAT` logits; exact argmax
remains the only public direction selection.

The position-size target is exactly `sigmoid((MFE-MAE)/(2*ATR_bps))`, with MAE
defined as a non-negative adverse magnitude. `FLAT` has a neutral training
target and zero executable units. Prediction is exported and journaled, but it
can become capital authority only through the separate immutable learned-size
calibration and OOS exposure/drawdown/parity contract. Missing or rejected
sizing proof emits no order; it is never converted to multiplier `1.0`.

## Runtime evidence ownership

`gx1/contracts/entry_model_native_runtime_evidence_v1.py` owns the complete
serve-time evidence schema. The same validator is called by the model-native
decision adapter, `TradeState` persistence/recovery, `TradeJournal` writes and
reloads, and the daily trade review. It requires exact calibrated direction,
trade/flat and side parity plus all hierarchy, path, utility, calibration,
MTF, eight-specialist, geometry, learned-size and exact Q/V/Advantage evidence.
It also validates `Adv=Q-V`. Missing, unexpected, non-finite, inconsistent or
retired overlay fields fail closed; consumers do not fill or infer them.

`position_size_logit` and `position_size_pred` are mandatory learned evidence.
The sizing diagnostic additionally binds its calibration bytes, XAU instrument
constraints, account scenarios, monotone capacity transform, step-rounded
units and actual-1-unit/equal-total-allocation controls. This label-horizon
proof does not grant capital authority. Entry paper/live remains blocked until
a full-TEST joint sizing-only replay binds every per-M1 HOLD-to-`EXIT_NOW` trace,
learned sizing is adopted, and a fresh post-adoption broker shadow
runtime-parity event passes. Strict finalizers and row-recomputing validators
now exist, but no real current-contract events do. Artifact guard
binds them to the accepted bundle and complete serve gates. A historical
fixed-1x comparison cannot satisfy launch authority or rescue a failed proof.

## Dataset and launch ownership

`gx1/scripts/build_entry_v10_ctx_training_dataset_v3.py` is the canonical
Entry dataset builder. Its filename is historical; its accepted behavior is
model-native only. It must emit exact split manifests and all direction/path/
utility/sizing targets without invoking Entry XGB inference.

The rebuild wrapper, TRAIN-only rank-reference producer and dataset builder
all require the same `--run-id`. The rank NPZ embeds it, the rank sidecar
repeats it, and the state contract/build proof/split manifests bind both
artifact hashes.
Missing or unequal IDs invalidate the chain. The ID is immutable provenance,
not an approval or direction authority.

All retained OANDA backfill writers also validate an explicit `--vedtak`
before creating or modifying files. A missing or invalid decision fails before
side effects; backfill output never supplies Entry direction authority.

The feature-ranking JSON and its derived seq513 signal manifest are explicit
immutable inputs, never files selected by glob, mtime or lexical "latest".
Preflight, wrapper and builder revalidate the nested ranking lineage, run ID,
source hash and exact requested TRAIN start/end. A ranking produced for a
different split cannot authorize a build even when its schema and 316+163
shape are otherwise valid.

`gx1/contracts/entry_model_native_train_launch_v1.py` binds explicit immutable
train/val/test manifests, data files, source tape, liveness, feature, target,
specialist, pretrain and readiness audits. It emits only an allowlisted recipe
to the smoke or candidate wrapper. Mutable `latest`, symlinks, missing hashes,
unlisted environment overrides and pre-existing output directories fail.

Both wrappers pass the exact TRAIN/VAL/TEST manifest and parquet paths to the
trainer, together with recipe-bound hashes. The trainer revalidates all six;
it has no dataset-directory glob, TRAIN-stem VAL/TEST inference or optional
split fallback.

`gx1/contracts/entry_dataset_split_artifacts_v1.py` owns the common foundation
audit boundary. For each TRAIN/VAL/TEST split it requires an explicit canonical
manifest path, manifest SHA-256 and parquet SHA-256; the parquet path may come
only from the hash-bound manifest. Feature, target and specialist audits emit
the normalized four-field identity, and smoke/adoption gates compare it with
the candidate manifest. Extra files in the dataset directory are inert.

The only model-native train wrappers are:

- `scripts/run_entry_model_native_seq513_smoke_train.sh`;
- `scripts/run_entry_model_native_seq513_candidate_train.sh`.

They do not grant authority by themselves. One valid `--run-id` and all bound
prerequisite evidence are required; evidence gates, not the ID, admit execution.

## Bundle, calibration and evidence ownership

`gx1/scripts/audit_entry_foundation_smoke_bundle_v1.py` is the canonical exact
bundle audit despite its historical filename. It strict-loads the state and
proves metadata/lock/state hashes, objective identity, dimensions, 22 active
heads, eight specialists, exact learned fusion, learned-component movement,
MTF/cross-TF/positional/FiLM/timeframe-scale wiring and immutable validation/
test behavior. Direction claims require exact support, confusion counts and
recomputed Wilson lower bounds globally, per class and in declared context
slices on both validation and test.

Foundation target audit schema v2 requires all 46 aux targets in every split,
including `time_to_mfe` and all Q targets. LONG/SHORT Q targets must be live,
FLAT must be exact zero reward, and every horizon must contain non-collapsed
unique LONG/SHORT/FLAT best actions. Prediction evidence schema v2 and smoke
bundle audit schema v3 then prove on both VAL and TEST that all 12 timing
outputs align with their targets, learned near-BOTTOM LONG and near-TOP SHORT
pockets meet immutable precision/support/Wilson floors, Q aligns with reward
targets, Q ranking selects the reward-best action, V tracks max-Q, and
`Advantage=Q-V`. These are audit-only evidence contracts; none is a live rule
or a second direction authority.

Smoke v3 distinguishes prediction/gate/component liveness from causal
direction influence and has no activation authority. Serve-parity v4 is the
later fail-closed influence owner: both exact input-family masking and encoder
hook ablation must move raw and calibrated class margins for all eight
specialists, while immutable VAL-mean replacement of each exact fusion slice
must move both surfaces for all 26 groups on deterministic TEST states. Exact
zero-mask ablations impose the same requirement on continuous/categorical
context and M5/M15/H1/H4/D1. Layout, reference, bundle and lock hashes are
admission-bound; an old event or one passive input/group blocks launch.

`gx1/scripts/fit_entry_direction_calibration_v1.py` may fit only the declared
calibration artifact. `gx1/contracts/immutable_event_authority_v1.py` and the
candidate evidence/readiness scripts bind events by content and lineage. A
mutable report path or a copied `PASS` decision is insufficient.

Selective-edge prediction carries exact VAL/TEST manifest and parquet
identities into its immutable report. Candidate replay and serve parity consume
those report-bound identities directly. Learned sizing provenance does the
same for TRAIN/VAL fitting and TEST diagnostics; it neither scans the dataset
directory nor grants sizing any direction authority.

`gx1/execution/model_native_entry_replay_v1.py` owns neutral source-tape and
offline label-horizon primitives. Replay consumes the model's final direction;
it does not decide direction. Candidate replay records the calibrated logits,
decision, supporting heads, target/outcome alignment and a unit-normalized
price-path outcome. It is an offline direction diagnostic with
`position_size_applied=false`, not an order or capital simulation. Executable
learned sizing must be proven separately against its OOS controls, the exact
adopted active Exit replay and post-adoption runtime parity. The source
contracts/finalizers now require a file-bound contiguous per-M1 Exit trace,
row-recomputed bid/ask results, exact registry identity, adopted bundle hashes
and fresh broker-shadow parity with zero orders. No current real chain has
passed, so capital authority remains `BLOCK`; live/paper emits `NO_ORDER`
whenever that sizing authority is missing or red.

`gx1/contracts/entry_model_native_adaptation_drift_v1.py` owns the only market
adaptation trigger. It row-recomputes immutable same-bundle candidate-TEST and
settled broker-shadow probabilities/outcomes, global LONG/SHORT edge and
direction-specific session/volatility slices. It can return only `STABLE` or
terminal `DRIFT`; it never selects direction, trains or submits an order.

`gx1/scripts/verify_entry_replay_readiness_v1.py` schema v2 binds current
bundle bytes and hands off with zero activation authority to
`gx1/contracts/entry_model_native_adaptation_lifecycle_v1.py`. The lifecycle
owns initial admission, monitor refresh, drift block, offline challenger,
zero-order shadow, explicit promotion and prior-incumbent rollback.
`gx1/contracts/entry_model_native_adaptation_shadow_v1.py` is the mandatory
promotion-comparison owner: identical paired paths, both exact argmax outputs,
bid/ask-recomputed outcomes, absolute challenger side edge and positive
lower-95% paired improvement globally and per supported direction/context.
Newest
terminal evidence wins; failed drift, replay or transition refresh invalidates
older green evidence. `gx1_guards/artifacts.py` accepts launch `ALLOW` only from
a fresh activating lifecycle event cross-bound to the exact accepted bundle,
serve gates, joint Exit proof and sizing runtime parity. No real lifecycle
chain exists; current launch remains `BLOCK`.

## Evidence retention and cleanup ownership

`gx1/contracts/evidence_retention_v1.py` is the byte-identity and authority
contract for destructive evidence cleanup. `gx1/scripts/cleanup_gx1_evidence_v1.py`
is the only admitted `GX1_DATA` deletion route. It accepts exact disjoint leaf
targets only; pins the canonical artifact registry, XAU launch contract and
deletion-incident record; writes a per-entry JSONL byte/topology manifest; and
forbids exclusions, symlinks and mount crossings. Immutable plan and separate
approval events precede explicit execution. Execution atomically stages each
target inside a new same-filesystem wrapper, proves the staged inventory again,
then writes durable staged/terminal evidence. Repository source cleanup remains
a reviewed source change and cannot use this data-deletion route.

`PROJECT_STATE_entry_iql_delete_incident.json` is the compact incident record
for the 2026-07-07 exclusion-path failure. Salvaged metadata is diagnostic only:
it contains no original row/model bytes and has no direction, comparison or
launch authority.

## Serving boundary

`gx1/execution/v12_model_native_state_live.py` builds the exact serve state.
`gx1/execution/v12_smart_entry_live.py` is a historical filename for the
model-native decision adapter; it loads only a launch-admitted bundle and
validates the shared exact runtime evidence contract before returning a result.
`gx1_guards/artifacts.py` requires both the artifact registry and
`PROJECT_STATE_xau_direction_launch.json` to agree on the exact bundle and
metadata hash. Current launch state is `BLOCK`, so resolution must fail.

Serving must reproduce the same ordered features, normalization, timeframe
alignment, architecture, calibration and final logits as immutable replay on
identical bars. A parity mismatch blocks launch.

Live direction must never be reconstructed from specialist votes, utility,
confidence, session, trend or a threshold. Those are learned evidence inside
the model, not downstream authorities.

`gx1/execution/v12_pipeline.py` owns immutable Entry freshness. It accepts
exactly 96 rows ending at the latest closed M5 bar. That row becomes available
five minutes after its bar-start; the decision must occur within the next 90
seconds, and canonical cutoff age may not exceed 390 seconds. These limits are
not environment-configurable. A missing/wrong window, older row or exceeded
limit raises structured unavailability and emits no `LONG`, `SHORT` or `FLAT`;
there is no cached-row substitution or backlog execution. Exit retains its
separate admitted freshness semantics.

The same fail-closed rule applies before inference to the 142+5 context and
session identity. Missing, invalid or session-inconsistent values, including a
fabricated ASIA flag, return
`MODEL_NATIVE_ENTRY_CONTEXT_NO_DIRECTION`; the runtime never manufactures
context, invokes a bridge or translates the failure to synthetic `FLAT`.

## Exit boundary

Exit is separately retained. Active Exit V3/Exit-IQL may still consume its
contracted XGB state. Entry's removal of XGB anchors and Entry-IQL does not
authorize changing Exit math, M1 cadence, artifacts or operating point.
Shared offline primitives must have neutral or Exit-owned names and must not
resurrect an Entry-IQL authority.

The retained V3 XGB bridge is an Exit-only owner of real ordered 7/41-field
validation for two active Exit consumers. Its import and order checks fail
closed and are intentionally not deleted as Entry residue. Conversely, the
retired Entry-IQL record in `PROJECT_STATE_artifacts.json` has `path=null` and
status `RETIRED_ARTIFACT_ABSENT`; it cannot be resolved as a fallback.

## Control and current status

Use:

```bash
bash scripts/gx1_handover.sh
bash scripts/gx1_handover.sh --check
.venv/bin/python -m json.tool PROJECT_STATE_xau_direction_launch.json
scripts/entry_next_edge_control.sh --help
```

`scripts/entry_next_edge_control.sh` is the single Entry control surface. It
must not expose direct legacy trainers, generic upstream pass-throughs or old
activation routes. `scripts/gx1_handover.sh` is the only handover script.

Current facts:

- source now has one chain-owned ranker/dataset path, a host-wide exclusive
  capped-job lock, immutable bounded Group-A chunks, one exact checkpoint
  retry and schema-v3 immutable terminal events; no fresh full execution has
  yet proven this repair;
- 2026-07-21 V3 has fresh ranking/manifest/preflight lineage but is
  non-authoritative failure evidence: the builder stopped after canonical join
  before Group-A completion and did not write terminal status;
- 2026-07-21 V4 is also non-authoritative failure evidence: transient service
  execution had no user bus, and the restored scoped runner lost the fresh
  rank process before any ranking artifact or checkpoint existed;
- Group-A output semantics must remain exact; reducing workers did not cure
  this process/terminal-status failure, so recovery needs bounded
  checkpointing and terminal-status durability rather than a direction
  fallback;

- source contracts and focused tests prove the intended exact architecture;
- no accepted fresh seq513 dataset/bundle/OOS result exists;
- no seq513 rebuild chain or training process is running;
- run lineage `XAU_SEQ513_REBUILD_20260718_V1` exists, but both July-19 rebuild
  attempts were terminated and invalidated after a reused feature-ranking
  TRAIN window (`2020-11-13..2026-03-31`) was found to mismatch the active
  TRAIN window (`2021-03-16..2026-03-31`);
- invalidated V1/V2/V3/V4 lineages are historical failure evidence and cannot
  be reused; the next chain must allocate a wholly fresh immutable run ID;
- no rebuild process is running; partial event artifacts have no authority,
  and schema-v2 `CHAIN_STATUS.json` terminally records `RED` with reason
  `FEATURE_RANKING_TRAIN_WINDOW_MISMATCH` and bound hashes;
- zero-reachability Entry adapters, critics, duplicate journal schemas,
  detached feature modules, manual sizing modules and stale research launchers
  have been deleted rather than retained as alternatives;
- no practical-precision or trading-edge claim exists without new immutable OOS
  and live-like proof;
- the report-only abstention metadata run is
  `BLOCK_ABSTENTION_EMPIRICAL_GATE`: FLAT labels are balanced on TRAIN
  (`1400/4095`, `34.19%`), validation (`530/1536`, `34.51%`) and TEST
  (`516/1536`, `33.59%`) and active FLAT/utility/margin weights are positive,
  but it read zero parquet and produced no learned-probe evidence;
- historical selection-benchmark bytes are absent and cannot be a satisfiable
  pre-rebuild gate. The old report-only abstention verifier/control route is
  deleted; fresh candidate TEST rows must instead pass the current immutable
  proxy comparison plus absolute OOT/cost/live-like gates;
- Entry candidate, replay, paper/demo/live and promotion remain blocked.

## Pipeline- og ingredienskart (seq513-datakjeden)

Oppdatert 2026-07-21 etter at V1/V2/V3/V4 ble ugyldiggjort. Ingen av forsøkene
ga en godkjent datasettartefakt; kartet beskriver den herdede, påkrevde
artefakt-DAG-en og kolonne-eierskapet. Les dette FØR du rg-jakter i builderen.

### Artefakt-DAG (produsent → output)

```text
kanonisk M1 bid/ask  GX1_DATA/data/oanda/canonical/xauusd_m1_bid_ask__CANONICAL/year=*/
  └─(repair ved tape-defekt) gx1/scripts/repair_m5_tape_dec2024_from_m1_v1.py
        --vedtak --m5-tape-root --m1-tape-root --out-root → EVENT/m5_tape_repaired_*/
kanonisk M5 bid/ask  .../xauusd_m5_bid_ask__CANONICAL/year=*/  (OANDA-native; IKKE M1-aggregert:
        high/low == M1-agg eksakt, open/close/volume kan avvike grense-ticks <=0.06)
  └─ gx1.scripts.materialize_build_canonical_features_v2  --m5-root --out-path
        → canonical_features_v2.parquet
      └─ gx1.scripts.materialize_canonical_v3_augment  --input --output-dir
            → cv3/ (113 kol = cv2 − 12 redundante + 6 nye)
          └─ gx1.scripts.materialize_cv3_modelrange_v1 --run-id --cv3
                --canonical-v2 --out (eksakt radakse, join 11 kol fra cf2,
                eksplisitt vindustrim og hashbundet provenance; 124 kol)
              └─ gx1.scripts.add_ctx_cont_columns_to_prebuilt
                    --prebuilt_parquet --output_parquet --raw_m5_parquet <7 år-parter>
                    --tape-root  (eksakt ctx16 + session5/cat5; ingen alternative dimensjoner)
                    → FULL_PLUS_CTX_v3src.parquet (208 kol; aktiv v2-kontrakt) + manifester
cv3 ─ gx1.scripts.prebuild_multi_tf_cache_v2 --m5-prebuilt --out-dir
        → MULTI_TF_V2_CACHE/ (builder_version må matche HTF_V2_CACHE_BUILDER_VERSION)
reparert tape + cv2 + cv3 + modelrange + cache + FULL_PLUS
  └─ gx1.scripts.audit_seq513_source_cascade_v1 --run-id --event-root --out
        → fersk hashbundet SOURCE_CASCADE_PROOF.json; alle self-paths må være event-lokale
FULL_PLUS + cache ─ scripts/run_seq513_rebuild_chain_v1.sh
      └─ materialize_model_native_train_rank_reference_v2
        --run-id --source-parquet --history-start --fit-start --fit-end --out
        → model_native_train_rank_reference_v4.npz + sidecar
      └─ materialize_entry_model_native_train_feature_ranker_v1
        --run-id --source-parquet --mtf-cache-dir --rank-reference-npz
        --history-start --train-start --train-end --out
        (bounded Group-A chunks + EVENT/_ranker_checkpoint.npz)
        → ENTRY_MODEL_NATIVE_TRAIN_FEATURE_RANKING_<stamp>.json
  └─ materialize_entry_model_native_seq513_signal_manifest_v1 --feature-ranking-json --out --run-id
      └─ entry_next_edge_control.sh model-native-rebuild-preflight
            <eksplisitte ranking-/manifeststier og øvrige flagg>
          └─ rebuild_entry_model_native_seq513_dataset.sh --run-id
                <samme eksplisitte ranking-/manifestidentitet og øvrige flagg>
                (capped 30G)
                → dataset/*__HOLD_03B_{train,val,test}.parquet + DATASET_BUILD_PROOF.json
Kjede-driver: scripts/run_seq513_rebuild_chain_v1.sh --run-id --event-root
--feature-ranking-json --signal-manifest --preflight-out-dir. Rankingstien,
manifeststien og preflight-mappen må være nye; kjeden produserer og binder
rankingen selv. Ingen inferred resume, glob/mtime eller leksikalsk latest.
Kun én eksakt hash-bundet checkpoint-retry er tillatt etter capped feil.
Telegram-ping er kun operasjonell status; validerte split-manifester er
terminal autoritet.
```

### Kolonne-/feature-eierskap (base 34 + ctx 142)

- Base 30/34: kilde-parquet direkte. Base 4 volum (vol_z_20, vol_ratio_5_20,
  vol_pct_96, signed_vol_z_20): `gx1/features/volume_features.py::add_volume_features`.
- ctx 62 kilde-bårne: FULL_PLUS direkte.
- ctx ~80 GROUP_A_PARITY + DIP_STRUCT (dist_to_*, dip_*, struct_*, atr_ratio_*,
  vol_pct_*_1yr): `augment_forward_outcome_v2` via `build_attach_context` +
  `compute_attach_rows` + `finalize_attach_columns` (attach_group_a… er
  komposisjonen). KOST: 85 ms/rad serielt; warmup 13 439 rader (~4 mnd,
  trimmes). Parallellisér ALLTID som i rankerens `_attach_group_a_parallel`:
  full-serie-kontekst én gang + rad-loop over workers. 1-års-persentilene
  ligger i konteksten → chunk-med-overlapp er FEIL tilnærming. De eksakte
  radområdene checkpointes uten overlapp i 4096-raders immutable NPZ-chunks;
  manifestet binder frame-, MTF-, felt- og run/window-identitet.
- ctx 13 ENTRY_SMART_DERIVED (smc_*_pressure, sr_*):
  `gx1/features/entry_smart_context.py::add_entry_smart_context_features`.
- ctx is_ASIA: builder ~linje 1799, `(session_id==0).astype(int8)`.
- ctx_cat 5: ctx-adderen emitterer; ORDERED_CTX_CAT_NAMES_V3.
- 316 obligatoriske specialist-felt: entry_model_native_feature_layers_v1 + de
  ni entry_*-lagmodulene, inkludert 11 eksakte M5 EMA50/200-felt; kandidat-
  ekstra fra samme modulers
  *_FEATURE_NAMES-konstanter; alt beregnes av builderens
  `_build_inline_seq_structure_extension` (~linje 674) som krever
  base+ctx142+cat5 i rammen.

### Builder-linjekart (build_entry_v10_ctx_training_dataset_v3.py)

~594 ctx-gate (CTX6CAT6, v1) · ~628 manifest-kontrakt · ~674 inline-extension ·
~1799 is_ASIA · ~1835 ctx-navn V3 · ~1982 df_ctx_cont · ~2096 merged3-assembly ·
~2115 cv2-lasteliste · ~2244 GROUP_A-attach (krever env
GX1_V10_MULTI_TF_V2_CACHE_DIR) · ~2330 smart-context · ~2337 ctx-komplett-sjekk ·
~3440 args.

### Kjente feller (verifisert 2026-07-18/19)

- July-19-forsøkene gjenbrukte en feature-ranking med TRAIN-start 2020-11-13
  mot aktiv TRAIN-start 2021-03-16. Den gamle preflighten sjekket ikke det
  nestede vinduet og ga derfor falsk GREEN. Forsøkene er terminert/ugyldige;
  schema-v2 `CHAIN_STATUS.json` er nå terminal `RED` med årsak
  `FEATURE_RANKING_TRAIN_WINDOW_MISMATCH` og bundne hasher.
- Des-2024-tapedefekten (2375 M5-rader, close utenfor [low, high], syntetiske
  helgebarer) finnes FORTSATT i kanonisk M5-rot og live-prebuilt; kun
  event-kopien er reparert. Kanonisk/live-reparasjon er en egen åpen beslutning.
- Skall-cwd kan resettes mellom kall: alltid `cd /home/andre2/src/GX1_ENGINE &&`
  først (rg gir ellers stille tomme treff); capped_run arver cwd og
  `python -m gx1...` krever repo-cwd.
- Aktiv vinduskontrakt er history-start 2021-01-05, TRAIN
  2021-03-16..2026-03-31, VAL 2026-04-01..2026-04-30 og TEST
  2026-05-01..2026-06-14T23:55. Rankingens TRAIN-vindu må matche eksakt.

Update this map whenever ownership or the active call graph changes. Remove
obsolete facts instead of appending a second historical architecture.
