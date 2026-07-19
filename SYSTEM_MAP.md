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
        |     +-- 305 mandatory outputs from 10 causal layers
        |     +-- 174 deterministic TRAIN-only ranked fields
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
20 supervised evidence heads
        |
        v
exact 23-group / 75-dimension evidence fusion -> 128 hidden -> 3 logits
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
- the specialist surface is 305 mandatory code-owned causal-layer outputs plus
  174 deterministic TRAIN-only ranked fields;
- sequence length 96;
- 142 continuous + 5 categorical context fields;
- no bridge source and no anchor source.

This contract is the one exact owner of the 34 base, 142 continuous-context
and 5 categorical-context identities. Active Entry has zero imports from
`signal_bridge_v1` or `signal_bridge_v3`; those modules cannot supply,
reorder or repair Entry state.

The complete 479-field order is manifest-owned and hash-bound. The first 305
positions must exactly equal the immutable causal-layer registry; the final
174 come from one validated deterministic TRAIN-only ranking. It cannot be
reconstructed from a mutable registry at train or serve time.

`gx1/features/entry_model_native_feature_layers_v1.py` owns the ten mandatory
causal layers: trend/EMA, SMC/liquidity, structure/swing, momentum/flow,
session/regime, volatility/compression, chart geometry,
price-action/candles, support/resistance memory and MTF confluence.
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

Direction training includes the public three-class objective, MTF direction,
tail and slice behavior, utility margins/triad, trade/side hierarchy and
validity. Supporting objectives include path quality, MFE, tradability,
bad-path, clean-edge, survival, timing, tail/volatility, TF agreement,
trendline rail and position size.

The final direction layer is one exact learned fusion. Its ordered input is 23
evidence groups / 75 values, followed by `LayerNorm(75)`, a learned
`75 -> 128` projection, GELU and a learned `128 -> 3` projection. Immutable
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
MTF, eight-specialist, geometry and learned-size evidence. Missing, unexpected,
non-finite, inconsistent or retired overlay fields fail closed; consumers do
not fill or infer them.

`position_size_logit` and `position_size_pred` are mandatory learned evidence.
The sizing diagnostic additionally binds its calibration bytes, XAU instrument
constraints, account scenarios, monotone capacity transform, step-rounded
units and actual-1-unit/equal-total-allocation controls. This label-horizon
proof does not grant capital authority. Entry paper/live remains blocked until
a joint sizing-only replay binds the exact adopted active Exit stack and a
fresh post-adoption broker runtime-parity event. A historical fixed-1x
comparison cannot satisfy launch authority or rescue a failed proof.

## Dataset and launch ownership

`gx1/scripts/build_entry_v10_ctx_training_dataset_v3.py` is the canonical
Entry dataset builder. Its filename is historical; its accepted behavior is
model-native only. It must emit exact split manifests and all direction/path/
utility/sizing targets without invoking Entry XGB inference.

The rebuild wrapper, TRAIN-only rank-reference producer and dataset builder
all require the same explicit `--vedtak`. The rank NPZ embeds it, the rank
sidecar repeats it, and the state contract/build proof/split manifests bind it.
Missing or unequal IDs invalidate the chain; a wrapper-only console value is
not authorization provenance.

All retained OANDA backfill writers also validate an explicit `--vedtak`
before creating or modifying files. A missing or invalid decision fails before
side effects; backfill output never supplies Entry direction authority.

The feature-ranking JSON and its derived seq513 signal manifest are explicit
immutable inputs, never files selected by glob, mtime or lexical "latest".
Preflight, wrapper and builder revalidate the nested ranking lineage, vedtak,
source hash and exact requested TRAIN start/end. A ranking produced for a
different split cannot authorize a build even when its schema and 305+174
shape are otherwise valid.

`gx1/contracts/entry_model_native_train_launch_v1.py` binds explicit immutable
train/val/test manifests, data files, source tape, liveness, feature, target,
specialist, pretrain and readiness audits. It emits only an allowlisted recipe
to the smoke or candidate wrapper. Mutable `latest`, symlinks, missing hashes,
unlisted environment overrides and pre-existing output directories fail.

The only model-native train wrappers are:

- `scripts/run_entry_model_native_seq513_smoke_train.sh`;
- `scripts/run_entry_model_native_seq513_candidate_train.sh`.

They do not grant authority by themselves. A valid explicit `--vedtak` and all
bound prerequisite evidence are still required.

## Bundle, calibration and evidence ownership

`gx1/scripts/audit_entry_foundation_smoke_bundle_v1.py` is the canonical exact
bundle audit despite its historical filename. It strict-loads the state and
proves metadata/lock/state hashes, objective identity, dimensions, 20 active
heads, eight specialists, exact learned fusion, learned-component movement,
MTF/cross-TF/positional/FiLM/timeframe-scale wiring and immutable validation/
test behavior. Direction claims require exact support, confusion counts and
recomputed Wilson lower bounds globally, per class and in declared context
slices on both validation and test.

`gx1/scripts/fit_entry_direction_calibration_v1.py` may fit only the declared
calibration artifact. `gx1/contracts/immutable_event_authority_v1.py` and the
candidate evidence/readiness scripts bind events by content and lineage. A
mutable report path or a copied `PASS` decision is insufficient.

`gx1/execution/model_native_entry_replay_v1.py` owns neutral source-tape and
offline label-horizon primitives. Replay consumes the model's final direction;
it does not decide direction. Candidate replay records the calibrated logits,
decision, supporting heads, target/outcome alignment and a unit-normalized
price-path outcome. It is an offline direction diagnostic with
`position_size_applied=false`, not an order or capital simulation. Executable
learned sizing must be proven separately against its OOS controls, the exact
adopted active Exit replay and post-adoption runtime parity. Today the
label-horizon sizing result is diagnostic only: capital authority remains
`BLOCK` because the adopted-Exit and post-adoption runtime bindings are not yet
implemented. Once implemented, live/paper must emit `NO_ORDER` whenever that
sizing authority is missing or red.

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
.venv/bin/python -m json.tool PROJECT_STATE_xau_direction_launch.json
scripts/entry_next_edge_control.sh --help
```

`scripts/entry_next_edge_control.sh` is the single Entry control surface. It
must not expose direct legacy trainers, generic upstream pass-throughs or old
activation routes. `scripts/gx1_handover.sh` is the only handover script.

Current facts:

- source contracts and focused tests prove the intended exact architecture;
- no accepted fresh seq513 dataset/bundle/OOS result exists;
- no seq513 rebuild chain or training process is running;
- vedtak `XAU_SEQ513_REBUILD_20260718_V1` exists, but both July-19 rebuild
  attempts were terminated and invalidated after a reused feature-ranking
  TRAIN window (`2020-11-13..2026-03-31`) was found to mismatch the active
  TRAIN window (`2021-03-16..2026-03-31`);
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
- immutable historical selection-benchmark bytes and exact learned-probe OOT
  evidence are absent; obtaining both is the next empirical gate and the
  metadata run authorizes no rebuild, training or launch;
- Entry candidate, replay, paper/demo/live and promotion remain blocked.

## Pipeline- og ingredienskart (seq513-datakjeden)

Oppdatert 2026-07-19 for vedtak `XAU_SEQ513_REBUILD_20260718_V1`. Forsøkene ga
ingen godkjent datasettartefakt; kartet beskriver den herdede, påkrevde
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
          └─ modelrange = cv3 join 11 kol fra cf2 + radtrim (inline; se
                cv3_modelrange.provenance.json; 124 kol)
              └─ gx1.scripts.add_ctx_cont_columns_to_prebuilt
                    --prebuilt_parquet --output_parquet --raw_m5_parquet <7 år-parter>
                    --ctx-cont-dim 16 --ctx-cat-dim 5 --tape-root
                    → FULL_PLUS_CTX_v3src.parquet (207 kol) + manifester
cv3 ─ gx1.scripts.prebuild_multi_tf_cache_v2 --m5-prebuilt --out-dir
        → MULTI_TF_V2_CACHE/ (builder_version må matche HTF_V2_CACHE_BUILDER_VERSION)
FULL_PLUS + cache ─ materialize_entry_model_native_train_feature_ranker_v1
        --vedtak --source-parquet --mtf-cache-dir --history-start --train-start
        --train-end --out-dir  (checkpoint: EVENT/_ranker_checkpoint.npz)
        → ENTRY_MODEL_NATIVE_TRAIN_FEATURE_RANKING_<stamp>.json
  └─ materialize_entry_model_native_seq513_signal_manifest_v1 --feature-ranking-json --out --vedtak
      └─ entry_next_edge_control.sh model-native-rebuild-preflight
            <eksplisitte ranking-/manifeststier og øvrige flagg>
          └─ rebuild_entry_model_native_seq513_dataset.sh --vedtak
                <samme eksplisitte ranking-/manifestidentitet og øvrige flagg>
                (capped 30G)
                → dataset/*__HOLD_03B_{train,val,test}.parquet + DATASET_BUILD_PROOF.json
Kjede-driver: scripts/run_seq513_rebuild_chain_v1.sh --vedtak --event-root
--feature-ranking-json --signal-manifest --preflight-out-dir. Ranking må finnes;
manifeststien og preflight-mappen må være nye. Ingen resume, glob/mtime eller
leksikalsk latest. Telegram-ping er kun operasjonell status; validerte
split-manifester er terminal autoritet.
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
  ligger i konteksten → chunk-med-overlapp er FEIL tilnærming.
- ctx 13 ENTRY_SMART_DERIVED (smc_*_pressure, sr_*):
  `gx1/features/entry_smart_context.py::add_entry_smart_context_features`.
- ctx is_ASIA: builder ~linje 1799, `(session_id==0).astype(int8)`.
- ctx_cat 5: ctx-adderen emitterer; ORDERED_CTX_CAT_NAMES_V3.
- 305 obligatoriske specialist-felt: entry_model_native_feature_layers_v1 + de
  ni entry_*-lagmodulene; 84 kandidat-ekstra fra samme modulers
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
