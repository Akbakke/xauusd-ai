# GX1 XAUUSD full-pipeline audit — 2026-07-23

## Decision

`BLOCK`. This is a source/data/model/runtime audit, not market-edge evidence.
V24 and V7 are immutable failure evidence. They must not be reused for another
training run, candidate, replay, paper/demo/live launch or promotion.

Three independent read-only audits covered:

1. XAU source, feature, target, split and MTF construction;
2. model routing, all learned heads, losses, sampling, metrics and checkpoint
   admission;
3. bundle loading, serving, replay, shadow, launch authority and handover.

The initial audits changed no file and did not interfere with V7. Their
repairs were then implemented in source; no dataset rebuild or training run
was started.

## Source-repair checkpoint — 2026-07-23T15:43:37Z

The following findings are repaired in source and regression-tested, but do
not rehabilitate V24/V7 and are not empirical edge evidence:

- all six dip-MFE targets preserve signed negative outcomes; MAE remains
  non-negative and the aux-target contract is versioned forward;
- selected-side bad-path penalizes LONG on LONG rows and SHORT on SHORT rows,
  with mirrored loss/gradient proof and FLAT fail-closed validation;
- clean-edge/survival target rates and positive weights use the same
  bidirectional selector semantics as TRAIN/VAL loss;
- the slice-balanced sampler visits every selected row exactly once per epoch,
  without replacement or hidden padding;
- the trainer itself requires the exact 162-value recipe, rejects ambient
  `ENTRY_*`/`GX1_*` controls and binds the external M5 source path/hash through
  launch env, all split manifests and state contracts;
- MTF construction has one causal V2 cache identity and no V1/double-build
  branch;
- all 22 active heads now require epoch-wide output/target liveness and
  class-centered influence through the final learned 96→128→3 fusion before
  checkpoint admission;
- raw-bps dip, forecast, tail-risk and volatility targets are normalized to
  one explicit 20-bps model unit before entering the shared fusion contract;
- all 513 signals, 142 continuous context values and five MTF 25-field
  surfaces use one ordered immutable robust TRAIN-fit normalization contract.
  Fit consumes the complete physical TRAIN population before sampling,
  preserves binary/categorical semantics, binds every field/statistic and
  per-TF causal-row hash, and records zero VAL/TEST fit rows;
- all 142 continuous and five categorical context fields have exactly one of
  eight specialist owners. Family-specific projections enter their specialist
  tokens before cross-attention; the current-bar alias set is derived from the
  actual ordered signal manifest and must remain bit-identical rather than
  being assumed to contain 82 fields;
- tradable, LONG and SHORT conditional bad-path/clean-edge/survival metrics
  plus incremental lift over the tradable baseline are checkpoint evidence;
- the five-timeframe V2 disk cache is mandatory for admitted training and
  binds the exact source M5 bytes, ten component arrays, sizes, hashes,
  11-file inventory and aggregate cache identity;
- trained, calibrated and sizing-finalized bundles are built in hidden sibling
  staging directories, carry one exact commit manifest, pass strict load
  before publication and use `renameat2(RENAME_NOREPLACE)` plus `fsync`;
- immutable JSON events are fully written and fsynced under a hidden staging
  name before atomic no-replace publication;
- launch approval is no longer accepted from arbitrary environment text.
  Future ALLOW state must bind the newest immutable one-time approval to the
  complete launch-state hash and exact bundle commit; the runner revalidates
  the unchanged launch/registry lease before every new exposure;
- joint Entry-sizing/Exit evidence now inventories and hash-binds every regular
  file under all three selected Exit artifacts, not only the mutable registry
  JSON;
- a missing broker `trade_id` can no longer trigger an opposite market-order
  “close”; it fails closed as unresolved exposure;
- `--grad-accum-steps` is the consumed trainer value and partial final
  accumulation is correctly rescaled;
- stale bundle-loader scaler/feature compatibility arguments and metadata are
  removed and explicitly rejected.

The transactional launch boundary is now repaired. Its producer is
`gx1/scripts/finalize_entry_model_native_launch_v1.py`, routed through the
existing public control surface. It requires a pre-existing identity-bound
vedtak, serializes canonical registry/state targets under one lock/CAS
boundary, and publishes exact COMMIT or restorative terminal FAIL evidence.
It cannot create empirical authority.

The adversarial replay re-audit first found the canonical producer P0: the joint
Exit finalizer accepts replay and per-M1 trace parquets from a caller and never
runs the byte-bound XGB→V3→Exit-IQL/Strategy-F artifacts. It therefore cannot
prove that the active models caused the actions. The launch finalizer and
runtime guard now reject this evidence before activation. The missing
canonical full-TEST producer must reuse `V12Pipeline.make_exit_decision`,
preserve the complete Entry snapshot, bind SourceTape plus
canonical/BASE28/MTF state, and emit its own traces with zero fallback or
horizon-cap pass. The later second-audit section below supersedes the
single-P0 count. A later source checkpoint closed this producer P0 in the
existing sizing/replay owner and retained the caller-parquet route as
diagnostic-only; no compliant real artifact chain has executed it.

Also open before any rebuild/training: repair and exact parity proof for the
canonical/live December-2024 M5 tape. Read-only inspection found 3,430
impossible-geometry rows in both copies, 2,799 on weekends; clean M1 supports
5,757 rebuilt December buckets and leaves 3,459 canonical rows unbacked.

Still empirically unproved after source repair: a fresh dataset, trained
checkpoint, calibrated LONG/SHORT/FLAT predictions, untouched OOS direction
edge, train==serve parity, replay, sizing/Exit performance and live-like
precision. V24/V7 predate these contracts.

The 1,725/1,720/5 repository result below was the first repair checkpoint. It
is historical and was superseded when the second audit added new contracts and
tests; the final current count belongs in the launch-state verification block.
These results prove source contracts, not a trading edge.

## Second audit correction — Exit and incremental chain

The later field-by-field audit proved that the retained Exit chain cannot be
used as a live incumbent:

- the old per-bar builder anchored near T+1 and emitted its first state at
  T+2/T+3 rather than using the exact T+5 fill and the closed fill-bar;
- the old lazy join used the M5 row carrying the same start label as the M1
  row. On four of five M1 phases that M5 bar was still forming; an empirical
  sample measured 80.026% exposed rows;
- V3 scoring placed the first trade-state overlay on the fill row one bar
  before the corresponding per-bar state;
- a fixed 96-M5 fetch covered only 96 of 104 M5 buckets in an observed
  512-M1 window, leaving 36 M1 rows with zero-filled canonical/XGB state;
- five active M1 microfeatures used simple returns/std live but log
  returns/RMS in training;
- TradeState mutated before V3/Exit validation, did not persist its last M1
  identity and lost Strategy-F deferral state across restart;
- the retained Exit-IQL summary declares `research_only_v1=true` and
  `iql_production_allowed_v1=false`; ordered feature names were not
  checkpoint-bound and the registry selected the first of three folds
  implicitly;
- the retained V3 training manifest points at an absent source dataset, and
  the old Exit policy was trained on an older Entry candidate distribution.

Source now repairs the timing, exact M5 join, exact overlay mapping, full
512-M1 coverage, microfeature parity, transactional/persisted TradeState,
Entry-frozen regime/session evidence, explicit serving fold, complete feature
coverage and ordered summary/checkpoint SHA. The old artifacts remain
invalid—they must be rebuilt, rescored and retrained on the repaired substrate.

The data/feature producer audit also found hidden sentinels, overlapping
feature owners, HTF availability errors, bounded-history state resets and
non-transactional lineage. Exact schema and BASE ownership, full-history
BASE augmentation and M5 decision-time HTF alignment are repaired in source.
The post-audit source checkpoint closes complete-history canonical-v2
recomputation, one atomic immutable canonical-v3/BASE28 generation pointer,
strict native-M5 market-closure/schema/hash ownership and reproducible V3
lineage bound to the exact XGB bridge identity. Because no causal
pre-decision slippage observation exists in the canonical tape, the
slippage-derived decision fields were removed; explicit replay stress remains
evaluation-only. PLUS5 ATR/ROC/VWAP, dependent normalized VWAP and published
SMC ATR use one formula path; H1/H4 aligns to M5 decision-availability without
the old extra lag. The existing V3 owner now contains the exact model-native
dataset writer/event and proves it with an end-to-end atomic-publication test.
The existing OANDA M5 owner now also contains the immutable native-source
producer, including retained source responses, complete-only rederivation,
streamed year output and atomic no-replace publication. Still open are
execution of that producer, a complete initial native→canonical-v3/BASE28
bootstrap, a fresh V3 dataset on compliant inputs, execution of the canonical
active-Exit full-TEST producer and fresh artifact rebuilds. Full-loader execution is
independently blocked by 2,375 invalid prebuilt OHLC rows between 2024-11-30
00:40Z and 2024-12-31 23:55Z.

The 2026-07-24 native-M5 producer checkpoint collected 1,895 tests: 1,890 passed, five were
explicitly skipped and zero failed. The manifest-bound loader also correctly
rejects the current live canonical parquet: the legacy updater changed its
bytes without advancing the canonical manifest SHA. This is fail-closed
evidence, not live readiness.

## Third audit correction — dataset, training and replay seams

Three fresh read-only agent passes re-audited the data/feature path, model/
training path and inference/launch path against the post-repair tree. Their
findings were corrected in the existing owners; no replacement launcher,
dataset builder or versioned compatibility script was added.

The repaired source now:

- maps every M1 phase to one shared last-closed-M5 key. An M1 row ending
  `xx:04` can no longer join the still-forming M5 bucket;
- requires the complete 95-row volume prefix and rejects incomplete history;
- validates exact XGB session domains, ordered bridge features, finite
  probabilities and the probability simplex before V3 consumes them;
- defines Exit IO V8 as a 173-field per-M1 historical context. The 78 market
  context fields are reconstructed at each historical M1 row rather than
  broadcast from the terminal row;
- validates the V3 training substrate itself: `N x 173` float32 market
  matrix, strict UTC minute order, zero base trade-state slots, exact
  seven-field XGB bridge recomputation, contiguous float32 overlays, exact
  overlay/record geometry, T+5 identity, 240-row teacher paths and terminal
  teacher equality. Producer-input and dataset-member bytes are rehashed and
  re-inventoried to detect substitution and time-of-check/time-of-use drift;
- fits path calibration only on declared tradable support and bad-path
  calibration only on the exact selector mask. The evaluator persists both
  LONG and SHORT selector masks, and off-support values cannot alter the fit;
- promotes prediction evidence to runtime V3 only when all active runtime
  heads are present, duplicated parquet/head values agree and the declaration
  explicitly carries runtime-head authority. Smoke, serve and launch
  consumers reject the older V2 declaration;
- separates canonical label-horizon bid/ask facts from active-Exit decisions
  and fills in replay schema v7. Each step binds the committed closed bar,
  following fresh quote, state price/PnL and model action; active fill is the
  final fresh quote and cannot overwrite the immutable label outcome;
- provides exact SourceTape open-quote lookup, an atomic frozen prebuilt-pair
  load and an Exit-only `V12Pipeline` factory that cannot load SmartEntry.

These closures do not create either missing producer. The model-native V3
materializer can validate and materialize one trade, but no canonical
end-to-end dataset event yet derives and publishes every matrix, overlay and
record from the bound sources. The Exit-only factory can load the frozen
runtime chain, but no canonical full-TEST loop/event yet owns every
`make_exit_decision` call and publishes its replay rows/traces. Launch and
training therefore remain fail-closed.

## V7 terminal result

`XAU_SEQ513_SMOKE_20260723_V7` ran from
`2026-07-23T12:50:30Z` to `2026-07-23T13:57:33Z` on the exact immutable V7
recipe and V24 split bytes. It completed six full TRAIN/VAL epochs, then the
hard-red slice gate stopped epochs seven and eight with
`TRAIN_FAIL_NO_BEST_STATE`.

- Highest raw VAL accuracy was `0.403455` at epoch 5, obtained with
  `85.1118% FLAT`, `14.3970% SHORT` and only `0.4912% LONG`.
- Final epoch accuracy was `0.381267`, with `71.4092% SHORT`,
  `24.5088% FLAT` and `4.0820% LONG`.
- Final direction-slice score was `-1.444065`, with 32 failed checks.
- Final bad-path/survival AUC was `0.478/0.514`.
- Six prediction-head pairs exceeded the collapse threshold; the largest was
  clean-edge versus path-quality at Spearman `+0.985`.
- Final VAL specialist/TF/family×TF minimum gate means were
  `0.000054/0.024166/0.000300`; specialist and family×TF failed the fixed
  `0.01` floor.
- No checkpoint or bundle was written. The temporary 72.71 GB memmap was
  removed automatically.

## P0 — signed dip-MFE target corruption

Status: **source repaired; fresh rebuild required**.

The dataset builder calculates signed spread-aware MFE, but then clips all six
`y_dip_mfe_{long,short}_K{12,48,96}` targets to a non-negative range.

- Producer: `gx1/scripts/build_entry_v10_ctx_training_dataset_v3.py`,
  `_build_dip_targets`, around lines 369-373 and 410-486.
- The test at `tests/test_entry_v10_outcome_targets.py`, around line 152,
  currently encodes the same invalid `max(0.0, ...)` behavior.
- Exact V24 has `7.46%` zero dip-MFE values in TRAIN and `3.12%` in VAL.

This destroys the distinction between “price exactly recovered spread” and
“price never recovered spread.” It violates the signed forward-outcome
invariant and contaminates the 18-value dip evidence group.

Required repair:

- preserve finite negative MFE and only keep MAE as a non-negative magnitude;
- add a monotonic adverse-path test that must produce negative MFE;
- rebuild every split from source; V24 cannot be patched in place.

## P0 — selected-side bad-path loss always suppresses LONG

Status: **source repaired and LONG↔SHORT gradient-tested**.

`y_bad_path` is selected from LONG or SHORT bad-path truth according to the
model-native direction side:

- `gx1/scripts/build_entry_v10_ctx_training_dataset_v3.py`, lines 550-561 and
  3153-3157.

The active probability penalty nevertheless always applies to
`probs[..., 0]`, which is LONG:

- TRAIN: `gx1/models/entry_v10/entry_v10_ctx_train_v3.py`, lines 5371-5383;
- VAL: the same file, lines 6576-6588.

The symmetric-negative block mirrors dead/teaser/hard negatives for SHORT but
does not mirror selected-side bad path. On the exact V7 cap, 814 rows were
bad-path-positive: 389 LONG and 425 SHORT. All 425 SHORT rows incorrectly
suppressed LONG. For bad LONG, CE and the penalty oppose each other; for bad
SHORT, both objectives favor SHORT.

Required repair:

- one shared TRAIN/VAL helper must penalize the probability of the selected
  bad side only;
- FLAT plus selected-side bad path must fail the target contract;
- LONG↔SHORT swap invariance and exact per-logit gradient tests are mandatory.

## P1 — replacement sampler hides 37% of selected rows per epoch

Status: **source repaired with exact no-replacement epoch coverage**.

`_SliceBalancedSampler` is documented as ordering the selected data, but uses
random choice with replacement in
`gx1/models/entry_v10/entry_v10_ctx_train_v3.py`, around lines 2799-2939.

Exact V7 reproduction:

- selected cap: 25,000 rows;
- per-epoch draws: 25,024;
- unique rows: 15,533-15,661 (`62.13-62.64%`);
- duplicate draws: 9,363-9,491;
- maximum repetition of one row: 7-9;
- unique bad-path-positive rows: only 556-613;
- class prior moves from `[20.37%, 18.47%, 61.16%]` to approximately
  `[25.4%, 23.6%, 51.0%]`.

Global and slice prior losses therefore learn the sampler distribution rather
than the declared selected TRAIN distribution.

Required repair: a coverage-preserving `BatchSampler` must visit every selected
row exactly once per epoch and form balanced batches by grouping/swapping
without replacement. Any padding must be explicit and separately measured.

## P1 — symmetric targets use LONG-only positive weights

Status: **source repaired with shared bidirectional target/rate semantics**.

Clean-edge and survival use bidirectional targets when symmetric mode is
active, but their positive weights are computed from LONG-only target rates.

Exact V24 TRAIN:

- clean-edge rate/weight: LONG-only `0.075268 / 12.2859`; correct bidirectional
  `0.144210 / 5.9344`;
- survival rate/weight: LONG-only `0.129365 / 6.7301`; correct bidirectional
  `0.245835 / 3.0678`.

The active positives are over-weighted by about `2.1-2.2x`. Rate, target and
weight must come from one shared semantic target producer, with LONG↔SHORT
swap tests.

## P1 — auxiliary AUC can pass by relearning tradable versus FLAT

Bad-path, clean-edge and survival park FLAT targets at zero, but checkpoint AUC
is computed globally. A predictor that uses only `y_tradable`—and has no path
skill—achieves on exact V24 VAL:

- bad-path AUC `0.707166`;
- clean-edge AUC `0.769777`;
- survival AUC `0.807505`.

These all exceed the current `0.52` floor. Path metrics must therefore be
conditioned on tradable support, reported separately for LONG/SHORT and prove
incremental skill over a tradable-only baseline.

## P1 — checkpoint admission does not cover every decision influence

Status: **source repaired for all 22 active heads and all 26 fusion inputs**.

The checkpoint aux gate currently audits only tradable, bad-path, clean-edge,
survival, path-quality and MFE. A checkpoint can be written while the following
active outputs are constant or anti-targeted:

- MTF direction;
- trade/side hierarchy, side utility/bad-path/MAE/validity;
- trendline, TF agreement and position size;
- dip, forecast, timing, tail risk and volatility;
- Q, V and advantage.

Every one of these contributes to the 26-group/96-value final fusion. Each
active head needs a support-aware metric and each evidence group needs a
class-margin ablation/Jacobian influence floor before checkpoint admission.
Global gate mean/entropy is liveness evidence, not proof that a specialist or
TF changes the correct class margin in the regimes where it should matter.

## P1 — raw input scales are not contract-normalized

Status: **source repaired with complete physical-TRAIN fit and immutable
field/statistic/causal-row lineage; fresh rebuild required**.

V24 TRAIN input standard deviations span approximately 6.4 million times:

- `session_regime.eu_structure_breakout_readiness`: `3.3438e-5`;
- `ctx_cont.d1_pct_change_5_canon_v2`: `215.0765`.

The 513 signals and 142 continuous context fields enter raw Linear projections.
The legacy `seq_scaler_path` and `snap_scaler_path` loader arguments only
validate/store paths and never transform input.

The repair fits robust median/IQR then positive absolute-deviation fallback,
rejects constant/invalid fields, preserves exact binary and categorical
domains, caps TRAIN clipping at two percent and binds the same state in
metadata, lock, model buffers, loader, replay and serve. The 82-field V24 alias
count is test evidence only; the live alias set is derived from ordered names.

## P1 — 96 fusion values have incompatible units

Status: **source repaired with 20-bps forward-head units, immutable per-field
input normalization, positive contract-bound TF scales and learned group
projections; fresh influence/edge evidence required**.

The 96 values combine raw bps, scaled bps, logits and `[0,1]` timing values,
then apply one cross-channel `LayerNorm(96)`. Example V7 target standard
deviations range from timing `0.194` and time-to-MFE `0.347` to dip-MFE
`45.280`, tail-MAE `55.142` and action value `143.84` before its separate
Q-scale.

The model now receives contract-normalized raw inputs, unit-normalized forward
heads and positive hash-bound TF scales before the learned 26-group fusion.
All 26 groups retain checkpoint-blocking class-margin influence. This proves
connectivity and scale ownership, not useful OOS cooperation.

## P1 — context specialist routing is taxonomy-only

Status: **source repaired with exact 142+5 one-owner routing and perturbation
proof**.

The specialist audit classifies all 142 continuous and five categorical
context fields and reports full mapping. The model specialist indices,
however, address only the 513 signal tensor. All context values enter generic
context projections and may affect later gates, but 60 continuous context
fields have no signal alias and can never enter the specialist token claimed
by the audit.

Family-specific continuous and categorical projections now enter the owned
specialist token before cross-attention. Categorical fields use separate
field/domain embeddings. Alias values must equal their signal snapshot bytes;
they have a single normalization-stat owner and no second independent value.

## P1 — MTF source identity is incomplete

Status: **source repaired end to end; fresh cache/rebuild proof required**.

The trainer binds the three split manifests and parquets but not the external
`--m5-prebuilt-path` used to build all M5/M15/H1/H4/D1 inputs. It checks
existence and columns but does not compare the file SHA to the split manifests.
Optional disk cache manifests also omit hashes for the ten component `.npy`
files.

The V7 CLI did point at the intended V24 file; no current byte divergence was
observed. The boundary is nevertheless replaceable.

The admitted trainer now requires an absolute V2 disk-cache directory and
rejects source-build fallback. Its schema binds source M5, the exact ten
arrays, file sizes/hashes, feature order, the 11-file inventory and one cache
identity; the trainer reuses only those verified bytes.

## P1 — launch authority safe completion path — source repaired

The current control surface still rejects direct promote/pin/shadow/live. The
canonical transactional finalizer updates launch authority plus artifact
registry as one recoverable transaction only after every upstream authority
passes. Its explicit active-Exit producer gate now accepts only replay-v7
producer-owned evidence and currently blocks because no compliant fresh chain
has executed that producer.

The former nonempty `GX1_SMART_LAUNCH_VEDTAK` pass-through is now removed from
both shell launcher and runner. Artifact validation requires the newest exact
approval event, complete launch-state payload hash and bundle-commit hash.
Runtime revalidates the unchanged authority lease before each new exposure.
The producer now requires a separately pre-existing one-time vedtak bound to
the exact bundle, transaction, targets, operating point and six prerequisite
evidence identities. Same-byte/no-symlink reads, a stable process lock,
compare-and-swap, immutable target-local backups and strict COMMIT/FAIL
recovery close hash/reopen, concurrency and partial-replacement gaps. The
public CLI owns canonical roots, and `.env` cannot supply launch authority.
Isolated transaction-mechanics tests prove commit, tamper rejection, exact
`ALLOW`, decoy Exit rejection, single-exposure enforcement,
partial-replacement rollback, strict recovery events and idempotent completed
retry. The synthetic upstream fixture is explicitly bypassed only inside those
transaction tests; production rejects it.

## P2 cleanup and quality findings

- Handover was hard-coded to pre-execution V7 state. It must validate explicit
  `READY_NOT_STARTED` and `TERMINAL_FAILED` states now, and later add
  independently validated produced/candidate/active states.
- Bundle export direct-write is repaired with hidden staging, exact commit
  inventory, strict pre-publication load, `fsync` and atomic no-replace rename.
- `feature_meta_path`, `seq_scaler_path` and `snap_scaler_path` are unused
  compatibility arguments in the loader.
- TRAIN and VAL component statistics mix weighted and raw loss units; the
  `bad_path_loss_sum` also combines BCE with a direction penalty.
- CLI `--grad-accum-steps` is not the value consumed by the train loop.
- Selector masks are dead indirection because both path-candidate arrays are
  always one.
- Some rates/weights use the full dataset instead of the selected subsample.
- Duplicate MTF construction and the dead V1 branch are removed.
- RSI, percentage change and rate-of-change fields route to momentum.
- Optional feature ranking now uses the exact spread-aware LONG-utility minus
  SHORT-utility target with final PnL, MFE, MAE and path-quality terms.
- System documentation claimed MTF sequence length 96; V7 actually used
  `16/16/16/8/8` for M5/M15/H1/H4/D1.
- The repaired December-2024 tape exists only in the V24 event copy; canonical
  M5 and live-prebuilt still carry the documented defect.

## Verified correct

- LONG=0, SHORT=1 and FLAT=2 are consistent.
- The 513 signal indices are unique, non-overlapping and fully covered by the
  eight specialist encoders.
- All 513 signal, 142 continuous context and five categorical inputs enter the
  model and are finite in V24.
- Five causal MTF branches, eight specialists and the learned family×TF path
  are physically present and receive gradients.
- The 22 supervised heads contribute exactly 26 groups and 96 values to one
  learned `96 -> 128 -> 3` fusion. No sibling direction head bypasses it.
- Causal split boundaries, closed-bar timing and common-history construction
  showed no active future shift, centered rolling, bfill, interpolation or
  forward merge.
- Bundle loading is strict and hash-bound; missing/invalid evidence raises
  direction unavailability rather than synthetic FLAT.
- Every context field has one specialist owner; signal/context aliases and
  seq-terminal/snapshot values must be bit-identical.
- Immutable TRAIN-fit normalization and five-timeframe cache bytes are
  persistent model/bundle state, not runtime defaults.
- Bundle/event publication is hidden-stage, fsynced and atomic no-replace.
- Active Exit artifact bytes, launch approval and runtime authority lease are
  independently fail-closed.
- Runtime spread/sizing may block order placement but cannot rewrite model
  direction.
- The forbidden-instrument scan is clean; `eu_*` is the European session
  label.
- Current launch authority is `BLOCK`.

## Ordered repair boundary

1. Record V7 and this audit as immutable failure evidence.
2. Preserve the completed source repairs for both P0s, sampling, auxiliary
   semantics, normalization, context routing, MTF binding, all-head influence,
   atomic bundle/event publication, Exit-byte binding, transactional launch
   finalization and runtime fail-close.
3. Preserve the exact V3 dataset producer in its existing owner and public
   control route. It must keep owning all 173-field rows, T+5 overlays/records,
   bound inputs/source inventory, immutable event and atomic publication.
4. Preserve the completed canonical full-TEST active-Exit replay producer in
   the existing sizing/replay owner. It owns its rows/traces and cross-binds
   all producer inputs, active artifacts, transitive source and outputs;
   caller-created parquets remain diagnostic-only.
5. Repair and prove canonical/live December-2024 tape parity.
6. Rebuild a fresh XAU-only dataset. Re-run every liveness, target, specialist,
   readiness and trainability audit.
7. Only then bind a new smoke recipe. Preserve the final TEST window for one
   declared untouched decision.
