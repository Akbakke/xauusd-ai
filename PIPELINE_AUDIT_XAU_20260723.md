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

Still open before any rebuild/training:

- one canonical transactional candidate/promotion/launch finalizer. The
  consumers now reject soft approval, mutable bundle content, expired evidence
  and mid-process authority changes, but no producer yet validates the entire
  candidate chain and publishes the activating state as one recoverable
  transaction;
- repair/parity of the canonical/live December-2024 M5 tape.

Still empirically unproved after source repair: a fresh dataset, trained
checkpoint, calibrated LONG/SHORT/FLAT predictions, untouched OOS direction
edge, train==serve parity, replay, sizing/Exit performance and live-like
precision. V24/V7 predate these contracts.

Repository verification after the repairs collected 1,725 tests: 1,720
passed, five were explicitly skipped and zero failed. Changed Python sources
and tests compile and pass Ruff (excluding the repository's intentional
import-bootstrap `E402` pattern); JSON, shell syntax, diff hygiene, handover
self-check and the exact forbidden-instrument scan are green. These results
prove source contracts, not a trading edge.

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

## P1 — launch authority has no safe completion path

The current control surface intentionally rejects promote/pin/shadow/live, but
there is no canonical transactional finalizer that can later validate a full
candidate chain and atomically update both launch authority and artifact
registry.

The former nonempty `GX1_SMART_LAUNCH_VEDTAK` pass-through is now removed from
both shell launcher and runner. Artifact validation requires the newest exact
approval event, complete launch-state payload hash and bundle-commit hash.
Runtime revalidates the unchanged authority lease before each new exposure.
The remaining gap is producer-side transactional promotion/finalization.

Required repair:

- one public promotion/launch finalizer that validates the newest immutable
  bundle, serve, sizing, Exit, replay, shadow and lifecycle evidence;
- atomic registry/state update with a terminal failure event on partial error;
- explicit one-time vedtak ID/hash bound to that launch event (consumer side
  is repaired);
- `.env` may not supply launch authority (repaired).

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
   atomic bundle/event publication, Exit-byte binding and runtime fail-close.
3. Implement the transactional promotion/launch finalizer and identity-bound
   vedtak.
4. Repair and prove canonical/live December-2024 tape parity.
5. Rebuild a fresh XAU-only dataset. Re-run every liveness, target, specialist,
   readiness and trainability audit.
6. Only then bind a new smoke recipe. Preserve the final TEST window for one
   declared untouched decision.
