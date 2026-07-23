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

The audits changed no file and did not interfere with V7.

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

V24 TRAIN input standard deviations span approximately 6.4 million times:

- `session_regime.eu_structure_breakout_readiness`: `3.3438e-5`;
- `ctx_cont.d1_pct_change_5_canon_v2`: `215.0765`.

The 513 signals and 142 continuous context fields enter raw Linear projections.
The legacy `seq_scaler_path` and `snap_scaler_path` loader arguments only
validate/store paths and never transform input.

Required repair:

- one immutable TRAIN-only, ordered scaler for 513 signals and 142 continuous
  context values;
- binary fields remain binary, categorical fields remain categorical and
  continuous fields use contract-owned robust scaling with scale floors;
- dataset, bundle, replay and live serving bind the same scaler bytes/hash;
- wrong field order, invalid scale or missing scaler fails closed.

## P1 — 96 fusion values have incompatible units

The 96 values combine raw bps, scaled bps, logits and `[0,1]` timing values,
then apply one cross-channel `LayerNorm(96)`. Example V7 target standard
deviations range from timing `0.194` and time-to-MFE `0.347` to dip-MFE
`45.280`, tail-MAE `55.142` and action value `143.84` before its separate
Q-scale.

One global LayerNorm does not create fieldwise semantic parity. Each of the 26
groups needs a contract-scaled output or group projection/normalization before
fusion, followed by scale-perturbation and group-influence tests.

## P1 — context specialist routing is taxonomy-only

The specialist audit classifies all 142 continuous and five categorical
context fields and reports full mapping. The model specialist indices,
however, address only the 513 signal tensor. All context values enter generic
context projections and may affect later gates, but 60 continuous context
fields have no signal alias and can never enter the specialist token claimed
by the audit.

Required repair:

- exact per-family `ctx_cont`/`ctx_cat` ownership;
- family-specific context projections fused into the corresponding specialist
  token before cross-attention;
- complete 142+5 coverage and perturbation tests;
- explicit policy against double-weighting the 82 aliased context values.

## P1 — MTF source identity is incomplete

The trainer binds the three split manifests and parquets but not the external
`--m5-prebuilt-path` used to build all M5/M15/H1/H4/D1 inputs. It checks
existence and columns but does not compare the file SHA to the split manifests.
Optional disk cache manifests also omit hashes for the ten component `.npy`
files.

The V7 CLI did point at the intended V24 file; no current byte divergence was
observed. The boundary is nevertheless replaceable.

Required repair: bind the M5 path/SHA into the recipe, require equality across
all split manifests and hash every cache component plus the aggregate cache
identity.

## P1 — launch authority has no safe completion path

The current control surface intentionally rejects promote/pin/shadow/live, but
there is no canonical transactional finalizer that can later validate a full
candidate chain and atomically update both launch authority and artifact
registry.

Additionally, live launch and artifact validation currently treat a nonempty
`GX1_LIVE_VEDTAK` environment value as sufficient. It is not bound to an
immutable launch event, exact candidate/bundle or one-time identity.

Required repair:

- one public promotion/launch finalizer that validates the newest immutable
  bundle, serve, sizing, Exit, replay, shadow and lifecycle evidence;
- atomic registry/state update with a terminal failure event on partial error;
- explicit one-time vedtak ID/hash bound to that launch event;
- `.env` may not supply launch authority.

## P2 cleanup and quality findings

- Handover was hard-coded to pre-execution V7 state. It must validate explicit
  `READY_NOT_STARTED` and `TERMINAL_FAILED` states now, and later add
  independently validated produced/candidate/active states.
- Bundle export writes directly into the final output directory before all
  checks finish. Export must use a staging directory, `fsync` and atomic rename.
- `feature_meta_path`, `seq_scaler_path` and `snap_scaler_path` are unused
  compatibility arguments in the loader.
- TRAIN and VAL component statistics mix weighted and raw loss units; the
  `bad_path_loss_sum` also combines BCE with a direction penalty.
- CLI `--grad-accum-steps` is not the value consumed by the train loop.
- Selector masks are dead indirection because both path-candidate arrays are
  always one.
- Some rates/weights use the full dataset instead of the selected subsample.
- MTF is built twice because the producer and consumer use different cache
  keys; a hard-coded V2 path leaves a dead V1 branch.
- RSI, percentage change and rate-of-change fields are assigned to trend
  before momentum matching.
- Optional feature ranking uses marginal Spearman against H24 mid-close return,
  not the spread-aware LONG/SHORT/FLAT path-utility target suite.
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
- Runtime spread/sizing may block order placement but cannot rewrite model
  direction.
- The forbidden-instrument scan is clean; `eu_*` is the European session
  label.
- Current launch authority is `BLOCK`.

## Ordered repair boundary

1. Record V7 and this audit as immutable failure evidence.
2. Fix both P0s and the sampling/aux semantic mismatches with exact symmetry,
   coverage and target-domain tests.
3. Add immutable input/group scaling, context specialist ownership, MTF byte
   binding and all-head/group-influence checkpoint admission.
4. Remove dead/duplicate compatibility paths and make bundle export atomic.
5. Implement the transactional promotion/launch finalizer and identity-bound
   vedtak.
6. Rebuild a fresh XAU-only dataset. Re-run every liveness, target, specialist,
   readiness and trainability audit.
7. Only then bind a new smoke recipe. Preserve the final TEST window for one
   declared untouched decision.
