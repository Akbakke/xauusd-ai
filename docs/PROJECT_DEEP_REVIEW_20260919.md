# Project-wide deep review — 2026-09-19

> **Repair status — 2026-09-20.** Four repair waves are committed on
> `audit/v9-premiere-20260905`, each with a green full canonical CPU suite:
> `6a2be2a7` (wave 1: trainer hygiene + EMA warmup — closes A-3, A-4's
> visibility, F1/F2/F5–F17/F19), `48d570ad` (wave 2: contract-derived
> intra-epoch fitted-Q target refresh, iteration-state schema v2 — closes
> A-1 and thereby the A-2 interaction; the monitor's absorbing property now
> judges a fully propagated value function, which is the honest criterion),
> `02babae8` (wave 3: cloud seams — closes B-1, B-3, B-4, B-5, A5, A8),
> `233a602d` (wave 4: dataset chain — closes C-1, C-3, C-4, C-7).
> **Still open, deliberately batched into ONE rebuild wave** because they
> change fit populations or targets: C-2 (gap-spanning Exit episodes), C-5
> (closed-interval upstream fits at a shared boundary), and the feature
> encodings D-1…D-6 with their mediums. 2026-09-20, source-only (working
> tree, no rebuild run): the level-registry share of that wave is
> implemented — D-1's emitted medium (the thresholded
> `recurrence_confirmed` vote is replaced by the raw
> `level_*_recurrence_dist_atr` birth distance; the fitted threshold stays
> bound as fit-chain lineage), the D-2 information restoration (four
> `level_*_pending_retest_{dist_atr,age_bars}` slots, NO confirmation band
> added to the event tape), and the reinstated pure-geometry
> `level_round_number_dist_{50,100}_atr` fields (+6 columns per lane;
> HTF matrix contract V21, cache manifest v31, per-TF liveness v21,
> registry state schema 10). Same day, source-only, part 3: D-3 (the three
> exactly-derived squeeze emissions retired, the two carriers stay; recovery
> identities at the owner), B15 (the squeeze module contract now states the
> implemented left-censored release-age clock instead of claiming NaN), A1
> (unsigned `level_bars_since_break` retired for its signed superset) and B9
> (`geomline_bars_since_break` renamed and signed by the remembered break
> side) — net −4 columns per lane; HTF matrix V22, cache manifest v32,
> per-TF liveness v22, squeeze manifest v2, level registry v16, trendline
> contract V6 (read all widths by executing the owners). The D-2 break-rule
> dynamics themselves (`SWING_LOOKBACK` sawtooth) and D-4…D-6 plus the D-3
> admission-gate medium remain open; every cache/bundle
> built on the V20/V21 surfaces is invalidated by the schema bumps and the
> rebuild wave is still pending. Cloud items still open: B-2
> (plausibility floor needs an independent witness bound into the smoke
> measurement), A6/A7 (billed instance time vs training time; FX staleness
> window), C2/C3-cloud (provider-side deadline off-host; monotonic clock
> backstop). Every remaining item requires either the rebuild or an
> operator/provider decision; no training or spend is authorized by this
> note.

Operator-ordered full-project defect hunt, executed 2026-09-19 by five parallel
read-only audits over (1) the decision-layer contracts, (2) the complete
canonical trainer, (3) the feature layer, (4) the committed cloud-training
package (`e4f0902e`), and (5) the dataset/normalization/leakage chain. This
document is the durable findings register required by CLAUDE.md rule 25c. It
grants no execution authority. Evidence classes: **source** (proven from
source/algebra), **measured** (real declared data, population and date stated),
**unmeasured** (mechanism proven, magnitude unknown).

## Headline verdict

The pipeline's causality, split hygiene, economics and decision losses are
sound (see "Verified clean"). The single most important result is a
**coherent four-part mechanism stack that predisposes the current candidate
configuration to select an all-FLAT checkpoint and stop early, regardless of
whether edge exists**. Do not launch the 30-epoch cloud candidate until the
selection-dynamics repairs below are decided and bound as a successor.

Supporting measurement (full TRAIN, 313,399 rows, 2026-09-19): per-side
diagnostic direction scores have means −1.34 (LONG) / −1.97 (SHORT) bps with
47.5 % / 44.8 % positive shares, p90 ≈ +26 bps; hindsight best-side mean
+16.9 bps with 92.3 % of rows positive on at least one side. An unskilled
converged Q-model therefore correctly predicts both sides slightly below the
exact FLAT anchor 0 → all-FLAT argmax. FLAT-everywhere is the correct
zero-skill answer, and the positive tail training must find is wide but must
overcome only a 1–2 bps margin. The columns measured are the horizon
diagnostics (`y_direction_*_score_bps`, bit-identical to
`*_final_pnl_at_direction_horizon_bps`), not the exact per-action
fitted-Q-to-Exit-teacher bridge.

## A. The FLAT-collapse mechanism stack (decision-critical)

- **A-1 CRITICAL (source).** One fitted-Q target refresh per epoch ×
  512-state exit chain: `Q_hold` bootstraps one state per refresh
  (`unified_exit_fitted_q_v1.py:191-195`; `copy.deepcopy(model)` inside the
  epoch loop, `entry_v10_ctx_train_v3.py:13268`, `:11631`). After k epochs
  `V(s0)` is a stopping value over ≈ the first k of 512 states — truncated
  strictly downward, and only for LONG/SHORT; FLAT's 0 anchor is exact.
  Counter-force: single-net `amax` maximization bias pushes upward
  (`double_q.active=False`). Net sign needs the per-epoch
  `first_side_values` histogram measurement.
- **A-2 CRITICAL (source).** All-FLAT scores exactly 0.0 on the checkpoint
  monitor (`entry_candidate_checkpoint_policy_v1.py:25-26`,
  `EARLY_STOP_MIN_DELTA=0.0`; FLAT PnL column structurally zero,
  `entry_v10_ctx_train_v3.py:6360`). Any policy not beating the round-trip
  spread scores negative, so once all-FLAT is reached no later epoch can
  register improvement without strictly positive net PnL in one step; early
  stop fires 5 epochs later with the FLAT checkpoint selected. FLAT is an
  absorbing selection state. Historical note: V9 epoch-1 monitor was
  −0.658 bps, i.e. not all-FLAT — the metric is sensitive.
- **A-3 HIGH (source+algebra).** EMA has no bias correction or warmup
  (`entry_v10_ctx_train_v3.py:5489-5517`): with decay `1−1/N` over an
  N-step epoch, the validated/selected epoch-1 weights carry e⁻¹ ≈ 36.8 %
  untrained initialization, epoch 2 ≈ 13.5 %, epoch 3 ≈ 5 %. Epoch 1 is
  always recorded as best (best_epoch<0 bypass), so `top_k/epoch_0001.pt`
  is a 63/37 blend by construction.
- **A-4 MEDIUM (source+algebra, lr-parametric).** Neutral log-variance init ×
  raw-bps MSE magnitudes × global `clip_grad_norm=1.0`: the learned task
  weights need ≈ `12/lr` optimizer steps (~120k at lr 1e-4) to reach their
  own optimum; early-epoch losses are dominated by the unweighted bps-scale
  task.
- **Interaction.** A-1..A-4 all corrupt exactly the epochs (1–5) in which the
  patience-5 selection passes judgment. They are one stack, not four
  independent nits. Codex's Q_mu-vs-pi512 flag is confirmed as the
  training-vs-selection objective split (different weights, different
  operators; coincide only at convergence).

Related admission-path defects: fail-open gate-entropy check
(`clamp(min=1e-12)` makes `entropy<=0` unreachable,
`entry_v10_ctx_train_v3.py:5102/:5173`); FLAT column exempted from the
*prediction* deadness check, not only the target (`:652-654`, `:9491-9502`);
primary Entry-Q diagnostics measure agreement with the teacher, not edge —
all-FLAT reads as good agreement (`:9218-9320`).

## B. Cloud package — fix before any spend (all source)

Structurally sound (no injection; atomic publication; nonce/replay handling;
GPU pinning; gate mandatory and re-derived; 19 areas verified clean). Gaps:

- **B-1 HIGH.** Capacity gate never checks the host's *remaining* life; a
  PASS with 4 h left to the provider deadline and a 43 h projection is
  possible (`cloud_training_capacity_gate_v1.py:617`; profile deadline only
  checked as `now >= deadline` at load).
- **B-2 HIGH.** No physical plausibility floor on `measured_train_seconds`;
  the whole PASS rides on one unverifiable self-reported float (`:403-410`).
- **B-3 HIGH.** Benchmark `train_rows`/`val_rows` are never bound to the
  candidate recipe's split populations (`:721-803`) — a projection for a
  different dataset can admit the candidate.
- **B-4 HIGH.** Deadline-fire path: one deletion attempt (120 s), then
  unconditional local poweroff — which does not stop provider billing and
  destroys the only agent able to retry; `provider_managed_delete: true` is
  self-asserted by the host to be deleted; no out-of-band provider-side stop.
- **B-5 MEDIUM-HIGH.** One failed telemetry sample terminates the whole paid
  run (no retry budget; single-threaded HTTPServer any local process can
  stall) — safe but the most likely way to burn NOK without a bundle.
- MEDIUM: restart reserve ignores lost work since last checkpoint (512-step
  interval); cost model bounds training time, not billed instance time
  (setup/storage/egress excluded); price/FX/`fx_observed_utc` are unchecked
  CLI inputs with no staleness bound; wall-clock deadline timer with no
  monotonic backstop; trainer wall limits equal the 48 h deadline per stage.
- LOW: trailing partial batch charged pro-rata (should ceil); no pessimistic
  Decimal rounding mode; duplicated limit literals across owners;
  `forbid_test_like=False` on the gate artifact itself; single-uid cloud VM
  makes signing self-attesting (threat-model note).

## C. Dataset/target chain

- **C-1 BLOCKING fail-closed (source+cross-artifact).** `train_policy_end =
  train_end − 300 s` shift is unilateral: builder applies it
  (`build_entry_v10_ctx_training_dataset_v3.py:5172`), ranker/preflight/
  legacy-chain assert the unshifted value against the same field; the
  preflight assertion is currently vacuous and legacy-mode ranking+build
  cannot agree (`policy_sha256` mismatch). One owner must compute it.
- **C-2 MAJOR (source, magnitude unmeasured).** Exit episodes are
  row-consecutive, not minute-consecutive: a 512-state episode silently
  bridges weekends; the elapsed-time channel says "one minute" across the
  gap and `exit_now_reward_bps` jumps by the whole gap with zero financing.
  Entry excludes closure-spanning rows; Exit includes and mislabels them —
  the two surfaces treat closures oppositely.
- **C-3 MAJOR (source).** Ten label-statistics log lines are not split-gated
  and print TEST label base rates (tradable rates, sizing quantiles, touch
  rates) to the build log before the seal
  (`build_...v3.py:4196-4455`) — a live operator-side selection channel.
  `_log_label_distribution_proof` already shows the correct gating.
- **C-4 MEDIUM (source).** No contract owner compares the MTF-cache/squeeze
  frozen `declared_train_window_*` against the dataset's declared TRAIN
  split; only the chain shell script enforces it. `--pretest-only` reuses a
  published cache with that window unverified.
- **C-5 MEDIUM.** Upstream TRAIN fits use closed `[start, end]` intervals
  (registry, squeeze) while builder permits `train_end == val_start` — a
  shared boundary bar can be in both fit and VAL. Preflight's stricter check
  is not required by the builder.
- **C-6 declared, note.** VAL is simultaneously the selection set, the
  early-stop set, and the sizing-calibration fit set
  (`MODEL_NATIVE_SIZING_FIT_SPLITS=("val",)`) — VAL-measured sizing utility
  is in-sample for the calibration.
- **C-7 MEDIUM (source, unmeasured).** Near-constant normalization scale
  fallback can take a single observation's deviation as the denominator
  (`entry_model_native_input_normalization_v1.py:863-878`) — a lone TRAIN
  event on a small D1 fit population flattens the whole column via asinh.
  `scale_source` records it; nothing inspects it.
- Sizing ECDF saturates to exactly 0/1 outside TRAIN support (declared).
- Symmetric train/serve population caveat: rows within one horizon of every
  closure are structurally absent from all splits but present at serve.

## D. Feature layer (all four documented debts confirmed; 15 new)

Measured on the hash-bound V46 cache liveness block unless noted.

- **D-1 blocking-grade (measured).** `level_recurrence_threshold_atr` spans
  1084× across lanes (D1 fitted on 11 observations); consequence:
  `recurrence_confirmed` base rate 0.97 (M5) vs 0.23 (M15) — the same field
  name means opposite things per lane; cross-TF confluence reads fitter
  variance.
- **D-2 blocking-grade (measured+source).** Level break rule has no
  confirmation band; fires 16–20 % of bars clock-invariantly (signature of
  `SWING_LOOKBACK=3`); provably always deletes the level nearest to price,
  so `level_above_dist_atr` can never reference a level price has reached.
  `bars_since_break` mean ≈ 5 bars — a sawtooth, not a memory.
- **D-3 HIGH (measured to 16 digits).** 18 of 30 squeeze columns are exact
  functions of `bars_in_squeeze` (`squeeze_active ≡ 1[b>0]`,
  `release_event ≡ 1[age==0]`, `duration_at_release ≡ b[t−1]·release[t]`)
  on all six clocks. Invisible to the duplicate detector because mandatory
  fields are excluded from its candidate pool.
- **D-4 HIGH (measured).** `vwap_local_cycle_dist_atr` is an exact
  session-phase indicator on H4 (16.7 % of rows exactly 0.0 = the 22:00 UTC
  session-open bars) and carries a different clock per lane; the sibling
  field was repaired in V30 for exactly this, the distance field kept the
  forbidden branch.
- **D-5 HIGH (numerically verified, 2.5e-13).**
  `chart.local_ema50_200_spread_delta_atr ≡ (2/49)·price_vs_ema50_atr −
  (2/199)·price_vs_ema200_atr` — exact affine duplicate inside its own
  tuple; `spread_accel_atr` near-duplicate. Same retirement criterion as the
  2026-08-19 wave applied to its sibling.
- **D-6 HIGH (measured).** Trendline block: sparse events on M5, saturated
  on D1 (31 % of bars "break", 34 % "retest hold", ~93 active lines/side;
  `bars_since_break` is a 3-bar counter on D1 vs 300-bar memory on M5).
- Additional (medium): squeeze admission gate lacks `var0<var1` and its
  reachability check is vacuous exactly when violated;
  `geomline_max_dev_atr` hard-censored at the per-lane fitted band (leaks a
  TRAIN constant as a feature scale); `geomchan_pos_0_1` clipping active on
  8.75 pp of D1 rows; `geomline_bars_since_break` sign-blind with no signed
  sibling; `recurrence_confirmed` is a birth-time binary that cannot count;
  Bollinger ddof=1 vs ddof=0 disagreement between htf and squeeze owners;
  `geomline_touch_*` mixes lag-0 and lag-3 impulses; squeeze
  `release_age_bars` fabricates 0 pre-first-release contradicting its own
  module contract; `raw_open_above/below_prev_*` below the project's own
  liveness floor on H1/H4/D1; four `candle.raw_*_change` fields saturate at
  ±1 exactly on gap bars. Corrections to stale documentation: squeeze
  occupancy on current artifacts is 44–64 % (not 87 %); threshold spread is
  three orders (not four); D1 fit support 11 (not 14).
- **Zero lookahead found.** MTF projection, session clock, warmup handling
  and all six audited feature files are causally clean; no zero-fill in the
  layer builders; bitwise-duplicate scan of all 176×5 cached fields: zero
  groups (the duplicates above are functional, not bitwise).

## E. Trainer sweep (closes the CLAUDE.md "not examined" item)

The declaration is **proven complete for the decision losses**: sole Entry
and Exit losses are masked raw-bps MSE on fitted-Q; one BCE on
`trendline_event`; learned log-variance weights; no fixed weights, rank
losses, composites, gate regularization or target scales anywhere in the
15,338 lines. Verified clean: gradient plumbing (incl. chunked Exit
re-injection), mask semantics, EMA tensor selection (parameters averaged,
buffers exact-copied), scheduler resume, determinism/FP32, movement proofs
against true epoch-0, selection reads the same weights it ships, empty-mask
task omission rule.

Seventeen unexplained magnitudes (six reach learning/admission): dip pinball
quantiles 0.5/0.9 selected by substring on target names (`:433-438`);
`_ACTIVE_HEAD_DIAGNOSTIC_MIN_ROWS=16` (`:644`); liveness eps 1e-8 scale-blind
vs bps (`:651`); the 1e-12 entropy floor (fail-open, §A); restated `2.0`
gate bound (four copies); module-level `_GRAD_CLIP_NORM=1.0` /
`_WEIGHT_DECAY=1e-5` defaults live on the `run_train` library route; AdamW
betas/eps unrecorded in the bundle identity. Structural: second UTC session
clock (boundaries 7/13/22, three sessions) inside the Entry-Q stability
diagnostic contradicting the four-session SSoT (`:9296-9300`); duplicated
inline selection authority in the smoke path dropping
`minimum_epochs_before_stop` (`:13549-13656`); `_require_pretest_recipe_cli_match`
returns silently on a legacy-schema recipe, skipping every numeric CLI
binding (`:14903-14906`); silent dense-mask fallback inside admission-gating
diagnostics (`:9418-9428`); `eligible_entry_rows` reported 0 on the chunked
(canonical) path (rule 2e); diagnostic clamps re-introducing the absorber
the neighbouring code documents removing (`:8702`, `:8721`).

## F. Recommended order (no authority granted here)

1. **Decision-dynamics successor (before any candidate spend):** decide and
   bind — target-refresh cadence decoupled from epochs (A-1); a selection
   rule that is not FLAT-absorbing (A-2: e.g. minimum epochs before the
   patience counter arms, or a secondary admission monitor on side-Q
   separation); EMA bias correction (A-3). Coordinate with the
   `work/gx1-current` (Codex) probe results; its regularized Entry-selector
   refutation probe and the per-epoch `first_side_values` histogram are the
   settling measurements.
2. **Cloud gate seams (hours, no rebuild):** B-1, B-2, B-3, deletion retry
   loop + no-poweroff-on-failure (B-4), telemetry retry budget (B-5).
3. **Trainer/diagnostic hygiene (small, no rebuild):** fail-open entropy
   check, second session clock, F8 silent return, dense-mask fallback,
   chunked `eligible_entry_rows`, dip-quantile ownership.
4. **Substrate decisions (operator; forces a rebuild — batch them):** C-1
   policy-end owner, C-2 gap-spanning episodes, C-3 TEST log gating, C-4
   window equality in a contract owner, D-1/D-2/D-3/D-4/D-5/D-6 feature
   repairs. One rebuild wave, not several.
5. Then Hopper qualification and the first full candidate under the
   repaired selection rule; ≥5 seeds before any edge statement; judge by the
   re-derived coin-flip null with abstention quality as the criterion.

## Not examined in this review

Model architecture internals (`entry_v10_ctx_hybrid_transformer.py` beyond
its interfaces), the serving/replay path, the retention owner, the M1/M5
native source producers, Codex's `work/gx1-current` branch code, and all
magnitude claims marked unmeasured above. The `first_side_values`
convergence histogram, `skipped_entry_rows` count, and the eligible-row FLAT
fraction remain the fastest settling measurements and are still unrun.
