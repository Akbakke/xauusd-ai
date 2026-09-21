# Pipeline feature/indicator fidelity review — 2026-09-21

Operator order: *"Gå igjennom HELE pipelinen i forhold til indikatorer og features og
se om alle er aktive og riktig oppsatte og samarbeider med hverandre og på tvers av
hverandre. Vi må kunne nøste opp i hva som gjør retningsedge og ikke."*

Method: six parallel read-only reviewers over (1) liveness/selection, (2) trend/EMA/
geometry, (3) momentum/vol/squeeze, (4) session/VWAP/SMC/candle, (5) cross-family model
cooperation, (6) edge-attribution machinery. Every claim is classified **[M]** measured
on real declared data, **[S]** proven from source, or **[N]** not examined. Measurement
substrates: the full native M5 tape (537,861 rows, 2019→2026-08), the PRETEST_V10
enriched cache (890 columns × 5 TFs), the FULLSUP dataset artifacts (313,399 TRAIN /
5,509 VAL), and the executed smoke/candidate bundles. Nothing was modified; the running
FULLSUP candidate (`ENTRY_V10_FULLSUP_CANDIDATE_20260921T115456Z`) was not disturbed.

This review supersedes nothing; it extends `docs/PROJECT_DEEP_REVIEW_20260919.md` and
`docs/INDICATOR_FIDELITY_AUDIT_20260813.md` with a full-surface pass at the current
240/71/178-surface.

---

## 1. Verified healthy (the positive map)

- **[M]** `classic_ema` (SMA-seed, α=2/(span+1)): bit-exact vs independent recursion,
  exact NaN warmup, causality proven by truncation test. One owner, no second EMA.
- **[M]** Wilder RSI exact to 5.7e-14 vs independent ewm; Wilder ATR-14 single owner;
  Wilder ADX/DI seed and warmup textbook-honest on all lanes (first DI at 14, first
  ADX at 27, verified).
- **[M]** Six-clock squeeze: fit and serve share one causal forward-filter step
  (bit-identical); admission gate F(0) ≥ +2.18 on all clocks; releases fire at sane
  rates (5,138 on M5 vs the incident's 1 in 352k); served occupancy matches fitted
  stationary distribution to ~1.5% on every clock. The 2026-08-15 absorbing-Viterbi
  class is structurally eliminated and measurably repaired.
- **[M]** Momentum G1/G2 event frame reproduces bit-exactly across the local and per-TF
  lanes (0 mismatches, 678k rows, 12 fields). Divergence-pivot lag exactly 3 bars on
  every event; construction exactly causal (zero slack).
- **[S]** One session clock (`gx1/time/session_detector.py`), zero restated hour
  literals repo-wide, D1 resample origin derived from the same constant, the trainer's
  second 3-session clock repaired at HEAD, decision-instant consistency enforced by a
  hard raise in the dataset builder.
- **[M]** VWAP D-4 repair landed end-to-end: 0.0000 exact-zero rate on all lanes
  (was 16.7% on H4); no cumulative-session operand survives; windows honest
  (`min_periods=window`).
- **[S+M]** `smc_swing_state` enum: docstring==code, consumed only as embedding
  (never ordinal), the inverting consumer is retired. Sweep/reclaim strictly causal,
  sides balanced (up 2.18% / down 2.21%).
- **[M]** Candle sign-blindness wave landed: `_neg()`/`clv` retired, every directional
  pair present and live on all 5 lanes, and a structural sign-semantics gate now exists.
- **[S]** ctx_cont↔seq 71/71 alias overlap is *by design*: manifest-derived, hash-bound,
  snap projection excludes aliases (no current-bar double count), one normalization
  owner (rule 19 held).
- **[S]** Per-lane branching in the EMA/VWAP/BB/ADX/stack block: zero `if timeframe`
  conditionals; one formula on all five lanes, verified by identical dispersion tables.
- **[S]** The 2×sigmoid feature gates initialize at exactly 1.0 (neutral, not dead) for
  every field×TF; gate statistics accumulate per-field (712-vector) with a health gate.
- **[M]** Zero nonfinite values across 312 fields × 2 splits; 7.2B TRAIN seq values
  scanned finite; `seq_last == snap` exact.

## 2. Findings register

### P0 — Attribution: what blocks "nøste opp retningsedgen" (the operator's goal)

- **F-1 [S]** No mechanism connects any family/field to bps of edge. Five attribution
  owners exist; all measure liveness (|ΔQ| > 1e-7) or fit-space deltas (fitted-Q MSE),
  none economics. The largest measured family ablation moves Q by **0.054 bps** (8-row
  sample, one-epoch candidate) against a selective edge of 4–28 bps.
- **F-2 [S]** Deleted prior art recovered: commit `0b139646` carried a full family-
  ablation arm matrix in the selective-edge evaluator (`--feature-mask-json`,
  per-family drop arms, pnl/drawdown/mae metric families, session/regime/direction/tail
  slices). Deleted by `3ef448e3`; the **schema slot survives** — every selective-edge
  report still publishes `feature_mask_ablation: {"enabled": false}` (evaluator :2186,
  :2281). Rule-21 home for the attribution protocol is therefore this evaluator.
- **F-3 [S]** Standing contradiction: `entry_exit_feature_usefulness_v1` forbids
  zero-substitution (`static_replacement_value: None`) while the trainer ablation and
  serve-parity gate ablate by writing 0.0 into normalized tensors (an off-manifold
  point ≡ (0−mean)/std). Two owners disagree on what a valid counterfactual is; any
  protocol must resolve this first or inherit a rule-25 failure.
- **F-4 [S]** The usefulness executor's Entry donor plan is degenerate: every row is
  its own block ⇒ donor = adjacent M5 bar ⇒ swaps are near-no-ops on autocorrelated
  fields ⇒ every Entry usefulness delta biased toward zero. (Exit episodes unaffected.)
- **F-5 [S]** Four of the strongest implemented mechanisms (serve-parity family masks,
  encoder-output hooks, 4×8 family×TF zero-mask grid, per-field gradients) have **never
  executed** (no `MODEL_NATIVE_SERVE_PARITY` event exists). Implemented ≠ measured.
- **F-6 [M]** Rule-2f power bounds for attribution on this VAL (measured HAC SEs):
  a family effect below ~4–5 bps at 25% coverage is unmeasurable; at 1% coverage the
  bound is ~28 bps. With 41 arms × 7 coverages × 5 seeds, an uncorrected sweep
  manufactures false families; the multiple-comparison rule must be preregistered.
- **F-7 [S]** Entry publishes no durable gate evidence in bundles (widths only), while
  Exit publishes full per-field gate vectors via `unified_exit_gate_evidence_v1`. The
  per-row 712-dim `family_tf_feature_gate` + 32-dim cooperation gate + realized bps
  **are** persisted in every VAL prediction parquet — the richest untapped attribution
  surface, consumable today with zero new training.
- **F-8 [S]** The mandatory 147 fields (61% of the signal surface) have no univariate
  diagnostic anywhere; the candidate ranking covers only the 67 ctx-cont candidates,
  is |Spearman| (sign discarded, interactions/non-monotonicity/temporal structure
  blind), and spans 0.0000937–0.0272 where 1σ ≈ 0.0073 — everything below ~rank 8 is
  within one sigma of zero. Correctly non-authoritative, but it is the only
  quantitative feature evidence in the chain.

### P1 — Surface defects for the next feature wave (rebuild required)

Exact duplicates / dead information (rule 4: retire the restatement, keep primitives):

- **F-9 [M]** `ema50_200_spread_atr` ≡ `ema200_dist_atr − ema50_dist_atr`: 100.000%
  float32 bit-equal on all 5 lanes. Same identity on the local layer
  (`chart.local_ema50_200_spread_atr`). Escaped the duplicate sweep because the fields
  live in different declared tuples.
- **F-10 [M]** `chart.local_ema50_200_spread_delta_atr` ≡ `local_ema50_slope_atr −
  local_ema200_slope_atr` (bit-equal) — **introduced by the 2026-09-20 D-5 repair**,
  which replaced one exact affine duplicate with another. The sibling accel field
  survives only on a denominator mismatch.
- **F-11 [M]** 35 columns per surface are exact functions of retained columns within a
  6-bar window: `rsi14_delta_5` ≡ 50·(rsi14_centered[t]−[t−5]); the four RSI-cross
  flags; both mom20 sign-flips (all bit-exact, 0 mismatches, 678k rows). Same class
  the project retired four times. (Ages with long tails — rsi_extreme 364 bars,
  divergence 457 — carry non-recoverable info; keep.)
- **F-12 [M]** `vwap20_dist_atr` vs `ema20_dist_atr`: Pearson 0.95–0.97 on every lane —
  a cross-family near-duplicate invisible to the within-family detector. After D-4 the
  session_regime family's entire per-TF content (4 VWAP fields) is ~95% a restatement
  of trend_ema.
- **F-13 [M]** `ctx_cont.close_range_observed`: 1 on 99.977% of TRAIN, constant 1 on
  VAL — a dead bit on two surfaces. `d1_ema_stack_aligned_v2` constant 0 on all of VAL.
  `bars_since_swing_high/low` saturate at 42 bars (cap folds the tail).

Missing trader primitives (measured gaps, not opinions):

- **F-14 [M]** No 20/50 EMA cross family — fires 3.3× more often than the 50/200 the
  surface does carry (M5: 5,334 vs 1,609 events). No `price_x_ema20_cross` (one-tuple
  edit at htf_features.py:3188). `ema100` has dist but no slope on the lanes (its slope
  exists only in the canonical base at k=20 — the one non-shift(5) lookback, better and
  grandfathered).
- **F-15 [S]** No MACD line/histogram anywhere (EMA-12/26 not on lanes; not linearly
  recoverable); no stochastic %K/%D; no hidden (continuation) divergence quadrants —
  pivot-pair RSI values never emitted, so unrecoverable; no second RSI/ATR period on
  lanes; squeeze intensity (bandwidth percentile) never emitted.
- **F-16 [S] (corrected 2026-09-21, rule 2d)** The first publication of this finding
  overstated it as "112/178 fields reach the model on no timeframe" — **withdrawn**.
  All 178 per-TF fields (ADX, DI, BB, VWAP, level/candle/smc/geometry blocks included)
  DO enter the model as the 178-wide MTF sequence tensors on M15/H1/H4/D1 (Entry) and
  all five TFs (Exit), each behind a learned per-field gate (the (4,178) gate evidence
  proves consumption). What is true and remains: (a) 112 of the 178 names have no
  M5-LOCAL twin on the 240 signal surface and no ctx scalar projection — the compact
  current-bar decision surface consumes 35 of 890 cached columns, so the *scalar* route
  is narrow by design or by omission (not adjudicated anywhere); (b) the M5 slice of
  the cache is ~61% byte-identical recomputation of local layers and its remainder is
  consumed only by the Exit route's M5 lane. The GPU-waste reading of the original
  claim is therefore wrong; the routing-asymmetry reading stands.
- **F-17 [S]** 136 fields retired by hand-edited constants with no attached evidence
  artifact (incl. `dow_cos`, leaving `dow_sin` non-injective; all cross-TF ATR ratios).
  The pipeline's own exclusion path is dead code (`decision="available"` hardcoded,
  availability_v1.py:268) — `excluded_feature_count = 0` is structural, not measured.

Sign/semantics/consistency defects:

- **F-18 [M]** MTF collapses the swing-state distinction the local lane preserves:
  `mtf_smc_structure_bias` maps state 1 (broadening) and state 2 (contraction) both to
  0.0 on **38.8% of bars** (208,752 M5 rows); the availability mask is computed and
  discarded. `smc_choch` is direction-blind on the decision clock while MTF has
  up/down siblings; the derived choch age cannot recover side.
- **F-19 [M]** Five side-merged event ages lose the side of the last event
  unrecoverably: `smc_sweep_event_age_bars` (both lanes), `divergence_event_age_bars`,
  `rsi_extreme_event_age_bars`, `foundation_choch_event_age_bars` — against the
  surface's own signed-age convention shown in three other fields.
- **F-20 [M]** `_v1_atr14` (signal index 0) is raw USD: Spearman +0.58 vs row index,
  IQR ×3.14 first→last third — the one surviving epoch proxy, known since 2026-08-13,
  measured again today, never removed.
- **F-21 [M]** `d1_dist_change_1bar_atr_v4` differences an ATR-normalized series
  (two denominators); the ATR-change term alone flips the sign on 6.44% of rows —
  violates the repo's own stated raw-spread-difference rule.
- **F-22 [M]** `ema200_slope_atr` at k=5 carries ρ = 0.985–0.990 with `ema200_dist_atr`
  on every lane (sign agreement 96%) — a redundancy collapse; the k=1 repair of
  2026-08-19 mitigated but did not fix the slow spans. Span-proportional lookback
  (k≈span/4) drops ρ to ~0.84–0.89.
- **F-23 [S]** Two Bollinger-20/2 owners with different ddof ship in the same lane:
  `bb_width_atr` uses ddof=1, the squeeze's bandwidth uses ddof=0 — measured gap
  exactly √(20/19)−1 = 2.5978%. The squeeze header's single-identity claim is false
  for one of its two named referents. (Squeeze state itself uncorrupted.)
- **F-24 [S]** Undeclared second conventions: `h4_mid_ema50_dist_atr_canon_v2` and
  `D1_dist_from_ema200_atr` use mid-price where the family uses close; "VWAP" is
  tick-count-weighted close (min volume = 1.0 measured), not volume-weighted typical
  price; `vwap_local_cycle_dist_atr` name is stale after D-4 (the file's own comment
  says such a rename is mandatory); three TR conventions coexist (wilder_atr includes
  row-0 TR, ADX excludes it, atr50 is SMA-smoothed).
- **F-25 [S]** No DST handling: session boundaries are UTC-fixed while London/NY shift,
  so ~7 months/year every session label is 60 min off the liquidity structure. Known
  since 2026-08-13, untouched. Also: live `session_change_flag` (forbidden route)
  disagrees with the build at row 0.
- **F-26 [M]** VAL is a materially different vol regime than TRAIN (`d1_atr14_bps`
  standardized shift 1.85σ vs a 1.0σ policy bound, recorded SHIFT_OBSERVED,
  diagnostic-only) and only 5,509 rows. Every VAL-based claim inherits this.

### P2 — Architecture (model-side; changes = new candidate, not this one)

- **F-27 [S]** No stage runs attention over the flattened 32 (TF×family) tokens. The
  operator's cross-signal example (M15 EMA-kryss + H4-trend + S/R) **is** learnable via
  two-hop composition (cross-TF within family → cross-family within TF, plus a second
  post-collapse cross-TF hop), but the evidence must survive a 128-d per-token
  bottleneck between hops.
- **F-28 [S]** The three fusion scales (0.25/0.25/0.5) have **no nameable origin**
  (rule 2a: guessed defaults) — required CLI args whose literals appear uncommented in
  the materializers, frozen as buffers, never learned. And the throttling is
  asymmetric: `mtf_repr` enters the Q head unscaled (full 128-d quarter of the joint
  input) while the dedicated cross-family cooperation residual has no bypass —
  **the architecture's headline mechanism is its most-attenuated path**.
- **F-29 [S]** Entry and Exit implement the axial stages in opposite order under the
  same module names (`family_axis_attn` attends across TFs in Entry, across families
  in Exit); `cross_family_fusion_scale` is Entry-only; `exit_tf_gate` is a marginal of
  the cooperation simplex while Entry's is independently parameterized. Nobody has
  examined whether the reversal is intentional.
- **F-30 [S]** Family projections are field-count-bound: `Linear(4,128)` confines
  session_regime/vol_compression tokens to a rank-4 manifold vs smc_liquidity's rank
  48. Token/gate/attention budgets are flat (good), representational capacity is not.
  And session_regime's per-TF content is four VWAP fields — zero session-clock fields
  (name/content gap, rule 25).
- **F-31 [S]** Latent hazards: divergence-strength has an unproven mid-series NaN path
  (zero-ATR bar; not triggered on this tape); ADX kills a whole lane on one flat seed
  window with a misleading warmup error; squeeze fit/serve share the step but provably
  not the initial condition (contraction burn-in ~95–136 bars covers it in practice;
  untested in the repo); the D-3 release-age identity is false at the first warmed row
  (censored away on this artifact); registries hard-raise on zero ATR where the EMA
  family emits NaN; `geomchan_pos_0_1` clips channel overshoot to [0,1] against the
  no-clip contract.

## 3. The attribution protocol (preregister BEFORE reading any number)

Home: `evaluate_entry_candidate_selective_edge_v1` (rule 21 — the schema slot
`feature_mask_ablation` already exists; the null/HAC/coverage machinery lives there;
seed-stability already imports from it). NOT the usefulness audit (wrong metric space,
iid SEs, degenerate donor, 205k forwards).

Preconditions: trained candidate (not smoke) · ≥5 seeds all `mixed_raw_q_actions` ·
re-derived exact coin-flip null (already mechanized as `coin_flip_mean_pnl_bps`).

Arms: 1 control + 8 local families + 32 family×TF (+8 family-across-TFs optional),
enumerated from `require_multi_tf_specialist_routing_v4`, hash-bound, one immutable
event for the whole matrix. Intervention must be *declared* (resolve F-3): mean-
substitution under the fitted normalization, not raw zeros.

Per arm × coverage × seed: paired Δbps on the fixed control population (HAC SE on the
paired series) **and** re-selected Δbps (separately — the population trap of 2026-06-23
false-refutations); decision-flip rate with sign split; abstention quality (the V29
success criterion); MAE/drawdown (from persisted `side_mae_bps`, today unaggregated);
gate-conditioned covariates from the existing parquet (free).

Nulls: (1) family-irrelevance, |Δbps| > 2·HAC-SE; (2) cardinality-matched random-block
null (≥200 draws, matched on surface and field count — otherwise smc's 48 fields vs
session's 4 makes any comparison a cardinality artifact), compare vs p95. Decision:
edge-bearing at coverage c iff both hold on ≥4/5 seeds at a c ≤ 25% the control itself
passed. Multiple-comparison rule committed in advance. **No family is retired by this
protocol** (rule 4). Stated up front: this VAL can attribute at most one or two
dominant families (F-6); a failing family is "not demonstrated at this power," never
"useless." Cost: 41 × one VAL forward pass ≈ hours under the producer guard.

## 4. Recommended sequencing

1. **Now (running):** FULLSUP candidate continues; epoch-1 VAL tells us if learning
   moves at all. Nothing in this review touches it.
2. **Attribution enablers (code wave, no rebuild):** reactivate the evaluator's
   feature-mask arm per §3; add Entry gate evidence to bundles (mirror the Exit
   contract, F-7); fix the usefulness donor plan (F-4); resolve the counterfactual
   contradiction (F-3). These change no feature bytes — recipes re-materialize but
   datasets stand.
3. **Next surface wave (one rebuild, batched):** retire F-9/F-10/F-11 exact duplicates;
   add F-14 (20/50 cross family + price_x_ema20) and the F-18/F-19 side-preserving
   repairs; kill the F-13 dead bits; F-20 (`_v1_atr14` → bps or retire); F-21; the
   F-24 renames/declarations; decide F-15 additions (MACD-hist, stochastic, hidden
   divergence, squeeze intensity) and the F-16 routing question (which of the 112
   orphans earn a seat — ADX/DI/BB/VWAP first). One rebuild, one readiness ladder.
4. **Architecture candidate (after attribution data exists):** F-27/F-28/F-30 are
   testable hypotheses once the ablation matrix runs — e.g. "does widening the
   cooperation path beat 0.25?" becomes a measured question, not a guess.
5. **F-25 (DST)** is a data-semantics decision for the operator: fixing it changes
   every session label historically; it must be its own preregistered comparison.

## 5. Not examined (stated uninvited, rule 25a)

The M1 Exit lane's candle/SMC emissions end-to-end; `MULTI_TF_SHIFT` availability
arithmetic (the HTF-alignment leak surface); level/trendline registries' D-1/D-2/D-6
open items from 2026-09-19 (not re-measured); shift(1) staleness split (2026-08-13,
never re-measured); whether the tiny measured family deltas (0.054 bps) reflect the
under-trained checkpoint, the max-over-8 statistic, or weak zero-masking — the single
cheapest open question; oracle/available-skill re-derivation (no owner exists);
normalization-fit internals; whether the Entry/Exit axial reversal is intentional;
d_model/n_heads/num_layers origin (same rule-2a exposure as the fusion scales).
