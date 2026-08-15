# Indicator fidelity audit — 2026-08-13

> Historical-scope notice, 2026-08-14: this document records the 608/158/189
> audit substrate and must not be read as the active contract. Current schema,
> dimensions, retirements and empirical status are owned by `AGENTS.md`,
> `SYSTEM_MAP.md` and `HANDOVER_XAU_DIRECTION_REPAIR_20260714.md`; do not infer
> them from any count or proposed surface below. No current-contract dataset,
> model, OOS/edge result or tick-resolution claim is established by this
> historical audit.

Five parallel read-only audits over EVERY emitted field in the surface
(608 signal / 158 ctx_cont / 5 ctx_cat / 189×5 per-TF), one question per
field: **does it carry what the real trading concept carries, or only its
name?** Verdicts: FULL (carries the concept) · PARTIAL (carries some) ·
NAME-ONLY (the name promises a mechanism the formula does not implement) ·
ABSENT (concept referenced nowhere in code).

Evidence classes per CLAUDE.md rule 2d/25a: `[M]` measured on the sealed
V29J artifacts · `[PS]` proven from source/algebra · `[J]` judgment ·
`[U]` unproven. **No audit opened a parquet** — every saturation and
degeneracy claim below is source+algebra and needs one measurement pass to
become `[M]`. That pass is listed as step 0 below.

This document is the durable answer to "are all indicators fully exploited
to their fullest?" The answer is **no**, per field, with the upgrade.

---

## 0. THE FINDING THAT BLOCKS EVERYTHING ELSE

**Both V29 registry tolerances are TRAIN-fitted on the wrong population,
and the fitted values are degenerate in opposite directions.** `[M]`+`[PS]`

Measured from the sealed V29J cache manifest (`v29_registry_constants`,
q=0.5, 2019-01-01 → 2026-05-31):

```
level_tol_atr   M5 0.009383  M15 0.015254  H1 0.028561  H4 0.055114  D1 0.138842
trendline_band  M5 1.507070  M15 1.406482  H1 1.376171  H4 1.285108  D1 1.220651
entry_m5        seq_len 96, band 1.290411
```

**0a. The level tolerance is ~0.0094 ATR ≈ 1 cent on XAUUSD M5.**
`fit_level_registry_tolerance` accumulates `earlier_prices` over the ENTIRE
TRAIN window and never prunes (`level_registry_v1.py:568-591`); the sample
is "distance from each pivot to the nearest of ALL previously seen pivot
prices" — `n_pivots_admitted=101,761` over a ~2,200 USD range, so the
nearest-neighbour median is ~1-2 cents by construction. **The statistic
tracks TRAIN window LENGTH, not a market property: double the window,
halve the tolerance.** The runtime population is entirely different — only
levels within `AGE_CAP` survive (M5: 240 bars ⇒ tens of levels, not 102k).
Rule 2f satisfied (sampling error reported); **rule 2g violated** (the fit
is not taken where the decision is made).

Consequences, all `[PS]` given the measured constant:
- merge requires |price − center| ≤ 0.0094 ATR ⇒ **pivot clustering never
  happens** ⇒ `member_pivot_count ≡ 1` ⇒ the V30 `test_count` split is
  `touch_count − 1`, an additive constant, not new information;
- break = first close ~1 cent past a pivot ⇒ `level_break_*_event` is a
  pivot-crossing detector, not a level break;
- **the level registry is a per-pivot crossing engine wearing a
  clustered-S/R name.** `level_kind = pivot_cluster` has no cluster.

**0b. The trendline band is the median of the NULL population.**
`fit_trendline_tolerance` measures deviation over EVERY ordered same-side
pivot pair with no validation filter (its docstring says so:
"threshold-free by construction"), then promotion at serve requires
`deviation ≤ band`. Setting the threshold at the median of exactly that
distribution means **≈50% of random 2-pivot pairs promote to ACTIVE** —
the "≥3-touch chartist validation" carries ~0 bits. `[M]`
`n_candidates_measured=874,314`. Band 1.29 ATR ≈ 1.5 USD on M5, wider than
most swings the line connects; the same band drives channel membership, so
`geomchan_active` is near-permanently 1 `[U]`.

**Why this blocks:** the V29J dataset was sealed with these constants, and
the V30 chain re-fits them the same way. Any conclusion about the event
surface's value is a conclusion about degenerate constants, not about the
mechanism. **Fix before the next chain run** — it is a fit-population
change in the existing owner, not new mechanism (rule 22's preferred order).

**REPAIRED 2026-08-13 (V30 package 6), in the existing owners:**

- **0a fixed.** `fit_level_registry_tolerance` now prunes its search set to
  the runtime merge's own window: at confirming bar `t` only pivots with
  `t − pivot_bar ≤ LEVEL_REGISTRY_FIT_AGE_CAP_BARS[tf]` are searchable (the
  engine's existing per-TF expiry cap; the `m1` fit lane maps to the M5 cap
  because the Exit lane executes the m5-bound block). `q` is unchanged (the
  recipe input) and the rule-2f bound is unchanged. The provenance payload now
  publishes `fit_population`, `age_cap_bars`,
  `searchable_pivot_population_mean` / `_max` and
  `n_pivots_dropped_no_in_window_neighbour`, so the searched-set size is
  auditable in every frozen manifest and can never silently track TRAIN
  length again. Caveat stated in the owner: a level whose zone is re-entered
  has its `last_touch_bar` refreshed and lives longer at runtime; the fit
  cannot replay touches without a tolerance it has not fitted yet, so the
  fitted population is a strict *subset* of the runtime one (exact for
  untouched levels), never a superset.
- **0b published, not moved.** No new quantile was invented. The band is still
  the median of the first-subsequent-pivot deviation population, and the
  payload now publishes `implied_validation_rate` — the measured share of
  arbitrary 2-pivot pairs that the fitted band promotes (~0.5 by construction,
  ≥0.5 exactly by the definition of a median). Conditioning the fit population
  on "pairs that became lines" is **circular** and was therefore not
  implemented: at serve a pair becomes a line exactly when a third pivot lands
  within the band, so selecting that subpopulation needs the band being
  fitted. Moving the statistic is now an operator decision with a measured
  rate to choose against.
- **Not repaired, found while repairing (`[PS]`, new):** the band fit records
  each candidate's FIRST subsequent same-side pivot and then drops the
  candidate, but serve keeps a non-promoted candidate alive and retests it
  against every later same-side pivot inside the `seq_len` window
  (`_ingest_confirmed_pivot` keeps `~promoted`). The fitted null population is
  therefore not the serve population, and the true serve promotion rate is
  *higher* than the published `implied_validation_rate` — the ~0 bits finding
  is, if anything, understated. Fixing it changes the statistic (a per-
  candidate minimum over its live window) and is left as a declared open item.
- Both fits are re-run by the chain, so no artifact was rewritten; the V29J
  frozen constants are now stale by construction, which is the intended
  outcome.

---

## 1. NAME-ONLY — the name promises what the formula does not compute

### 1a. Chart geometry: 15 of 18 MANDATORY-PINNED fields, 0 FULL `[PS]`
`CHART_GEOMETRY_MODEL_NATIVE_FEATURE_NAMES` is pinned un-removable, and it
contains **no field with full concept coverage**. Proven by construction:
the layer's input tuple is pre-reduced scalars — no price, no pivot
coordinate, no bar index, no slope, no anchor. Two points define a line;
the layer receives one distance per side. So every field named
`trendline` / `rail` / `channel` / `triangle` / `flag` / `apex` is
NAME-ONLY *by algebra*:
- `trendline_break_up/down_pressure` — no line; "break" is `smc_bos_up`, a
  horizontal swing break. **Plus a sign inversion**: the sided-proximity
  terms return 0 once price passes the level, so the field peaks BEFORE
  the break and decays AT it.
- `mtf_channel_retest_*_quality` — no retest; the "window" is `_lag1`
  (one bar), and nothing tests it is the SAME level that broke.
- `rising_support_rail_*` (4 fields) — no rail, no slope; "rising" is the
  sign of a 9-term EMA blend. The "short trap" variant contains no failure
  and no breakdown — it is a strictly more bullish rescaling of its twin.
- `ascending_triangle_pressure` vs `descending_triangle_pressure` —
  identical formulas except the EMA sign; nothing tests flat resistance,
  rising lows, touch counts or convergence.
- `h8_proxy_trend_score` — **there is no H8 timeframe in the engine.**
- `fib_extension_*` (3) — **algebraically impossible**: `fib_position` is
  `clip01`-ed, an extension is by definition >100%. `fib_golden_zone` is
  mislabelled (0.5-0.618 vs the classic 0.618-0.786). The 5 fib columns
  carry ONE degree of freedom (deterministic shifts of one scalar through
  `exp(-12·|p-k|)`), and `fib_position` itself is a hand-weighted blend of
  a real retracement (55%) with a premium proxy and a 20-DAY range
  position — quantities on different ranges and different clocks.

**The real implementations exist and are ignored.** V29 shipped
`trendline_registry_v1` (fitted 2-anchor lines, slope, touch counts,
ACTIVE/BROKEN, real retest window, apex solve for converging lines) and
`level_registry_v1` (touch counts, break, retest hold/fail) on 2026-08-11.
The proxy layer's version string is 2026-08-09 — **it was never reconciled
with its own replacements.** The upgrade is wiring, not new code.

### 1b. SMC: the core concepts are absent or renamed `[PS]`
- **Order blocks, fair value gaps, equal-high/low pools: ABSENT.** Zero
  producers in the tracked tree (the only FVG code is in a detached
  worktree, env-gated, unreachable). What `smc_v1` owns is ONE fractal
  pivot engine: every SMC field compares `close/high/low` against at most
  four pivot prices. No object identity, no mitigation, no volume.
- **`smc_choch` is NAME-ONLY.** A real change-of-character is the break of
  the last opposing swing while a trend is in force — level, direction,
  displacement. Here there is **no break test at all**: it fires when a
  newly confirmed pivot flips the HH/HL→LH/LL pattern, ≥3 bars after the
  pivot and typically long after price broke. On M5 it is **unsigned**.
- **BOS carries no displacement** — 0/1 flag; a 0.05-ATR scratch and a
  3-ATR break are the same number. The displacement IS computed for the
  geometry sibling (`mtf_geometry_*_break_displacement_atr`) and routed to
  a DIFFERENT encoder, so the SMC specialist never sees it.
- **`smc_sweep` is half a sweep** — real 1-bar false break with
  close-back-inside, but the level is a 7-bar fractal (not a pool),
  multi-bar sweeps are invisible, there is **no de-duplication** (the same
  level poked five bars running fires five times and resets the age, while
  BOS was explicitly made one-shot), and the level is never marked
  consumed.
- **premium/discount is a rolling proxy**, not an anchored dealing range:
  it re-anchors every time a 7-bar fractal confirms, and is clipped to
  [0,1] so a breakout saturates with no "trading outside the range".
- **"liquidity" is distance-to-extreme, not a pool**:
  `liquidity_hi/lo_nearest_abs_atr` take `min` over five per-TF distances
  then `abs()`, destroying the sign the producer deliberately used to mean
  "already swept". No pool size, no equal-high clustering, no age.
- **"reclaim" has no producer.** 12 fields are named after an event
  nothing computes; "reclaim" everywhere means "the sweep bar closed back
  inside".

### 1c. Session/vol: 57 of 68 session fields are pre-fused votes `[PS]`
Only **5 of 68** carry information the fusion cannot form from inputs it
already has (best: `spread_cost_ratio` = cost per unit expected movement).
The rest are mask × pressure products and hand-weighted votes — the exact
`mtf_confluence` class rule 4 retired on 2026-08-05. One is literally
named `regime_transition_abstain_score`: a hand-written abstain vote.

### 1d. sr_memory: 34 fields that never read the level registry `[PS]`
"Repeated test" / "memory" are EWM decays (0.82/0.96, **origin-less**) of
a continuous, always-positive proximity blend over an identity-less "level"
(a `max` over five different level KINDS whose winner changes bar to bar).
There is no discrete touch to remember and nothing whose identity persists.
`SUPPORT_RESISTANCE_MEMORY_SOURCE_FIELDS` contains **not one `level_*`
field** — the V29 registry with genuine `touch_count`, `test_count`,
`bars_since_touch`, `mean/max/last_reaction_atr` was shipped BESIDE a
34-field layer that fabricates the same concepts from proximity blends.

### 1e. `struct_*_v3` (28 ctx fields): momentum wearing a structure name `[PS]`
The producer's own comment says "HH/HL/LH/LL via mom_5 + mom_20 signs".
There is no pivot, no high, no low anywhere in it: four boolean sign
quadrants of `(close−close[−20], close−close[−5])/atr` plus their
magnitude ratio, ×5 TFs. They are routed to the STRUCTURE specialist,
they define `mtf_structure_agreement/divergence`, and they leak into the
foundation `pullback` term that four more structure fields depend on —
**contamination three owners deep.** Also: `struct_tf_agree_count_v3` is a
mean named `count`; `struct_smc_swing_x_dip_v3` ordinal-encodes an
UNORDERED enum where the warmup sentinel is the maximum value;
`struct_all_tf_pullback_v3` is produced and not in the contract (dead).

### 1f. Candles: context-free by construction `[PS]`
`CANDLESTICK_PATTERN_SOURCE_FIELDS = (time, open, high, low, close)`.
Therefore, for all 60 fields (×5 TFs = 300 mandatory columns), as algebra
not preference: **no volume** (it exists in the engine and is withheld),
**no level/location**, **no size-vs-recent-range** except four
`*_vs_prev5` fields that gate NOTHING, and **maximum 2 bars of prior-trend
memory**. A hammer is only a hammer at a level after a downtrend; the
layer cannot see either. Firing condition: lower wick > 16.3% of range
⇒ ~60% of the feasible domain `[PS exhaustive]`, consistent with the
measured ~51% `[M]`.
`tail_rejection_risk` is **exactly 1.0 over 66.7% of its feasible domain**
(exhaustive simplex algebra) — a censored copy of `1 − body_share`, which
is already column 1. ×6 columns (M5 + 5 TFs), dead.

---

## 2. DEAD MATH — provably inert terms `[PS+U]`

**2a. Wrong transform on the wrong family.** `center_atr_ratio =
tanh(log₂(ratio))` is correct for SAME-TF `ATR14/ATR100` (centred at 1).
It is applied to CROSS-TF ratios, which centre near 0.58/0.14/0.10/0.20
because ATR scales as √(bar duration). So `log₂` is always negative,
`tanh` always negative, `_pos(...)` ≡ **0**. Consequences:
`short_tf_vol_expansion_pressure` has **70% of its weight ≈0**,
`higher_tf_vol_expansion_pressure` **75%**, and `vol_term_structure_slope`
is a **constant ≈ −0.43** — a vol term structure that can never cross zero,
i.e. can never say "short-dated vol above long-dated".

**2b. A 44,721× unit mismatch inside one expression.** `rvol_20` is
bps×√20 ≈ 20 into `tanh(x/2.5)` ⇒ **constant 1.0**; in the SAME 7-term sum,
`_v1_pk_sigma20` (dimensionless ≈3e-4) through the SAME scale ⇒ pinned at
~1e-4. The `rvol_20` saturation repeats in three families (vol, foundation,
chart) — a rule-25b sweep case.

**2c. Dead at the only consumer.** `_v1_close_ema_slope_3` (~5e-5) into
`tanh(1.0)` contributes ~6e-6 to a ±1 composite;
`_v1_bb_bandwidth_delta_10` (noise-amplifier repaired in V30 pkg 1, scale
NOT repaired) ~1e-4 into `tanh(1.0)`.

**2d. Resolution thrown away.** `spread_bucket_pressure` maps TRAIN
quintiles 2, 3 AND 4 to 1.0 — 60% of rows including every rollover spike
become one value, while the ATR sibling two lines later is correctly
graded.

---

## 3. THE LAST ERA PROXY, AND TWO CLOCKS

**3a. `_v1_atr14` is raw USD** — no `/close`, no bps `[PS]`. Gold ran
1454→5588 over the tape, so the field's LEVEL is the date. The 2026-08-09
wave repaired every sibling to bps or ATR-multiples and never touched this
one — and it is field #1 of the contract-frozen BASE 34. Normalization
recenters but cannot remove a monotone era trend, and guarantees a
TRAIN↔VAL/TEST distribution shift. This is the defect class that produced
the verdict "the probes measured broken encoding, not the market".

**3b. Two bar clocks inside one contract** `[PS]`: 15 of the 34 BASE
fields are `shift(1)` (one M5 bar stale), 19 are on the decision bar. Both
causal; the shift is a legacy device the decision-delay contract now
duplicates. Composites that mix them compute cross-bar interactions while
their names claim same-bar.

**3c. A third daily clock survives** `[PS]`: `_session_vwap` resets at
midnight UTC while D1 now opens 22:00 (V30 pkg 3) — and those four VWAP
fields × 5 TFs are the ENTIRE per-TF content the session specialist
receives besides `regime_class_id`.

**3d. DST: absent, and the consequence is systematic** `[PS]`. Boundaries
are UTC-fixed year-round. London is UTC+0/+1, New York UTC−5/−4, and the
transitions are ~3 weeks apart. For ~7 months a year every session mask —
and therefore all 51 pre-fused products that multiply by one — is
phase-shifted 60 minutes against the liquidity structure it claims to
encode. The LBMA PM fix (the largest scheduled XAU liquidity event) reads
as `session_age 0.75` in winter and `0.50` in summer. A systematic
seasonal label error, invisible to every gate.

---

## 4. LIVE BUGS INTRODUCED BY EARLIER REPAIRS `[PS]`

**4a. `hl_state` / `ll_state` are built from the CONTRADICTING state.**
`smc_swing_state` enum: 0=HH+HL, 1=HH+LL, 2=LH+HL, 3=LH+LL, 4=tie/warmup.
The foundation layer names them `up_bias=state1`, `down_bias=state2` and
weights `up_bias` (HH **and lower low**) at 0.55 into `hl_state` — evidence
AGAINST a higher low — and `down_bias` (LH **and higher low**) at 0.45 into
`ll_state`. Before the 2026-08-09 partition repair states 1/2 were
unreachable and the mislabel was inert; **the repair made them live.**
Five emitted foundation fields carry it, `structure_up_minus_down`
inherits it, and all 28 `chart.structure_swing_*` derivations consume it.
Fix = re-indexing an existing enum, zero new numbers.

**4b. `_v1_atr14`, the sweep de-dup asymmetry and the M5-vs-MTF SMC
divergences** (BOS re-fire, CHoCH sensitivity/direction, PD envelope and
warmup fill) are all cases where a fix landed in one owner and not its
sibling — rule 25b exists because of exactly this.

---

## 5. STATE THAT EXISTS IN MEMORY AND IS NEVER EMITTED `[PS]`

- `armed_high` / `armed_low` in `swing_structure_v1` is EXACTLY "the last
  confirmed swing high/low has not been closed through" — **structure
  intact**. Set on adoption, cleared on break, never emitted.
- `level_registry` keeps `status ∈ {active,broken}` and `retest_state ∈
  {none,pending,held,failed}` per level and emits neither; a broken level
  VANISHES from the surface except through a 999-capped age.
- `trendline_registry` keeps `state ∈ {candidate,active,broken}` and emits
  only ACTIVE masks.
- **Polarity flip** ("old support is new resistance", the most-used S/R
  construct there is): on retest-hold the trendline registry marks
  `resolved = True` and **deletes the line on the bar it proved itself as
  flipped resistance**; the level registry keeps `status="broken"` forever
  and excludes it from the nearest-ACTIVE scan. PARTIAL as a 1-bar event,
  ABSENT as an object.
- Only 2 of 4 structure run-counters exist (`consecutive_higher_lows`,
  `consecutive_lower_highs`) — so **"uptrend structure, unbroken for N
  pivots" is not formable from any emitted field.**

---

## 6. WHAT IS GENUINELY GOOD (so it is not broken by the repair wave)

- `swing_structure_v1` — 14/14 FULL. Real pivot arithmetic, honest NaN
  prefixes, correct edge semantics, ATR-normalized displacement.
  `consecutive_higher_lows_count` is the best structure field in the
  surface.
- The level registry's **reaction measurement**: signed 12-bar lift-off
  from the touch bar with the touch bar's ATR and center frozen — a touch
  that never lifts scores negative. The strongest single mechanism audited.
- The trendline registry's **channel geometry**: real apex intersection,
  strict convergence test, algebraic midline slope. `geomline_*_age_bars`
  counts from the anchor's CONFIRMATION bar — causally exact.
- V30's `geomline_*_active_count` (graded occupancy) genuinely fixes the
  saturated-mask problem and IS the density feature.
- `regime_v4` V29 flip block — named origin for every number, mirrored
  formula owner, explicit refusal to build a cross-TF aggregate citing
  rule 4. The template the other families should follow.
- `entry_volatility_semantics_v1` — highest-quality owner audited; its
  problem is under-adoption (two of its five functions are unused while
  consumers open-code worse variants).
- V30 packages 1-3 verified in place: the three noise-amplifier repairs,
  DI spread, RSI velocity, divergence strength, Kaufman ER, M5-local ages
  and cross events, the unified session clock, the 22:00 D1 origin (stub
  rate 16.99% → 0.051% `[M]`), and the spread-dynamics block (FULL, no
  defect found).
- `atr_bucket` / `spread_bucket` are textbook rule-18 TRAIN-fitted;
  `vol_regime_id` is a proper causal rolling quantile.

---

## 7. CONSTANT INVENTORY (rule 2a) `[PS]`

| owner | unsourced magnitudes | fitted/derived constants |
|---|---|---|
| `entry_session_regime_interactions_v1` | ~115 | 1 (`SPREAD_RATIO_TANH_SCALE_TRAIN_P90`) |
| `entry_vol_compression_v1` | ~120 | 1 (an algebraic max, derived in-comment) |
| `entry_smc_liquidity_quality_v1` | ~40 | 3 (2 algebraic maxima + one weight tuple) |
| `entry_chart_geometry_v1` | ~60 incl. the fib kernel 12.0 and the blend 0.55/0.30/0.15 | 0 |
| `entry_candlestick_patterns_v1` | ~25 incl. doji 0.18, hammer 1.8/0.7/0.35 | 0 |
| `entry_support_resistance_memory_v1` | ~60 blend weights + 2 EWM decays | 4 derived algebraic bounds (exemplary) |
| `augment_forward_outcome_v2` (dip/struct) | 2 (`/2.0`, `·5.0`) + 5 liquidity lookbacks | 0 |
| `regime_v4_features` | 2 (`0.8` maturity, `±5` ROC clip) | rest derived |

`0.50` appears 27 times in one file with no origin. The `0.8` maturity
threshold is `[M]` **the top-1 ranked feature overall** and `[M]` has
**exactly zero variance on the June VAL split**.

---

## 8. UPGRADE ORDER (rule 22: measure → recipe value → extend owner → build)

**Step 0 — MEASURE (blocks everything; ~10 min each, capped audits).**
1. Registry degeneracy: per-bar histograms of `level_above_touch_count`,
   `level_above_test_count`, `member_pivot_count`, `geomline_*_active_count`,
   `geomchan_active` on the sealed V29J TRAIN rows. Settles §0.
2. Saturation sweep: activity/variance of `_pos(atr_ratio_*)`,
   `rvol_unit`, `sigma_pressure`, `vol_term_structure_slope`,
   `tail_rejection_risk`, `spread_bucket_pressure`. Settles §2.
3. `_v1_atr14` TRAIN-vs-VAL distribution shift. Settles §3a.

**Step 1 — recipe-value fixes (no new mechanism).**
4. ~~Re-fit `level_tol_atr` on the AGE-CAP-pruned pivot population (the set
   the runtime merge actually searches); publish `implied_validation_rate`
   for `band_atr` so the null-population defect cannot recur silently.~~
   **DONE 2026-08-13 (V30 package 6)** — both in the existing owners, no new
   constant; see the REPAIRED block in §0, including the one open item it
   uncovered (the band fit's first-pivot-only population vs serve's retested
   candidates).

**Step 2 — one-line repairs, zero new numbers.**
5. `hl_state`/`ll_state` enum re-indexing (§4a).
6. `_v1_atr14` bps sibling in the mandatory causal layer (the frozen
   34-tuple stays untouched — the V30 Kaufman-ER precedent).
7. The stale `/5.0` divisor; the graded spread ladder; the unit repairs in
   §2b/2c via the existing TRAIN-fit precedent.
8. Session-anchored VWAP → `multi_tf_bar_label` (kills the third clock).

**Step 3 — emission only (state already in memory).**
9. `swing_{high,low}_level_intact` from `armed_*`; the two missing run
   counters (§5).
10. `level_*_member_pivot_count` + second-nearest slot (density);
    measured-move from break-bar state.
11. Polarity flip as a state-machine edge in both registries.
12. BOS/CHoCH displacement; sided sweep depth; sweep de-duplication.

**Step 4 — rewire to the real owners (deletes unsourced constants).**
13. `sr_memory` repeated-test/respect/break → registry `test_count` /
    `bars_since_touch` / `last_reaction_atr`; retire the two EWM decays.
14. Chart geometry's 15 NAME-ONLY fields → the registries that already
    emit the real thing.
15. `struct_*_v3` → honest name (`mom5_mom20_quadrant_*`); structure
    consumers → the per-TF `swing_*_sequence_delta_atr` already emitted.
16. Candle gates: `hammer × range_expansion_vs_prev5` (free), level
    proximity from the registry (already computed in the same function),
    volume confirmation on the follow-through fields.

**Step 5 — retire the hand-fused rulebook (rule 4, mtf_confluence
precedent), AFTER the pre-registered evaluation verdict.** The 51 session
products + 7 votes, the 24-field SMC quality layer, the 6 candle
aggregates, the geometry composites, the 6 provable duplicates. Every
consumed input remains a model input; this removes ~300 unsourced
magnitudes in one move.

**Step 6 — genuinely missing primitives** (each with a home and existing
constant origins): reclaim event · volume confirmation (per-TF volume-z,
volume-at-level, volume divergence via the existing pivot-pair machinery) ·
squeeze state/duration/release latch/box · pool identity (equal highs as
a ≥2-member cluster) · session-anchored levels (engine ready, kind fenced) ·
order blocks / FVG (registry gives them a home, rule 21).

---

## 9. WHAT NO AUDIT EXAMINED (rule 25a, stated uninvited)

- **No field's real distribution.** Every saturation, degeneracy and
  "near-always 1" claim is source+algebra. Step 0 converts them to `[M]`
  or kills them.
- The Exit/M1 lane's consumption of these families.
- Whether the live path reproduces these formulas bit-identically (that is
  the train==serve gate's territory — and per rule 25 it proves
  consistency, not fidelity).
- The normalization statistics fitted over any of these fields.
- Every "would pay" claim is `[U]`. The 2026-08-09 walk-forward refutation
  stands; this audit argues representability and correctness only.

---

## 10. STEP-0 MEASUREMENTS — verdicts on the audit's own claims (2026-08-13)

Measured on the sealed V29J bytes (TRAIN n=369,303, the exact declared
decision rows; entry-M5 registry replay reproduced all 22 sealed level
fields with max |diff| = 0.0). Raw logs:
`GX1_DATA/logs/v30_step0_measurements_20260813/`.

**CONFIRMED**
- The level registry is a per-pivot crossing engine: **71.9% / 70.8%** of
  present rows carry `touch_count == 1`, and when it is 1 the level's only
  touch is its founding pivot (`bars_since_touch − age_bars == 3` on
  **100.000%** of those rows). `level_break_*` fires on 5.7% / 5.3% with
  `level_broken_touch_count` p75 = 0.
- The ≥3-touch line validation carries ~0 bits: a line touch fires on
  **29.85%** of rows (one per ~3.4 bars), line `touch_count` p50 = 5,
  and `geomline_max_dev_atr` saturates exactly at the fitted band 1.2904.
- Geometry occupancy on the entry lane: `geomline_above/below_active`
  **98.70% / 98.72%**; M15→D1 95.9→100.0%. (Per-TF M5 is only 43.3% —
  its `seq_len` is 16, not 96.)
- `higher_tf_vol_expansion_pressure`: 75% of its weight is dead on
  **every one of 369,303 rows** — `_pos(atr_ratio_m15_d1)` and
  `_pos(atr_ratio_h1_d1)` are **exactly 0 rows**. Field max 0.2273 of [0,1].
- The unit mismatch: measured magnitude ratio **45,004** (algebra predicted
  44,721); `tanh(rvol_20/2.5)` is exactly 1.0 on **29.9%** and ≥0.999999 on
  **51.5%**; `_v1_pk_sigma20` p50 = **1.64e-4**.
- `spread_bucket_high_pressure` is exactly 1.0 on **60.0011%** of rows.
- `_v1_atr14` is an era clock: pearson **+0.4945** vs time, spearman
  **+0.6009**; TRAIN mean 2.177 → VAL 5.693 (**z_val +1.331**); VAL p1 sits
  above TRAIN p50.

**SOFTENED / REFUTED — the audit overstated these**
- "`member_pivot_count ≡ 1`, clustering never happens": member==1 on
  **98.03% / 98.32%** of present rows, not always. Clustering happens on
  ~2%. The re-fit is still justified; the absolute was wrong.
- "V30 `test_count` is `touch_count − 1`, an additive constant": the
  identity is **violated on 1.97% / 1.68%** of present rows — it carries
  genuinely new information there.
- "`vol_term_structure_slope` is a constant ≈ −0.43": **REFUTED** — std
  0.0880, 360,024 unique values. What IS true: it is positive on only
  **0.077%** of rows, so it can never say "short-dated vol above
  long-dated" in practice.
- `_pos(center_atr_ratio(cross-TF))` ≡ 0: true for 3 of 4 pairs;
  **`atr_ratio_m5_m15` is alive on 1.205%** of rows (max 0.882).
- `candle.pattern_tail_rejection_risk` is **not in the sealed 592** — the
  TRAIN ranker already dropped it (candidate #200 of 210, score 4.7e-4).
  The rankable pool works.
- The `_v1_atr14` bps repair removes only **26.8%** of the standardized
  shift and **39.4%** of the era correlation — the residual is real
  volatility-regime change, not a unit defect — and the proposed sibling
  is **0.996-correlated** with `ctx_cont.atr_bps`, already a model input.
  Re-scope: this is not the win the audit implied.

**INCIDENTAL, unrequested (rule 25a)** — a cold-start replay of the per-TF
level block does not reproduce the sealed cache for M5/M15/H1 (the sealed
cache carries no NaN warmup prefix where a cold start emits 15; a handful
of rows differ: 6/30 of 369,303 on M5). H4 and D1 reproduce bit-for-bit.
Plausibly benign (longer warmup history feeding the cache build), but no
gate currently asks this question. Separately: the V30 22:00 D1 origin
yields a **completely different D1 grid** (1,734 vs 2,088 bars, zero shared
labels) — expected, and it means V30's D1 features are a new population,
not a perturbation of V29's.

---

## 10. V30 PACKAGE 7 — OPERATOR-AUTHORIZED REMOVALS PERFORMED (2026-08-13)

Step 5 of §8 is taken for the two owners the operator named. Recording what
IS and is NOT verified (rule 25c).

**Removed — 43 chart-geometry columns** (`entry_chart_geometry_v1`, layer
58 → 15, version → `entry_chart_geometry_v5_20260813_name_only_composite_removal`):
`h8_proxy_trend_score` (no H8 exists in `MULTI_TF_RESAMPLE_RULES`);
`ema_stack_bull/bear_pressure` (products of two still-emitted fields, zero
consumers); the five exact-affine duplicates of the two sided stacks
(`support_minus_resistance_stack`, `major_level_proximity_max`,
`channel_position_low_to_high`, `channel_center_bias`, `channel_edge_pressure`);
the whole Fibonacci block (15: `fib_position_proxy`, the five
`fib_retracement_*_proximity`, `fib_golden_zone_proximity`,
`fib_pullback_long/short_pressure`, `fib_extension_breakout_up/down_pressure`,
`fib_support/resistance_confluence_*_pressure`, `fib_extension_exhaustion_risk`);
and the remaining NAME-ONLY composites (`support_bounce_long_pressure`,
`resistance_reject_short_pressure`, `trendline_break_up/down_pressure`,
`ascending/descending_triangle_pressure`, `bull/bear_flag_pullback_pressure`,
`line_pattern_tail_risk`, `trendline_channel_confluence_pressure`,
`channel_edge_rejection_pressure`, the four `*_rail_*`,
`triangle_apex_compression_pressure`, `flag_breakout_readiness_pressure`,
`mtf_channel_breakout_up/down_quality`, `mtf_channel_retest_long/short_quality`).

**Removed — 7 candlestick columns** (`entry_candlestick_patterns_v1`, layer
60 → 53, version → `entry_candlestick_patterns_v3_20260813_vote_and_affine_duplicate_removal`),
each also on all five per-TF `mtf_pattern_*` lanes: the six aggregate votes
(`bull/bear_reversal_pressure`, `bull/bear_continuation_pressure`,
`indecision_breakout_setup`, `tail_rejection_risk`) and the exact affine
duplicate `close_pressure_signed` (= 2·`close_location` − 1, clip inactive).

**Derived counts after the removal** (owner tuples, never restated literals):
per-TF `MULTI_TF_FEATURE_COUNT_V4` 189 → **182**; mandatory causal
441 → **424** (chart_geometry_smart2 18 → 2, price_action_candle_smart3
32 → 31); `MODEL_NATIVE_SIGNAL_DIM` 608 → **591** (34 base + 424 + 133 ranked
over the same 16 families); `MODEL_NATIVE_CTX_CONT_DIM` unchanged at 158.

**Rule-4 evidence retention.** `ctx_cont.retracement_from_last_impulse` — the
REAL retracement the removed `fib_position_proxy` blended at 0.55 — remains a
`MODEL_NATIVE_CTX_CONT_FIELDS` member and a declared, executed source of
`entry_trend_ema_v1` and `entry_foundation_structure_v1`. The same holds for
every other consumed input; see the removal comments in both owners.

**Measured** (synthetic contract fixture, 240 rows, 2026-08-13): all 15
surviving chart-geometry columns and all 53 surviving candlestick columns are
**bit-identical** to their pre-removal (`f7c3a7a8`) emissions. `[M-synthetic]`
— it proves the edit deleted emissions and unused intermediates only; it is not
a claim about production values.

**Proof coverage that was REDUCED, stated uninvited (rule 25a):**
- The pretrain **channel-position polarity statistic is retired**, because both
  of its subject columns were removed. `entry_pretrain_polarity_signal_v1`
  drops to schema v2 / two required fields, and
  `audit_xau_direction_repair_pretrain_v1` now reports only the pocket-occupancy
  measurement it can still take (rule 2e). Nothing replaces the inversion check.
- `REQUIRED_RAIL_FEATURES` (a `_rail_` substring filter over the mandatory
  geometry tuple) would have become EMPTY, i.e. a silently vacuous gate. It is
  re-pointed at the complete mandatory geometry tuple as
  `REQUIRED_MANDATORY_GEOMETRY_FEATURES` and raises if that tuple is ever empty.
- Four `STRUCTURAL_AUX_LABEL_SIGNAL_REQUIREMENTS` entries (schema v3) and the
  three OR-disjuncts they fed in `_rising_support` / `_falling_resistance` are
  gone. Those two masks are now **strictly narrower**. No threshold was retuned.
- Three consumer composites lost a term with no reweighting (renormalizing
  would invent a magnitude): `momentum.flow_bodyflow_bull/bear_pressure` lost
  the 0.24/0.16 candle-vote terms; `chart.sr_memory_liquidity_*_rejection_*`
  lost the 0.15 geometry term (its DERIVED divisor
  `SR_LEVEL_REJECTION_ALGEBRAIC_MAX` moves 6.09 → 5.94 to match);
  `chart.sr_memory_liquidity_*_break_continuation_*` lost the 0.20
  trendline-break term; `chart.sr_memory_support_minus_resistance_level_balance`
  lost a double count of its own expression.

**NOT examined / unproved.** No real-tape values, liveness, TRAIN ranking or
saturation exist for the post-removal surface — the V30 rebuild chain has not
run. Whether removing pre-fused evidence improves abstention quality is `[U]`
until the pre-registered evaluation ladder says otherwise. The MTF disk-cache
manifests bound to the old `MULTI_TF_FEATURE_NAMES_SHA256_V4` are now
unloadable by design, and every V29J-era artifact is bound to the 608 surface.

---

## 11. V30 PACKAGE 8B — the 75 hand-fused composites (2026-08-13)

Operator decision: **nothing is amputated.** The SMC quality fields keep their
concepts, the session products stay but stop being mandatory, and only what is
provably nothing is removed. Recording what IS and is NOT verified (rule 25c).

### 11.1 Measured-dead math, repaired

**Cross-TF ATR centring** (`entry_volatility_semantics_v1`, new owner
`center_cross_tf_atr_ratio` + `cross_tf_atr_ratio_scaling_expectation`).
`center_atr_ratio = tanh(log2(r))` is correct only for a SAME-TF ratio
(centred at 1). ATR scales as sqrt(bar duration), so a cross-TF ratio is
centred at `sqrt(bars_short/bars_long)`. MEASURED on the complete declared
TRAIN population (V29J, 369,303 rows — complete, so no sampling error):

| pair | mean | p50 | `sqrt(bars_s/bars_l)` | old `_pos(...)` alive |
|---|---|---|---|---|
| m5/m15 | 0.5697 | 0.5471 | 0.5774 | 1.205% |
| m5/h4 | 0.1435 | 0.1256 | 0.1443 | 0.007% |
| m15/d1 | 0.1064 | 0.0936 | 0.1021 | **0 of 369,303** |
| h1/d1 | 0.2182 | 0.2072 | 0.2041 | **0 of 369,303** |

The observed centres ARE the scaling law. The bar counts come from the existing
named per-TF constant `htf_features.MULTI_TF_BARS_IN_M5` (itself derived from
`MULTI_TF_RESAMPLE_RULES`), so **no magnitude was introduced**. The original
function's docstring is narrowed to same-TF ratios so the misuse cannot recur.
Consumers updated: `entry_vol_compression_v1` (which also carried a SECOND
wrong centre, `tanh(ratio - 1.0)`, pinned at -0.406/-0.891/-0.782 — the two
duplicate transforms are collapsed into one centred value per pair) and, swept
in the same wave per rule 25b, `entry_session_regime_interactions_v1`, whose
`atr_ratio_pressure` used `tanh(ratio/2.0)` on raw cross-TF ratios and sat near
+0.148 on every row.

**Volatility unit mismatch.** `rvol_20` is bps*sqrt(20) (TRAIN p50 18.4375);
`_v1_pk_sigma20` is a dimensionless Parkinson sigma (TRAIN p50 4.0968e-4) — a
measured **45,004x** magnitude ratio inside the single 7-term
`high_vol_tail_risk` sum, both through the same z-score scale 2.5.
`tanh(rvol_20/2.5)` was exactly 1.0 on **29.90%** of rows and >= 0.999999 on
**51.51%**. Both legs are now put on ONE declared unit (per-bar bps) by the
semantics owner:

- `rvol_20 / sqrt(20)` — the sqrt(window) is the producer's own declared factor
  (`materialize_build_canonical_features_v1.rvol_window`), not a fitted number;
- `_v1_pk_sigma20 * 1e4` — the repo's bps convention (the `atr_bps` precedent).

After the repair the two estimators agree to **0.6% at p50** (4.1227 vs 4.0968)
and **1.1% at p90** (9.1725 vs 9.0751) — the proof that they are one quantity
and may share one scale. `VOL_PER_BAR_BPS_TANH_SCALE_TRAIN_P90 = 9.172530` is
the p90 of `rvol_20/sqrt(20)` on that complete TRAIN population, the same role
and naming as the existing `SPREAD_RATIO_TANH_SCALE_TRAIN_P90`. Raw log:
`GX1_DATA/logs/v30_package8b_20260813/train_vol_unit_scale_fit.json`.

> Correction to §10: it recorded "`_v1_pk_sigma20` p50 = 1.64e-4". That is the
> p50 of `tanh(_v1_pk_sigma20/2.5)`, not of the field. The field's measured p50
> is **4.0968e-4**; 18.4375 / 4.0968e-4 = 45,004, which is the ratio §10 also
> reports. `[M]`

**`spread_bucket_high_pressure`** was `0.50*1{b>=1} + 0.50*1{b>=2}` over a
five-code declared domain, so codes 2, 3 and 4 all mapped to 1.0 — measured
exactly 1.0 on **60.0011%** of rows. It now uses the graded form its own
sibling `atr_bucket_pressure` uses (one indicator per bucket step) over the
domain owner `MODEL_NATIVE_CTX_CAT_DOMAINS`, so bucket b reads b/4.

**Stale `/5.0` divisor** on `chart.foundation_compression_release_trigger`
removed. Its producer declares `[0, 1]` (`entry_foundation_structure_v1._add`
bounds) and `entry_vol_compression_v1` already states "no rescale is needed".
Three features were capped at 0.2 of their declared range:
`eu_structure_breakout_readiness`, `session_vol_spread_breakout_readiness`,
`session_vol_spread_tail_risk`. **Checked and NOT changed:** the sibling
divisors are correct — `structure_up_minus_down` is declared `[-2, 2]` and
`sweep_reclaim_balance_proxy` `[-5, 5]`.

### 11.2 The session products became rankable, not deleted

`SESSION_REGIME_INTERACTION_FEATURE_NAMES` still emits every field. What
changed is the registry entry in
`entry_model_native_feature_layers_v1.MODEL_NATIVE_SPECIALIST_LAYER_FEATURES`,
which now points at `SESSION_REGIME_INTERACTION_MANDATORY_FEATURE_NAMES` — the
five measured-genuine primitives (`spread_cost_ratio`,
`session_age_progress_norm`, `mtf_regime_class_vote_agreement`,
`mtf_regime_short_long_mismatch`, `h4_d1_regime_sign_agreement`). The other 62
drop into the TRAIN-ranked candidate pool, where the ranker discovers them by
its `*_FEATURE_NAMES` reflection. **VERIFIED by execution:** the pool contains
exactly those 62 `session_regime.*` names.

This is the shape two families already had (chart geometry 2 of 15, candlestick
smart3 31 of 53), so the contract expresses it without deleting the family.
**One wiring gap had to be closed**: both precedents are unconditionally
emitted layers, while session/regime is a gated `smart_builder` whose run/skip
guard in `build_entry_v10_ctx_training_dataset_v3` tested the MANDATORY tuple.
During a ranker pass the requested set contains only candidates — never a
mandatory name — so the layer would have been skipped and its own rankable
fields left uncomputable (`FEATURE_RANKER_EXTENSION_MISSING_CANDIDATES`). The
guard now iterates the new `MODEL_NATIVE_SPECIALIST_LAYER_EMITTED_FEATURES`
registry (full emission per family, with an import-time guard that every
mandatory name is a member of its family's emission).

**`regime_transition_abstain_score` was removed outright** — a hand-written
abstain vote, and abstention is the model's own decision authority (rule 3).
Rule 4 holds: all five of its inputs remain model inputs. The expression
survives only as an internal factor of three composites that are themselves now
rankable, so no hand-written abstain vote is pinned into the surface. **Stated
uninvited (rule 25a):** `spread_atr_boundary_abstain_score` is also an
abstain-named product; it was not in scope, it is now rankable rather than
mandatory, and nobody has judged it.

### 11.3 SMC quality: concepts kept, two honest changes

No SMC field removed; all 24 still emitted. A fidelity header now records, in
the owner, that these are hand-authored scoring rulebooks over approximate
ingredients (no reclaim, no pool identity, an unanchored dealing range) and
names the bounded upgrade path (a real reclaim event; pool identity from the
level registry; a BOS-anchored dealing range).

`_count_proxy` was the fraction of twelve DIFFERENT proximity rows exceeding
0.55 on the SINGLE CURRENT bar — a cross-sectional confluence share with no
time axis — consumed under the names `support_touch_count` /
`resistance_touch_count`. It is replaced by the level registry's genuine
temporal counts `level_{below,above}_touch_count`, added to
`SMC_LIQUIDITY_QUALITY_SOURCE_FIELDS` (they are emitted unprefixed on the
entry-M5 surface by `level_registry_m5_layer`, which the inline extension
materializes before this layer runs). The bounded map is
`n / (1 + n)` — the exact algebraic complement of this file's own `_recency`
owner, so no magnitude was introduced. The orphaned 12-row confluence stack was
then removed: every one of its rows is an independent model input in its own
right and the side-specific maxima of the same rows are still computed as
`support_stack` / `resistance_stack`.

### 11.4 Derived counts

`MODEL_NATIVE_SIGNAL_DIM` **drops by exactly the number demoted**, and that is
correct rather than a defect. The dim is `34 + mandatory + 133`, where 133
(`MODEL_NATIVE_RANKED_REMAINDER_FEATURE_COUNT`) is a pinned literal with its
own tripwire: the ranked remainder is a fixed-size top-N selection from the
pool, not "everything in the pool". Demoting 62 fields enlarges the pool, not
the selection. Holding the dim constant would require raising 133 to 195, which
is inventing a magnitude to preserve a number (rule 2b) and would hand the
demoted fields guaranteed slots again through a different door.

This package's own delta on the mandatory count is **-63** (62 demoted + 1
removed). The absolute values in the working tree at the time of writing also
carry a concurrent agent's in-flight registry growth (+10) and are therefore
not attributable to this package alone; read them from the owner tuples.

### 11.5 NOT examined / unproved

- **No real-tape values exist for any repaired field.** Every "this is now
  alive" statement is proven from source and algebra plus the pre-repair
  measurement; the post-repair distributions are `[U]` until the V30 rebuild
  runs.
- Whether any of this improves abstention quality or bps is `[U]`. The
  2026-08-09 walk-forward refutation stands.
- The ranker has not been RUN against the enlarged pool; the pool membership
  was verified by executing `_candidate_universe`, the ranking was not.
- No claim is made that the five kept primitives are the right five. The
  partition is the operator's explicit list; the audit's supporting statement
  is `[PS]` (structure of the expressions), not a measured ablation.
- The `two_sided_liquidity_pressure` weights (0.50/0.25/0.25) were NOT retuned
  when the quantity they weigh changed from a same-bar confluence share to a
  registry touch count. Re-tuning would have invented magnitudes; the value
  change is real and unmeasured.

---

## 11. V30 PACKAGE 8A — LIVE BUG + EMISSION-ONLY WINS (2026-08-13)

§4a is repaired and §5 is emitted. Recording what IS and is NOT verified
(rule 25c). Evidence class of every claim below: **proven from source** unless
it says otherwise; every execution result is `[M-synthetic]` (rule 2c — it
proves the code runs and the algebra holds, never a production claim).

**§4a LIVE BUG — repaired.** `entry_foundation_structure_v1` read the
`smc_swing_state` enum as a direction bias: state 1 (HH+**LL**) was weighted
0.55 into `hl_state` — evidence AGAINST a higher low — and state 2
(LH+**HL**) 0.45 into `ll_state`. The repair is the literal decomposition of
the enum (HH ∈ {0,1}, HL ∈ {0,2}, LH ∈ {1,3}, LL ∈ {2,3}); every weight keeps
its original clean/mixed role and only the state it multiplies moves, so no
magnitude was invented (rule 2b). `hh_state` and `ll_state` were already the
correct pairs and are byte-identical; `hl_state` and `lh_state` swap states 1
and 2. `[M-synthetic]` with all other sources zeroed the four emissions are
exactly `[0.85,0.45,0,0,0]`, `[0.75,0,0.55,0,0]`, `[0,0.55,0,0.45,0]`,
`[0,0,0.45,0.85,0]` over states 0..4, so
`structure_up_minus_down` reads +1.60 / −0.10 / +0.10 / −1.30 / 0 — the
pre-repair layer emitted **+1.00 for state 1 and −1.00 for state 2**, i.e.
exactly inverted on the two states the 2026-08-09 partition repair made live.
Five emitted foundation fields, `structure_up_minus_down` and every
`chart.structure_swing_*` derivation carried the inversion. Layer version →
`entry_foundation_structure_v5_20260813_swing_state_enum_decomposition_repair`.

**§5 STATE THAT EXISTED IN MEMORY — now emitted.**

- `swing_structure_v1` (+6, on the ctx surface, the M5 mandatory event layer
  and all five per-TF lanes): `consecutive_higher_highs_count` /
  `consecutive_lower_lows_count` (the two MISSING run counters — same
  arithmetic, opposite strict comparison, same cap and log1p normalization as
  the two that existed, so "uptrend structure unbroken for N pivots" is now
  formable); `swing_high_level_intact` / `swing_low_level_intact` (the G1
  `armed_*` loop state = "the last confirmed swing high/low has not been
  closed through"); `bars_since_swing_high_norm` / `_low_norm` (the ONE age
  convention, `htf_features._event_age_norm`, imported — the raw V1 fields are
  untouched). The intact flags are honestly NaN before the first confirmed
  pivot on their side; that prefix is a strict SUBSET of the sequence-delta
  prefix, so the shared HTF warmup trim is unchanged.
- `level_registry_v1` (+4, M5/513 lane): `level_above/below_member_pivot_count`
  (already in state, already the subtrahend of `*_test_count`; the §10
  measurement said member==1 on 98.0/98.3% and ==2 on ~2%, so it is sparse but
  real) and `level_above2/below2_dist_atr` (the SECOND-nearest ACTIVE level per
  side — a four-level shelf and an isolated level emitted identical rows).
  Same absent-slot convention as the nearest slot; no new constant.
- `smc_v1` MTF owner (+3, on all five per-TF lanes):
  `mtf_smc_bos_displacement_atr` (signed `(close − level)/atr` at the firing
  bar, 0 off-event) and `mtf_smc_sweep_up/down_event` (the de-duplicated
  first-bar sweep events — the repeating flags fired once per bar of an
  excursion and are left untouched).
- `smc_v1` M5 owner (+6, **declared but not wired**): the same three
  quantities plus the sided sweep depths and the normalized sweep age, behind
  an explicit `include_v30_additions` call-site contract switch
  (`SMC_V30_ADDITION_NAMES_V1`), default off. See "NOT wired" below.

**POLARITY FLIP — performed in both registries.** "Old support is new
resistance" existed as a one-bar impulse and never as an object: the level
registry kept `status="broken"` forever and excluded the level from the
nearest-ACTIVE scan, and the trendline registry DELETED the line on the very
bar it proved itself. A held retest now flips the level's `side_of_origin` /
the line's `side` and returns it to ACTIVE with identity, anchors, member
pivots, touch history and reaction memory preserved. A FAILED retest is
unchanged. No new constant — the existing retest window and band decide it.
Two consequences stated uninvited:

- The level registry's reaction window now freezes the level's side at its own
  `t0` alongside `atr0`/`center0`, so a flip inside an open window cannot
  rewrite the direction that window was measuring (rule 2g). That is a
  per-level state-schema change → `LEVEL_REGISTRY_STATE_VERSION` = `..._state_3`.
- `side_of_origin` / `TrendlineV1.side` are no longer immutable identity; the
  immutable identity is the level/line id and the anchors.

**POLARITY FLIP COST — measured, stated uninvited.** The trendline flip keeps
held-retest lines alive instead of deleting them, so the ACTIVE-line
population grows. `[M-synthetic]` on the 5000-bar random-walk fixture,
before → after (ms/bar, final ACTIVE lines):

| `seq_len` | before | after |
|---|---|---|
| 16 (per-TF M5 lane) | 0.040 (0) | 0.042 (1) |
| 96 (`MODEL_NATIVE_SEQ_LEN`, the declared `trendline_seq_len`) | 0.109 (41) | 0.169 (65) |
| 512 (no lane uses this) | 0.954 (1110) | **2.546 (2584)** |

The loose 1.5 ms/bar guard is unchanged and still passes at both declared
windows with 8.9x and 36x headroom. At 512 the flip is 2.65x over it. The
benchmark's WINDOW moved to `MODEL_NATIVE_SEQ_LEN` (rule 2g — the measurement
must be taken where the decision is made; no lane runs at 512), and the 512
numbers are recorded in the test docstring so the superlinear scaling is not
hidden. If a lane is ever configured at that window, this cost is real and the
guard must be re-derived from a declared budget, never relaxed.

The LEVEL registry flip costs almost nothing by the same measurement: on a
5000-bar synthetic M5 walk (`q=0.5` fitted tolerance) it moves 0.0415 →
0.0444 ms/bar with the level population unchanged (37 → 36). Its population is
bounded by the merge tolerance and the per-TF expiry cap; the trendline
registry has no merge, so every validated pivot PAIR is a distinct line.

**Derived counts after the package** (owner tuples, never restated literals):
per-TF `MULTI_TF_FEATURE_COUNT_V4` 182 → **191** (+6 swing, +3 mtf_smc);
`MODEL_NATIVE_CTX_CONT_DIM` 158 → **164**; the M5 level block 25 → **29**, the
M5 swing event layer 9 → **15** and the mtf_smc block 11 → **14**. The
mandatory causal count and `MODEL_NATIVE_SIGNAL_DIM` therefore grow by **10**
relative to package 7; their absolute values also move with the concurrent
package 8B in the same worktree, so read them from the owner tuples.

**NOT wired / NOT examined (rule 25a), stated uninvited:**

- The six M5 `smc_*` additions are computed but reach no model input. A raw
  canonical column cannot: the ranker's `_candidate_universe` scans only
  specialist-layer owners and the dataset builder's inline extension exposes
  only the frozen 34 base fields plus specialist-layer outputs. Enabling the
  historical fixed-width source-cascade artifacts. Those obsolete artifacts
  are now retired; the current-pair canonical owner binds its dynamic ordered
  column hash instead. The same three quantities DO reach the model on every
  timeframe through the MTF owner.
- No rare-event floor was registered for any new event. Floors must come from
  a measured TRAIN build (the 2026-08-12 precedent); a new event that lands
  below the 1% liveness floor on the next real build is a correct fail-loud.
- No real-tape values, liveness, TRAIN ranking, saturation or abstention
  effect exist for any field in this package. Whether the polarity flip or the
  repaired enum improves direction quality is `[U]`.

### 11.6 Test state at hand-off

Full capped suite (`scripts/gx1_capped_run.sh --class audit --mem 4G --swap 512M`,
`pytest tests/`) run after the repair: **2 failures, both outside this package
and proven so by authorship** —
`test_evidence_retention_v1.py::test_resume_refuses_a_target_whose_source_is_still_present`
(the concurrent retention-`resume` work: +358 lines in
`gx1/scripts/cleanup_gx1_evidence_v1.py` and +336 in its test, none of them from
this package) and
`test_guard_hooks_versioned.py::test_guard_artifact_matches_versioned_reference[settings.reference.json]`
(drift between the live `~/.claude/settings.json` and the tracked reference; both
files are untouched here and `.claude/` is unmodified in the tree).

The eight failures this package DID cause were all repaired and re-verified:
two vol-compression fixtures and one bounded-orchestration fixture that
fabricated NEGATIVE values for `rvol_20` / `_v1_pk_sigma20` — encoding the exact
misunderstanding this package removed, since both quantities are non-negative by
construction — plus the `session_regime_encoder` specialist count, which is now
derived from the owner tuple instead of restated. `tests/` fixtures were moved
onto the measured TRAIN quantile scale with their original rank order preserved.
Raw log: `GX1_DATA/logs/v30_package8b_20260813/full_suite.log`.
