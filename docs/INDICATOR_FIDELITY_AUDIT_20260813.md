# Indicator fidelity audit — 2026-08-13

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
