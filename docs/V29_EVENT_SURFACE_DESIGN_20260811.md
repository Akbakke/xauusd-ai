# V29 EVENT SURFACE — unified design (2026-08-11)

Status: **PROPOSAL ONLY.** This document carries no authority and authorizes no
heavy job. It is the synthesis of the eight family event-gap reports of
2026-08-11 (`/home/andre2/GX1_DATA/logs/event_gap_review_20260811/{structure_swing,
smc_liquidity,trend_ema,vol_compression,momentum_flow,session_regime,
chart_geometry,price_action_candle}.md`) into one buildable V29 contract
proposal. Every underlying grade is proven from source by those reports; every
count in this document is derived arithmetic over their specs; nothing here is
measured on data. Execution is sequenced BEHIND the pending immutable recipe
decision for the normalized substrate (V8-smoke FLAT collapse on the repaired
V27/V28 substrate is the active-track blocker); adopting this design changes
nothing until the recipe owner adopts it and the evidence ladder runs.

Shared finding of all eight reports, stated once: the current surface is a
dense continuous conditioning system with almost no discrete events, no level
identity, no touch/reaction memory, and no retest semantics. The operator's
confluence-of-events reading (level break + retest hold + EMA cross + same
event on M15 and H4) is unrepresentable today. V29 adds the missing EVENT
PRIMITIVES per family and per timeframe — and deliberately nothing above them:
per rule 4 and the `mtf_confluence` removal precedent (2026-08-05), the fusion
transformer learns every binding.

---

## 1. DECISION — one unified level registry (three kinds, one object model)

Three reports independently proposed overlapping level stores:
smc_liquidity's `LEVEL_REGISTRY_V1` (clustered pivot levels + round-number
grid), structure_swing's G5 `swing_level_registry` (nearest-active-level
slots), and session_regime's G1 session-anchored registry (PDH/PDL/PDC,
Asia range, session opens, week open, prior-week close).

**Decided: ONE new bounded authority, `gx1/features/level_registry_v1.py`,**
carrying a level KIND enum:

```
LevelKind = {pivot_cluster, session_anchored, round_number}
```

- **One object model** (extends the smc report §2 object with `kind` and
  `anchor_name`): `{level_id, kind, anchor_name (session_anchored only),
  side_of_origin, center_price, zone half-width TOL·ATR, member_pivot_count,
  touch_count, birth_bar, last_touch_bar, bars_since_touch, reaction_sum_atr,
  reaction_max_atr, reaction_last_atr, completed_reaction_count,
  status ∈ {active, broken}, break_bar, break_side,
  retest_state ∈ {none, pending, held, failed}}`.
- **One lifecycle** for all kinds, fixed intra-bar order (smc report §3):
  (1) admission/merge, (2) break check (first close beyond the zone,
  edge-triggered `cond & ~prev_cond`), (3) retest check (first zone re-entry
  within RETEST_WINDOW; outcome hold/fail at that bar's close), (4) touch
  check (first-entry-per-excursion, deduplicated), (5) reaction-window
  completion (signed, confirmation-lagged at t0+W — sign-blindness lesson of
  2026-08-09 not repeated).
- **Kind-specific admission only**: pivot_cluster admits/merges lookback-3
  confirmed pivots from `smc_v1` (`_detect_swing_pivots` +
  `_track_recent_swings`, `SWING_LOOKBACK = 3` — the one pivot truth; no
  second detector); session_anchored admits one immutable price per anchor on
  the SESSION_BOUNDARIES trading-day clock (ASIA open 22:00 UTC, named
  constant in `session_detector.py`) and expires it at its next anchor;
  round_number is stateless grid distance (no lifecycle).
- This decision **overrides** the session report's owner choice
  (`augment_forward_outcome_v2.py`): day-keyed session levels are the same
  bounded authority as pivot-cluster levels (persistent price-level identity
  with touch/break/retest lifecycle), so they live in the one registry, wired
  through the existing ctx emission path. `_build_daily_pivots` and every
  existing `dist_to_*` field stay byte-identical (persistent model fields).
- structure_swing's G5 registry is **superseded entirely** by this registry
  (its 12 proposed slot fields and its per-TF variant are duplicates of the
  slot template below). Its G1–G4 single-level swing events remain in
  `swing_structure_v1.py` — see §3; boundary in §1.4.

### 1.1 One emission template

One per-level template, reused across kinds in two addressing modes:

- **Slot-addressed** (pivot_cluster): nearest ACTIVE level above and below
  current close — 7 state fields per side — plus registry-wide event fields.
- **Name-addressed** (session_anchored): the same vocabulary per named anchor,
  7 fields each.

Reconciliation of the three reports' field lists (dedup decisions):

| Source proposal | Disposition |
|---|---|
| smc M5 list (22) | ADOPTED verbatim as the pivot_cluster M5 lane |
| smc per-TF list (11) | ADOPTED verbatim as the per-TF lane |
| structure_swing G5 slots (6×2 M5 + per-TF) | DROPPED as duplicates; its only non-overlap field `level_*_broken` is covered by the registry's break/`bars_since_break` events (broken levels stop being "nearest ACTIVE") |
| session G1 template (8/level: dist, break_up, break_dn, bars_since_break, touch_count, retest_flag, bars_since_retest, break_held) | NORMALIZED to 7/level: `retest_flag`+`bars_since_retest`+`break_held` are replaced by the registry's signed retest vocabulary (`retest_hold_signed`, `retest_fail_signed`) + `bars_since_break`. `break_held` is recoverable (break event + no opposite-side break + no fail event; session anchors expire at their next anchor, bounding staleness). One retest vocabulary across all kinds. |

### 1.2 Deduplicated M5/513 list — final count **78** (22 Phase A + 56 Phase B)

Pivot_cluster slots, prefix `level_` (14):
`level_above_dist_atr`, `level_above_touch_count`, `level_above_age_bars`,
`level_above_bars_since_touch`, `level_above_mean_reaction_atr`,
`level_above_max_reaction_atr`, `level_above_last_reaction_atr` + the 7
`level_below_*` mirrors.

Pivot_cluster registry events (6):
`level_break_up_event`, `level_break_down_event`, `level_broken_touch_count`
(event-gated 0), `level_bars_since_break`, `level_retest_hold_signed`,
`level_retest_fail_signed` (sign = break direction).

Round_number (2): `level_round_50_dist_atr`, `level_round_100_dist_atr`.

Session_anchored (Phase B), prefix `sess_level_`, 8 anchors × 7 = 56:
anchors `pdh, pdl, pdc, asia_hi, asia_lo, sess_open, week_open, pwc`; per
anchor `dist_atr, break_up_event, break_dn_event, bars_since_break,
touch_count, retest_hold_signed, retest_fail_signed`. `pwc` doubles as the
weekend-gap tracker with zero extra machinery (session report G4): its
`dist_atr` at the reopen bar IS the gap magnitude, its break IS the fill.

### 1.3 Per-TF list — final count **11** per TF, prefix `mtf_level_`

`mtf_level_above_dist_atr`, `mtf_level_below_dist_atr`,
`mtf_level_above_touch_count`, `mtf_level_below_touch_count`,
`mtf_level_above_mean_reaction_atr`, `mtf_level_below_mean_reaction_atr`,
`mtf_level_break_up_event`, `mtf_level_break_down_event`,
`mtf_level_bars_since_break`, `mtf_level_retest_hold_signed`,
`mtf_level_retest_fail_signed` — pivot_cluster kind only, run independently
per TF clock inside the existing V4 lane next to
`compute_smc_mtf_primitives_v1`. Session anchors are single prices on the M5
clock (identical on every TF) and round numbers are stateless — neither
replicates per TF. This makes "the same break on M15 AND H4" two input events
whose conjunction the fusion learns.

### 1.4 Constants, routing, boundaries

- Constant origins (all named in one sentence, rule 2a): pivot source
  `SWING_LOOKBACK=3` (named, smc_v1); `TOL_LEVEL_ATR[tf]` TRAIN-fitted
  quantile of pivot-to-nearest-earlier-pivot distances **measured inside the
  runtime merge's own `AGE_CAP[tf]` window** (rule 2g; the unpruned
  whole-history version tracked TRAIN length and was repaired 2026-08-13, V30
  package 6 — see `docs/INDICATOR_FIDELITY_AUDIT_20260813.md` §0a), quantile
  `q` an
  immutable-recipe key, fitted values frozen bundle state (rule 18), sample
  size + sampling bound stated per TF (rule 2f); expiry `AGE_CAP[tf]` = the
  liquidity-zone lookbacks 240/192/168/168/60 (named constants in
  `augment_forward_outcome_v2._liquidity_zones`); reaction window `W=12`
  (named tau-12 family convention, flagged for TRAIN-fitted upgrade at the
  recipe decision); `RETEST_WINDOW=24` (named tau-24 convention, recipe key);
  round grid 50/100 USD (operator-declared XAUUSD convention, named constants
  in the registry contract); distance saturation 20 and count/age caps 999
  (named `exp(-min(x,20))` and `smc_bars_since_sweep=999` sentinel
  conventions); session clock `SESSION_BOUNDARIES` (named constants).
- Routing (rule 19, one specialist owner per field): `level_*` and
  `mtf_level_*` → `smc_liquidity_encoder` (the existing `"level"` token in
  `classify_entry_specialist_feature` already routes these names — verified
  by the smc report); `sess_level_*` → `session_regime_encoder` via the
  existing ctx overrides map.
- `entry_support_resistance_memory_v1` is **upgraded, not bypassed**
  (follow-up wave): registry raw fields enter its source contract so its
  pressures compound on true touch counts; its EWM fields remain (different,
  still-valid smoothed-proximity quantity).
- Serve-state continuity: exact serializable registry state; declared
  cold-start warmup = `AGE_CAP[tf]` bars per TF (honest contaminated prefix);
  session kind declares first-complete-day/week warmup (fail-closed, rule 2e:
  no partial ranges emitted as complete ones).
- Warmup: NaN prefix until the first admitted level, single chronological
  prefix trimmed by the shared HTF matrix owner; no mid-series NaN
  (side-absence saturates at the distance cap).

## 2. DECISION — trendline registry stays separate

**Confirmed: `gx1/features/trendline_registry_v1.py` is a second, separate
new bounded authority** (chart_geometry report Part B), NOT merged into the
level registry and NOT hosted in `entry_chart_geometry_v1` (stateless
no-price-access contract) or `smc_v1` (stateless per-bar primitive contract).
Boundary contract:

- The level registry owns HORIZONTAL levels (discrete prices/zones with touch
  identity). The trendline registry owns SLOPED lines and channels
  (two-point-anchored, ≥3-touch-validated, immutable line identity, state
  machine CANDIDATE → ACTIVE → BROKEN → retired-after-retest).
- **One pivot truth:** the level registry exports its clustered-pivot
  membership (level_id → member pivot indices/prices) as an internal API; the
  line fitter consumes that stream (and the raw confirmed-pivot stream) and
  never re-detects pivots (rules 13/19/21).
- No shared emitted field between the two registries (rule 19). The existing
  10 `mtf_geometry_*` fields stay (horizontal last-pivot evidence is distinct
  from sloped-line evidence, rule 4).
- Emission: the 30-field per-TF block exactly as specified (B.5): 2 slots
  (nearest ACTIVE projection above/below) × 7 attributes, 10 event/impulse
  fields (touch, break with broken-line touch_count/age, retest hold/fail per
  direction), 6 channel/triangle fields with apex proximity. Constants: band
  per TF TRAIN-fitted (median 3rd-touch |deviation|/ATR over the complete
  TRAIN candidate population, N stated; break margin = the same band — one
  constant; since that population is the *null* it then judges, the fit also
  publishes the measured `implied_validation_rate` it implies, ~0.5 by
  construction — V30 package 6, audit §0b); retest window
  `TRENDLINE_RETEST_WINDOW_BARS_V1 = 2·SWING_LOOKBACK+1` bars (named constant
  in the registry owner, derived from a named constant — read it by executing
  the owner, never from this line); candidate window AND the ACTIVE staleness
  bound = `per_tf_seq_lens` (explicit recipe input). Aux-target replacement:
  §5.
  - 2026-08-15: `a94f5c6e` replaced BOTH of those bounds with one TRAIN-fitted
    `identity_expiry_bars` (`ceil(RMST)`, 2 on M5/M15/H1/H4 and the entry-M5
    lane, 1 on D1). Since the promotion path stamps
    `last_touch_bar = t - SWING_LOOKBACK`, every fitted value was `<= 3` and
    deleted each line on its own promotion bar: 28 of the emitted fields went
    constant 0.0, `geomline_bars_since_break` went all-NaN, and the M5 lane
    died on `HTF_V4_CACHE_WARMUP_INVALID`. Both bounds are restored to the
    state this paragraph always described, and the fitted lifetime is no longer
    consumed anywhere. Measured after the repair on the complete declared tape
    (`XAU_M5_NATIVE_2019_20260804_V4`, 537,861 M5 rows, M5 lane): 0 of 31
    trendline columns constant, 537,657 complete rows, no all-NaN column.
  - The emitted block is 31 fields, not the 30 written above; the authoritative
    tuple is `TRENDLINE_REGISTRY_FEATURE_NAMES_V1` (rule 13 — this document may
    not restate it). That drift predates the 2026-08-15 repair.
- Entry-M5 visibility: the same 30-field emission also runs on the entry M5
  clock into the 513 lane (`chart.geomline_*`) — see §4.1 block E.

## 3. Per-TF event primitives per family — consolidated table

All events are binary/signed closed-bar edge triggers (`cond & ~prev_cond`,
the repaired-BOS idiom); ages use `log1p(x)/log1p(500)` or the 999-sentinel —
in each owner ONE of the two named conventions is chosen and cited. "Ext" =
extension of the existing owner (rule 21); the only NEW files in V29 are the
two registries (§1, §2).

| Family | Primitives (fields) | Owner file | New/Ext | Constant origins | Per-TF | M5/513 lane | Phase |
|---|---|---|---|---|---|---|---|
| smc_liquidity | level registry pivot_cluster + round grid (M5 22, per-TF 11) | `level_registry_v1.py` | **NEW** (bounded authority) | §1.4 | 5 TFs | yes (22) | A |
| chart_geometry | trendline/channel registry (30/TF) | `trendline_registry_v1.py` | **NEW** (bounded authority) | §2 | 5 TFs | yes (30, block E) | A |
| structure_swing | G1 swing break events + displacement (3); G2 break ages (2); G4 pivot-sequence deltas + higher-low/lower-high run counts (4) | `swing_structure_v1.py` | Ext | `SWING_LOOKBACK_V1=2`, ATR14, `FOUNDATION_EVENT_AGE_CAP=96`, log1p convention — zero new numbers | 5 TFs (`MULTI_TF_V4_SWING_FEATURES` 5→17) | yes (9) | A |
| structure_swing | G3 last-swing retest touch/fail events + held state (3) | `swing_structure_v1.py` | Ext | ε, K = recipe inputs or TRAIN-fitted (origin decided at recipe time, none invented) | 5 TFs | yes (3) | B |
| trend_ema | GAP-1 per-TF 50/200 spread/state/cross (4); GAP-2 cross age (1); GAP-3 price×EMA50/200 cross events + above-ages (6) | `htf_features.py` + local layer (`entry_model_native_feature_layers_v1.py`) | Ext | bit-for-bit the local layer's formulas; clip ±30, 500-cap log1p — all named, zero new numbers | 5 TFs (trend slice 10→21) | yes (7: GAP-2 local + GAP-3 local) | A |
| trend_ema | GAP-4 EMA50 retest touch/hold/fail + retests-held-in-leg count (4) | same owners | Ext | tau TRAIN-fitted quantile (rule 2f: N + bound stated); confirmation window K = recipe input | 5 TFs | yes (4) | B |
| momentum_flow | G1 RSI divergence event/strength/age (3); G2 RSI 30/70/50 threshold crosses, mom20 sign flip + ages (6) | `htf_features.py` (`compute_per_bar_features_v4`) | Ext | Wilder 1978 published 30/70 + midline 50 (fixed points of the existing affine map); /50 strength = existing constant; zero-line natural | 5 TFs | yes (9, block E) | A |
| momentum_flow | G3 raw-RSI ctx scalars `m5_rsi14`, `h1_rsi14_raw`, `h4_rsi14_raw` (3) — z-fields kept | ctx path via `entry_model_native_signal_v1.py` | Ext | one `_rsi` producer, one unit | ctx (M5 clock) | yes (3) | A |
| momentum_flow | G4 `bos_on_volume_signed`, `sweep_on_volume_signed` (2) | base assembly / `entry_smc_liquidity_quality_v1.py` (smaller wins) | Ext | product of two existing owned fields, NO threshold | M5 first (per-TF needs per-TF vol-z, deferred) | yes (2) | B |
| session_regime | G2 per-TF regime flip flags + bars-since for m5/m15/h1/h4 (8) — D1 exists | `regime_v4_features.py` | Ext | identical F8/F9 algorithm, class-id keyed, own-TF clock via `tf_bars` — zero new numbers | ctx per-TF flags | yes (8) | A |
| session_regime | G1 session-anchored levels (56) → §1 registry, session_anchored kind | `level_registry_v1.py` | (in NEW file) | §1.4; θ/touch-norm TRAIN-fitted, N stated | M5 ctx only | yes (56) | B |
| session_regime | G3 VWAP cross event, bars-since, respect count (3) | `htf_features.py` (existing VWAP owner) | Ext | pure algebra; θ TRAIN-fitted; existing anchor kept (midnight-UTC field untouched; session-clock VWAP = one more registry anchor if the operator wants it) | 5 TFs | no | B |
| vol_compression | G1 squeeze_active state + squeeze_release event (2); G2 bars_in_squeeze + duration-at-release latch (2) | `htf_features.py` per-TF lanes; direct raw/event routing (the former hand-fused `entry_vol_compression_v1.py` consumer was retired in v10) | Ext | percentile windows per `D1_atr_percentile_252` convention; `p_low`/`p_release` recipe inputs or TRAIN-fitted; 500-cap log1p | 5 TFs | yes (4, block E) | B |
| vol_compression | G3 consolidation-box object (10: width, age, dist top/bottom, touch counts, break up/dn, displacement, edge retest) — **conditional on G0 measurement** (else interaction terms in the specialist layer instead) | same owners | Ext | τ, k recipe/TRAIN-fitted; box born at squeeze-on, causal running extrema | 5 TFs | yes (10, block E) | B |
| vol_compression | G5 `H4_range_compression_ratio` (1) | `htf_features.py` scalar lane | Ext | exact H1/M15 sibling convention incl. `H4_ATR100_MIN_BARS` warmup gate | ctx | yes (1) | B |
| price_action_candle | P1 `hammer_event_quality`, `shooting_star_event_quality` (Stage A, zero new numbers), `doji_event_flag` (Stage B, TRAIN-percentile) (3); P3 gap-gate repair (no new fields) | `entry_candlestick_patterns_v1.py` | Ext | Stage A: existing fields × existing composition convention; Stage B: percentile = recipe input, threshold = TRAIN-fitted (rule 18/2f) | 5 TFs (layer 60→63, candle slice 64→67) | yes (3, mandatory — hammer/doji currently have no mandatory M5 representative) | B |

Explicitly NOT built in V29 (measure-first, rule 22): candle P5
bars-since-pattern (attention already sees in-window history; build only on a
measured failure), momentum G5 divergence-count/RSI-trendline, vol G4
failed-break counts, G6 bandwalk, G7 NR-N (pending the G0/nonzero-rate
audits), per-TF volume-z (prerequisite for per-TF G4), candle-side sweep bar
(smc owns sweep truth — confirmed non-gap).

Retest-authority boundary (dedup decision): the level registry is the ONE
retest authority for identified horizontal levels; trendline registry owns
retest of sloped lines; structure_swing G3 is scoped strictly to the
last-lookback-2-swing lifecycle (a level definition the registry does not
carry); EMA GAP-4 owns dynamic-EMA retest. Four disjoint objects, no shared
field (rule 19).

## 4. DECISION — no precomputed confluence, and the removal candidates

**No cross-TF confluence votes and no pattern×level conjunction features are
built.** Rule 4 and the 2026-08-05 `mtf_confluence` removal are the binding
precedent: primitives ship per TF; the 26/96-fusion learns every binding
(M15∧H4 break, hammer-at-level, doji-at-squeeze, break-on-volume beyond the
two G4 magnitude products). Consequences applied consistently:

- session G2's proposed `regime_flip_alignment_v3` and
  `regime_flip_direction_agreement_v3` are **excluded** (cross-TF aggregate
  counts — the per-TF flags + bars-since make the coincidence learnable).
- vol G5's `n_tfs_in_squeeze` is **excluded by default** for the same reason
  — flagged as an operator-decision open question (§8) because it is a
  non-directional multiplicity count, arguably not the removed vote class.
- Candle P2 interface contract adopted: patterns stay pure sparse events;
  the registries publish distance/age/touch/break/retest on the same
  closed-bar clock; fusion binds them. No `hammer_at_support` ever.

**Removal candidates (separate decision, NOT part of the V29 dimension
math; rule-4-compliant retirement = every consumed input remains a model
input):**

1. The 6 candle aggregate votes: `candle.pattern_bull_reversal_pressure`,
   `bear_reversal_pressure`, `bull_continuation_pressure`,
   `bear_continuation_pressure`, `indecision_breakout_setup`,
   `tail_rejection_risk` (intra-family hand-weighted votes; candle report
   P4). Effect if adopted: layer 63→57, per-TF slice −6, M5 −6.
2. The `entry_chart_geometry_v1` composite pressures (58-field layer of
   hand-weighted same-bar products, including the falsely-named
   `trendline_break_*`/`*_retest_*`) — candidates for the mtf_confluence
   treatment once real line features are live (chart report B.7). Their 55
   source scalars all remain model inputs.

## 5. Contract impact

### 5.1 Dimensions

Active composition (rule 4): 34 base + 346 mandatory causal + 133
TRAIN-ranked = **513**; per-TF V4 surface = **111**. All new fields register
as mandatory causal-layer outputs; the 133 count is held constant and
re-ranked TRAIN-only on the V29 substrate.

Per-family additions (from §1–§3; per-TF numbers are per timeframe):

| Family | M5 lane A | M5 lane B | per-TF A | per-TF B |
|---|---|---|---|---|
| smc_liquidity (level registry) | 22 | — | 11 | — |
| chart_geometry (trendline registry) | 30 | — | 30 | — |
| structure_swing | 9 | 3 | 9 | 3 |
| trend_ema | 7 | 4 | 11 | 4 |
| momentum_flow | 12 | 2 | 9 | — |
| session_regime | 8 | 56 | — | 3 |
| vol_compression | — | 15 | — | 14 |
| price_action_candle | — | 3 | — | 3 |
| **Totals** | **88** | **83** | **70** | **27** |

**513 → 601 (Phase A) → 684 (Phase B)** — mandatory causal 346 → 434 → 517.
**111 → 181 (Phase A) → 208 (Phase B)** per TF.

> **STAGE-2 CORRECTION (2026-08-11, wiring wave — derived reality):**
> the counts above were pre-implementation arithmetic; the stage-1 owners and
> the stage-2 wiring bind the following exact derived counts, and the code
> tuples — not this table — are the authority (rule 13):
>
> 1. **momentum events are 10, not 9** (accepted): the built G1/G2 block is
>    4 RSI threshold crosses + `rsi_extreme_age_norm` + 2 mom20 sign flips +
>    2 divergence events + `divergence_age_norm`
>    (`MULTI_TF_V4_MOMENTUM_EVENT_FEATURES`).
> 2. **Wired by stage 2** (operator decision: block E KEPT):
>    per-TF surface `MULTI_TF_FEATURE_COUNT_V4` **111 → 173**
>    (= 111 + 11 trend events + 10 momentum events + 11 `mtf_level_*` +
>    30 `geomline_*`/`geomchan_*`); 513 lane
>    `MODEL_NATIVE_SIGNAL_DIM` **513 → 592** with mandatory causal
>    **346 → 425** over **11 → 16** families (= +22 `level_*` + 30
>    `chart.geomline_*`/`chart.geomchan_*` + 9 swing events + 10 momentum
>    events + 8 regime flips); ranked remainder held at 133.
> 3. **Declared by Phase A, wired by the V30 wave** (2026-08-13, package 2 —
>    "Phase-A completion"; the engines already existed and only the call
>    sites/contracts moved, at one rebuild boundary):
>    - structure_swing per-TF additions — DONE, +9/TF (the nine
>      `SWING_V29_ADDITION_NAMES_V1`, not the sketched 12:
>      `MULTI_TF_V4_SWING_FEATURES` is 5→14, and the per-TF lane calls the
>      producer with `include_v29_additions=True`).  The same nine names were
>      adopted into `MODEL_NATIVE_CTX_CONT_SWING_FIELDS`, performing the
>      producer's own "the stage-2 V29 wiring adopts these names into the
>      ctx/111-surface contracts together with the V29 rebuild" declaration
>      (+9 ctx).
>    - momentum G3 raw-RSI ctx scalars — DONE, +3 ctx
>      (`m5_rsi14_canon_v2`, `h1_rsi14_canon_v2`, `h4_rsi14_canon_v2`; spelled
>      as the existing `<tf>_rsi14_canon_v2` M15/D1 siblings they mirror, not
>      the sketched `*_raw`, and emitted by the same one `_rsi` producer on
>      each TF's own closed bars).
>    - trend_ema GAP-2/3 M5-local fields — DONE, all 7 (package 2 landed the
>      three durations `chart.local_ema50_200_cross_age_norm` /
>      `chart.local_price_above_ema{50,200}_age_norm`; package 3, 2026-08-13,
>      landed the four cross EVENTS
>      `chart.local_price_x_ema{50,200}_cross_{up,down}`).  All seven are wired
>      through the same `_trend_age_bars` / `_event_age_norm` /
>      `_cross_up_event` / `_cross_down_event` owners as the per-TF lane —
>      imported, never duplicated.  The layer's causal warmup floor is
>      unchanged at 201 rows (re-verified on the declared canonical_v3 M5
>      surface: source index 199 and 200 fail the layer's own finiteness gate,
>      201 passes, on the full 19-column layer).
>    Derived reality after the V30 wave (packages 1-3):
>    `MODEL_NATIVE_SIGNAL_DIM` **608** (34 base + 441 mandatory causal + 133
>    ranked), ctx **155**/5, per-TF `MULTI_TF_FEATURE_COUNT_V4` **189**.  The
>    code tuples remain the authority; these figures are recorded, not
>    restated as a target.
> 4. The two aux tautologies are replaced by the forward-realized
>    `y_line_support_touch_held` / `y_line_resistance_touch_held` plus their
>    touch-event masks (§5.2.8; head stays 6-dim, masked loss on the two
>    line dims).

Entry-M5 visibility block (**block E**, inside the M5 numbers above): the
Entry route consumes M15/H1/H4/D1 windows of the 111 surface only
(`m5_route_consumption.entry = False`); Exit consumes all five TFs. Every
family's M5-clock event must therefore exist in the 513 causal lane for Entry
to see it — the architecture's existing convention (M5 candles, M5 BOS
events already live there). Block E = geomline M5 (30, A) + RSI/mom events
M5 (9, A) + squeeze M5 (4, B) + box M5 (10, B) = 39 A + 14 B. Striking block
E (operator decision, §8) gives 513 → 562 → 631 and makes M15 the finest
Entry event resolution for lines/oscillator-events/squeeze. Recommendation:
keep block E (the operator enters on M5).

Conditional: −10/TF and −10 M5 if the vol G0 measurement replaces the box
object with interaction terms; −2 candle M5/per-TF Stage-B hammer/star flags
are added only if Stage A measures dense (not counted above).

### 5.2 Required downstream updates (all fail-closed, one rebuild boundary per phase)

1. `gx1/contracts/entry_model_native_signal_v1.py`: `MODEL_NATIVE_SIGNAL_DIM`
   513 → X, ctx_cont 142 → +, new `MODEL_NATIVE_SMART_FAMILY_CONTRACT`
   entries for the two registries.
2. `gx1/features/htf_features.py`: `MULTI_TF_FEATURE_COUNT_V4` 111 → Y, new
   `MULTI_TF_FEATURE_NAMES_SHA256_V4`, group tuples, MTF cache rebuild.
3. `entry_specialist_feature_groups_v1.py`: routing tokens +
   `expected_specialist_counts` (candle 60→63; per-TF `chart_geometry_encoder`
   10→40, `smc_liquidity_encoder` 11→22, `trend_ema_encoder` 10→21+4,
   `momentum_flow_encoder` +9, `structure_swing_encoder` 5→17, etc.);
   `MULTI_TF_SPECIALIST_FEATURE_GROUPS_V4`.
4. `entry_exit_production_architecture_v1.py`: encoder input dims re-derived;
   parameter/step-RSS budget re-measured under the 20G cap before any train.
5. Normalization refit — once, complete physical TRAIN population, before
   sampling (rule 18); all TRAIN-fitted constants (`TOL_LEVEL_ATR`, band_tf,
   tau, θ, candle percentile thresholds) frozen as bundle state with sample
   sizes and bounds recorded (rule 2f).
6. **Full V29 dataset rebuild** with one new immutable build `--run-id`
   (both native lanes, M15 cache, ordered surfaces, preflight, liveness);
   training runs bind `dataset_run_id` (rules 8/14/15). The Exit local M1
   surface changes too (the shared local layer grew) — its contract is
   re-derived in the same wave.
7. TRAIN-only re-ranking of the 133 over the V29 substrate.
8. **Aux rail-target replacement** in
   `build_entry_v10_ctx_training_dataset_v3.py`: delete the two tautologies
   `y_rising_channel_support_touch` / `y_falling_channel_resistance_touch`;
   add the forward-realized line-touch labels `y_line_support_touch_held` /
   `y_line_resistance_touch_held` (defined only on registry touch-event bars,
   label = that same line not BROKEN within the existing named forward
   horizon; event-masked loss per the `y_side_mask` pattern). Head count
   unchanged.
9. Event-liveness gates: every new event field enters the auto fail-closed
   liveness battery; a never-firing event on the declared TRAIN population is
   a red gate, not a warning.
10. Tests: `test_htf_v4_per_bar_contract` expected counts, specialist-count
    tests, alias tests, registry unit tests (causality: no emission may use
    an unconfirmed pivot or an incomplete session block).

## 6. Build sequencing (rule 22) and measurement gates

**Phase A — the operator's core scenario end-to-end** (higher lows into a
level; the level breaks; the retest holds; EMA regime agrees; the same break
visible on M15 and H4): `level_registry_v1.py` (pivot_cluster +
round_number; session_anchored kind declared in the enum, produced in B),
`trendline_registry_v1.py`, trend_ema GAP-1/2/3, momentum G1/G2/G3, session
G2 regime-flip flags, structure_swing G1/G2/G4, aux-target replacement.
= 513→601, 111→181. One immutable V29 rebuild → normalization + fits →
TRAIN re-rank → capped smoke → candidate, on the existing evidence ladder.

**Phase B — after Phase A admission and the measurements below**:
session_anchored registry kind + VWAP events, squeeze state/release/duration
+ H4 ratio + box-or-interactions (per G0), candle rarity gates + gap-gate
repair, trend GAP-4, structure G3, momentum G4, and the §4 vote-removal
decision. = 601→684, 181→208. Own immutable rebuild `--run-id`.

**Measured BEFORE adoption (pre-registered, capped 4G audits):**

1. Registry + trendline compute cost on the real declared tape during the
   Phase-A build (chart report's minutes-estimate is unproven; a blowout is a
   red gate).
2. Nonzero-rate audit of all 60 candle fields per TF on declared TRAIN bytes
   (precondition for P1; 20 quoted rates are structural, not measured) +
   piercing/dark-cloud ±1-tick flip-sensitivity (P3).
3. Vol G0 TRAIN-only measurement (squeeze × SMC-envelope interaction) —
   decides box object vs interaction terms.
4. trend GAP-6 tanh-input quantile measurement (saturation prevalence) — with
   the unit-comment repair, §7.2.
5. All TRAIN fits with per-TF sample sizes and sampling bounds (rule 2f).
6. Event base-rate audit for every new event on V29 TRAIN (liveness, §5.2.9).

**Pre-registered evaluation plan — V29 vs V28:** identical walk-forward
protocol to the 2026-08-09 snapshot-edge refutation (same folds, same
TRAIN/OOS boundaries); the coin-flip null (−13.16 bps TRAIN), oracle
(+17.76) and available-skill (+30.91) references re-measured on the V29
substrate and mandatory in every claim; multi-seed (≥2 seeds — the paused
logit-adjust seed protocol, §7.1); abstention quality is the success
criterion; admission only through the existing gate battery; no gate
threshold moves without its rule-2f bound. A single-fold win is not
adoption evidence.

## 7. Open items folded in

1. **Paused logit-adjust 2-seed validation.** `ENTRY_DIRECTION_LOGIT_ADJUST_TAU=1.0`
   was adopted 2026-08-11 (recipe owner + trainer; see
   `docs/RECIPE_DECISION_DRAFT_20260808.md` adoption note) on seed-flip
   collapse evidence; the confirming 2-seed training validation is paused.
   It resumes on the first trained candidate — V29 Phase A, or earlier on
   V28 if training resumes there first. V29 does not supersede it.
2. **Historical trend_ema unit-comment mismatch (GAP-6).** The comments were
   corrected, and the hand-fused `entry_trend_ema_v1.py` layer was retired in
   v11. Raw local and independently clocked MTF EMA evidence remains routed to
   the learned trend specialist; any future scale must still be TRAIN-fitted,
   never chosen by eye.
3. **Recipe-draft deferred items** (`docs/RECIPE_DECISION_DRAFT_20260808.md`
   item 4): "V28 divergence/M5-RSI feature owners" — **ABSORBED** by V29
   momentum G1/G3 (this document is their design). "y_side FLAT-parking
   rewrite" (rebuild-coupled) — **ABSORBED** into the V29 Phase-A rebuild
   wave. "Anchoring-weight sweep" — **NOT absorbed**; stands as its own
   deferred decision (no origin for new magnitudes without a sweep on real
   data).
4. **Aux-target tautologies** — absorbed (§5.2.8).
5. **sr_memory upgrade** (registry fields into its source contract) —
   follow-up wave after Phase A, build-on never remove (§1.4).

## 8. Open questions requiring the operator's decision

1. **Block E** (Entry-M5 visibility: +39 A / +14 B M5 fields): keep
   (recommended) or strike (Entry's finest event resolution becomes M15)?
2. **`n_tfs_in_squeeze`** (and the regime-flip alignment counts): excluded
   here per rule 4 — override only if the operator rules non-directional
   multiplicity counts outside the removed vote class.
3. **Vote removals** (§4: 6 candle aggregates + geometry composite
   pressures): approve retirement in the Phase-B rebuild?
4. **Recipe keys needing declared values at the recipe decision:** `q`
   (TOL quantile), `p_low`/`p_release` (squeeze percentiles), candle Stage-B
   percentile, EMA-retest K, W upgrade (reaction window: keep tau-12
   convention or TRAIN-fit).
5. **Session template normalization** (§1.1: `break_held` +
   `bars_since_retest` folded into the unified retest vocabulary): confirm.
6. **Surface growth** (513→684, 111→208 nearly doubles the per-TF width):
   accept the re-derived encoder/compute budget under the 20G cap, or trim
   Phase B?

## 9. What remains unproved

Everything numerical about behaviour: event base rates, registry costs,
fitted constants, and any direction-edge claim (the 2026-08-09 walk-forward
refutation stands; this design promises representability of the operator's
confluence events, not edge). All eight source reports are proven-from-source
reviews; this synthesis adds only arithmetic and the boundary decisions
recorded above. No repo code was changed by this document.

## 10. Stage-4 amendment (2026-08-11): presence-mask saturation at D1

First real-tape measurement (native-M5 V4 tape → D1, TRAIN fit band 1.2207
ATR, N=2304 D1 TRAIN bars, N_candidates=9505): at D1 scale an ACTIVE
≥3-touch line above/below the close almost always exists inside the 252-bar
candidate window. `geomline_below_active` is exactly 1.0 on every declared
2021–2026 D1 row (871 touch events fired; all six sibling attributes
non-constant); `geomline_above_active` escapes constancy only via a handful
of 2024 zero-days. The first V29 chain run (`…20260811_V29B`) therefore went
RED on `HTF_V4_CACHE_FULL_INPUT_LIVENESS_FAIL: D1:constant_fields=
['geomline_below_active']` — the gate measured a true saturation, not wiring
death.

Resolution (owner: `htf_features.HTF_V4_PRESENCE_MASK_SATURATION_CONTRACT`,
liveness schema v2→v3): the B.5 flag-disambiguated-zero masks must stay (a
0-attribute row is only readable through them), and a constant mask is
admitted ONLY as saturation — exact value 1.0, every sibling attribute
non-constant on the same TF, paired touch event firing — and is recorded
explicitly as `saturated_presence_masks` in the liveness payload, re-proved
by the strict validator. Constant 0.0, dead siblings, silent events and all
non-mask constants remain RED. Two admitted masks are necessarily identical
constant columns; exactly that pair is exempt from the duplicate check
(each admission proved independent wiring). v2 payloads (the immutable V28
baseline cache) stay valid under their own exact key set.

## 11. Stage-4 amendment superseded (2026-08-15): the masks are retired

§10's resolution no longer holds and its mechanism no longer exists. The
`saturated_presence_masks` admission was withdrawn (the V4 liveness owner now
carries a strict "no constant-field or saturation exception exists" contract,
bound by `tests/test_htf_v4_liveness_saturation_contract.py`), which left the
saturating masks with no route to green.

`geomline_above_active` and `geomline_below_active` are therefore retired from
`TRENDLINE_REGISTRY_SLOT_FEATURE_NAMES_V1` and from every surface derived from
it. PROVEN FROM SOURCE in `trendline_registry_v1._emit_row`: one branch wrote
both the mask and `geomline_*_active_count`, and the mask was set exactly when
that side's ACTIVE population was non-zero — the mask WAS the count's `>= 1`
indicator, which is also what §10's own B.5 note said. Nothing the mask
carried is lost (CLAUDE.md rule 4): the graded, raw, uncapped count stays and
strictly dominates it, and the "0-attribute row" §10 wanted the mask for is
now read as `count == 0`.

One thing this is NOT: a repair of an `exact_duplicate` failure in
`entry_model_native_feature_availability_v1`. Mask and count are bit-identical
on any lane whose per-side ACTIVE population never exceeds 1, but that owner's
duplicate check is fed only the CANDIDATE pool
(`materialize_entry_model_native_train_feature_ranker_v1` builds it with
`name not in mandatory`), and both fields are MANDATORY. The duplicate would
have surfaced in the per-TF liveness owner instead. Stated here so a later
session does not inherit a mechanism that was never reachable.

Everything §10 measured about the market stands as of its own date — at D1
scale an ACTIVE three-touch line above or below the close almost always exists
inside the 252-bar candidate window. Note that §10 was measured on the
midnight-UTC D1 origin, which V30 package 3 replaced with the trading-day
origin on 2026-08-13; it has NOT been re-measured on the current axis, which
would require a TRAIN-fitted registry-constants artifact that does not exist.
Only the representation changed.
