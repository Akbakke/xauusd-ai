# Feature value review — 2026-08-13

> Historical-scope notice, updated 2026-08-28: this V29 review is retained for
> provenance only. It is not the active V46 field contract and may not be used
> to add/remove features or infer current liveness. V46's full-surface proof,
> canonical smoke status and next action are owned by the handover. The latest
> smoke's latest metadata repairs include `57d4ebcb`, `e0cf52ed` and `64d648da`;
> they change none of
> the historical review's feature conclusions.

Five parallel read-only reviews over the complete V29 surface (592 signal /
142+5 ctx / 173×5 MTF), graded against the REAL sealed V29J artifacts
(full-input liveness stats, TRAIN ranking scores, specialist audits) and the
source owners. Evidence classes: [M] measured on the sealed artifacts,
[PS] proven from source, [J] judgment, [U] unproven. No direction-edge claim
is made anywhere; the 2026-08-09 walk-forward refutation stands.

## A. USELESS / passengers (measured or proven)

1. **[M] 91 of 142 ctx_cont fields are mirrored bit-identically into the
   signal surface** — 15.4% of the 592 are exact copies (same std to 1e-12,
   same active_count).
2. **[PS] Three noise-amplifier bugs in `basic_v1.py`** (the exact class
   repaired in `htf_features` 2026-08-09, never backported):
   `_v1_kama_slope_30` (`np.diff n=5`), `_v1_tema_slope_20` (`n=3`),
   `_v1_bb_bandwidth_delta_10` (`n=3`; TRAIN mean 1.8e-09 [M]). They are
   high-pass noise, feed `slope_score` (0.08×2) and five vol composites.
3. **[PS] `DIP_STRUCT`: 30 of 36 ctx fields are a 2-bit quantization of
   `mom_5_atr`×`mom_20_atr` signs** (the four quadrant flags × 5 TFs, depth
   = same ratio, agree/dip = sums/products of the flags); 25 are terminal
   (no consumer). [M] the m5/m15/h1 ladder rungs are statistically
   indistinguishable (active rates within 0.0011).
4. **[PS] Six exact-affine duplicates in chart/candle**:
   `pattern_close_pressure_signed` ≡ 2·`close_location`−1 (clip inactive);
   `geometry_major_level_proximity_max`, `channel_position_low_to_high`,
   `channel_center_bias`, `channel_edge_pressure`,
   `support_minus_resistance_stack` — all exact functions of two stacks +
   one ctx field; three are pinned mandatory.
5. **[M] `session_tradable` ≡ ¬`is_ASIA`** (active counts sum exactly to
   row count); `smc_choch_recent_tau12` ≡ `tau24` (bit-identical ranking
   scores); `mtf_pattern_{body,upper,lower}_share` ≡ `*_pct` on H1/H4/D1
   (identical values_sha256 — 6 columns the duplicate rejector missed).
6. **[M] The 133 cut removes almost none of the measured noise floor**: 51
   of 77 ranker-rejected candidates remain mandatory via ctx_cont —
   including the single worst measured feature (`bars_since_swing_low`,
   score 4.4e-06 = 5,600× below top).
7. **[PS] `V3_EXTENSION` (5/5)**: `smc_premium_state` = exact product of two
   BASE fields; `hour/dow_sin/cos` strictly dominated by the session group.
8. **[M] hammer/shooting_star fire on ~51/50% of bars** (no gate) — the
   free gate (`range_expansion_vs_prev5`) is computed and gates nothing.
9. Already-decided removal candidates re-confirmed with stronger evidence:
   6 candle aggregate votes + 58 chart-geometry composite pressures (incl.
   4 misnomers containing no line/retest); the last removal blocker (aux
   tautologies) is resolved.
10. **[M] VAL-design defect quantified**: 18 D1/H4-regime fields have
    exactly zero variance on the June VAL (61 D1 bars) — including the
    top-1 ranked feature overall (`d1_trend_age_mature_flag_v3`).

## B. DEFINITELY MISSING (ranked by leverage)

1. **Session-anchored levels** (PDH/PDL/PDC, Asia hi/lo, week open, pwc) —
   engine built, kind not admitted (`LEVEL_KINDS_IMPLEMENTED_V1` excludes
   it); plus the pivot "day" uses UTC-midnight while the session owner
   opens 22:00 (two day-clocks).
2. **Bid/ask spread dynamics** — `bid_high/bid_low/ask_high/ask_low` are
   read by NO feature owner; Δspread, session-conditional spread z,
   widening events, intrabar quote geometry. Role: abstention/regime
   evidence (fenced from the refuted microstructure-as-direction probe).
   Ten-minute variance audit on declared bytes first (the legacy
   `_v1_spread_z` died constant on the OLD route).
3. **Per-TF volume normalization** — `_resample_ohlcv` already sums volume
   per TF; zero volume fields in the 173. Declared prerequisite of
   already-approved momentum G4 (break/sweep-on-volume).
4. **Level density / second-nearest slot** — registry state holds the whole
   population; emission takes nearest-per-side only. A 4-level shelf and an
   isolated level emit identical rows. Mirror in trendlines.
5. **Volume-at-level/touch** — one extra accumulator per level (same shape
   as `reaction_sum_atr`).
6. **Session-conditional baselines** — x_t vs TRAIN-fitted
   minute-of-week/session-slot median/IQR (the
   `SPREAD_RATIO_TANH_SCALE_TRAIN_P90` fit precedent); coarse slots per
   rule 2f.
7. **Squeeze state/release/duration/box** (vol G1/G2/G3/G5) — no duration
   field exists anywhere in vol; `compression_persistence_score` is a 2-bar
   smoother.
8. **VWAP bands + session-clock anchor** — current VWAP dist is
   ATR-normalized, not σ_vwap-normalized; anchor is a third clock
   (midnight).
9. **Measured-move projections** — range-height at break is already in
   registry state.
10. **Volume profile (POC/value area) from M5 candles** — largest
    conceptual hole, highest arbitrariness (4 unsourced constants);
    G0-style pre-measurement required before any build.
11. Un-flipped Phase-A remnants [PS]: per-TF swing V29 events (9×5) built
    but not composed; momentum G3 raw-RSI ctx scalars absent (M5/H1/H4 raw
    RSI missing while M15/D1 present — inconsistency, not design).

## C. MAKE SMARTER (implemented but too crude)

1. **Duration everywhere it's missing**: squeeze `bars_in_squeeze` +
   release-latch; `geomline_bars_since_break` (trendline registry has no
   persistent break memory — levels do); M5-local EMA-cross/state ages
   (Entry sees a 1-bar spike, Exit sees the aged version).
2. **Momentum rate**: RSI velocity (4 RSI representations, zero
   velocities); divergence STRENGTH (declared in design §3, built as
   event+age only — magnitude discarded); mom_5-vs-mom_20 horizon spread
   (impulse acceleration).
3. **Signed DI spread** beside `adx_centered` — plus_di/minus_di computed
   and discarded on the next line; TR-based direction, orthogonal to every
   EMA-based sign.
4. **Distance-to-flip per TF** for the regime enum (D1 has it —
   `d1_dist_to_boundary_v3`; other TFs emit only the class id) + fit the
   0.3 threshold as a TRAIN quantile (currently unsourced literal).
5. **Graded occupancy instead of saturated masks**: count of ACTIVE lines
   per side (cannot saturate, disambiguates the zero-attribute row, and IS
   the density feature from B.4) — converts the D1 saturation exemption
   into evidence.
6. **Level registry precision**: split `touch_count` into `member_pivot_count` +
   post-birth `test_count`; sign `level_bars_since_break` by break side.
7. **Candle gates**: `hammer_event_quality = hammer ×
   range_expansion_vs_prev5` (zero new constants); doji Stage-B
   TRAIN-percentile flag; replace the tick-exact piercing/dark-cloud gap
   gate with the family's ATR tolerance.
8. **bps→ATR sibling emission** for the local layer + micro momentum
   (era-proxy removed, vol-proxy not; pre-registered saturation measurement
   first per design §6.4).
9. **Elapsed-wall-clock ages beside bar-count ages** — every age in the
   system counts observed bars; 12 bars = 60 min midweek or 60 hours across
   a weekend. Correctness observation on all V29 age fields.
10. `atr_percentile_blend` (7 hand-weighted terms) → real per-TF rolling
    ATR percentiles (mechanism exists: `D1_atr_percentile_252`);
    `bb_position` unclipped signed σ; `H4_range_compression_ratio` (hole in
    the term structure); Kaufman efficiency ratio (computed inside KAMA and
    discarded).

## D. DEPENDENCY DEFECTS (check before the V29 verdict)

1. **Two session clocks with different hour sets** [PS] — **RESOLVED, V30
   package 3, 2026-08-13.** The `augment_forward_outcome_v2.py` hour sets
   (ASIA {22..8}, EU {7..16}, US {13..21}) are retired; the four
   `is_*_overlap/only` flags are derived from the ONE `session_detector.py`
   partition by `session_overlap_flags`. Mapping: `is_eu_us_overlap` :=
   OVERLAP, `is_us_only` := US, and EU splits into `is_asia_eu_overlap`
   (first 120 min — the width of the retired ASIA_HOURS ∩ EU_HOURS
   intersection, re-expressed from the surviving EU open) and `is_eu_only`
   (the 180-min remainder). The `active_session ≡ 1` proof was re-derived on
   the single partition and is now strictly stronger (no cross-clock side
   condition). DST posture is UNCHANGED and still open: the boundaries remain
   UTC-fixed, so the ~1-hour seasonal phase error is not addressed here.
2. **Two daily clocks** [PS] — **RESOLVED, V30 package 3, 2026-08-13.** The
   D1 bin now opens at 22:00 UTC (`MULTI_TF_RESAMPLE_ORIGIN_OFFSET`), the same
   trading-day clock the session-anchored V29 levels use. MEASURED on the
   complete declared native M5 tape: stub bins (<=10% of 288 M5 bars) fell
   from 401/2,360 = 16.99% to 1/1,958 = 0.05%, and Sunday-bin median coverage
   rose from 8.33% to 95.83%; the one remaining stub is a genuine 2019-02-15
   tape gap, not a clock artifact. NOT covered by this decision: the intraday
   session VWAP (`htf_features._session_vwap`) still resets at midnight UTC —
   a third daily clock, now the only one left, and its own recipe decision.
3. **Four ATR-14 conventions live simultaneously** [PS]; the swing
   normalizer (`min_periods=1`) inflates all `*_atr` swing fields for the
   first ~13 bars of any segment.
4. **Regime threshold 0.3 unsourced + chatter risk**: flip events are rare
   (421-1,543 TRAIN); denominator wobble near |slope_atr|≈0.3 could be a
   meaningful fraction of the event population. Ten-minute diagnostic:
   distribution of |slope_atr|−0.3 + ε-sensitivity of flip counts.
5. **Routing findings** [PS]: vol_compression owns the
   compression_expansion objective but receives 2 fields/TF while the
   candle encoder holds five compression/expansion measures;
   chart_geometry (48 mandatory incl. all trendline fields) and
   price_action get ONE ctx dim each; BOS/CHoCH owned by smc on the MTF
   lane and structure_swing on the M5 lane; `struct_*` (28 ctx dims) are
   momentum quadrants wearing a structure name; two stale prose counts in
   the signal contract ("479/346", "513-field") vs derived 558/425/592;
   the lexical classifier disagrees with the explicit tuples on 70/173
   names (the `_atr`-suffix-captures-ownership class).
6. **Dead alternative normalization owner**:
   `share_temporal_alias_stats_from_ctx` has no production caller and
   inverts the declared direction — rule-10 removal candidate.
7. **Weakest unprotected liveness margins** [M]: five event fields pass the
   1% floor by 0.2-0.5pp without rare-event registration
   (`bull/bear_divergence_event`, `rsi_cross_up_30`, `rsi_cross_down_70`,
   `session_change_flag`, `m15_regime_changed_flag_v3`).

## Suggested order (rule 22: measure → recipe value → extend owner → build)

1. Ten-minute diagnostics first: regime-threshold chatter; D1 stub-bar
   histogram; spread/quote-column variance on declared bytes.
2. One-line bug repairs: the three `basic_v1` noise amplifiers (adopt the
   existing k-bar-change convention).
3. Recipe decisions: session hour-set unification (+DST posture); D1
   rollover clock.
4. Emission-only wins: DI spread, divergence strength, level
   density/second slot, graded occupancy, signed break age,
   `H4_range_compression_ratio`, efficiency ratio.
5. Phase-A completion: swing per-TF flip, raw-RSI ctx, M5-local ages.
6. Phase B per the design doc: session-anchored levels first (engine
   ready), then squeeze/box, volume-z per TF, VWAP bands, spread dynamics.
7. Reduction wave (rule 4-safe, after the V29 eval verdict): struct_* block,
   V3_EXTENSION, exact-affine duplicates, mirrors policy for ctx→signal.

Every "would pay" claim is [U] until the pre-registered evaluation ladder
says otherwise.

## V30 package 4 — spread dynamics performed (2026-08-13)

Item 6 of the suggested order above ("spread dynamics") is taken, as an MVP of
exactly three fields. Recording what IS and is NOT verified (rule 25c).

**Measured** (complete declared native M5 tape, N=537,861 rows, 2026-08-13):
`spread_bps` mean 2.23 / std 2.61 / p1 1.11 / p99 14.66 over ~3261 distinct
rounded values; the 1-bar change is nonzero on 99.93% of rows with std 1.161;
the intrabar quote envelope `ask_high - bid_low` is 10.21 bps mean / 8.64 std;
the quote-range asymmetry is nonzero on 87.85% of rows with std 0.969 bps;
hourly median spread ranges 1.62 (h11) to 2.08 (h22), a 1.3x band. This is the
evidence that the quote surface is alive on the native tape — the legacy
`_v1_spread_z` died CONSTANT only on the retired canonical route.

**Built** — `gx1/features/micro_structure_v1.compute_spread_dynamics_features`,
a second bounded producer beside the five-field OHLC micro surface (that
function is untouched):

| field | formula | domain |
|---|---|---|
| `spread_bps_delta_1` | `spread_bps[t] - spread_bps[t-1]` | signed bps, honest NaN on row 0 |
| `spread_extremes_sum_bps` | `((ask_high - bid_high) + (ask_low - bid_low)) / close * 1e4` | non-negative by the enforced quote geometry (repaired 2026-08-19: the retired `spread_intrabar_range_bps` envelope was r=0.9259 with the mid bar range) |
| `quote_range_asymmetry_bps` | `((ask_high-ask_low) - (bid_high-bid_low)) / close * 1e4` | signed bps |

**Purpose fence:** abstention and execution-regime evidence, NOT a direction
signal. Orthogonal microstructure sources were OOS-refuted for DIRECTION on
the retired chain and that fence stands; nothing post-model may consume these
(rule 3).

**Proven from source**
- One spread owner. The level is not recomputed: `derive_observed_spread_bps`
  is called on `frame[["bid_close","ask_close"]]`, the exact call
  `entry_model_native_state_v2.compute_causal_market_rank_inputs` makes to put
  `spread_bps` on the ctx surface, so `spread_bps_delta_1` is the difference of
  the emitted ctx column by construction.
- Causal: every input is an aggregate of the decision bar's OWN closed quotes;
  the delta additionally reads `t-1` only. Same closed-bar convention as every
  other current-bar ctx field.
- No new magnitude: `/ close * 1e4` is this owner's own bps convention; no
  threshold and no clip is introduced (this owner clips nothing).
- Source availability on every offline lane: the four quote extremes are
  members of `CANONICAL_NATIVE_REQUIRED_COLUMNS` and survive unprojected from
  the native stage through `build_canonical_v2` (`basic_v1.build_basic_v1`
  mutates and returns the same frame) into `_finish_canonical_v3_context`,
  where the block is emitted; the Entry dataset builder reads them from its
  existing `_risk_tape_cols` tape join. The M1/M5 enriched OUTPUT_COLUMNS
  projection happens AFTER the ctx block is computed.
- Routing: all three are pinned explicitly to `session_regime_encoder`
  (`quote_range_asymmetry_bps` would otherwise lose the lexical race on
  `range`).

**NOT examined / unproved**
- No real-tape values, liveness or ranking exist for these three fields yet:
  they await the V30 rebuild chain. The distributional numbers above were
  measured on the raw quote columns, not on the emitted ctx columns.
- The fields enter the ctx surface, i.e. the TRAIN-ranked candidate pool
  (like `spread_bps` itself), NOT the mandatory causal families. They may be
  ranked out. `MODEL_NATIVE_SIGNAL_DIM` is unchanged at 608; only
  `MODEL_NATIVE_CTX_CONT_DIM` moves 155 -> 158.
- Whether an execution-regime signal actually improves abstention quality is
  [U] until the pre-registered evaluation ladder says otherwise.
- The LIVE serve lane is not exercised here. `ModelNativeStateBuilder`
  receives the joined cv3+BASE28 frame; the four intrabar quote columns are
  required by the persisted pair-v2 contract
  (`CANONICAL_NATIVE_REQUIRED_COLUMNS[1:]`) and are consumed by
  `v12_trade_state`, but no test in this wave proves the live frame carries
  them. A live frame without them now fails closed loudly, which is correct.
