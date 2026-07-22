# Foundation feature routing audit — 2026-07-22

## Decision

V19 is rejected for training. Its real signal-manifest producer emitted zero
of the 57 implemented foundation fields, while the feature audit requires all
57. The active v4 contract retains the complete layer as mandatory:

- 34 base + 479 specialist = 513 signals;
- 378 mandatory fields from 12 code-owned layers;
- 101 deterministic TRAIN-only ranked fields;
- 57/57 foundation fields retained and routed;
- no foundation field is an exact duplicate of any field in the former
  316-field mandatory surface.

The V19 comparison below is diagnostic only. It uses all 5,808 validation rows
after discarding the first 96 rows so event-age state is settled. `r` is the
Pearson correlation with the closest former mandatory field. Correlation does
not select, remove or authorize a field.

## Root causes and repairs

1. `build_chart_layer()` computed the foundation layer, and downstream smart
   layers consumed it, but the code-owned mandatory registry did not retain it
   in the final model tensor.
2. The ranker's reflective candidate discovery did not expose
   `FOUNDATION_STRUCTURE_FEATURE_NAMES`; therefore ranking could not rescue any
   foundation field.
3. Synthetic unit manifests manually inserted the fields and masked the real
   producer mismatch.
4. H1/M15 `ATR14 / ATR100` and relative Bollinger bandwidth were interpreted
   with reversed compression signs in four builders. A single strict transform
   owner now distinguishes compression from expansion, and release requires
   lagged compression followed by positive expansion acceleration.
5. The split-constant allowlist was deleted. Selected-field learnability is
   TRAIN-owned; all selected fields must remain finite in every split, while
   all foundation outputs and sources must remain live in TRAIN, VAL and TEST.
6. V20 exposed one additional cross-contract mismatch after this audit: four
   geometry fields consumed by the structural auxiliary-label producer were
   not all guaranteed by the mandatory registry. All four are now mandatory,
   and a single 19-requirement registry is imported by both the signal
   contract and dataset builder. This raises the mandatory partition from 373
   to 377 and reduces the TRAIN-ranked remainder from 106 to 102 without
   changing 513 total.
7. V21 exposed the remaining pretrain-polarity mismatch:
   `chart.geometry_support_minus_resistance_stack` was consumed by
   support/resistance memory and required by the channel-polarity audit, but
   was not guaranteed by the mandatory registry. It is now the fifth promoted
   geometry prerequisite, raising the mandatory partition from 377 to 378 and
   reducing the TRAIN-ranked remainder from 102 to 101. The audit computes
   target liveness/consistency independently of polarity availability, while a
   missing polarity field remains hard RED.

The `eu` token in field names denotes European trading-session hours. It is a
time/session feature, not a price series or an independently traded instrument.

## Exact family routing

| Foundation evidence | Count | Learned owner |
|---|---:|---|
| HH/HL/LH/LL, BOS/CHoCH age, impulse/pullback | 19 | structure/swing |
| Sweep/reclaim and false-breakout proxies | 5 | SMC/liquidity |
| Compression, expansion and release | 5 | volatility/compression |
| Session × structure interactions | 28 | session/regime |
| **Total** | **57** | **4 specialists; all join learned cross-specialist fusion** |

## Field-by-field duplicate and routing check

| Foundation field | Route | Closest former mandatory field | r |
|---|---|---|---:|
| `chart.foundation_hh_state` | structure | `chart.structure_swing_swing_leg_quality_up` | +0.970 |
| `chart.foundation_hl_state` | structure | `chart.structure_swing_hh_hl_consistency_up` | +0.992 |
| `chart.foundation_lh_state` | structure | `chart.structure_swing_lh_ll_consistency_down` | +0.924 |
| `chart.foundation_ll_state` | structure | `chart.structure_swing_swing_leg_quality_down` | +0.954 |
| `chart.foundation_structure_up_minus_down` | structure | `chart.structure_swing_swing_leg_quality_balance` | +0.986 |
| `chart.foundation_bos_up_age_bars` | structure | `chart.structure_swing_bos_choch_recency_alignment_up` | -0.804 |
| `chart.foundation_bos_down_age_bars` | structure | `chart.structure_swing_bos_choch_recency_alignment_down` | -0.840 |
| `chart.foundation_bos_up_recent_tau24` | structure | `chart.structure_swing_bos_choch_recency_alignment_up` | +0.925 |
| `chart.foundation_bos_down_recent_tau24` | structure | `chart.structure_swing_bos_choch_recency_alignment_down` | +0.927 |
| `chart.foundation_bos_recent_balance` | structure | `chart.structure_swing_bos_choch_recency_alignment_down` | -0.865 |
| `chart.foundation_choch_age_bars` | structure | `chart.structure_swing_choch_failure_up_risk` | -0.767 |
| `chart.foundation_choch_recent_tau24` | structure | `chart.structure_swing_choch_failure_down_risk` | +0.794 |
| `chart.foundation_bars_since_structure_break_min` | structure | `chart.structure_swing_structure_compression_pressure` | +0.801 |
| `chart.foundation_sweep_low_reclaim_up_proxy` | SMC/liquidity | `chart.smc_liquidity_false_breakout_quality_long` | +0.946 |
| `chart.foundation_sweep_high_reclaim_down_proxy` | SMC/liquidity | `chart.smc_liquidity_false_breakout_quality_short` | +0.963 |
| `chart.foundation_false_breakout_high_followthrough_down_proxy` | SMC/liquidity | `chart.smc_liquidity_false_breakout_quality_short` | +0.281 |
| `chart.foundation_false_breakout_low_followthrough_up_proxy` | SMC/liquidity | `chart.smc_liquidity_mtf_confluence_sweep_reclaim_long` | +0.461 |
| `chart.foundation_sweep_reclaim_balance_proxy` | SMC/liquidity | `chart.smc_liquidity_mtf_confluence_sweep_reclaim_short` | -0.798 |
| `chart.foundation_compression_state` | volatility | `vol_compression.range_compression_stack` | -0.806 |
| `chart.foundation_expansion_state` | volatility | `vol_compression.expansion_state_score` | +0.909 |
| `chart.foundation_compression_release_trigger` | volatility | `session_regime.session_vol_spread_tail_risk` | +0.564 |
| `chart.foundation_compression_release_up` | volatility | `vol_compression.compression_release_up_pressure` | +0.542 |
| `chart.foundation_compression_release_down` | volatility | `vol_compression.compression_release_down_pressure` | +0.593 |
| `chart.foundation_impulse_direction` | structure | `trend.ema_mtf_score` | +0.930 |
| `chart.foundation_impulse_age_proxy` | structure | `trend.ema_trend_age_mean` | +0.858 |
| `chart.foundation_pullback_phase_up` | structure | `chart.structure_swing_pullback_phase_continuation_up` | +0.824 |
| `chart.foundation_pullback_phase_down` | structure | `chart.structure_swing_pullback_phase_continuation_down` | +0.834 |
| `chart.foundation_pullback_depth_norm` | structure | `chart.structure_swing_pullback_depth_quality` | +0.889 |
| `chart.foundation_impulse_pullback_alignment` | structure | `chart.structure_swing_pullback_phase_continuation_down` | -0.809 |
| `chart.foundation_asia_x_hh_state` | session | `session_regime.mtf_confluence_abstain_score` | +0.572 |
| `chart.foundation_asia_x_hl_state` | session | `chart.structure_swing_hh_hl_consistency_up` | +0.560 |
| `chart.foundation_asia_x_lh_state` | session | `session_regime.asia_structure_mean_reversion_pressure` | +0.804 |
| `chart.foundation_asia_x_ll_state` | session | `session_regime.asia_structure_mean_reversion_pressure` | +0.741 |
| `chart.foundation_asia_x_bos_balance` | session | `chart.structure_swing_bos_choch_recency_alignment_up` | +0.543 |
| `chart.foundation_asia_x_choch_recent` | session | `chart.structure_swing_choch_failure_up_risk` | +0.892 |
| `chart.foundation_asia_x_sweep_reclaim_balance` | session | `chart.smc_liquidity_premium_discount_reclaim_confluence_short` | -0.514 |
| `chart.foundation_eu_x_hh_state` | session | `chart.structure_swing_swing_leg_quality_up` | +0.587 |
| `chart.foundation_eu_x_hl_state` | session | `chart.structure_swing_hh_hl_consistency_up` | +0.610 |
| `chart.foundation_eu_x_lh_state` | session | `session_regime.eu_h4_d1_structure_continuation_short` | +0.815 |
| `chart.foundation_eu_x_ll_state` | session | `session_regime.eu_h4_d1_structure_continuation_short` | +0.824 |
| `chart.foundation_eu_x_bos_balance` | session | `chart.structure_swing_bos_choch_recency_alignment_down` | -0.569 |
| `chart.foundation_eu_x_choch_recent` | session | `chart.structure_swing_choch_failure_down_risk` | +0.882 |
| `chart.foundation_eu_x_sweep_reclaim_balance` | session | `chart.smc_liquidity_mtf_confluence_sweep_reclaim_short` | -0.561 |
| `chart.foundation_us_x_hh_state` | session | `session_regime.us_late_session_structure_chase_risk` | +0.462 |
| `chart.foundation_us_x_hl_state` | session | `chart.structure_swing_hh_hl_consistency_up` | +0.460 |
| `chart.foundation_us_x_lh_state` | session | `session_regime.us_late_session_structure_chase_risk` | +0.643 |
| `chart.foundation_us_x_ll_state` | session | `session_regime.us_late_session_structure_chase_risk` | +0.666 |
| `chart.foundation_us_x_bos_balance` | session | `chart.structure_swing_bos_choch_recency_alignment_down` | -0.510 |
| `chart.foundation_us_x_choch_recent` | session | `chart.structure_swing_choch_failure_down_risk` | +0.883 |
| `chart.foundation_us_x_sweep_reclaim_balance` | session | `chart.smc_liquidity_mtf_confluence_sweep_reclaim_short` | -0.457 |
| `chart.foundation_overlap_x_hh_state` | session | `session_regime.spread_cost_x_overlap_risk` | +0.555 |
| `chart.foundation_overlap_x_hl_state` | session | `session_regime.spread_cost_x_overlap_risk` | +0.505 |
| `chart.foundation_overlap_x_lh_state` | session | `session_regime.eu_h4_d1_structure_continuation_short` | +0.723 |
| `chart.foundation_overlap_x_ll_state` | session | `session_regime.eu_h4_d1_structure_continuation_short` | +0.735 |
| `chart.foundation_overlap_x_bos_balance` | session | `chart.structure_swing_break_confirmation_balance` | +0.481 |
| `chart.foundation_overlap_x_choch_recent` | session | `chart.structure_swing_choch_failure_down_risk` | +0.886 |
| `chart.foundation_overlap_x_sweep_reclaim_balance` | session | `chart.smc_liquidity_mtf_confluence_sweep_reclaim_short` | -0.437 |

The negative compression correlation is expected in this diagnostic: the
comparison column is the rejected V19 value built with the inverted source
semantics. Both the primitive foundation field and its higher-order volatility
counterpart were corrected before V20.

## Learned cooperation path

The 57 fields do not form another direction policy. They enter four specialist
encoders, then participate in learned specialist interaction, gate and
cross-attention blocks. Their evidence joins the five-timeframe cooperation
surface and the exact 26-group/96-value fusion. Only the final calibrated
three-logit model output may select LONG, SHORT or FLAT.

This routing proves presence, liveness and differentiable connectivity. It
does not prove predictive edge. V20 later failed closed on a separate
ranking-owned structural-label prerequisite, and V21 later failed on the
pretrain-polarity dependency recorded above. V22 must pass fresh ranking, rebuild,
foundation/source liveness, target/specialist audits, smoke training,
untouched OOS evaluation, replay, train-equals-serve parity, learned sizing and
shadow evidence.
