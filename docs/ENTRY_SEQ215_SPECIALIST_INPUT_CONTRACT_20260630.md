# Entry Seq215 Specialist Input Contract - 2026-06-30

Status: report-only proof note for `challenger_seq215`. Do not treat this as
training approval. Smoke training, candidate training, replay, IQL, shadow,
live and promotion remain closed behind their normal gates.

## Sources

- Machine contract: `gx1/features/entry_specialist_feature_groups_v1.py`
- Audit writer: `gx1/scripts/audit_entry_specialist_feature_groups_v1.py`
- Active seq215 specialist audit:
  `/home/andre2/GX1_DATA/reports/entry_specialist_feature_group_audit_20260628_v1/challenger_seq215_20260630_contract8/ENTRY_SPECIALIST_FEATURE_GROUP_AUDIT_latest.json`
- Seq215 smoke dataset manifest:
  `/home/andre2/GX1_DATA/runs/FASE2B_REGIME_V4_20260605/v10_6yr_rebuild_20260628_foundation_seq146/v10_dataset_challenger_seq215_smoke_20260630/SMOKE_DATASET_MANIFEST.json`
- Contract proof tests: `tests/test_entry_seq215_model_contract_proof.py`

## Contract Summary

- `contract_mode`: `challenger_seq215`
- `signal_field_count`: 215
- `selected_feature_count`: 174
- Context: `ctx_cont_dim=142`, `ctx_cat_dim=5` in the specialist audit
  context contracts.
- Trainable specialists: 8 exact specialists:
  `structure_swing_encoder`, `smc_liquidity_encoder`, `trend_ema_encoder`,
  `vol_compression_encoder`, `momentum_flow_encoder`,
  `session_regime_encoder`, `chart_geometry_encoder`,
  `price_action_candle_encoder`
- Diagnostic/non-trainable groups: `neutral_bridge_anchor` has 7 inputs;
  `unmapped` has 0 inputs.
- Active heads: `direction`, `tradable`, `path_quality`, `mfe_first_n`,
  `bad_path`, `clean_edge`, `survival`, `tf_agreement`,
  `path_quality_log_var`, `position_size`, `dip`, `forecast`, `timing`,
  `tail_risk`, `vol_forecast`, `mtf_direction`
- Blocked heads: `hold_horizon` only.
- Audit status: `decision=PASS`, `specialist_model_contract_valid=true`,
  `foundation_objective_routing_all_present_and_expected=true`,
  `specialist_input_liveness_all_live=true`.

Note: the latest machine audit observes 31 `price_action_candle_encoder`
inputs. If another handoff says "candlestick 28", use the audit JSON above as
the source of truth for the current seq215 contract.

## Family Map

| Family | Trainable in seq215 | Input count | Role |
| --- | --- | ---: | --- |
| `neutral_bridge_anchor` | no | 7 | Explicit neutral XGB bridge priors only. |
| `structure_swing_encoder` | yes | 37 | HH/HL/LH/LL, BOS/CHoCH, structure breaks, impulse and pullback phase. |
| `smc_liquidity_encoder` | yes | 23 | Sweeps, reclaim, false breakout, support/resistance, wick liquidity and premium/discount. |
| `trend_ema_encoder` | yes | 6 | EMA stack/slope, price-vs-EMA and MTF trend agreement proxies. |
| `vol_compression_encoder` | yes | 21 | ATR, volatility regime, compression, squeeze and expansion/release. |
| `momentum_flow_encoder` | yes | 5 | Returns, CLV, dip confirmation and short-term follow-through. |
| `session_regime_encoder` | yes | 44 | EU/session/regime and session x structure interactions. |
| `chart_geometry_encoder` | yes | 41 | Numeric trendlines, S/R channels, Fib zones, EMA crosses and chart-pattern pressure. |
| `price_action_candle_encoder` | yes | 31 | Closed-bar body/wick and candlestick pattern pressure. |
| `unmapped` | no | 0 | Must remain zero. |

## Input Index Map

`neutral_bridge_anchor`:
0 `p_long`; 1 `p_short`; 2 `p_flat`; 3 `p_hat`; 4 `uncertainty_score`;
5 `margin_top1_top2`; 6 `entropy`.

`structure_swing_encoder`:
28 `smc_swing_state`; 29 `smc_bos_up`; 30 `smc_bos_down`; 31 `smc_choch`;
53 `chart.lh_x_ema50_200`; 54 `chart.lh_x_ema50_200_atr`;
61 `chart.lh_x_d1_upper`; 63 `ctx_cont.struct_pullback_depth_m5_v3`;
64 `chart.bos_x_ema50_200_atr`; 67 `chart.bos_x_ema50_200`;
68 `chart.lh_x_price_vs_ema200`; 72 `chart.bos_x_price_vs_ema200`;
75 `chart.near_recent_swing_low`; 81 `chart.bos_x_vol_stack`;
83 `chart.bos_x_h1_vol_pct`; 84 `chart.bos_x_tf_agreement`;
85 `chart.bos_x_d1_atr_pct`; 88 `chart.bos_x_d1_upper`;
89 `chart.foundation_hh_state`; 90 `chart.foundation_hl_state`;
91 `chart.foundation_lh_state`; 92 `chart.foundation_ll_state`;
93 `chart.foundation_structure_up_minus_down`;
94 `chart.foundation_bos_up_age_bars`;
95 `chart.foundation_bos_down_age_bars`;
96 `chart.foundation_bos_up_recent_tau24`;
97 `chart.foundation_bos_down_recent_tau24`;
98 `chart.foundation_bos_recent_balance`;
99 `chart.foundation_choch_age_bars`;
100 `chart.foundation_choch_recent_tau24`;
101 `chart.foundation_bars_since_structure_break_min`;
112 `chart.foundation_impulse_direction`;
113 `chart.foundation_impulse_age_proxy`;
114 `chart.foundation_pullback_phase_up`;
115 `chart.foundation_pullback_phase_down`;
116 `chart.foundation_pullback_depth_norm`;
117 `chart.foundation_impulse_pullback_alignment`.

`smc_liquidity_encoder`:
32 `smc_sweep_up`; 33 `smc_sweep_down`; 34 `smc_sweep_size_atr`;
35 `smc_bars_since_sweep`; 36 `smc_premium_discount`;
41 `chart.premium_discount_x_level`;
52 `ctx_cont.sr_support_minus_resistance_prox`;
55 `chart.wick_level_x_level_prox`; 58 `chart.bos_x_level_prox`;
59 `chart.wick_x_major_level`; 60 `chart.wick_level_x_h1_vol_pct`;
66 `chart.wick_level_x_ema50_200`; 69 `chart.wick_level_x_vol_stack`;
71 `chart.sweep_size_x_vol_stack`;
76 `ctx_cont.smc_sweep_bull_pressure_last48`;
77 `ctx_cont.dist_to_m5_hi_atr`;
80 `ctx_cont.liquidity_hi_nearest_abs_atr`;
87 `chart.wick_level_x_d1_lower`;
102 `chart.foundation_sweep_low_reclaim_up_proxy`;
103 `chart.foundation_sweep_high_reclaim_down_proxy`;
104 `chart.foundation_false_breakout_high_followthrough_down_proxy`;
105 `chart.foundation_false_breakout_low_followthrough_up_proxy`;
106 `chart.foundation_sweep_reclaim_balance_proxy`.

`trend_ema_encoder`:
15 `ema20_slope`; 16 `pos_vs_ema200`; 18 `_v1_ema_diff`;
19 `_v1_close_ema_slope_3`; 22 `_v1_kama_slope_30`;
23 `_v1_tema_slope_20`.

`vol_compression_encoder`:
7 `_v1_atr14`; 8 `atr_z`; 12 `rvol_20`; 17 `_v1_pk_sigma20`;
21 `_v1_range_z`; 24 `_v1_bb_squeeze_20_2`;
25 `_v1_bb_bandwidth_delta_10`; 27 `_v1_kurt_r`; 37 `vol_z_20`;
38 `vol_ratio_5_20`; 39 `vol_pct_96`; 40 `signed_vol_z_20`;
73 `ctx_cont.H1_range_compression_ratio`; 78 `ctx_cont.d1_atr14_canon_v2`;
79 `ctx_cont.atr_ratio_h1_d1`; 86 `ctx_cont._v1h4_atr`;
107 `chart.foundation_compression_state`;
108 `chart.foundation_expansion_state`;
109 `chart.foundation_compression_release_trigger`;
110 `chart.foundation_compression_release_up`;
111 `chart.foundation_compression_release_down`.

`momentum_flow_encoder`:
9 `ret_1`; 10 `ret_5`; 11 `ret_20`; 20 `_v1_clv`;
70 `ctx_cont.dip_confirmed_h1_v3`.

`session_regime_encoder`:
42 `chart.eu_x_level_prox`; 43 `chart.is_eu_only_x_bos`;
44 `chart.wick_level_x_regime_div`; 45 `chart.eu_x_hh`;
46 `chart.eu_x_bos`; 47 `chart.eu_x_price_vs_ema200`;
48 `chart.is_eu_only_x_wick_level`; 49 `chart.eu_x_wick_level`;
50 `chart.eu_x_ema50_200_atr`; 51 `chart.eu_x_ema50_200`;
56 `chart.eu_x_d1_upper`; 57 `chart.is_eu_only_x_d1_loc`;
62 `chart.eu_x_vol_stack`; 65 `chart.is_eu_only_x_pullback`;
74 `chart.eu_x_pullback`; 82 `chart.is_eu_only_x_trend_proxy`;
118 `chart.foundation_asia_x_hh_state`;
119 `chart.foundation_asia_x_hl_state`;
120 `chart.foundation_asia_x_lh_state`;
121 `chart.foundation_asia_x_ll_state`;
122 `chart.foundation_asia_x_bos_balance`;
123 `chart.foundation_asia_x_choch_recent`;
124 `chart.foundation_asia_x_sweep_reclaim_balance`;
125 `chart.foundation_eu_x_hh_state`;
126 `chart.foundation_eu_x_hl_state`;
127 `chart.foundation_eu_x_lh_state`;
128 `chart.foundation_eu_x_ll_state`;
129 `chart.foundation_eu_x_bos_balance`;
130 `chart.foundation_eu_x_choch_recent`;
131 `chart.foundation_eu_x_sweep_reclaim_balance`;
132 `chart.foundation_us_x_hh_state`;
133 `chart.foundation_us_x_hl_state`;
134 `chart.foundation_us_x_lh_state`;
135 `chart.foundation_us_x_ll_state`;
136 `chart.foundation_us_x_bos_balance`;
137 `chart.foundation_us_x_choch_recent`;
138 `chart.foundation_us_x_sweep_reclaim_balance`;
139 `chart.foundation_overlap_x_hh_state`;
140 `chart.foundation_overlap_x_hl_state`;
141 `chart.foundation_overlap_x_lh_state`;
142 `chart.foundation_overlap_x_ll_state`;
143 `chart.foundation_overlap_x_bos_balance`;
144 `chart.foundation_overlap_x_choch_recent`;
145 `chart.foundation_overlap_x_sweep_reclaim_balance`.

`chart_geometry_encoder`:
146 `chart.geometry_mtf_trend_score`;
147 `chart.geometry_h8_proxy_trend_score`;
148 `chart.geometry_mtf_trend_agreement_pressure`;
149 `chart.geometry_mtf_trend_divergence_pressure`;
150 `chart.geometry_ema_stack_bull_pressure`;
151 `chart.geometry_ema_stack_bear_pressure`;
152 `chart.geometry_ema_cross_up_pressure`;
153 `chart.geometry_ema_cross_down_pressure`;
154 `chart.geometry_support_line_proximity_stack`;
155 `chart.geometry_resistance_line_proximity_stack`;
156 `chart.geometry_support_minus_resistance_stack`;
157 `chart.geometry_major_level_proximity_max`;
158 `chart.geometry_major_level_proximity_mean`;
159 `chart.geometry_channel_position_low_to_high`;
160 `chart.geometry_channel_center_bias`;
161 `chart.geometry_channel_edge_pressure`;
162 `chart.geometry_support_bounce_long_pressure`;
163 `chart.geometry_resistance_reject_short_pressure`;
164 `chart.geometry_trendline_break_up_pressure`;
165 `chart.geometry_trendline_break_down_pressure`;
166 `chart.geometry_failed_breakout_high_reversal_pressure`;
167 `chart.geometry_failed_breakout_low_reversal_pressure`;
168 `chart.geometry_fib_position_proxy`;
169 `chart.geometry_fib_retracement_236_proximity`;
170 `chart.geometry_fib_retracement_382_proximity`;
171 `chart.geometry_fib_retracement_500_proximity`;
172 `chart.geometry_fib_retracement_618_proximity`;
173 `chart.geometry_fib_retracement_786_proximity`;
174 `chart.geometry_fib_golden_zone_proximity`;
175 `chart.geometry_fib_pullback_long_pressure`;
176 `chart.geometry_fib_pullback_short_pressure`;
177 `chart.geometry_fib_extension_breakout_up_pressure`;
178 `chart.geometry_fib_extension_breakout_down_pressure`;
179 `chart.geometry_ascending_triangle_pressure`;
180 `chart.geometry_descending_triangle_pressure`;
181 `chart.geometry_bull_flag_pullback_pressure`;
182 `chart.geometry_bear_flag_pullback_pressure`;
183 `chart.geometry_compression_breakout_up_pressure`;
184 `chart.geometry_compression_breakout_down_pressure`;
185 `chart.geometry_late_trend_reversal_risk`;
186 `chart.geometry_line_pattern_tail_risk`.

`price_action_candle_encoder`:
13 `body_pct`; 14 `wick_asym`; 26 `_v1_body_share_1`;
187 `candle.pattern_body_share`;
188 `candle.pattern_upper_wick_share`;
189 `candle.pattern_lower_wick_share`;
190 `candle.pattern_close_location`;
191 `candle.pattern_body_direction`;
192 `candle.pattern_doji_score`;
193 `candle.pattern_hammer_bull_reversal_score`;
194 `candle.pattern_shooting_star_bear_reversal_score`;
195 `candle.pattern_marubozu_bull_score`;
196 `candle.pattern_marubozu_bear_score`;
197 `candle.pattern_bullish_engulfing_score`;
198 `candle.pattern_bearish_engulfing_score`;
199 `candle.pattern_inside_bar_compression_score`;
200 `candle.pattern_outside_bar_expansion_score`;
201 `candle.pattern_piercing_line_bull_score`;
202 `candle.pattern_dark_cloud_bear_score`;
203 `candle.pattern_tweezer_bottom_score`;
204 `candle.pattern_tweezer_top_score`;
205 `candle.pattern_morning_star_bull_score`;
206 `candle.pattern_evening_star_bear_score`;
207 `candle.pattern_three_white_soldiers_score`;
208 `candle.pattern_three_black_crows_score`;
209 `candle.pattern_bull_reversal_pressure`;
210 `candle.pattern_bear_reversal_pressure`;
211 `candle.pattern_bull_continuation_pressure`;
212 `candle.pattern_bear_continuation_pressure`;
213 `candle.pattern_indecision_breakout_setup`;
214 `candle.pattern_tail_rejection_risk`.

`unmapped`: none.
