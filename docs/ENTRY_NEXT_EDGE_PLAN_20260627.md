# Entry Edge Rebuild Plan 2026-06-27

Status: historical seed plan, superseded as active roadmap by
`docs/ACTIVE_SUPER_AI_BOT_GOAL_20260702.md`,
`docs/ENTRY_FOUNDATION_AUDIT_20260628.md` and
`docs/ENTRY_SEQUENTIAL_AI_SPECIALIST_BLUEPRINT_20260628.md`.

This plan replaced blind Entry Transformer retraining. It remains useful
historical context for why the project moved from broad XGB/V10 accuracy toward
sequence-aware specialist evidence. The current active path is now
foundation_seq146 -> challenger_seq215 -> smart_seq520_candidate -> capped smart
smoke -> candidate -> selective-edge/no-XGB ablation -> replay evidence ->
Entry IQL -> Entry-to-Exit handoff -> Exit Transformer/IQL. No shadow/live
decision can be made from this document alone.

Current 2026-07-02 rule: `smart_seq520_candidate` is structurally ready but
evidence-red on direction/class balance. The next smart smoke must prove the
FLAT/class-balance repair in the main direction head and MTF direction auxiliary
head, preserve exact specialist contracts, and improve replay-relevant slices.

## 2026-06-28 Sequential Pivot

Decision:

```text
XGBoost is dropped as a primary candidate.
Allowed use: cheap diagnostic / benchmark only.
Primary model track: sequential Transformer V10 + repaired Entry-IQL / other sequential models.
No shadow/live run in this phase. Use offline 2026 replay/backtest only.
```

Reason:

```text
XGBoost is non-sequential. It can see engineered lag/state columns, but it does
not model the market path over the prior bars the way the Transformer sequence
stack does. The current problem is tail risk/drawdown/path quality, so the main
candidate must explicitly consume sequence/history.
```

Sequential feature coverage audit:

```text
script=gx1/scripts/analyze_sequential_feature_coverage_v1.py
report=/home/andre2/GX1_DATA/reports/sequential_feature_coverage_20260628_v1
transformer_bundle=/home/andre2/GX1_DATA/runs/FASE2B_REGIME_V4_20260605/v10_6yr_rebuild_20260626_spreadfix/v10_bundle_6yr_symneg_smartctx_xgbfixed_h24_cwp030_res025_wd1e4
transformer_seq_len=96
transformer_seq_dim=41
transformer_ctx_dim=142
entry_iql_status=ENTRY_IQL_V2_PARTIAL_DEGRADES_VS_BASELINE
entry_iql_production_allowed=false
```

Audit finding:

```text
Transformer already has per-bar SMC/BOS/CHoCH/sweep/wick/volatility fields in
the 96-bar sequence. However, pivot distance, impulse/pullback structure,
HH/HL/LH/LL structure, and dip-distance are mostly context/state fields, not
per-bar sequence fields. Next sequential work should promote the strongest
tail-risk structure concepts into sequence-aware inputs or a dedicated temporal
structure encoder before another long retrain.
```

## 2026-06-28 Sequence Structure Layer V1

Tail/drawdown promotion analysis:

```text
script=gx1/scripts/analyze_sequence_feature_promotion_v1.py
report=/home/andre2/GX1_DATA/reports/sequence_feature_promotion_20260628_v1
ranked_features=588
ranked_non_seq_interactions=171807
manifest_features=48
xgboost_primary_candidate=false
```

Top promoted concepts:

```text
premium/discount x level
EU/session x level proximity
EU/session x BOS
wick x level x regime divergence
EU x HH
EU x EMA50/200 and price-vs-EMA200
LH x EMA50/200
pullback/dip/compression/liquidity context promoted from ctx-only to temporal candidates
```

Sequence structure feature-order/provenance layer:

```text
script=gx1/scripts/materialize_sequence_structure_features_v1.py
report=/home/andre2/GX1_DATA/reports/sequence_structure_feature_layer_20260628_v1
parquet=/home/andre2/GX1_DATA/reports/sequence_structure_feature_layer_20260628_v1/sequence_structure_features.parquet
rows=399417
selected_features=48
missing_features=0
base_seq_dim=41
proposed_seq_dim=89
parquet_sha256=98a8d220d412ba0d1cbe0f9519c4cd1f0b837233853d311584591a5c4565bc28
```

Important correction:

```text
The materialized parquet above is sample-aligned to existing V10 dataset rows
and must not be joined directly as raw per-bar sequence history. The V10 dataset
builder now supports true temporal seq extension by computing these same
features inline on merged3 for every raw per-bar row in the 96-bar window.
```

Builder integration smoke:

```text
builder=gx1/scripts/build_entry_v10_ctx_training_dataset_v3.py
new_flags=--seq-structure-manifest --seq-structure-compute-inline
smoke_report=/home/andre2/GX1_DATA/reports/sequence_structure_builder_smoke_20260628_v1
smoke_output=/home/andre2/GX1_DATA/reports/sequence_structure_builder_smoke_20260628_v1/v10_seq_structure_inline_smoke__HOLD_03B.parquet
smoke_rows=141
seq_shape=(96,89)
snap_shape=(89,)
ctx_cont_shape=(142,)
ctx_cat_shape=(5,)
extension_mode=inline_from_merged3
```

Neutral XGB bridge smoke:

```text
decision=XGBoost remains contract-compatible only, not predictive input
new_flag=--neutral-xgb-bridge
smoke_output=/home/andre2/GX1_DATA/reports/sequence_structure_builder_smoke_20260628_v1/v10_seq_structure_inline_neutral_bridge_smoke__HOLD_03B.parquet
smoke_rows=141
seq_shape=(96,89)
snap_shape=(89,)
ctx_cont_shape=(142,)
ctx_cat_shape=(5,)
neutral_xgb_bridge=true
xgb_bridge_source=neutral_uniform_proba
xgb_model_sha256=null
seq_structure_extension_dim=48
extension_mode=inline_from_merged3
compile_check=PASS
entry_next_edge_control_verify=PASS checks_passed=479
```

Trainer dynamic-input patch:

```text
trainer=gx1/models/entry_v10/entry_v10_ctx_train_v3.py
bundle_loader=gx1/models/entry_v10/entry_v10_bundle.py
status=patched
reason=old trainer hard-coded seq/snap dim 41; structure dataset emits seq/snap dim 89
sanity_manifest=/home/andre2/GX1_DATA/reports/sequence_structure_builder_smoke_20260628_v1/v10_seq_structure_inline_neutral_bridge_smoke__HOLD_03B.manifest.json
sanity_bundle=/home/andre2/GX1_DATA/reports/sequence_structure_builder_smoke_20260628_v1/v10_seq_structure_trainer_sanity_bundle
sanity_forward=PASS seq=(4,96,89) snap=(4,89) ctx_cont=(4,142) ctx_cat=(4,5)
strict_runtime_load=PASS seq_input_dim=89 snap_input_dim=89
dataset_getitem=PASS rows=141 signal_names=89 neutral_xgb_bridge=true
entry_next_edge_control_verify=PASS checks_passed=479
```

Full sequential dataset build:

```text
run_dir=/home/andre2/GX1_DATA/runs/FASE2B_REGIME_V4_20260605/v10_6yr_rebuild_20260626_seq_structure_neutral
dataset_train=/home/andre2/GX1_DATA/runs/FASE2B_REGIME_V4_20260605/v10_6yr_rebuild_20260626_seq_structure_neutral/v10_dataset_seq_structure_neutral__HOLD_03B_train.parquet
dataset_val=/home/andre2/GX1_DATA/runs/FASE2B_REGIME_V4_20260605/v10_6yr_rebuild_20260626_seq_structure_neutral/v10_dataset_seq_structure_neutral__HOLD_03B_val.parquet
dataset_test_2026=/home/andre2/GX1_DATA/runs/FASE2B_REGIME_V4_20260605/v10_6yr_rebuild_20260626_seq_structure_neutral/v10_dataset_seq_structure_neutral__HOLD_03B_test.parquet
train_window=2020-11-09T07:55:00Z..2025-09-30T20:52:00Z rows=350219 y_rates_long_short_flat=0.192911/0.175733/0.631356
val_window=2025-10-01T07:56:00Z..2025-12-31T19:55:00Z rows=17646 y_rates_long_short_flat=0.330046/0.251728/0.418225
test_window=2026-01-02T06:55:00Z..2026-06-26T01:25:00Z rows=34030 y_rates_long_short_flat=0.324949/0.320071/0.354981
seq_shape=(96,89)
snap_shape=(89,)
ctx_cont_shape=(142,)
ctx_cat_shape=(5,)
neutral_xgb_bridge=true
xgb_bridge_source=neutral_uniform_proba
xgb_model_sha256=null
seq_structure_extension_dim=48
extension_mode=inline_from_merged3
finite_seq_snap=true
```

MTF cache freshness repair:

```text
issue=initial 2026 test build failed because default MTF cache ended 2026-06-08 while data ended 2026-06-26
decision=do not bypass stale cache; regenerate cache for full 2026 cutoff
cache=/home/andre2/GX1_DATA/runs/FASE2B_REGIME_V4_20260605/v10_6yr_rebuild_20260626_seq_structure_neutral/MULTI_TF_V2_CACHE
source=/home/andre2/GX1_DATA/runs/FASE2B_REGIME_V4_20260605/v10_6yr_rebuild_20260626_spreadfix/FULL_PLUS_CTX_v3src.parquet
source_rows=402252
source_range=2020-11-09T00:00:00Z..2026-06-26T03:25:00Z
cache_tfs=M5/M15/H1/H4/D1
feature_count_per_tf=25
freshness_check=PASS cache_cutoff=2026-06-26T03:25:00Z
```

Trainer full-contract smoke:

```text
bundle=/home/andre2/GX1_DATA/runs/FASE2B_REGIME_V4_20260605/v10_6yr_rebuild_20260626_seq_structure_neutral/transformer_seq_structure_neutral_smoke
command_mode=1_epoch_subsample_smoke_not_performance_candidate
seq_input_dim=89
snap_input_dim=89
ordered_signal_names=89
neutral_xgb_bridge=true
xgb_bridge_source=neutral_uniform_proba
multi_tf=enabled v2_mode=true tf_dim=25 tf_len=96
strict_runtime_load=PASS
feature_liveness=PASS full_advanced_arrays rows=350219 skip_neutral_bridge_slots=7
best_val_loss=6.372947993522246
best_dir_acc=0.4182250935056103
entry_next_edge_control_verify=PASS checks_passed=479
```

Trainer bug fixed during smoke:

```text
bug=EntryV10CtxDataset advanced __getitem__ ignored self.indices, so --subsample-rows and smoke-date did not select requested rows
impact=post-export liveness could false-fail sparse D1 regime features by auditing the first rows instead of sampled rows
fix=advanced __getitem__ now resolves t=int(self.indices[i]) before slicing df/_np_seq/_np_snap/_np_ctx_cont/_np_ctx_cat
extra_fix=post-export liveness now audits full advanced ctx/snap arrays and full MTF cache arrays instead of a single random batch
sparse_feature_checked=d1_regime_changed_flag_v3 train_nonzero=85/350219 val_nonzero=2/17646 test_nonzero=4/34030
```

Exit alignment audit:

```text
script=gx1/scripts/analyze_exit_feature_alignment_v1.py
report=/home/andre2/GX1_DATA/reports/exit_feature_alignment_20260628_v1
exit_iql_summary=/home/andre2/GX1_DATA/runs/FASE2B_REGIME_V4_20260605/exit_iql_train_clean/summary_v1.json
exit_status=EXIT_IQL_V3_M1_PASS_BEATS_BASELINE
exit_production_allowed=false
exit_features=209
sequence_layer_features=48
matched_in_exit=10
missing_from_exit=38
```

Interpretation:

```text
Entry now has a concrete temporal structure layer candidate. Exit does not yet
share most of it. If entry learns to avoid tail structures but exit cannot see
the same structures while managing open risk, drawdown can remain bad through
giveback/late-exit behavior. Next work must integrate the same structure layer
into both entry sequence training and exit state/temporal training.
```

Next no-shadow execution plan:

```text
1. Train/evaluate sequential candidates only:
   - baseline latest Transformer V10, seq_dim=41
   - Transformer V10 + structure extension, seq_dim=89
   - Transformer V10 + structure extension + stronger tail/path auxiliary loss
   - repaired Entry-IQL with same structure state summaries
   - optional non-XGB sequential baselines: TCN/GRU/TFT-style temporal encoder

2. Replay/backtest only on 2026:
   - no shadow/live
   - cost20 and cost30
   - session slices: ALL, NO_EU, EU_ONLY, US_ONLY, OVERLAP_ONLY, ASIA_ONLY
   - required metrics: net, trades, win_rate, max_drawdown, worst_loss, tail counts,
     monthly stability, side/session breakdown

3. Upgrade exit in parallel:
   - add same 48 structure features to Exit-IQL state or exit temporal encoder
   - test giveback-loss and late-exit tail reduction
   - retest entry+exit together in 2026 replay

4. Promotion remains blocked until 2026 replay proves lower drawdown/tails at
   cost30 without destroying net/trade count. Shadow remains explicitly deferred.
```

Next full offline build command:

```bash
GX1_DATA=/home/andre2/GX1_DATA \
GX1_ALLOW_LEGACY_ENTRY_V10_RESEARCH=20260627_ALLOW_LEGACY_ENTRY_V10_RESEARCH \
python3 -m gx1.scripts.build_entry_v10_ctx_training_dataset_v3 \
  --source-parquet-override /home/andre2/GX1_DATA/runs/FASE2B_REGIME_V4_20260605/v10_6yr_rebuild_20260626_spreadfix/FULL_PLUS_CTX_v3src.parquet \
  --xgb-feature-contract-path gx1/xgb/contracts/xgb_input_features_base80_v1.json \
  --xgb-sanitizer-config-path gx1/xgb/contracts/xgb_input_sanitizer_base80_v1.json \
  --xgb_bundle /home/andre2/GX1_DATA/runs/FASE2B_REGIME_V4_20260605/v10_6yr_rebuild_20260626_spreadfix/xgb_v7_fixed_h24_15bps_cwp030 \
  --canonical_v2_parquet /home/andre2/GX1_DATA/runs/FASE2B_REGIME_V4_20260605/v10_6yr_rebuild_20260626_spreadfix/FULL_PLUS_CTX_v3src.parquet \
  --output /home/andre2/GX1_DATA/runs/FASE2B_REGIME_V4_20260605/v10_6yr_rebuild_20260626_seq_structure_neutral/v10_dataset_seq_structure_neutral.parquet \
  --time_split \
  --train_start 2020-11-09T00:00:00Z --train_end 2025-09-30T23:59:59Z \
  --val_start 2025-10-01T00:00:00Z --val_end 2025-12-31T23:59:59Z \
  --test_start 2026-01-01T00:00:00Z --test_end 2026-06-26T03:25:00Z \
  --hold-bars 3 --seq_len 96 \
  --seq-structure-manifest /home/andre2/GX1_DATA/reports/sequence_structure_feature_layer_20260628_v1/sequence_structure_feature_layer_manifest.json \
  --seq-structure-compute-inline \
  --neutral-xgb-bridge
```

## Fixed References

Workspace:

```text
/home/andre2/GX1_DATA/runs/FASE2B_REGIME_V4_20260605/v10_6yr_rebuild_20260626_spreadfix
```

Current baseline bundle:

```text
/home/andre2/GX1_DATA/runs/FASE2B_REGIME_V4_20260605/v10_6yr_rebuild_20260626_spreadfix/v10_bundle_6yr_symneg_smartctx_xgbfixed_h24_cwp05
```

Current best candidate bundle:

```text
/home/andre2/GX1_DATA/runs/FASE2B_REGIME_V4_20260605/v10_6yr_rebuild_20260626_spreadfix/v10_bundle_6yr_symneg_smartctx_xgbfixed_h24_cwp030_res025_wd1e4
```

Current best candidate dataset:

```text
/home/andre2/GX1_DATA/runs/FASE2B_REGIME_V4_20260605/v10_6yr_rebuild_20260626_spreadfix/v10_dataset_6yr_smartctx_xgbfixed_h24_cwp030
```

Current best XGB bridge:

```text
/home/andre2/GX1_DATA/runs/FASE2B_REGIME_V4_20260605/v10_6yr_rebuild_20260626_spreadfix/xgb_v7_fixed_h24_15bps_cwp030
```

## Known Results

Baseline ET with XGB `cwp05`:

```text
test_acc=0.4855822143957737
```

Standard ET with XGB `cwp030`:

```text
best_epoch=4
test_acc=0.4686330618534008
verdict=reject; overfit / anchor override
```

Best candidate ET with XGB `cwp030`, residual `0.25`, weight decay `1e-4`:

```text
best_epoch=5
best_dir_acc=0.4654205607476635
test_acc=0.49262601804974687
verdict=candidate only; requires selective EV/PnL proof
```

XGB-only selective edge smoke, `test` split:

```text
artifact=/tmp/entry_selective_smoke/selective_edge_metrics.csv
model=xgb_signal7
full_argmax_accuracy=0.425709883337002
top1_all_mean_pnl_bps=3.702250
top2_all_mean_pnl_bps=3.105486
top5_all_mean_pnl_bps=-10.749012
top10_all_mean_pnl_bps=-10.886699
top20_all_mean_pnl_bps=-6.088080
verdict=not enough; only very tight top 1-2% is positive in smoke
```

This smoke is a plumbing check and an early warning, not a final XGB verdict.
It proves the evaluator can connect split rows to spread-aware fixed-horizon PnL,
and it raises the bar for any ET+XGB run: ET must improve the top 5-10% edge,
not merely copy the XGB confidence shape.

Full selective edge eval, `val,test` splits:

```text
artifact=/home/andre2/GX1_DATA/reports/entry_selective_edge_20260627_cwp030_res025_wd1e4/selective_edge_metrics.csv
summary=/home/andre2/GX1_DATA/reports/entry_selective_edge_20260627_cwp030_res025_wd1e4/summary.json
```

Top-score `ALL` mean spread-aware PnL bps:

| split | model | top1 | top2 | top5 | top10 | top20 |
|---|---:|---:|---:|---:|---:|---:|
| val | xgb_signal7 | 17.80 | 14.13 | -3.82 | -7.52 | -8.00 |
| val | cwp05 | 39.98 | 38.72 | 31.93 | 28.59 | 22.56 |
| val | cwp030_res025_wd1e4 | 57.45 | 48.83 | 42.35 | 31.68 | 26.65 |
| test | xgb_signal7 | 3.70 | 3.11 | -10.75 | -10.89 | -6.09 |
| test | cwp05 | 98.86 | 78.61 | 55.61 | 46.11 | 33.98 |
| test | cwp030_res025_wd1e4 | 60.58 | 64.18 | 52.08 | 42.95 | 33.78 |

Side/session notes:

- ET bundles are positive on both LONG and SHORT at top 5-10%.
- XGB-only SHORT is strongly negative at top 5-10%.
- OVERLAP is the most stable strong session across `val` and `test`.
- `cwp030_res025_wd1e4` is strongest on `val`, but `cwp05` is strongest on
  `test`; no bundle is promoted from this result alone.

Current Phase 1 verdict:

```text
selective_edge_exists=yes for ET bundles
xgb_only_edge=weak/narrow; top 5-10% fails
promotion_verdict=no promotion; require ablations, prior baselines, and cost stress
```

Prior baseline and cost-stress eval:

```text
artifact=/home/andre2/GX1_DATA/reports/entry_selective_edge_20260627_priors_coststress/selective_edge_metrics.csv
summary=/home/andre2/GX1_DATA/reports/entry_selective_edge_20260627_priors_coststress/summary.json
extra_cost_bps=0,2,5,10
prior_source_split=train
```

Train-derived prior state:

| prior item | value |
|---|---:|
| train rows | 391129 |
| global long mean PnL | -1.15 bps |
| global short mean PnL | -2.31 bps |
| global side | LONG |
| ASIA selected | LONG, -1.10 bps |
| EU selected | LONG, -1.60 bps |
| OVERLAP selected | SHORT, -1.12 bps |
| US selected | LONG, +0.01 bps |

Top-score `ALL` mean PnL bps at the actual gate levels:

| split | model | top5 cost0 | top5 cost10 | top10 cost0 | top10 cost10 |
|---|---:|---:|---:|---:|---:|
| val | xgb_signal7 | -3.82 | -13.82 | -7.52 | -17.52 |
| val | prior best | 3.27 | -6.73 | 8.89 | -1.11 |
| val | cwp05 | 31.93 | 21.93 | 28.59 | 18.59 |
| val | cwp030_res025_wd1e4 | 42.35 | 32.35 | 31.68 | 21.68 |
| test | xgb_signal7 | -10.75 | -20.75 | -10.89 | -20.89 |
| test | prior best | 14.84 | 4.84 | 3.66 | -6.34 |
| test | cwp05 | 55.61 | 45.61 | 46.11 | 36.11 |
| test | cwp030_res025_wd1e4 | 52.08 | 42.08 | 42.95 | 32.95 |

Prior/cost-stress verdict:

```text
ET_edge_beats_priors=yes
ET_edge_survives_10bps_extra_cost=yes
xgb_only_fails_coststress=yes
best_val=cwp030_res025_wd1e4
best_test=cwp05
promotion_verdict=no promotion; model selection is not stable enough
then_required_gate=no-XGB ablation and non-XGB tabular baseline
```

No-XGB inference ablation:

```text
artifact=/home/andre2/GX1_DATA/reports/entry_selective_edge_20260627_no_xgb_ablation/selective_edge_metrics.csv
summary=/home/andre2/GX1_DATA/reports/entry_selective_edge_20260627_no_xgb_ablation/summary.json
method=neutralize signal bridge fields at inference only
neutralized_fields=p_long,p_short,p_flat,p_hat,uncertainty_score,margin_top1_top2,entropy
neutral_values=1/3,1/3,1/3,1/3,2/3,0,ln(3)
extra_cost_bps=0,10
```

Top-score `ALL` mean PnL bps:

| split | model | top5 cost0 | top5 cost10 | top10 cost0 | top10 cost10 |
|---|---:|---:|---:|---:|---:|
| val | cwp05 | 31.93 | 21.93 | 28.59 | 18.59 |
| val | cwp05_no_xgb | 26.74 | 16.74 | 24.45 | 14.45 |
| val | cwp030_res025_wd1e4 | 42.35 | 32.35 | 31.68 | 21.68 |
| val | cwp030_res025_wd1e4_no_xgb | 42.00 | 32.00 | 34.51 | 24.51 |
| test | cwp05 | 55.61 | 45.61 | 46.11 | 36.11 |
| test | cwp05_no_xgb | 50.76 | 40.76 | 41.38 | 31.38 |
| test | cwp030_res025_wd1e4 | 52.08 | 42.08 | 42.95 | 32.95 |
| test | cwp030_res025_wd1e4_no_xgb | 42.44 | 32.44 | 34.71 | 24.71 |

No-XGB ablation verdict:

```text
edge_without_xgb_signal=yes
xgb_is_not_the_whole_edge=yes
xgb_still_helps_on_test=yes
remove_xgb_now=no
then_required_gate=train/evaluate a true non-XGB tabular baseline, then decide
```

True no-XGB tabular LightGBM baseline:

```text
artifact=/home/andre2/GX1_DATA/reports/entry_selective_edge_20260627_tabular_no_xgb_lgbm/selective_edge_metrics.csv
summary=/home/andre2/GX1_DATA/reports/entry_selective_edge_20260627_tabular_no_xgb_lgbm/summary.json
model=/home/andre2/GX1_DATA/reports/entry_selective_edge_20260627_tabular_no_xgb_lgbm/lightgbm_no_xgb_model.joblib
feature_importance=/home/andre2/GX1_DATA/reports/entry_selective_edge_20260627_tabular_no_xgb_lgbm/feature_importance.csv
train_rows=391129
val_rows=3745
best_iteration=39
features=snap[7:]+ctx_cont+ctx_cat
excluded_xgb_fields=p_long,p_short,p_flat,p_hat,uncertainty_score,margin_top1_top2,entropy
```

Top-score `ALL` mean PnL bps:

| split | model | top5 cost0 | top5 cost10 | top10 cost0 | top10 cost10 |
|---|---:|---:|---:|---:|---:|
| val | cwp05 | 31.93 | 21.93 | 28.59 | 18.59 |
| val | cwp030_res025_wd1e4 | 42.35 | 32.35 | 31.68 | 21.68 |
| val | lightgbm_tabular_no_xgb | 54.74 | 44.74 | 42.93 | 32.93 |
| test | cwp05 | 55.61 | 45.61 | 46.11 | 36.11 |
| test | cwp030_res025_wd1e4 | 52.08 | 42.08 | 42.95 | 32.95 |
| test | lightgbm_tabular_no_xgb | 67.40 | 57.40 | 50.66 | 40.66 |

Top feature families:

- D1 range/location: `d1_range_z_20_canon_v2`,
  `d1_close_pct_in_20day_range_canon_v2`, `d1_pct_change_5_canon_v2`
- MTF structure: `struct_pullback_depth_h4_v3`,
  `struct_pullback_depth_h1_v3`
- MTF trend/volatility: `_v1h1_ema_diff`, `_v1h4_ema_diff`, D1/H1/H4 ATR
- Time/session: `hour_sin`, `hour_cos`, `session_id`

Shuffle-label negative control:

```text
artifact=/home/andre2/GX1_DATA/reports/entry_selective_edge_20260627_tabular_no_xgb_lgbm_shuffle_control/selective_edge_metrics.csv
summary=/home/andre2/GX1_DATA/reports/entry_selective_edge_20260627_tabular_no_xgb_lgbm_shuffle_control/summary.json
train_rows=391129
labels=randomly permuted before LightGBM fit
```

Shuffle-control top-score `ALL` mean PnL bps:

| split | model | top5 cost0 | top5 cost10 | top10 cost0 | top10 cost10 |
|---|---:|---:|---:|---:|---:|
| val | shuffled-label control | -1.16 | -11.16 | 0.59 | -9.41 |
| test | shuffled-label control | 8.20 | -1.80 | 3.72 | -6.28 |

Tabular no-XGB verdict:

```text
tabular_no_xgb_beats_et_plus_xgb=yes
tabular_no_xgb_survives_10bps_extra_cost=yes
shuffle_control_pass=yes; control collapses near/under zero under cost stress
xgb_required_for_future_architecture=no
then_promotion_verdict=no live promotion yet; required walk-forward/LOSO and replay gate
then_required_gate=walk-forward or LOSO tabular no-XGB validation, then replay/policy integration
```

Walk-forward no-XGB tabular LightGBM validation:

```text
artifact=/home/andre2/GX1_DATA/reports/entry_selective_edge_20260627_tabular_no_xgb_walkforward/walkforward_selective_edge_metrics.csv
summary=/home/andre2/GX1_DATA/reports/entry_selective_edge_20260627_tabular_no_xgb_walkforward/summary.json
feature_importance=/home/andre2/GX1_DATA/reports/entry_selective_edge_20260627_tabular_no_xgb_walkforward/walkforward_feature_importance_mean.csv
models_dir=/home/andre2/GX1_DATA/reports/entry_selective_edge_20260627_tabular_no_xgb_walkforward/models
folds=2023H1,2023H2,2024H1,2024H2,2025H1,2025H2,2026YTD,2026HOLDOUT
training_rule=train only on rows before the evaluated fold
validation_rule=pre-fold validation tail, default 30 days
features=snap[7:]+ctx_cont+ctx_cat
excluded_xgb_fields=p_long,p_short,p_flat,p_hat,uncertainty_score,margin_top1_top2,entropy
```

Walk-forward top-score `ALL`, extra cost stress `+10 bps`:

| fold | top5 mean pnl bps | top5 win rate | top10 mean pnl bps | top10 win rate |
|---|---:|---:|---:|---:|
| 2023H1 | 39.58 | 0.846 | 27.20 | 0.761 |
| 2023H2 | 26.16 | 0.773 | 16.51 | 0.685 |
| 2024H1 | 35.75 | 0.822 | 23.09 | 0.734 |
| 2024H2 | 30.31 | 0.780 | 20.86 | 0.694 |
| 2025H1 | 29.21 | 0.780 | 22.71 | 0.737 |
| 2025H2 | 41.02 | 0.862 | 31.66 | 0.802 |
| 2026YTD | 56.63 | 0.798 | 39.98 | 0.714 |
| 2026HOLDOUT | 56.03 | 0.875 | 37.78 | 0.759 |

Walk-forward summary under `+10 bps`:

```text
real_top5_positive_folds=8/8
real_top5_min_mean_median_max_bps=26.16,39.34,37.66,56.63
real_top10_positive_folds=8/8
real_top10_min_mean_median_max_bps=16.51,27.48,25.15,39.98
shuffle_top5_positive_folds=0/8
shuffle_top5_min_mean_median_max_bps=-13.05,-10.76,-11.90,-5.60
shuffle_top10_positive_folds=0/8
shuffle_top10_min_mean_median_max_bps=-12.40,-10.57,-11.85,-4.37
```

Top walk-forward feature families:

- D1 range/location: `d1_range_z_20_canon_v2`,
  `d1_close_pct_in_20day_range_canon_v2`, `d1_pct_change_5_canon_v2`
- MTF trend/volatility: `_v1h4_ema_diff`, `_v1h1_ema_diff`,
  `_v1h4_rsi14_z`, `_v1h1_rsi14_z`, H1/H4 ATR, D1 ATR
- Structure and trend-age: `struct_pullback_depth_h4_v3`,
  `struct_pullback_depth_h1_v3`, H1/H4/M15/D1 trend age
- Raw snap context: `pos_vs_ema200`, `_v1_kurt_r`

Walk-forward verdict:

```text
walkforward_gate_pass=yes
tabular_no_xgb_is_current_lead_path=yes
xgb_required_for_future_architecture=no
et_training_status=paused; do not train more ET before replay/policy gate
promotion_verdict=no live promotion yet; replay/policy and risk gates still required
then_required_gate=offline replay with fixed thresholds, cooldown, sizing, risk limits, and slippage/cost stress
```

Offline replay/policy gate for tabular no-XGB:

```text
script=gx1/scripts/replay_entry_tabular_no_xgb_policy_v1.py
main_artifacts_dir=/home/andre2/GX1_DATA/reports/entry_selective_edge_20260627_tabular_no_xgb_policy_replay
slippage_artifacts_dir=/home/andre2/GX1_DATA/reports/entry_selective_edge_20260627_tabular_no_xgb_policy_replay_slip5
latest_2026_artifacts_dir=/home/andre2/GX1_DATA/reports/entry_selective_edge_20260627_tabular_no_xgb_policy_replay_2026_only_slip5
threshold_rule=per-fold threshold calibrated on pre-fold validation tail only
model_source=walk-forward models from entry_selective_edge_20260627_tabular_no_xgb_walkforward/models
policy=one-position-at-a-time, cooldown 6 bars, max 8 trades/day, daily loss limit 150 bps, fixed-horizon exit
features=snap[7:]+ctx_cont+ctx_cat
excluded_xgb_fields=p_long,p_short,p_flat,p_hat,uncertainty_score,margin_top1_top2,entropy
```

Replay aggregate metrics:

| policy | cost | slippage | trades | net sum bps | net mean bps | win rate | profit factor | max DD bps |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| top5 | 0 | 0 | 1608 | 53002.50 | 32.96 | 0.845 | 12.30 | 263.52 |
| top5 | 10 | 0 | 1605 | 36879.17 | 22.98 | 0.731 | 5.59 | 283.52 |
| top10 | 0 | 0 | 2557 | 66372.80 | 25.96 | 0.805 | 8.28 | 286.68 |
| top10 | 10 | 0 | 2557 | 40802.80 | 15.96 | 0.667 | 3.58 | 316.68 |
| top5 | 10 | 5 | 1605 | 28854.17 | 17.98 | 0.669 | 3.76 | 293.52 |
| top10 | 10 | 5 | 2550 | 27907.36 | 10.94 | 0.596 | 2.37 | 352.15 |

Replay stability:

```text
top5_cost10_slip0_positive_folds=8/8
top5_cost10_slip0_positive_months=42/42
top10_cost10_slip0_positive_folds=8/8
top10_cost10_slip0_positive_months=42/42
top5_cost10_slip5_positive_folds=8/8
top5_cost10_slip5_positive_months=42/42
top10_cost10_slip5_positive_folds=8/8
top10_cost10_slip5_positive_months=41/42
lead_policy=top5, not top10; top10 is less robust under slippage stress
```

2026-only replay, run after deciding not to start live shadow yet:

```text
artifact=/home/andre2/GX1_DATA/reports/entry_selective_edge_20260627_tabular_no_xgb_policy_replay_2026_only_slip5/summary.json
folds=2026YTD,2026HOLDOUT
period=2026-01-01_to_2026-06-13
slippage_bps=5
cost_stress_bps=0,10
threshold_top_fracs=0.05,0.10
status=PASS
shadow_status=deferred; do not start shadow before 2026 replay robustness review
```

2026-only aggregate metrics:

| policy | cost | slippage | trades | net sum bps | net mean bps | win rate | profit factor | max DD bps | positive months |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| top5 | 0 | 5 | 174 | 6574.69 | 37.79 | 0.770 | 6.48 | 273.52 | 6/6 |
| top5 | 10 | 5 | 174 | 4834.69 | 27.79 | 0.661 | 3.85 | 293.52 | 6/6 |
| top10 | 0 | 5 | 258 | 7768.56 | 30.11 | 0.729 | 4.98 | 301.68 | 6/6 |
| top10 | 10 | 5 | 258 | 5188.56 | 20.11 | 0.632 | 2.87 | 331.68 | 6/6 |

2026-only verdict:

```text
offline_2026_replay_gate_pass=yes
top5_cost10_slip5_positive_months=6/6
top10_cost10_slip5_positive_months=6/6
holdout_may_jun_positive=yes
promotion_verdict=no live promotion; offline result is positive but must pass stress/feature diagnostics first
then_required_gate=review 2026-only replay robustness, then run targeted stress/feature diagnostics; shadow deferred
```

Replay verdict:

```text
offline_replay_gate_pass=yes for top5 tabular no-XGB policy
lead_policy=top5 threshold calibrated from pre-fold validation tail
promotion_verdict=no live promotion yet; serve/parity/shadow gates still required
then_required_gate=package no-XGB candidate inference path, prove train/serve feature parity, then run shadow/paper gate
```

Candidate package and serve-parity gate:

```text
candidate_id=entry_tabular_no_xgb_top5_v1_20260627
package_dir=/home/andre2/GX1_DATA/reports/entry_tabular_no_xgb_candidates/entry_tabular_no_xgb_top5_v1_20260627
candidate_manifest=/home/andre2/GX1_DATA/reports/entry_tabular_no_xgb_candidates/entry_tabular_no_xgb_top5_v1_20260627/candidate_manifest.json
feature_manifest=/home/andre2/GX1_DATA/reports/entry_tabular_no_xgb_candidates/entry_tabular_no_xgb_top5_v1_20260627/feature_manifest.json
policy_config=/home/andre2/GX1_DATA/reports/entry_tabular_no_xgb_candidates/entry_tabular_no_xgb_top5_v1_20260627/policy_config.json
serve_parity_report=/home/andre2/GX1_DATA/reports/entry_tabular_no_xgb_candidates/entry_tabular_no_xgb_top5_v1_20260627/serve_parity/serve_parity_report.json
package_script=gx1/scripts/package_entry_tabular_no_xgb_candidate_v1.py
serve_parity_script=gx1/scripts/verify_entry_tabular_no_xgb_candidate_parity_v1.py
runtime_candidate_module=gx1/runtime/entry_tabular_no_xgb_candidate.py
package_status=NOT_PROMOTED_NOT_LIVE_READY
feature_contract_hash=1d11fce818060ad5aeaabc0c00b369d22d75bc741859bbdf4b0eb03c9743c573
n_features=181
no_xgb_feature_guard=PASS
artifact_hashes_checked=22
serve_parity_status=PASS
val_runtime_research_max_abs_diff=0.0
test_runtime_research_max_abs_diff=0.0
promotion_verdict=no live promotion; shadow/paper and risk gates still required
then_required_gate=shadow/paper gate from candidate manifest with no-XGB runtime module
```

Manifest-resolved shadow/paper gate:

```text
script=gx1/scripts/run_entry_tabular_no_xgb_shadow_paper_gate_v1.py
summary=/home/andre2/GX1_DATA/reports/entry_tabular_no_xgb_candidates/entry_tabular_no_xgb_top5_v1_20260627/shadow_paper_val_to_test_cost10_slip5/shadow_paper_summary.json
candidate_manifest=/home/andre2/GX1_DATA/reports/entry_tabular_no_xgb_candidates/entry_tabular_no_xgb_top5_v1_20260627/candidate_manifest.json
calibration_split=val
paper_split=test
threshold_top_frac=0.05
score_threshold=0.39048198845884335
cost_stress_bps=10
slippage_bps=5
paper_rows=4543
above_threshold_rows=269
paper_trades=23
net_sum_bps=823.01
net_mean_bps=35.78
win_rate=0.739
profit_factor=4.77
max_drawdown_bps=85.09
positive_months=2/2
shadow_paper_status=PASS
promotion_verdict=no live promotion; sample is short and needs review/live-shadow wiring
then_required_gate=manual review, explicit live-shadow wiring, operational risk guard, then paper/live shadow observation
```

Live-shadow wiring preflight:

```text
pipeline_wiring=gx1/execution/v12_pipeline.py
v10_shadow_input_export=gx1/execution/v12_v10_live.py
runtime_shadow_scorer=gx1/runtime/entry_tabular_no_xgb_candidate.py
preflight_script=gx1/scripts/verify_entry_tabular_no_xgb_live_shadow_wiring_v1.py
preflight_report=/home/andre2/GX1_DATA/reports/entry_tabular_no_xgb_candidates/entry_tabular_no_xgb_top5_v1_20260627/live_shadow_wiring_preflight/live_shadow_wiring_preflight.json
env_file=/home/andre2/GX1_DATA/reports/entry_tabular_no_xgb_candidates/entry_tabular_no_xgb_top5_v1_20260627/live_shadow_wiring_preflight/entry_tabular_no_xgb_live_shadow.env
canonical_control=scripts/entry_next_edge_control.sh
canonical_control_executable=true
shadow_launcher_direct_executable=false
shadow_launcher_direct_invocation_policy=bash_only_guarded_by_control_surface
plan_state_verifier=gx1/scripts/verify_entry_next_edge_plan_state_v1.py
plan_state_verification_report=/home/andre2/GX1_DATA/reports/entry_tabular_no_xgb_candidates/entry_tabular_no_xgb_top5_v1_20260627/plan_state_verification/entry_next_edge_plan_state_verification.json
onboarding_override_docs=3
claude_active_entry_next_edge_override=true
agents_active_entry_next_edge_override=true
system_map_active_entry_next_edge_override=true
guardrail_selftest=gx1/scripts/verify_entry_next_edge_guardrails_v1.py
guardrail_selftest_report=/home/andre2/GX1_DATA/reports/entry_tabular_no_xgb_candidates/entry_tabular_no_xgb_top5_v1_20260627/guardrail_verification/entry_next_edge_guardrail_verification.json
shadow_only_launcher=scripts/run_entry_tabular_no_xgb_shadow_only.sh
shadow_telemetry_verifier=gx1/scripts/verify_entry_tabular_no_xgb_shadow_telemetry_v1.py
shadow_review_template=docs/ENTRY_NEXT_EDGE_SHADOW_REVIEW_TEMPLATE_20260627.md
runner_preflight_dir=/home/andre2/GX1_DATA/reports/entry_tabular_no_xgb_candidates/entry_tabular_no_xgb_top5_v1_20260627/live_shadow_runner_preflight
shadow_only_run_context=/home/andre2/GX1_DATA/reports/entry_tabular_no_xgb_candidates/entry_tabular_no_xgb_top5_v1_20260627/live_shadow_runner_preflight/shadow_only_run_context.json
legacy_live_practice_guard=scripts/launch_live_practice.sh requires GX1_ALLOW_LEGACY_ENTRY_LIVE_PRACTICE=20260627_ALLOW_LEGACY_ENTRY_LIVE_PRACTICE
legacy_live_trading_guard=scripts/run_live_trial160.sh requires GX1_ALLOW_LEGACY_ENTRY_LIVE_TRADING=20260627_ALLOW_LEGACY_ENTRY_LIVE_TRADING
legacy_live_surface_guard=scripts/entry_next_edge_live_legacy_block.sh requires GX1_ALLOW_LEGACY_ENTRY_LIVE_PRACTICE=20260627_ALLOW_LEGACY_ENTRY_LIVE_PRACTICE by default; OANDA exec smoke requires GX1_ALLOW_LEGACY_OANDA_EXEC_SMOKE=20260627_ALLOW_LEGACY_OANDA_EXEC_SMOKE
legacy_disabled_oanda_practice_canary_guard=scripts/_legacy_disabled/run_canary_oanda_practice.sh requires GX1_ALLOW_LEGACY_OANDA_PRACTICE_CANARY=20260627_ALLOW_LEGACY_OANDA_PRACTICE_CANARY
legacy_stop_live_practice_shadow_guard=scripts/stop_live_practice.sh skips active no-XGB shadow runner unless GX1_ALLOW_STOP_ENTRY_NEXT_EDGE_SHADOW=20260627_ALLOW_STOP_ENTRY_NEXT_EDGE_SHADOW
legacy_generic_replay_guard=scripts/run_replay.sh requires GX1_ALLOW_LEGACY_ENTRY_V10_RESEARCH=20260627_ALLOW_LEGACY_ENTRY_V10_RESEARCH
legacy_nightly_learning_guard=scripts/gx1_nightly_learning.sh requires GX1_ALLOW_LEGACY_NIGHTLY_LEARNING=20260627_ALLOW_LEGACY_NIGHTLY_LEARNING
legacy_v12_counterfactual_daemon_guard=gx1/execution/v12_daily_counterfactual.sh requires GX1_ALLOW_LEGACY_V12_COUNTERFACTUAL_DAEMON=20260627_ALLOW_LEGACY_V12_COUNTERFACTUAL_DAEMON
legacy_v12_prebuilt_refresh_daemon_guard=gx1/execution/v12_prebuilt_refresh_daemon.sh requires GX1_ALLOW_LEGACY_V12_PREBUILT_REFRESH_DAEMON=20260627_ALLOW_LEGACY_V12_PREBUILT_REFRESH_DAEMON
v12_data_maintenance_direct_modules_guard=direct gx1.execution V12 data-maintenance entrypoints require GX1_ALLOW_V12_DATA_MAINTENANCE=20260627_ALLOW_V12_DATA_MAINTENANCE
oanda_order_placement_guard=gx1/execution/oanda_client.py create_market_order/close_trade require GX1_ALLOW_OANDA_ORDER_PLACEMENT=20260627_ALLOW_OANDA_ORDER_PLACEMENT
direct_runner_guard=gx1/execution/v12_paper_runner.py requires --dry-run --shadow-only plus no-XGB shadow env plus GX1_ENTRY_NEXT_EDGE_SHADOW_RUNNER_ACK=20260627_ENTRY_NEXT_EDGE_SHADOW_RUNNER, or GX1_ALLOW_LEGACY_ENTRY_PAPER_RUNNER=20260627_ALLOW_LEGACY_ENTRY_PAPER_RUNNER
legacy_entry_v10_research_guard=scripts/entry_next_edge_legacy_block.sh requires GX1_ALLOW_LEGACY_ENTRY_V10_RESEARCH=20260627_ALLOW_LEGACY_ENTRY_V10_RESEARCH
legacy_entry_v10_scripts_blocked=24
legacy_entry_python_runners_blocked=2
legacy_entry_policy_training_modules_blocked=15
legacy_live_surface_scripts_blocked=8
legacy_entry_direct_modules_blocked=17
v12_data_maintenance_direct_modules_blocked=4
legacy_oanda_demo_runner_module_absent=true
legacy_shell_surface_scan_open_scripts=0
legacy_python_runner_scan_open_scripts=0
legacy_live_surface_scan_open_scripts=0
legacy_gx1_runner_import_scan_open_scripts=0
shadow_start_dirty_worktree_policy=clean required unless --allow-dirty-worktree
shadow_start_dirty_worktree_override_ack=GX1_ALLOW_ENTRY_SHADOW_DIRTY_WORKTREE=20260627_ALLOW_ENTRY_SHADOW_DIRTY_WORKTREE
shadow_run_context_records_git_status=true
shadow_preview_allow_dirty_worktree=false
shadow_control_surface_passthrough_policy=canonical preview/start args only
shadow_control_preview_duration_seconds=60
shadow_control_start_duration_seconds=3600
shadow_control_poll_seconds=10
shadow_launcher_preview_duration_seconds=60
shadow_launcher_start_duration_seconds=3600
shadow_launcher_poll_seconds=10
shadow_launcher_duration_override_ack=GX1_ALLOW_ENTRY_SHADOW_DURATION_OVERRIDE=20260627_ALLOW_ENTRY_SHADOW_DURATION_OVERRIDE
shadow_launcher_poll_override_ack=GX1_ALLOW_ENTRY_SHADOW_POLL_OVERRIDE=20260627_ALLOW_ENTRY_SHADOW_POLL_OVERRIDE
shadow_journal_suffix=noxgb_shadow
shadow_journal_suffix_override_ack=GX1_ALLOW_ENTRY_SHADOW_JOURNAL_SUFFIX=20260627_ALLOW_ENTRY_SHADOW_JOURNAL_SUFFIX
shadow_launcher_canonical_env_file=/home/andre2/GX1_DATA/reports/entry_tabular_no_xgb_candidates/entry_tabular_no_xgb_top5_v1_20260627/live_shadow_wiring_preflight/entry_tabular_no_xgb_live_shadow.env
shadow_launcher_env_override_ack=GX1_ALLOW_ENTRY_SHADOW_ENV_OVERRIDE=20260627_ALLOW_ENTRY_SHADOW_ENV_OVERRIDE
shadow_runner_canonical_manifest=/home/andre2/GX1_DATA/reports/entry_tabular_no_xgb_candidates/entry_tabular_no_xgb_top5_v1_20260627/candidate_manifest.json
shadow_runner_canonical_threshold=0.39048198845884335
shadow_telemetry_all_rows_contract_check=true
shadow_telemetry_missing_fields_override_ack=GX1_ALLOW_ENTRY_SHADOW_MISSING_FIELDS=20260627_ALLOW_ENTRY_SHADOW_MISSING_FIELDS
shadow_telemetry_expected_candidate_id=entry_tabular_no_xgb_top5_v1_20260627
shadow_telemetry_expected_feature_hash=1d11fce818060ad5aeaabc0c00b369d22d75bc741859bbdf4b0eb03c9743c573
shadow_telemetry_expected_score_threshold=0.39048198845884335
shadow_telemetry_offline_expected_would_take_rate=0.05921197446621176
shadow_telemetry_contract_override_ack=GX1_ALLOW_ENTRY_SHADOW_CONTRACT_OVERRIDE=20260627_ALLOW_ENTRY_SHADOW_CONTRACT_OVERRIDE
primary_objective=selective_ev_pnl_not_full_bar_accuracy
post_shadow_accept_gate=telemetry_exists_no_order_side_effects_feature_hash_match_candidate_rate_explainable_manual_review
post_shadow_fail_action=stop_promotion_and_enter_feature_label_objective_redesign_not_accuracy_tuning
post_shadow_allowed_decisions=ACCEPT_FOR_NEXT_REVIEW_GATE,HOLD_FOR_MORE_SHADOW,FAIL_TO_FEATURE_LABEL_OBJECTIVE_REDESIGN
feature_backlog_quantified_patterns=hh_hl_lh_ll,bos_choch,liquidity_sweep,false_breakout,compression_expansion,wick_rejection,pivot_distance,impulse_pullback,vol_session_interaction
existing_model_dependencies_available=lightgbm,sklearn,xgboost_reference
missing_optional_model_dependencies=catboost,sktime,tsfresh
model_challenger_policy=no_new_dependency_before_shadow_review
latest_2026_policy_replay_dir=/home/andre2/GX1_DATA/reports/entry_selective_edge_20260627_tabular_no_xgb_policy_replay_2026_only_slip5
latest_2026_policy_replay_status=PASS
latest_2026_policy_replay_period=2026-01-01_to_2026-06-13
latest_2026_policy_replay_folds=2026YTD,2026HOLDOUT
latest_2026_policy_replay_slippage_bps=5.0
latest_2026_policy_replay_top05_cost10_trades=174
latest_2026_policy_replay_top05_cost10_net_sum_bps=4834.685411743672
latest_2026_policy_replay_top05_cost10_net_mean_bps=27.78554834335444
latest_2026_policy_replay_top05_cost10_positive_months=6/6
latest_2026_policy_replay_top10_cost10_trades=258
latest_2026_policy_replay_top10_cost10_net_sum_bps=5188.560298773748
latest_2026_policy_replay_top10_cost10_net_mean_bps=20.11069883245639
latest_2026_policy_replay_top10_cost10_positive_months=6/6
shadow_env_required=GX1_ENTRY_TABULAR_NO_XGB_SHADOW_ACK=20260627_ENTRY_NO_XGB_LIVE_SHADOW
shadow_runner_env_required=GX1_ENTRY_NEXT_EDGE_SHADOW_RUNNER_ACK=20260627_ENTRY_NEXT_EDGE_SHADOW_RUNNER
shadow_manifest_env=GX1_ENTRY_TABULAR_NO_XGB_SHADOW_MANIFEST
shadow_threshold_env=GX1_ENTRY_TABULAR_NO_XGB_SHADOW_THRESHOLD
shadow_score_threshold=0.39048198845884335
rows_checked=4543
live_research_max_abs_diff=0.0
would_take_rows=269
would_take_rate=0.0592
would_take_long_rows=57
would_take_short_rows=212
live_order_placement=NOT_STARTED_NOT_ENABLED
live_shadow_wiring_preflight_status=PASS
plan_state_verification_status=PASS
plan_state_verification_checks=479
guardrail_selftest_status=PASS
guardrail_selftest_cases=53
guardrail_selftest_runner_started=false
guardrail_selftest_shadow_run_context_status=PASS
guardrail_selftest_shadow_run_context_allow_dirty_worktree=false
guardrail_selftest_shadow_run_context_records_git_status=true
latest_control_preview_status=PASS
latest_control_preview_mode=preview
latest_control_preview_runner_started=false
latest_default_start_shadow_status=BLOCKED_BY_DIRTY_WORKTREE
latest_verify_shadow_status=BLOCKED_MISSING_JOURNAL
latest_shadow_attempt_status=ABORTED_BY_USER_BEFORE_JOURNAL_ROWS
latest_shadow_observation_started=false
latest_shadow_journal_rows=0
current_dirty_worktree_requires_ack=true
handover_default_status=REDIRECTS_TO_ENTRY_NEXT_EDGE_PLAN
shadow_only_launcher_preview_status=PASS
shadow_launcher_noncanonical_env_without_ack_status=BLOCKED_BY_ENTRY_NEXT_EDGE_PLAN
shadow_launcher_noncanonical_journal_suffix_without_ack_status=BLOCKED_BY_ENTRY_NEXT_EDGE_PLAN
shadow_launcher_dirty_worktree_override_without_ack_status=BLOCKED_BY_ENTRY_NEXT_EDGE_PLAN
shadow_launcher_noncanonical_duration_without_ack_status=BLOCKED_BY_ENTRY_NEXT_EDGE_PLAN
shadow_launcher_noncanonical_poll_without_ack_status=BLOCKED_BY_ENTRY_NEXT_EDGE_PLAN
shadow_launcher_direct_exec_permission_status=BLOCKED_BY_FILE_MODE
shadow_control_preview_arg_override_status=BLOCKED_BY_ENTRY_NEXT_EDGE_PLAN
shadow_control_start_arg_override_status=BLOCKED_BY_ENTRY_NEXT_EDGE_PLAN
shadow_control_dirty_worktree_override_without_ack_status=BLOCKED_BY_ENTRY_NEXT_EDGE_PLAN
legacy_live_practice_default_status=BLOCKED_BY_ENTRY_NEXT_EDGE_PLAN
legacy_live_trial160_default_status=BLOCKED_BY_ENTRY_NEXT_EDGE_PLAN
legacy_stop_live_practice_shadow_guard_status=SKIPS_ENTRY_NEXT_EDGE_SHADOW_BY_DEFAULT
legacy_live_demo_tombstone_status=BLOCKED_BY_ENTRY_NEXT_EDGE_PLAN
legacy_dipfix_asia_experiment_status=BLOCKED_BY_ENTRY_NEXT_EDGE_PLAN
legacy_volbal_baseline_oneshot_status=BLOCKED_BY_ENTRY_NEXT_EDGE_PLAN
legacy_oanda_exec_smoke_status=BLOCKED_BY_ENTRY_NEXT_EDGE_PLAN
legacy_disabled_oanda_practice_canary_status=BLOCKED_BY_ENTRY_NEXT_EDGE_PLAN
direct_v12_paper_runner_default_status=BLOCKED_BY_ENTRY_NEXT_EDGE_PLAN
direct_v12_paper_runner_shadow_without_launcher_ack_status=BLOCKED_BY_ENTRY_NEXT_EDGE_PLAN
direct_v12_paper_runner_shadow_wrong_manifest_status=BLOCKED_BY_ENTRY_NEXT_EDGE_PLAN
direct_v12_paper_runner_shadow_wrong_threshold_status=BLOCKED_BY_ENTRY_NEXT_EDGE_PLAN
legacy_entry_v10_training_pipeline_status=BLOCKED_BY_ENTRY_NEXT_EDGE_PLAN
legacy_entry_v10_replay_status=BLOCKED_BY_ENTRY_NEXT_EDGE_PLAN
legacy_entry_v10_worker_smoke_status=BLOCKED_BY_ENTRY_NEXT_EDGE_PLAN
legacy_entry_replay_perf_runner_status=BLOCKED_BY_ENTRY_NEXT_EDGE_PLAN
legacy_generic_replay_launcher_status=BLOCKED_BY_ENTRY_NEXT_EDGE_PLAN
legacy_nightly_learning_status=BLOCKED_BY_ENTRY_NEXT_EDGE_PLAN
legacy_v12_counterfactual_daemon_status=BLOCKED_BY_ENTRY_NEXT_EDGE_PLAN
legacy_v12_prebuilt_refresh_daemon_status=BLOCKED_BY_ENTRY_NEXT_EDGE_PLAN
direct_v12_backfill_to_present_status=BLOCKED_BY_ENTRY_NEXT_EDGE_PLAN
direct_v12_m1_to_m5_downsample_status=BLOCKED_BY_ENTRY_NEXT_EDGE_PLAN
direct_v12_oanda_data_collector_status=BLOCKED_BY_ENTRY_NEXT_EDGE_PLAN
direct_v12_canonical_incremental_status=BLOCKED_BY_ENTRY_NEXT_EDGE_PLAN
oanda_create_market_order_status=BLOCKED_BY_ENTRY_NEXT_EDGE_PLAN
oanda_close_trade_status=BLOCKED_BY_ENTRY_NEXT_EDGE_PLAN
direct_legacy_v10_ctx_dataset_module_status=BLOCKED_BY_ENTRY_NEXT_EDGE_PLAN
direct_legacy_v10_sweep_module_status=BLOCKED_BY_ENTRY_NEXT_EDGE_PLAN
direct_legacy_xgb_prebuilt_module_status=BLOCKED_BY_ENTRY_NEXT_EDGE_PLAN
direct_legacy_xgb_flow_ablation_module_status=BLOCKED_BY_ENTRY_NEXT_EDGE_PLAN
legacy_oanda_demo_runner_import_status=MODULE_ABSENT
direct_legacy_sniper_quarter_replay_status=BLOCKED_BY_ENTRY_NEXT_EDGE_PLAN
direct_legacy_entry_r3_retrain_status=BLOCKED_BY_ENTRY_NEXT_EDGE_PLAN
direct_legacy_entry_iql_v2_builder_status=BLOCKED_BY_ENTRY_NEXT_EDGE_PLAN
verify_shadow_feature_hash_mismatch_status=BLOCKED_BY_ENTRY_NEXT_EDGE_PLAN
direct_verify_shadow_contract_override_without_ack_status=BLOCKED_BY_ENTRY_NEXT_EDGE_PLAN
verify_shadow_missing_fields_default_status=BLOCKED_BY_ENTRY_NEXT_EDGE_PLAN
control_verify_shadow_allow_missing_without_ack_status=BLOCKED_BY_ENTRY_NEXT_EDGE_PLAN
direct_verify_shadow_allow_missing_without_ack_status=BLOCKED_BY_ENTRY_NEXT_EDGE_PLAN
primary_verify_command=scripts/entry_next_edge_control.sh verify
primary_selftest_command=scripts/entry_next_edge_control.sh selftest
primary_preview_command=scripts/entry_next_edge_control.sh preview-shadow --duration-seconds 60 --poll-seconds 10
primary_2026_replay_command=.venv/bin/python -m gx1.scripts.replay_entry_tabular_no_xgb_policy_v1 --folds '2026YTD=2026-01-01:2026-05-01,2026HOLDOUT=2026-05-01:2026-06-13' --threshold-top-fracs 0.05,0.10 --cost-stress-bps 0,10 --slippage-bps 5
deferred_shadow_start_command=scripts/entry_next_edge_control.sh start-shadow
primary_verify_shadow_command=scripts/entry_next_edge_control.sh verify-shadow --journal /home/andre2/GX1_DATA/reports/v12_paper_runs/v12_paper_journal_<UTC_YYYYMMDD>_noxgb_shadow.jsonl
planned_2026_replay_review_1=review top5/top10 cost10+slip5 by fold/month/day/session
planned_2026_replay_review_2=run targeted stress/feature diagnostics before any shadow/live step
deferred_shadow_observation_1=scripts/entry_next_edge_control.sh start-shadow
deferred_shadow_observation_2=scripts/entry_next_edge_control.sh verify-shadow --journal /home/andre2/GX1_DATA/reports/v12_paper_runs/v12_paper_journal_<UTC_YYYYMMDD>_noxgb_shadow.jsonl --min-shadow-rows 20
planned_review_gate=manual review of 2026 offline replay robustness before any promotion, training restart, pin, shadow, or live order placement
shadow_only_runner_command=.venv/bin/python -m gx1.execution.v12_paper_runner --dry-run --shadow-only --journal-suffix noxgb_shadow --units 1 --max-trades 1 --max-spread-bps 9999 --poll-seconds 10
shadow_only_journal=/home/andre2/GX1_DATA/reports/v12_paper_runs/v12_paper_journal_<UTC_YYYYMMDD>_noxgb_shadow.jsonl
promotion_verdict=no live promotion; 2026 offline replay is positive but robustness review is not complete
next_required_gate=review 2026-only replay robustness, then run targeted stress/feature diagnostics; shadow deferred
```

## Non-Negotiable Direction

Do not optimize full-bar 3-class accuracy as the main objective anymore.
`0.49` accuracy is not a trading edge. It only proves the model learns some
structure above the 3-class majority baseline.

From now on, the main question is:

```text
Does a small selected subset of signals produce positive forward EV after spread?
```

## Phase 1: Selective Edge / PnL Eval

Build or run an evaluator that reports, at minimum:

- top 1%, 2%, 5%, 10%, and 20% by confidence/margin
- LONG and SHORT separately
- ASIA, EU, OVERLAP, US separately
- precision, win rate, forward PnL bps, MFE, MAE, path quality
- coverage rate
- comparison against baseline ET, best candidate ET, XGB-only, majority/session priors

Gate:

```text
If top 5-10% does not show clear positive EV after spread, stop model tuning.
Fix labels/features/objective first.
```

## Phase 2: XGB Ablation

Answer whether XGB is needed at all.

Required lanes:

- Lane A: XGB-only selective edge
- Lane B: ET with XGB anchor, current best candidate
- Lane C: ET without XGB, signal7 neutralized
- Lane D: LightGBM/CatBoost without XGB-derived probabilities

Gate:

```text
If Lane C or D >= Lane B on selective EV/PnL, remove XGB from the future live architecture.
If Lane A ~= Lane B, deprioritize ET.
If none has selective EV, redesign labels/features before more deep learning.
```

## Phase 3: Label Redesign

Test labels only if Phase 1 fails or shows weak/unstable edge.

Candidates:

- H12, H24, H48 side-by-side
- ATR/session dynamic triple-barrier
- tradeable-only directional labels
- separate direction and skip/quality models
- expected-value bps target instead of pure 3-class direction

Gate:

```text
New labels must improve top-k EV/PnL, not just accuracy.
```

## Phase 4: Pattern Feature Pack

Add only quantified features, not discretionary visual chart claims.

Candidate feature groups:

- HH/HL/LH/LL swing state
- break of structure / CHoCH
- liquidity sweep of prior high/low
- false breakout with close back inside range
- compression before expansion
- wick rejection at session high/low
- distance to D1/H4/M15 pivots
- impulse then first pullback
- volatility/session interaction

Gate:

```text
Feature pack must beat the current raw-feature baseline in a tabular model before
it is allowed into another GPU Transformer run.
```

## Phase 5: Model Order

Use this order:

1. Selective EV/PnL evaluator
2. LightGBM/CatBoost tabular baseline
3. no-XGB ET ablation
4. TCN/InceptionTime/MiniROCKET sequence baseline
5. ensemble only if simple models prove independent edge

Do not run RL/IQL for entry until selective EV exists.

## Next Run Queue After First Selective Eval

The first selective eval proves that ET confidence can select positive
spread-aware EV on the current `val,test` split. It does not prove that either
bundle is ready to promote.

Run in this order:

1. DONE: Extend selective eval with simple prior baselines and cost stress.
   - always-long, always-short, train-derived session side prior
   - extra cost stress: 2, 5, and 10 bps after spread-aware bid/ask PnL
   - result: pass; ET remains positive and beats priors at top 5-10%
2. DONE: Re-run full selective eval with these baselines for `cwp05`,
   `cwp030_res025_wd1e4`, and XGB-only.
3. DONE: Run no-XGB inference ablation before any new ET training is trusted.
   - ET with signal7 neutralized at inference
   - result: edge survives; XGB helps on test but is not the whole edge
4. DONE: Run tabular LightGBM without XGB-derived probabilities.
   - result: tabular no-XGB beats ET+XGB on top 5-10% in `val,test`
   - result: shuffle-label control collapses under cost stress
5. DONE: Remove XGB as a required future live architecture dependency.
   - keep XGB only as a historical/reference baseline until a later result
     proves independent incremental value
6. DONE: Run walk-forward tabular no-XGB validation.
   - include cost stress, priors, shuffle control, side/session slices
   - result: pass; real model positive in 8/8 folds at top 5-10% after
     `+10 bps`, shuffled-label control negative in 8/8 folds
7. DONE: Run offline replay/policy integration for tabular no-XGB.
   - use fixed thresholds selected without seeing the replay interval
   - add cooldown, one-position-at-a-time policy, sizing, stop/TP or exit rule,
     daily loss limit, max trades/day, and explicit slippage/cost stress
   - report equity curve, drawdown, trade count, expectancy, tail losses,
     session/side slices, and month-by-month stability
   - result: pass for top5; top5 stays positive in 8/8 folds and 42/42 months
     under `+10 bps` cost and `+5 bps` slippage
   - result: top10 is positive overall, but less robust under slippage stress
     because it has 1 negative month
8. DONE: Package no-XGB candidate inference path and prove serve parity.
   - no live pin until train/serve feature parity, shadow/paper, and risk gates pass
   - build a deterministic candidate manifest with model path, feature contract,
     threshold rule, policy parameters, and no-XGB field exclusions
   - hard-fail if any XGB-derived signal field is present in the live candidate
   - result: package written as `NOT_PROMOTED_NOT_LIVE_READY`
   - result: serve parity pass on `val,test` with runtime/research feature
     matrix max abs diff `0.0`
9. DONE: Run shadow/paper gate from the candidate manifest.
   - use only `gx1/runtime/entry_tabular_no_xgb_candidate.py` for inference
   - resolve candidate by explicit `candidate_manifest.json`, not latest glob
   - compare shadow decisions, candidate rate, side/session distribution,
     veto reasons, paper trades, PnL, drawdown, and operational rejects
   - result: pass on `test` with val-calibrated top5 threshold, `+10 bps`
     cost and `+5 bps` slippage
   - limitation: only 23 paper trades over 2 months, so this is not a live pin
10. DONE: Manual review and explicit live-shadow wiring preflight.
   - wire candidate by explicit manifest path only
   - add live-shadow-only mode that logs decisions without order placement
   - enforce no-XGB feature guard, feature hash check, artifact hash check,
     max trades/day, daily loss limit, and one-position-at-a-time policy
   - compare live-shadow telemetry against offline decision rates before any pin
   - result: pipeline wiring is default-off and env-gated with explicit ACK,
     manifest path, and score threshold
   - result: preflight pass; live-style feature vector matches research path
     with max abs diff `0.0`
   - status: live order placement not started/enabled
11. CURRENT: Review the 2026-only offline replay before any shadow attempt.
   - use the existing no-XGB walk-forward models and existing dataset/source tape
   - do not start shadow just to get comfort if the current question is offline
     robustness
   - use `gx1/scripts/replay_entry_tabular_no_xgb_policy_v1.py` for 2026 replay
   - latest 2026-only replay artifact:
     `/home/andre2/GX1_DATA/reports/entry_selective_edge_20260627_tabular_no_xgb_policy_replay_2026_only_slip5/summary.json`
   - status: top5 and top10 are positive across 2026YTD and 2026HOLDOUT under
     `cost10 + slippage5`
   - next diagnostics: review by fold/month/day/session, threshold sensitivity,
     adverse excursion, and feature/regime concentration
   - source only the generated env file if shadow is deliberately revisited later,
     no latest-glob manifest resolution
   - use `scripts/entry_next_edge_control.sh` as the canonical shadow command surface
   - run `scripts/entry_next_edge_control.sh selftest` after guard changes; it
     behavior-tests verify, preview, training block, legacy live block, direct
     runner block, direct shadow-without-launcher-ack block, and missing journal
     block, plus legacy Entry V10 training/replay blocks
   - blocked commands in this surface: `train`, `retrain`, `promote`, `pin`,
     `live`, `xgb-train`, `et-train`, and equivalent old Entry actions
   - legacy full-stack live launcher `scripts/launch_live_practice.sh` is blocked
     by default because it can start XGB -> V10 -> Entry-IQL with OANDA orders;
     override requires explicit `GX1_ALLOW_LEGACY_ENTRY_LIVE_PRACTICE` ack
   - legacy Entry V10/V10.1 research/replay shell scripts are blocked by
     `scripts/entry_next_edge_legacy_block.sh`; override requires explicit
     `GX1_ALLOW_LEGACY_ENTRY_V10_RESEARCH` ack
   - direct V12 data-maintenance entrypoints (`v12_backfill_to_present`,
     `v12_m1_to_m5_downsample`, `v12_oanda_data_collector`, and
     `v12_canonical_incremental`) are blocked by default; override requires
     explicit `GX1_ALLOW_V12_DATA_MAINTENANCE` ack because they mutate the tape
     or prebuilts used by shadow validation
   - low-level OANDA order mutation (`create_market_order` / `close_trade`) is
     blocked inside `gx1/execution/oanda_client.py`; override requires explicit
     `GX1_ALLOW_OANDA_ORDER_PLACEMENT` ack even if a higher-level legacy script
     was also acknowledged
   - direct `gx1.execution.v12_paper_runner` is also blocked by default; it only
     runs without legacy ack when invoked by the canonical shadow launcher with
     `--dry-run --shadow-only`, the generated no-XGB shadow env, and the runner
     launcher ack present
   - launcher first runs `gx1/scripts/verify_entry_next_edge_plan_state_v1.py`
     and fails if plan, manifest, env, serve parity, shadow/paper gate, or
     live-shadow preflight drift away from the no-XGB shadow-only path
   - launcher writes `shadow_only_run_context.json` for preview/start with git
     status, manifest path, threshold, journal suffix, expected journal path,
     and exact runner command
   - verify command:
     `scripts/entry_next_edge_control.sh verify`
   - guardrail selftest command:
     `scripts/entry_next_edge_control.sh selftest`
   - 2026 replay command:
     `.venv/bin/python -m gx1.scripts.replay_entry_tabular_no_xgb_policy_v1 --folds '2026YTD=2026-01-01:2026-05-01,2026HOLDOUT=2026-05-01:2026-06-13' --threshold-top-fracs 0.05,0.10 --cost-stress-bps 0,10 --slippage-bps 5`
   - shadow preview command, deferred:
     `scripts/entry_next_edge_control.sh preview-shadow --duration-seconds 60 --poll-seconds 10`
   - shadow start command, deferred until explicitly needed:
     `scripts/entry_next_edge_control.sh start-shadow`
   - if deliberately running from the current dirty worktree, require explicit
     `--allow-dirty-worktree` and record `git status --short` with the run
   - require journal fields `shadow_no_xgb_*` on candidate rows
   - verify the resulting journal:
     `scripts/entry_next_edge_control.sh verify-shadow --journal /home/andre2/GX1_DATA/reports/v12_paper_runs/v12_paper_journal_<UTC_YYYYMMDD>_noxgb_shadow.jsonl`
   - compare live-shadow candidate rate, side/session distribution, disabled
     reasons, and rejects against offline/test expectations
   - pass condition: shadow telemetry exists, no order side effects, no disabled
     scorer, feature hash matches package, and live candidate rate is explainable
     versus offline `would_take_rate=0.0592`
   - no production pin until 2026 replay robustness and any later shadow-only
     telemetry are reviewed
12. If all model lanes lose edge under replay/cost/risk stress, stop model
   training and move to label/objective redesign.
13. Only after reviewed 2026 replay robustness and any deliberately scheduled
   shadow/risk gate: decide between tabular no-XGB promotion path, ET
   distillation/ensemble, or label/objective redesign.

Archived reproduction command for the prior full selective eval.
This is not the next run:

```bash
.venv/bin/python -m gx1.scripts.evaluate_entry_selective_edge_v1 \
  --dataset-dir /home/andre2/GX1_DATA/runs/FASE2B_REGIME_V4_20260605/v10_6yr_rebuild_20260626_spreadfix/v10_dataset_6yr_smartctx_xgbfixed_h24_cwp030 \
  --source-parquet /home/andre2/GX1_DATA/runs/FASE2B_REGIME_V4_20260605/v10_6yr_rebuild_20260626_spreadfix/FULL_PLUS_CTX_v3src.parquet \
  --m5-prebuilt-path /home/andre2/GX1_DATA/runs/FASE2B_REGIME_V4_20260605/v10_6yr_rebuild_20260626_spreadfix/cv3/xauusd_m5_CANONICAL_V3_2020_2026.parquet \
  --out-dir /home/andre2/GX1_DATA/reports/entry_selective_edge_20260627_cwp030_res025_wd1e4 \
  --splits val,test \
  --include-xgb-only \
  --bundle cwp05=/home/andre2/GX1_DATA/runs/FASE2B_REGIME_V4_20260605/v10_6yr_rebuild_20260626_spreadfix/v10_bundle_6yr_symneg_smartctx_xgbfixed_h24_cwp05 \
  --bundle cwp030_res025_wd1e4=/home/andre2/GX1_DATA/runs/FASE2B_REGIME_V4_20260605/v10_6yr_rebuild_20260626_spreadfix/v10_bundle_6yr_symneg_smartctx_xgbfixed_h24_cwp030_res025_wd1e4 \
  --device cpu --batch-size 512 --num-workers 0
```

## Chart-Structure Ablation Status 2026-06-27

Offline only. No shadow/live run was started for this work.

New runner:

```text
script=gx1/scripts/experiment_entry_chart_structure_ablation_v1.py
feature_layer=544 total features: 181 base + 127 chart + 236 deep chart
interactions from existing no-XGB matrix
feature_groups=HH/HL/LH/LL proxies, BOS/CHoCH, sweep/false-breakout rejection,
compression-to-expansion, wick/level rejection, pivot proximity, impulse/pullback,
session x structure, session x volatility, structure x volatility
deep_feature_groups=EMA/position crosses and deltas, regime freshness/divergence,
compression release, vol stack, tail pressure, session x structure/context,
structure x context, breakout/breakdown tail risk
models_tested=LightGBM, HistGradientBoosting, ExtraTrees, XGBoost reference
catboost_status=not installed
transformer_tcn_iql_status=not retrained in this gate; too slow until tabular
structure edge is exhausted
```

Breadth artifact:

```text
/home/andre2/GX1_DATA/reports/entry_chart_structure_ablation_20260627_breadth_v2
```

Focused retest artifact:

```text
/home/andre2/GX1_DATA/reports/entry_chart_structure_ablation_20260627_focused_v1
```

Deep interaction and tail-risk artifacts:

```text
deep_finalist_replay=/home/andre2/GX1_DATA/reports/entry_chart_structure_deep_finalist_20260627_v1
tail_exit_audit=/home/andre2/GX1_DATA/reports/entry_feature_tail_exit_deep_audit_20260627_v1
fold_calibrated_veto_retest=/home/andre2/GX1_DATA/reports/entry_chart_structure_veto_retest_20260627_v1
veto_sensitivity=/home/andre2/GX1_DATA/reports/entry_chart_structure_veto_sensitivity_20260627_v1
coststress_initial=/home/andre2/GX1_DATA/reports/entry_chart_structure_best_candidate_coststress_20260628_v1
exit_retimer_default=/home/andre2/GX1_DATA/reports/entry_feature_exit_retimer_20260628_v1
exit_retimer_cost10_20_grid=/home/andre2/GX1_DATA/reports/entry_feature_exit_retimer_cost10_20_grid_20260628_v1
multimodel_veto_cost10_20=/home/andre2/GX1_DATA/reports/entry_chart_structure_multimodel_veto_cost10_20_20260628_v1
repro_probe=/home/andre2/GX1_DATA/reports/entry_chart_structure_repro_probe_20260628_v1
feature_interaction_tail=/home/andre2/GX1_DATA/reports/entry_feature_interaction_tail_20260628_v1
pairwise_veto_retest=/home/andre2/GX1_DATA/reports/entry_chart_structure_pairwise_veto_retest_20260628_v1
pairwise_exit_retimer=/home/andre2/GX1_DATA/reports/entry_feature_exit_retimer_pairwise_20260628_v1
regime_model_stability=/home/andre2/GX1_DATA/reports/entry_chart_structure_regime_model_stability_20260628_v1
residual_tail_interaction_mining=/home/andre2/GX1_DATA/reports/entry_feature_interaction_tail_regime_model_stability_20260628_v1
residual_tail_veto_retest=/home/andre2/GX1_DATA/reports/entry_chart_structure_residual_tail_veto_retest_20260628_v1
residual_tail_exit_retimer=/home/andre2/GX1_DATA/reports/entry_feature_exit_retimer_residual_tail_20260628_v1
finalist_seed_stability=/home/andre2/GX1_DATA/reports/entry_chart_structure_finalist_seed_stability_20260628_v1
seed9001_tail_interaction_mining=/home/andre2/GX1_DATA/reports/entry_feature_interaction_tail_seed9001_finalists_20260628_v1
voltail_guard_retest=/home/andre2/GX1_DATA/reports/entry_chart_structure_voltail_guard_retest_20260628_v1
voltail_exit_retimer=/home/andre2/GX1_DATA/reports/entry_feature_exit_retimer_voltail_guard_20260628_v1
voltail_residual_interaction_mining=/home/andre2/GX1_DATA/reports/entry_feature_interaction_tail_voltail_guard_20260628_v1
asia_residual_guard_retest=/home/andre2/GX1_DATA/reports/entry_chart_structure_asia_residual_guard_retest_20260628_v1
bodylow_exit_retimer=/home/andre2/GX1_DATA/reports/entry_feature_exit_retimer_bodylow_guard_20260628_v1
lowtail_cost_session_stress=/home/andre2/GX1_DATA/reports/entry_chart_structure_lowtail_cost_session_stress_20260628_v1
```

Baseline to beat, old 2026 replay top10 cost10/slip5:

```text
trades=258
net_sum_bps=5188.5603
net_mean_bps=20.1107
win_rate=0.6318
profit_factor=2.8731
max_loss_bps=-265.2037
max_drawdown_bps=331.6795
positive_months=6/6
```

Main focused candidates, all cost10/slip5/horizon and 6/6 positive months:

| candidate | trades | net bps | mean bps | win | PF | max loss | max DD |
|---|---:|---:|---:|---:|---:|---:|---:|
| chart_layer_only + HistGB + ALL + top3.0 | 178 | 9798.39 | 55.05 | 79.78% | 14.88 | -93.62 | 93.62 |
| chart_layer_only + HistGB + NO_EU + top4.0 | 168 | 8960.35 | 53.34 | 81.55% | 15.79 | -93.62 | 93.62 |
| chart_layer_only + HistGB + ALL + top3.5 | 196 | 10235.54 | 52.22 | 79.08% | 13.84 | -111.69 | 143.78 |
| chart_layer_only + HistGB + ALL + top4.0 | 212 | 10346.57 | 48.80 | 76.89% | 11.02 | -111.69 | 143.78 |
| base_plus_chart + HistGB + OVERLAP_US + top5.0 | 126 | 6685.66 | 53.06 | 80.95% | 18.86 | -73.49 | 73.49 |

Current interpretation:

- Chart-structure features are useful. The old 2026 replay max drawdown
  331.68 bps was materially reduced by structure/deep finalists.
- Deep chart interactions improved low-tail behavior when combined with
  fold-calibrated entry vetoes. However, the initial DD=20 result below did
  not reproduce after the 2026-06-28 multimodel/repro reruns and must be
  treated as stale/non-promotable until explained:

| candidate | trades | net bps | mean bps | win | PF | max loss | max DD |
|---|---:|---:|---:|---:|---:|---:|---:|
| chart_deep_only + HistGB + NO_EU + eu5_body_low5 + top2.0 + horizon | 122 | 8365.37 | 68.57 | 86.89% | 74.00 | -20.12 | 20.12 |
| chart_deep_only + HistGB + NO_EU + eu_trend_delta_low5 + top2.0 + horizon | 125 | 8369.39 | 66.96 | 85.60% | 61.90 | -28.18 | 28.18 |
| chart_deep_only + HistGB + ALL + eu5_range_low5 + top1.8 + horizon | 133 | 8770.19 | 65.94 | 85.71% | 43.60 | -28.18 | 46.98 |
| base_plus_chart_deep + HistGB + OVERLAP_US + top4.0 + horizon | 100 | 6051.15 | 60.51 | 86.00% | 31.57 | -34.96 | 34.96 |
| base_plus_chart_deep + XGBoost + ALL + top5.0 + horizon | 268 | 10351.52 | 38.63 | 75.75% | 6.63 | -132.83 | 148.37 |

2026-06-28 repeatable/current-code candidates:

| candidate | cost | trades | net bps | mean bps | win | PF | max loss | max DD |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| chart_deep_only + HistGB + ALL + eu5_range_low5 + top2.2 + horizon | 10 | 147 | 8492.80 | 57.77 | 84.35% | 23.51 | -81.69 | 81.69 |
| chart_deep_only + HistGB + ALL + eu5_range_low5 + top2.2 + horizon | 20 | 147 | 7022.80 | 47.77 | 73.47% | 11.45 | -91.69 | 98.61 |
| chart_deep_only + HistGB + ALL + eu_trend_delta_low5 + top2.0 + horizon | 20 | 148 | 6759.02 | 45.67 | 73.65% | 10.46 | -91.69 | 91.69 |
| chart_deep_only + HistGB + NO_EU + eu5_body_low5 + top2.0 + horizon | 10 | 109 | 6455.23 | 59.22 | 86.24% | 22.50 | -81.69 | 81.69 |
| chart_deep_only + HistGB + NO_EU + eu5_body_low5 + top2.0 + horizon | 20 | 109 | 5365.23 | 49.22 | 78.90% | 12.10 | -91.69 | 91.69 |
| chart_deep_only + XGBoost + ALL + eu5_range_low5 + top2.0 + horizon | 10 | 178 | 9539.19 | 53.59 | 79.21% | 10.92 | -144.50 | 144.50 |

2026-06-28 pairwise chart-structure tail mining and retest:

```text
single_feature_conditions_screened=all 544 feature-layer columns
pairwise_feature_conditions_screened=37835
compound_veto_syntax=feature_a:side:q&feature_b:side:q inside one rule
rule_combination=compound conditions require all hits; multiple rules remain union vetoes
calibration=fold/session validation thresholds, applied to eval/replay rows
```

Diagnostic pairwise tail themes:

- ASIA wick/level/proximity with squeeze or D1 boundary proximity.
- D1 EMA200 distance low combined with BB squeeze high.
- Distance to last swing low low combined with squeeze or sweep/H1 volume low.
- ASIA expansion/wick combinations. Some improve net, some worsen tail after
  non-leaky retest, so diagnostic hits are not promotion evidence by themselves.

Retested pairwise candidates, all `chart_deep_only + HistGB + ALL + horizon`
and 6/6 positive months:

| candidate | cost | trades | net bps | mean bps | win | PF | max loss | max DD |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| eg_asia_wick_squeeze + top2.2 | 10 | 146 | 8721.80 | 59.74 | 85.62% | 27.32 | -81.69 | 81.69 |
| eg_asia_wick_squeeze + top2.2 | 20 | 146 | 7261.80 | 49.74 | 75.34% | 13.22 | -91.69 | 91.69 |
| eg_squeeze_ema + top2.2 | 20 | 134 | 6313.94 | 47.12 | 73.88% | 12.78 | -48.11 | 63.61 |
| eg_swinglow_squeeze + top2.0 | 20 | 141 | 7086.92 | 50.26 | 73.76% | 12.41 | -87.00 | 87.00 |
| eg_sweep_swinglow + top2.0 | 20 | 137 | 6993.31 | 51.05 | 76.64% | 13.41 | -87.00 | 87.00 |

Pairwise retest interpretation:

- New repeatable net lead is `eg_asia_wick_squeeze + top2.2`: it improves the
  current-code `eu5_range_low5 + top2.2` baseline from `8492.80` to `8721.80`
  bps at cost10, and from `7022.80` to `7261.80` bps at cost20.
- Best tail cut is `eg_squeeze_ema + top2.2` at cost20: max DD falls from
  `98.61` to `63.61` bps and max loss improves from `-91.69` to `-48.11`,
  but net falls to `6313.94` bps. This is a risk-budget candidate, not the
  current net lead.
- `eg_swinglow_squeeze` and `eg_sweep_swinglow` are balanced alternatives:
  roughly `6993-7087` bps at cost20 with `87.00` bps max DD.
- `asia_level_boundary` and `asia_level_trend` were useful diagnostically but
  worsened replay tail in some retests, so they are not current leads.
- Exit retiming on the new pairwise candidates did not beat the fixed horizon.
  `horizon`, `time48`, and `time72` are identical here because the replay
  horizon is 24 bars; tighter tail/volatility exits cut net or raise drawdown.

2026-06-28 regime/model stability and residual-tail retest:

```text
models=HistGB, LightGBM, ExtraTrees, LogisticRegression, XGBoost reference
session_policies=ALL, NO_EU, OVERLAP_US, OVERLAP_ONLY, US_ONLY, ASIA_ONLY, EU_ONLY
seed=4242
feature_set=chart_deep_only
exit=horizon, then residual-tail exit retimer
```

Stability findings:

- The chart/deep layer is useful across the tabular model grid, but the exact
  low-tail row is seed/model sensitive. The stale DD=20 row remains rejected.
- In the seed-4242 stability grid, XGBoost produced the highest cost20 net
  before residual-tail retest (`+7322.44` bps on `eg_asia_wick_squeeze`), but
  its tail/PF profile was weaker (`DD 124.84`, PF `6.46`) than the best HistGB
  low-tail rows.
- The best seed-4242 low-tail row before residual-tail retest was
  `HistGB + NO_EU + eg_squeeze_ema + top1.8`: cost20 `110` trades,
  `+6171.81` bps, PF `17.82`, max loss/DD `-44.65/44.65`.
- Residual tail mining on the new stability artifact showed that the remaining
  large tails cluster around support/resistance pressure with weak H1 trend,
  premium/discount level pressure, sweep/false-breakout-low near support, and
  ASIA/June HOLDOUT tails for the XGBoost ALL policies.

Fold-calibrated residual-tail veto retest, all 6/6 positive months:

| candidate | cost | trades | net bps | mean bps | win | PF | max loss | max DD |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| HistGB + ALL + eg_sweep_support_h1weak + top2.2 | 20 | 161 | 7559.21 | 46.95 | 74.53% | 9.32 | -105.64 | 105.64 |
| HistGB + ALL + eg_sweep_support_h1weak + top1.8 | 20 | 140 | 7474.00 | 53.39 | 77.86% | 12.15 | -105.64 | 105.64 |
| HistGB + ALL + eg_support_h1weak + top1.8 | 20 | 128 | 6775.14 | 52.93 | 78.91% | 13.81 | -52.35 | 63.93 |
| HistGB + NO_EU + eg_support_h1weak + top2.0 | 20 | 112 | 6211.82 | 55.46 | 82.14% | 18.24 | -50.35 | 50.35 |
| HistGB + NO_EU + eg_squeeze_ema + top1.8 | 20 | 110 | 6171.81 | 56.11 | 81.82% | 17.82 | -44.65 | 44.65 |
| HistGB + NO_EU + eg_combo_support_squeeze + top2.2 | 20 | 107 | 5944.37 | 55.55 | 80.37% | 17.03 | -43.68 | 43.68 |
| XGBoost + ALL + eg_sweep_support_h1weak + top2.2 | 20 | 172 | 7401.58 | 43.03 | 72.09% | 7.19 | -105.64 | 148.99 |

Residual-tail interpretation:

- `eg_sweep_support_h1weak` is a new seed-4242 net lead, but it accepts about
  `105.64` bps DD. It is not a low-tail lead.
- `eg_support_h1weak`, `eg_squeeze_ema`, and `eg_combo_support_squeeze` are the
  current low-tail challenger family. They cut DD to roughly `43.68-63.93` bps
  at cost20 while keeping `+5944` to `+6775` bps.
- XGBoost remains useful as a challenger/reference, but the residual-tail retest
  again shows weaker drawdown than the best HistGB low-tail rows.
- Exit retiming on these finalists did not materially reduce tail. The only
  positive exit row was `sweep_low_tight`, which added `+13.77` to `+43.14`
  bps on the sweep-support policies with unchanged DD. Premium/support/tail
  tight exits can lower DD by `10-31` bps, but gave up roughly `670-1137` bps.
  Horizon remains the default exit for low-tail finalists.

2026-06-28 five-seed finalist robustness and volatility-tail guard:

```text
seeds=1337,2027,4242,7777,9001
models=HistGB, LightGBM, ExtraTrees, LogisticRegression, XGBoost in broad finalist grid
focused_retest=HistGB + NO_EU premium/support family with volatility/compression guards
cost=20 bps, slippage=5 bps, exit=horizon unless stated otherwise
```

Five-seed robustness findings:

- The old single-seed low-DD rows did not fully survive seed stress. With
  enough trade coverage, no original candidate held `DD <= 80` across all five
  seeds.
- The only original finalist with `DD <= 120`, `min_trades >= 90`, positive net,
  and 6/6 positive months in all five seeds was
  `HistGB + NO_EU + eg_premium_support_cluster + top2.2`: mean `+5329.90` bps,
  min `+5054.02` bps, mean DD `73.67`, max DD `117.61`, worst loss `-78.48`,
  mean trades `106.0`.
- Residual mining on seed 9001 showed the new tail was no longer mainly
  support/resistance structure. It clustered in volatility/compression shock:
  high `ctx_cont.atr_ratio_m15_d1`, high `ctx_cont.atr_bps`, high
  `ctx_cont.H1_range_compression_ratio`, high
  `chart.compression_to_expansion_proxy`, and ASIA expansion interaction.
- Fold-calibrated retest of those guards produced a stronger current
  low-tail lead:

| candidate | seeds | trades mean/min | net mean/min | win mean | PF min | max loss worst | DD mean/max |
|---|---:|---:|---:|---:|---:|---:|---:|
| HistGB + NO_EU + eg_premium_vol_atrbps_m15d1 + top2.2 | 5 | 97.2 / 80 | 5238.95 / 4716.70 | 76.96% | 12.62 | -65.92 | 55.69 / 74.86 |
| HistGB + NO_EU + eg_premium_support_cluster + top2.2 | 5 | 106.0 / 93 | 5329.90 / 5054.02 | 74.58% | 8.97 | -78.48 | 73.67 / 117.61 |
| HistGB + NO_EU + eg_premium_vol_comp_exp_m15d1 + top2.2 | 5 | 98.6 / 84 | 5134.66 / 4569.30 | 76.48% | 12.43 | -65.92 | 70.31 / 88.63 |
| HistGB + NO_EU + eg_premium_vol_m15d1 + top2.2 | 5 | 89.6 / 74 | 4774.64 / 4363.89 | 77.12% | 12.63 | -65.92 | 55.07 / 74.86 |

Interim vol-tail candidate before ASIA residual retest:

```text
feature_set=chart_deep_only
model=HistGB
session_policy=NO_EU
entry_veto_set=eg_premium_vol_atrbps_m15d1
entry_veto_rules=
  chart.eu_x_trend_delta:low:0.05
  + ctx_cont.m15_range_z_20_canon_v2:low:0.05
  + chart.premium_discount_x_level:high:0.8
    & ctx_cont.sr_support_minus_resistance_prox:high:0.8
  + ctx_cont.atr_bps:high:0.8
    & ctx_cont.atr_ratio_m15_d1:high:0.9
threshold_top_frac=0.022
cost_stress_bps=20
exit=horizon
```

Exit retimer on the current candidate:

- `horizon`, `time48`, and `time72` are identical because the replay horizon is
  24 bars.
- `comp_exp_tight` added only `+18.95` bps mean across five seeds with unchanged
  DD/loss. This is a tiny accounting improvement, not a tail-risk fix.
- `vol_m15d1_tight` cut net by about `197.30` bps mean with unchanged DD/loss.
- Stop/TP and broad `atrbps_tight` exits materially worsened DD. Example:
  `stoptp90_45` raised five-seed DD max to `255.96`.
- Conclusion remains: the edge is entry/veto side. Keep `horizon` as default.

2026-06-28 ASIA residual guard and cost/session stress:

After `eg_premium_vol_atrbps_m15d1`, the remaining `net <= -40` losses across
five seeds were narrow: six trades out of 486 total. They clustered mostly in
ASIA short, especially March/June. Residual interaction mining with
`net <= -35` showed repeated ASIA-structure/body signatures:

```text
chart.asia_x_ll high
chart.is_ASIA_x_trend_proxy low
chart.asia_x_lh high
chart.asia_x_d1_lower high
chart.sweep_x_d1_upper low
chart.asia_x_wick_level high
snap.body_pct low
```

Fold-calibrated residual guard retest:

| candidate | seeds | trades mean/min | net mean/min | win mean | PF min | max loss worst | DD mean/max |
|---|---:|---:|---:|---:|---:|---:|---:|
| HistGB + NO_EU + eg_premium_vol_body_low + top2.2 | 5 | 95.0 / 82 | 5207.89 / 4727.75 | 77.72% | 14.98 | -48.11 | 48.84 / 67.83 |
| HistGB + NO_EU + eg_premium_vol_atrbps_m15d1 + top2.2 | 5 | 97.2 / 80 | 5238.95 / 4716.70 | 76.96% | 12.62 | -65.92 | 55.69 / 74.86 |
| HistGB + NO_EU + eg_premium_vol_asia_lltrend + top2.2 | 5 | 84.4 / 71 | 4775.53 / 4214.45 | 77.95% | 13.83 | -48.11 | 46.87 / 54.93 |
| HistGB + NO_EU + eg_premium_vol_sweep_d1upper + top2.2 | 5 | 82.2 / 71 | 4602.73 / 3965.56 | 78.10% | 16.60 | -43.68 | 42.56 / 54.59 |

This five-seed result made `eg_premium_vol_body_low + top2.2` the temporary
low-tail lead. The later 10-seed replay invalidated it as the main lead:
body-low kept useful tail information, but it did not generalize as cleanly as
the simpler volatility-tail guard.

Current preferred offline candidate:

```text
feature_set=chart_deep_only
model=HistGB
session_policy=NO_EU
entry_veto_set=eg_bos_m15comp_atrratio_m15d1_low5
entry_veto_rules=
  chart.eu_x_trend_delta:low:0.05
  + ctx_cont.m15_range_z_20_canon_v2:low:0.05
  + chart.premium_discount_x_level:high:0.8
    & ctx_cont.sr_support_minus_resistance_prox:high:0.8
  + ctx_cont.atr_bps:high:0.8
    & ctx_cont.atr_ratio_m15_d1:high:0.9
  + ctx_cont.M15_range_compression_ratio:low:0.10
    & ctx_cont.smc_bos_pressure_last12:high:0.90
  + ctx_cont.atr_ratio_m15_d1:low:0.05
threshold_top_frac=0.022
cost_stress_bps=20
exit=horizon
```

Five-seed cost stress on temporary body-low lead vs prior vol-tail lead:

| candidate | cost | net mean/min | DD mean/max | worst loss | trades mean/min | PF min |
|---|---:|---:|---:|---:|---:|---:|
| eg_premium_vol_body_low | 10 | 6157.89 / 5587.75 | 34.74 / 45.53 | -38.11 | 95.0 / 82 | 33.87 |
| eg_premium_vol_atrbps_m15d1 | 10 | 6210.95 / 5516.70 | 40.60 / 55.92 | -55.92 | 97.2 / 80 | 28.27 |
| eg_premium_vol_body_low | 20 | 5207.89 / 4727.75 | 48.84 / 67.83 | -48.11 | 95.0 / 82 | 14.98 |
| eg_premium_vol_atrbps_m15d1 | 20 | 5238.95 / 4716.70 | 55.69 / 74.86 | -65.92 | 97.2 / 80 | 12.62 |
| eg_premium_vol_body_low | 30 | 4257.89 / 3867.75 | 69.52 / 104.32 | -58.11 | 95.0 / 82 | 7.64 |
| eg_premium_vol_atrbps_m15d1 | 30 | 4266.95 / 3916.70 | 79.57 / 104.86 | -75.92 | 97.2 / 80 | 6.50 |
| eg_premium_vol_body_low | 40 | 3307.89 / 3007.75 | 113.66 / 144.32 | -68.11 | 95.0 / 82 | 4.30 |
| eg_premium_vol_atrbps_m15d1 | 40 | 3294.95 / 3116.70 | 122.15 / 144.32 | -85.92 | 97.2 / 80 | 3.70 |

Session stress, cost20 aggregated across seed experiments:

- `OVERLAP` remains the cleanest session for both candidates.
- `ASIA` is still the residual tail source, but `body_low` improves ASIA
  max loss from `-65.92` to `-48.11` while keeping ASIA net positive.
- `US` remains small but positive and low-tail after the NO_EU policy filter.

Exit retimer on `eg_premium_vol_body_low`:

- `horizon`, `time48`, `time72`, and `bodylow_tight` are identical.
- `comp_exp_tight` adds only `+18.95` bps mean with unchanged DD/loss.
- `vol_m15d1_tight` loses about `196.43` bps mean with no useful DD reduction.
- ASIA feature exits and stop/TP exits worsen DD materially. Keep `horizon`.

2026-06-28 10-seed stability and multimodel follow-up:

```text
histgb_10seed_artifact=/home/andre2/GX1_DATA/reports/entry_chart_structure_bodylow_10seed_stability_20260628_v1
multimodel_tail_seed_artifact=/home/andre2/GX1_DATA/reports/entry_chart_structure_10seed_lead_multimodel_check_20260628_v1
xgboost_10seed_artifact=/home/andre2/GX1_DATA/reports/entry_chart_structure_xgboost_10seed_stability_20260628_v1
agreement_probe_artifact=/home/andre2/GX1_DATA/reports/entry_chart_structure_model_agreement_probe_20260628_v1
```

10-seed HistGB stability, cost20/top2.2:

| candidate | seeds | trades mean/min | net mean/min | DD mean/max | worst loss | PF min | positive months min |
|---|---:|---:|---:|---:|---:|---:|---:|
| HistGB + NO_EU + eg_premium_vol_atrbps_m15d1 | 10 | 93.0 / 73 | 5153.85 / 4648.25 | 59.25 / 86.93 | -69.73 | 12.62 | 5 |
| HistGB + NO_EU + eg_premium_vol_body_low | 10 | 89.7 / 69 | 5009.32 / 4161.09 | 61.23 / 113.56 | -69.73 | 14.98 | 5 |
| HistGB + NO_EU + eg_premium_vol_sweep_d1upper | 10 | 77.9 / 63 | 4454.13 / 3965.56 | 48.80 / 113.08 | -87.41 | 16.60 | 5 |
| HistGB + NO_EU + eg_premium_vol_asia_lltrend | 10 | 80.0 / 61 | 4630.86 / 3912.28 | 55.64 / 131.24 | -87.41 | 13.83 | 5 |

10-seed conclusion:

- No 10-seed HistGB row with enough coverage kept 6/6 positive months on every
  seed. The `eg_premium_vol_atrbps_m15d1` row missed only because seed `16180`
  had a tiny April sample: two trades, `-28.40` bps net.
- `eg_premium_vol_atrbps_m15d1 + top2.2` is the current low-DD lead because it
  is the only 10-seed row with `trades_min >= 60`, positive net, and
  `DD max <= 90`.
- `eg_premium_vol_body_low` is demoted to challenger. It was better in the
  first five seeds, but the extra seeds raised max DD to `113.56`.

Focused multimodel check on tail seeds
`12345,16180,4242,9001,1337` tested `HistGB`, `LightGBM`, `ExtraTrees`,
`LogReg`, and `XGBoost` on the same `chart_deep_only` feature layer and veto
grid. XGBoost was strongest on these five tail seeds:

| model + candidate | seeds | trades mean/min | net mean/min | DD mean/max | worst loss | PF min | positive months min |
|---|---:|---:|---:|---:|---:|---:|---:|
| XGBoost + eg_premium_vol_atrbps_m15d1 + top2.2 | 5 | 102.4 / 91 | 5149.29 / 4814.98 | 68.69 / 70.58 | -69.36 | 9.39 | 6 |
| HistGB + eg_premium_vol_atrbps_m15d1 + top2.2 | 5 | 94.4 / 85 | 5157.01 / 4648.25 | 62.19 / 86.93 | -69.73 | 14.71 | 5 |

Because XGBoost looked competitive on tail seeds, it was completed to 10 seeds:

| XGBoost candidate | seeds | trades mean/min | net mean/min | DD mean/max | worst loss | PF min | positive months min |
|---|---:|---:|---:|---:|---:|---:|---:|
| eg_premium_vol_atrbps_m15d1 + top2.2 | 10 | 101.5 / 91 | 5073.03 / 4485.54 | 72.24 / 107.79 | -87.41 | 8.94 | 6 |
| eg_premium_vol_body_low + top2.2 | 10 | 95.0 / 86 | 4729.21 / 4298.23 | 71.63 / 101.73 | -87.41 | 9.37 | 6 |
| eg_premium_vol_sweep_d1upper + top2.2 | 10 | 85.0 / 78 | 4447.48 / 4152.41 | 72.85 / 107.79 | -87.41 | 10.78 | 6 |

XGBoost verdict:

- XGBoost improves month consistency: 6/6 positive months across all 10 seeds
  on the tested top2.2 rows.
- XGBoost does not replace HistGB as low-tail lead: its best 10-seed DD max is
  `100.04-107.79`, and worst loss is `-87.41`. HistGB
  `eg_premium_vol_atrbps_m15d1 + top2.2` keeps lower max DD (`86.93`) and
  lower worst loss (`-69.73`), with higher PF floor.
- XGBoost remains the main challenger if the next objective becomes smoother
  month-level positivity rather than minimum drawdown.

HistGB/XGBoost agreement probe:

| agreement candidate | seeds | trades mean/min | net mean/min | DD mean/max | worst loss | PF min | positive months min |
|---|---:|---:|---:|---:|---:|---:|---:|
| agreement + eg_premium_vol_atrbps_m15d1 + top1.8 | 10 | 37.2 / 27 | 2889.89 / 2441.44 | 30.27 / 45.96 | -38.18 | 16.78 | 5 |
| agreement + eg_premium_vol_body_low + top2.2 | 10 | 43.1 / 36 | 3024.38 / 2544.31 | 41.96 / 67.83 | -47.24 | 15.33 | 5 |

Agreement verdict:

- Model agreement materially reduces tail risk, but coverage is too low for the
  main entry policy today.
- It is a valid next offline branch as a risk-off gate: implement agreement in
  the replay runner with fold-calibrated model scores, then retest cost/session
  stress. Do not treat the post-hoc trade intersection as a deployable policy.

2026-06-28 residual guard v2 and exit retimer:

```text
current_lead_synthetic_replay=/home/andre2/GX1_DATA/reports/entry_chart_structure_current_lead_10seed_synthetic_replay_20260628_v1
current_lead_interaction_tail=/home/andre2/GX1_DATA/reports/entry_feature_interaction_tail_current_lead_10seed_20260628_v1
bos_m15comp_guard_retest=/home/andre2/GX1_DATA/reports/entry_chart_structure_residual_tail_guard_10seed_retest_20260628_v1
bos_m15comp_exit_retimer=/home/andre2/GX1_DATA/reports/entry_feature_exit_retimer_residual_guard_bos_m15comp_10seed_20260628_v1
bos_m15comp_synthetic_replay=/home/andre2/GX1_DATA/reports/entry_chart_structure_bos_m15comp_10seed_synthetic_replay_20260628_v1
bos_m15comp_interaction_tail=/home/andre2/GX1_DATA/reports/entry_feature_interaction_tail_bos_m15comp_10seed_20260628_v1
guard2_retest=/home/andre2/GX1_DATA/reports/entry_chart_structure_residual_tail_guard2_10seed_retest_20260628_v1
guard2_atrratio_exit_retimer=/home/andre2/GX1_DATA/reports/entry_feature_exit_retimer_guard2_atrratio_low5_10seed_20260628_v1
```

The first residual guard pass promoted `eg_atrbps_bos_m15comp` over the older
`eg_premium_vol_atrbps_m15d1` lead:

| candidate | seeds | trades mean/min | net mean/min | DD mean/max | worst loss | PF min | positive months min |
|---|---:|---:|---:|---:|---:|---:|---:|
| eg_atrbps_bos_m15comp + top2.2 | 10 | 92.4 / 71 | 5202.02 / 4638.32 | 57.53 / 74.86 | -69.73 | 12.49 | 5 |
| eg_premium_vol_atrbps_m15d1 + top2.2 | 10 | 93.0 / 73 | 5153.85 / 4648.25 | 59.25 / 86.93 | -69.73 | 12.62 | 5 |

This reduced drawdown path risk but did not remove the largest single loss.
The second residual mining pass found the remaining tail split between
`OVERLAP`/EU-structure shorts and ASIA-short low H1/D1 or M15/D1 ratio
conditions. Fold-calibrated guard2 results:

| candidate | seeds | trades mean/min | net mean/min | DD mean/max | worst loss | PF min | positive months min |
|---|---:|---:|---:|---:|---:|---:|---:|
| eg_bos_m15comp_atrratio_m15d1_low5 + top2.2 | 10 | 82.8 / 65 | 5064.68 / 4547.40 | 49.35 / 71.19 | -48.11 | 13.57 | 5 |
| eg_atrbps_bos_m15comp + top2.2 | 10 | 92.4 / 71 | 5202.02 / 4638.32 | 57.53 / 74.86 | -69.73 | 12.49 | 5 |
| eg_bos_m15comp_eu_hl95_eubos80 + top2.2 | 10 | 89.0 / 67 | 5147.45 / 4090.78 | 53.19 / 74.86 | -65.92 | 14.06 | 5 |

Guard2 conclusion:

- `eg_bos_m15comp_atrratio_m15d1_low5 + top2.2` is the new low-tail lead. It
  gives up about `137` bps mean net and about `10` trades versus
  `eg_atrbps_bos_m15comp`, but cuts worst loss by `21.62` bps and lowers DD max
  by `3.67` bps.
- The extra low `ctx_cont.atr_ratio_m15_d1` veto is not a blunt volatility cap.
  It removes the residual low M15/D1 ratio tail while the existing high
  `atr_bps x atr_ratio_m15_d1` rule still removes the opposite high-volatility
  shock state.
- Positive-months min remains `5`. Some seeds have a no-trade or tiny-negative
  month after the extra guard, but the worst negative month observed in the
  selected monthly table is only `-14.58` bps.
- `eg_bos_m15comp_eu_hl95_eubos80` removes the `-69.73` event in seed `12345`
  but does not beat the low M15/D1 ratio guard on 10-seed net-min/tail profile.

Exit retimer on `eg_bos_m15comp_atrratio_m15d1_low5`:

- `horizon`, `time48`, `time72`, `range_low5_tight`,
  `atrm15d1_low5_tight`, and `atrm15d1_low5_time12` are identical. The entry
  veto has already removed the low-M15/D1 condition, so those conditional exits
  do not trigger.
- Stop/TP, broad feature exits, short time stops, and trailing exits worsen net
  and/or DD. Keep `horizon`.

2026-06-28 all-feature interaction pass on the current low-tail lead:

```text
current_lowtail_synthetic_replay=/home/andre2/GX1_DATA/reports/entry_chart_structure_current_lowtail_10seed_synthetic_replay_20260628_v2
current_lowtail_triple_interaction_tail=/home/andre2/GX1_DATA/reports/entry_feature_interaction_tail_current_lowtail_10seed_triple_20260628_v2
guard3_retest=/home/andre2/GX1_DATA/reports/entry_chart_structure_residual_tail_guard3_10seed_retest_20260628_v1
guard3_tailpress_exit_retimer=/home/andre2/GX1_DATA/reports/entry_feature_exit_retimer_guard3_tailpress_vol_wick_10seed_20260628_v1
```

The feature audit now screens all `544` available features: `181` base
features, `127` chart-structure features, and `236` deep/cross features. The
deep layer includes EMA-stack/cross features such as `chart.ema_stack_cross_up`,
`chart.m5_ema_fast_slow_cross_up/down`, `chart.h1_ema_fast_slow_cross_up/down`,
`chart.h4_ema_fast_slow_cross_up/down`, and D1 EMA-slope cross features.

`analyze_entry_feature_interaction_tail_v1.py` now also emits
`triple_feature_condition_diagnostics.csv`. This lets the offline miner test
three-way feature interactions from the strongest condition pool. These rows are
diagnostic/in-sample only and still require fold-calibrated replay before use.

Residual-tail pattern after `eg_bos_m15comp_atrratio_m15d1_low5`:

- The remaining tails are mostly `ASIA SHORT`: `14` tail losses out of `327`
  ASIA trades, with `ASIA SHORT` tail rate `7.87%`.
- The strongest cross-feature cluster is low-volatility / low-expansion near
  D1-upper context: low `ctx_cont.m15_range_z_20_canon_v2`, low
  `chart.vol_stack`, low `ctx_cont.D1_atr_percentile_252`, low
  `ctx_cont.atr_ratio_h1_d1`, low `chart.sweep_x_d1_upper`, and low
  `chart.tail_pressure_x_d1_upper` / `chart.wick_level_x_d1_upper`.
- This is a different residual regime from the earlier high-volatility shock
  state. The current model is now losing on occasional low-expansion ASIA short
  continuation/rejection states, not the original `-69` overlap event.

Fold-calibrated guard3 retest, top2.2/cost20:

| candidate | seeds | trades mean/min | net mean/min | DD mean/max | worst loss | PF min | positive months min |
|---|---:|---:|---:|---:|---:|---:|---:|
| eg_lowtail_current | 10 | 82.8 / 65 | 5064.68 / 4547.40 | 49.35 / 71.19 | -48.11 | 13.57 | 5 |
| eg_lowtail_vol05_h1d1 | 10 | 82.4 / 64 | 5059.45 / 4565.55 | 49.04 / 71.19 | -48.77 | 14.34 | 5 |
| eg_lowtail_d1atr_vol08_m15range | 10 | 82.1 / 65 | 5056.39 / 4563.36 | 48.33 / 71.19 | -48.11 | 13.45 | 5 |
| eg_lowtail_tailpress_vol_wick | 10 | 75.8 / 60 | 4781.85 / 4369.21 | 42.57 / 70.92 | -43.91 | 17.54 | 5 |

Guard3 conclusion:

- No guard3 candidate replaces the current main low-tail lead. The best
  net-min variants improve net-min slightly but do not improve worst loss.
- `eg_lowtail_tailpress_vol_wick` is a valid risk-off branch: it cuts worst loss
  from `-48.11` to `-43.91` and slightly lowers DD max, but gives up about
  `283` bps mean net and `7` trades versus the main lead.
- `eg_lowtail_m15range20` lowers DD max to `67.83`, but worsens worst loss to
  `-52.96`, so it is not a low-tail promotion candidate.

Exit retimer on `eg_lowtail_tailpress_vol_wick`:

- `horizon`, `time48`, `time72`, and the direct tailpress/vol/wick conditional
  exits are effectively identical.
- `range_low20_tight` adds only about `+63.03` bps mean with unchanged DD/worst
  loss on the risk-off branch. It does not recover enough net/coverage to
  replace the main lead.
- Broad stop/TP, short time stops, and vol/d1-expansion exits still worsen
  net and/or DD.

2026-06-28 ASIA SHORT scoped residual pass:

```text
asia_short_scoped_interaction_tail=/home/andre2/GX1_DATA/reports/entry_feature_interaction_tail_current_lowtail_asia_short_scoped_20260628_v1
asia_short_scoped_guard_retest=/home/andre2/GX1_DATA/reports/entry_chart_structure_asia_short_scoped_guard_10seed_retest_20260628_v1
asia_short_d1dip_exit_replay=/home/andre2/GX1_DATA/reports/entry_chart_structure_asia_short_d1dip_top022_replay_for_exit_20260628_v1
asia_short_d1dip_exit_retimer=/home/andre2/GX1_DATA/reports/entry_feature_exit_retimer_asia_short_d1dip_top022_10seed_20260628_v1
```

The entry runner and exit retimer now support scoped conditions:
`feature:low|high:quantile:LONG|SHORT:ASIA|EU|US|OVERLAP`. This matters because
the residual tail is specifically `ASIA SHORT`; previous entry vetoes could
only be session-scoped and could not target model direction.

ASIA SHORT diagnostic mining on the current low-tail lead screened all `544`
features and filtered the trade universe to `178` ASIA SHORT trades. The scoped
slice had `14` losses <= `-30` bps, net `+8874.66` bps, and worst loss
`-48.11`. Strong diagnostic clusters:

- Liquidity below price / lower-liquidity proximity:
  `ctx_cont.liquidity_lo_nearest_abs_atr`, `ctx_cont.liquidity_lo_minus_hi_prox`,
  and `chart.sweep_x_atr_ratio_h1_d1`.
- D1-upper-distance / dip-context shorts:
  `ctx_cont.dist_to_d1_hi_atr` with `ctx_cont.dip_proximity_d1_v3` or
  `ctx_cont.dip_confirmed_d1_v3`.
- Momentum/EMA proxy cluster:
  low `ctx_cont._v1h1_rsi14_z`, high `ctx_cont.dip_confirmed_m5_v3`, and
  `chart.m5_ema_fast_slow_delta x snap.pos_vs_ema200 x D1 distance`.

Fold-calibrated scoped ASIA SHORT guard retest, top2.2/cost20:

| candidate | seeds | trades mean/min | net mean/min | DD mean/max | worst loss | PF min | positive months min |
|---|---:|---:|---:|---:|---:|---:|---:|
| eg_asia_d1dip_hi85 | 10 | 82.5 / 65 | 5079.28 / 4547.40 | 46.22 / 70.92 | -47.71 | 15.36 | 5 |
| eg_asia_rsi_dipm5 | 10 | 82.5 / 65 | 5079.12 / 4547.40 | 46.22 / 70.92 | -47.71 | 15.36 | 5 |
| eg_asia_d1hi90 | 10 | 82.0 / 64 | 5066.48 / 4565.55 | 47.39 / 67.83 | -52.96 | 16.65 | 5 |
| eg_lowtail_current | 10 | 82.8 / 65 | 5064.68 / 4547.40 | 49.35 / 71.19 | -48.11 | 13.57 | 5 |

Scoped guard conclusion:

- `eg_asia_d1dip_hi85` is the new low-tail lead. It improves mean net, DD mean,
  DD max, worst loss, and PF min versus `eg_lowtail_current`, while preserving
  min net and min trades within practical equivalence.
- The guard removes only `3` trades across the 10-seed top2.2 replay but cuts
  all-trade tail30 count from `18` to `15` and tail45 count from `6` to `3`.
  ASIA SHORT tail30 falls from `14` to `11`; ASIA SHORT tail45 falls from `6`
  to `3`.
- `eg_asia_rsi_dipm5` is nearly identical in aggregate and likely catches the
  same residual trades. Keep it as a challenger, not a separate lead.
- `eg_asia_d1hi90` lowers DD max to `67.83` but introduces a worse worst loss
  (`-52.96`), so it is not a low-tail promotion despite better net-min.

Exit retimer on `eg_asia_d1dip_hi85`:

- `horizon` remains the structural lead exit. Scoped exits do not reduce
  tail30/tail45 further after the entry guard.
- `asia_rsi_dipm5_time12` improves net mean/min (`5094.62 / 4602.15`) with the
  same DD max `70.92` and worst loss `-47.71`, but it triggers on only about
  `0.7%` of trades. Treat it as an optional micro-retimer, not a tail fix.
- `asia_m15low_tight` improves mean net but not net-min or tails enough to
  replace horizon. Broad time stops, trail, and broad stop/TP still worsen net
  or tails.

2026-06-28 true EMA50/EMA200 price-feature opt-in test:

```text
price_ema_v3_retest=/home/andre2/GX1_DATA/reports/entry_chart_structure_price_ema_v3_lead_10seed_retest_20260628_v1
```

The source parquet contains raw `open/high/low/close`, `mid`, `ema100_slope`,
`pos_vs_ema200`, and multi-timeframe regime columns. The research runner now
has an explicit `--include-price-ema-features` option that computes past-only
M5 EMA50/EMA200 features from the source tape and adds session/regime/structure
interactions. Default remains the proven `544`-feature v2 layer; opt-in v3 has
`635` features.

Opt-in v3 was tested over 10 seeds with the current lead vetoes plus true-EMA
scoped veto candidates:

| candidate | top frac | seeds | trades mean/min | net mean/min | DD mean/max | worst loss | PF min |
|---|---:|---:|---:|---:|---:|---:|---:|
| eg_asia_rsi_dipm5 | 0.022 | 10 | 88.2 / 81 | 5214.96 / 4747.32 | 124.91 / 176.60 | -151.07 | 10.32 |
| eg_asia_d1dip_hi85 | 0.022 | 10 | 88.2 / 81 | 5194.96 / 4741.87 | 124.91 / 176.60 | -151.07 | 10.32 |
| eg_lowtail_current | 0.022 | 10 | 88.3 / 81 | 5190.15 / 4741.87 | 124.91 / 176.60 | -151.07 | 10.32 |

Price/EMA conclusion:

- Do not promote price/EMA v3 into the default model feature set. It increases
  coverage and mean/min net, but admits a new `ASIA SHORT` tail:
  `2026-02-03 06:45 UTC`, seed `27182`, net `-151.07` bps, MAE `159.39` bps.
- Tightening top fraction did not solve it: top2.0, top2.2, and top2.4 all
  carried DD max `176.60` and worst loss `-151.07`.
- True EMA scoped veto candidates did not catch the new tail. This suggests
  the current HistGB model overuses the added trend/regime features for score
  ranking before the veto layer can help.
- Keep EMA50/EMA200 features as explicit opt-in research only. The default
  `chart_deep_only` path is restored to the `544`-feature v2 layer.

Current working interpretation after the 2026-06-28 reruns:

- The robust low-tail lead is now `chart_deep_only + HistGB + NO_EU +
  eg_asia_d1dip_hi85 + top2.2 + horizon`. It has the best 10-seed cost20 tail
  profile among rows with enough coverage: `DD max 70.92`, worst loss `-47.71`,
  min net `+4547.40` bps, min trades `65`, PF min `15.36`.
- `eg_bos_m15comp_atrratio_m15d1_low5 + top2.2` is now the prior low-tail
  baseline: `net mean 5064.68`, min net `4547.40`, DD max `71.19`, worst loss
  `-48.11`.
- `eg_atrbps_bos_m15comp + top2.2` is the higher-coverage/higher-net sibling:
  `net mean 5202.02`, min net `4638.32`, min trades `71`, but worst loss remains
  `-69.73`.
- `eg_premium_vol_body_low` remains a useful five-seed tail signature but is no
  longer the preferred lead after 10-seed stress.
- `HistGB + ALL` remains the higher-net research branch
  (`eg_asia_wick_squeeze` / `eg_sweep_support_h1weak`), but the five-seed tail
  is larger and not the current low-tail branch.
- `eg_squeeze_ema`, `eg_support_h1weak`, and `eg_combo_support_squeeze` remain
  useful comparators, but have been superseded by the premium + vol-tail guard
  for low-tail robustness.
  The stale DD=20 NO_EU row is still not a promotion candidate.
- XGBoost is useful but not the low-tail lead. It gives better all-month
  positivity in the 10-seed replay, but it carries worse max DD/worst loss than
  the HistGB low-tail lead.
- LightGBM can produce lower DD rows, but with much lower coverage/net
  (`87` trades and `4259.29` bps at cost10 for the best low-DD row).
- The runner now uses `policy_hash_version=2` for future reports so policy IDs
  include feature-layer/model-training signature and do not collide across
  stale/new model outputs.

- The fold-calibrated vetoes are non-leaky in these replays: thresholds are
  fit on each fold/session validation slice and then applied to eval. Earlier
  single-condition guards such as `eu5_body_low5` helped. The current 10-seed
  low-tail lead is now the scoped ASIA SHORT compound `eg_asia_d1dip_hi85`;
  `eg_bos_m15comp_atrratio_m15d1_low5` is the prior low-tail baseline and
  `eg_premium_vol_body_low` remains a challenger, not the lead.
- `EU_ONLY` is not a lead session. `NO_EU` and `ALL` with explicit session
  interaction features are stronger in the current artifacts.
- Stop/TP, short time-stop, trailing, and feature-conditioned exit variants
  did not improve net in the 2026-06-28 exit-retimer grids. Horizon remains
  the lead exit. Tightening exits can reduce some tails, but only with too much
  net give-up for current candidates.
- XGBoost can score well as a reference and challenger, but the best tail-risk
  candidate is still `HistGB` with chart-deep features and a calibrated veto.
  XGBoost is not required as lead.

2026-06-28 post-EMA multimodel and session/side follow-up:

```text
xgboost_d1dip_5seed=/home/andre2/GX1_DATA/reports/entry_chart_structure_d1dip_xgboost_5seed_20260628_v1
histgb_xgb_agreement_same5=/home/andre2/GX1_DATA/reports/entry_chart_structure_d1dip_histgb_xgb_agreement_same5_20260628_v1
session_side_seed_stability=/home/andre2/GX1_DATA/reports/entry_chart_structure_asia_short_scoped_guard_10seed_retest_20260628_v1/selected_top022_session_side_seed_stability.csv
```

A broad first-seed model screen on the new `eg_asia_d1dip_hi85` guard tested
`HistGB`, `LightGBM`, `ExtraTrees`, `LogReg`, and `XGBoost` on the same
`chart_deep_only` feature layer. Seed `12345`, top2.2/cost20:

| model | trades | net | DD | worst loss | note |
|---|---:|---:|---:|---:|---|
| HistGB | 80 | 5103.24 | 38.18 | -38.18 | clear low-tail lead |
| XGBoost | 81 | 4764.85 | 67.83 | -47.24 | useful challenger |
| ExtraTrees | 28 | 1559.83 | 60.80 | -48.35 | too little coverage |
| LightGBM | 28 | 1607.80 | 120.46 | -120.46 | tail failure |
| LogReg | 134 | 3970.67 | 370.94 | -364.91 | not viable |

The broad run was stopped after the first complete seed because LightGBM,
ExtraTrees, and LogReg were not competitive enough to justify full runtime.
XGBoost was then rerun alone over five tail-sensitive seeds
`12345,16180,4242,9001,1337`.

Same-five top2.2/cost20 comparison:

| model + guard | seeds | trades mean/min | net mean/min | DD mean/max | worst loss | PF min | positive months min |
|---|---:|---:|---:|---:|---:|---:|---:|
| HistGB + eg_asia_d1dip_hi85 | 5 | 83.8 / 74 | 5122.55 / 4812.86 | 43.19 / 54.59 | -43.68 | 19.57 | 5 |
| XGBoost + eg_asia_rsi_dipm5 | 5 | 89.0 / 80 | 4886.13 / 4726.41 | 71.02 / 83.97 | -83.97 | 10.55 | 5 |
| XGBoost + eg_asia_d1dip_hi85 | 5 | 89.4 / 81 | 4885.93 / 4717.71 | 71.02 / 83.97 | -83.97 | 10.61 | 5 |

XGBoost conclusion after the new guard:

- XGBoost is not required and does not replace HistGB. It has slightly higher
  coverage on the same five seeds, but lower net, higher DD, much worse worst
  loss, and lower PF floor.
- XGBoost top2.2 tail counts over five seeds were `19` losses <= `-30`, `10`
  <= `-45`, and `6` <= `-60`; `13` of the <= `-30` tails were ASIA SHORT.
- Keep XGBoost as an offline challenger and possible agreement/risk-off signal,
  not as the main model dependency.

Post-hoc HistGB/XGBoost agreement on the same five seeds:

| policy | trades mean/min | net mean/min | DD mean/max | worst loss | tail <= -45 | note |
|---|---:|---:|---:|---:|---:|---|
| HistGB d1dip | 83.8 / 74 | 5122.55 / 4812.86 | 43.19 / 54.59 | -43.68 | 0 | main lead |
| XGBoost d1dip | 89.4 / 81 | 4885.93 / 4717.71 | 71.02 / 83.97 | -83.97 | 10 | challenger only |
| agreement intersection | 42.0 / 33 | 3454.77 / 3007.82 | 34.24 / 56.34 | -38.18 | 0 | risk-off branch, not deployable yet |

Agreement materially cuts tails but gives up too much coverage/net for the main
entry policy. It is worth implementing later as a native replay gate only if
the objective becomes a lower-frequency risk-off mode. The current intersection
is post-hoc and should not be treated as deployable policy logic.

Current lead session/side tail stability, top2.2/cost20 over 10 seeds:

| slice | trades mean/min | net mean/min | DD mean/max | worst loss | tail <= -30 | tail <= -45 |
|---|---:|---:|---:|---:|---:|---:|
| ASIA SHORT | 17.5 / 13 | 901.90 / 782.69 | 46.02 / 89.07 | -47.71 | 11 | 3 |
| ASIA LONG | 14.9 / 11 | 777.26 / 661.98 | 6.11 / 14.58 | -14.58 | 0 | 0 |
| OVERLAP SHORT | 19.1 / 15 | 1429.87 / 1240.99 | 28.13 / 39.88 | -39.88 | 1 | 0 |
| OVERLAP LONG | 18.4 / 15 | 930.28 / 824.91 | 15.06 / 32.78 | -18.44 | 0 | 0 |
| US LONG | 8.0 / 5 | 758.01 / 644.80 | 27.25 / 43.68 | -43.68 | 3 | 0 |
| US SHORT | 4.6 / 3 | 281.96 / 174.40 | 20.00 / 34.20 | -27.00 | 0 | 0 |

Session/side conclusion: the remaining tail is still concentrated in
`ASIA SHORT`. EU exclusion remains correct; adding a broad ASIA ban would throw
away too much profitable ASIA LONG and ASIA SHORT net. Further work should be
ASIA SHORT-specific structure, not session-wide removal.

Tail/feature audit interpretation:

- Dataset-level MAE/path tails are dominated by volatility and range context:
  `atr_bps`, `rvol_20`, `_v1_atr14`, H1/H4/D1 ATR, `vol_regime_id`,
  `atr_bucket`, `chart.vol_stack`, `chart.d1_range_x_expansion`, and
  `chart.compression_to_expansion_proxy`.
- Useful chart-structure signal exists in D1 range expansion, session x
  trend/volatility, pullback x range, HH/LL x D1 location, sweep/wick/level
  interactions, the fold-calibrated `chart.eu_x_trend_delta` guard, and the
  two-sided M15/D1 ATR-ratio tail filter.
- High volatility is not simply bad; it also carries winners. Blunt ATR-high
  vetoes reduced drawdown but destroyed net edge. Tail guards must be structure
  aware, not just volatility caps.

2026-06-28 current-code chart-structure and risk-overlay pass:

```text
currentcode_d1dip_residual_guard_10seed=/home/andre2/GX1_DATA/reports/entry_chart_structure_d1dip_residual_guard_currentcode_10seed_20260628_v1
hh_rsi_bos_guard_10seed=/home/andre2/GX1_DATA/reports/entry_chart_structure_hh_rsi_bos_guard_probe_5seed_20260628_v1
hh_rsi_residual_fast_asia_short=/home/andre2/GX1_DATA/reports/entry_residual_tail_fast_hh_rsi_asia_short_10seed_20260628_v1
hh_rsi_residual_fast_all_moderate=/home/andre2/GX1_DATA/reports/entry_residual_tail_fast_hh_rsi_all_moderate_10seed_20260628_v1
risk_overlay_10seed=/home/andre2/GX1_DATA/reports/entry_risk_overlay_hh_bos_10seed_20260628_v1
threshold_cost_stress_10seed=/home/andre2/GX1_DATA/reports/entry_risk_overlay_hh_bos_threshold_cost_stress_10seed_20260628_v1
exit_retimer_current_lead=/home/andre2/GX1_DATA/reports/entry_exit_retimer_hh_bos_session75_10seed_20260628_v1
xgboost_current_lead_10seed=/home/andre2/GX1_DATA/reports/entry_chart_structure_hh_bos_session75_xgboost_10seed_20260628_v1
xgboost_current_lead_overlay=/home/andre2/GX1_DATA/reports/entry_risk_overlay_hh_bos_xgboost_10seed_20260628_v1
```

Important correction: the older `eg_asia_d1dip_hi85` 10-seed artifact should be
treated as stale for leadership comparisons after the price-EMA feature default
fix. Current-code replays are authoritative.

Current-code entry-only top2.2/cost20, 10 seeds:

| policy | trades mean/min | net mean/min | DD mean/max | worst loss | tail <= -35 | tail <= -45 |
|---|---:|---:|---:|---:|---:|---:|
| `eg_asia_hh_rsi_d1range` | 80.9 / 70 | 5020.72 / 4297.73 | 45.59 / 84.54 | -52.42 | 3 | 1 |
| `eg_asia_hh_bos_session75` | 77.2 / 67 | 5002.40 / 4251.17 | 37.79 / 84.54 | -39.88 | 2 | 0 |
| `eg_asia_hh_hhregime80` | 79.1 / 68 | 4993.63 / 4355.21 | 40.02 / 84.54 | -39.88 | 2 | 0 |

Interpretation:

- `eg_asia_hh_bos_session75` and `eg_asia_hh_hhregime80` remove the `-52.42`
  ASIA SHORT tail and all `<= -45` losses over 10 seeds.
- They do not solve max drawdown by entry filtering alone; the `84.54` DD is a
  mixed-session loss sequence in seed `16180`, not a single ASIA SHORT tail.
- Static exits and feature-conditioned exits were retested over the full
  10-policy current lead. No retimer rule beat `horizon` while keeping at least
  95 percent of horizon net. Keep `horizon` for now.

Risk-overlay replay, top2.2/cost20, 10 seeds:

| policy + overlay | trades mean/min | skipped mean | net mean/min | DD mean/max | worst loss | tail <= -45 |
|---|---:|---:|---:|---:|---:|---:|
| `eg_asia_hh_bos_session75 + two_loss25_24h` | 77.0 / 67 | 0.2 | 5005.30 / 4251.17 | 34.89 / 55.57 | -39.88 | 0 |
| `eg_asia_hh_hhregime80 + two_loss25_24h` | 78.9 / 68 | 0.2 | 4996.52 / 4355.21 | 37.12 / 63.64 | -39.88 | 0 |
| `eg_asia_hh_bos_session75` no overlay | 77.2 / 67 | 0.0 | 5002.40 / 4251.17 | 37.79 / 84.54 | -39.88 | 0 |

Threshold and cost stress for `eg_asia_hh_bos_session75 + two_loss25_24h`:

| threshold/cost | trades mean/min | net mean/min | DD mean/max | worst loss | tail <= -45 | status |
|---|---:|---:|---:|---:|---:|---|
| top2.0 / cost20 | 72.1 / 61 | 4837.65 / 3652.00 | 37.94 / 55.57 | -52.96 | 1 | reject vs top2.2 |
| top2.2 / cost20 | 77.0 / 67 | 5005.30 / 4251.17 | 34.89 / 55.57 | -39.88 | 0 | current lead |
| top2.4 / cost20 | 81.2 / 69 | 5053.17 / 4346.28 | 50.26 / 121.69 | -121.69 | 3 | reject |
| top2.2 / cost10 | 77.0 / 67 | 5775.30 / 4941.17 | 21.35 / 35.57 | -29.88 | 0 | sensitivity only |
| top2.2 / cost30 | 77.0 / 67 | 4235.30 / 3561.17 | 52.26 / 82.44 | -49.88 | 2 | positive but tail-stressed |

Current-lead exit retimer aggregate, 10 policies:

| exit rule | trades mean/min | net mean/min | DD mean/max | worst loss | PF min | verdict |
|---|---:|---:|---:|---:|---:|---|
| `horizon` | 77.2 / 67 | 5002.40 / 4251.17 | 37.79 / 84.54 | -39.88 | 24.66 | keep |
| `asia_short_bos_trail` | 77.2 / 67 | 4746.37 / 3984.24 | 37.13 / 84.54 | -39.88 | 24.00 | lower net, no DD-max gain |
| `asia_short_bos_time12` | 77.2 / 67 | 4788.15 / 4036.04 | 38.36 / 84.54 | -39.88 | 20.88 | lower net, no DD-max gain |
| `trail80_35` | 77.2 / 67 | 3326.19 / 2791.55 | 33.27 / 57.13 | -39.88 | 17.20 | DD lower, net destroyed |

Current-lead model check:

| model branch | threshold | trades mean/min | net mean/min | DD mean/max | worst loss | tail <= -45 | verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| HistGB `eg_asia_hh_bos_session75` | top2.2 | 77.2 / 67 | 5002.40 / 4251.17 | 37.79 / 84.54 | -39.88 | 0 | lead before overlay |
| HistGB + `two_loss25_24h` | top2.2 | 77.0 / 67 | 5005.30 / 4251.17 | 34.89 / 55.57 | -39.88 | 0 | current lead |
| XGBoost `eg_asia_hh_bos_session75` | top2.0 | 80.6 / 72 | 4769.36 / 4442.16 | 69.97 / 90.01 | -63.73 | 14 | reject |
| XGBoost `eg_asia_hh_bos_session75` | top2.2 | 85.9 / 74 | 4870.73 / 4502.52 | 71.61 / 90.01 | -83.97 | 16 | reject |
| XGBoost `eg_asia_hh_bos_session75` | top2.4 | 89.0 / 77 | 4886.69 / 4590.41 | 68.10 / 69.36 | -69.36 | 17 | reject |

The broad first-seed model probe also rejected LogReg, LightGBM, and
ExtraTrees for this current lead: LogReg had a `-364.91` bps tail, LightGBM had
low trade coverage/net, and ExtraTrees had low trade coverage/net. XGBoost was
the only alternate model worth completing to 10 seeds, and it still failed the
tail-risk requirement.

The best current offline branch is therefore:

```text
chart_deep_only + HistGB + NO_EU + eg_asia_hh_bos_session75 + top2.2 + horizon
+ risk overlay: after two consecutive closed losses <= -25 bps, pause new
  entries for 24h from the second loss exit time
```

This branch is not deployable yet. The overlay must be implemented in native
replay/paper logic and retested with exact closed-trade timing, extra seeds,
month/session/day stability, cost stress, and train/serve parity.

Latest local verification:

```text
py_compile_status=PASS for chart-structure runner, pairwise miner, exit retimer,
  and deep tail audit
scripts/entry_next_edge_control.sh verify=PASS; checks_passed=479
scripts/entry_next_edge_control.sh selftest=PASS; cases_passed=53
shadow_runner_started=false
live_order_placement=NOT_STARTED_NOT_ENABLED
pgrep_shadow_live_runner=none
```

Immediate next action:

```text
Keep iterating offline around the current risk-controlled lead:
  chart_deep_only + HistGB + NO_EU + eg_asia_hh_bos_session75 + top2.2 + horizon
  + two_loss25_24h risk overlay
Compare, but do not currently prefer, the higher-net research branches:
  chart_deep_only + HistGB + NO_EU + eg_asia_hh_rsi_d1range + top2.2 + horizon
  chart_deep_only + HistGB + NO_EU + eg_asia_hh_hhregime80 + top2.2 + horizon
  chart_deep_only + HistGB + NO_EU + eg_asia_d1dip_hi85 + top2.2 + horizon
  chart_deep_only + HistGB + NO_EU + eg_bos_m15comp_atrratio_m15d1_low5 + top2.2 + horizon
  chart_deep_only + HistGB + NO_EU + eg_atrbps_bos_m15comp + top2.2 + horizon
  chart_deep_only + HistGB + ALL + eg_asia_wick_squeeze + top2.0/top2.2 + horizon
  chart_deep_only + HistGB + ALL/NO_EU + eg_sweep_support_h1weak + top2.0/top2.2 + horizon
  chart_deep_only + HistGB + NO_EU + eg_premium_vol_body_low + top2.2 + horizon
  chart_deep_only + HistGB + NO_EU + eg_premium_support_cluster + top2.2 + horizon
Do not currently prefer XGBoost:
  chart_deep_only + XGBoost + NO_EU + eg_asia_hh_bos_session75 failed
  10-seed tail stress at top2.0/top2.2/top2.4
  HistGB/XGBoost agreement gate remains research-only, not a lead branch
Then run robustness checks before any packaging:
  additional seed/repeat stability, cost/slippage stress, fold/month/day/session
  diagnostics, and train/serve feature parity for the generated chart/deep/veto
  feature layer
Do not start shadow unless that is the explicit next decision after replay review.
Do not start new ET/XGB training before 2026 replay diagnostics are reviewed.
Do not start more ET/XGB training unless a new chart-structure hypothesis first
passes 2026 replay diagnostics under HistGB.
Do not start new ET/Transformer/IQL/TCN training before the chart-structure
finalists are replayed/stressed offline.
```

2026-06-28 drawdown-atlas and veto-only EMA pass:

```text
drawdown_atlas_currentlead=/home/andre2/GX1_DATA/reports/entry_feature_drawdown_atlas_hh_bos_currentlead_20260628_v1
dd_atlas_guard_10seed=/home/andre2/GX1_DATA/reports/entry_chart_structure_hh_bos_dd_atlas_guard_10seed_20260628_v1
dd_atlas_guard_risk_overlay=/home/andre2/GX1_DATA/reports/entry_risk_overlay_hh_bos_dd_atlas_guard_10seed_20260628_v1
dd_atlas_guard_exit_retimer=/home/andre2/GX1_DATA/reports/entry_exit_retimer_hh_bos_dd_atlas_guard_10seed_20260628_v1
dd_atlas_price_ema=/home/andre2/GX1_DATA/reports/entry_feature_drawdown_atlas_hh_bos_dd_atlas_price_ema_20260628_v1
ema_vetoonly_10seed=/home/andre2/GX1_DATA/reports/entry_chart_structure_hh_bos_dd_atlas_ema_vetoonly_10seed_20260628_v1
ema_vetoonly_risk_overlay=/home/andre2/GX1_DATA/reports/entry_risk_overlay_hh_bos_dd_atlas_ema_vetoonly_10seed_20260628_v1
ema_vetoonly_costgrid=/home/andre2/GX1_DATA/reports/entry_risk_overlay_hh_bos_dd_atlas_ema_vetoonly_costgrid_10seed_20260628_v1
ema_vetoonly_exit_retimer=/home/andre2/GX1_DATA/reports/entry_exit_retimer_hh_bos_dd_atlas_ema_vetoonly_10seed_20260628_v1
```

Implementation update:

- `gx1/scripts/analyze_entry_feature_drawdown_atlas_v1.py` adds an offline
  feature/drawdown atlas for residual drawdown episodes and candidate
  single/pair/triple entry/exit rules.
- `gx1/scripts/experiment_entry_chart_structure_ablation_v1.py` now supports
  `--veto-only-price-ema-features`, so EMA/price-derived features can be used
  in fold-calibrated veto rules without adding those features to model
  training. This avoids repeating the failed price-EMA-as-model-feature path.

Drawdown atlas on the old current lead found the remaining `84.54` bps max DD
was not a pure ASIA/SHORT or pure high-volatility issue. It was mostly a May
HOLDOUT mixed-session sequence, especially seed `16180`, with risk concentrated
around low H4 RSI, low S1 distance, short H4 trend-age, pullback/regime, and
EU/session interaction features.

New drawdown-atlas guard:

```text
eg_dd_h4rsi_s1_h4age =
  eg_asia_hh_bos_session75
  + ctx_cont._v1h4_rsi14_z:low:0.05
    & ctx_cont.dist_to_S1_atr:low:0.08
    & ctx_cont.h4_trend_age_bars_norm_v2:low:0.08
```

Entry-only top2.2/cost20, 10 seeds:

| policy | trades mean/min | net mean/min | DD mean/max | worst loss | tail <= -35 | tail <= -45 |
|---|---:|---:|---:|---:|---:|---:|
| old `eg_asia_hh_bos_session75` | 77.2 / 67 | 5002.40 / 4251.17 | 37.79 / 84.54 | -39.88 | 2 | 0 |
| old + `two_loss25_24h` overlay | 77.0 / 67 | 5005.30 / 4251.17 | 34.89 / 55.57 | -39.88 | 2 | 0 |
| `eg_dd_h4rsi_s1_h4age` | 76.8 / 67 | 5013.87 / 4251.17 | 30.45 / 39.88 | -39.88 | 2 | 0 |
| `eg_dd_s1_lhpull` | 75.5 / 64 | 4936.87 / 4158.21 | 30.45 / 39.88 | -39.88 | 2 | 0 |

This replaces the previous risk-overlay-led branch as the entry-only offline
lead. The old `84.54` drawdown sequence in seed `16180` drops to `28.98` bps
without a post-loss overlay.

Veto-only EMA replay, top2.2/cost20, 10 seeds:

| policy | trades mean/min | net mean/min | DD mean/max | worst loss | tail <= -25 | tail <= -35 | verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| `eg_dd_h4rsi_s1_h4age` | 76.8 / 67 | 5013.87 / 4251.17 | 30.45 / 39.88 | -39.88 | 16 | 2 | baseline |
| `eg_dd_h4rsi_s1_h4age_ema_bos_level` | 75.5 / 64 | 5006.64 / 4214.68 | 27.91 / 37.41 | -34.62 | 13 | 0 | net-friendly lead |
| `eg_dd_h4rsi_s1_h4age_ema_sr` | 76.0 / 67 | 4998.10 / 4246.85 | 27.77 / 37.41 | -34.62 | 7 | 0 | tail-friendly challenger |
| `eg_dd_h4rsi_s1_h4age_ema_ext` | 74.1 / 64 | 4546.76 / 3746.50 | 27.77 / 37.41 | -34.62 | 7 | 0 | reject, net too low |

The EMA features were veto-only. Model feature count stayed `497`; veto feature
count was `635`. The useful EMA rule for the net-friendly lead is:

```text
chart.eu_x_ema50_200:high:0.7
& chart.eu_x_level_prox:high:0.92
& chart.is_eu_only_x_bos:high:0.95
```

Per-seed diagnosis: the EMA veto helps seed `27182` and `54321`; it was not
needed for seed `16180`, which was already fixed by the H4 RSI / S1 /
trend-age guard.

Cost stress, no overlay:

| policy | cost | net mean/min | DD mean/max | worst loss | tail <= -35 | tail <= -45 |
|---|---:|---:|---:|---:|---:|---:|
| `eg_dd_h4rsi_s1_h4age` | 10 | 5781.87 / 4941.17 | 18.60 / 29.88 | -29.88 | 0 | 0 |
| `eg_dd_h4rsi_s1_h4age_ema_bos_level` | 10 | 5761.64 / 4884.68 | 15.82 / 24.62 | -24.62 | 0 | 0 |
| `eg_dd_h4rsi_s1_h4age_ema_sr` | 10 | 5758.10 / 4926.85 | 15.57 / 24.62 | -24.62 | 0 | 0 |
| `eg_dd_h4rsi_s1_h4age` | 30 | 4245.87 / 3561.17 | 45.82 / 61.84 | -49.88 | 16 | 2 |
| `eg_dd_h4rsi_s1_h4age_ema_bos_level` | 30 | 4251.64 / 3544.68 | 44.52 / 61.84 | -44.62 | 13 | 0 |
| `eg_dd_h4rsi_s1_h4age_ema_sr` | 30 | 4238.10 / 3566.85 | 45.82 / 61.84 | -44.62 | 7 | 0 |

Risk overlay on the EMA finalists:

| policy + overlay | trades mean/min | skipped mean | net mean/min | DD mean/max | worst loss | tail <= -35 |
|---|---:|---:|---:|---:|---:|---:|
| `ema_bos_level` no overlay | 75.5 / 64 | 0.0 | 5006.64 / 4214.68 | 27.91 / 37.41 | -34.62 | 0 |
| `ema_bos_level + cool_loss30_6h` | 75.3 / 64 | 0.2 | 5005.87 / 4214.68 | 27.35 / 34.62 | -34.62 | 0 |
| `ema_sr` no overlay | 76.0 / 67 | 0.0 | 4998.10 / 4246.85 | 27.77 / 37.41 | -34.62 | 0 |
| `ema_sr + cool_loss30_6h` | 75.8 / 67 | 0.2 | 4997.34 / 4246.85 | 27.22 / 34.62 | -34.62 | 0 |

The light `cool_loss30_6h` overlay is a small secondary improvement, but it is
not required to remove the `<= -35` tail. Treat it as optional until native
closed-trade timing is implemented and retested.

Exit retimer on EMA finalists:

| policy | best exit | net mean/min | DD mean/max | worst loss | verdict |
|---|---|---:|---:|---:|---|
| `ema_bos_level` | `horizon` | 5006.64 / 4214.68 | 27.91 / 37.41 | -34.62 | keep |
| `ema_sr` | `horizon` | 4998.10 / 4246.85 | 27.77 / 37.41 | -34.62 | keep |

`time48` and `time72` are equivalent to horizon because base horizon is 24
bars. Static stop/TP and trail exits reduce net or reintroduce larger DD/tails.
EMA-conditioned exits also failed: they either reduce net or increase max DD.

Best current offline branch:

```text
chart_deep_only + HistGB + NO_EU
+ eg_dd_h4rsi_s1_h4age_ema_bos_level
+ top2.2 + horizon
```

Risk-first challenger:

```text
chart_deep_only + HistGB + NO_EU
+ eg_dd_h4rsi_s1_h4age_ema_sr
+ top2.2 + horizon
```

Current interpretation:

- XGBoost remains rejected for this branch; it increased tail risk in 10-seed
  stress.
- New Transformer/IQL/TCN training is not justified before locking this
  chart-structure feature/veto layer through additional fold/month/session
  stress and train/serve parity.
- EU is not "always best" here. Pure EU was filtered out by `NO_EU`; the useful
  signal is an EU-context feature used as a veto on non-EU allowed trades, not
  an instruction to trade EU-only.
- Do not start shadow. Next work remains offline replay/stress and parity.

## Hygiene Rules

Never promote or pin a bundle from this plan on accuracy alone.

Do not use these as live/promotion candidates:

```text
v10_bundle_6yr_symneg_smartctx_xgbfixed_h24_cwp030
v10_bundle_6yr_symneg_smartctx_xgbfixed_h24_cwp05
v10_bundle_6yr_baseline_smartctx
v10_bundle_6yr_symneg_smartctx
```

Do not build new ET datasets from implicit XGB defaults. Always pass an explicit
`XGB_BUNDLE_DIR`.

Do not make XGB a required dependency for new Entry live architecture unless a
future artifact proves incremental selective EV over the tabular no-XGB baseline.
Current status: XGB is reference/comparison only, not the lead path.

Do not run ET retraining through `scripts/v10_6yr_rebuild_20260626.sh` unless:

```text
GX1_ENTRY_REBUILD_PLAN_ACK=20260627_SELECTIVE_EDGE
DATASET_DIR=<explicit dataset>
OUT_BUNDLE_DIR=<explicit output bundle>
ENTRY_RESIDUAL_SCALE=<explicit residual scale>
TRAIN_EXTRA_ARGS="--weight-decay <explicit value>"
```

Reference candidate training args:

```text
ENTRY_RESIDUAL_SCALE=0.25
TRAIN_EXTRA_ARGS="--weight-decay 1e-4"
```

## Completion Criteria For This Plan

This plan is complete only when one of these is true:

1. A candidate passes selective EV/PnL gates and promotion gates, with artifacts
   recorded.
2. The current label/objective is proven non-viable and replaced by a new locked
   label/objective plan.

Until then, this is active research, not a live-system change.
