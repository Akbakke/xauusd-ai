# 72-Hour Tuning Orchestration Plan

**Date**: 2026-02-04  
**Status**: TRUTH SAFE (airbag test PASS)  
**Goal**: Systematic hyperparameter search for XGB + Transformer + thresholds without breaking determinism or contracts

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                   72-HOUR TUNING ORCHESTRATOR                    │
│              gx1/scripts/research/run_72h_tuning.sh              │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ├──────────────────────────────────┐
                              │                                  │
                    ┌─────────▼─────────┐              ┌─────────▼─────────┐
                    │   PHASE A: XGB    │              │   PHASE B: TRANS │
                    │   Optuna Tuning   │              │   Depth Ladder   │
                    │   (CPU Heavy)     │              │   (GPU Heavy)    │
                    │   ~24-36 hours    │              │   ~12-24 hours   │
                    └─────────┬─────────┘              └─────────┬─────────┘
                              │                                  │
                              │                                  │
                    ┌─────────▼──────────────────────────────────▼─────────┐
                    │         PHASE C: Operating Region Scan                │
                    │         (Cheap, Decisive)                            │
                    │         ~2-4 hours                                   │
                    └──────────────────────────────────────────────────────┘
                              │
                              │
                    ┌─────────▼─────────┐
                    │   TOP-K REPORT    │
                    │   Decision Packet │
                    └───────────────────┘
```

### Data Flow

```
PREBUILT (TRIAL160/2024,2025)
    │
    ├─► Phase A: XGB Training (per session head)
    │   └─► Optuna Study (SQLite)
    │       └─► Candidate Models (joblib + metadata)
    │
    ├─► Phase B: Transformer Training (depth variants)
    │   └─► Checkpoint Bundles (model_state_dict.pt + metadata)
    │
    └─► Phase C: Evaluation (TRUTH_CHAIN_COMPUTE)
        └─► YEAR_METRICS + PAYOFF_PANEL
            └─► Decision Surface (PnL/DD/trade-rate)
```

---

## Reused Existing Modules

### Training Infrastructure
- `gx1/scripts/train_xgb_universal_multihead_v2.py` - XGB training patterns
- `gx1/scripts/train_entry_v10_ctx_depth_ladder.py` - Transformer training patterns
- `gx1/xgb/multihead/xgb_multihead_model_v1.py` - XGB model class
- `gx1/models/entry_v10/entry_v10_ctx_train.py` - Transformer training utilities

### Evaluation Infrastructure
- `gx1/scripts/replay_eval_chain_compute.py` - TRUTH_CHAIN_COMPUTE runner
- `gx1/scripts/build_year_metrics.py` - Metrics aggregation
- `gx1/scripts/run_chain_compute_fullyear_2025.sh` - Evaluation harness pattern

### Utilities
- `gx1/xgb/preprocess/xgb_input_sanitizer.py` - Feature sanitization
- `gx1/time/session_detector.py` - Session detection
- `gx1/features/feature_contract_v10_ctx.py` - Contract SSoT
- `gx1/runtime/run_identity.py` - RUN_IDENTITY generation

### Optuna Patterns
- `gx1/scripts/optuna_tune_xgb_us_drift_2024_2025.py` - Optuna study structure

---

## Phase A: XGB Multihead Optuna (24-36 hours)

### Objective
Improve per-session score distributions (especially US/ASIA) while preserving stability.

### Strategy
- **One Optuna study per session head** (EU, OVERLAP, US, ASIA)
- **SQLite storage**: `$GX1_DATA/optuna/xgb_phase_a_<session>_<ts>.db`
- **Parallel trials**: Max CPU cores, but per-trial determinism (fixed seed per trial)
- **Pruning**: Early stop if score distribution collapses or drift exceeds thresholds

### Search Space
```python
{
    "max_depth": [3, 4, 5, 6, 7, 8],
    "min_child_weight": [1, 2, 5, 10, 15, 20],
    "subsample": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    "eta": [0.03, 0.05, 0.07, 0.10, 0.12, 0.15],
    "reg_alpha": [0.0, 0.1, 0.5, 1.0, 2.0],
    "reg_lambda": [0.0, 0.1, 0.5, 1.0, 2.0],
    "gamma": [0.0, 0.5, 1.0, 2.0, 5.0],
}
```

### Evaluation (Fast)
- **Window**: 2025 only (fast smoke)
- **Metrics**: Per-session score quantiles, trade-rate vs threshold (coarse), PnL/DD tails
- **Harness**: TRUTH_CHAIN_COMPUTE with frozen Transformer (baseline bundle)

### Outputs
- Top K candidates per session (K=5)
- Each candidate: `xgb_candidate_<session>_<trial_id>.joblib` + metadata JSON
- Metadata includes: hyperparams, score stats, drift metrics, dataset fingerprint

### Timeline
- **Trial time**: ~5-10 min per trial (2025 window)
- **Trials per session**: 100-200
- **Parallelization**: 8-16 workers (CPU-bound)
- **Total**: ~24-36 hours for all sessions

---

## Phase B: Transformer Depth Ladder (12-24 hours)

### Objective
Test deeper transformer (4, 6 layers) for non-EU utilization without killing edge.

### Strategy
- **Depth ladder**: {3 baseline, 4, 6}
- **Fixed params**: d_model=128, n_heads=4, dim_feedforward=512, dropout=0.05
- **Training**: Same data windows, fixed seeds, same epochs
- **Verification**: Bundle loads, XGB channel telemetry correct

### Evaluation
- **Windows**: Q1 2025 (smoke) + Full 2025 (confirm)
- **Harness**: TRUTH_CHAIN_COMPUTE with best XGB from Phase A
- **Metrics**: Same as Phase A + bundle compatibility checks

### Outputs
- Checkpoint bundles: `transformer_depth_<layers>_<ts>/`
- Each bundle: `model_state_dict.pt` + `bundle_metadata.json` + SHA256
- GO/NO-GO decision per depth variant

### Timeline
- **Training per depth**: ~2-4 hours (GPU)
- **Evaluation per depth**: ~1-2 hours
- **Total**: ~12-24 hours for all depths

---

## Phase C: Operating Region Scan (2-4 hours)

### Objective
Choose thresholds (global or per-session) that maximize PnL/DD under costs.

### Strategy
- **Finalists**: Best XGB from Phase A + best Transformer depth from Phase B
- **Thresholds**: Global {0.56, 0.60, 0.64, 0.68}
- **Optional**: Per-session threshold grid (small, 2-3 values per session)
- **Cost sensitivity**: {0, realistic, stress(2x)}

### Evaluation
- **Window**: Full 2025
- **Harness**: TRUTH_CHAIN_COMPUTE
- **Metrics**: PnL, MaxDD, trade-rate, per-session contribution

### Outputs
- Decision surface: `operating_region_scan.json`
- Top-K configs: `topk_summary.md` + `decision_packet.json`

### Timeline
- **Per config**: ~10-15 min
- **Total configs**: ~20-30 (thresholds × cost levels)
- **Total**: ~2-4 hours

---

## Execution Plan

### Step 1: Preflight Checks
```bash
# Verify TRUTH mode
export GX1_RUN_MODE=TRUTH
export GX1_TRUTH_MODE=1

# Verify prebuilt exists
test -f "$GX1_DATA/data/data/prebuilt/TRIAL160/2025/xauusd_m5_2025_features_v10_ctx.parquet"
test -f "$GX1_DATA/data/data/prebuilt/TRIAL160/2024/xauusd_m5_2024_features_v10_ctx.parquet"

# Verify baseline bundle
test -d "$GX1_DATA/models/models/entry_v10_ctx/FULLYEAR_2024_2025_V12AB91"

# Verify XGB session policy
test -f "$WORKSPACE_ROOT/gx1/configs/xgb_session_policy.json"
```

### Step 2: Phase A Execution
```bash
gx1/scripts/run_phase_a.sh \
    --sessions EU OVERLAP US ASIA \
    --years 2024 2025 \
    --n-trials-per-session 150 \
    --n-jobs 12 \
    --output-dir "$GX1_DATA/reports/research_72h/$RUN_TS/phase_a" \
    --optuna-db-dir "$GX1_DATA/optuna"
```

### Step 3: Phase B Execution
```bash
gx1/scripts/run_phase_b.sh \
    --depths 3 4 6 \
    --best-xgb-dir "$GX1_DATA/reports/research_72h/$RUN_TS/phase_a/top_candidates" \
    --output-dir "$GX1_DATA/reports/research_72h/$RUN_TS/phase_b" \
    --device cuda
```

### Step 4: Phase C Execution
```bash
gx1/scripts/run_phase_c.sh \
    --best-xgb-dir "$GX1_DATA/reports/research_72h/$RUN_TS/phase_a/top_candidates" \
    --best-transformer-dir "$GX1_DATA/reports/research_72h/$RUN_TS/phase_b/best_depth" \
    --thresholds 0.56 0.60 0.64 0.68 \
    --output-dir "$GX1_DATA/reports/research_72h/$RUN_TS/phase_c"
```

### Step 5: Generate Reports
```bash
/home/andre2/venvs/gx1/bin/python gx1/scripts/research/generate_topk_report.py \
    --research-dir "$GX1_DATA/reports/research_72h/$RUN_TS" \
    --output-dir "$GX1_DATA/reports/research_72h/$RUN_TS"
```

---

## Hard Gates (Invariants)

### TRUTH Mode Required
- All evaluations must run with `GX1_RUN_MODE=TRUTH` or `GX1_TRUTH_MODE=1`
- Missing policy/bundle/prebuilt/payoff → hard-fail (exit non-zero)

### Prebuilt-Only
- `GX1_FEATURE_BUILD_DISABLED=1`
- `GX1_REPLAY_USE_PREBUILT_FEATURES=1`

### Identity Gates
- XGB session policy must be present and hashed into RUN_IDENTITY
- Bundle SHA256 must match RUN_IDENTITY
- Prebuilt identity gates enforced

### Payoff Panel Required
- If exits exist, PAYOFF_PANEL must be generated
- Fatal if missing in TRUTH mode

### Candidate Artifacts
Each candidate must include:
- Model SHA256
- Training config JSON
- Dataset fingerprint (prebuilt hash + window)
- Session/head coverage

---

## Safety Checks

### Cost/Slippage Sensitivity
- Evaluate with cost=0, realistic, stress(2x)
- Reject candidates with unrealistic PnL under stress

### Sanity Checks
- **Tails**: P95 MAE/MFE within bounds
- **Concentration**: No single session >90% of trades
- **Session dominance**: Reject if one session dominates unrealistically

### Stability Checks
- **Drift**: KS/PSI metrics within thresholds
- **Score distribution**: p95 not near 0.5, variance not near 0
- **Reproducibility**: Same inputs → same outputs (determinism)

---

## Resource Allocation

### CPU (Phase A)
- **Workers**: 12-16 (XGB training is CPU-bound)
- **Memory**: ~8-16 GB per worker
- **Storage**: ~10-20 GB for Optuna DBs + candidate models

### GPU (Phase B)
- **Device**: CUDA (single GPU sufficient)
- **Memory**: ~8-12 GB VRAM
- **Storage**: ~5-10 GB for checkpoints

### Storage (Total)
- **Optuna DBs**: ~1-2 GB
- **XGB candidates**: ~500 MB - 1 GB
- **Transformer checkpoints**: ~200-500 MB each
- **Evaluation outputs**: ~100-200 MB per config
- **Total**: ~20-30 GB

---

## Timeline Summary

| Phase | Duration | Resource | Parallelism |
|-------|----------|----------|-------------|
| **A: XGB Optuna** | 24-36h | CPU | 12-16 workers |
| **B: Transformer Depth** | 12-24h | GPU | Sequential |
| **C: Operating Region** | 2-4h | CPU | 4-8 workers |
| **Reporting** | 0.5h | CPU | Sequential |
| **Total** | **~40-65h** | Mixed | - |

---

## Deliverables

1. **Top-K Summary** (`TOPK_SUMMARY.md`):
   - Top 10 candidates (model + threshold configs)
   - Metrics table (PnL, DD, trade-rate, winrate)
   - Per-session breakdown
   - Stability notes (tails, concentration, drift)

2. **Decision Packet** (`DECISION_PACKET.json`):
   - Machine-readable results
   - All candidate metadata
   - Evaluation artifacts paths

3. **Research Artifacts**:
   - Phase A: Optuna studies + top candidates
   - Phase B: Transformer checkpoints
   - Phase C: Operating region scan results

---

## STOP THE BLEED Mechanism

If any of the following occur, the orchestration **immediately exits non-zero**:

1. **Invariant breach**: Missing policy/bundle/prebuilt in TRUTH mode
2. **Missing metrics**: PAYOFF_PANEL missing when exits exist
3. **Contract violation**: Feature contract mismatch
4. **Determinism failure**: Same inputs → different outputs
5. **Sanity check failure**: Unrealistic metrics (e.g., PnL >1000 bps, winrate >99%)

---

## Next Steps

1. Implement `run_phase_a_xgb_optuna.py`
2. Implement `run_phase_b_transformer_depth_ladder.py`
3. Implement `run_phase_c_operating_region_scan.py`
4. Implement `run_72h_tuning.sh` (orchestrator)
5. Implement `generate_topk_report.py` (reporting)
