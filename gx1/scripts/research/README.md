# 72-Hour Tuning Research Suite

**Status**: ✅ Complete and TRUTH SAFE  
**Last Updated**: 2026-02-04

## Quick Start

```bash
# Set environment
export GX1_DATA=/path/to/GX1_DATA
export GX1_RUN_MODE=TRUTH
export GX1_TRUTH_MODE=1

# Run full 72-hour tuning orchestration
bash gx1/scripts/research/run_72h_tuning.sh
```

## Architecture

See `plan_72h_tuning.md` for full architecture diagram, execution plan, and resource allocation.

## Files

| File | Purpose |
|------|---------|
| `plan_72h_tuning.md` | Full plan with architecture, timelines, and safety gates |
| `run_72h_tuning.sh` | Main orchestrator (runs all phases sequentially) |
| `run_phase_a_xgb_optuna.py` | Phase A: XGB Optuna tuning per session |
| `run_phase_b_transformer_depth_ladder.py` | Phase B: Transformer depth ladder (3, 4, 6 layers) |
| `run_phase_c_operating_region_scan.py` | Phase C: Threshold + cost sensitivity sweep |
| `generate_topk_report.py` | Generate TOPK_SUMMARY.md and DECISION_PACKET.json |

## Phase Execution

### Phase A: XGB Optuna (24-36 hours, CPU-heavy)

```bash
gx1/scripts/run_phase_a.sh \
    --sessions EU OVERLAP US ASIA \
    --years 2024 2025 \
    --n-trials-per-session 150 \
    --n-jobs 12 \
    --output-dir "$GX1_DATA/reports/research_72h/$RUN_TS/phase_a" \
    --optuna-db-dir "$GX1_DATA/optuna" \
    --baseline-bundle-dir "$GX1_DATA/models/models/entry_v10_ctx/FULLYEAR_2024_2025_V12AB91"
```

**Outputs**:
- Optuna SQLite DBs: `$GX1_DATA/optuna/xgb_phase_a_<session>_<ts>.db`
- Top candidates: `phase_a/candidates/<session>/xgb_<session>_trial_<n>.joblib`
- Summary: `phase_a/phase_a_summary.json`

### Phase B: Transformer Depth Ladder (12-24 hours, GPU-heavy)

```bash
gx1/scripts/run_phase_b.sh \
    --depths 3 4 6 \
    --best-xgb-dir "$GX1_DATA/reports/research_72h/$RUN_TS/phase_a/top_candidates" \
    --output-dir "$GX1_DATA/reports/research_72h/$RUN_TS/phase_b" \
    --device cuda \
    --baseline-bundle-dir "$GX1_DATA/models/models/entry_v10_ctx/FULLYEAR_2024_2025_V12AB91"
```

**Outputs**:
- Evaluation results per depth: `phase_b/eval_depth_<n>_<window>/`
- Summary: `phase_b/phase_b_summary.json`

### Phase C: Operating Region Scan (2-4 hours, CPU)

```bash
gx1/scripts/run_phase_c.sh \
    --best-xgb-dir "$GX1_DATA/reports/research_72h/$RUN_TS/phase_a/top_candidates" \
    --best-transformer-dir "$GX1_DATA/reports/research_72h/$RUN_TS/phase_b/best_depth" \
    --thresholds 0.56 0.60 0.64 0.68 \
    --output-dir "$GX1_DATA/reports/research_72h/$RUN_TS/phase_c"
```

**Outputs**:
- Decision surface: `phase_c/operating_region_scan.json`
- Per-config evaluations: `phase_c/thr_<n>_cost_<m>/`

## Reporting

After all phases complete, reports are automatically generated:

- **`TOPK_SUMMARY.md`**: Human-readable summary with top 10 candidates
- **`DECISION_PACKET.json`**: Machine-readable results for downstream analysis

## TRUTH Safety Gates

All phases enforce:

✅ **TRUTH mode required** - Hard-fail if `GX1_RUN_MODE != TRUTH`  
✅ **Prebuilt-only** - `GX1_FEATURE_BUILD_DISABLED=1` enforced  
✅ **XGB session policy** - Must exist and be hashed into RUN_IDENTITY  
✅ **Payoff panel required** - Fatal if missing when exits exist  
✅ **Identity gates** - Bundle SHA, policy hash, prebuilt fingerprint verified  

## Perf harness (TRUTH 2025 gated-parallel)

Goal: a deterministic throughput harness for canonical TRUTH (`run_truth_e2e_sanity`) that:
- Uses **PREBUILT only** (`GX1_FEATURE_BUILD_DISABLED=1`)
- Produces run-root `PERF_SUMMARY.{json,md}` and per-window `PERF_SUMMARY.{json,md}`
- Surfaces **dominant cost** from `RUN_COMPLETED.json.stage_timings_s`
- Records **wiring proof counters** (`xgb_flows`, `transformer_forward_calls`) from `ENTRY_FEATURES_USED_MASTER.json`
- Records **exit journaling counters** from `EXIT_COVERAGE_SUMMARY.json`

### Usage

```bash
export GX1_DATA=/home/andre2/GX1_DATA
export GX1_RUN_MODE=TRUTH
export GX1_TRUTH_MODE=1
export GX1_GATED_FUSION_ENABLED=1
export GX1_REPLAY_USE_PREBUILT_FEATURES=1
export GX1_FEATURE_BUILD_DISABLED=1
export GX1_REPLAY_ACCOUNTING_CLOSE_AT_END=0
export GX1_REPLAY_EXIT_SEMANTICS=CALENDAR_TRUE
export GX1_SEED=42
export GX1_CANONICAL_BUNDLE_DIR="$GX1_DATA/models/models/entry_v10_ctx/FULLYEAR_2024_2025_V12AB91"
export GX1_CANONICAL_POLICY_PATH="$GX1_DATA/configs/policies/canonical/TRUTH_BASELINE_V12AB91.yaml"

# Optional: enable replay cProfile (if implemented in replay master)
export GX1_PERF_PROFILE=0

/home/andre2/venvs/gx1/bin/python gx1/scripts/research/run_perf_harness_2025_gated_parallel.py --windows W2
```

### Outputs

Run root:
- `$GX1_DATA/reports/perf_harness/<RUN_TS>/PERF_SUMMARY.json`
- `$GX1_DATA/reports/perf_harness/<RUN_TS>/PERF_SUMMARY.md`

Per-window:
- `$GX1_DATA/reports/perf_harness/<RUN_TS>/<Wn>/PERF_SUMMARY.json`
- `$GX1_DATA/reports/perf_harness/<RUN_TS>/<Wn>/PERF_SUMMARY.md`

## Determinism

- Fixed seeds (42) across all phases
- Deterministic windows (2024-2025)
- Immutable artifacts (bundle SHA, policy hash)
- Reproducible results (same inputs → same outputs)

## Safety Checks

- **Cost sensitivity**: Evaluated at cost=0, realistic, stress(2x)
- **Sanity checks**: Reject unrealistic metrics (PnL >1000 bps, winrate >99%)
- **Stability**: Tails (MAE/MFE P95), concentration, session dominance
- **Drift**: KS/PSI metrics within thresholds

## Resource Requirements

| Phase | Duration | CPU | GPU | Memory | Storage |
|-------|----------|-----|-----|--------|---------|
| A: XGB Optuna | 24-36h | 12-16 cores | - | 8-16 GB/worker | ~10-20 GB |
| B: Transformer Depth | 12-24h | 2-4 cores | 1 GPU | 8-12 GB VRAM | ~5-10 GB |
| C: Operating Region | 2-4h | 4-8 cores | - | 4-8 GB | ~5-10 GB |
| **Total** | **~40-65h** | Mixed | 1 GPU | - | **~20-30 GB** |

## Output Structure

```
$GX1_DATA/reports/research_72h/$RUN_TS/
├── phase_a/
│   ├── phase_a_summary.json
│   ├── candidates/
│   │   ├── EU/
│   │   ├── OVERLAP/
│   │   ├── US/
│   │   └── ASIA/
│   └── phase_a.log
├── phase_b/
│   ├── phase_b_summary.json
│   ├── eval_depth_3_2025/
│   ├── eval_depth_4_2025/
│   └── eval_depth_6_2025/
├── phase_c/
│   ├── operating_region_scan.json
│   ├── thr_0.56_cost_0.0/
│   ├── thr_0.56_cost_1.0/
│   └── ...
├── TOPK_SUMMARY.md
├── DECISION_PACKET.json
└── reporting.log
```

## Troubleshooting

### Phase A fails with "Prebuilt not found"
- Verify: `$GX1_DATA/data/data/prebuilt/TRIAL160/2024/` and `2025/` exist
- Check: `GX1_DATA` environment variable is set correctly

### Phase B fails with "Baseline bundle not found"
- Verify: `$GX1_DATA/models/models/entry_v10_ctx/FULLYEAR_2024_2025_V12AB91/` exists
- Check: Bundle contains `model_state_dict.pt` and `bundle_metadata.json`

### TRUTH mode violations
- Ensure: `GX1_RUN_MODE=TRUTH` and `GX1_TRUTH_MODE=1`
- Verify: `gx1/configs/xgb_session_policy.json` exists
- Check: All prebuilt files are present and valid

### Evaluation timeouts
- Increase timeout in subprocess calls (default: 3600s for Phase A, 7200s for Phase B/C)
- Reduce window size for faster evaluation (e.g., Q1 2025 only)

### Phase A flight recorder (MASTER_EARLY.json)
If Phase A appears stuck or exits unexpectedly, inspect the heartbeat flight recorder:

```bash
OUT="$GX1_DATA/reports/research_72h/<RUN_TS>/phase_a"
/home/andre2/venvs/gx1/bin/python -c "import json; print(json.dumps(json.load(open('$OUT/MASTER_EARLY.json')), indent=2))"
```

## Next Steps

1. **Review plan**: Read `plan_72h_tuning.md` for full details
2. **Preflight check**: Run orchestrator with `--dry-run` (if implemented) or verify paths manually
3. **Start tuning**: `bash gx1/scripts/research/run_72h_tuning.sh`
4. **Monitor progress**: Check phase logs in `$GX1_DATA/reports/research_72h/$RUN_TS/`
5. **Review results**: Read `TOPK_SUMMARY.md` and `DECISION_PACKET.json`

## Notes

- Phase A uses fast-path pruning initially; full TRUTH evaluation is optional (`--skip-full-eval`)
- Phase B currently evaluates baseline bundle as proxy; full implementation would train new bundles per depth
- All scripts include `[RUN_CTX]` headers with provenance (git SHA, GX1_DATA, TRUTH flags)
- Orchestrator includes "STOP THE BLEED" mechanism: exits non-zero on any invariant breach

## Feature Metadata Schema

The `feature_metadata.json` file is required in bundle directories for evaluation. If missing, it is auto-generated from SSoT (`gx1/features/feature_contract_v10_ctx.py`).

**Minimal required fields**:
- `base_seq_dim`: int - Base sequence features count (13)
- `base_snap_dim`: int - Base snapshot features count (84)
- `snap_xgb_channels_names`: List[str] - XGB channel names in SNAP (e.g., `["p_long_xgb", "p_hat_xgb", ...]`)
- `snap_xgb_start`: int - Start index of XGB channels in SNAP array (84)
- `contract_sha256`: str - SHA256 hash of `feature_contract_v10_ctx.py` (for validation)

**Invariant**: `contract_sha256` must match the current `feature_contract_v10_ctx.py` hash. If mismatch is detected, a warning is logged.

**TRUTH behavior**: If generation is impossible (missing contract file, import failure), hard-fails with `RuntimeError`.

**DEV behavior**: Auto-generates and logs a warning that the bundle was patched for evaluation.
