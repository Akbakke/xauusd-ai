# Proof v1 SSoT (Single Source of Truth)

**Frozen:** 2026-02-03  
**Status:** CONDITIONAL_GO (observability-gap, ikke signal)

## Frozen Artifacts

### Prebuilt Features
- **Identifier:** `v12ab_clean_2025`
- **Path:** `/home/andre2/GX1_DATA/data/data/prebuilt/TRIAL160/2025/xauusd_m5_2025_features_v10_ctx.parquet`
- **Schema:** V12AB91 (91 features, no `prob_*` columns)
- **Build invariant:** Hard-fail if any column starts with `prob_`

### XGB Model Bundle
- **Identifier:** `FULLYEAR_2024_2025_V12AB91`
- **Path:** `/home/andre2/GX1_DATA/models/models/entry_v10_ctx/FULLYEAR_2024_2025_V12AB91/`
- **Model:** `xgb_universal_multihead_v2.joblib`
- **Lock:** `MASTER_MODEL_LOCK.json` (SHA256: c8b80f61ccecab41c9d9eccfdb4900372c492441c4ef3d3c01b7a4328b2ad161)
- **Features:** 91 (V12AB91), no `prob_*`, no `_v1_*`
- **Sessions:** EU, OVERLAP

### TRUTH Replay Run
- **Run ID:** `PROOF_2025_TRUTH_V12AB91_FPRINT_20260203_113032`
- **Path:** `/home/andre2/GX1_DATA/reports/replay_eval/PROOF_2025_TRUTH_V12AB91_FPRINT_20260203_113032/`
- **Date range:** 2025-01-01 to 2025-01-08
- **Fingerprint logging:** Active (`GX1_XGB_INPUT_FINGERPRINT=1`)
- **Status:** `RUN_COMPLETED`

### Truth Report Artifacts
- **Path:** `/home/andre2/GX1_DATA/reports/xgb_transformer_truth/`
- **Truth Report:** `XGB_TRANSFORMER_TRUTH_unknown.json/md`
- **GO/NO-GO:** `XGB_TRANSFORMER_GO_NOGO_unknown.json/md`
- **Decision:** CONDITIONAL_GO

## Verification Metrics (EU/OVERLAP)

### EU Session
- **Correlation (tf prob_long vs xgb p_long_raw):** 0.4124 (n=60)
- **Correlation (tf confidence vs xgb uncertainty):** -0.0989 (n=60)
- **Disagreement rate (sign(p-0.5)):** 0.0 (0/60)

### OVERLAP Session
- **Correlation (tf prob_long vs xgb p_long_raw):** 0.3051 (n=40)
- **Correlation (tf confidence vs xgb uncertainty):** 0.4868 (n=40)
- **Disagreement rate (sign(p-0.5)):** 0.0 (0/40)

## Notes

- **Observability gap:** `ENTRY_FEATURES_USED.json` does not contain `xgb_flows` (by design, not a signal issue)
- **Correlation computation:** Performed via ts+session join between `transformer_outputs` and `XGB_INPUT_FINGERPRINT.*.jsonl`
- **Next phase:** Model/edge work (no plumbing changes)

## Canonical Python Environment

- **Path:** `/home/andre2/venvs/gx1/bin/python`
- **Verified by:** `gx1/utils/env_identity_gate.py` (TRUTH/SMOKE mode)
- **Runner:** **ALWAYS run via `gx1/scripts/run_replay_canonical.sh`** to ensure canonical Python is used

---

*This SSoT snapshot is frozen and serves as the baseline for subsequent model/edge development.*
