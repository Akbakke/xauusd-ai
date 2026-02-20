# Session Gate Root-Cause (Truth)

Purpose: Provide a canonical, replay-only method to diagnose why EU/OVERLAP sessions
produce zero entry evaluations, and to select the minimal diagnostic bypass gate.

## Inputs
- `ENTRY_FEATURES_USED.json` (run-level, master)
- `RUN_IDENTITY.json`

## Output
Run-level reports under the replay output directory:
- `SESSION_FUNNEL_ROOT_CAUSE_<run_id>.json`
- `SESSION_FUNNEL_ROOT_CAUSE_<run_id>.md`

## How to Run
```bash
python3 gx1/scripts/build_session_funnel_root_cause_report.py \
  --existing-output-dir /path/to/replay_output_dir
```

## What It Reports
1) Bars per session (ASIA/EU/OVERLAP/US)
2) Funnel per session:
   - post_warmup_count
   - pregate_pass_count / pregate_block_count
   - eval_called_count
   - predict_entered_count
   - pre_call_count
   - transformer_forward_calls
   - exceptions_count
3) Block reasons per session:
   - eligibility_blocks
   - session_blocks
   - vol_regime_blocks
   - score_blocks
   - cost_blocks
   - stage2_block_reasons / stage3_block_reasons / pre_model_return_reasons (if present)
4) First Gate Kill per session:
   - `first_kill_stage`
   - `likely_reason`

## Diagnostic Override (Replay-Only)
Purpose: Force EU/OVERLAP to reach eval stage without altering production logic.

Env toggles:
- `GX1_DIAGNOSTIC_FORCE_EVAL_SESSIONS="EU,OVERLAP"`
- `GX1_DIAGNOSTIC_BYPASS_GATE="<gate>"`

Gate options:
- `hard_eligibility`
- `soft_eligibility`
- `pregate`
- `vol_guard`
- `score_gate`

Rules:
- Only active when `GX1_TRUTH_TELEMETRY=1`
- Only affects sessions in `GX1_DIAGNOSTIC_FORCE_EVAL_SESSIONS`
- Bypasses exactly one gate (the first kill identified by the report)
- Records bypass in RUN_IDENTITY and ENTRY_FEATURES_USED.json

## Contract Notes
- This is a diagnostic, replay-only feature.
- It must be disabled for production runs.
- It is used only to obtain proof that EU/OVERLAP can reach eval stage.
