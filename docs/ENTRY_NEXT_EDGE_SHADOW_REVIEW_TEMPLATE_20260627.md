# Entry Next-Edge Shadow Review Template 2026-06-27

Use this after `scripts/entry_next_edge_control.sh verify-shadow` has produced
`shadow_telemetry_verification.json`.

This is a manual decision gate. A technical PASS from `verify-shadow` is not a
promotion, paper-trading, or live-trading approval.

## Required Inputs

```text
plan_doc=docs/ENTRY_NEXT_EDGE_PLAN_20260627.md
telemetry_report=<shadow_telemetry_verification.json>
shadow_journal=<v12_paper_journal_<UTC_YYYYMMDD>_noxgb_shadow.jsonl>
run_context=/home/andre2/GX1_DATA/reports/entry_tabular_no_xgb_candidates/entry_tabular_no_xgb_top5_v1_20260627/live_shadow_runner_preflight/shadow_only_run_context.json
candidate_manifest=/home/andre2/GX1_DATA/reports/entry_tabular_no_xgb_candidates/entry_tabular_no_xgb_top5_v1_20260627/candidate_manifest.json
```

## Hard Rejects

Reject the shadow run and do not promote if any item is true:

```text
order_side_effects_detected=true
shadow_disabled_reason_present=true
feature_contract_hash_mismatch=true
candidate_id_mismatch=true
missing_required_shadow_fields=true
runner_started_outside_canonical_control=true
live_order_placement_not_NOT_STARTED_NOT_ENABLED=true
```

## Review Checks

Record the observed values:

```text
shadow_rows=<int>
would_take_rate=<float>
offline_expected_would_take_rate=0.05921197446621176
would_take_long_rows=<int>
would_take_short_rows=<int>
score_threshold=0.39048198845884335
score_p95=<float>
score_p99=<float>
order_status_counts=<json>
git_status_short_entries=<int>
```

Explain deviations before any next step:

```text
candidate_rate_explanation=<required if materially different from offline>
side_skew_explanation=<required if long/short distribution is materially skewed>
session_skew_explanation=<required if one session dominates>
dirty_worktree_explanation=<required if git_status_short_entries > 0>
```

## Decision

Choose exactly one:

```text
decision=ACCEPT_FOR_NEXT_REVIEW_GATE
meaning=technical telemetry is coherent; still no production pin

decision=HOLD_FOR_MORE_SHADOW
meaning=telemetry is technically clean but sample is too small or distribution is unclear

decision=FAIL_TO_FEATURE_LABEL_OBJECTIVE_REDESIGN
meaning=do not tune accuracy or restart XGB/ET; redesign quantified features, labels, or objective
```

## Non-Negotiables

```text
no_live_promotion_from_shadow_review_alone=true
do_not_optimize_full_bar_accuracy=true
xgb_reference_only=true
no_new_model_dependency_before_shadow_review=true
```
