# Cleanup Deletion Manifest - 2026-07-02

Status: applied as repo hygiene only. This cleanup does not train, replay,
distill, stage, shadow, live, promote or change active model contracts.

## Active Keep Rules

Do not delete these during historical cleanup:

- `docs/ACTIVE_SUPER_AI_BOT_GOAL_20260702.md`
- `docs/ENTRY_FOUNDATION_AUDIT_20260628.md`
- `docs/ENTRY_SEQUENTIAL_AI_SPECIALIST_BLUEPRINT_20260628.md`
- `docs/ENTRY_SEQ215_SPECIALIST_INPUT_CONTRACT_20260630.md`
- `AGENTS.md`, `CLAUDE.md`, `SYSTEM_MAP.md`, `PROJECT_STATE.md`,
  `DECISION_LOG.md`
- `docs/ENTRY_NEXT_EDGE_PLAN_20260627.md`
- `docs/ENTRY_NEXT_EDGE_SHADOW_REVIEW_TEMPLATE_20260627.md`
- active guardrail/tombstone docs carrying the 2026-07-02 active override
- code, tests, scripts, data contracts, manifests, model artifacts and active
  gate reports

## Deleted Batch

This batch removes 144 tracked files:

- 86 old SNIPER 2025 generated OOS, quarter metrics, delta and backtest docs
- 11 old PREBUILT mapping/status docs that are not active control surfaces
- 17 old FASE1/PHASE/STEP/FULLYEAR/BACKFILL status docs
- 7 top-level historical status or incident notes
- 3 generated non-doc artifacts: one accidental root tmp file that looked like
  a restore command and two parity audit CSV sidecars
- 20 additional historical cleanup, preflight, replay, worktree inventory,
  tuning and status docs

The exact deletion list is the `git diff --name-status` for this commit. Git
history remains the archive for the removed historical reports.

## Intentionally Kept For Review

The cleanup kept older-looking files when they are referenced by active code,
tests, readiness/control surfaces or surviving documentation. In particular:

- frozen SNIPER anchors:
  `docs/SNIPER_2025_FULLYEAR_REPORT__20251218.md`,
  `docs/SNIPER_2025_ANALYSIS_FROZEN.md`,
  `docs/SNIPER_2025_OOS_SUMMARY__baseline_20251218_145523.md`,
  `docs/SNIPER_2025_OOS_SUMMARY__guarded_20251218_145523.md`,
  `docs/SNIPER_2025_DELTA_BASELINE_vs_GUARDED_20251218_145523.md`
- legacy or contract-shaped docs such as `docs/RUNBOOK.md`,
  `docs/FEATURE_MANIFEST.md`, `docs/ENTRY_CONTEXT_FEATURES_CONTRACT.md`,
  `docs/ENTRY_TELEMETRY_CONTRACT_SNIPER.md`,
  `docs/XGB_CALIBRATION_CONTRACT.md` and
  `docs/V10_CTX_FEATURE_CONTRACT_ANALYSIS.md`
- legacy snapshots under `gx1/legacy/_legacy_disabled/`

Any future deletion from the kept set should update references in the same
commit and rerun the same fail-closed verification gates.
