#!/usr/bin/env bash
# Retire the superseded baseline datasets V28 (513 surface) and V29J (592),
# per the operator decision of 2026-08-14. Plan -> approve -> execute, all
# through the retention owner, all in one process so no authority byte can
# move between the stages.
set -euo pipefail
cd /home/andre2/src/GX1_ENGINE
P=/home/andre2/GX1_DATA/data/data/prebuilt
R=/home/andre2/GX1_DATA/reports
M=gx1.scripts.cleanup_gx1_evidence_v1

REASON="The V30 fidelity surface (2026-08-13, derived 538 = 34 + 371 + 133 over 16 families) supersedes both the 513 surface V28 was built on and the 592 surface V29J was built on; both are forbidden as substrate for new training. Measured 2026-08-14: no model, bundle, calibration event or metric was ever derived from either dataset - V28's only external referrer is the 2026-08-11 logit-adjustment smoke, which was itself killed at batch 211 without publishing a bundle, and V29J's are the V29 smoke-ladder and V30 step-0 measurement logs, which are records of measurements already written down. Reachability proof: no manifest anywhere under GX1_DATA binds either path; every external referrer is a log or report, not a data-to-data lineage binding, and neither is a parent root of any successor. No active process holds either. Their documentary role as the frozen comparison baseline was unexecutable - producing that arm requires training on a forbidden surface - and PROJECT_STATE, GX1_RULES.md and AGENTS.md were repaired in commit ced766a6 to name the coin-flip null (-13.16 bps TRAIN / -18.58 bps VAL) as the evaluation reference instead."

VEDTAK=GX1_V30_RETIRE_SUPERSEDED_BASELINE_DATASETS_20260814

echo "=== PLAN ==="
PLAN_OUT=$(scripts/gx1_capped_run.sh --class audit -- \
  .venv/bin/python -m $M plan \
  --target "$P/XAU_ENTRY_EXIT_M15_20260809_V28" \
  --target "$P/XAU_ENTRY_EXIT_M15_20260811_V29J" \
  --reason "$REASON" --vedtak "$VEDTAK")
PLAN_JSON=$(sed -n 's/.*"plan_json": "\([^"]*\)".*/\1/p' <<<"$PLAN_OUT" | tail -1)
PLAN_SHA=$(sed -n 's/.*"plan_sha256": "\([^"]*\)".*/\1/p' <<<"$PLAN_OUT" | tail -1)
echo "plan_json=$PLAN_JSON"
echo "plan_sha256=$PLAN_SHA"
test -n "$PLAN_JSON" && test -n "$PLAN_SHA"

echo "=== APPROVE ==="
APP_OUT=$(scripts/gx1_capped_run.sh --class audit -- \
  .venv/bin/python -m $M approve \
  --plan-json "$PLAN_JSON" --plan-sha256 "$PLAN_SHA" --vedtak "$VEDTAK" \
  --approved-by "andre2 (explicit operator decision 2026-08-14: slett begge, 106 GB)" \
  --approve)
APP_JSON=$(sed -n 's/.*"approval_json": "\([^"]*\)".*/\1/p' <<<"$APP_OUT" | tail -1)
APP_SHA=$(sed -n 's/.*"approval_sha256": "\([^"]*\)".*/\1/p' <<<"$APP_OUT" | tail -1)
echo "approval_json=$APP_JSON"
test -n "$APP_JSON" && test -n "$APP_SHA"

echo "=== EXECUTE ==="
scripts/gx1_capped_run.sh --class audit -- \
  .venv/bin/python -m $M execute \
  --plan-json "$PLAN_JSON" --plan-sha256 "$PLAN_SHA" \
  --approval-json "$APP_JSON" --approval-sha256 "$APP_SHA" \
  --vedtak "$VEDTAK" --execute
