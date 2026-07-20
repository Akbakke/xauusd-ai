# GX1 operating rules

`AGENTS.md` is the current operational constitution. `SYSTEM_MAP.md` is the
current architecture map. `HANDOVER_XAU_DIRECTION_REPAIR_20260714.md` records
the current work state. If these disagree, fail closed and repair the
documentation and code together.

## Non-negotiable rules

1. Trade XAUUSD only. Cross-asset data may be read only as model features and
   must never become another traded output.
2. No fallback, guessed default, mutable `latest`, stale artifact, synthetic
   decision input or soft pass-through is allowed.
3. Entry direction comes only from the accepted model's calibrated
   `LONG/SHORT/FLAT` logits. No post-model trend, session, confidence, utility
   or threshold rule may veto, flip or manufacture direction.
4. Keep every genuine feature family in the learned path. Removing a retired
   rule must never remove its underlying market evidence. The 305 registered
   causal-layer outputs are mandatory; only 174 additional specialist fields
   may be selected by deterministic TRAIN-only ranking.
5. The learned size head is mandatory evidence. Label-horizon sizing proof is
   diagnostic only; paper/live additionally requires an exact joint adopted-
   Exit sizing replay and fresh post-adoption broker runtime parity. Until both
   pass, emit no order. Fixed size is not a fallback.
6. Train equals serve: exact ordered fields, dimensions, normalization,
   timeframe construction, hashes and final-logit semantics must match.
7. Newest valid terminal evidence wins. A newer red event blocks every older
   green event. Missing or malformed evidence is red.
8. Every Entry train/rebuild needs one immutable `--run-id`, immutable inputs
   and a clean, resource-safe execution plan. The ID is lineage, not manual
   approval; evidence contracts alone admit execution. Never auto-promote an
   artifact.
9. Do not delete anything under `/home/andre2/GX1_DATA` or active run paths
   without an explicit verified cleanup decision. Preserve active collectors,
   canonical builders, dashboards and their files.
10. Remove disconnected repository code and stale docs once call-site scans,
    tests and evidence ownership show they are unnecessary.
11. Never expose secrets, force-push, hard-reset shared work or overwrite
    unrelated working-tree changes.
12. Finish every change with focused tests, syntax checks, stale-path scans,
    `git diff --check` and an honest statement of what remains unproved.
