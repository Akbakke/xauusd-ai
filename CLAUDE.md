# GX1 operating rules

`AGENTS.md` is the current operational constitution. `SYSTEM_MAP.md` is the
current architecture map. `HANDOVER_XAU_DIRECTION_REPAIR_20260714.md` records
the current work state. If these disagree, fail closed and repair the
documentation and code together.

## Non-negotiable rules

1. Trade and model XAUUSD only. Entry contracts must not depend on market data
   from another instrument or expose another traded output.
2. No fallback, guessed default, mutable `latest`, stale artifact, synthetic
   decision input or soft pass-through is allowed.
3. Entry direction comes only from the accepted model's calibrated
   `LONG/SHORT/FLAT` logits. No post-model trend, session, confidence, utility
   or threshold rule may veto, flip or manufacture direction.
4. Keep every genuine feature family in the learned path. Removing a retired
   rule must never remove its underlying market evidence. The 378 registered
   causal-layer outputs are mandatory; only 101 additional specialist fields
   may be selected by deterministic TRAIN-only ranking.
5. The learned size head is mandatory evidence. Label-horizon sizing proof is
   diagnostic only; paper/live additionally requires an exact joint adopted-
   Exit sizing replay and fresh post-adoption broker runtime parity. Until both
   pass, emit no order. Fixed size is not a fallback.
6. Train equals serve: exact ordered fields, dimensions, normalization,
   timeframe construction, hashes and final-logit semantics must match.
7. Newest valid terminal evidence wins. A newer red event blocks every older
   green event. Missing or malformed evidence is red. A GREEN dataset admits
   only those exact bytes to the next evidence gate; it does not admit a model,
   direction, bundle or launch. Keep that distinction explicit in both
   Markdown and `PROJECT_STATE_xau_direction_launch.json`.
8. Every Entry rebuild needs one immutable dataset-build `--run-id`. Every
   train has a distinct output `--run-id` plus a launch-derived
   `dataset_run_id` that must match post-rebuild and all split manifests; it is
   not a caller override. IDs are lineage, not manual approval; evidence
   contracts alone admit execution. Never auto-promote an artifact.
9. Do not delete anything under `/home/andre2/GX1_DATA` or active run paths
   without an explicit verified cleanup decision. Preserve active collectors,
   canonical builders, dashboards and their files.
10. Remove disconnected repository code and stale docs once call-site scans,
    tests and evidence ownership show they are unnecessary.
11. Never expose secrets, force-push, hard-reset shared work or overwrite
    unrelated working-tree changes.
12. Finish every change with focused tests, syntax checks, stale-path scans,
    `git diff --check` and an honest statement of what remains unproved.
13. Source-wiring audits must prove import and executable use of the exact
    contract owner. Repeating the expected mode, dimension or field literal in
    a consumer is not ownership proof and must never be required as one.
14. Model-native training may receive decision-affecting environment values
    only from the canonical exact recipe owner. The immutable recipe must bind
    all 162 keys, split artifacts, prerequisite audits and executable source
    bytes; ambient values, wrapper defaults and hand-authored recipe evidence
    are forbidden.
15. Dataset-build and training-output identities are separate roles. Recipe,
    wrapper, trainer, bundle metadata/lock and handover must bind both; missing,
    collapsed or split-brain lineage fails closed.
16. Forward-outcome target domains are exact. Spread-aware MFE and path quality
    remain signed through validation and both train/validation losses; MAE
    remains a non-negative adverse magnitude. Clipping, taking absolute values
    or substituting parked zeros is a forbidden target rewrite.
17. Active head liveness checks must require the exact batch keys emitted by
    the canonical Dataset mapping. `y_direction` is converted once to class
    tensor `y`; adding aliases, defaults or duplicated targets to satisfy a
    head check is forbidden.
