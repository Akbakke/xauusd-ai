# ACTIVE ENTRY/EXIT SUPER AI BOT OVERRIDE - 2026-07-02

The current Entry/Exit AI workstream is governed by
`docs/ACTIVE_SUPER_AI_BOT_GOAL_20260702.md`,
`docs/ENTRY_FOUNDATION_AUDIT_20260628.md` and
`docs/ENTRY_SEQUENTIAL_AI_SPECIALIST_BLUEPRINT_20260628.md`. The older
2026-06-27 no-XGB shadow plan and older live-practice instructions are
historical evidence, not the active operating point.

Active goal: build one replay-proven XAUUSD policy where all Entry and Exit
inputs cooperate as a shared market-state language. `smart_seq520_candidate` is
structurally ready but old smart smoke evidence remains fail-closed on
direction/class balance. The next SMART/SEQ520 smoke must prove repaired FLAT
calibration, main direction loss, MTF direction auxiliary loss, checkpoint guard,
specialist liveness, path calibration, selected-tail direction and Exit-bound
smart state preservation.

Foundation smoke readiness is the literal gate
`READY_FOR_VEDTAK_SMOKE_TRAIN`; even then trainer start still requires clean git
and an explicit matching vedtak. Canonical gated smoke command:
`smoke-train --vedtak <id> --require-edge-audit`.

Current allowed path:
- `scripts/entry_next_edge_control.sh verify`
- `scripts/entry_next_edge_control.sh selftest`
- `scripts/entry_next_edge_control.sh foundation-guardrails`
- `scripts/entry_next_edge_control.sh worktree-hygiene`
- optional, explicit cleanup staging:
  `scripts/entry_next_edge_control.sh stage-foundation-cleanup --apply --vedtak <id>`
- `scripts/entry_next_edge_control.sh train-readiness`
- after explicit user vedtak only:
  `scripts/entry_next_edge_control.sh smoke-train --vedtak <id> --require-edge-audit`
- after explicit SMART/SEQ520 user vedtak only:
  `scripts/entry_next_edge_control.sh smart-smoke-train --vedtak <id> --require-edge-audit`

Current blocked path:
- no generic train/retrain, promote, pin, shadow, paper/live order placement, or
  legacy live/practice launch
- no candidate train, replay-readiness, IQL distillation, IQL replay comparison,
  promotion review, shadow, or live until the preceding foundation gates produce
  their explicit PASS/READY artifacts
- readiness is `READY_FOR_VEDTAK_SMOKE_TRAIN` only when foundation gates and
  git-clean execution hygiene both pass; `READY_FOR_VEDTAK_SMOKE_TRAIN_AFTER_GIT_CLEAN`
  means the foundation contract is ready but real trainer start is still blocked
- downstream gates remain closed
- no shadow/live/promotion until Entry Transformer, Entry IQL, Exit Transformer
  and Exit IQL are replay-proven as one live-equivalent policy
- heavy jobs must preserve RAM headroom and use capped runners where required

# GX1 — operational constitution (read every session)

These are the HARD rules. They are operational, not architectural — they apply to every
session, every turn. Make a violation impossible or loud, never "remember to be careful".
Architecture, data-flow, and the detailed guardrails live in @AGENTS.md — read it before acting.

## OPERATING MODE — DRIVE TO COMPLETION (user vedtak 2026-06-05)
Default to AUTONOMOUS execution. When I authorize a goal/wave (a `--vedtak`, or "kjør alt ferdig /
fortsett til du er fornøyd"), execute every step of it — builds, runs, retrains, data-prep, cleanup —
to completion WITHOUT per-step confirmation. Verify each step yourself; report at MILESTONES, not before
each step. One wave `--vedtak` covers every retrain in that wave (rule 3). Stop ONLY for (a) a genuine
hard blocker you cannot resolve after actually trying, or (b) a catastrophe-floor breach (rules 2/4/6:
git-clean-before-run, manifest, XAU-only / no-secrets / no-force-push). Continue until DONE + verified.
Asking me to confirm every little thing creates misery — DON'T. (Protected-core edits are no longer
marker-gated as of 2026-06-05 — the hook warns, doesn't block; edit the live chain deliberately per
rule 1, no per-edit `touch`. Reversible-first cleanup per rule 5 needs no per-item ask within an authorized wave.)

## ALWAYS BUILD, NEVER REMOVE — FIX AND IMPLEMENT (user vedtak 2026-06-13, MUST ALWAYS BE FOLLOWED)
When two artifacts/schemas/feature-sets differ, the answer is NEVER to DROP capability to match the
lesser one — it is to UPGRADE the lesser one UP to the richer one. We drop NOTHING. A richer schema
(more features / more regime cols / more capability) is the target; bring everything else up to it,
fix the wiring, implement the missing side. Example (the mistake that birthed this rule): a rebuilt
BASE34 had 114 cols WITH the REGIME_V4 block; the live serve had 62 WITHOUT it. The WRONG move is
"rebuild regime-OFF to match live's 62"; the RIGHT move is "keep the 114 WITH regime and upgrade the
live serve + daemon to the 114-col regime-aware schema." Removing features to make a swap easier is
forbidden. If a model/serve path lacks a feature the build has, that is a GAP to CLOSE (build it in),
never a reason to delete the feature. Always build, fix, implement — never amputate.

1. **Protected core — edit DELIBERATELY (hard marker-gate removed 2026-06-05, user vedtak).** The live
   chain / SACRED transformer contracts — `gx1/execution/`, `gx1/contracts/`, `gx1/exits/contracts/`,
   `gx1/models/entry_v10/`, `gx1/core/` — are no longer marker-gated (the per-edit `touch` friction was
   killing the workflow); the PreToolUse hook now WARNS instead of blocks. The discipline is UNCHANGED:
   verify in-use, ONE truth, minimal change, train==serve, and NEVER coarsen the M1 exit grid (that edit
   is still HARD-blocked). The gate is gone; the care is not — treat every live-chain edit as a real change.

2. **Git clean before any run.** Never start a train / retrain / backtest / Phase-6 / live launch
   while `git status --short` is non-empty. A dirty tree means we don't know what we're running.
   No exceptions.

3. **Never auto-retrain.** Every retrain requires an explicit `--vedtak <id>`. The `gx1_guards`
   are fail-closed; do not route around them.

4. **Manifest before every run.** Always produce and log a run-manifest first (`/run-experiment`):
   commit hash, config path+hash, dataset path+rows+hash, checkpoint, seed, feature-set version.
   No silent defaults — an explicit `--config` is required; missing config is a hard error, not a
   guessed default.

5. **Never delete artifacts without confirmation.** No deletes under `GX1_DATA/` or `runs/`.
   Destructive cleanup is always backup → inventory → dry-run → my confirmation → delete, in that
   order.

6. **XAUUSD only. No secrets. No force-push.** Never introduce another instrument. Never commit
   `.env`/credentials. Never `git push --force` or `git reset --hard` shared history.

7. **Extend, don't fork; check before you build.** Change the existing component; new files only
   for a genuinely-new shared one-truth helper. Before creating a file, name the existing one you
   considered extending and why it didn't fit. Keep ONE truth, fail-closed, minimal change.

8. **ONE gjeldende — single selection truth, no version roulette (user vedtak 2026-06-05).** Exactly
   ONE artifact is active per role (xgb / v10_entry / v3_exit / entry_iql / exit_iql), named in the ONE
   selection contract `PROJECT_STATE_artifacts.json`. Build / decision / serve code resolves bundles ONLY
   through that contract (`gx1_guards.load_decision_artifact`) — NEVER by glob, "latest", mtime, or a
   hardcoded/default path. Missing or ambiguous = hard error, never a guessed fallback. A new artifact is
   PENDING until it passes gates and I flip the contract; the live-active bundle is never deleted while
   active. **AUTO-PARK on supersede (user vedtak 2026-06-06): the moment a new version supersedes an old
   one (e.g. V7→V8 exit contract), AUTOMATICALLY quarantine the superseded build-artifacts — reversible
   move to `runs/_SUPERSEDED_<date>/`, WITHOUT being asked.** Superseded DATASETS / intermediate builds /
   non-active dupes park immediately (rollback re-activates the BUNDLE, not the dataset, so datasets are
   safe to park once a newer dataset exists); the still-live-ACTIVE BUNDLE parks automatically at the
   cement-flip that deactivates it — never before (it is the rollback). Quarantine→delete via rule 5
   (backup→inventory→dry-run→my confirm) so we never drown in v1/v2/v3… The goal is that running the wrong
   version is IMPOSSIBLE (a fail-closed resolver refusing to guess), not "remember to pick the right one".

9. **NOTHING ignored — feature-liveness is AUTO-checked every run, NEVER hand-verified (user vedtak
   2026-06-06).** Every input feature + dependency the chain consumes MUST be alive (non-constant on a
   SHUFFLED sample) or on the documented `gx1.audit.feature_liveness.KNOWN_ALLOWED_DEAD` allowlist. The
   ONE-TRUTH cross-chain check `gx1.audit.feature_liveness` runs AUTOMATICALLY + fail-closed: the XGB-gain
   gate at XGB-retrain post-export, the V10 ctx/snap/multi-TF (all 5 TFs alive + DISTINCT) check at
   V10-retrain post-export, the zeroed/constant-hand-off guard in the Entry/Exit-IQL builds, and
   `python -m gx1.audit.feature_liveness --strict …` before any cement. A NEW dead/ignored feature (or a
   broken/duplicated TF, or a zeroed hand-off) = a silent-ignore regression → the build/retrain/cement
   FAILS LOUD; fix the wiring or add to the allowlist with a documented reason. NEVER hand-verify "are all
   features used" again — the chain refuses to run-wrong. **LIVE-TAIL leg (user vedtak 2026-06-11): the
   check also covers the LIVE prebuilt tails** — `--live-tail` scans cv3+BASE34 for the freeze signature
   (was-varying column now constant; the 2026-05-25 BASE34 copy-forward freeze lived 17 days while every
   training-side audit was green). Runs AUTO: hard-fail preflight in `launch_live_practice.sh` + hourly
   ERROR-loud self-check in the canonical_incremental daemon. Allowlist `LIVE_TAIL_ALLOWED_CONST`
   (documented reasons only). **EVERY (re)start preflight = THREE legs (user-direktiv 2026-06-12):**
   (1) live-tail FREEZE — hard fail; (2) CONTINUITY/gap-sjekk mot helg/pause/helligdager/KNOWN_DATA_GAPS —
   fresh UNKNOWN gap = hard fail (hull i historikken blokkerer oppstart); (3) KS DISTRIBUTION-DRIFT —
   siste 7d live-states vs ACTIVE-bundlets `drift_reference_v1.parquet`, ADVISORY-loud (markedsdrift
   flagger et retrain-VEDTAK, blokkerer aldri en launch — rule 3; en bot som nekter å starte i nytt
   regime er feil). Drift-referansen regenereres ved hvert cement (`--write-drift-reference`).

@AGENTS.md
