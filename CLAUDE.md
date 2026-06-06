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
   features used" again — the chain refuses to run-wrong.

@AGENTS.md
