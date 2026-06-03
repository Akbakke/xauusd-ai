# GX1 — operational constitution (read every session)

These are the HARD rules. They are operational, not architectural — they apply to every
session, every turn. Make a violation impossible or loud, never "remember to be careful".
Architecture, data-flow, and the detailed guardrails live in @AGENTS.md — read it before acting.

1. **Protected core is frozen.** Never `Edit`/`Write` the live chain or the SACRED transformer
   contracts — `gx1/execution/`, `gx1/contracts/`, `gx1/exits/contracts/`, `gx1/models/entry_v10/`,
   `gx1/core/` — without my explicit, per-change confirmation. A PreToolUse hook blocks these by
   default; lifting it is a deliberate act by me, never assumed by you.

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

@AGENTS.md
