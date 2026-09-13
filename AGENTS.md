The user explicitly authorized the September 13 VAL throughput restart.
Successor f40ec16f preserves completed TRAIN and uses Exit-VAL batch 128;
TRAIN/Entry-VAL remain 16. This specific restart supersedes the earlier hold
on performance changes. After the restart, return to sparse monitoring.

# GX1 takeover instructions

Read `GX1_ARBEIDSMAAL.md`, `CURRENT_HANDOVER.md`, then `SYSTEM_MAP.md`.
Run `bash scripts/gx1_handover.sh --check` and `bash scripts/gx1_handover.sh`
from this documentation checkout on the training host. They verify the small
immutable bindings and observe the separate frozen training source/runtime.
A cloned repo on another host needs the named artifacts restored first;
missing files are missing evidence, not permission to start a new run.

## Current operator scope — 2026-09-13

The current instructions are in `GX1_ARBEIDSMAAL.md` and `CURRENT_HANDOVER.md`.
The operator authorized the full local lifecycle-v2 campaign: full one-year
TRAIN smoke plus full June VAL are complete; full five-year TRAIN now runs
for up to 30 epochs, with full June VAL after each epoch and patience 5.
TEST remains sealed; no paper/live, broker activity or external spending.
Standing authorization covers ordinary necessary in-scope work; do not ask
again for the same approval. The Windows controller owns automatic guarded
pause/reboot/resume. The frozen active source is identified by
`CURRENT_NATIVE_RUN.json`; never edit or rebind it for documentation.

Conserve tokens and time: one agent and one heavy job by default, approximately
15-minute observations, silence on routine healthy progress, no minute polling.
Fix only a concrete observed blocker with the smallest necessary change.
No speculative refactors, repeated passed smokes/full suites, opportunistic
benchmarks or extra cleanup. Targeted verification for a real change suffices.
The completed September 12 three-agent audit was explicitly requested and is
finished; it does not authorize continuing parallel agents.

Current plan/recipe values supersede the old operational limits below: 300 W
physical cap, 310 W actual-draw stop, 85 C core, 80 C memory junction, 12 GiB
VRAM; the keeper reduces power to 200 W at 80 C core. These are local operating
limits. Do not restore the obsolete 160/200 W policy or disable the campaign
because an old paragraph says training is blocked. Earlier dated hold, retry,
power, sampler and launch-state instructions below are historical for this
campaign. Technical contracts still apply; source and runtime evidence outrank
stale prose. `scripts/gx1_handover.sh` observes the explicit native binding;
it neither authorizes nor starts training.

The historical architecture and evidence rules remain in `GX1_RULES.md`;
its latest operator scope overrides earlier dated launch holds. Source owners
own dimensions, feature order, economics and model decisions. No handwritten
threshold or maximum holding period has been added. Never replace the learned
policy while a bound evaluation is running.

For documentation/observer work, inspect the diff, run shell syntax and focused
observer tests. Preserve installed Git hooks. Avoid a second heavy job while
the trainer owns its lock; commit-time capped checks need a natural idle slot.
Do not use a frozen training checkout for edits or commit a trained checkpoint,
raw data, runtime logs, credentials or private configuration to Git.
