---
name: run-experiment
description: Gate any GX1 training/backtest/retrain behind a deterministic run-manifest (git-clean + hashed config/dataset/checkpoint + explicit confirmation). User-invoked only via /run-experiment.
disable-model-invocation: true
allowed-tools: Bash Read Glob Grep
---

# /run-experiment — never run the wrong thing

You (Claude) MUST follow this procedure exactly. Its job is to make "we thought we ran
config X but actually ran Y" impossible. Do not skip a step, do not invent defaults, do
not proceed on a dirty tree.

## Live repo state (captured at skill load)

- HEAD:    !`git -C /home/andre2/src/GX1_ENGINE rev-parse --short HEAD 2>&1`
- Branch:  !`git -C /home/andre2/src/GX1_ENGINE rev-parse --abbrev-ref HEAD 2>&1`
- Status (MUST be empty to run):
```!
git -C /home/andre2/src/GX1_ENGINE status --short 2>&1 || echo "(git error)"
```

## Procedure

1. **Parse the request — no silent defaults.** From the user's `/run-experiment` input,
   extract: `--config` (REQUIRED), and the run command they want gated. Also collect
   `--seed`, `--feature-set` (REQUIRED), and where present `--dataset` and `--checkpoint`.
   If `--config`, `--seed`, or `--feature-set` is missing, STOP and ask the user for it —
   never guess. If no explicit run command is given, STOP and ask what is to be executed.

2. **Hard precondition — git clean.** If the "Status" block above is non-empty, STOP.
   Tell the user the tree is dirty and list it. Do not run anything until they commit or
   stash. (Rule 2 in CLAUDE.md.)

3. **Build the manifest (dry-run first).** Run:
   ```
   /home/andre2/src/GX1_ENGINE/.venv/bin/python /home/andre2/src/GX1_ENGINE/scripts/run_manifest.py \
     --dry-run --config <cfg> --seed <seed> --feature-set <ver> \
     [--dataset <p>] [--checkpoint <ckpt>] --run-cmd "<exact run command>" --label "<short>"
   ```
   Use `--hash-mode meta` only if a dataset is multi-GB and a full content hash is
   impractical (say so explicitly). Show the user the full manifest JSON it prints.

4. **WAIT for explicit confirmation.** Present the manifest and ask, verbatim:
   *"Bekreft: skriv `kjør` for å persistere manifestet og starte kjøringen."*
   Do nothing further until the user replies with an explicit `kjør` (or equivalent
   clear go-ahead). Anything else = abort, no run.

5. **Persist + run.** On confirmation, re-run `run_manifest.py` with `--commit` (NOT
   `--dry-run`) so it re-checks git-clean and writes `manifest.json` into a fresh
   `GX1_DATA/runs/<utc>-<commit8>-<cfg8>/` dir. Capture that run-dir path (last stdout
   line). Then execute the user's run command, directing its outputs/logs INTO that same
   run dir so results and manifest live together.

6. **Report.** Tell the user the run-dir path and confirm the manifest + results are
   co-located. The run is now fully reproducible: commit, config-hash, dataset-hash+rows,
   checkpoint-hash, seed, feature-set are all pinned on disk.

## Hard stops (never override silently)
- Missing `--config` / `--seed` / `--feature-set` → ask, never default.
- Dirty git tree → refuse.
- A retrain without an explicit `--vedtak <id>` → refuse (CLAUDE.md rule 3 / gx1_guards).
- Any edit to protected core (`gx1/execution|contracts|exits/contracts|models/entry_v10|core`)
  needed to make the run work → STOP and get my explicit confirmation first (rule 1).
