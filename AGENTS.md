# GX1 agent handover

Read `GX1_RULES.md` first. It is binding.

## Current truth

- Scope is offline XAUUSD only.
- Architecture is fixed: one shared eight-owner featurebase, Entry M5 and Exit
  M1 in one model/shared encoder.
- Entry is 96×513 plus 142 continuous and 5 categorical context values.
- Exit is the same feature contract at M1, a 480-bar M1 sequence, the frozen
  Entry representation and the additive 14-field causal path.
- Unique model argmax is the only Entry/Exit authority; ties fail closed.
- There is no admitted dataset, recipe, model, edge, win-rate or PnL proof.
- V18 was invalid because training `run_id` equalled `dataset_run_id`; it was
  stopped safely. Source/lifecycle fixes invalidate V8/V13/V18 as resume input.

## Takeover sequence

```bash
bash scripts/gx1_handover.sh --check
bash scripts/gx1_handover.sh
git status --short --untracked-files=all
git worktree list --porcelain
```

The executable handover is the status owner. Do not infer state from old run
directories. A registered dirty nested worktree may belong to another agent;
do not modify or delete it.

## Implementation discipline

- Extend an existing owner instead of adding another implementation.
- One formula, one ordered field contract and one normalization state must serve
  M5 and M1. Resolution-specific mechanics stay in the shared surface owner.
- No ambient environment flag may change feature bytes, dimensions, sampling,
  objectives or model decisions. Recipe-owned values are exact and audited.
- There is one deterministic FP32 trainer path. Feature producers use one
  worker and DataLoaders use zero subprocess workers; do not add compile,
  autocast, TF32, hardware-derived workers or soft performance fallbacks.
- Candidate training uses the full TRAIN population. Smoke subsampling is an
  explicit storage/compute profile and cannot authorize a candidate.
- Calibration events are immutable, split-bound and retained across direction,
  path and sizing stages. TEST cannot fit anything.
- M5 label horizons remain M5 bars even when outcomes are reconstructed from
  M1; never count M1 rows as M5 bars.
- Missing source, context, path, model or provenance is an error, never a
  neutral value or fallback.

## Machine safety

Every heavy command uses `scripts/gx1_capped_run.sh`: 4G for audits/tests and
at most 10G for the canonical trainer, 512 MiB swap, CPU 0-1, one job at a
time. Communicate before any run lasting more than a minute. Never launch live,
paper, broker, dashboard, collector, notifier or adaptation work. Do not stop
pre-existing processes unless the user explicitly authorizes that action.

The verified takeover environment is CPython 3.10.12 with the direct packages
in `requirements.txt`. Do not reproduce the workstation by freezing unrelated
packages from `.venv`.

## Required verification before commit

```bash
git diff --check
bash -n scripts/*.sh
.venv/bin/python -m compileall -q gx1 tests
scripts/gx1_capped_run.sh --mem 4G --swap 512M -- \
  .venv/bin/python -m pytest -q
```

Inspect the exact diff and preserve unrelated user changes. No destructive Git
commands. Generated-run cleanup must use the retention contract, not `rm`.

## Next implementation sequence

1. Verify the audited producer commit with the executable handover.
2. Publish a fresh native M1/M5 pair under a new dataset run ID.
3. Rebuild both resolution surfaces, MTF cache, lifecycle v3 and split dataset.
4. Materialize a distinct training run ID and bounded smoke recipe.
5. Train/audit smoke, then full candidate if every gate passes.
6. Fit allowed calibration, freeze the candidate, evaluate untouched TEST and
   run the same bundle's unified Exit replay.

Run `bash scripts/gx1_handover.sh` whenever authority or status changes.
