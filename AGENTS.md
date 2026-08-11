# GX1 agent handover

Read `GX1_RULES.md` first. It is binding.

## Current truth

- Scope is offline XAUUSD only.
- Architecture is fixed: the same eight code-owned feature implementations run
  independently on local M5 for Entry and local M1 for Exit, in one model and
  shared encoder. There is no combined pre-owner M1/M5 package.
- Entry is 96×592 plus 142 continuous and 5 categorical context values (592 =
  34 base + 425 mandatory causal + 133 TRAIN-ranked, 16 mandatory families —
  the V29 event surface with level/trendline registries and per-TF event
  primitives; counts derive from the owner tuples).
- Entry consumes one immutable native M5 feature surface across all splits;
  exact contiguous timestamp views are required. Never restore per-split
  inline reconstruction of the 558 specialist fields.
- Exit is the same feature contract at M1, a 480-bar M1 sequence, the frozen
  Entry representation and the additive 14-field causal path.
- Entry context is closed M15/H1/H4/D1. Exit context is closed
  M5/M15/H1/H4/D1. Build closed OHLCV bars before features; never resample
  already computed M1 indicator values into a higher timeframe.
- Unique model argmax is the only Entry/Exit authority; ties fail closed.
- There is no admitted model, recipe, edge, win-rate or PnL proof. The V28
  dataset chain ran GREEN (frozen comparison baseline, retired 513 surface);
  no V29 dataset exists yet.
- Fresh native M1/M5 V4 sources and canonical pair generation
  `9b18e215...077232cd` (2026-08-09) exist; the 2026-08-04 parent
  `64d62c1f...a11b84c` is untouched history. They are source authority only.
- The V29 rebuild chain requires the explicit registry-fit inputs
  (`--level-tol-quantile-q 0.5` adopted 2026-08-11; fit window defaults to
  the chain's `--train-end`). Registry fits freeze into the hash-bound build
  manifests; both lanes fail closed without them.
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
  native M5 and native M1. The exact Entry signal-manifest hash and TRAIN-rank
  state must bind the Exit surface. Resolution-specific values stay separate.
- Fit that rank state from the pair-bound canonical M5 market fields, then
  require exact canonical-to-final-M5 market identity through TRAIN. Never
  revive the circular rank-from-downstream-source path.
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
at most 20G for the canonical trainer (raised from 10G 2026-08-09 on real
batch=640 RSS measurement, see CLAUDE.md), 512 MiB swap, CPU 0-1, one job at a
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
2. Use the published V4 native/canonical pair and run the current-pair rebuild
   chain under a new dataset run ID.
3. Rebuild both resolution surfaces, then pass the exact M5 surface to Entry
   and the exact M1 surface to lifecycle/Exit before building the splits.
4. Materialize a distinct training run ID and bounded smoke recipe.
5. Train/audit smoke, then full candidate if every gate passes.
6. Fit allowed calibration, freeze the candidate, evaluate untouched TEST and
   run the same bundle's unified Exit replay.

Run `bash scripts/gx1_handover.sh` whenever authority or status changes.
