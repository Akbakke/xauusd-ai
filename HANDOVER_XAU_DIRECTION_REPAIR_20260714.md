# GX1 XAUUSD handover

Updated 2026-08-04. Run `bash scripts/gx1_handover.sh` before relying on this
document. `GX1_RULES.md` is binding.

## Current verdict

The car's major components are connected in source, and a fresh immutable
native/canonical source pair is published, but it is not ready to drive.
Launch remains `BLOCK`. No admitted featurebase/dataset, valid training recipe,
accepted model, calibrated bundle, untouched-TEST edge, PnL or win-rate proof
exists.

The prior producer-tree audit on 2026-08-04 used five independent review lanes.
The current rank-source/current-pair repair passed repo-wide Ruff, Python
compilation, shell syntax and all 2,006 collected tests under the 4G cgroup.
Source consistency never proves trading edge or profitability.

The old V8 dataset, V13 lifecycle and V18 recipe cannot be resumed. Contract
repairs changed source lineage and M1 episode semantics; V18 also reused the
dataset run ID as the training run ID and was stopped safely during epoch 2.
No OOM occurred in that stop.

## Current feature architecture

- the same eight feature owners use one implementation each, run independently
  at native M5 for Entry and native M1 for Exit; no combined pre-owner M1/M5
  package;
- 513 ordered signals, 142 continuous and 5 categorical context fields;
- Entry: 96 local M5 bars plus leak-safe M15/H1/H4/D1 context;
- Exit: the same ordered fields on a 480-bar M1 local sequence plus leak-safe
  M5/M15/H1/H4/D1 context, frozen 128-value Entry representation and additive
  causal path;
- closed OHLCV is built before each timeframe's features; finished M1 features
  are never resampled upward or copied into Entry;
- one shared encoder and one committed bundle;
- unique calibrated LONG/SHORT/FLAT argmax for Entry;
- unique HOLD/EXIT_NOW argmax for Exit;
- exact ties and missing evidence fail closed.

## What is implemented

- native OANDA M1/M5 immutable source and pair contracts;
- exact M1/M5 owner, ordered-field, signal-manifest, TRAIN-rank and source
  identity validation;
- one required immutable M5 Entry surface loaded once through bounded memmaps
  and shared as exact zero-copy timestamp windows across TRAIN/VAL/TEST;
- all eight specialist families and five-timeframe grid;
- TRAIN-only ranking and normalization contracts;
- model-native Entry direction and unified Exit heads;
- M1 lifecycle builder/loader and same-bundle replay path;
- immutable direction/path calibration provenance;
- learned sizing and serve/replay parity contracts;
- capped-run resource owner and immutable event machinery.

The latest repair pass also:

- made every one of the eight owners explicitly resolution-symmetric at local
  M5/M1 and locked the M15 bridge into both routes;
- made preflight and dataset construction reject an Exit surface whose ordered
  513 fields, signal-manifest hash or TRAIN-rank state differs from Entry;
- removed Entry's three split-local 479-feature rebuilds; the M5 producer is
  now the sole model-input surface and missing/noncontiguous rows fail closed;
- made M1 windows include the current closed row;
- changed lifecycle continuity from wall-clock minutes to authoritative
  observed M1 rows across proven closures;
- bound exact M1 feature artifacts into bundle/runtime evidence;
- separated dataset and training run IDs;
- removed TEST from smoke/candidate selection and forbade candidate subsampling;
- retained sequential direction/path calibration events;
- wired the ×10 price corruption guard into active producers and replay;
- fixed M5 label horizons so they are not counted as M1 rows;
- removed ambient feature flags, repaired the ATR regime feature and removed
  unused duplicate feature/target/source routes;
- collapsed training to one deterministic FP32 path, fixed feature production
  to one worker and model DataLoaders to zero subprocess workers;
- removed the old label-horizon replay commands from the active control path;
- completed the authorized retention workflow for 33 exact old-run targets
  with `DELETE_COMPLETE` and no recorded failure;
- replaced the stale handover that falsely treated V18 as runnable;
- published native V4 sources and canonical pair generation
  `64d62c1f29e5d2b30f4e187af1ec65cabd48bb50fe4638a3ec5af2523a11b84c`:
  M1 has 2,661,631 rows through `2026-08-04T07:54:00Z`, M5 has 537,861
  rows through `07:50:00Z`, canonical has 470,139 rows and BASE28 has
  2,335,830 rows;
- replaced the circular rank dependency with one canonical-M5 TRAIN-rank
  source plus exact canonical-to-final-M5 market identity proof;
- rewired `scripts/run_seq513_rebuild_chain_v1.sh` to build current M5 and M1
  owner lanes, the mandatory M15 cache, both ordered feature surfaces and then
  preflight/rebuild. The legacy event-local source-cascade route is rejected.

## What remains empirically unproven or unadmitted

1. Rebuilt M1/M5 feature surfaces from the published pair.
2. A lifecycle-v3 and split dataset with complete liveness and exact identity.
3. A distinct, valid bounded smoke recipe and successful smoke bundle.
4. A full candidate trained on all TRAIN rows.
5. Immutable calibration using only its declared non-TEST split.
6. Untouched-TEST precision, PnL, drawdown and slice evidence.
7. Same-candidate unified Entry/Exit full-TEST replay and runtime parity.

Until all seven exist, practical precision and profitability are unknown.

## Machine and process safety

Use `scripts/gx1_capped_run.sh`: 4G for tests/audits, at most 10G for the
canonical trainer, 512 MiB swap, CPU 0-1 and one job at a time. Never increase
limits to force progress. A partial or killed run is failed evidence.

The verified environment is CPython 3.10.12. `requirements.txt` contains only
the pinned direct runtime and verification packages needed for takeover; it is
not a dump of unrelated packages present in `.venv`.

At this handover, old collector/dashboard/notifier processes may already be
running outside the frozen scope. Do not restart, use or stop them without
explicit user authority. They do not prove or authorize the offline model.

A registered dirty nested worktree exists. It belongs outside this repair;
do not inspect, clean or delete it. Preserve all unrelated user changes.

## Next implementation sequence

1. Verify the audited producer commit with `scripts/gx1_handover.sh --check`.
2. Run the current-pair chain from the published pair with one new dataset run
   ID; it builds both native feature lanes one capped job at a time.
3. Admit the resulting M1/M5 surfaces, lifecycle and TRAIN/VAL/TEST only if all
   preflight/liveness/identity gates pass.
4. Produce a distinct smoke training run ID and exact 4G/10G-safe recipe.
5. Run smoke and audit every class, head, gate and Exit path.
6. If smoke passes, train one full candidate, calibrate and freeze it.
7. Open TEST once, calculate actual PnL/win rate and run unified Exit replay.

No architecture redesign is planned. Failures should be repaired in the
existing owner, with the smallest exact change that preserves the full model.

## Takeover

```bash
bash scripts/gx1_handover.sh --check
bash scripts/gx1_handover.sh
git status --short --untracked-files=all
git worktree list --porcelain
```

Then read `GX1_RULES.md`, `AGENTS.md`, `SYSTEM_MAP.md` and
`docs/DATA_CONTRACT.md`. Use only `scripts/entry_next_edge_control.sh` for the
active workflow. It intentionally exposes no live or legacy replay route.
