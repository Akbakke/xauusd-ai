# GX1 agent constitution

## Current objective and status

Build one XAUUSD Entry model that learns tops, bottoms and abstention from the
full evidence stack, then selects `LONG`, `SHORT` or `FLAT` through one
model-native decision path. Near-perfect practical precision is the target,
not a current claim. The system stays closed until immutable out-of-sample
contracts prove the required direction edge.

Current Entry status is **BLOCK**. There is no accepted fresh seq513 bundle.
Old Smart520, neutral-XGB, anchored Entry and Entry-IQL evidence is historical
and cannot authorize a run or launch.

Read in this order:

1. `AGENTS.md`
2. `SYSTEM_MAP.md` — including the "Pipeline- og ingredienskart" section: read
   it BEFORE grepping for artifact producers or feature-column owners; it is
   the one-truth map of the data DAG and saves large amounts of re-scanning.
3. `HANDOVER_XAU_DIRECTION_REPAIR_20260714.md`
4. `PROJECT_STATE_xau_direction_launch.json`
5. relevant code contract and test files

Run `bash scripts/gx1_handover.sh` for a read-only takeover snapshot.

## Hard architecture invariants

### One Entry decision authority

The public Entry decision is exactly the accepted bundle's calibrated final
direction logits and `argmax([LONG, SHORT, FLAT])`. Nothing downstream may
veto, flip, threshold, recover or replace it. A missing decision, field,
contract, artifact or proof is an error, never `FLAT` by convenience and never
permission to use another policy.

### Full learned evidence

The exact model-native surface is:

- 513 ordered signals = 34 genuine base fields + 479 specialist fields;
- the 479 specialist fields = all 305 outputs from ten code-owned causal
  layers in exact registry order + 174 deterministic TRAIN-only ranked fields;
- 142 ordered continuous context fields;
- 5 ordered categorical context fields;
- sequence length 96;
- M5, M15, H1, H4 and D1 market context;
- cross-timeframe attention, positional encoding, FiLM context conditioning
  and learned timeframe scales;
- eight specialists: structure/swing, SMC/liquidity, trend/EMA,
  volatility/compression, momentum/flow, session/regime, chart geometry and
  price-action/candles.

Every genuine trend, session, structure, liquidity, volatility, momentum,
chart, candle, path-quality and utility input/target stays in the learned
path. Removing an old filter means removing competing decision authority, not
amputating its evidence.

All advertised output heads must be present, trained with positive weight,
exported and audited. This includes direction, MTF direction, specialist
fusion/gates, path/MFE/tradability/bad-path/clean-edge/survival evidence,
trade-side hierarchy, side validity, trendline rail, timing/tail/volatility,
TF agreement and position size.

The exact active output declaration has 20 heads. Their ordered evidence feeds
one learned 23-group/75-value direction fusion (`LayerNorm(75)`, `75 -> 128`,
GELU, `128 -> 3`) before immutable calibration and exact three-class argmax.

### Sizing boundary

The learned `position_size` head is mandatory and must be trained from the
realized future path target, parity-checked and journaled. Label-horizon TEST
utility/exposure/drawdown is a sizing-head diagnostic, not capital authority.
No fresh accepted current-contract sizing result exists.
Paper/live additionally requires a joint sizing-only replay with the exact
adopted active Exit stack and a fresh post-adoption broker runtime-parity event.
Until both pass, emit no order. Never substitute fixed units or multiplier
`1.0`; historical fixed 1x may be used only as an explicitly named comparison
benchmark.

### Exact contracts; no compatibility lane

The model-native lane has no XGB Entry bridge, neutral constants, direction
anchor, legacy direction mode, Entry-IQL fallback, warm start, post-model
filter or alternate launcher. Do not preserve dead arguments or outputs for
compatibility. Exact consumers either match the current schema or fail.

Active Exit components are a separate scope. Do not delete Exit XGB or
Exit-IQL primitives merely because XGB and IQL are forbidden as Entry
direction authorities.

## Evidence and artifact rules

- Use explicit absolute immutable paths and content hashes. Never select by
  glob, mtime, lexical version or a path containing mutable `latest`.
- Bind dataset splits, ordered fields, recipe, source revision, model state,
  metadata, lock, calibration, predictions, replay journal and audit events.
- Validate file identity again at every authority boundary. A report saying
  `PASS` without proving its bound bytes is not evidence.
- Newest terminal event for the same family wins. Newer red, malformed or
  incomplete evidence invalidates older green evidence.
- Train/validation/test must be chronological and leakage-safe. Calibration
  may only fit on its declared calibration split and must be immutable before
  evaluation.
- Full-field liveness covers all 513+142+5 inputs on train, val and test.
  Unallowlisted constant, non-finite, dead, duplicated or reordered fields
  fail. Multi-timeframe values must be alive and genuinely distinct.
- Precision claims require untouched OOS evidence, meaningful support,
  calibration, class/slice confusion, path/utility quality, costs and
  live-like replay. Code-contract tests alone never prove trading edge.
- `PROJECT_STATE_xau_direction_launch.json` remains `BLOCK` until one exact
  bundle and its newest terminal evidence are explicitly admitted.

## Run authority and resource safety

Source edits, read-only audits and tests are authorized by the active goal.
A dataset rebuild, training run, broad sweep, replay that writes large
artifacts, paper/demo launch, live launch or promotion needs its explicit
contract and `--vedtak`. Documentation never supplies that authority.

Before a heavy authorized run:

- inspect `git status --short`, disk, RAM and active Python processes;
- do not collide with persistent collectors/canonical builders/dashboard;
- use capped execution where the contract requires it;
- stop a clearly hard-red recipe instead of consuming compute;
- monitor `/home/andre2` and `/home/andre2/GX1_DATA`; clean repository debris
  continuously, but never delete GX1_DATA evidence without explicit approval.

Operational discipline (paid for in the 2026-07-18/19 campaign; follow always):

- Any computation over ~15 minutes MUST persist a checkpoint of its expensive
  intermediate result before its final write step, bound to input sha and
  window key, and reuse it on rerun (pattern: ranker `_ranker_checkpoint.npz`).
- A watcher that polls for a process must match the exact python command line
  (anchored `pgrep -f '^...python -m module'`), never a substring a wrapper
  shell also contains; a dead job must alert immediately, not wait silently.
- Every shell invocation starts with `cd /home/andre2/src/GX1_ENGINE &&`
  (the working directory can reset between calls; rg then returns silently
  empty results and `python -m gx1...` fails to resolve).
- Never let a pipe swallow an exit code: capture command output to a file and
  append the real `$?`, or check PIPESTATUS.
- Per-row loops over the tape are ~85 ms/row: parallelize with one shared
  full-series context and a fanned row loop (exact by construction), never
  with chunk overlap — long-memory features (trailing-1yr percentiles) make
  overlap both wrong and slow.

Use `/home/andre2/src/GX1_ENGINE/.venv/bin/python` for repository Python.
Use `rg`/`rg --files` for source discovery so `.gitignore` excludes `.venv`,
artifacts and data. Never recursively scan `.venv`, `.git` or
`/home/andre2/GX1_DATA` when a source-scoped query is sufficient.

## Repository ownership map

- `gx1/contracts/`: exact schemas, validators and immutable authority helpers.
- `gx1/features/`: genuine feature construction and specialist ownership.
- `gx1/models/entry_v10/`: model-native Entry architecture and trainer.
- `gx1/scripts/build_entry_v10_ctx_training_dataset_v3.py`: canonical Entry
  dataset builder; despite the historical filename it must be model-native only.
- `gx1/execution/v12_model_native_state_live.py`: model-native serve state.
- `gx1/execution/v12_smart_entry_live.py`: admitted Entry bundle adapter; must
  fail while launch state is blocked.
- `gx1/execution/model_native_entry_replay_v1.py`: neutral offline replay
  primitives; it does not select direction.
- `gx1/scripts/audit_entry_foundation_smoke_bundle_v1.py`: exact bundle audit;
  historical filename, current model-native responsibility.
- `scripts/entry_next_edge_control.sh`: single read-only/control entrypoint.
- `scripts/gx1_handover.sh`: single handover entrypoint.
- `PROJECT_STATE_xau_direction_launch.json`: current Entry launch decision.
- `PROJECT_STATE_artifacts.json`: retained artifact registry, including the
  separately active Exit chain; it cannot override the Entry launch contract.
- `/home/andre2/GX1_DATA`: large immutable data, bundles and evidence.

Do not create a parallel implementation when an active owner exists. If a new
shared helper is truly necessary, move all consumers to it and delete the old
owner in the same bounded change.

## Cleanup discipline

Continuously remove disconnected repository scripts, archived source copies,
sole-purpose stale tests, obsolete configs and historical Markdown. Before
deletion, prove no active imports, subprocess calls, control routes, process
command lines or unique evidence ownership remain. Preserve user changes and
anything touched by an active process. Empty compatibility shells are not a
valid substitute for deletion.

Never use destructive Git recovery commands. Do not modify or remove secrets.
Never force-push.

## Verification before handoff

For each bounded change:

1. run syntax/import checks for modified Python and shell;
2. run the smallest contract tests that prove the change;
3. run broader collection/integration tests when shared surfaces changed;
4. scan for stale filenames, modes, arguments, fallbacks and duplicate owners;
5. run `git diff --check`;
6. state which facts are code-proven, which are empirically proven and which
   remain blocked.

Never claim near-perfect precision, launch readiness or a completed goal from
green unit tests alone.
