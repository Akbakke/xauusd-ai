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
V19 was the first current-data dataset lineage to reach a `GREEN` chain
terminal under the former v1 surface, intentionally stopped at the smoke gate. It
binds exact `XAU_USD` M1/M5 tape provenance, source cascade v5, TRAIN-only
ranking, 513 ordered signals, 142 continuous context fields, five categorical
fields, all three chronological split manifests, exhaustive schema-v3 input
liveness and pretrain audit v2. The splits contain 369,081 TRAIN, 5,904 VAL
and 3,934 TEST rows through 2026-07-21T20:00:00Z. All 1,980 field/split
liveness records validate; June/July H4/D1 ATR shift is recorded as genuine
OOD diagnostic evidence, not rewritten or waived.

The subsequent foundation audit proved that V19 omitted all 57 required
`chart.foundation_*` fields from the model input. V19 is therefore immutable
superseded evidence, not an accepted training dataset. No model was trained on
V19. V20 rebuilt fresh current source through 2026-07-22T07:35:00Z and passed
source audit, TRAIN-only ranking, the 513-field manifest and preflight. Dataset
construction then failed closed before any split publication because
`chart.geometry_channel_position_low_to_high`, a structural auxiliary-label
prerequisite, was optional ranking evidence. V20 is terminal `RED` and must
never be reused. The active v3 contract owns all 19 structural-label signal
requirements in the mandatory prefix: 377 mandatory plus 102 ranked fields.
Only a wholly fresh V21 may proceed. Launch remains `BLOCK` until rebuild,
smoke, candidate, untouched OOS, replay, serve parity, sizing and shadow gates
all pass.

ROADMAP.md is the current execution/takeover plan. Read it after this
constitution; it records active rebuild incidents but never overrides the
machine-readable launch BLOCK.

Read in this order:

1. `AGENTS.md`
2. `SYSTEM_MAP.md` — including the "Pipeline- og ingredienskart" section: read
   it BEFORE grepping for artifact producers or feature-column owners; it is
   the one-truth map of the data DAG and saves large amounts of re-scanning.
3. `HANDOVER_XAU_DIRECTION_REPAIR_20260714.md`
4. `PROJECT_STATE_xau_direction_launch.json`
5. relevant code contract and test files

Run `bash scripts/gx1_handover.sh` for a read-only takeover snapshot.

Token/credit discipline:

- Run the compact snapshot once per root takeover. On continuations and in
  subagents, run `bash scripts/gx1_handover.sh --check` first; when its
  `authority_fingerprint` is unchanged, do not reread the authoritative
  documents or repeat an all-Markdown inventory.
- `--verbose` is an explicit diagnostic, not a takeover default. Do not print
  the full handover or a raw process table again after they have been read.
- Start discovery from the ownership paths in `SYSTEM_MAP.md`. Read exact
  files or bounded sections and cap search output; broaden a source scan only
  when the map cannot identify the owner, and record why it was necessary.
- Repository cleanup is a token/credit control, not optional polish. Whenever
  work exposes apparently unused code, perform one bounded ownership/reference
  check immediately. If no active caller, process, control route, unique
  evidence producer or reproducibility duty remains, delete the code and its
  sole-purpose tests/config/docs in the same change instead of postponing it.
  If safe deletion is not yet proven, record the exact unresolved owner; do not
  repeatedly rescan the repository or retain commented/renamed dead copies.
- Source cleanup uses reviewed source edits. Destructive `GX1_DATA` cleanup has
  exactly one owner: `gx1/scripts/cleanup_gx1_evidence_v1.py`. Direct `rm`,
  ad-hoc `unlink`/`rmtree`, parent deletion with exclusions, abbreviated `...`
  paths and cleanup-script copies are forbidden. The only admitted sequence is
  byte-inventoried exact leaf targets, pinned registry/launch/incident authority,
  immutable plan, separate immutable approval, explicit execute, same-device
  atomic quarantine, revalidation and immutable terminal evidence. Without a
  user-approved vedtak, run only the plan/dry-run validation path.

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
- the 479 specialist fields = all 377 outputs from twelve code-owned causal
  layers in exact registry order + 102 deterministic TRAIN-only ranked fields;
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
TF agreement, position size, counterfactual action value and expectile value.

Head liveness alone is not learned evidence. Current target audit requires the
complete canonical 46-target surface in TRAIN/VAL/TEST. Current immutable
prediction/smoke evidence must prove on VAL and TEST that LONG adverse-turn
timing identifies BOTTOM pockets, SHORT adverse-turn timing identifies TOP
pockets, Q ranks the counterfactual reward-best action, V tracks learned max-Q
and Advantage equals Q minus V. Missing old-schema fields fail closed; do not
forward-fill them.

Smoke liveness is not a pass-through or direction-influence proof. Only
serve-parity v4 may supply that launch evidence: both specialist ablation
methods and all 26 exact fusion-slice replacements must move class-centred raw
and calibrated logits above the immutable row/epsilon floors. The same applies
to both context tensors and every retained timeframe. Missing or pre-v4
evidence fails closed.

The exact active output declaration has 22 heads. Their ordered evidence feeds
one learned 26-group/96-value direction fusion (`LayerNorm(96)`, `96 -> 128`,
GELU, `128 -> 3`) before immutable calibration and exact three-class argmax.

### Sizing boundary

The learned `position_size` head is mandatory and must be trained from the
realized future path target, parity-checked and journaled. Label-horizon TEST
utility/exposure/drawdown is a sizing-head diagnostic, not capital authority.
No fresh accepted current-contract sizing result exists.
Paper/live additionally requires a joint sizing-only replay with the exact
adopted active Exit stack and a fresh post-adoption broker runtime-parity event.
The joint proof must cover full TEST and bind the complete per-M1 active Exit
HOLD-to-`EXIT_NOW` trace for every non-FLAT row; runtime parity is broker-live
shadow-only and submits zero orders. Until both pass, emit no order. Never
substitute fixed units or multiplier `1.0`; historical fixed 1x may be used
only as an explicitly named comparison benchmark.

### Exact contracts; no compatibility lane

The model-native lane has no XGB Entry bridge, neutral constants, direction
anchor, legacy direction mode, separate Entry-IQL policy/fallback, warm start,
post-model filter or alternate launcher. Learned offline-RL Q/V/advantage
heads may exist only inside the shared encoder and final learned direction
fusion, with positive training objectives and exact export/serve evidence;
they never become a second policy. Do not preserve dead arguments or outputs
for compatibility. Exact consumers either match the current schema or fail.

Active Exit components are a separate scope. Do not delete Exit XGB or
Exit-IQL primitives merely because XGB and IQL are forbidden as Entry
direction authorities.

Continual Entry adaptation has one source path: immutable same-bundle drift,
replay-readiness v2, offline challenger, zero-order shadow, explicit promotion
and rollback to a prior incumbent. Replay never activates launch directly.
Shadow promotion requires identical incumbent/challenger paths,
bid/ask-recomputed outcomes, absolute candidate LONG/SHORT edge and positive
lower-95% paired improvement globally and per supported direction/context.
Live/online weight updates, post-model direction rules and stale lifecycle
events are forbidden. Launch must cross-bind the fresh activating lifecycle
event to the exact accepted bundle, serve, active Exit and learned-sizing
evidence; otherwise fail closed.

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
  TRAIN must prove learnable variability or an explicit sparse-event support
  floor. Untouched chronological VAL/TEST may truthfully contain one regime
  state, but every value remains fully scanned, finite, ordered and hash-bound;
  categorical OOS values must be inside the exact TRAIN vocabulary. No
  constant pass-through or synthetic OOS activity is allowed. Multi-timeframe
  values must be alive and genuinely distinct on TRAIN.
- Precision claims require untouched OOS evidence, meaningful support,
  calibration, class/slice confusion, path/utility quality, costs and
  live-like replay. Code-contract tests alone never prove trading edge.
- `PROJECT_STATE_xau_direction_launch.json` remains `BLOCK` until one exact
  bundle and its newest terminal evidence are explicitly admitted.

## Run authority and resource safety

Source edits, read-only audits and tests are authorized by the active goal.
Entry rebuild, training and evaluation use one validated `--run-id` only to
bind immutable lineage; no separate manual approval token exists. Their exact
evidence contracts decide whether execution may proceed. Paper/demo launch,
live launch, promotion and destructive GX1_DATA cleanup retain their separate
explicit safety boundaries. Documentation never overrides those contracts.

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
  window key, and reuse it on an exact same-attempt rerun. Group-A uses
  immutable 4096-row chunk checkpoints plus a hash-bound completion event;
  the ranker also binds its final matrix checkpoint to run/source/cache/window.
- Every heavy command must enter through `scripts/gx1_capped_run.sh`, whose
  host-wide nonblocking lock forbids overlapping ranker, builder, trainer or
  replay jobs. Lock refusal is terminal failure, never permission to bypass
  the capped runner.
- A watcher that polls for a process must match the exact python command line
  (anchored `pgrep -f '^...python -m module'`), never a substring a wrapper
  shell also contains; a dead job must alert immediately, not wait silently.
- Every shell invocation starts with `cd /home/andre2/src/GX1_ENGINE &&`
  (the working directory can reset between calls; rg then returns silently
  empty results and `python -m gx1...` fails to resolve).
- Never let a pipe swallow an exit code: capture command output to a file and
  append the real `$?`, or check PIPESTATUS.
- Per-row Group-A lookup must remain zero-copy (`int64` timestamps,
  `float32` MTF matrices) and resolve each TF snapshot once. The 2026-07-21
  six-year probe measured ~2,062 complete rows/s; a dtype cast inside
  `_tf_cache_row` regresses to full-matrix copies and is forbidden.
- Decision slices MUST receive an explicit full causal M5 prefix for Group-A.
  Hash and validate exact decision OHLC against it, then fan disjoint row
  ranges over the one shared context. Never reset 60-D1 liquidity at a split
  boundary and never use chunk overlap; long-memory features make both wrong.

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
valid substitute for deletion. This rule applies during every task: clean up
proven-dead code when encountered, rather than accumulating a separate future
cleanup backlog.

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
