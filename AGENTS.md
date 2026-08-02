# GX1 agent constitution

## Scope freeze — the only active work

`GX1_RULES.md` is the short, binding scope contract. GX1 is **offline shared
featurebase only**: immutable XAUUSD snapshot → one shared featurebase →
Entry M5 `LONG/SHORT/FLAT` and Exit M1 `HOLD/EXIT_NOW` → offline train/OOS/
replay evidence. Entry and Exit use the same eight feature owners, formulas,
normalization and lineage; resolution is the only intentional difference.

Do not drift outside this path. Live, paper, demo, broker, daemon, polling,
watchdog, live-tail admission, promotion, launcher and continual-drift or
online-adaptation work are forbidden. Do not add a feature family, duplicate
owner, compatibility lane or operational route. Reuse exact immutable caches
and append only after overlap/hash proof; never recompute full history merely
to process a new tail. Invalid evidence stops closed. Historical live and
adaptation text below is reference only and cannot expand the active scope.

## Current objective and status

Build one XAUUSD model bundle that learns tops, bottoms, abstention and
position lifecycle from the full evidence stack. The same shared encoder
selects Entry `LONG/SHORT/FLAT` and Exit `HOLD/EXIT_NOW`. Near-perfect
practical precision is the target, not a current claim. The system stays
closed until immutable out-of-sample contracts prove both edges.

Current system status is **BLOCK**. There is no accepted unified model bundle,
no empirical Entry/Exit edge proof and no launch authority. Old Smart520,
external decision-bridge, anchored Entry and Entry-IQL evidence is retired and
cannot authorize a run or launch.

V26 (`XAU_SEQ513_REBUILD_20260725_V26`) was the latest completed pre-V4
dataset lineage:
the first dataset built end to end on an event-local strict native-v3 M5
tape (2019-01-01→2026-07-24T20:55, 369,303/5,904/4,776 split rows), terminal
GREEN with all dataset/readiness audits PASS. Its smoke training
`XAU_SEQ513_SMOKE_20260725_V8` — the first run on the fully repaired
substrate — ended `TRAIN_FAIL_NO_BEST_STATE` with total FLAT collapse (VAL
100% FLAT from epoch 3, 58 slice failures, path auxiliaries at chance); no
checkpoint or bundle exists. Seventeen never-executed post-audit boundaries
were repaired in existing owners during that campaign (DECISION_LOG
2026-07-25). V25 is chain-RED evidence; V24/V7 below are historical.
On 2026-07-29 the rejected V18 bundle plus the stale V19/V26 dataset, audit
and smoke-manifest bytes were retired from launch authority and deleted
through the immutable cleanup owner. Their measured results remain historical
chronology only. No dataset is currently admitted. The event-local native
source and frozen V4 cache remain input evidence, not model or launch
authority.
V24 (`XAU_SEQ513_REBUILD_20260722_V24`) was the previous immutable failed
dataset/training lineage. It rebuilt a fresh XAU source cascade through the
last complete M5 bar at `2026-07-22T12:05:00Z` and terminalized `GREEN` under
the pre-V7 dataset gates.
The exact 369,081 TRAIN / 5,904 VAL / 4,115 TEST rows bind 513 ordered signals,
142 continuous context fields, five categorical fields, exhaustive 1,980-row
input liveness and pretrain target/polarity evidence. Post-rebuild, foundation
feature, all-46-target and eight-specialist audits pass on the same six split
bytes. TRAIN has zero dead signals, zero exact duplicate signal groups and
zero unmapped signal/context fields. One exact six-field D1 duplicate group in
June VAL is recorded as a truthful one-regime OOS observation; it is not a
TRAIN duplicate and is not fabricated away. The post-V7 full-pipeline audit
supersedes that former admission: six signed dip-MFE targets are clipped to
zero, so V24 must be rebuilt and cannot authorize another training run.

V22 previously exposed two exact TRAIN duplicate SMC-liquidity/SR pairs and a
sparse-event policy mismatch. V23 proved those repairs and all specialist
contracts, then smoke readiness failed solely because the preflight producer
omitted the required explicit `iql_distillation=false` side-effect key. V24
proves the repaired six-key preflight and reaches smoke/readiness/trainability
review. The first V24 trainability attempt then exposed a raw-source-text
contract check that rejected correct imports from the signal-contract owner;
commit `0f2b9468` replaced literal duplication with AST-proven import/use and
the immutable trainability review now passes.

This was a data and contract breakthrough, not a model or trading-edge
breakthrough. Commits `f08cd904`, `b5a61e21` and `bf5c61a0` closed the
former source-level smoke-launch gap. Seven capped executions then failed closed
without a bundle: V1 exposed an
over-strict static-versus-emitted aux-target check; V2 exposed that the trainer
incorrectly treated V24's dataset-build ID as the new training/output ID; V3
crossed both walls and completed the M5/M15/H1/H4/D1 prebuild, then exposed an
invalid non-negative requirement on signed, spread-aware MFE. The same review
found silent zero-clipping of signed MFE and path-quality regression targets;
V4 crossed that wall, built the complete 72.71 GB tensor surface, entered the
first batch and exposed an incorrect `y_direction` batch-alias requirement in
the mandatory MTF direction head. No optimizer step completed in V1–V4.
Commits `9459babe`, `b986c8db`, `c9e2569f` and `f05b3390` repair those
boundaries. V5 then completed one full train/validation epoch with optimizer
steps, but direction-slice evidence and auxiliary tradable/bad-path AUC
failed the fixed checkpoint gates. No best state or bundle was admitted; this
is empirical model-quality rejection, not a reason to soften a contract. V6
then completed six epochs, briefly reached near-label global balance, but
never passed local slices or auxiliary health; it ended with LONG starvation
and clean-edge/path-quality head collapse. No bundle was written.

Commit `37128985` records exact epoch-wide health for specialist,
timeframe and family×timeframe gates and makes it part of checkpoint admission
at the unchanged minimum. It strengthens direction-neutral gate balance
without encoding a live direction. Recipe schema v2 binds distinct
`run_id=XAU_SEQ513_SMOKE_20260723_V7` and
`dataset_run_id=XAU_SEQ513_REBUILD_20260722_V24`; launch derives the latter
from post-rebuild plus all three manifests, and trainer/bundle contracts
revalidate the separation.

V7 then ran six full TRAIN/VAL epochs before the hard-red slice stop emitted
`TRAIN_FAIL_NO_BEST_STATE`. Raw accuracy peaked at 0.403455 only with 85.1118%
FLAT; the final epoch predicted 71.4092% SHORT, failed 32 slices, retained
bad-path/survival AUC 0.478/0.514, six cross-head collapses and near-zero
specialist/family×TF minimum use. No checkpoint or bundle was written.

The independent full-pipeline audit in
`PIPELINE_AUDIT_XAU_20260723.md` found two P0s and multiple P1s. Source now
repairs target/objective semantics,
no-replacement sampling, conditional auxiliary evidence, exact recipe/M5/MTF
identity, full-TRAIN normalization, direct 142+5 family ownership,
all-22-head/26-group influence, atomic bundle/event publication, exact
same-candidate bundle identity, identity-bound approval/vedtak, recoverable
candidate-to-launch finalization, runtime lease rechecks and execution
fail-close. The public finalizer serializes the canonical registry/state
targets, requires a pre-existing one-time vedtak, binds the accepted bundle
and exact single-exposure operating point, and either commits both targets or
restores both with durable failure evidence.

The adversarial re-audit proved that the former separate Exit chain was both
causally invalid and an architecture violation. Its implementation,
contracts, registry roles and artifacts are retired. Entry and Exit now have
one authority boundary: the exact same immutable bundle and shared encoder
must emit calibrated `LONG/SHORT/FLAT` Entry logits and `HOLD/EXIT_NOW` Exit
logits. Exit must consume a hash-bound frozen Entry snapshot plus an exact,
contiguous, closed-M1 post-entry path envelope. No auxiliary model, bridge,
overlay, runtime rule or synthetic HOLD is permitted.

The existing OANDA owner can publish immutable native M1/M5 source bundles,
and pair generation `077e5419…` remains frozen historical source evidence.
It is not a current serving identity. Native schema v4 successors bind the
exact parent manifest by CAS, reuse verified historical chunks, refetch only
one bounded overlap plus the new tail and refuse any overlap rewrite. The pair
owner publishes the candidate event before pointer activation. The live-tail
contract requires two consecutive events, stores a static launch anchor and
revalidates the newest admission against the exact inference pair before each
new Entry and order. Stale evidence must block new exposure without blocking
same-bundle Exit recovery. No real successor/admission has been published or
launch-bound.
The unified Exit head, lifecycle dataset/loss, exact closed-M1 runtime envelope
and canonical full-TEST producer also exist in source. Runtime and replay
retain the same frozen shared Entry representation, call only the same
candidate bundle, use the same path transform and bind exact model-output
hashes. No producer output, serve parity or trained unified artifact exists,
so runtime still fails closed. V24/V7 remain immutable failure evidence. Every
fresh empirical dataset, model, edge, sizing and launch gate remains open.

ROADMAP.md is the current ordered execution plan. It records historical
incidents and the next admissible stages, but no V4 rebuild or training is
active and it never overrides the machine-readable launch BLOCK.

Read in this order:

1. `AGENTS.md`
2. `PIPELINE_AUDIT_XAU_20260723.md`
3. `SYSTEM_MAP.md` — read it before grepping for artifact producers or
   feature-column owners; it is the one-truth map of the data DAG.
4. `HANDOVER_XAU_DIRECTION_REPAIR_20260714.md`
5. `PROJECT_STATE_xau_direction_launch.json`
6. relevant code contract and test files

Run `bash scripts/entry_next_edge_control.sh handover` for a read-only takeover
snapshot.

Token/credit discipline:

- Run the compact snapshot once per root takeover. On continuations and in
  subagents, run `bash scripts/entry_next_edge_control.sh handover --check`
  first. `authority_fingerprint` binds the ordered documents and launch state;
  when it is unchanged, do not reread them or repeat an all-Markdown
  inventory. `worktree_fingerprint` separately binds HEAD, the complete
  tracked diff and every untracked file byte. If it changed, inspect the
  current diff and rerun the affected contracts before resuming even when the
  changed-path count and authority fingerprint are unchanged.
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
- the 479 specialist fields = all 378 outputs from twelve code-owned causal
  layers in exact registry order + 101 deterministic TRAIN-only ranked fields;
- 142 ordered continuous context fields;
- 5 ordered categorical context fields;
- sequence length 96;
- exact V4 M5, M15, H1, H4 and D1 market context: 111 ordered
  timeframe-native fields at each resolution, 555 feature×timeframe cells,
  40 family×timeframe cooperation routes and all eight non-empty specialists;
- cross-timeframe attention, positional encoding, FiLM context conditioning
  and learned timeframe scales;
- eight specialists: structure/swing, SMC/liquidity, trend/EMA,
  volatility/compression, momentum/flow, session/regime, chart geometry and
  price-action/candles.

Every genuine trend, session, structure, liquidity, volatility, momentum,
chart, candle, path-quality and utility input/target stays in the learned
path. Removing an old filter means removing competing decision authority, not
amputating its evidence.

### Causal multi-resolution evidence pyramid

Multi-timeframe means that all eight specialist families have a real,
timeframe-native causal surface on M5, M15, H1, H4 and D1. Eight M5 specialists
beside five generic timeframe tokens is not this contract. The learned path
must expose and audit the complete family×timeframe surface; a missing, dead,
unrouted or non-influential cell fails closed.

History becomes coarser as it becomes older. Fine M5 context is retained only
for the recent horizon declared by the immutable recipe; progressively older
context is carried by M15, H1, H4 and D1 instead of duplicating distant history
at M5 resolution. The exact per-timeframe windows are explicit recipe inputs
and must form a strictly increasing wall-clock coverage pyramid. They are
selected on declared TRAIN/VAL evidence, never hidden in a wrapper, ambient
environment or model default.

Feature relevance is learned independently for each feature×timeframe and may
change with age, regime and the other evidence. A feature may be weak on M5 and
decisive on D1, or the reverse. No fixed timeframe preference, manual
confluence rule or hard-coded live direction weight is allowed. Per-timeframe
TRAIN-only normalization gives every cell a comparable numerical path; shared
family semantics, timeframe-specific normalization/gates, temporal encoding and learned
cross-family/cross-timeframe cooperation determine conditional influence.
Attention or gate values alone are not influence proof: immutable VAL/TEST
ablation must move raw and calibrated class margins for every retained
family×timeframe route, while window and architecture selection remain
TRAIN/VAL-only.

Cache reach is not model-window reach. The active cache contract must publish
only fully closed resample buckets, and training must prove the exact declared
M5/M15/H1/H4/D1 window at the first and last decision of TRAIN, VAL and TEST.
The preflight cache check proves causal warmup/reach only; it may never claim a
shared 96-bar MTF window.

All advertised output heads must be present, trained with positive weight,
exported and audited. This includes direction, MTF direction, specialist
fusion/gates, path/MFE/tradability/bad-path/clean-edge/survival evidence,
trade-side hierarchy, side validity, trendline rail, timing/tail/volatility,
TF agreement, position size, counterfactual action value and expectile value.

Forward-outcome domains are exact. Spread-aware MFE and path quality are
signed and must remain signed through validation, scaling, train loss and
validation loss; MAE is a non-negative adverse magnitude. Silent clipping,
absolute-value conversion, target substitution or parked-zero aliasing is a
forbidden target rewrite.

Head liveness alone is not learned evidence. Current target audit requires the
complete canonical 46-target surface in TRAIN/VAL/TEST. Current immutable
prediction/smoke evidence must prove on VAL and TEST that LONG adverse-turn
timing identifies BOTTOM pockets, SHORT adverse-turn timing identifies TOP
pockets, Q ranks the counterfactual reward-best action, V tracks learned max-Q
and Advantage equals Q minus V. Missing old-schema fields fail closed; do not
forward-fill them.

Smoke liveness is not a pass-through or direction-influence proof. Only the
current serve-parity v11 contract may supply that launch evidence: both
specialist ablation methods and all 26 exact fusion-slice replacements must
move class-centred raw and calibrated logits above the immutable row/epsilon
floors. The same applies to both context tensors, all five timeframes and all
40 family×timeframe routes. All 555 feature×timeframe gates must be finite,
ordered, non-saturated and context-responsive. In addition, all 1,723 numeric
routes must have sampled local raw and calibrated class-margin sensitivity:
513 sequence-field routes, 513 snapshot-field routes, 142 continuous-context
routes and 555 MTF feature×timeframe routes. Each of the five categorical
context fields must move both surfaces under a valid next-category
counterfactual. These are local gradient/counterfactual proofs, not a claim
that individual zero ablation is globally causal. Missing or pre-v11 evidence
fails closed.

The exact Entry evidence declaration has 22 heads. Their ordered evidence
feeds one learned 26-group/96-value direction fusion (`LayerNorm(96)`,
`96 -> 128`, GELU, `128 -> 3`) before immutable calibration and exact
three-class argmax. The same model adds one positively trained unified Exit
action head; it does not join the 26-group direction fusion and cannot be
trained or swapped after the unified candidate is selected.

### Value origin and evidence class

Every decision-affecting number has exactly one legitimate origin: a named
constant in a contract owner, a statistic fitted on real declared data, or an
explicit CLI/recipe input. If the origin cannot be named in one sentence, the
value is a guessed default and fails closed. Never invent a magnitude to make
something work; when a value must change, adopt the convention the surrounding
code already uses and name it. Removing an exception is allowed, inventing a
constant is not.

Synthetic, random, placeholder or toy-dimension data proves only that code
runs. It may never support a conclusion about production behaviour, justify a
code change or back a claim. Conclusions require real declared bytes at real
contract dimensions, or a proof from source and algebra that holds independent
of data. Every claim carries its evidence class — proven from source, measured
on real data, measured on synthetic data, or unproven — and a claim whose
evidence turns out weaker than stated is downgraded or withdrawn on the spot,
with the withdrawal recorded. A diagnostic instrument reports only what is
valid where it runs; a field that cannot be measured there is omitted, never
emitted as a zero or placeholder that reads as a result.

### Every decision-affecting value is declared by the caller

Rule 14 forbids ambient values and wrapper defaults, and it covers more than
recipe keys. Seven hidden defaults were found in one campaign, each a value that
changes the result supplied by something other than the caller: per-timeframe
lookback windows hardcoded in both training wrappers and disagreeing sixfold, a
`GX1_MTF_TAPERED` environment ladder nobody had run, three windows that reached
the argument parser but never the call site, a model reading the global length
instead of the per-timeframe one, a lineage hash taken from an imported constant
rather than from the manifest the artifact published, and `--num-workers 0`
costing an order of magnitude in throughput.

A wrapper literal is a default. An environment variable is a default. A constant
imported into a consumer instead of read from the artifact is a default. If the
caller cannot state the value, it is not declared — make it a required input and
let the chain fail closed when it is missing.

Artifacts declare what they are, and consumers verify against that declaration
rather than against an assumption. An unknown contract, a width mismatch, a
column-order mismatch, or surfaces that disagree with each other all fail closed.

### Measure where the decision is made

A threshold compared against a sampled statistic must be at least that
statistic's sampling error at the sample size where the comparison is actually
made, or the comparison must be taken on the complete declared population. A
tolerance tighter than the noise of the quantity it judges does not measure the
quantity — it trains on, or fails on, the noise. State the sample size and the
resulting bound whenever a threshold is introduced or moved.

Measurements are taken where the decision is made: on the rows the model trains
or serves on, at the time the quantity is live, after any declared warmup, and
through the same index mapping the model uses. When a gate reports a failure,
prove the gate looked at the right rows before believing it about the system.

Five defects of exactly this shape have been found and repaired: gradient norms
read after the epoch cleared them, multi-timeframe liveness sampled from
indicator warmup, a liveness index built from a subsampled length but applied to
the full arrays, deadness judged on 1,024 of 369,303 rows, and a 0.02
prior-match tolerance against a batch rate whose standard error is 0.0625.

### Advanced enough to trade, never more

Complexity must earn its place by removing a real failure mode. Before writing
code, prefer in order: measure the existing system, change one recipe value,
extend the existing owner, then add something new. When two designs fail closed
equally well, keep the smaller one. A measurement that eliminates a hypothesis
in ten minutes outranks a mechanism that might address it.

### Profile-separated checkpoint admission

`_checkpoint_admission_ok` in the trainer is the one owner of checkpoint
admission. `candidate` requires auxiliary head health, active head health and
cooperation gate health, unchanged. `smoke` admits on active-head liveness
plus non-degenerate class support only; auxiliary and cooperation health stay
computed, logged and journaled as diagnostics. This is a trainability
ratchet, not an acceptance change: a smoke bundle carries zero edge,
promotion or launch authority, and the smoke bundle audit, candidate
    readiness, selective-edge, replay, current serve-parity v11, sizing, joint Exit,
lifecycle and launch finalizer contracts are unchanged. Only a candidate
bundle may enter the acceptance chain. Never widen smoke admission further,
and never let a smoke result stand in for candidate evidence.

### Sizing boundary

The learned `position_size` head is mandatory and must be trained from the
realized future path target, parity-checked and journaled. Label-horizon TEST
utility/exposure/drawdown is a sizing-head diagnostic, not capital authority.
No fresh accepted current-contract sizing result exists.
Paper/live additionally requires a joint sizing replay with the exact unified
bundle and a fresh post-adoption broker runtime-parity event. The joint proof
must cover full TEST and bind the complete per-M1 unified-model
HOLD-to-`EXIT_NOW` trace for every non-FLAT row; runtime parity is broker-live
shadow-only and submits zero orders. Until both pass, emit no order. Never
substitute fixed units or multiplier `1.0`; historical fixed 1x may be used
only as an explicitly named comparison benchmark.

### Exact contracts; no compatibility lane

The model-native lane has no external decision bridge, neutral constants,
direction anchor, legacy direction mode, separate Entry or Exit policy,
post-model filter or alternate launcher. One immutable bundle and shared
encoder own both decisions. Learned auxiliary heads may exist only inside that
model, with positive objectives and exact export/serve evidence; they never
become a second policy.

The former separate Exit stack is permanently forbidden. Do not reintroduce
an auxiliary tree model, V3 scorer, Exit-IQL policy, Strategy-F overlay,
hand-written close rule or compatibility bridge under a new name. The unified
Exit head must be trained from causal lifecycle samples and must bind the
frozen Entry snapshot and exact closed-M1 path. Missing or invalid evidence is
an error, never `HOLD` or `EXIT_NOW`. Do not preserve dead arguments or
outputs for compatibility; exact consumers either match the current schema or
fail.

A same-bundle `EXIT_NOW` is durable execution intent. Persist it before broker
mutation, recover its journal record idempotently after restart, and retry the
same close until broker reconciliation resolves it. Never run a later model
bar that can replace pending `EXIT_NOW` with `HOLD`.

Lifecycle M1 authority is an immutable native/pair manifest that revalidates
the stored raw OANDA responses and literal `complete=true` candles. A
standalone parquet path, collector snapshot, inferred cadence or producer-code
claim is never sufficient authority for training, replay or serving.

Continual Entry adaptation has one source path: immutable same-bundle drift,
replay-readiness v2, offline challenger, zero-order shadow, explicit promotion
and rollback to a prior incumbent. Replay never activates launch directly.
Shadow promotion requires identical incumbent/challenger paths,
bid/ask-recomputed outcomes, absolute candidate LONG/SHORT edge and positive
lower-95% paired improvement globally and per supported direction/context.
Live/online weight updates, post-model direction rules and stale lifecycle
events are forbidden. Launch must cross-bind the fresh activating lifecycle
event to the exact accepted bundle, serve, unified Exit and learned-sizing
evidence; otherwise fail closed.

## Evidence and artifact rules

- Use explicit absolute immutable paths and content hashes. Never select by
  glob, mtime, lexical version or a path containing mutable `latest`.
- Bind dataset splits, ordered fields, recipe, source revision, model state,
  metadata, lock, calibration, predictions, replay journal and audit events.
- Validate file identity again at every authority boundary. A report saying
  `PASS` without proving its bound bytes is not evidence.
- Source-wiring audits must validate the exact contract owner and actual use.
  Do not require consumers to duplicate resolved mode/dimension literals merely
  so a raw-text grep can find them; conversely, an unused import is not proof.
- Newest terminal event for the same family wins. Newer red, malformed or
  incomplete evidence invalidates older green evidence.
- Current takeover fields may bind only artifacts admissible at the current
  stage. When no dataset is admitted, current smoke evidence is null; a
  historical recipe remains diagnosis and is never a handover prerequisite.
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
Entry rebuild uses one validated `--run-id` across the complete immutable
dataset-build lineage. Training/evaluation uses a new `--run-id` for its own
output lineage and a separate launch-derived `dataset_run_id` for the exact
input lineage; operators cannot supply or override the latter. Collapsing or
mixing those roles fails closed. No separate manual approval token exists.
Their exact evidence contracts decide whether execution may proceed.
Paper/demo launch, live launch, promotion and destructive GX1_DATA cleanup
retain their separate explicit safety boundaries. Documentation never
overrides those contracts.

Before a heavy authorized run:

- inspect `git status --short`, disk, RAM and active Python processes;
- do not collide with valid persistent collectors or unrelated dashboard
  processes; verify that any claimed canonical builder still has an admitted
  current CLI before preserving it;
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
- Model-native normalization MUST fit the complete physical TRAIN population
  before any cap, sampler or weighting step. The exact 513+142 order,
  categorical domains, current-bar alias ownership and five-timeframe causal
  source-row hashes are bundle/model state. VAL, TEST, replay and live never
  refit or accept an external scaler path.
- Every 142+5 context field MUST have exactly one family owner and enter that
  specialist before cross-attention. Alias count is derived from the exact
  ordered signal surface, never pinned from a historical dataset.
- Final bundle/event names MUST remain absent until hidden staging, byte/hash
  validation and `fsync` pass. Bundles require the exact commit manifest and
  atomic no-replace directory publication; immutable JSON events use the same
  visibility rule.
- Launch authority cannot come from `.env` or a nonempty string. A future
  ALLOW must bind the newest immutable one-time approval, complete launch-state
  payload, exact bundle commit and unchanged runtime lease. Revalidate before
  every new exposure. Missing broker `trade_id` blocks close execution; it
  never authorizes a counter-market order.

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
- `PROJECT_STATE_artifacts.json`: unified-bundle artifact registry. It has no
  separate Entry/Exit policy role and cannot override the launch contract.
- `/home/andre2/GX1_DATA`: large immutable data, bundles and evidence.

Do not create a parallel implementation when an active owner exists. If a new
shared helper is truly necessary, move all consumers to it and delete the old
owner in the same bounded change.

Extend the existing script, contract or control route whenever the requested
behavior belongs to its current responsibility. Do not create a new versioned
script for a minor edit, compatibility spelling or local workaround. A new
file is admissible only for a genuinely new, bounded authority/responsibility
whose inclusion in an existing owner would mix contracts; document why it is
new and route it through the existing public control surface.

No dashboard, watchdog, launcher or systemd unit may advertise or restart a
removed CLI/daemon mode. The implemented snapshot successor/publication route
must execute two fresh events and bind their admission before launch; source
implementation alone is not authority. Runtime staleness rejection is a
second wall, not a substitute for launch-time completeness.

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

## Host-capacity hard stop

This machine has a 43 GiB RAM envelope. Every heavy offline producer,
dataset build, audit, train, selective-edge run or replay MUST enter through
`scripts/gx1_capped_run.sh`. The runner is the only capacity authority and
enforces: one heavy job, `MemoryMax/MemoryHigh <= 14G`, swap `<= 1G`, at least
16G host-available RAM before launch, two CPU cores and one numerical-library
thread. A request above those limits, missing host state, lock contention or a
missing cgroup is a hard failure; never bypass it, lower the guard, run in the
background or start a second copy. Large intermediate files do not justify a
higher RAM cap. A partial output after a cap, crash or reboot is invalid until
the producer's immutable completion manifest and all hashes pass.

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
