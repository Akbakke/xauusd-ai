# GX1 XAUUSD model-native trading engine

GX1 is being rebuilt around one learned XAUUSD Entry decision: calibrated
`LONG`, `SHORT` or `FLAT`. The model must fuse the complete market-evidence
stack and prove its edge through immutable out-of-sample contracts. Missing,
stale or contradictory evidence blocks the system; there is no fallback
direction policy.

Current status: **BLOCKED FOR MODEL/EDGE/LAUNCH**, with one current audited
dataset lineage. V24 rebuilt fresh XAU source through
`2026-07-22T12:05:00Z`, terminalized GREEN at the designed smoke gate and
passed post-rebuild, full-input, pretrain, foundation-feature, all-46-target,
eight-specialist, smoke-readiness and trainability review. Its exact splits
contain 369,081 TRAIN, 5,904 VAL and 4,115 TEST rows. TRAIN has zero dead
signals, zero exact duplicate signal groups and zero unmapped signal/context
fields. This proves data/routing contracts, not direction edge.

No smoke model, bundle, calibration or learned prediction evidence exists, so
no Entry launch is authorized. Smoke execution remains source-blocked until a
canonical immutable recipe-audit producer and the post-smoke bundle-audit
control route exist; manually composing those artifacts is forbidden. The old
Smart520 evidence and rejected V1-V23 lineages are historical only and cannot
be used for training, replay, paper trading, live trading or promotion.

A report-only abstention metadata check is
`BLOCK_ABSTENTION_EMPIRICAL_GATE`: its balanced FLAT-label counts and positive
objective weights are not learned evidence. It read zero parquet; immutable
historical selection-benchmark bytes and exact learned-probe OOT evidence are
absent, so that historical comparison cannot be a pre-rebuild gate. The next
empirical gate is a fresh accepted seq513 dataset/candidate followed by an
immutable proxy comparison and absolute untouched OOT/cost/live-like proof.

The source recovery creates and binds the TRAIN-rank reference before
feature ranking, then routes ranking and dataset construction through one
chain. It serializes all capped heavy jobs with one host-wide lock, checkpoints
Group-A in exact hash-bound 4096-row chunks, including the complete causal M5
context identity, permits one strict checkpoint
retry, and emits immutable schema-v4 terminal chain events. This has source-test proof
and current V24 artifact proof. V22 exposed duplicate liquidity/SR semantics;
V23 proved their separation but exposed an omitted preflight side-effect key;
V24 proves both repairs. V21/V22/V23 large split parquets were deleted while
their small failure/audit evidence was retained.

## Active Entry contract

- XAUUSD only; M5 decision cadence with M5/M15/H1/H4/D1 context.
- 513 ordered signals: 34 genuine base price-state fields plus 479 specialist
  fields. Of those 479, all 378 outputs from twelve registered causal feature
  layers are code-owned and mandatory; only the remaining 101 positions come
  from deterministic TRAIN-only ranking.
- 142 continuous and 5 categorical context fields.
- Eight learned specialists: structure/swing, SMC/liquidity, trend/EMA,
  volatility/compression, momentum/flow, session/regime, chart geometry and
  price-action/candles.
- Twenty-two positively supervised evidence heads feeding one exact learned
  26-group/96-value fusion (`96 -> 128 -> 3`).
- One final direction authority: calibrated model logits and exact
  `argmax([LONG, SHORT, FLAT])`.
- Continual adaptation is offline and immutable: same-bundle row-recomputed
  drift, challenger replay, zero-order shadow, explicit promotion and rollback
  to a prior incumbent. Replay has no direct activation authority; live weight
  updates and post-model direction rules are forbidden. The launch guard
  requires the fresh lifecycle event to bind the exact bundle, serve, active
  Exit and learned-sizing evidence. Promotion additionally requires incumbent
  and challenger on identical immutable price paths with bid/ask-recomputed
  outcomes, absolute challenger side edge and positive lower-95% paired
  improvement. No real lifecycle chain exists yet.
- Learned path, utility, timing, tail-risk, volatility, trade/side hierarchy,
  trendline-rail, validity, position-size and internal Q/V/Advantage evidence
  is mandatory. Q/V never forms a separate policy.
- VAL and TEST must prove that learned LONG timing aligns with realized
  `BOTTOM` outcomes, SHORT timing aligns with realized `TOP` outcomes, and
  Q/V/Advantage aligns with the full counterfactual reward surface. Merely
  finite or non-constant head output cannot satisfy the smoke edge gate.
- Position sizing is learned and must be immutably calibrated, parity-checked
  and journaled. Any label-horizon TEST result is diagnostic only, and no fresh
  accepted sizing result exists for the current contract.
  Paper/live capital remains blocked pending a joint replay with the exact
  adopted active Exit stack and a fresh post-adoption broker runtime-parity
  event. Strict finalizers/validators now require the complete bound per-M1 Exit
  trace and broker-shadow observations, but no fresh real event has passed.
  Fixed 1x is a historical comparison only, never a fallback.

Real trend/session/liquidity/volatility/momentum evidence belongs inside the
model. Retired filters are only the disconnected rules that could veto, flip,
threshold or silently pass through a model decision after inference.
Full-stack coverage is proven by causal timing, field liveness and learned
connectivity, but coverage alone does not prove useful influence or trading
edge. Each family's influence and the fused decision must be proved
empirically on immutable OOS/live-like evidence; duplicate indicator aliases
and future-leaking variants are not additional robustness.

Smoke evidence is non-activating liveness only. Serve-parity v4 separately
requires raw and calibrated class-margin movement from both ablations of every
specialist and from immutable slice replacement of every one of the 26 fusion
groups, plus zero-mask influence from both context tensors and all five
timeframes. Any passive input/group or older parity schema blocks launch.

TRAIN/VAL/TEST identities are explicit and hash-bound end to end: foundation
audits, smoke/adoption, selective-edge prediction, replay, serve parity and
learned sizing all consume the declared manifest/parquet bytes. No stage may
select a split by directory glob, infer it from another split's filename or
accept an unbound artifact merely because it is present in the same directory.

## Start here

```bash
bash scripts/gx1_handover.sh
bash scripts/gx1_handover.sh --check  # continuations with unchanged authority
.venv/bin/python -m json.tool PROJECT_STATE_xau_direction_launch.json
scripts/entry_next_edge_control.sh --help
```

Read `AGENTS.md`, `SYSTEM_MAP.md` and
`HANDOVER_XAU_DIRECTION_REPAIR_20260714.md` before changing the pipeline.
Training or rebuild commands require their exact immutable prerequisites and
one `--run-id` shared by every artifact. The ID is provenance, not approval;
documentation never overrides the evidence gates.

Code lives in this repository. Large datasets, bundles and run evidence live
under `/home/andre2/GX1_DATA` and must not be deleted without an explicit,
verified cleanup decision. Repository cleanup should continuously remove
disconnected scripts, tests and stale documentation once their callers and
evidence value have been disproved.
