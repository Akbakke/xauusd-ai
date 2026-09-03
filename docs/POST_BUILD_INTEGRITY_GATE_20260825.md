# Post-build integrity gate: XAUUSD directional bot

> **2026-09-03 re-entry note:** V9 reached full technical TRAIN+VAL, but this
> is not candidate acceptance or a TEST release. Its selected VAL PnL is
> negative, and the post-restart signed host bridge reports a 390 W physical
> limit rather than the required 160 W. Use
> [`CURRENT_HANDOFF_20260903.md`](CURRENT_HANDOFF_20260903.md) for the current
> gate.

> Runtime-state rule: this gate does not identify a resumable candidate.
> `bash scripts/gx1_handover.sh` verifies the declared recipe/source closure,
> immutable session contract, active pointer and state before reporting it.

Status: **V46 REVIEWED — `PASS_FOR_BOUNDED_SMOKE_RECIPE_AND_DRY_RUN_ONLY`; this is not a training, demo, live, or edge sign-off.**

> **30 August 2026 current-status override:** an immutable technical
> checkpoint bundle now has exact clean-CPU reload parity and a 70,880-row
> VAL-only label journal exists with `TEST ACCESSED: NO`.  The journal's
> negative smoke-label PnL, win share, MFE and MAE are permitted here only as
> model/label plumbing diagnostics; they are not Gate 8 backtest or edge
> evidence. A static positive/open learned multiplier is provisional only on
> the Exit surface, after its direct-input report is bound to selected model,
> VAL, MTF-cache and lifecycle bytes; Entry gates remain strict. Full Exit
> trajectory evidence is also bound to online/target states on bundle load.
> Candidate CUDA has exercised partial TRAIN through checkpoint 640 and a
> fresh-process resume, but it has not reached candidate VAL or emitted a
> candidate bundle. External compute and every execution route remain blocked.

## Runtime amendment — 2026-08-28

The data/feature gate remains PASS, but the requested canonical CUDA proof has
not produced a bundle. Two batch-32 V46 recipes passed their hash-bound
preflight and then stopped safely at the unchanged 70 C core boundary: 71 C /
263.77 W / 8,951 MiB for 10,000 TRAIN rows and 71 C / 261.33 W / 8,951 MiB for
1,000 rows. This establishes a local thermal hold, not a data, feature, target
or VRAM failure. The first batch-8 attempt was intentionally stopped before a
result because its 1,000-row sample applied to TRAIN only, leaving all 70,880
VAL rows. A source repair now bounds both model-compute splits only after the
full data/feature preflight; it changes no V46 bytes, feature or target. One
repaired batch-8 smoke with 32 rows per compute split may run next and its
terminal evidence must be assessed before any repeat. If it cannot create a valid bundle safely, a
separately approved remote canonical smoke must still pass every immutable
recipe and bundle-audit condition below.

**Later runtime amendment — 2026-08-28:** the final fresh 32/32 smoke completed
full preflight, four CUDA optimizer steps, validation, strict load and
post-export structural liveness. It atomically published a diagnostic bundle
inside 63 C / 212.37 W / 8,751 MiB. Commit `c3026c0f` makes bounded smoke use
the separate immutable full-population liveness proof for rare-event variability
while retaining strict candidate checks. Candidate remains strict: immutable VAL
predictions and a smoke-bundle audit are now required, and create no edge, TEST,
demo or live authority.

**Current CPU rebind — 2026-08-28:** the retained V46 source passed a fresh
full-input liveness v10 scan across all TRAIN/VAL sequence values. The
post-rebuild, smoke-manifest, smoke-readiness and trainability reports were
re-issued from that exact evidence, followed by a CPU-only recipe preflight
that rehashed the live Train/Val bytes. All report checks passed; TEST remained
byte-opaque and no CUDA, model training, dataset rebuild or broker activity was
started. These refreshed reports are the canonical state pointers, but they are
still only evidence of wiring and provenance, not an edge or execution sign-off.

## V46 review record — 2026-08-25

**Decision:** `PASS_FOR_BOUNDED_SMOKE_RECIPE_AND_DRY_RUN_ONLY`.

The exact V46 build passed its rebuild chain, full-input liveness, foundation
feature/target, eight-specialist, pretrain, execution-causality, trainability,
and immutable recipe gates.  The canonical smoke wrapper also passed
`--dry-run`; it did not create a bundle, start a trainer, allocate CUDA, rebuild
data, open a broker connection, or place an order.

This is deliberately narrower than `PASS_FOR_TRAINING`.  A future actual smoke
remains an explicit, resource-controlled execution decision; it is not implied
by this record.  There is no PnL, win rate, MAE/MFE, backtest, demo, or live
claim at this stage.

### Immutable V46 evidence

- Chain root: `/home/andre2/GX1_DATA/data/data/prebuilt/V46_20260825T170935Z_CHAIN`.
- Green terminal: `CHAIN_TERMINAL_20260825T195501470746Z_GREEN.json`.
- Current full-input liveness: `audit/ENTRY_FULL_INPUT_LIVENESS_CONTRACT_20260828T174938Z.json`
  (`PASS`; 238 signal fields, 71 continuous context fields, one categorical
  context field, and all eight MTF family surfaces live).
- Pretrain audit: `audit/XAU_DIRECTION_REPAIR_PRETRAIN_AUDIT_20260825T195458720518Z.json`
  (`PASS`).
- Foundation feature / target audits: `audit/foundation_features/ENTRY_FEATURE_FOUNDATION_AUDIT_20260825T200552Z.json`
  and `audit/foundation_targets/ENTRY_TARGET_FOUNDATION_AUDIT_20260825T200719Z.json`
  (`PASS`).
- Eight-specialist audit: `audit/specialist_features/ENTRY_SPECIALIST_FEATURE_GROUP_AUDIT_20260825T200754Z.json`
  (`PASS`; all required families are bound to the 238-field model surface).
- Execution causality: `audit/ENTRY_EXECUTION_CAUSALITY_AUDIT_20260825T2010Z.json`
  (`PASS`; no legacy same-close M5 label and active auxiliary targets are M1-fill bound).
- Exact smoke recipe: `train_recipe_audit_20260825T202900Z/ENTRY_MODEL_NATIVE_SEQ513_TRAIN_RECIPE_AUDIT_20260825T201737076503Z.json`
  (`PASS`, report-only, source commit `980fa0f17a7f2d5b1a92253dd00123935b90c527`).

### Repair captured before execution

The recipe gate exposed an auxiliary-target proof validator error before any
training.  It incorrectly required all 96 global K=96 tail rows to reach the
auxiliary completeness gate, even though the earlier causal-M1 filter can
legitimately remove part of that tail.  The corrected validator still requires
the global tail to be exactly 96, requires exact candidate/emitted accounting,
and forbids non-finite emitted targets; it now permits the bounded remainder
observed in V46 (76 train and 56 validation rows).  The repair is commit
`980fa0f1` and was covered by the contract tests.

### Test-isolation and robustness caveats

- The original V46 test seal remains the authoritative pre-training artifact.
  During this review, a structural manifest query exposed four aggregate test
  count fields; no test parquet, labels, feature values, predictions, or
  performance were read.  Record this as a metadata disclosure, not a model
  selection input.  V46 test results must not be used to tune anything, and a
  fresh explicit final-evaluation sign-off is required before any OOS claim.
- Liveness found genuine train/validation ATR distribution shift on H4/D1.
  It is not a broken feature, but it is a regime-robustness risk to be tested
  with walk-forward validation after a model exists.
- The current entry-fitted-Q target is gross-spread-inclusive research only;
  it has no production authority.  Costs, financing, execution latency, and
  portfolio constraints still need the later evaluation gates.

## Purpose

This is the mandatory, fail-closed review gate between a completed feature/dataset
rebuild and any model training.  It exists to prevent a familiar failure mode:
columns, manifests, or family names appear present, while the underlying feature is
constant, causally invalid, duplicated, unused, or disconnected from the model.

Passing a narrow unit test, a successful parquet write, or a `PASS` from one
manifest is not sufficient.  Training is allowed only after the evidence listed in
every section below has been inspected and each required check has passed.

## Required evidence package

The reviewer must bind the review to one immutable rebuild run and record the paths
and SHA-256 values for at least:

- chain terminal event and `CHAIN_STATUS.json`;
- source-cascade proof;
- signal manifest and feature-ranking artifact;
- M1 and M5 enriched frames and their manifests;
- M1 and M5 feature-base surfaces;
- rebuild preflight report;
- cross-surface audit report;
- final dataset manifest, label coverage report, and terminal dataset decision;
- source commit, requirements hash, input data manifests, and volatility-squeeze
  artifact binding.

Any missing, non-PASS, mismatched, or unhashed required artifact is a hard fail.

## Gate 1 — raw data and temporal identity

- [ ] Instrument is the intended XAUUSD feed, with expected price scale, UTC clock,
  OHLCV schema, and source manifests.
- [ ] Source bars are monotonic, unique, internally consistent (`low <= open/close
  <= high`), and free from unintended synthetic fills.
- [ ] Market closures and gaps are represented as gaps; no sequence is stitched
  across missing physical bars.
- [ ] The chronological train, validation, and sealed-test windows match the run
  declaration exactly.
- [ ] Spread, fees, slippage, and executable prices are either sourced or replaced
  by an explicitly conservative, versioned assumption.  They may not be silently
  omitted from later evaluation.

## Gate 2 — causality, alignment, and ownership

- [ ] Every decision-time value is available on or before the closed decision bar.
- [ ] Higher-timeframe values are joined `as-of` completed source bars only; no
  in-progress M5/M15/H1/H4/D1 value is visible early.
- [ ] M1 and M5 surfaces have the required physical clock alignment and preserve
  observed feature history when a future outcome label is unavailable.
- [ ] Rows with incomplete causal M1 outcomes are excluded from emitted training
  labels without deleting predecessor rows from a model sequence.
- [ ] Each MTF field has exactly one declared owner.  Legacy owners, parallel
  implementations, fallback fields, and future feature reuse are rejected.
- [ ] All source/cache/manifest hashes are mutually bound to the same market tape
  and run identity.

## Gate 3 — every individual feature is alive

For every field exposed to ranking or the model, including mandatory fields:

- [ ] the physical column occurs in the final model input surface;
- [ ] values are finite over its valid support (no hidden `NaN`, `inf`, sentinel,
  zero-fill, or fallback population);
- [ ] it has non-trivial variation on train, validation, and test support;
- [ ] it is not an exact or near-exact duplicate of another feature under a new
  name;
- [ ] its scale, bounds, sign convention, and warmup behaviour match its defined
  semantic contract;
- [ ] it has meaningful availability after warmup rather than a tiny surviving
  population;
- [ ] it is selected or deliberately rejected by a recorded train-only ranking
  decision — not silently dropped between surface and model.

Constant/duplicate/invalid fields must be removed or fixed, then the affected
surface must be rebuilt.  They may not be tolerated as "harmless" dead weight.

## Gate 4 — the eight feature families

Each family needs an input-to-model trace, liveness statistics, causality proof,
and a recorded role in the final surface:

| Family | Must be demonstrated |
| --- | --- |
| Structure | Confirmed swings, prior highs/lows, support/resistance, age/distance and their causal confirmation delay. |
| SMC | Liquidity/structural-state concepts used by the system, with causal event timing and non-empty state changes. |
| Trend | EMA levels, slopes, stack/alignment, regime and regime age across local and higher clocks. |
| Volatility | ATR-derived state, volatility/squeeze features, valid fit binding and variation across regimes. |
| Momentum | Local and cross-timeframe momentum/returns, correctly aligned and non-duplicative. |
| Session | Session/overlap fields, UTC/DST correctness and expected market-hour population. |
| Chart / technical | The declared local technical owners, with valid warmup and runtime values. |
| Candlestick | Body, wick, range and pattern/state fields computed from closed bars only. |

A family counts as present only when at least one of its live, contractual fields
reaches the final model input through the declared route.  A name in a registry,
an unused dataframe column, or a unit-test-only implementation does not count.

## Gate 5 — sequential model surface

- [ ] The Entry and Exit windows have their declared lengths and ordering (including
  the Entry 513-bar contract where applicable).
- [ ] Sequence tensors are composed from physical predecessor rows only, with no
  discontinuity caused by removed labels, closure gaps, or reset leakage.
- [ ] Warmup and confirmation delays are preserved: a swing/support/resistance or
  regime value cannot appear before it is observable.
- [ ] All required M1/M5 and MTF members are aligned at every sequence row.
- [ ] Sequence construction is identical in dataset build, replay/backtest, and
  serving code, or an explicit parity test proves equivalence.

## Gate 6 — directional target and label quality

The system must predict an **executable risk-adjusted outcome**, not merely the
next bar's sign.  The trade decision must allow `long`, `short`, and `no trade`.

- [ ] Label policy, entry timing, horizon/barriers, stop/target/timeout, and
  intrabar ordering are explicit and bound to the run.
- [ ] Outcome labels use only future market data after the decision timestamp;
  feature inputs use none of it.
- [ ] Long and short construction are symmetric where intended and separately
  measured where market microstructure makes them asymmetric.
- [ ] Label coverage, abstention/no-trade population, class balance, and missing
  label exclusion are reported per split and per regime.
- [ ] Costs, spread, slippage, and execution constraints are incorporated before
  later claims of directionality or edge.
- [ ] The diagnostic direction target cannot be mistaken for the executable
  training target.

## Gate 7 — training and selection isolation

- [ ] Imputation, scaling, ranking, thresholding, and calibration are fitted on
  the permitted chronological train population only.
- [ ] Validation chooses configuration; the sealed test set is untouched until a
  final, preregistered evaluation.
- [ ] Sequence/label overlap is purged or embargoed where required.
- [ ] Seeds, data hashes, architecture, optimizer, stopping rule, and all selected
  thresholds are recorded.
- [ ] Family-level and feature-level ablations are run out of sample.  A family
  that provides no robust contribution is identified rather than assumed useful.

## Gate 8 — edge and execution, after training only

No PnL, win-rate, MAE/MFE, or edge claim is permitted before this gate.

- [ ] Walk-forward, validation, and sealed out-of-sample results are separately
  reported after costs.
- [ ] Trade-level metrics include expectancy, net PnL, profit factor, win rate,
  drawdown, MAE before MFE, MFE/MAE, holding time, turnover, and tail loss.
- [ ] Performance is broken out by long/short/no-trade, session, volatility,
  trend/range regime, and significant news/illiquid periods.
- [ ] Thresholds are chosen without touching the sealed test results.
- [ ] Backtest fills follow the same executable pricing, spread, latency and order
  rules planned for OANDA demo execution.

## Gate 9 — operational safety before demo trading

- [ ] Shadow/replay mode agrees with the offline feature surface and produces
  complete decision audit logs.
- [ ] Demo order handling enforces position size, stop loss, maximum daily loss,
  maximum exposure, stale-data checks, and a manual/automatic kill switch.
- [ ] Missing data, failed feature computation, hash mismatch, or uncalibrated
  output causes abstention — never an unguarded trade.
- [ ] Resource limits, checkpoints, recovery behaviour, and disk retention have
  been tested under representative load.

## Sign-off rule

The reviewer must write a dated report containing one of:

- `PASS_FOR_TRAINING`: every required check above has direct evidence;
- `FAIL_REBUILD_REQUIRED`: one or more data/feature/causality/surface checks fail;
- `FAIL_DESIGN_DECISION_REQUIRED`: target, execution, or policy ambiguity remains.

Only `PASS_FOR_TRAINING` authorizes the next stage.  It does **not** authorize
demo or live trading; those require Gates 8 and 9 after a trained model exists.
