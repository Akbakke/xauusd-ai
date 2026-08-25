# Post-build integrity gate: XAUUSD directional bot

Status: **PENDING — V44 must finish before this document can be signed off.**

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
