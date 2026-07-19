# GX1 project state

Updated 2026-07-17.

## Entry direction

Status: **BLOCK**.

There is no accepted model-native seq513 bundle. The active source contract is
513 signals, 142 continuous context fields, 5 categorical context fields,
sequence length 96, five timeframes, eight specialists and one calibrated
`LONG/SHORT/FLAT` direction authority. Twenty positively supervised heads feed
one exact learned 23-group/75-value fusion (`75 -> 128 -> 3`).

The 513 signals are 34 code-owned base fields plus 479 specialist fields. The
first 305 specialist fields are every output from ten registered causal
full-stack layers and are mandatory in exact registry order. Only the final
174 positions are chosen by deterministic TRAIN-only ranking. This prevents
feature selection from ranking away an entire trend, session, liquidity,
structure, volatility, momentum, price-action, support/resistance or MTF
evidence layer.

All earlier Entry datasets, bundles, reports and promotion records are rejected
by the current exact contract and cannot override the launch block.

Source contracts and focused tests are being hardened. No fresh dataset build,
training run or OOS candidate result has been produced in this cleanup, so no
empirical precision or launch-readiness claim exists.

`PROJECT_STATE_xau_direction_launch.json` is the machine-readable Entry launch
decision. Both it and the artifact guard must admit the same immutable bundle
before Entry resolution can succeed.

## Evidence boundary

All real structure, trend, liquidity, volatility, momentum, session, price-
action, path-quality and utility evidence remains in the learned model. Old
post-model filters and manual sizing overlays are retired because they were
competing authorities, not because their underlying market information was
unwanted.

The eight learned specialists cover structure/swing, SMC/liquidity, trend/EMA,
volatility/compression, momentum/flow, session/regime, chart geometry and
price-action/candles. Their evidence is fused with hierarchy, MTF, path and
utility objectives before the final calibrated three-class argmax. None is a
separate live direction rule.

The learned size head is mandatory and has no implicit capital fallback. Its
logit and prediction must survive decision, state, journal and review parity.
Hash-bound calibration, the account grid and any label-horizon TEST controls
are sizing-head diagnostics only; no fresh accepted current-contract result
exists. Execution admission remains structurally blocked
until a joint sizing-only replay binds the exact adopted active Exit stack and
a fresh post-adoption broker runtime-parity event. Missing proof means no
order, never a silent multiplier `1.0`; fixed 1x is only a named historical
benchmark.

## Runtime boundary

One exact runtime evidence contract is shared by the model-native decision,
`TradeState`, journal persistence/recovery and daily review. It accepts neither
missing auxiliary evidence nor retired overlay fields. Entry also requires an
exact 96-row window ending at the latest closed M5, a fixed 90-second decision
limit after the bar becomes available, and a fixed 390-second canonical-cutoff
age limit. Failure emits no direction and cannot be softened by an environment
override, cached row or synthetic `FLAT`.

## Exit

The retained Exit V3/Exit-IQL chain is a separate contract. Its XGB use and M1
exit semantics are not removed by the Entry cleanup. Shared helpers have
neutral or Exit-owned modules; active Exit math is unchanged.

## Next admissible milestone

The repository-wide audit is complete (2026-07-17): stale references,
zero-reachability files and duplicate owners are removed, contracts are
hardened (mandatory 305-prefix order, launch-JSON partition constants, one
latency owner) and the full suite is green (1341/0). The next admissible work
is the fresh seq513 rebuild per the runbook in
`HANDOVER_XAU_DIRECTION_REPAIR_20260714.md`: explicit `--vedtak`, green
preflight, rebuild, then smoke with the abstention criterion (DECISION_LOG
2026-07-17) — a smoke with zero FLAT predictions is hard-red by definition.
No rebuild/training or empirical precision result exists yet.
