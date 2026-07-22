# GX1 project state

Updated 2026-07-22.

## Entry direction

Status: **BLOCK**.

GX1 has no accepted Entry dataset, trained model, prediction evidence, bundle
or launch authority under the active contract. Paper, demo and live Entry
trading remain closed. The absence of proof means no direction and no order;
it must never become a guessed direction, synthetic FLAT, cached decision or
manual overlay.

The active signal contract is
`xau_seq513_model_native_direction_v3`:

- 513 ordered signal fields: 34 base + 479 specialist fields;
- 377 mandatory fields from twelve code-owned causal feature layers;
- 102 deterministic TRAIN-only ranked remainder fields;
- 142 continuous and five categorical context fields;
- sequence length 96;
- five causal timeframe caches with 25 values per timeframe;
- eight learned specialist encoders;
- 26 learned evidence groups producing one exact 96-value fusion;
- one final model-native `96 -> 128 -> 3` LONG/SHORT/FLAT direction path.

All 57 `chart.foundation_*` fields are now part of the mandatory 377-field
prefix. They route to four learned specialists: 19 structure/swing, five
SMC/liquidity, five volatility/compression and 28 session/regime fields. They
remain available beside their higher-order derivations so feature selection
cannot silently discard primitive HH/HL/LH/LL, BOS/CHoCH age, sweep/reclaim,
compression/release, impulse/pullback or session-conditioned structure
evidence. The complete field-by-field routing and duplicate comparison is in
`docs/FOUNDATION_FEATURE_ROUTING_AUDIT_20260722.md`.

The learned cooperation path is the sole direction authority. Features first
enter their specialist and timeframe representations, then learned
interaction, gating and cross-attention blocks combine compatible and
conflicting evidence across families and timeframes. Structure, trend,
liquidity, volatility, momentum, session, price action, path quality and
utility therefore do not vote through separate rules. Their combined evidence
reaches the final three logits; no post-model trend, EMA, regime, support,
resistance, momentum, SMC or MTF rule may change the result.

## Dataset evidence

V19 terminalized GREEN under the former v1 dataset contract only. Its source
reached the last complete M5 bar at `2026-07-21T20:00:00Z` and proved 392,995
rows x 188 columns, all 187 numeric source fields finite and live, full-v3
five-timeframe ownership, exact chronological TRAIN/VAL/TEST splits and all
1,980 input-field/split liveness records. Its immutable terminal event is:

- run ID: `XAU_SEQ513_REBUILD_20260721_V19`;
- terminal SHA-256:
  `5ea7d75fd093677e2d6104ca4e8bfc649ed9dc8693cbab06b163ca7b2fbdc0df`;
- terminal reason: `stopped at smoke gate`.

The corrected post-rebuild readiness event passed with SHA-256
`405f827162779ecea57e01c361a0c73e2d933501dd5f9e3dcfabc977ee3dd920`.
The first real foundation audit then failed closed: V19 contains zero of the
57 required foundation fields in its selected surface and split metadata.
V19 is therefore immutable superseded evidence, not an accepted dataset, and
must not be trained, copied forward or rehabilitated by editing metadata.

The omission had two source causes. The chart builder computed the foundation
layer but the old mandatory registry did not retain it; the ranker's reflective
candidate discovery also did not expose its name registry. Synthetic test
manifests had manually inserted foundation fields and masked the producer
mismatch. The real producer now binds the exact 377 + 102 partition,
foundation version/count/all-selected metadata and the mandatory prefix.

A separate formula audit found inverted volatility semantics in four active
builders. H1/M15 range ratios are `ATR14 / ATR100`, so values below one mean
compression and values above one mean expansion. Relative Bollinger bandwidth
is `bandwidth / mean_bandwidth - 1`, so negative means squeeze and positive
means expansion. One strict transform owner now supplies compression and
expansion pressure to foundation, volatility, chart geometry, chart core and
deep interactions. Release requires lagged compression followed by positive
expansion acceleration; non-positive ATR ratios fail closed.

On 5,808 settled V19 validation rows, all 57 corrected foundation outputs were
finite and non-constant. No field was exactly equal to a field in the former
mandatory surface. This is diagnostic liveness and semantic evidence only; it
does not authorize V19 or prove predictive edge.

V20 (`XAU_SEQ513_REBUILD_20260722_V20`) rebuilt the full source cascade from
canonical roots through `2026-07-22T07:35:00Z`. Its source audit passed on
393,122 x 188 FULL_PLUS rows: all 187 numeric fields were finite and live,
with no constants, exact duplicates or fallback. A fresh TRAIN-only ranking,
513-field manifest and rebuild preflight also passed.

Dataset construction then failed closed before any split was published. The
structural auxiliary-label producer requires
`chart.geometry_channel_position_low_to_high`, but that field was still
ranking-owned and did not win V20 selection. The exact checkpoint retry failed
identically. V20 is terminal `RED` at
`dataset-rebuild-exact-checkpoint-resume`; its terminal SHA-256 is
`0b60ceda8b72f45cc76d83c3e4bb681bc5f190f1b0200a67391140e0a293e606`.
No V20 artifact may be reused.

The repair gives one code-owned registry to all 19 current-bar requirements
used by structural auxiliary-label construction. The dataset builder resolves
signals only through that registry, while the signal contract proves every
requirement has a mandatory candidate. Four geometry prerequisites are now
mandatory, producing the exact 377 + 102 partition without changing the total
513-field surface. This removes the target/ranking dependency; it does not
prove predictive edge.

## Evidence and runtime boundary

Foundation feature, target and specialist audits bind explicit immutable
TRAIN/VAL/TEST manifests and parquet SHA-256 values. Selected-field
learnability is assessed on TRAIN. Every selected field must be finite in all
splits, and every required foundation output, objective, family and source
must be present and live in TRAIN, VAL and TEST. The former split-constant
allowlist is deleted; no allowlist or CLI threshold can soften the fixed
policy.

The internal Q/V/Advantage, TOP/BOTTOM timing, path-quality, utility,
calibration and learned-size heads remain supporting learned evidence, not
parallel live policies. Adoption still requires untouched OOS predictions,
non-degenerate LONG/SHORT/FLAT support, calibration, cost robustness,
row-recomputed replay, all-specialist/all-group serve influence, exact
train-equals-serve parity, joint active Exit replay, sizing parity and settled
broker shadow evidence. No current-contract event proves these conditions.

Live model loading begins from exact launch-declared paths and hashes. Entry
requires a complete 96-row window ending at the latest closed M5 bar, exact
context and causal timeframe evidence. A failed model decision remains
structured direction unavailability and leaves the bucket retryable. Missing,
invalid, stale or session-inconsistent evidence cannot be repaired by an
environment override, artifact search, default value, cached row or synthetic
FLAT.

`PROJECT_STATE_xau_direction_launch.json` is the machine-readable launch
decision. It is `BLOCK`, has no accepted dataset or bundle, identifies V19 as
superseded evidence and V20 as the latest terminal failure.

## Verification state

The active v3 source changes pass:

- the complete affected-area suite: 139 tests, zero failures;
- the full repository suite: 100% pass, five explicit skips, zero failures;
- Python compilation for `gx1` and `tests`;
- `git diff --check`;
- exact contract/count/routing assertions;
- repository and active-hook forbidden-instrument zero-scan.

These are source and contract proofs. They do not prove market edge,
near-perfect precision or live readiness.

## Exit

The retained Exit V3/Exit-IQL chain is a separate contract. Its XGB use and M1
exit semantics are not removed by the Entry cleanup. Shared helpers have
neutral or Exit-owned modules; active Exit math is unchanged.

The Exit-only V3 XGB bridge owns exact 7/41 field validation for two active
Exit consumers; both its import and ordered field contract fail closed. The
retired Entry-IQL registry record has `path=null` and status
`RETIRED_ARTIFACT_ABSENT`, so it cannot act as an Entry fallback.

## Next admissible milestone

Allocate a fresh immutable V21 lineage and rebuild the complete source,
TRAIN-only ranking, v3 signal manifest and TRAIN/VAL/TEST dataset chain. V21
must then pass post-rebuild readiness, foundation feature/target/specialist
audits, smoke manifest/readiness/trainability and a capped smoke training run.
Stop at the first red gate; do not reuse any V1-V20 ranking, checkpoint,
manifest, split or dataset artifact.

Only after smoke passes may a candidate be calibrated and evaluated. Compare a
declared full-history baseline with a TRAIN-only recent-regime challenger,
select and calibrate without touching the final TEST window, then open TEST
once for the declared final gate. Zero FLAT predictions is an automatic hard
red. A candidate still needs replay, serve parity, learned sizing, joint Exit
and shadow lifecycle evidence before paper/demo/live can open.

Selective, well-supported high precision is a feasible research target.
Near-perfect practical precision is not assumed and may only be claimed from
immutable OOS and live-like empirical evidence.
