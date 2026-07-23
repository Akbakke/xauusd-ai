# GX1 project state

Updated 2026-07-23.

## Entry direction

Status: **BLOCK**.

GX1 has one current audited Entry dataset lineage, V24, but no trained model,
prediction evidence, accepted bundle or launch authority. Paper, demo and live
Entry trading remain closed. The absence of model/edge proof means no direction
and no order; it must never become a guessed direction, synthetic FLAT, cached
decision or manual overlay.

The active signal contract is
`xau_seq513_model_native_direction_v4`:

- 513 ordered signal fields: 34 base + 479 specialist fields;
- 378 mandatory fields from twelve code-owned causal feature layers;
- 101 deterministic TRAIN-only ranked remainder fields;
- 142 continuous and five categorical context fields;
- sequence length 96;
- five causal timeframe caches with 25 values per timeframe;
- eight learned specialist encoders;
- 26 learned evidence groups producing one exact 96-value fusion;
- one final model-native `96 -> 128 -> 3` LONG/SHORT/FLAT direction path.

All 57 `chart.foundation_*` fields are now part of the mandatory 378-field
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

V24 (`XAU_SEQ513_REBUILD_20260722_V24`) is the current immutable data lineage.
It rebuilt a fresh source cascade through the last complete M5 bar at
`2026-07-22T12:05:00Z`; FULL_PLUS contains 393,176 rows x 188 columns and its
source-cascade proof is PASS. The chain terminal is GREEN, stopped at the
designed smoke gate, with SHA-256
`aaf5458fa53e83f16c436031650ff7ede322094b2376a9747fbe30f388891e48`.
The terminal producer Git head is `c18a07181140b4fb2838c7fc62a59306cdf4709b`.

The exact V24 split identity is:

- TRAIN: 369,081 rows, parquet SHA-256
  `b4c7455cd16ed01d815af546ca59fb39d7790c242b3a37c8df2bb1cd64a479ee`;
- VAL: 5,904 June rows, parquet SHA-256
  `15488f69ee8ce5b48b458bc9e9e091b2672103ba1a18cbfe4e9d354fd02f1bc3`;
- TEST: 4,115 July rows through 12:05 UTC, parquet SHA-256
  `2cbc718fb8aa7d8122c5847c62ae9950bdb2dc862fccfd09388494398802a666`.

Exhaustive input liveness and pretrain pass. Post-rebuild readiness, foundation
feature, complete 46-target and specialist audits all pass on those same
bytes. The specialist audit proves 513/513 signals and 479/479 selected
features, all eight model contracts, zero TRAIN dead signals, zero TRAIN exact
duplicate groups and zero unmapped signal/context fields. Sparse TRAIN support
includes CHoCH 375, bullish/bearish outside-after-inside 3,345/3,129 and M5
EMA50/200 cross-up/down 1,114/1,114. A six-field D1 exact duplicate group exists
only in June VAL because that short OOS window occupies one regime state; it is
diagnostic OOD evidence, not a TRAIN code duplicate.

V22 had failed this specialist audit because SMC liquidity-pool proximity was
exactly identical to S/R-memory proximity and because sparse events were judged
by the wrong generic floor. V23 proved the semantic separation and canonical
sparse floors, then smoke readiness failed only because preflight omitted the
required `iql_distillation=false` key. V24 proves all six side-effect keys
explicitly false and smoke readiness is READY.

The first V24 trainability audit caught one more source mismatch: it searched
downstream source text for duplicated contract literals even though all four
consumers correctly imported the exact mode and width from the signal-contract
owner. Commit `0f2b9468f396bdfd7d850749fd045294662a9bd4` replaces that check
with AST-proven import and use.

Commits `f08cd90474336b5632c19aa8cd734f6e9bf65f9a`,
`b5a61e21118693491edf4975edec793fbc47d794` and
`bf5c61a00500aa50890f118b6eb41ab5e91bb0c6` then close the exact
source-level smoke-launch gap. One canonical recipe owner and producer now
construct all 162 decision-affecting trainer settings without ambient
pass-through/default values, validate the real split-native pretrain audit and
bind every executable source file by SHA-256. The single control surface now
exposes both recipe production and the existing exact post-smoke bundle audit.

Fresh V24 smoke readiness is READY with SHA-256
`d8c09f0d20e3928b55d38a33d0a4b8fb1d0db5bf29b5a0939db7fc2213f12c9e`;
trainability is READY with SHA-256
`29d03b3fd45b31f5f7c9df64dd985c9f6892ed0c4d5f758ee712e2dfb260508e`.
The immutable smoke recipe is PASS with SHA-256
`fa2404603a435d8dc47e26fb2d7345e25b3a2d81b3760e9a0a6c7cf1078ec040`,
source commit `bf5c61a0`, run ID `XAU_SEQ513_SMOKE_20260723_V1`, one
epoch, 10,000-row cap and 30G/2G memory/swap caps. Its exact public wrapper
dry-run passes. No smoke training has started and the declared output bundle
does not exist. V21/V22/V23 large rejected split parquets have been deleted,
while their small terminal/manifest/audit evidence remains.

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
decision. It is `BLOCK`, identifies V24 as the current audited dataset
lineage, binds the non-activating recipe/dry-run evidence and has no accepted
bundle or launch evidence.

## Verification state

The active v4 source and V24 data contracts pass:

- the full repository suite: 100% pass, five explicit skips, zero failures;
- Python compilation for `gx1` and `tests`;
- `git diff --check`;
- exact contract/count/routing assertions;
- V24 post-rebuild, foundation feature/target/specialist, smoke-readiness,
  trainability, immutable 162-setting recipe and exact wrapper dry-run;
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

Run the exact capped one-epoch/10,000-row V24 smoke recipe through the public
wrapper, then immediately audit the produced bundle through
`model-native-smoke-bundle-audit`. Stop at the first red gate; do not
hand-author an authority artifact or bypass the control surface. Zero FLAT
predictions or missing/passive required evidence is an automatic hard red.

Only after smoke passes may a candidate be calibrated and evaluated. Compare a
declared full-history baseline with a TRAIN-only recent-regime challenger,
select and calibrate without touching the final TEST window, then open TEST
once for the declared final gate. Zero FLAT predictions is an automatic hard
red. A candidate still needs replay, serve parity, learned sizing, joint Exit
and shadow lifecycle evidence before paper/demo/live can open.

Selective, well-supported high precision is a feasible research target.
Near-perfect practical precision is not assumed and may only be claimed from
immutable OOS and live-like empirical evidence.
