# Direct Entry outcome hypothesis — 2026-09-25

Status: predeclared research hypothesis; no target extraction, fit, model change, or optimizer update has been run. TEST remains sealed.

## Objective

Replace the Entry training signal that copies a frozen Exit-Q at the first post-fill state with direct, side-specific future market outcomes from the existing M1 BID/ASK history. Preserve all current causal Entry inputs, features, feature families, and timeframes. Keep the existing Exit-facing `entry_action_q_bps` semantics unchanged; any direct-outcome prediction must have separate output semantics so it cannot silently alter Exit inputs.

The hypothesis is that direct, executable LONG/SHORT outcome targets provide more useful entry feedback than the current Exit-Q proxy, and that a separately learned wait value can make Entry selective. This is a research hypothesis, not a profitability claim.

## Target definition to bind before any data run

- Decision features are only those available after the completed M5 bar. Use the existing causal M5-to-M1 quote contract and source arrays; do not rebuild or broaden feature inputs.
- Compare two fixed schedules on identical eligible dates and labels: every valid M5 decision, and every third M5 decision as M15 cadence. Higher-timeframe features remain unchanged.
- First research label horizon: 19 M5 intervals (95 minutes) from the completed decision timestamp. This is a fixed target-measurement horizon, not a production holding-time rule.
- Reuse the existing exact clock: an M5 bar is timestamped at its UTC start; its completed decision is at start + 300 seconds; entry uses the exact M1 open quote at that decision time. LONG enters ASK and exits/marks BID; SHORT enters BID and exits/marks ASK. The outcome endpoint is the exact M1 open 95 minutes after the decision. If the quote or any required minute in the path is missing, mark the label invalid. Never interpolate or forward-fill.
- Apply the already versioned prospective cost policy as a scenario: observed side-correct BID/ASK spread, adverse slippage 2 bps per entry and exit, commission 0, zero execution fee under its hash-bound policy, and the policy's elapsed-time financing rates. Preserve gross move and every cost component separately. The existing cost audit says historical slippage and financing truth are not fully qualified; do not call these labels realized historical broker PnL.
- Produce both continuous net outcomes, `R_long_net` and `R_short_net`, for every valid decision. Do not make future prices or label metadata model inputs.

Before training, report total eligible rows, valid full-path labels, invalid reasons, timestamp coverage, purged chronological blocks, and block-based effective sample size. Report M5 and M15 separately. The existing exact-timestamp counts are not full-path coverage and cannot stand in for this measurement.

## Prebound source identities

These are existing TRAIN-only source bindings; they are not permission to launch:

- Entry M5 TRAIN manifest: `/home/andre2/GX1_DATA/data/data/prebuilt/V46_20260825T170935Z_CHAIN/artifacts/V9_FIVE_YEAR_ENTRY_NOTIONAL_20260905T132535Z/dataset/entry_dataset__ENTRY_FITTED_Q_train.manifest.json`, SHA-256 `9311e6d8617c92b7b3dec1c6fab8042002b9610431828266c39976460689f7e6`.
- Existing exact M1 TRAIN child view manifest: `/home/andre2/GX1_DATA/data/data/prebuilt/LIFECYCLE_V2_FULL_TRAIN_20260912/M1_CHILD_VIEWS_V1/train.manifest.json`, SHA-256 `e5c99322bbe878a7a148dd840f9ae5547fb2a1fe8f90c0138058b20fd795c843`; parquet SHA-256 `ee0bdee2bf76846ef0a715a537457cb0653ecac1585742603b84b51de87874f8`. Manifest reports 1,764,512 fit rows, from 2021-06-01 through 2026-05-31 UTC; use only the M1 TRAIN child, not VAL.
- Existing full-train input binding: `/home/andre2/GX1_DATA/data/data/prebuilt/LIFECYCLE_V2_FULL_TRAIN_20260912/FULL_TRAIN_DATA_INPUTS.json`, SHA-256 `56167496540bc045975343b1f80587d1ce52da564c2eeaab6f566889b8b2bee1`; it binds the M5 TRAIN parquet and explicitly has `test_data_used=false` and `training_launch_authority=false`.
- Prospective cost policy: `.../PROSPECTIVE_COST_POLICY_V1/policy.json`, SHA-256 `a48f8e56da21cfa670a80c3b4bfdf735d8ce0b29a25e269184bd5f44fb240a69`; parameter authority SHA-256 `b5cfc8ebaf5b5116747266c667edbbf73f4b862f6c4259c4c0fa202fd1095e37`; existing cost audit SHA-256 `d05a0f2a5f929f95bd0bfc16e813d37b90d93770d037c6cf784e7c835b127f13`. Revalidate the existing policy before any run; never tune it on VAL.
- Reusable implementation is `gx1/contracts/entry_causal_m1_outcomes_v1.py` at the current canonical source commit. Its current fill clock is decision-time M1 open, and its 19-bar outcome path checks exact start/end times and the full required path. No edit to this implementation is part of the coverage-only scope.


## Three distinct model/evaluation quantities

1. **Market feedback:** direct M1 net outcomes tell whether a side offered a cost-positive move over the research horizon. Entry must not learn these values from Exit-Q.
2. **Wait value:** FLAT is the continuation value of remaining flat and preserving the next eligible opportunity, not a constant zero and not merely a class meaning “both fixed-horizon returns were negative.” The eventual action-value target must account for the value of the next decision and the next decision after an actual Exit. Do not fit this head until its transition/Exit semantics and chronological provenance are bound.
3. **Strategy truth:** score selected Entry decisions by realized net PnL under one identical, causal Exit policy, with one-position overlap handling and open positions included. Exit model Q is never treated as realized PnL. Existing reused-TRAIN Exit results do not prove fit-as-of causality and cannot be the teacher for all historical rows.

If a before-period Exit checkpoint cannot be proven for a chronological block, that block may support market-outcome analysis only; it cannot support a causal strategy-PnL claim or a wait-value teacher.

## Fixed sequence and stop conditions

1. Bind the exact source manifests, cost-policy version, decision/fill clocks, fixed horizon, M5/M15 schedule, label-validity rules, and output hashes before measuring coverage.
2. Run one bounded target-coverage/ESS measurement only. Zero model fits, zero optimizer steps, no full epoch/VAL, and no TEST.
3. If labels are valid and coverage/ESS is sufficient under the predeclared block method, define the separate Entry outputs and continuation target in a reviewed training contract. Preserve the old Exit-facing Q function and measure its baseline/parity.
4. Train only after the direct-target contract, causal wait-value teacher, and canonical run policy are all bound. Compare one chronological training/evaluation design to the same constant baseline; no feature, threshold, horizon, or model sweep.
5. Evaluate market prediction, selective LONG/SHORT/FLAT quality, and full one-position strategy economics separately. Positive TRAIN fit or direct market prediction alone is not PASS. Use only a later permitted chronological control for generalization; TEST stays sealed.

Any missing fill, cost, provenance, continuation, or portfolio semantics stops the affected claim rather than being imputed. No source code, weights, datasets, or model state are changed by this document.
