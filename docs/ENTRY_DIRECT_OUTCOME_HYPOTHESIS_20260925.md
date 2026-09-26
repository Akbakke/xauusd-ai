# Direct Entry outcome hypothesis — 2026-09-25

> **Review 2026-09-26 (operator-ordered consolidation) — do not execute this plan.**
> Measured in [DIRECTION_TIMESCALE_20260926.md](DIRECTION_TIMESCALE_20260926.md):
> (1) the 95-minute side targets already exist bit-identically in the bound V9 dataset and in
> V12 (`y_{long,short}_final_pnl_at_direction_horizon_bps`, 313,399/313,399 rows) and were
> tested teacher-free by the walk-forward instrument without a robust result (direct HGB sign
> AUC 0.497–0.500); (2) `Y_wait = max(0, R_long_net(t+cadence), R_short_net(t+cadence))`
> averages +13.7 bps against −5.3/−6.0 for the side targets — a hindsight side choice that
> forces FLAT on 57.9 % of rows even with perfect foresight and on every row without
> information. The operator moved the direction question to multi-day/week horizons
> (see VEIEN_VIDERE.md). Kept as history.

Status: predeclared research hypothesis; no target extraction, fit, model change, or optimizer update has been run. TEST remains sealed.

## Objective

Replace the Entry training signal that copies a frozen Exit-Q at the first post-fill state with direct, side-specific future market outcomes from the existing M1 BID/ASK history. Preserve all current causal Entry inputs, features, feature families, and timeframes. Keep the existing Exit-facing `entry_action_q_bps` semantics unchanged; any direct-outcome prediction must have separate output semantics so it cannot silently alter Exit inputs.

The hypothesis is that direct, executable LONG/SHORT outcome targets provide more useful entry feedback than the current Exit-Q proxy, and that a separately learned wait value can make Entry selective. This is a research hypothesis, not a profitability claim.

## Target definition to bind before any data run

- Decision features are only those available after the completed M5 bar. Use the existing causal M5-to-M1 quote contract and source arrays; do not rebuild or broaden feature inputs.
- Compare two fixed schedules on identical eligible dates and labels: every valid M5 decision, and only M5 bars whose completed decision time is exactly on a UTC 15-minute boundary for M15 cadence. Higher-timeframe features remain unchanged.
- First research label horizon: 19 M5 intervals (95 minutes) from the completed decision timestamp. This is a fixed target-measurement horizon, not a production holding-time rule.
- Reuse the existing exact clock: an M5 bar is timestamped at its UTC start; its completed decision is at start + 300 seconds; entry uses the exact M1 open quote at that decision time. LONG enters ASK and exits/marks BID; SHORT enters BID and exits/marks ASK. The outcome endpoint is the exact M1 open 95 minutes after the decision. If the quote or any required minute in the path is missing, mark the label invalid. Never interpolate or forward-fill.
- Apply the already versioned prospective cost policy as a scenario: observed side-correct BID/ASK spread, adverse slippage 2 bps per entry and exit, commission 0, zero execution fee under its hash-bound policy, and the policy's elapsed-time financing rates. Preserve gross move and every cost component separately. The existing cost audit says historical slippage and financing truth are not fully qualified; do not call these labels realized historical broker PnL.
- Produce both continuous net outcomes, `R_long_net` and `R_short_net`, for every valid decision. Do not make future prices or label metadata model inputs.

Before training, report total eligible rows, valid full-path labels, invalid reasons, timestamp coverage, and block-based effective sample size. Report M5 and M15 separately. For block ESS, use fixed UTC calendar weeks; purge labels whose 95-minute outcome window intersects the first or last 95 minutes of a week, then report valid rows per remaining week and ESS = (sum weekly counts)^2 / sum(weekly counts^2). This is a block-count estimate, not an iid-row sample count. The existing exact-timestamp counts are not full-path coverage and cannot stand in for this measurement.

## Separate Entry targets: side outcomes and the value of waiting

For each eligible decision at time t, define two supervised market-outcome targets from the exact executable quote path and the bound prospective cost scenario:

- Y_long(t) = R_long_net(t): enter LONG at the decision-time ASK and mark/exit at BID after 95 minutes, less the bound entry/exit costs and elapsed-time financing.
- Y_short(t) = R_short_net(t): enter SHORT at the decision-time BID and mark/exit at ASK after 95 minutes, with the same cost accounting.
- Y_wait(t, cadence) = max(0, R_long_net(t + cadence), R_short_net(t + cadence)): the best cost-adjusted opportunity available at the next eligible decision, measured over the following 95 minutes. Cadence is 5 minutes for the M5 schedule and 15 minutes for the M15 schedule.

The model receives only the existing causal features at t and predicts these three targets. Future quotes, realized side choice, target-validity flags, and cost components are labels/audit metadata only; they must not enter model inputs. Use a separate direct-outcome head and preserve the existing Exit-facing entry_action_q_bps unchanged. Select LONG or SHORT only when its predicted net value is positive and strictly exceeds both the other side and predicted wait value; otherwise select FLAT. Exact ties resolve to FLAT. This makes FLAT a learned, potentially nonzero continuation forecast, not a constant zero or an instruction to trade.

The next-decision opportunity target is a supervised forecast target, not realized strategy action value: it assumes the best future side is known when the future label is constructed. It can teach whether waiting is likely to preserve a cost-positive opportunity, but must not be reported as strategy PnL or recursively treated as a calibrated Bellman value. Measure actual selectivity and economics in a separate chronological replay under one identical causal Exit, one-position overlap accounting, and open positions included. Do not use Exit-Q as the direct market target.

The existing 95-minute coverage is insufficient for these three targets because waiting shifts the start of the future path. Require complete exact-quote paths through 100 minutes for M5 and 110 minutes for M15 (95-minute horizon plus the schedule cadence), including all required minutes and the endpoint. Keep the existing weekly purge/block-ESS method, shifted to each target's full information window. Report gross returns and each cost separately, then net-return distributions and invalid reasons by side, cadence, and year. The versioned cost schedule remains a prospective scenario, not verified historical broker PnL.

Target materialization remains a separate bounded audit requiring canonical policy authority. It must write only aggregate evidence unless a later reviewed contract explicitly authorizes row labels. No model fit or weight change follows automatically from sufficient coverage.

## Candidate fit and comparison, only after a new policy binding

If the cost-adjusted target audit is complete and its receipts support the target, the first fit should be one controlled comparison, not a search:

- Train one separate direct scorer per schedule, using the existing causal feature/timeframe inputs and the same fresh initialization for both arms. Train all scorer layers; do not warm-start them from the old Exit-Q-trained Entry checkpoint. Leave the original Entry and Exit checkpoints frozen. During every replay, feed the fixed Exit the exact original Entry-Q/context it received before this experiment; only the separate scorer may choose whether a new position is opened.
- Train all backbone layers and this new output against the corresponding direct continuous labels. Use mean-squared error on each output after scaling each target by a scale estimated from the chronological fit segment only; convert predictions back to bps for selection and reports. Keep labels uncapped and keep the existing optimizer recipe fixed. No target, threshold, feature, horizon, architecture, or optimizer search.
- Use one fixed chronological split inside TRAIN, with the later segment held out once for a check. Apply the same calendar cutoff to M5 and M15; M15 is the exact 15-minute subset. Exclude fit examples whose full target window crosses the cutoff; purge 110 minutes at the boundary. Fit all feature/target normalization from the earlier segment only. Do not use June 2026 VAL or TEST to tune anything.
- Compare each arm's three predictions to a constant predictor fitted on that same earlier segment. Then apply the frozen rule above without fitting a threshold. Separately replay selected actions with one identical causal Exit policy and one-position accounting; report every eligible opportunity, selected trades, open-position marks, net PnL, and results by month. A market-target metric or positive TRAIN fit is not an economics PASS.
- Proceed to any fit only if a new canonical run policy explicitly binds the data, split, target/head semantics, budgets, and receipts. This document itself grants no fit or launch authority. TEST stays sealed.

## Prebound source identities

These are existing TRAIN-only source bindings; they are not permission to launch:

The coverage-only audit is CPU-only, limited to the bound M5 TRAIN timestamps and M1 TRAIN child view, invoked once through `scripts/gx1_capped_run.sh --class audit --mem 4G --swap 512M`. It writes only aggregated counts and hashes to a private run receipt; it does not persist row-level labels. It does not use VAL, launch a model, or access TEST.

- Entry M5 TRAIN manifest: `/home/andre2/GX1_DATA/data/data/prebuilt/V46_20260825T170935Z_CHAIN/artifacts/V9_FIVE_YEAR_ENTRY_NOTIONAL_20260905T132535Z/dataset/entry_dataset__ENTRY_FITTED_Q_train.manifest.json`, SHA-256 `9311e6d8617c92b7b3dec1c6fab8042002b9610431828266c39976460689f7e6`.
- Existing exact M1 TRAIN child view manifest: `/home/andre2/GX1_DATA/data/data/prebuilt/LIFECYCLE_V2_FULL_TRAIN_20260912/M1_CHILD_VIEWS_V1/train.manifest.json`, SHA-256 `e5c99322bbe878a7a148dd840f9ae5547fb2a1fe8f90c0138058b20fd795c843`; parquet SHA-256 `ee0bdee2bf76846ef0a715a537457cb0653ecac1585742603b84b51de87874f8`. Manifest reports 1,764,512 fit rows, from 2021-06-01 through 2026-05-31 UTC; use only the M1 TRAIN child, not VAL.
- Existing full-train input binding: `/home/andre2/GX1_DATA/data/data/prebuilt/LIFECYCLE_V2_FULL_TRAIN_20260912/FULL_TRAIN_DATA_INPUTS.json`, SHA-256 `56167496540bc045975343b1f80587d1ce52da564c2eeaab6f566889b8b2bee1`; it binds the M5 TRAIN parquet and explicitly has `test_data_used=false` and `training_launch_authority=false`.
- Prospective cost policy: `/home/andre2/GX1_DATA/data/data/prebuilt/LIFECYCLE_V2_FULL_TRAIN_20260912/PROSPECTIVE_COST_POLICY_V1/policy.json`, SHA-256 `a48f8e56da21cfa670a80c3b4bfdf735d8ce0b29a25e269184bd5f44fb240a69`; parameter authority SHA-256 `b5cfc8ebaf5b5116747266c667edbbf73f4b862f6c4259c4c0fa202fd1095e37`; existing cost audit SHA-256 `d05a0f2a5f929f95bd0bfc16e813d37b90d93770d037c6cf784e7c835b127f13`. Revalidate the existing policy before any run; never tune it on VAL.
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
