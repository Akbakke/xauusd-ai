> **2026-09-14 current operator state: TRAINING STOPPED after complete first-epoch June VAL.**
> The Windows campaign task is disabled; no next epoch until Entry/Exit and MAE/MFE findings are resolved.
> Use CURRENT_HANDOVER.md, GX1_ARBEIDSMAAL.md and docs/ENTRY_EXIT_REVIEW_20260914.md.
> V40 remains frozen; V41 performance work is prepared only. Model monitoring is hourly when a long job is active.
> Older operational observations below are historical.

<!-- GX1_DOCUMENT_CLASS: CANONICAL | stable lifecycle-v2 system map -->
# GX1 system map

Clock deployment correction, September 13: applying locks before GPU work
triggered the unchanged high-idle keeper and blocked telemetry on boot 409.
The launcher now applies clocks only after GPU memory exceeds the keeper's
384 MiB idle boundary and utilization is positive. It resets clocks when
memory returns to idle and on controller exit. No guard limits were changed.
Canonical keeper recovery and resumed startup must be verified from receipts.

## Current pause-envelope continuation

Current source: 03592fe6f1113736d0499c35ef98a3d9267e558c at /home/andre2/src/GX1_VAL_PAUSE_ENVELOPE_V40.
Branch fix/native-val-pause-envelope-20260913 is pushed; exact remote HEAD verified.
Use CURRENT_NATIVE_RUN.json for current bindings. Older source references are historical.

V39 durably saved 18,353,548 VAL views / 145,254 forwards, cursor 3860/1444,
then failed CANDIDATE_EXECUTION_PAUSE_VAL_RECEIPT_INVALID. This was a bug in
the new receipt validator: the producer wraps its pause result. V40 validates
that actual envelope and permits exact-contract continuation of saved v2 VAL.
Three actual evaluator pauses now pass through the production envelope helper
and receipt writer in one regression test. Five targeted cases and mandatory
commit hooks passed. No numerical/performance model code changes in V40.

All saved VAL progress, TRAIN checkpoint 309 / 19,588 steps and the same
immutable EMA snapshot are explicitly bound and preserved. The failed V39
window has no outer PASS receipt; never claim otherwise. No TRAIN/VAL prefix
is repeated. Source is frozen; documentation checkout is not executable source.

Measured V39 host GPU clock correction improved steady throughput from
1,077 to 1,528 views/s (+41.9%), with identical Q/actions in runtime parity.
Windows launcher reapplies 1395–1695 MHz graphics / requested 9751 MHz memory
before the unchanged bound controller. Effective CUDA P2 memory was 9501 MHz.
300 W, 85 C core, 80 C memory and all existing guards remain. Automatic
application after physical reboot still needs the APPLIED.jsonl observation.

Current deployment is PREPARED, not yet verified running. One heavy job,
four CPU workers, CPU 0–18, sparse checks around 15 minutes. Full five-year
TRAIN, June VAL each epoch, max 30 epochs/patience 5. TEST remains sealed.
Full-policy positive net Bps and live readiness remain unproven; no reliable ETA.

Windows task entry point: `scripts/windows/GX1-NativeClockProfileLauncher.ps1` applies supported GPU clocks, then invokes the original bound controller unchanged. Resource/numerical guards remain unchanged. This driver-setting layer is external to the frozen TRAIN/VAL source. Its binding/measurement is in `handover_snapshot/GPU_CLOCK_PROFILE_20260913.json`.

Current source: `1548dd7c47d7f5a83ede4ccca1ef635b443d51f0` at `/home/andre2/src/GX1_VAL_HOTPATH_V39`. Exact bindings: `CURRENT_NATIVE_RUN.json`. Frozen VAL shares identical path encoding and immutable metadata; durable VAL progress also identifies pause receipts. Resume origin is the saved 13,176,595 views. Older runtime references below are historical.

Current runtime (2026-09-13): `39bdb3ce327b2ba408e43573b3e535fdc30bea06` at `/home/andre2/src/GX1_VAL_CPU_PIPELINE_V38`. Exact live bindings are in `CURRENT_NATIVE_RUN.json`. This successor adds compact market-input reuse, batched economics and four CPU preparation workers; trainer affinity is 0–18 with normal priority, inside the existing 20G/512M/128-task and GPU guards. It resumes the prior EMA snapshot and 1,344,280 verified VAL views. Older source/runtime references below are historical.

Current operating state: `CURRENT_HANDOVER.md`. Immutable active source,
plan, recipe and index: `CURRENT_NATIVE_RUN.json`. Source owners define shapes
and semantics; observations in the dated audit are not replacement constants.
The documentation branch does not change the running source.

| Component | Actual role | Source owner / evidence |
|---|---|---|
| Prices | Native M1/M5 bid/ask OHLC; executable side and costs | bound source/index; `unified_exit_random_access_val_factory_v1.py` |
| Features | Same eight owners on separate native clocks; TRAIN-fit normalization | `entry_model_native_signal_v1.py`, `htf_features.py`, bound manifests |
| Entry | Native M5 sequence with closed M15/H1/H4/D1 context; LONG/SHORT/FLAT by unique Q argmax | `entry_fitted_q_v1.py` |
| Exit | Closed M1 history plus M5/M15/H1/H4/D1 context, Entry-decision token, trade path and lifetime summary | lifecycle-v2 model and state factory |
| Lifecycle | 480 local M1 history bars, at most 512 detailed post-entry tail bars; no 512-bar lifetime cap | `unified_exit_lifecycle_v2.py` |
| MFE/MAE | Causal bid/ask extrema since fill and current executable PnL in lifetime summary | state factory `_summary`; not zero merely because trade is open |
| Fill clock | Observe closed M5, fill at following M1 open; first Exit decision one M1 bar later | first-state bridge/index |
| TRAIN | Full epoch Entry population; bounded outcome-blind sampled Exit transitions for both sides | full-population session, TRAIN factory, epoch sampler |
| Optimization | Shared model, online/target, optimizer, learned task weights, EMA, scheduler, exact order and RNG | resumable candidate coordinator |
| VAL | Entire June Entry cohort, both potential sides until learned EXIT or natural censoring | resumable random-access evaluator |
| Selection | Couple actual Entry choice with learned Exit net Bps; full-cohort and quality gates | entry policy evaluator, checkpoint policy |
| Early stop | June VAL after each epoch; maximum 30, patience 5 | exact recipe; terminal computation is not admission |
| Campaign | Native windows, atomic pointer/receipts, fresh physical boot per invocation | Windows CampaignV2 controller and native candidate runner |
| Safety | Local signed resource/GPU guard and power keeper | plan policy; controller, not Codex polling, owns safety |
| Handover | Observe source, immutable small bindings, PID, checkpoint and VAL progress | `scripts/gx1_handover.sh` → existing collector native mode |

The 2026-09-12 audit observed 238 local signals, 71 continuous context fields,
one categorical context field, and 176 MTF fields partitioned across eight
families. Entry has 32 family/timeframe tokens; Exit has 40. Full wiring is
verified; useful contribution from every route is not yet proven.

TRAIN fitting spans 2021-06-01–2026-05-31; June 2026 is VAL. The smoke covered
all eligible entries of 2025-06-01–2026-05-31. TEST is sealed. Existing base
normalization v7 is deliberately preserved; full-TRAIN lifetime-summary fitting
is separate. Never substitute a newer-looking normalization file.

A model forward is a batch calculation, not a trade. The 5,508 June entry
opportunities produce 11,016 alternative long/short paths, not 11,016 account
orders. Cohort net Bps is not a capital/concurrency-constrained portfolio
backtest. Q agreement is not win rate or calibrated confidence. No manual
confidence threshold, stop loss or maximum trade age was added.

Source audit findings and limits are in `docs/audit_20260912/`. Old forced-512
training, failed zero-step launches and disabled-bootstrap instructions are
historical. `PROJECT_STATE_xau_direction_launch.json` belongs to the earlier
launch owner; native takeover explicitly uses `CURRENT_NATIVE_RUN.json`.

The September 13 successor keeps TRAIN and Entry-VAL batch 16 and increases
Exit-VAL batch to 128. The first production batch compares Q/actions and
inference time with 16-row calls. Completed TRAIN state is preserved in a
fresh session; June accumulators restart. See CURRENT_HANDOVER.md for the
predecessor measurements, exact binding and scope of the runtime check.

VAL market reuse: the evaluator owns one temporary cache per frozen-model invocation, keyed by absolute M1 row. The model reuses only local/MTF states and routing outputs; Entry token, path, summary and Q remain position-specific. No cache crosses weights/epochs or enters TRAIN. The actual cache comparison and timing are logged as VAL_MARKET_CACHE_VERIFIED.
