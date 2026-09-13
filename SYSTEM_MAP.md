<!-- GX1_DOCUMENT_CLASS: CANONICAL | stable lifecycle-v2 system map -->
# GX1 system map

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
