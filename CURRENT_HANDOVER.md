<!-- GX1_DOCUMENT_CLASS: CANONICAL | current operational takeover -->
# GX1 lifecycle-v2 takeover — 2026-09-14

Full June VAL after first five-year TRAIN epoch completed on 2026-09-14.
USER-REQUESTED STOP: GX1RandomAccessCampaignV2 is disabled/stopped and native
PID 723 was terminated. No next epoch before the Entry/Exit, MAE/MFE and
month-end HOLD findings are resolved. Epoch 2 had already started before the
request; final durable pointer is checkpoint 315, epoch_index 1, 19,908 total
optimizer steps, batch offset 320. Do not call invocation 7 an outer PASS.

The first-epoch immutable EMA (19,588 steps) and full VAL are preserved.
VAL completed 57,845,748 state views / 467,371 forwards, with 7,472 model exits
and 3,544 month-end censored sides out of 11,016 hypothetical side paths.
Entry chose 4,180 LONG / 1,328 SHORT / 0 FLAT. Of those 5,508 selected trades,
only 2,227 exited; 3,281 remained HOLD. Official full-policy net Bps is unavailable.
See docs/ENTRY_EXIT_REVIEW_20260914.md and handover_snapshot/ENTRY_EXIT_SUMMARY_20260914.json.
Do not present closed-winner statistics as complete-policy profitability.

Frozen executed source remains 03592fe6f1113736d0499c35ef98a3d9267e558c in
/home/andre2/src/GX1_VAL_PAUSE_ENVELOPE_V40. CURRENT_NATIVE_RUN.json retains
the exact source, data, recipe and session bindings. No executable source was
changed in place. V41 capacity preparation is separate, not active.

One agent, one heavy job; hourly model observations only when a long run is
active. Existing signed local guards retain their frequent safety measurements.
300 W cap, 85 C core, 80 C memory, 12 GiB VRAM, 20 GiB RAM and 512 MiB swap.
TEST remains sealed; no live/paper trading, promotion or external spend.

Capacity preparation: `2ab85a7548aa0e2132779de26066ecb4bc10c6f6` in `/home/andre2/src/GX1_VAL_CAPACITY_V41`. CPU-stage comparison measured +17.36% with eight workers/batch 128 and +24.65% with eight workers/batch 256, identical state bytes. GPU batch 256 and end-to-end speed remain unmeasured. Long windows are prepared (10,800-second VAL, 12,000-second native budget, 13,800-second independent guard); no new campaign or resume migration exists. Preserve the stop while Entry/Exit fixes are decided.
