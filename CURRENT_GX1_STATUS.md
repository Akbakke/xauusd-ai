<!-- GX1_DOCUMENT_CLASS: CANONICAL | current status -->
# Current GX1 status

Verified 2026-09-13 21:40 UTC: V40 invocation 1 ended RESUMABLE with guard
PASS and both exit codes 0. Invocation 2 resumed automatically on physical
boot 410 and is running as PID 719; VAL reached 26,061,367 views. Workload
clocks applied automatically at 21:15:53 on the new boot. No idle blocker,
telemetry running, 57 C core / 66 C memory / 125 W. User now requires hourly
model observations; automatic local guards keep their existing cadence.
Earlier pending pause/boot observations below are historical.

Verified running 2026-09-13 19:48 UTC: V40 PID 961 on boot 409 resumed all
18,353,548 saved VAL views and advanced to 18,441,891. TRAIN stays 309/19,588.
Actual GPU Q difference is zero; actions, dynamic state, economics/hashes agree.
The corrected launcher applied clocks automatically at 19:43:38 after workload
memory reached 530 MiB; idle blocker absent and telemetry task running.
GPU 1395/9501 MHz, 55 C core / 64 C memory / 124 W under unchanged 300 W limit.
The V40 first outer pause receipt, idle reset and next boot remain to be observed.
Use handover_snapshot/VAL_PAUSEFIX_LIVE_VERIFICATION.json. Earlier prepared and
failed startup observations below are historical. Continue sparse monitoring.

Clock deployment correction, September 13: applying locks before GPU work
triggered the unchanged high-idle keeper and blocked telemetry on boot 409.
The launcher now applies clocks only after GPU memory exceeds the keeper's
384 MiB idle boundary and utilization is positive. It resets clocks when
memory returns to idle and on controller exit. No guard limits were changed.
Canonical keeper recovery and resumed startup must be verified from receipts.

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
