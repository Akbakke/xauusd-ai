# GX1 roadmap

Updated 2026-08-03.

## Objective

Produce one XAUUSD bundle with one shared encoder that learns tops, bottoms,
abstention and position lifecycle from the complete causal evidence stack.
Entry emits `LONG/SHORT/FLAT` on closed M5. Exit emits `HOLD/EXIT_NOW` on
closed M1 from the same feature ownership plus the additive M1 path. Exact
model argmax owns both decisions.

The active scope is offline train, OOS and replay only. Live, paper, broker,
publisher, promotion and drift work are outside the frozen scope in
`GX1_RULES.md`.

## Completed authority

- One shared 513-signal, 142-continuous and five-categorical feature contract.
- The same eight specialists and formulas for Entry M5 and Exit M1.
- Exact M5/M15/H1/H4/D1 V4 surface: 111 fields per timeframe, 40
  family×timeframe routes and 555 feature×timeframe gates.
- Current MTF cache schema v3 with complete trailing buckets and liveness PASS.
- Chronological V8 dataset: 369,303 TRAIN, 5,904 VAL and 6,071 untouched TEST
  rows.
- V13 unified Exit lifecycle PASS: 96 M5 Entry bars and 480 M1 Exit bars,
  identical 513-field ownership, split boundaries and TRAIN normalization.
- One positively trained unified Exit head in the same model; no separate
  Entry/Exit policy or compatibility lane.
- Current immutable smoke recipe audit PASS: CPU, batch 8, six epochs,
  patience 3, 512 TRAIN rows, zero workers and explicit windows
  `16/64/96/96/252`.
- Heavy-job authority fixed at 10 GiB memory, 512 MiB swap, CPU 0–1 and one
  numerical thread.

These are data, wiring and recipe contracts. They do not prove model edge.

## Current boundary

The first six-epoch V8/V13 smoke was interrupted in epoch 3 by a Windows
`HYPERVISOR_ERROR (0x20001)` bugcheck and produced no completion bundle. The
post-restart WSL VM now matches the configured 32 GB/4 GB envelope. The first
post-restart launch stopped before data on a runner/trainer environment
collision, repaired in `667d704b`. The next exact V16 launch reached the first
training step, where the cgroup safely killed only the job at its 10 GiB RAM
and 512 MiB swap ceilings; no bundle was produced and the PC remained healthy.

Commit `45421e70` replaces retained Transformer activations with exact
per-layer backward recomputation. It removes no feature, head, timeframe or
Exit sample and changes no batch objective, dropout stream, output or
gradient. The 118 focused tests pass under 4 GiB, and a full-size batch-8 plus
32-row Exit forward/backward proof peaked at 3,732,668 KiB RSS with zero swap.
Immutable recipe V17 binds this source and its exact dry-run passes.

Launch and model status remain `BLOCK`: no completed smoke bundle, candidate,
calibration, untouched-TEST edge, train==serve proof or accepted replay exists.

## Next execution sequence

1. Run the V17-bound V8/V13 six-epoch smoke through the single control
   surface under 10 GiB/512 MiB/CPU0–1. Do not rebuild the current dataset.
2. Require a terminal immutable result. A missing bundle, cap kill, class
   collapse or failed component-movement contract remains hard red.
3. If smoke produces a contract-valid trainability bundle, run the existing
   smoke-bundle audit. Smoke still carries zero edge authority.
4. Train one candidate with the same shared encoder and both heads. Preserve
   TEST untouched and never retrain or swap Exit afterward.
5. Fit calibration only on its declared non-TEST split.
6. Evaluate untouched TEST for direction, balanced accuracy, abstention
   coverage, LONG/BOTTOM and SHORT/TOP alignment, costs, path quality,
   utility and supported slices.
7. Prove train==serve and raw/calibrated margin movement for all eight
   specialists, five timeframes, 40 family×timeframe routes, 26 fusion groups,
   1,723 numeric routes and five categorical routes for both decisions.
8. Execute the candidate-bound full-TEST Entry/Exit replay from exact T+5
   Entry fills through actual model `EXIT_NOW`.
9. Admit nothing unless every current byte-bound contract passes. Otherwise
    remain `BLOCK` with terminal evidence.

## Permanent no-go conditions

- hand-written direction, confluence weights, thresholds, vetoes or overlays;
- a separate Exit model or synthetic `FLAT`/`HOLD`;
- a missing family or copied/zero-filled timeframe surface;
- stale, mutable, glob-selected or `latest` artifacts;
- hidden recipe values or TEST-driven selection;
- target clipping, substitution or fabricated feature values;
- gate values presented as predictive edge;
- overlapping or uncapped heavy jobs;
- direct GX1_DATA deletion outside the evidence-retention owner;
- a precision or profitability claim without immutable OOS and replay proof.
