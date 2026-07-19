# GX1 XAUUSD model-native trading engine

GX1 is being rebuilt around one learned XAUUSD Entry decision: calibrated
`LONG`, `SHORT` or `FLAT`. The model must fuse the complete market-evidence
stack and prove its edge through immutable out-of-sample contracts. Missing,
stale or contradictory evidence blocks the system; there is no fallback
direction policy.

Current status: **BLOCKED**. No fresh seq513 bundle is accepted and no Entry
launch is authorized. The old Smart520 evidence is historical and cannot be
used for training, replay, paper trading, live trading or promotion.
The July-19 seq513 rebuild attempts under
`XAU_SEQ513_REBUILD_20260718_V1` were terminated and invalidated after a
feature-ranking TRAIN-window mismatch; no rebuild process or accepted dataset
exists now, no seq513 training process is running, and partial artifacts have
no authority. V1 cannot be reused; any future rebuild requires a new explicit
vedtak after the abstention-baseline decision. A report-only abstention metadata check is
`BLOCK_ABSTENTION_EMPIRICAL_GATE`: its balanced FLAT-label counts and positive
objective weights are not learned evidence. It read zero parquet; immutable
historical selection-benchmark bytes and exact learned-probe OOT evidence are
still absent and are the next empirical gate.

## Active Entry contract

- XAUUSD only; M5 decision cadence with M5/M15/H1/H4/D1 context.
- 513 ordered signals: 34 genuine base price-state fields plus 479 specialist
  fields. Of those 479, all 305 outputs from ten registered causal feature
  layers are code-owned and mandatory; only the remaining 174 positions come
  from deterministic TRAIN-only ranking.
- 142 continuous and 5 categorical context fields.
- Eight learned specialists: structure/swing, SMC/liquidity, trend/EMA,
  volatility/compression, momentum/flow, session/regime, chart geometry and
  price-action/candles.
- Twenty positively supervised evidence heads feeding one exact learned
  23-group/75-value fusion (`75 -> 128 -> 3`).
- One final direction authority: calibrated model logits and exact
  `argmax([LONG, SHORT, FLAT])`.
- Learned path, utility, timing, tail-risk, volatility, trade/side hierarchy,
  trendline-rail, validity and position-size evidence is mandatory.
- Position sizing is learned and must be immutably calibrated, parity-checked
  and journaled. Any label-horizon TEST result is diagnostic only, and no fresh
  accepted sizing result exists for the current contract.
  Paper/live capital remains blocked pending a joint replay with the exact
  adopted active Exit stack and a fresh post-adoption broker runtime-parity
  event. Fixed 1x is a historical comparison only, never a fallback.

Real trend/session/liquidity/volatility/momentum evidence belongs inside the
model. Retired filters are only the disconnected rules that could veto, flip,
threshold or silently pass through a model decision after inference.
Full-stack coverage is proven by causal timing, field liveness and learned
connectivity, but coverage alone does not prove useful influence or trading
edge. Each family's influence and the fused decision must be proved
empirically on immutable OOS/live-like evidence; duplicate indicator aliases
and future-leaking variants are not additional robustness.

## Start here

```bash
bash scripts/gx1_handover.sh
bash scripts/gx1_handover.sh --check  # continuations with unchanged authority
.venv/bin/python -m json.tool PROJECT_STATE_xau_direction_launch.json
scripts/entry_next_edge_control.sh --help
```

Read `AGENTS.md`, `SYSTEM_MAP.md` and
`HANDOVER_XAU_DIRECTION_REPAIR_20260714.md` before changing the pipeline.
Training or rebuild commands require their exact immutable prerequisites and
an explicit `--vedtak`; no heavy run is authorized by documentation alone.

Code lives in this repository. Large datasets, bundles and run evidence live
under `/home/andre2/GX1_DATA` and must not be deleted without an explicit,
verified cleanup decision. Repository cleanup should continuously remove
disconnected scripts, tests and stale documentation once their callers and
evidence value have been disproved.
