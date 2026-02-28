# GX1 Active FARM Pipeline

## Overview
The active GX1 stack targets the ASIA + LOW/MEDIUM regime with ENTRY_V9_FARM_V2B as the only production entry policy. Runtime is split into managers:

- **EntryManager** – wraps the V9 feature pipeline + policy gating and owns metadata logging for FARM_V2B.
- **ExitManager** – orchestrates FARM exits (V1 baseline + V2 rules + fixed/random sanity exits) and brokers PnL accounting.
- **BrokerClient** – thin wrapper around `OandaClient` so the runner stays broker-agnostic.
- **ReplayEngine (skeleton)** – placeholder for a pure replay harness (entry → exit → pnl) without live plumbing.

Legacy entry models (V4/V6/V7/V8/hybrid blend) and exit stacks (EXIT_V2 drift, EXIT_V3 adaptive, shadow exits) now live in `gx1/legacy/` and are not imported by the active runner.

## Data → Feature → Decision Flow
1. **Data ingestion**
   - `data/features/xauusd_gx1_m5_compatible.parquet` (default) filtered via `scripts/run_replay.sh` or live pricing.
   - Candle alignment + run guards happen in `EntryFeatureBundle`.
2. **Feature building**
   - `gx1/features/runtime_v9.build_v9_runtime_features` builds leakage-safe features at runtime.
   - Guardrails enforce ASIA session + LOW/MED volatility via FARM guard V2.
3. **Entry decision**
   - EntryManager evaluates ENTRY_V9_FARM_V2B (hard p_long ≥ 0.72, p_profitable logged only).
   - `should_enter_trade` gating + BigBrain entry gater remain inside the manager.
   - Meta-model predictions for p_profitable stored for telemetry but never gate trades.
4. **Exit decision**
   - ExitManager routes to FARM exits only:
     - `FARM_EXIT_V2_RULES_A` (rules-based profit capture / trailing).
     - `FARM_EXIT_V1_STABLE` (baseline) when policy requests it.
     - `FARM_EXIT_FIXED_BARS` / `_FIXED` for sanity harnesses.
   - EXIT_V2/EXIT_V3 drift paths and shadow exits are entirely removed.
5. **PnL + logging**
   - PnL computed via `gx1.utils.pnl.compute_pnl_bps` (bid/ask aware) the moment ExitManager emits a closure.
   - `trade_log.csv` contains FARM-specific metadata (`entry_policy_version`, `farm_guard_version`, `entry_p_long`, `entry_p_profitable`).

## Active Policies (gx1/configs/policies/active)
- `ENTRY_V9_FARM_V2B.yaml` – primary runtime entry config.
- `ENTRY_V9_FARM_V2.yaml` – documented fallback while finishing the migration.
- `GX1_V11_OANDA_DEMO_V9_FARM_V2B.yaml` – hygiene/coverage runs with FARM_EXIT_V2_AGGRO.
- `GX1_V11_OANDA_DEMO_V9_FARM_V2B_RULES_A.yaml` – default production bundle (ENTRY_V2B + EXIT_RULES_A).
- `GX1_V11_OANDA_DEMO_V9_FARM_V2B_EXIT_A_FULL.yaml` – long-window audit runner.
- `GX1_V11_OANDA_DEMO_V9_FARM_V2B_RANDOM_EXIT.yaml` / `_FIXED_EXIT.yaml` – sanity exits for entry-only EV probes.

All other policy bundles live under `gx1/archive/configs/policies/` and retain their original YAML names for reference.

## Script Layout
- `scripts/run_replay.sh` – single parameterised entry/exit replay harness (policy, date range, workers, optional output dir).
- `scripts/active/` – Python helpers (`replay_entry_exit*.py`, `run_parallel_replay.py`).
- `scripts/legacy/` – diagnostic/tuning utilities kept for reference.
- `scripts/archive/` – one-off FARM shell wrappers superseded by the new runner.

## Design Principles
- **Single source of truth**: active runtime code resides in `gx1/active`/`gx1/execution`; legacy experiments are isolated under `gx1/legacy` or `gx1/archive`.
- **Composability**: `EntryManager`, `ExitManager`, and `BrokerClient` can be reused by the upcoming pure replay runner without touching live plumbing.
- **Guardrails first**: FARM entry/exit guardrails are always enforced before touching broker state; EXIT_V2/EXIT_V3 paths are removed to avoid accidental regressions.
- **Documentation-driven**: active policies list and script layout are versioned so future audits immediately see what is in play.
