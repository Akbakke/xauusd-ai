# SNIPER Regime Analysis (Offline)

This package contains **offline analysis tools** for SNIPER 2025. They operate only on
existing `gx1/wf_runs/SNIPER_OBS_...` trade journals and **do not run replay or change
any trading logic**.

## 1. Verify that regime fields are present

Script: `gx1/sniper/analysis/verify_sniper_regime_fields.py`

This script scans one or more SNIPER run directories, loads their
`trade_journal/trade_journal_index.csv`, and checks that the expected FARM-style
regime fields exist with good coverage:

- `trend_regime`
- `vol_regime`
- `atr_bps`
- `spread_bps`
- `session`
- `range_pos`, `distance_to_range`, `range_edge_dist_atr`
- `router_decision` (optional)

It writes a markdown report under `reports/`, e.g.:

```bash
python -m gx1.sniper.analysis.verify_sniper_regime_fields
```

or with explicit runs:

```bash
python gx1/sniper/analysis/verify_sniper_regime_fields.py \
  --run-dirs gx1/wf_runs/SNIPER_OBS_Q1_2025_baseline_... \
             gx1/wf_runs/SNIPER_OBS_Q1_2025_guarded_...
```

## 2. Regime classification helper

Module: `gx1/sniper/analysis/regime_classifier.py`

Provides a simple function:

```python
from gx1.sniper.analysis.regime_classifier import classify_regime

regime_class, reason = classify_regime(row_dict_or_series)
```

The classifier maps each trade into one of three coarse regime classes:

- `A_TRENT` – clearly trending, normal/high energy, acceptable spreads
- `B_MIXED` – not clearly trending or choppy
- `C_CHOP` – choppy / range-bound / ultra-low volatility

It relies only on fields already present in SNIPER trade journals
(`trend_regime`, `vol_regime`, `atr_bps`, `spread_bps`).

## 3. Analyze performance by regime class

Script: `gx1/sniper/analysis/analyze_sniper_by_regime.py`

This script loads existing SNIPER 2025 run_dirs (or uses `--run-dirs`), classifies each
trade into a regime class, and aggregates performance metrics per
`quarter × variant × regime_class`:

- trade count
- EV/trade (mean `pnl_bps`)
- win rate
- avg win / avg loss / payoff
- p90 loss (90th percentile of loss magnitude)
- median duration (if `entry_time`/`exit_time` available)

Usage:

```bash
python -m gx1.sniper.analysis.analyze_sniper_by_regime
```

or with explicit run directories:

```bash
python gx1/sniper/analysis/analyze_sniper_by_regime.py \
  --run-dirs gx1/wf_runs/SNIPER_OBS_Q1_2025_baseline_... \
             gx1/wf_runs/SNIPER_OBS_Q1_2025_guarded_... \
             gx1/wf_runs/SNIPER_OBS_Q2_2025_baseline_... \
             gx1/wf_runs/SNIPER_OBS_Q2_2025_guarded_... \
             ...
```

It writes a markdown report under `reports/`, for example:

```
reports/SNIPER_2025_REGIME_SPLIT_REPORT__YYYYMMDD_HHMMSS.md
```

> Note: These tools are **analysis-only**; they do not trigger new replays, do not
> update trade journals, and do not modify any live or backtest policies.

## 4. Q4 overlay evaluation (A/B after replay)

Script: `gx1/sniper/analysis/eval_q4_overlay.py`

After you have run Q4 replays with and without the SNIPER size overlay variants
(`baseline`, `baseline_overlay`, `guarded`, `guarded_overlay`), you can evaluate
the overlay effect and correctness without rerunning replay:

```bash
python -m gx1.sniper.analysis.eval_q4_overlay
```

The script:
- Autodetects latest Q4 runs for:
  - `SNIPER_OBS_Q4_2025_baseline_*`
  - `SNIPER_OBS_Q4_2025_baseline_overlay_*`
  - `SNIPER_OBS_Q4_2025_guarded_*`
  - `SNIPER_OBS_Q4_2025_guarded_overlay_*`
- Verifies that:
  - `entry_snapshot.sniper_overlay.overlay_applied == True` **only** for `Q4 × B_MIXED`
  - `overlay_name == "Q4_B_MIXED_SIZE"`
  - `multiplier` matches config (default `0.30`)
  - `size_after_units` matches the expected rounded size (preserving sign)
- Produces a markdown report:

```
reports/SNIPER_Q4_OVERLAY_EVAL__YYYYMMDD_HHMMSS.md
```

with tables for:
- Q4 baseline vs baseline_overlay (total + B_MIXED only)
- Q4 guarded vs guarded_overlay (total + B_MIXED only)

## 5. Regime source comparison (index vs JSON)

Script: `gx1/sniper/analysis/compare_regime_sources.py`

After running Q4 replays, you can diagnose differences between the
index/CSV-based regime fields and the JSON-first view used by the overlay eval:

```bash
python -m gx1.sniper.analysis.compare_regime_sources
```

This script:
- Autodetects the latest Q4 2025 baseline and baseline_overlay run_dirs.
- Loads regime inputs from:
  - `trade_journal/trade_journal_index.csv` (INDEX source)
  - `trade_journal/trades/*.json` (JSON source)
- Applies the same `classify_regime(...)` function to both sources.
- Produces a markdown report:

```
reports/SNIPER_Q4_REGIME_SOURCE_DIFF__YYYYMMDD_HHMMSS.md
```

The report includes:
- Field coverage (trend_regime, vol_regime, atr_bps, spread_bps, session) for index vs JSON.
- Regime class distribution (A/B/C) for each source.
- Agreement matrix (index regime vs JSON regime).
- Drilldown for trades where index=B_MIXED but JSON!=B_MIXED.
- An auto "hypothesis" summary explaining likely causes of any mismatch.

## 6. Q4 C_CHOP analysis and size sensitivity

Script: `gx1/sniper/analysis/analyze_q4_c_chop.py`

To analyze Q4 × C_CHOP performance and hypothetical size multipliers using
existing JSON trade journals:

```bash
python -m gx1.sniper.analysis.analyze_q4_c_chop
```

This script:
- Autodetects the latest Q4 2025 `baseline` and `guarded` run_dirs.
- Loads trades from `trade_journal/trades/*.json` (no CSV/index dependency).
- Classifies regimes with `classify_regime(...)` and keeps only `C_CHOP`.
- Computes, for Q4 × C_CHOP:
  - trades, EV/trade (mean `pnl_bps`), winrate, payoff, p90 loss,
    median and p90 duration (if available).
- Runs an offline size-sensitivity sweep over multipliers:
  - `[1.0, 0.75, 0.50, 0.30]`
  - Scales `pnl_bps` and recomputes EV, winrate, payoff, p90 loss.
- Optionally splits by `session` (ASIA/LONDON/NY) where there are ≥200 trades.

It writes a report:

```
reports/SNIPER_Q4_C_CHOP_ANALYSIS__YYYYMMDD_HHMMSS.md
```

with:
- Baseline vs guarded C_CHOP metrics.
- Size-sensitivity tables per variant (and per session where applicable).




