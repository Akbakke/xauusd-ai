# Counter semantics (observability)

## n_model_calls / bars_evaluated

- **Definition:** Number of **actual transformer forward calls** (one per bar where the entry model was evaluated / invoked).
- **Source:** `entry_manager.entry_feature_telemetry.transformer_forward_calls`, set after each successful forward in the runner; at end of chunk the runner sets `perf_n_model_calls` from this so the chunk footer and perf export get correct counts.
- **Footer:** `chunk_footer.json` exposes `n_model_calls` and `bars_evaluated` (same value). If `t_transformer_forward_sec > 0` but `n_model_calls == 0`, the footer sets `counter_invariant_violation: true` (observability bug, no PnL impact).
- **Perf:** `perf_<run_id>.json` aggregates `total_model_calls` from chunk footers; when footer has `status=ok`, it requires `total_bars` and either `n_model_calls` or `bars_evaluated` (with fallback path for footer location and `t_feature_build_total_sec` / `feature_time_total_sec`).

**When is n_model_calls == 0 valid?**

- Policy or session filters removed all bars from entry evaluation, and **n_trades == 0** (no trades were opened, so no inconsistency).

**When is n_model_calls == 0 invalid?**

- When **n_trades > 0**: at least one bar led to a trade, so the entry model must have been called. n_model_calls must be > 0 or the run has an observability bug (counters not wired).

## Why this matters for E2E gates

The E2E **forward_calls** gate uses these semantics to avoid false GO:

- **n_trades > 0** → require **forward_calls > 0** (from merged metrics or chunk_footer.n_model_calls / bars_evaluated). If all are 0 or missing → NO-GO.
- **n_trades == 0** → allow **forward_calls == 0** (no-forward-window; log as policy/session-filtered). No use of `t_transformer_forward_sec` as proof of forward (timing only, not semantics).

This keeps canonical results and PnL unchanged while making observability and gates logically consistent.

## Full-year run re-classification

When the E2E **forward_calls** gate passes (from merged metrics or from `chunk_footer.n_model_calls` / `bars_evaluated`), that run is **de facto GO**. There is no need to re-run the full year for that run: the gate proves the entry model ran and counters are consistent. Re-running is only needed if you change code/data or need a fresh run for another reason.

### Full-year FY2025_BASE28_CTX2PLUS_20260218_122441 (SSoT)

- **Run ID:** FY2025_BASE28_CTX2PLUS_20260218_122441  
- **Status:** GO (economically valid; outcomes canonical)

**Economic / strategic validity:** Trade outcomes, PnL, and policy execution for this run are canonical. No re-run is required for economic or strategy purposes.

**Observability note:** This run was produced before the counter-wiring fix. Artifacts such as `n_model_calls`, `perf_*.json`, and forward-counters are known to be incomplete and must not be used for counter/perf assertions or gates.

**Revalidation:** Forward-calls semantics were revalidated post-fix via a 7-day E2E run (GO) with the same setup: canonical truth, BASE28 XGB, CTX2PLUS transformer (ctx_cont_dim=4), prebuilt parquet, and policy.

**Conclusion:** The full-year run is strategically and economically valid. A re-run is not necessary unless full observability artifacts (perf, counters) from a post-fix pipeline are required.
