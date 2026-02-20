# TRUTH pipeline ritual and canary commands

## Artifact layout (single source of truth)

- **Root artifacts (run_root):**  
  `RUN_IDENTITY.json`, `E2E_SANITY_SUMMARY.md`, `metrics_*_MERGED.json`, `trade_outcomes_*_MERGED.parquet`, `MERGE_PROOF_*.json`, `RUN_COMPLETED.json`, `PREFLIGHT_E2E.json`, `POSTRUN_E2E.json`, etc.
- **Replay internals (run_root/replay/):**  
  `chunk_0/` (chunk_footer.json, trade_outcomes_<run_id>.parquet, attribution_, ZERO_TRADES_DIAG.json, etc.).
- **ML exit audit (when exit_ml_enabled):** `run_root/replay/chunk_0/logs/exits/exits_<run_id>.jsonl` — one jsonl line per bar per trade (state, signals, entry_snapshot, deltas, computed exit_score/threshold/decision/reason).

So `run_root/replay/` is internal replay output; all “root” artifacts live in `run_root`. ML exit: `GX1_EXIT_ML_DECIDER=1` enables score_v1; footer has `exit_ml_enabled`, `exit_ml_decision_mode`, `exit_ml_config_hash`; when enabled, postrun requires `replay/chunk_0/logs/exits/exits_*.jsonl`.

---

## Normal TRUTH E2E sanity (expects trades on default window)

Preflight + replay + postrun. Use canonical 1-week window or full-year runner.

```bash
export GX1_DATA=/path/to/GX1_DATA
export GX1_CANONICAL_TRUTH_FILE=/path/to/gx1/configs/canonical_truth_signal_only.json
python -m gx1.scripts.run_truth_e2e_sanity --start-ts 2025-06-03 --end-ts 2025-06-10
```

---

## Zero-trades canary (0-trades contract test)

Dedicated mode to verify the pipeline is deterministic and robust when replay yields **0 trades**. Not the default. No change to canonical truth or policy.

- **Flag:** `--force-zero-trades`
- **Mechanism:** Sets `GX1_ANALYSIS_MODE=1` and `GX1_ENTRY_THRESHOLD_OVERRIDE=1.1`. Entry logic uses `p_long >= threshold`; with threshold 1.1 no probability can trigger entry → 0 trades guaranteed.
- **Contract:** All TRUTH artifacts must still be produced (root in run_root, chunk in run_root/replay/chunk_0).
- **Proof in artifacts (SSoT):** `RUN_IDENTITY.json` and `E2E_SANITY_SUMMARY.md` (section “Zero-trades canary”) in run_root.
- **Gate:** If `n_trades > 0` when canary is enabled → **hard fail** (exit non-zero, E2E_FATAL_CAPSULE).
- **Log:** Script logs `MODE=ZERO_TRADES_CANARY` at start and `ZERO_TRADES_CANARY: n_trades=0, contract OK` on success.

### Canary command (short window, e.g. 1 day)

```bash
python -m gx1.scripts.run_truth_e2e_sanity --force-zero-trades --start-ts 2025-06-03 --end-ts 2025-06-04
```

Expected: `n_trades == 0`, all artifacts present, post-run gate green, exit 0.

---

## A/B Trade Outcomes Delta (Exit ML)

After running A/B full-year (baseline vs ML-frozen), compare trade-outcomes parquets and get per-trade deltas + summary. See **gx1/docs/AB_TRADE_DELTA.md** for full options and output paths.

```bash
/home/andre2/venvs/gx1/bin/python -m gx1.scripts.ab_trade_outcomes_delta \
  --run-a /home/andre2/GX1_DATA/reports/truth_e2e_sanity/A_BASELINE_AB_EXIT_ML_20260219_152051 \
  --run-b /home/andre2/GX1_DATA/reports/truth_e2e_sanity/B_ML_FROZEN_AB_EXIT_ML_20260219_152051 \
  --ab-run-id AB_EXIT_ML_20260219_152051
```

Outputs: `<out-dir>/<ab-run-id>/AB_TRADE_DELTA.md`, `AB_TRADE_DELTA.csv`, `AB_TRADE_DELTA_META.json` (default out-dir: `/home/andre2/GX1_DATA/reports/ab_fullyear_2025_exit_ml`).

---

## Verification commands

### 1) Layout-sjekk (root vs replay)

Kjør canary med fast `--run-id`, deretter sjekk at MERGED er i ROOT og ZERO_TRADES_DIAG i replay/chunk_0.

```bash
RUN_ID="ZERO_TRADES_CANARY_$(date -u +%Y%m%d_%H%M%S)"
RUN_DIR="/home/andre2/GX1_DATA/reports/truth_e2e_sanity/$RUN_ID"

# (Kjør canary)
cd /home/andre2/src/GX1_ENGINE && GX1_DATA=/home/andre2/GX1_DATA \
/home/andre2/venvs/gx1/bin/python -m gx1.scripts.run_truth_e2e_sanity \
  --truth-file gx1/configs/canonical_truth_signal_only.json \
  --force-zero-trades \
  --start-ts 2025-06-03 --end-ts 2025-06-04 \
  --run-id "$RUN_ID"

echo "=== ROOT ==="
ls -la "$RUN_DIR" | sed -n '1,160p'

echo "=== REPLAY ==="
ls -la "$RUN_DIR/replay" | sed -n '1,160p'

echo "=== MERGED (must be in ROOT) ==="
find "$RUN_DIR" -maxdepth 1 -type f \
  \( -name "metrics_*_MERGED.json" -o -name "trade_outcomes_*_MERGED.parquet" -o -name "MERGE_PROOF_*.json" -o -name "RUN_COMPLETED.json" -o -name "RUN_IDENTITY.json" -o -name "E2E_SANITY_SUMMARY.md" \) \
  -print | sort

echo "=== ZERO TRADES DIAG (must be in replay/chunk_0) ==="
find "$RUN_DIR/replay/chunk_0" -maxdepth 1 -type f -name "ZERO_TRADES_DIAG.json" -print
```

### 2) Canary "contract OK" sjekk (hard)

Etter en canary-run: RUN_IDENTITY, n_trades=0, og at summary inneholder canary-seksjon.  
**RUN_DIR er alltid derivert fra RUN_ID** — hvis du glemmer RUN_ID, får du tydelig feilmelding i stedet for «jq finner ikke fil».

```bash
RUN_ID="${RUN_ID:?set RUN_ID first}"
RUN_DIR="/home/andre2/GX1_DATA/reports/truth_e2e_sanity/$RUN_ID"
ROOT="$RUN_DIR"

echo "RUN_IDENTITY:"
jq -r '"mode=\(.mode) threshold=\(.entry_threshold_override) run_id=\(.run_id)"' "$ROOT/RUN_IDENTITY.json"

echo "n_trades (must be 0):"
jq -r '.n_trades' "$ROOT"/metrics_*_MERGED.json

echo "summary contains canary section:"
rg -n "Zero-trades canary|entry_threshold_override|MODE=ZERO_TRADES_CANARY" "$ROOT/E2E_SANITY_SUMMARY.md" || true
```

### 3) One-shot streng verifisering (CI-ish)

Når ritualet skal være maskinvennlig: feil hardt (exit) hvis noe mangler. Sett `RUN_ID` før du kjører.

```bash
set -euo pipefail

RUN_ID="${RUN_ID:?set RUN_ID}"
RUN_DIR="/home/andre2/GX1_DATA/reports/truth_e2e_sanity/$RUN_ID"
ROOT="$RUN_DIR"

test -f "$ROOT/RUN_IDENTITY.json"
test -f "$ROOT/RUN_COMPLETED.json"
ls "$ROOT"/metrics_*_MERGED.json >/dev/null
ls "$ROOT"/trade_outcomes_*_MERGED.parquet >/dev/null
ls "$ROOT"/MERGE_PROOF_*.json >/dev/null
test -f "$ROOT/E2E_SANITY_SUMMARY.md"
test -f "$ROOT/replay/chunk_0/ZERO_TRADES_DIAG.json"

jq -e '.n_trades == 0' "$ROOT"/metrics_*_MERGED.json >/dev/null
```

### 4) Nerde-helse-sjekk: hvor havnet EXIT_COVERAGE_SUMMARY?

Postrun sjekker root først, deretter replay. Denne kommandoen viser begge steder (hvis de finnes).

```bash
ls -la "$RUN_DIR/EXIT_COVERAGE_SUMMARY.json" "$RUN_DIR/replay/EXIT_COVERAGE_SUMMARY.json" 2>/dev/null || true
```

### 5) Hard-fail test (valgfritt)

Kjør en normal TRUTH-run (samme korte vindu, **uten** `--force-zero-trades`). Hvis den gir 1+ trades, kjør samme vindu **med** `--force-zero-trades` og bekreft exit 0 og n_trades=0. Da er canary den eneste modus som hard-failer når n_trades≠0.
