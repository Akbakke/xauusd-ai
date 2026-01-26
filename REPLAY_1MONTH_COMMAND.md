# 1 Måned Replay - Kommando og Instruksjoner

## Kommando

```bash
# Sett opp miljøvariabler
export GX1_REPLAY_NO_CSV=1
export GX1_FEATURE_USE_NP_ROLLING=1

# Kjør med thread limits wrapper
POLICY="gx1/configs/policies/sniper_snapshot/2025_SNIPER_V10_1/GX1_V12_SNIPER_ENTRY_V10_1_FLAT_THRESHOLD_0_18.yaml"
DATA_FILE="data/temp/january_2025_replay.parquet"  # Juster til faktisk fil
OUTPUT_DIR="data/temp/replay_1month_$(date +%Y%m%d_%H%M%S)"

mkdir -p "$OUTPUT_DIR"

./scripts/run_replay_with_thread_limits.sh \
  scripts/run_mini_replay_perf.py \
  "$POLICY" \
  "$DATA_FILE" \
  "$OUTPUT_DIR"
```

## Validering (valgfritt, kun for debug)

For å validere NP kurtosis mot pandas på første 500 bars:

```bash
export GX1_VALIDATE_NP_KURTOSIS=1
# (legg til før replay-kommandoen over)
```

## Etter Replay: Sjekk Resultater

```bash
# Vis perf-summary
cat "$OUTPUT_DIR/REPLAY_PERF_SUMMARY.md"

# Vis prioritert kill list
python3 scripts/list_pandas_rolling_calls.py --perf "$OUTPUT_DIR/REPLAY_PERF_SUMMARY.json"
```

## Suksesskriterier

- `completed=True` i REPLAY_PERF_SUMMARY.md
- Ingen segfault/timeout
- `rolling.pandas.apply.w48` fortsatt 0 eller eliminert
- `feat.basic_v1.misc_roll` betydelig lavere enn før
- `next_pandas_rolling_target` vises i summary

