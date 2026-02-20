# Model Compare 2025

FULLYEAR + Q1–Q4 comparison. **Default runs the canonical model only** (SSoT: `config["canonical_model"]` = BASE28). Single-model only; multi-model and `ALL` are banned.

**BASE28_CTX2PLUS_T1** is research-only: allowed only with explicit `--models BASE28_CTX2PLUS_T1` (script prints a WARNING).

## Prerequisites

- **BASE28 (canonical):** XGB bundle, prebuilt parquet, transformer bundle (paths in `model_configs.json`).
- **BASE28_CTX2PLUS_T1:** Optional; same config key when running research.

Check status without starting: `--validate_only` (validates canonical model).

## Usage

**Validation (--models default tom streng `""`, resolves via `config["canonical_model"]`; currently BASE28):**
```bash
export GX1_DATA=/home/andre2/GX1_DATA
python -m gx1.scripts.run_model_compare_2025 --validate_only
```
Expected: exit 0; evidence at `$GX1_DATA/reports/model_compare_2025/model_compare_validation_evidence.json` (includes `canonical_model`).

**Quarter pack (canonical):**
```bash
python -m gx1.scripts.run_model_compare_2025 --mode quarters_2025 --parallel --max_parallel 8
```
Omitting `--models` uses canonical (BASE28).

**Full run (fullyear + quarters):**
```bash
python -m gx1.scripts.run_model_compare_2025 --mode all --parallel --max_parallel 8
```

**Threshold sweep (parallel cases):**
```bash
python -m gx1.scripts.run_model_compare_2025 --thresholds "0.44,0.46,0.48" --parallel --max_parallel 3
```
Each case = 1W1C TRUTH; output under `model_id/cases/thr_0p48/` etc.

## 0-trades behavior

When a run produces **0 trades**:

- Chunk writes **empty** `trade_outcomes_*.parquet` with canonical schema (TRUTH SSoT contract).
- Merge writes **empty** `trade_outcomes_*_MERGED.parquet` + MERGE_PROOF.json + metrics.
- No TRUTH_SSOT_FAIL – run completes successfully.
- `ZERO_TRADES_DIAG.json` in chunk dir: threshold, counts, reject_reason_histogram, etc.

## Sanity check (after run)

```bash
BASE=$GX1_DATA/reports/model_compare_2025
ls -la "$BASE"/RUN_MANIFEST.json
ls -la "$BASE"/fullyear_summary.csv "$BASE"/quarter_summary.csv
find "$BASE" -maxdepth 2 \( -name "RUN_FAILED.json" -o -name "MASTER_FATAL.json" -o -name "RUN_COMPLETED.json" \) | head
```

Modes: `fullyear` | `quarters` | `quarters_2025` | `all` | `extract_only`

## Output

```
GX1_DATA/reports/model_compare_2025/
  fullyear_summary.csv
  quarter_summary.csv
  MODEL_COMPARE_2025_SUMMARY.md
  BASE28/
  (BASE28_CTX2PLUS_T1/ if used)
```
