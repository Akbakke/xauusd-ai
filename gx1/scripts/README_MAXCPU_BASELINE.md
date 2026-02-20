# High-Throughput Baseline Evaluation Runner

## Overview

`run_baseline_eval_maxcpu.sh` is a wrapper script that maximizes CPU utilization on multi-core systems (e.g., 3090-box) by running baseline evaluation with multiple parallel workers while maintaining deterministic invariants.

## Key Features

- **Auto-detects optimal worker count**: `min(nproc-2, 16)` (minimum 2)
- **Prevents CPU oversubscription**: Sets OMP/MKL/OpenBLAS threads to 1 per worker
- **Optimizes I/O**: PyArrow uses all cores for I/O operations
- **Global lock**: Ensures only one baseline eval runs at a time
- **Automatic harvesting**: Runs Phase 1B analysis after replay completes
- **Hard invariant verification**: Checks PREBUILT and US XGB disable policy

## CPU Thread Configuration

**CRITICAL:** The script sets CPU thread environment variables **BEFORE** Python imports to avoid oversubscription:

```bash
OMP_NUM_THREADS=1          # OpenMP threads per worker
MKL_NUM_THREADS=1          # Intel MKL threads per worker
OPENBLAS_NUM_THREADS=1     # OpenBLAS threads per worker
NUMEXPR_NUM_THREADS=1      # NumExpr threads per worker
VECLIB_MAXIMUM_THREADS=1   # Accelerate (macOS) threads per worker
BLIS_NUM_THREADS=1         # BLIS threads per worker
```

**Why?** We want many processes (workers), each single-threaded in BLAS/OMP. This maximizes CPU utilization across all cores.

**I/O threads** (can use multiple threads):
```bash
PYARROW_NUM_THREADS=$(nproc)  # PyArrow I/O threads
ARROW_NUM_THREADS=$(nproc)    # Arrow I/O threads
```

## Usage

### Basic (auto-detect workers)
```bash
./gx1/scripts/run_baseline_eval_maxcpu.sh
```

### Custom workers
```bash
./gx1/scripts/run_baseline_eval_maxcpu.sh --workers 8
```

### Custom days
```bash
./gx1/scripts/run_baseline_eval_maxcpu.sh --days 7
```

### Full options
```bash
./gx1/scripts/run_baseline_eval_maxcpu.sh \
  --days 2 \
  --workers 16 \
  --policy-yaml /path/to/policy.yaml \
  --bundle-dir /path/to/bundle \
  --prebuilt-parquet /path/to/prebuilt.parquet \
  --data-path /path/to/data.parquet
```

## Output

The script creates an output directory:
```
$GX1_DATA/reports/transformer_baseline_eval/BASELINE_MAXCPU_<timestamp>/
```

Contains:
- `replay.log` - Full replay output
- `harvest.log` - Phase 1B harvesting output
- `RUN_IDENTITY.json` - Run metadata
- `TRANSFORMER_BASELINE_REPORT.md` - Comprehensive baseline report
- `TRANSFORMER_BASELINE_METRICS.json` - Machine-readable metrics
- `TRANSFORMER_PATHOLOGIES.md` - Detailed pathology analysis
- Chunk directories with telemetry

## Hard Invariants

The script verifies:
- ✅ `feature_build_call_count == 0` (PREBUILT mode)
- ✅ `xgb_predict_count_us == 0` (US XGB disabled)
- ✅ `transformer_forward_calls_by_session` sums > 0

If any invariant fails, the script exits with code 1.

## Performance Expectations

On a 3090-box with many CPU cores:
- **CPU usage**: ~80-100% total during run
- **Workers**: Should not be idle due to I/O
- **Throughput**: Significantly higher than `workers=1`

## Troubleshooting

### Low CPU usage
- Check if workers are waiting on I/O (disk I/O bound)
- Verify OMP_NUM_THREADS=1 is set (check `replay.log`)
- Check if prebuilt parquet is on slow storage

### Workers idle
- May indicate I/O bottleneck (prebuilt parquet reading)
- Consider using faster storage or reducing workers

### Lock file issues
```bash
# If lock file is stale:
rm /tmp/gx1_baseline_eval_maxcpu.lock
```

## Notes

- **Determinism**: Multiple workers may process chunks in different order, but each chunk is deterministic
- **Memory**: Each worker loads its own copy of the model bundle (ensure sufficient RAM)
- **I/O**: Prebuilt parquet is read once per worker (not per bar)
