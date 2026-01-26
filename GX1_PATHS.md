# GX1 Paths Configuration

**Last Updated:** 2025-01-21  
**Purpose:** Canonical environment variables and path configuration for GX1 Engine and GX1_DATA separation.

## Environment Variables

### Core Paths

| Variable | Default | Description |
|----------|---------|-------------|
| `GX1_DATA_ROOT` | `../GX1_DATA` | Root directory for GX1_DATA (data, reports, models, etc.) |
| `GX1_REPORTS_ROOT` | `../GX1_DATA/reports` | Reports directory (replay evaluations, truth decomposition, etc.) |
| `GX1_DATA_DATA` | `../GX1_DATA/data` | Data directory (candles, prebuilt features) |
| `GX1_MODELS_ROOT` | `../GX1_DATA/models` | Models directory (bundles, checkpoints) |
| `GX1_ARCHIVE_ROOT` | `../GX1_DATA/archive` | Archive directory (historical artifacts) |
| `GX1_OPTUNA_ROOT` | `../GX1_DATA/optuna` | Optuna database directory |

## Directory Structure

### GX1_ENGINE (Repository)

```
GX1 XAUUSD/
├── gx1/              # Engine code
├── docs/              # Documentation (SSoT manifests)
├── scripts/           # Shell scripts
├── tests/             # Tests
├── policies/          # Policy files
├── gx1/configs/       # Config files
├── GX1_PATHS.md       # This file
└── reports/repo_audit/ # Audit logs (kept in engine)
```

### GX1_DATA (Sibling Directory)

```
../GX1_DATA/
├── data/              # Data files (candles, prebuilt features)
│   ├── oanda/
│   └── features/
├── reports/           # Generated reports
│   ├── replay_eval/
│   ├── truth_decomp/
│   └── ...
├── archive/           # Archived artifacts
├── models/            # Model bundles and checkpoints
│   ├── entry_v10_ctx/
│   └── ...
├── checkpoints/       # Training checkpoints
└── optuna/            # Optuna databases
```

## Local Setup Example

### Default Setup (Relative Paths)

```bash
# Engine and data are siblings
/Users/username/Desktop/
├── GX1 XAUUSD/        # Engine repository
└── GX1_DATA/          # Data directory
```

**No environment variables needed** - defaults work out of the box.

### Custom Setup (Absolute Paths)

```bash
# Set custom paths
export GX1_DATA_ROOT="/path/to/custom/data"
export GX1_REPORTS_ROOT="/path/to/custom/data/reports"
export GX1_DATA_DATA="/path/to/custom/data/data"
export GX1_MODELS_ROOT="/path/to/custom/data/models"
```

## Command Examples

### Before Split (Old)

```bash
# Data and reports in same repo
python3 gx1/scripts/run_depth_ladder_eval_multiyear.py \
  --data-root data/oanda/years \
  --prebuilt-parquet data/features/xauusd_m5_2025_features_v10_ctx.parquet \
  --out-root reports/replay_eval/DEPTH_LADDER
```

### After Split (New)

```bash
# Data and reports in GX1_DATA
python3 gx1/scripts/run_depth_ladder_eval_multiyear.py \
  --data-root ../GX1_DATA/data/oanda/years \
  --prebuilt-parquet ../GX1_DATA/data/features/xauusd_m5_2025_features_v10_ctx.parquet \
  --out-root ../GX1_DATA/reports/replay_eval/DEPTH_LADDER

# Or use defaults (env vars)
export GX1_DATA_DATA="../GX1_DATA/data"
export GX1_REPORTS_ROOT="../GX1_DATA/reports"
python3 gx1/scripts/run_depth_ladder_eval_multiyear.py \
  --data-root $GX1_DATA_DATA/oanda/years \
  --prebuilt-parquet $GX1_DATA_DATA/features/xauusd_m5_2025_features_v10_ctx.parquet
```

## Script Default Paths

Scripts use environment variables for default paths. You can override with command-line arguments:

- `--data-root` - Override `GX1_DATA_DATA`
- `--out-root` - Override `GX1_REPORTS_ROOT`
- `--bundle-dir` - Override `GX1_MODELS_ROOT`

**Example:**
```bash
# Use defaults (from env vars)
python3 gx1/scripts/run_depth_ladder_eval_multiyear.py --arm baseline

# Override with explicit paths
python3 gx1/scripts/run_depth_ladder_eval_multiyear.py \
  --arm baseline \
  --data-root /custom/path/data \
  --out-root /custom/path/reports
```

## Migration Notes

- **Engine code unchanged:** All runtime logic remains identical
- **Paths are defaults only:** Scripts still accept `--data-root`, `--out-root`, etc.
- **PREBUILT mode:** Works identically, just with different default paths
- **Existing commands:** Continue to work if you provide explicit paths

## References

- **Data Contract:** `docs/DATA_CONTRACT.md`
- **Feature Manifest:** `docs/FEATURE_MANIFEST.md`
- **Prune Plan:** `reports/repo_audit/PRUNE_PLAN.md`
