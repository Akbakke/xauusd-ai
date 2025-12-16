# GX1 Config Surface - YAML/CLI Parameters Reference

**Last Updated:** 2025-12-15

---

## Policy YAML Structure

### Top-Level Keys

```yaml
meta:
  router_version: V3_RANGE
  role: PROD_BASELINE
  frozen_date: 2025-12-15
  description: "..."

policy_name: GX1_V11_OANDA_DEMO_V9_FARM_V2B_EXIT_HYBRID_V3_RANGE_PROD
version: "GX1_V11_OANDA_DEMO_V9_FARM_V2B_EXIT_HYBRID_V3_RANGE_PROD"
mode: "REPLAY"  # or "LIVE", "ENTRY_ONLY"
instrument: "XAU_USD"
timeframe: "M5"
warmup_bars: 288

entry_config: gx1/configs/policies/prod_snapshot/2025_FARM_V2B_HYBRID/ENTRY_V9_FARM_V2B_PROD.yaml
exit_config: gx1/configs/exits/FARM_EXIT_V2_RULES_A_mp68_max5_rule5_cd0.yaml

exit_policies:
  rule5: gx1/configs/exits/FARM_EXIT_V2_RULES_A_mp68_max5_rule5_cd0.yaml
  rule6a: gx1/configs/exits/FARM_EXIT_V2_RULES_ADAPTIVE_v1.yaml

exit_hybrid:
  enabled: true
  mode: RULE5_RULE6A_ATR_SPREAD_V1
  atr_low_pct: 0.0
  atr_high_pct: 75.0
  max_spread_pct: 40.0
  allowed_regimes:
    - FARM_ASIA_MEDIUM
  exit_params:
    enable_tp: true
    enable_trailing: true
    enable_be: true

hybrid_exit_router:
  version: HYBRID_ROUTER_V3
  v3_range_edge_cutoff: 1.0  # Guardrail cutoff
  model_path: gx1/configs/policies/prod_snapshot/2025_FARM_V2B_HYBRID_V3_RANGE/exit_router_models_v3/exit_router_v3_tree.pkl

trade_log_csv: gx1/wf_runs/.../trade_log.csv

logging:
  level: INFO  # or DEBUG, WARNING, ERROR
  log_dir: gx1/wf_runs/.../logs

execution:
  max_open_trades: 5

risk:
  min_time_between_trades_sec: 0
```

---

## Key Parameters

### `hybrid_exit_router.v3_range_edge_cutoff`

**Type:** `float`  
**Default:** `1.0`  
**Range:** `[0.0, 10.0]` (typically `[0.5, 2.0]`)

**Effect:**
- Guardrail cutoff for RULE6A allocation
- If `range_edge_dist_atr >= cutoff`, RULE6A is overridden to RULE5
- Lower values = more restrictive (fewer RULE6A trades)
- Higher values = less restrictive (more RULE6A trades)

**Location:** `gx1/policy/exit_hybrid_controller.py:45`  
**Runtime Check:** `gx1/policy/exit_hybrid_controller.py:104`

**Example:**
```yaml
hybrid_exit_router:
  v3_range_edge_cutoff: 1.0  # PROD_BASELINE
```

---

### `exit_hybrid.enabled`

**Type:** `boolean`  
**Default:** `false`

**Effect:**
- Enables/disables hybrid exit routing
- If `false`, uses default exit policy (RULE5)
- If `true`, uses router to select between RULE5/RULE6A

**Location:** `gx1/execution/oanda_demo_runner.py:1336`

---

### `exit_hybrid.mode`

**Type:** `string`  
**Default:** `RULE5_RULE6A_ATR_SPREAD_V1`  
**Values:** `RULE5_RULE6A_ATR_SPREAD_V1` (only supported value)

**Effect:**
- Routing mode (currently only one mode supported)
- Passed to `ExitModeSelector` but not actively used

**Location:** `gx1/execution/oanda_demo_runner.py:1337`

---

### `exit_hybrid.atr_low_pct` / `atr_high_pct`

**Type:** `float`  
**Default:** `0.0` / `75.0`  
**Range:** `[0.0, 100.0]`

**Effect:**
- ATR percentile range for hybrid routing (legacy, not actively used in V3)
- Originally used for rule-based routing (V1)
- V3 router uses ML model instead

**Location:** `gx1/execution/oanda_demo_runner.py:1341-1342`

---

### `exit_hybrid.max_spread_pct`

**Type:** `float`  
**Default:** `40.0`  
**Range:** `[0.0, 100.0]`

**Effect:**
- Maximum spread percentile for hybrid routing (legacy, not actively used in V3)
- Originally used for rule-based routing (V1)
- V3 router uses ML model instead

**Location:** `gx1/execution/oanda_demo_runner.py:1343`

---

### `hybrid_exit_router.version`

**Type:** `string`  
**Default:** `HYBRID_ROUTER_V1`  
**Values:** `HYBRID_ROUTER_V1`, `HYBRID_ROUTER_V2`, `HYBRID_ROUTER_V2B`, `HYBRID_ROUTER_V3`, `HYBRID_ROUTER_ADAPTIVE`

**Effect:**
- Selects router version
- `HYBRID_ROUTER_V3`: Uses ML decision tree with range features
- Other versions: UKJENT – må verifiseres

**Location:** `gx1/execution/oanda_demo_runner.py:1340`  
**Router Selection:** `gx1/policy/exit_hybrid_controller.py:71`

---

### `hybrid_exit_router.model_path`

**Type:** `string` (file path)  
**Default:** `gx1/analysis/exit_router_models_v3/exit_router_v3_tree.pkl` (hardcoded fallback)

**Effect:**
- Path to trained router model (for V3)
- If not set, uses hardcoded path
- Model loaded lazily on first router call

**Location:** `gx1/core/hybrid_exit_router.py:214`  
**Note:** Currently hardcoded, config value may not be used

---

### `logging.level`

**Type:** `string`  
**Default:** `INFO`  
**Values:** `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`

**Effect:**
- Sets Python logging level
- `DEBUG`: Includes range feature values, router decisions, guardrail overrides
- `INFO`: Major events (trade creation, exit execution)
- `WARNING`: Guardrail overrides, fallback logic

**Location:** `gx1/execution/oanda_demo_runner.py:421`

---

### `mode`

**Type:** `string`  
**Default:** `LIVE`  
**Values:** `LIVE`, `REPLAY`, `ENTRY_ONLY`

**Effect:**
- `LIVE`: Live trading mode (OANDA API)
- `REPLAY`: Offline replay mode (historical data)
- `ENTRY_ONLY`: Entry-only mode (no exits, for entry model evaluation)

**Location:** `gx1/execution/oanda_demo_runner.py:1305`

---

### `warmup_bars`

**Type:** `integer`  
**Default:** `288` (24 hours for M5)

**Effect:**
- Number of bars to warm up before trading starts
- Used for feature initialization (ATR history, etc.)

**Location:** `gx1/execution/oanda_demo_runner.py:1305`

---

### `execution.max_open_trades`

**Type:** `integer`  
**Default:** `5`

**Effect:**
- Maximum number of concurrent open trades
- Prevents over-leveraging

**Location:** UKJENT – må verifiseres

---

## CLI Parameters

### `scripts/run_replay.sh`

**Usage:**
```bash
bash scripts/run_replay.sh <policy_yaml> <start_date> <end_date> [n_workers] [output_dir]
```

**Parameters:**
- `policy_yaml`: Path to policy YAML file (required)
- `start_date`: Start date (YYYY-MM-DD) (required)
- `end_date`: End date (YYYY-MM-DD) (required)
- `n_workers`: Number of parallel workers (default: 7)
- `output_dir`: Output directory (default: `gx1/wf_runs/<policy_name>`)

**Environment Variables:**
- `M5_DATA`: Path to M5 price data (parquet/CSV)
- `GX1_RUN_ID`: Custom run ID (optional, for testing)

**Example:**
```bash
bash scripts/run_replay.sh \
  gx1/configs/policies/active/GX1_V11_OANDA_DEMO_V9_FARM_V2B_EXIT_HYBRID_V3_FULLYEAR.yaml \
  2025-01-01 2025-12-31 7 \
  gx1/wf_runs/FARM_V2B_EXIT_HYBRID_ROUTER_V3_RANGE_FULLYEAR
```

---

### `scripts/active/replay_entry_exit_parallel.py`

**Usage:**
```bash
python scripts/active/replay_entry_exit_parallel.py \
  --price-data <path> \
  --base-policy <path> \
  [--exit-config <path>] \
  [--limit-bars <n>] \
  [--output <path>] \
  [--n-workers <n>]
```

**Parameters:**
- `--price-data`: Path to price data (parquet/CSV) (required)
- `--base-policy`: Path to policy YAML (required)
- `--exit-config`: Override exit config (optional)
- `--limit-bars`: Limit to last N bars (optional)
- `--output`: Output path for results JSON (default: `results/entry_exit_v2_drift_only.json`)
- `--n-workers`: Number of parallel workers (default: 7)

**Environment Variables:**
- `GX1_PARALLEL_LOG_LEVEL`: Log level for parallel execution (default: `INFO`)
- `GX1_PARALLEL_BACKEND`: Parallel backend (default: `loky`)

---

### `gx1/analysis/build_exit_policy_training_dataset_v3.py`

**Usage:**
```bash
python gx1/analysis/build_exit_policy_training_dataset_v3.py \
  --run <TAG> <PATH> \
  [--run <TAG> <PATH> ...]
```

**Parameters:**
- `--run`: Add a run to dataset (can be used multiple times)
  - Format: `--run TAG PATH`
  - `TAG`: Run identifier (e.g., `BASELINE_2024`)
  - `PATH`: Path to run directory (e.g., `gx1/wf_runs/FARM_V2B_EXIT_HYBRID_ROUTER_V3_RANGE_2024_BASELINE`)

**Example:**
```bash
python gx1/analysis/build_exit_policy_training_dataset_v3.py \
  --run BASELINE_2024 gx1/wf_runs/FARM_V2B_EXIT_HYBRID_ROUTER_V3_RANGE_2024_BASELINE \
  --run GUARDRAIL_2024 gx1/wf_runs/FARM_V2B_EXIT_HYBRID_ROUTER_V3_RANGE_2024_GUARDRAIL
```

**Output:**
- `gx1/analysis/exit_policy_training_dataset_v3.csv`
- `gx1/analysis/exit_policy_training_dataset_v3.parquet`
- `gx1/analysis/exit_policy_training_dataset_v3_metadata.json`

---

### `gx1/analysis/router_training_v3.py`

**Usage:**
```bash
python gx1/analysis/router_training_v3.py \
  --dataset <path> \
  [--output-dir <path>] \
  [--max-depth <n>] \
  [--min-samples-split <n>]
```

**Parameters:**
- `--dataset`: Path to training dataset (CSV/parquet) (required)
- `--output-dir`: Output directory for model files (default: `gx1/analysis/exit_router_models_v3/`)
- `--max-depth`: Maximum tree depth (default: UKJENT – må verifiseres)
- `--min-samples-split`: Minimum samples for split (default: UKJENT – må verifiseres)

**Output:**
- `exit_router_v3_tree.pkl`
- `exit_router_v3_tree_rules.txt`
- `exit_router_v3_metrics.json`

---

### `gx1/analysis/compare_exit_routers_fullyear.py`

**Usage:**
```bash
python gx1/analysis/compare_exit_routers_fullyear.py \
  --run <TAG> <PATH> \
  [--run <TAG> <PATH> ...] \
  [--regime-analysis] \
  [--intratrade-risk] \
  [--baseline-run <PATH>] \
  [--guardrail-run <PATH>]
```

**Parameters:**
- `--run`: Add a run to compare (can be used multiple times)
- `--regime-analysis`: Run regime analysis (range-edge buckets, time windows)
- `--intratrade-risk`: Run intratrade risk analysis (requires `--baseline-run` and `--guardrail-run`)
- `--baseline-run`: Baseline run path (for intratrade risk)
- `--guardrail-run`: Guardrail run path (for intratrade risk)

**Example:**
```bash
python gx1/analysis/compare_exit_routers_fullyear.py \
  --run BASELINE_2024 gx1/wf_runs/FARM_V2B_EXIT_HYBRID_ROUTER_V3_RANGE_2024_BASELINE \
  --run GUARDRAIL_2024 gx1/wf_runs/FARM_V2B_EXIT_HYBRID_ROUTER_V3_RANGE_2024_GUARDRAIL \
  --regime-analysis
```

---

## Runtime Parameter Access

**Entry Manager:**
- `self.exit_mode_selector`: `ExitModeSelector` instance
- `self.exit_hybrid_enabled`: Boolean flag
- Range features: `trade.extra["range_pos"]`, `trade.extra["distance_to_range"]`, `trade.extra["range_edge_dist_atr"]`

**Exit Manager:**
- `self.exit_farm_v2_rules_factory`: Exit policy factory (if enabled)
- `trade.extra["exit_profile"]`: Selected exit profile (RULE5 or RULE6A)

**Router:**
- `ctx.v3_range_edge_cutoff`: Guardrail cutoff (from config)
- `ctx.range_edge_dist_atr`: Range edge distance (from trade.extra)

---

## Parameter Validation

**Guardrail Cutoff:**
- Must be `>= 0.0` and `<= 10.0` (typically `[0.5, 2.0]`)
- Validated: `gx1/policy/exit_hybrid_controller.py:45` (float conversion)

**Router Version:**
- Must be one of supported values
- Validated: `gx1/policy/exit_hybrid_controller.py:71` (if/elif chain)

**Model Path:**
- Must exist (or fallback used)
- Validated: `gx1/core/hybrid_exit_router.py:215` (Path.exists())

---

## Notes

- **Config Merging:** Entry config and exit config are merged into main policy dict
- **Default Values:** Most parameters have defaults (see code locations above)
- **Environment Overrides:** Some parameters can be overridden via environment variables
- **Runtime Changes:** Config changes require restart (policy lock prevents mid-run changes)

