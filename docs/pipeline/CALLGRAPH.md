# GX1 Call Graph - Module → Functions → Callers

**Last Updated:** 2025-12-15

---

## Entry Point → Runtime Flow

### Level 1: Script Entry

```
scripts/run_replay.sh
  └─> scripts/active/replay_entry_exit_parallel.py::main()
        ├─> load_yaml_config() [gx1/execution/oanda_demo_runner.py:416]
        ├─> split_dataframe() [local]
        ├─> run_replay_chunk() [local]
        │     └─> GX1DemoRunner.__init__() [gx1/execution/oanda_demo_runner.py:1263]
        │     └─> GX1DemoRunner.run_replay() [gx1/execution/oanda_demo_runner.py:7652]
        └─> merge_chunk_results() [local]
```

---

### Level 2: GX1DemoRunner Initialization

```
GX1DemoRunner.__init__() [gx1/execution/oanda_demo_runner.py:1263]
  ├─> load_yaml_config() [gx1/execution/oanda_demo_runner.py:416]
  ├─> EntryManager.__init__() [gx1/execution/entry_manager.py:20]
  │     └─> ExitModeSelector.__init__() [gx1/policy/exit_hybrid_controller.py:24]
  ├─> ExitManager.__init__() [gx1/execution/exit_manager.py:27]
  ├─> load_entry_models() [gx1/execution/oanda_demo_runner.py:444]
  │     └─> Called from line 2184 with:
  │         - metadata_path: gx1/models/GX1_entry_session_metadata.json
  │         - model_paths: {"EU": ..., "US": ..., "OVERLAP": ...}
  │     └─> Returns EntryModelBundle with models dict, feature_names, metadata
  └─> setup_logging() [gx1/execution/oanda_demo_runner.py:421]
```

---

### Level 3: Replay Execution

```
GX1DemoRunner._run_replay_impl() [gx1/execution/oanda_demo_runner.py:~7652]
  └─> (bar loop)
        ├─> GX1DemoRunner.evaluate_entry() [gx1/execution/oanda_demo_runner.py:5500]
        │     └─> EntryManager.evaluate_entry() [gx1/execution/entry_manager.py:196]
        └─> GX1DemoRunner.evaluate_and_close_trades() [gx1/execution/oanda_demo_runner.py:6072]
              └─> ExitManager.evaluate_and_close_trades() [gx1/execution/exit_manager.py:39]
```

---

## Entry Manager Call Chain

```
EntryManager.evaluate_entry() [gx1/execution/entry_manager.py:196]
  ├─> build_live_entry_features() [gx1/execution/live_features.py]
  │     └─> build_v9_runtime_features() [gx1/features/runtime_v9.py]
  │           └─> (feature pipeline computes ATR, spread, etc.)
  │
  ├─> entry_policy.predict() [UKJENT - entry_v9_policy_farm_v2b.py]
  │     └─> (FARM_V2B transformer model inference)
  │
  ├─> EntryManager._compute_range_features() [gx1/execution/entry_manager.py:69]
  │     └─> (computes range_pos, distance_to_range)
  │
  ├─> EntryManager._compute_range_edge_dist_atr() [gx1/execution/entry_manager.py:147]
  │     └─> (computes range_edge_dist_atr)
  │
  ├─> ExitModeSelector.choose_exit_profile() [gx1/policy/exit_hybrid_controller.py:47]
  │     ├─> ExitRouterContext() [gx1/core/hybrid_exit_router.py:24]
  │     ├─> hybrid_exit_router_v3() [gx1/core/hybrid_exit_router.py:177]
  │     │     ├─> joblib.load() [lazy load model]
  │     │     └─> model.predict() [sklearn DecisionTreeClassifier]
  │     └─> (guardrail check) [gx1/policy/exit_hybrid_controller.py:101]
  │
  └─> LiveTrade.__init__() [gx1/execution/oanda_demo_runner.py]
        └─> (sets trade.extra with range features)
```

---

## Exit Manager Call Chain

```
ExitManager.evaluate_and_close_trades() [gx1/execution/exit_manager.py:39]
  ├─> ExitManager._compute_runtime_atr_bps() [UKJENT - må verifiseres]
  │
  └─> (for each open trade)
        ├─> trade.extra.get("exit_profile") [reads from trade object]
        │
        ├─> get_exit_policy_farm_v2_rules() [gx1/policy/exit_farm_v2_rules.py]
        │     └─> (RULE5 exit logic)
        │
        └─> get_exit_policy_farm_v2_rules_adaptive() [gx1/policy/exit_farm_v2_rules_adaptive.py]
              └─> (RULE6A exit logic)
```

---

## Router Call Chain

```
ExitModeSelector.choose_exit_profile() [gx1/policy/exit_hybrid_controller.py:47]
  ├─> ExitRouterContext() [gx1/core/hybrid_exit_router.py:24]
  │     └─> (dataclass with atr_pct, spread_pct, range_pos, distance_to_range, range_edge_dist_atr)
  │
  ├─> hybrid_exit_router_v3() [gx1/core/hybrid_exit_router.py:177]
  │     ├─> joblib.load("exit_router_v3_tree.pkl") [lazy load, cached]
  │     ├─> pd.DataFrame() [prepare features]
  │     ├─> model.predict(X) [sklearn DecisionTreeClassifier]
  │     └─> (fallback to hardcoded tree if model fails)
  │
  └─> (guardrail post-processing) [gx1/policy/exit_hybrid_controller.py:101]
        └─> if policy == "RULE6A" and range_edge_dist_atr >= cutoff:
              └─> policy = "RULE5"
```

---

## Training Pipeline Call Chain

### Dataset Building

```
build_exit_policy_training_dataset_v3.py::main() [gx1/analysis/build_exit_policy_training_dataset_v3.py:368]
  ├─> _extract_trade_data() [gx1/analysis/build_exit_policy_training_dataset_v3.py:92]
  │     ├─> json.loads(trade.extra) [parse extra JSON]
  │     └─> (extracts range_pos, distance_to_range, range_edge_dist_atr)
  │
  ├─> build_dataset_v3() [gx1/analysis/build_exit_policy_training_dataset_v3.py:~343]
  │     ├─> pd.to_numeric() [clean range features]
  │     ├─> fillna() [handle missing values]
  │     └─> clip() [clamp to valid ranges]
  │
  └─> (save dataset + metadata)
```

### Router Training

```
router_training_v3.py::main() [gx1/analysis/router_training_v3.py:~200]
  ├─> load_dataset_v3() [gx1/analysis/router_training_v3.py:26]
  ├─> prepare_features_v3() [gx1/analysis/router_training_v3.py:65]
  │     ├─> _clean_distance_to_range() [gx1/analysis/router_training_v3.py:40]
  │     └─> (prepare numeric + categorical features)
  │
  ├─> build_preprocessor() [gx1/analysis/router_training_v3.py:139]
  │     ├─> OneHotEncoder() [for categoricals]
  │     └─> ColumnTransformer() [combine transformers]
  │
  ├─> train_test_split() [sklearn]
  ├─> DecisionTreeClassifier.fit() [sklearn]
  ├─> joblib.dump() [save model]
  └─> export_text() [export tree rules]
```

---

## Key Function Locations

| Function | File | Line | Called From |
|----------|------|------|-------------|
| `load_yaml_config()` | `gx1/execution/oanda_demo_runner.py` | 416 | `GX1DemoRunner.__init__()`, `replay_entry_exit_parallel.py` |
| `load_entry_models()` | `gx1/execution/oanda_demo_runner.py` | 444 | `GX1DemoRunner.__init__()` (line 2184) |
| `EntryManager.evaluate_entry()` | `gx1/execution/entry_manager.py` | 196 | `GX1DemoRunner.evaluate_entry()` |
| `_compute_range_features()` | `gx1/execution/entry_manager.py` | 69 | `EntryManager.evaluate_entry()` |
| `_compute_range_edge_dist_atr()` | `gx1/execution/entry_manager.py` | 147 | `EntryManager.evaluate_entry()` |
| `ExitModeSelector.choose_exit_profile()` | `gx1/policy/exit_hybrid_controller.py` | 47 | `EntryManager.evaluate_entry()` |
| `hybrid_exit_router_v3()` | `gx1/core/hybrid_exit_router.py` | 177 | `ExitModeSelector.choose_exit_profile()` |
| `ExitManager.evaluate_and_close_trades()` | `gx1/execution/exit_manager.py` | 39 | `GX1DemoRunner.evaluate_and_close_trades()` |
| `build_live_entry_features()` | `gx1/execution/live_features.py` | UKJENT | `EntryManager.evaluate_entry()` |
| `build_v9_runtime_features()` | `gx1/features/runtime_v9.py` | UKJENT | `build_live_entry_features()` |
| `_run_once_impl()` | `gx1/execution/oanda_demo_runner.py` | 6169 | Main replay/live loop |
| `_execute_entry_impl()` | `gx1/execution/oanda_demo_runner.py` | 5913 | Entry execution (sets kill-switch on failures) |

---

## Guardrail Call Chain

```
EntryManager.evaluate_entry() [gx1/execution/entry_manager.py:196]
  └─> ExitModeSelector.choose_exit_profile() [gx1/policy/exit_hybrid_controller.py:47]
        └─> hybrid_exit_router_v3() [gx1/core/hybrid_exit_router.py:177]
              └─> (returns "RULE5" or "RULE6A")
        └─> (guardrail check) [gx1/policy/exit_hybrid_controller.py:101]
              └─> if policy == "RULE6A" and range_edge_dist_atr >= cutoff:
                    └─> policy = "RULE5" [override]
```

**Guardrail Location:** `gx1/policy/exit_hybrid_controller.py`, lines 101-110  
**Guardrail Cutoff:** `self.v3_range_edge_cutoff` (default: 1.0, from config)

---

## Model Loading Call Chain

```
hybrid_exit_router_v3() [gx1/core/hybrid_exit_router.py:177]
  └─> (first call)
        └─> joblib.load("exit_router_v3_tree.pkl") [gx1/core/hybrid_exit_router.py:217]
              └─> (cached in hybrid_exit_router_v3._model_cache)
  └─> (subsequent calls)
        └─> (uses cached model)
```

**Model Path:** `gx1/analysis/exit_router_models_v3/exit_router_v3_tree.pkl`  
**Fallback:** Hardcoded tree logic (lines 264-291) if model load fails

---

## Analysis Scripts Call Chain

```
compare_exit_routers_fullyear.py::main() [gx1/analysis/compare_exit_routers_fullyear.py]
  ├─> load_trade_logs() [UKJENT - må verifiseres]
  ├─> run_regime_analysis() [UKJENT - må verifiseres]
  │     ├─> analyze_range_edge_buckets() [UKJENT - må verifiseres]
  │     └─> analyze_time_windows() [UKJENT - må verifiseres]
  └─> run_intratrade_risk_analysis() [UKJENT - må verifiseres]
        └─> calculate_intratrade_metrics() [UKJENT - må verifiseres]
```

---

## Notes

- **Lazy Loading:** Router model loaded on first call, cached thereafter
- **Fallback Logic:** Router falls back to hardcoded tree if model load fails
- **Guardrail Post-Processing:** Applied after router prediction, before returning profile
- **Range Features:** Computed in EntryManager before router selection
- **Parallel Execution:** Each chunk runs independently, results merged at end

