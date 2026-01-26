# PREBUILT REPLAY - FILKARTLEGGING

**Dato:** 2025-01-13  
**Status:** Systematisk gjennomgang av workspace

## ✅ FUNNET I WORKSPACE

### Bash Scripts (Implementert)
- ✅ `scripts/go_nogo_prebuilt.sh` - Preflight check, oppretter marker
- ✅ `scripts/run_fullyear_prebuilt.sh` - FULLYEAR replay med gates

### Dokumentasjon
- ✅ `docs/PREBUILT_REPLAY_FLOW.md` - Kanonisk dokumentasjon
- ✅ `docs/PREBUILT_IMPLEMENTATION_STATUS.md` - Implementasjonsstatus

### Refererte Filer (Må Verifiseres)
- `gx1/scripts/replay_eval_gated_parallel.py` - **REFERERT MEN IKKE FUNNET**
  - Referert i: `scripts/go_nogo_prebuilt.sh` (linje 69, 88, 184)
  - Referert i: `scripts/run_fullyear_prebuilt.sh` (linje 129)
  - **STATUS:** Må opprettes eller finnes et annet sted

## ⚠️ FEATURE-BUILDING FUNKSJONER

### build_v9_runtime_features
- **Lokasjon:** `_archive_v9/features/runtime_v9.py` (linje 438)
- **Importeres fra:** `gx1.features.runtime_v9` (i flere filer)
- **Status:** Filen ligger i archive, men importeres som aktiv
- **TODO:** Verifiser om det er en aktiv versjon eller om archive-versjonen brukes

### build_basic_v1
- **Importeres fra:** `gx1.features.basic_v1` (i `_archive_v9/features/runtime_v9.py` linje 20)
- **Status:** Filen `gx1/features/basic_v1.py` finnes IKKE i workspace
- **TODO:** Må finnes eller opprettes

### build_live_entry_features
- **Referert i dokumentasjon:** `gx1/execution/live_features.py:build_live_entry_features`
- **Status:** Filen `gx1/execution/live_features.py` finnes IKKE i workspace (slettet i git)
- **TODO:** Må finnes eller funksjonaliteten må være flyttet

## 📋 RELEVANTE FILER I WORKSPACE

### gx1/features/
- `__init__.py`
- `array_utils.py`
- `feature_manifest.py`
- `htf_aggregator.py`
- `pandas_ops_timer.py`
- `rolling_logger.py`
- `rolling_np.py`
- `rolling_state_numba.py`
- `rolling_timer.py`
- **Mangler:** `basic_v1.py`, `runtime_v9.py` (aktiv versjon)

### gx1/execution/
- `broker_client.py`
- `entry_context_features.py`
- `exec_smoke_test.py`
- `exit_critic_controller.py`
- `exit_manager.py`
- `fast_path_verification.py`
- `oanda_backfill.py`
- `oanda_client.py`
- `oanda_credentials.py`
- `replay_engine.py`
- `replay_features.py`
- `runtime_mode.py`
- `telemetry.py`
- `trade_log_schema.py`
- **Mangler:** `live_features.py` (slettet i git)

### gx1/scripts/
- `analyze_live_day.py`
- `analyze_sniper_live_p_long.py`
- `archive_checkpoints_no_resume.py`
- `audit_hash_duplicates_and_cleanup.py`
- `audit_runs_inventory.py`
- `audit_v9_models.py`
- `backfill_xauusd_m5_from_oanda.py`
- `build_entry_timing_dataset_v1.py`
- `build_entry_v10_ctx_training_dataset.py`
- `canonicalize_exit_router.py`
- `check_live_coverage.py`
- `cleanup_execution_plan.py`
- `eval_entry_policies_v1.py`
- `generate_v9_archive_script.py`
- `preflight_full_build_sanity.py`
- `replay_with_shadow_parallel.py`
- `replay_with_shadow.py`
- `resolve_unknown_artifacts.py`
- `resume_sniper_after_sleep.py`
- `run_shadow_counterfactual.py`
- `run_sniper_quarter_replays.py`
- `train_xgb_calibrators.py`
- `verify_runtime_after_archive.py`
- **Mangler:** `replay_eval_gated_parallel.py`

## 🔍 IMPORTERINGER SOM REFERERER TIL MANGLENDE FILER

### Fra `_archive_v9/features/runtime_v9.py`:
```python
from gx1.features.basic_v1 import build_basic_v1  # Linje 20
```

### Fra flere filer:
```python
from gx1.features.runtime_v9 import build_v9_runtime_features
```

### Fra dokumentasjon:
- `gx1/execution/live_features.py:build_live_entry_features` (referert men filen slettet)

## 📝 KONKLUSJON

1. **replay_eval_gated_parallel.py** må opprettes eller finnes
2. **gx1/features/basic_v1.py** må finnes eller opprettes
3. **gx1/features/runtime_v9.py** (aktiv versjon) må finnes eller archive-versjonen må brukes
4. **gx1/execution/live_features.py** er slettet - funksjonaliteten må være flyttet eller gjenopprettes

## 🎯 NESTE STEG

1. Verifiser om filene faktisk finnes (kanskje utenfor worktree)
2. Hvis de mangler, må de opprettes eller funksjonaliteten må lokaliseres
3. Implementer sikkerhetssjekker i de faktiske filene når de er lokalisert
