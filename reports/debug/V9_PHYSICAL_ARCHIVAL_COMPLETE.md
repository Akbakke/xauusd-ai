# ✅ V9 PHYSICAL ARCHIVAL COMPLETE

**Date:** 2026-01-10  
**Status:** ✅ V9 ERADICATED, ARCHIVED, AND TOMBSTONED

## Executive Summary

V9 has been **physically archived** and **tombstoned**. Replay mode is now **100% V9-free** with **zero possibility of import side effects**. SSoT headers are written for easy audit.

## Implementation

### ✅ DEL 1: Physical Archival
- **Archived files:**
  - `gx1/policy/entry_v9_policy_sniper.py` → `_archive_v9/policy/entry_v9_policy_sniper.py`
  - `gx1/features/runtime_v9.py` → `_archive_v9/features/runtime_v9.py`

- **Archive structure:**
  ```
  _archive_v9/
  ├── policy/
  │   └── entry_v9_policy_sniper.py (original implementation)
  └── features/
      └── runtime_v9.py (original implementation)
  ```

### ✅ DEL 2: Tombstones
- **Original files replaced with tombstones:**
  - `gx1/policy/entry_v9_policy_sniper.py` - Tombstone (hard-fails in replay mode)
  - `gx1/features/runtime_v9.py` - Tombstone (hard-fails in replay mode)

- **Tombstone behavior:**
  - **Replay mode:** `RuntimeError("V9_REMOVED: ...")` - **HARD FAIL**
  - **Live mode:** Loads from archive (backward compatibility, deprecated)

- **Test results:**
  - ✅ Tombstone blocks V9 import in replay mode
  - ✅ Tombstone allows V9 import in live mode (backward compatibility)
  - ✅ Runtime tombstone blocks V9 import in replay mode

### ✅ DEL 3: SSoT Assert (Hard-Fail in Replay-Start)
- **Location:** `gx1/execution/oanda_demo_runner.py` → `_run_replay_impl()`
- **Assertions:**
  - ✅ `policy_module` must be `gx1.policy.entry_policy_sniper_v10_ctx`
  - ✅ `entry_model_id` must be present and explicit
  - ✅ `runtime_feature_module` must be `gx1.features.runtime_v10_ctx`
  - ✅ `bundle_sha256` must be present (for audit trail)
  
- **Hard-fail on mismatch:**
  - `REPLAY_SSoT_ASSERT_FAILED` - stops replay immediately

### ✅ DEL 4: SSoT Logging (Explicit One-Line)
- **Location:** `[REPLAY_SSoT]` log block at replay-start
- **Printed once at replay-start:**
  ```
  [REPLAY_SSoT]   policy_module: gx1.policy.entry_policy_sniper_v10_ctx
  [REPLAY_SSoT]   runtime_feature_module: gx1.features.runtime_v10_ctx
  [REPLAY_SSoT]   entry_model_id: ENTRY_V10_CTX_GATED_FUSION
  [REPLAY_SSoT]   bundle_sha256: 97d90be83aa980cd82a0b321d0edf9369363298effcb6ca4a043eebd6e381deb
  ```

### ✅ DEL 5: REPLAY_SSoT_HEADER.json (Machine-Readable)
- **Location:** `gx1/wf_runs/<run_id>/REPLAY_SSoT_HEADER.json`
- **Structure:**
  ```json
  {
    "run_id": "20260110_203509",
    "chunk_id": "single",
    "timestamp": "2026-01-10T19:35:10.576425+00:00",
    "replay_sso": {
      "policy_module": "gx1.policy.entry_policy_sniper_v10_ctx",
      "policy_class": "EntryPolicySniperV10Ctx",
      "policy_id": "entry_policy_sniper_v10_ctx",
      "runtime_feature_module": "gx1.features.runtime_v10_ctx",
      "runtime_feature_impl": "v10_ctx",
      "entry_model_id": "ENTRY_V10_CTX_GATED_FUSION",
      "bundle_path_abs": "/path/to/bundle",
      "bundle_sha256": "97d90be83aa980cd82a0b321d0edf9369363298effcb6ca4a043eebd6e381deb",
      "policy_config_path": "/path/to/policy.yaml",
      "entry_config_path": "gx1/configs/entry_configs/ENTRY_V10_CTX_SNIPER_REPLAY.yaml",
      "v9_forbidden": true
    },
    "provenance": {
      "git_commit": "2a79bcdfee56cdc6c92586d4b069bb2b15fb758b",
      "worker_pid": 89329
    }
  }
  ```

## ✅ Verification

### 1. Tombstone Test
```bash
# Replay mode (should fail)
export GX1_REPLAY=1
python3 -c "from gx1.policy.entry_v9_policy_sniper import apply_entry_v9_policy_sniper"
# Result: RuntimeError("V9_REMOVED: entry_v9_policy_sniper is forbidden in replay mode...")

# Live mode (should work, backward compatibility)
export GX1_REPLAY=0
python3 -c "from gx1.policy.entry_v9_policy_sniper import apply_entry_v9_policy_sniper"
# Result: ✅ Import successful (loads from archive)
```

### 2. Mini Replay Test
- ✅ Mini replay runs successfully
- ✅ SSoT header written: `gx1/wf_runs/20260110_203509/REPLAY_SSoT_HEADER.json`
- ✅ All invariants pass
- ✅ Ghostbusters scan: 0 violations
- ✅ No V9 references in artifacts

### 3. SSoT Assert Test
- ✅ SSoT assert hard-fails if `policy_module` is missing/wrong
- ✅ SSoT assert hard-fails if `runtime_feature_module` is missing/wrong
- ✅ SSoT assert hard-fails if `bundle_sha256` is missing

## Audit Trail (6 Months Later)

To audit a replay run from 6 months ago:
1. **Read SSoT header:** `cat gx1/wf_runs/<run_id>/REPLAY_SSoT_HEADER.json`
2. **Verify bundle:** Check `bundle_sha256` matches expected bundle
3. **Verify policy:** Check `policy_module` is V10_CTX (not V9)
4. **Verify runtime:** Check `runtime_feature_module` is `gx1.features.runtime_v10_ctx`
5. **Check timestamp:** Verify `timestamp` matches run time

**One-liner audit:**
```bash
jq '.replay_sso | {policy_module, runtime_feature_module, entry_model_id, bundle_sha256, v9_forbidden}' gx1/wf_runs/<run_id>/REPLAY_SSoT_HEADER.json
```

## Files Changed

### Tombstones (Hard-Fail in Replay Mode)
- `gx1/policy/entry_v9_policy_sniper.py` - Tombstone
- `gx1/features/runtime_v9.py` - Tombstone

### Archive (Original Implementation)
- `_archive_v9/policy/entry_v9_policy_sniper.py` - Original (for live mode backward compatibility)
- `_archive_v9/features/runtime_v9.py` - Original (for live mode backward compatibility)

### Code Changes
- `gx1/execution/oanda_demo_runner.py` - Added SSoT assert + REPLAY_SSoT_HEADER.json writer

## Conclusion

**V9 IS PHYSICALLY ARCHIVED, TOMBSTONED, AND VERIFIED.**

- ✅ **Zero import side effects:** Tombstones hard-fail in replay mode
- ✅ **Backward compatibility:** Live mode can still use archived V9 (deprecated)
- ✅ **SSoT headers:** Machine-readable audit trail for every replay run
- ✅ **Hard-fail asserts:** SSoT asserts stop replay if critical fields are missing/wrong
- ✅ **Easy audit:** One JSON file contains all critical metadata for audit

**Next steps:**
- This can now be committed as: `"V9 physically archived and tombstoned - SSoT headers added"`
- V9 code in `_archive_v9/` can be optionally deleted later (but keeping for live mode backward compatibility for now)
- This becomes the **new SSoT** for replay mode

