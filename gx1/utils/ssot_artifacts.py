"""
SSoT artifacts loader for v13_core.

In TRUTH/SMOKE:
- Requires a single canonical config file to exist
- Requires contract_id match
- Requires all required paths to be absolute + exist
- No fallback

In DEV:
- Allows empty paths (placeholder config)
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class SSoTArtifacts:
    feature_contract_id: str
    prebuilt_schema_manifest_path: str
    prebuilt_parquet_path: str
    bundle_dir: str
    master_model_lock_path: str


def _is_truth_or_smoke() -> bool:
    mode = os.getenv("GX1_RUN_MODE", "").upper()
    return os.getenv("GX1_TRUTH_MODE", "0") == "1" or mode in {"TRUTH", "SMOKE"}


def load_ssot_artifacts_v13_core(config_path: Optional[str] = None) -> SSoTArtifacts:
    """
    Load and validate SSoT artifacts config for v13_core.

    Args:
        config_path: Optional override. If None, uses gx1/configs/ssot_artifacts_v13_core.json.
    """
    is_strict = _is_truth_or_smoke()

    if config_path is None:
        workspace_root = Path(__file__).resolve().parents[2]
        config_path_obj = workspace_root / "gx1" / "configs" / "ssot_artifacts_v13_core.json"
    else:
        config_path_obj = Path(str(config_path))

    if not config_path_obj.exists():
        raise RuntimeError(f"[SSOT_ARTIFACTS_FAIL] Missing ssot config: {config_path_obj}")

    obj: Dict[str, Any] = json.loads(config_path_obj.read_text(encoding="utf-8"))
    feature_contract_id = str(obj.get("feature_contract_id") or "")
    if feature_contract_id != "v13_core":
        raise RuntimeError(
            "[SSOT_ARTIFACTS_FAIL] feature_contract_id mismatch in ssot config: "
            f"expected=v13_core got={feature_contract_id or 'MISSING'} path={config_path_obj}"
        )

    def get_path(key: str) -> str:
        return str(obj.get(key) or "")

    prebuilt_schema_manifest_path = get_path("prebuilt_schema_manifest_path")
    prebuilt_parquet_path = get_path("prebuilt_parquet_path")
    bundle_dir = get_path("bundle_dir")
    master_model_lock_path = get_path("master_model_lock_path")

    # Strict validation
    if is_strict:
        required = {
            "prebuilt_schema_manifest_path": prebuilt_schema_manifest_path,
            "prebuilt_parquet_path": prebuilt_parquet_path,
            "bundle_dir": bundle_dir,
            "master_model_lock_path": master_model_lock_path,
        }
        for k, v in required.items():
            if not v:
                raise RuntimeError(
                    f"[SSOT_ARTIFACTS_FAIL] TRUTH_NO_FALLBACK: required path is empty: {k} (config={config_path_obj})"
                )
            p = Path(v)
            if not p.is_absolute():
                raise RuntimeError(
                    f"[SSOT_ARTIFACTS_FAIL] TRUTH_NO_FALLBACK: path must be absolute: {k}={v}"
                )
            if k == "bundle_dir":
                if not p.exists() or not p.is_dir():
                    raise RuntimeError(f"[SSOT_ARTIFACTS_FAIL] bundle_dir missing/not dir: {v}")
            else:
                if not p.exists() or not p.is_file():
                    raise RuntimeError(f"[SSOT_ARTIFACTS_FAIL] path missing/not file: {k}={v}")
    else:
        # DEV mode: if provided, must be absolute.
        for k, v in {
            "prebuilt_schema_manifest_path": prebuilt_schema_manifest_path,
            "prebuilt_parquet_path": prebuilt_parquet_path,
            "bundle_dir": bundle_dir,
            "master_model_lock_path": master_model_lock_path,
        }.items():
            if v:
                if not Path(v).is_absolute():
                    raise RuntimeError(f"[SSOT_ARTIFACTS_FAIL] path must be absolute (DEV too): {k}={v}")

    return SSoTArtifacts(
        feature_contract_id=feature_contract_id,
        prebuilt_schema_manifest_path=prebuilt_schema_manifest_path,
        prebuilt_parquet_path=prebuilt_parquet_path,
        bundle_dir=bundle_dir,
        master_model_lock_path=master_model_lock_path,
    )


def load_ssot_artifacts_v13_refined(config_path: Optional[str] = None) -> SSoTArtifacts:
    """
    Load and validate SSoT artifacts config for v13_refined.

    Args:
        config_path: Optional override. If None, uses gx1/configs/ssot_artifacts_v13_refined.json.
    """
    is_strict = _is_truth_or_smoke()

    if config_path is None:
        workspace_root = Path(__file__).resolve().parents[2]
        config_path_obj = workspace_root / "gx1" / "configs" / "ssot_artifacts_v13_refined.json"
    else:
        config_path_obj = Path(str(config_path))

    if not config_path_obj.exists():
        raise RuntimeError(f"[SSOT_ARTIFACTS_FAIL] Missing ssot config: {config_path_obj}")

    obj: Dict[str, Any] = json.loads(config_path_obj.read_text(encoding="utf-8"))
    feature_contract_id = str(obj.get("feature_contract_id") or "")
    if feature_contract_id != "v13_refined":
        raise RuntimeError(
            "[SSOT_ARTIFACTS_FAIL] feature_contract_id mismatch in ssot config: "
            f"expected=v13_refined got={feature_contract_id or 'MISSING'} path={config_path_obj}"
        )

    def get_path(key: str) -> str:
        return str(obj.get(key) or "")

    prebuilt_schema_manifest_path = get_path("prebuilt_schema_manifest_path")
    prebuilt_parquet_path = get_path("prebuilt_parquet_path")
    bundle_dir = get_path("bundle_dir")
    master_model_lock_path = get_path("master_model_lock_path")

    if is_strict:
        required = {
            "prebuilt_schema_manifest_path": prebuilt_schema_manifest_path,
            "prebuilt_parquet_path": prebuilt_parquet_path,
            "bundle_dir": bundle_dir,
            "master_model_lock_path": master_model_lock_path,
        }
        for k, v in required.items():
            if not v:
                raise RuntimeError(
                    f"[SSOT_ARTIFACTS_FAIL] TRUTH_NO_FALLBACK: required path is empty: {k} (config={config_path_obj})"
                )
            p = Path(v)
            if not p.is_absolute():
                raise RuntimeError(
                    f"[SSOT_ARTIFACTS_FAIL] TRUTH_NO_FALLBACK: path must be absolute: {k}={v}"
                )
            if k == "bundle_dir":
                if not p.exists() or not p.is_dir():
                    raise RuntimeError(f"[SSOT_ARTIFACTS_FAIL] bundle_dir missing/not dir: {v}")
            else:
                if not p.exists() or not p.is_file():
                    raise RuntimeError(f"[SSOT_ARTIFACTS_FAIL] path missing/not file: {k}={v}")
    else:
        for k, v in {
            "prebuilt_schema_manifest_path": prebuilt_schema_manifest_path,
            "prebuilt_parquet_path": prebuilt_parquet_path,
            "bundle_dir": bundle_dir,
            "master_model_lock_path": master_model_lock_path,
        }.items():
            if v:
                if not Path(v).is_absolute():
                    raise RuntimeError(f"[SSOT_ARTIFACTS_FAIL] path must be absolute (DEV too): {k}={v}")

    return SSoTArtifacts(
        feature_contract_id=feature_contract_id,
        prebuilt_schema_manifest_path=prebuilt_schema_manifest_path,
        prebuilt_parquet_path=prebuilt_parquet_path,
        bundle_dir=bundle_dir,
        master_model_lock_path=master_model_lock_path,
    )


def load_ssot_artifacts_v13_refined2(config_path: Optional[str] = None) -> SSoTArtifacts:
    """
    Load and validate SSoT artifacts config for v13_refined2.
    """
    is_strict = _is_truth_or_smoke()

    if config_path is None:
        workspace_root = Path(__file__).resolve().parents[2]
        config_path_obj = workspace_root / "gx1" / "configs" / "ssot_artifacts_v13_refined2.json"
    else:
        config_path_obj = Path(str(config_path))

    if not config_path_obj.exists():
        raise RuntimeError(f"[SSOT_ARTIFACTS_FAIL] Missing ssot config: {config_path_obj}")

    obj: Dict[str, Any] = json.loads(config_path_obj.read_text(encoding="utf-8"))
    feature_contract_id = str(obj.get("feature_contract_id") or "")
    if feature_contract_id != "v13_refined2":
        raise RuntimeError(
            "[SSOT_ARTIFACTS_FAIL] feature_contract_id mismatch in ssot config: "
            f"expected=v13_refined2 got={feature_contract_id or 'MISSING'} path={config_path_obj}"
        )

    def get_path(key: str) -> str:
        return str(obj.get(key) or "")

    prebuilt_schema_manifest_path = get_path("prebuilt_schema_manifest_path")
    prebuilt_parquet_path = get_path("prebuilt_parquet_path")
    bundle_dir = get_path("bundle_dir")
    master_model_lock_path = get_path("master_model_lock_path")

    if is_strict:
        required = {
            "prebuilt_schema_manifest_path": prebuilt_schema_manifest_path,
            "prebuilt_parquet_path": prebuilt_parquet_path,
            "bundle_dir": bundle_dir,
            "master_model_lock_path": master_model_lock_path,
        }
        for k, v in required.items():
            if not v:
                raise RuntimeError(
                    f"[SSOT_ARTIFACTS_FAIL] TRUTH_NO_FALLBACK: required path is empty: {k} (config={config_path_obj})"
                )
            p = Path(v)
            if not p.is_absolute():
                raise RuntimeError(
                    f"[SSOT_ARTIFACTS_FAIL] TRUTH_NO_FALLBACK: path must be absolute: {k}={v}"
                )
            if k == "bundle_dir":
                if not p.exists() or not p.is_dir():
                    raise RuntimeError(f"[SSOT_ARTIFACTS_FAIL] bundle_dir missing/not dir: {v}")
            else:
                if not p.exists() or not p.is_file():
                    raise RuntimeError(f"[SSOT_ARTIFACTS_FAIL] path missing/not file: {k}={v}")
    else:
        for k, v in {
            "prebuilt_schema_manifest_path": prebuilt_schema_manifest_path,
            "prebuilt_parquet_path": prebuilt_parquet_path,
            "bundle_dir": bundle_dir,
            "master_model_lock_path": master_model_lock_path,
        }.items():
            if v:
                if not Path(v).is_absolute():
                    raise RuntimeError(f"[SSOT_ARTIFACTS_FAIL] path must be absolute (DEV too): {k}={v}")

    return SSoTArtifacts(
        feature_contract_id=feature_contract_id,
        prebuilt_schema_manifest_path=prebuilt_schema_manifest_path,
        prebuilt_parquet_path=prebuilt_parquet_path,
        bundle_dir=bundle_dir,
        master_model_lock_path=master_model_lock_path,
    )


def load_ssot_artifacts_v13_refined3(config_path: Optional[str] = None) -> SSoTArtifacts:
    """
    Load and validate SSoT artifacts config for v13_refined3.
    """
    is_strict = _is_truth_or_smoke()

    if config_path is None:
        workspace_root = Path(__file__).resolve().parents[2]
        config_path_obj = workspace_root / "gx1" / "configs" / "ssot_artifacts_v13_refined3.json"
    else:
        config_path_obj = Path(str(config_path))

    if not config_path_obj.exists():
        raise RuntimeError(f"[SSOT_ARTIFACTS_FAIL] Missing ssot config: {config_path_obj}")

    obj: Dict[str, Any] = json.loads(config_path_obj.read_text(encoding="utf-8"))
    feature_contract_id = str(obj.get("feature_contract_id") or "")
    if feature_contract_id != "v13_refined3":
        raise RuntimeError(
            "[SSOT_ARTIFACTS_FAIL] feature_contract_id mismatch in ssot config: "
            f"expected=v13_refined3 got={feature_contract_id or 'MISSING'} path={config_path_obj}"
        )

    def get_path(key: str) -> str:
        return str(obj.get(key) or "")

    prebuilt_schema_manifest_path = get_path("prebuilt_schema_manifest_path")
    prebuilt_parquet_path = get_path("prebuilt_parquet_path")
    bundle_dir = get_path("bundle_dir")
    master_model_lock_path = get_path("master_model_lock_path")

    if is_strict:
        required = {
            "prebuilt_schema_manifest_path": prebuilt_schema_manifest_path,
            "prebuilt_parquet_path": prebuilt_parquet_path,
            "bundle_dir": bundle_dir,
            "master_model_lock_path": master_model_lock_path,
        }
        for k, v in required.items():
            if not v:
                raise RuntimeError(
                    f"[SSOT_ARTIFACTS_FAIL] TRUTH_NO_FALLBACK: required path is empty: {k} (config={config_path_obj})"
                )
            p = Path(v)
            if not p.is_absolute():
                raise RuntimeError(
                    f"[SSOT_ARTIFACTS_FAIL] TRUTH_NO_FALLBACK: path must be absolute: {k}={v}"
                )
            if k == "bundle_dir":
                if not p.exists() or not p.is_dir():
                    raise RuntimeError(f"[SSOT_ARTIFACTS_FAIL] bundle_dir missing/not dir: {v}")
            else:
                if not p.exists() or not p.is_file():
                    raise RuntimeError(f"[SSOT_ARTIFACTS_FAIL] path missing/not file: {k}={v}")
    else:
        for k, v in {
            "prebuilt_schema_manifest_path": prebuilt_schema_manifest_path,
            "prebuilt_parquet_path": prebuilt_parquet_path,
            "bundle_dir": bundle_dir,
            "master_model_lock_path": master_model_lock_path,
        }.items():
            if v:
                if not Path(v).is_absolute():
                    raise RuntimeError(f"[SSOT_ARTIFACTS_FAIL] path must be absolute (DEV too): {k}={v}")

    return SSoTArtifacts(
        feature_contract_id=feature_contract_id,
        prebuilt_schema_manifest_path=prebuilt_schema_manifest_path,
        prebuilt_parquet_path=prebuilt_parquet_path,
        bundle_dir=bundle_dir,
        master_model_lock_path=master_model_lock_path,
    )

