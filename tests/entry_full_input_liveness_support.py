from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from gx1.contracts.entry_full_input_liveness_v1 import (
    classify_field_name_semantics,
    MULTI_TF_FEATURE_NAMES,
    MULTI_TF_TIMEFRAMES,
    build_full_input_liveness_artifact,
    sha256_file,
)
from gx1.contracts.entry_cross_surface_overlap_v1 import (
    DECISION_ROUTES,
    POLICY_VERSION as CROSS_SURFACE_POLICY_VERSION,
    SCHEMA_VERSION as CROSS_SURFACE_SCHEMA_VERSION,
    classify_active_duplicate_pairs,
    declared_context_mtf_aliases,
)
from gx1.features.htf_features import (
    HTF_V4_MATRIX_CONTRACT,
    build_multi_tf_v4_liveness_contract,
)
from gx1.contracts.entry_model_native_signal_v1 import (
    MODEL_NATIVE_CONTRACT_MODE,
    MODEL_NATIVE_CTX_CAT_DIM,
    MODEL_NATIVE_CTX_CONT_DIM,
    MODEL_NATIVE_CTX_CONT_FIELDS,
    MODEL_NATIVE_SIGNAL_DIM,
)


def full_input_field_order() -> dict[str, list[str]]:
    signal = [
        "smc_choch",
        "ctx_cont.d1_atr14_bps_canon_v2",
        "ctx_cont._v1h4_atr_bps",
        "chart.local_ema50_200_cross_up",
        "chart.local_ema50_200_cross_down",
        "chart.geomline_retest_fail_up",
        "chart.geomline_retest_fail_down",
        "h1_regime_changed_flag_v3",
        "h4_regime_changed_flag_v3",
    ]
    signal.extend(
        f"signal_feature_{idx:03d}"
        for idx in range(MODEL_NATIVE_SIGNAL_DIM - len(signal))
    )
    ctx_cont = [
        "d1_atr14_bps_canon_v2",
        "_v1h4_atr_bps",
        "d1_regime_changed_flag_v3",
    ]
    ctx_cont.extend(
        f"ctx_cont_feature_{idx:03d}"
        for idx in range(MODEL_NATIVE_CTX_CONT_DIM - len(ctx_cont))
    )
    ctx_cat = [f"ctx_cat_feature_{idx}" for idx in range(MODEL_NATIVE_CTX_CAT_DIM)]
    return {"signal": signal, "ctx_cont": ctx_cont, "ctx_cat": ctx_cat}


def full_input_stats(
    field_order: dict[str, list[str]],
    *,
    rows: int = 10000,
) -> dict[str, dict[str, dict[str, dict]]]:
    result: dict[str, dict[str, dict[str, dict]]] = {}
    for split in ("train", "val", "test"):
        split_rows: dict[str, dict[str, dict]] = {}
        for surface, fields in field_order.items():
            surface_rows: dict[str, dict] = {}
            for idx, field in enumerate(fields):
                if surface == "ctx_cat":
                    surface_rows[field] = {
                        "row_count": rows,
                        "finite_count": rows,
                        "nonfinite_count": 0,
                        "unique_count": 3,
                        "integer_like_count": rows,
                        "unique_values": [0, 1, 2],
                    }
                    continue
                # Synthetic values must honour the field's own name contract,
                # otherwise the fixture asserts a surface the contract forbids.
                # A monotone all-positive ramp silently violated every signed
                # field the moment the semantics gate landed.
                semantics = classify_field_name_semantics(field)
                if semantics == "signed":
                    low, high = -1.0 - idx * 0.001, 1.5 + idx * 0.001
                elif semantics == "unit_interval":
                    low, high = 0.0, 1.0
                else:
                    low, high = 0.5 + idx * 0.001, 1.5 + idx * 0.001
                surface_rows[field] = {
                    "row_count": rows,
                    "finite_count": rows,
                    "nonfinite_count": 0,
                    "mean": (low + high) / 2.0,
                    "std": 0.5,
                    "min": low,
                    "max": high,
                    "value_range": high - low,
                    "active_count": rows,
                    "active_rate": 1.0,
                }
            split_rows[surface] = surface_rows
        result[split] = split_rows
    return result


def cross_surface_hash_fixture(
    *,
    decision: str,
) -> tuple[dict[str, str], dict[str, str]]:
    """Build an exact-width synthetic report with owner-derived aliases.

    Multiple local representation paths may intentionally point at one MTF
    field.  They must therefore share the digest of that MTF value, while the
    total local signal-key count remains the model owner's exact width.
    """

    route = DECISION_ROUTES[decision]
    aliases = sorted(declared_context_mtf_aliases(decision=decision))
    signal_aliases = sorted(
        local for local, _mtf in aliases if local.startswith("local.signal.")
    )
    local_hashes = {
        f"local.signal.fixture_{index}": hashlib.sha256(
            f"signal:{decision}:{index}".encode("utf-8")
        ).hexdigest()
        for index in range(MODEL_NATIVE_SIGNAL_DIM - len(signal_aliases))
    }
    local_hashes.update(
        {
            local: hashlib.sha256(
                f"signal:{decision}:{local}".encode("utf-8")
            ).hexdigest()
            for local in signal_aliases
        }
    )
    local_hashes.update(
        {
            f"local.ctx_cont.{field}": hashlib.sha256(
                f"context:{decision}:{field}".encode("utf-8")
            ).hexdigest()
            for field in MODEL_NATIVE_CTX_CONT_FIELDS
        }
    )
    active_mtf_hashes = {
        f"mtf.{timeframe.lower()}.{field}": hashlib.sha256(
            f"mtf:{decision}:{timeframe}:{field}".encode("utf-8")
        ).hexdigest()
        for timeframe in route["active_mtf_timeframes"]
        for field in MULTI_TF_FEATURE_NAMES
    }
    for local, mtf in aliases:
        digest = hashlib.sha256(
            f"alias:{decision}:{mtf}".encode("utf-8")
        ).hexdigest()
        local_hashes[local] = digest
        active_mtf_hashes[mtf] = digest
    return local_hashes, active_mtf_hashes


def write_full_input_liveness_fixture(
    tmp_path: Path,
    *,
    dataset_dir: Path | None = None,
    field_order: dict[str, list[str]] | None = None,
    mutate_stats: Callable[[dict[str, dict[str, dict[str, dict]]]], None] | None = None,
) -> tuple[Path, dict, dict[str, dict[str, str]]]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    order = field_order or full_input_field_order()
    stats = full_input_stats(order)
    if mutate_stats is not None:
        mutate_stats(stats)
    dataset = (dataset_dir or (tmp_path / "smart_dataset")).resolve()
    dataset.mkdir(parents=True, exist_ok=True)
    manifest_bindings: dict[str, dict[str, str]] = {}
    scan_proof: dict[str, dict] = {}
    for split in ("train", "val", "test"):
        manifest = dataset / f"fixture_{split}.manifest.json"
        manifest.write_text(json.dumps({"split": split}) + "\n", encoding="utf-8")
        manifest_bindings[split] = {
            "path": str(manifest),
            "sha256": sha256_file(manifest),
        }
        parquet = dataset / f"fixture_{split}.parquet"
        parquet.write_bytes(f"{split}-fullscan-fixture".encode("utf-8"))
        scan_proof[split] = {
            "parquet_path": str(parquet),
            "size_bytes": parquet.stat().st_size,
            "mtime_ns": parquet.stat().st_mtime_ns,
            "total_rows": 10000,
            "scanned_rows": 10000,
            "fullscan": True,
            "scan_complete": True,
        }
    mtf_cache_dir = tmp_path / "MULTI_TF_CACHE"
    mtf_cache_dir.mkdir()
    mtf_manifest = mtf_cache_dir / "manifest.json"
    mtf_manifest.write_text(
        json.dumps({"cache_identity_sha256": "a" * 64}) + "\n",
        encoding="utf-8",
    )
    mtf_frames: dict[str, pd.DataFrame] = {}
    for timeframe in MULTI_TF_TIMEFRAMES:
        row = np.arange(32, dtype=np.float32)[:, None]
        column = np.arange(len(MULTI_TF_FEATURE_NAMES), dtype=np.float32)[None, :]
        values = row + column * np.float32(0.001)
        frame = pd.DataFrame(values, columns=MULTI_TF_FEATURE_NAMES)
        frame.attrs["feats_np"] = values.copy()
        frame.attrs["causal_warmup_rows"] = 1
        frame.attrs["htf_feature_contract"] = HTF_V4_MATRIX_CONTRACT
        mtf_frames[timeframe] = frame
    mtf_liveness = build_multi_tf_v4_liveness_contract(mtf_frames)
    cross_report = (
        tmp_path / "ENTRY_CROSS_SURFACE_INPUT_OVERLAP_20260716T000000000000Z.json"
    )
    cross_run_id = "FIXTURE_CROSS_SURFACE_20260716"
    cross_payload: dict[str, object] = {
        "schema_version": CROSS_SURFACE_SCHEMA_VERSION,
        "entry_run_id": cross_run_id,
        "decision": "PASS",
        "failures": [],
        "policy": {
            "version": CROSS_SURFACE_POLICY_VERSION,
            "decision_population": "manifest_bound_history_start_through_surface_end",
            "decision_routes": {
                decision: {
                    "local_timeframe": route["local_timeframe"],
                    "active_mtf_timeframes": list(route["active_mtf_timeframes"]),
                }
                for decision, route in DECISION_ROUTES.items()
            },
        },
        "input_bindings": {
            "signal_manifest": {
                "feature_history_start_utc": "2026-01-01T00:00:00+00:00"
            }
        },
        "eight_family_coverage": {
            f"family_{index}": {"local_field_count": 1, "mtf_field_count": 1}
            for index in range(8)
        },
    }
    for decision, route in DECISION_ROUTES.items():
        local_hashes, active_mtf_hashes = cross_surface_hash_fixture(
            decision=decision
        )
        cross_payload[decision] = {
            "local_timeframe": route["local_timeframe"],
            "active_mtf_timeframes": list(route["active_mtf_timeframes"]),
            "row_count": 1,
            "source_row_count": 2,
            "excluded_pre_history_row_count": 1,
            "audit_start_time_ns": int(pd.Timestamp("2026-01-01T00:00:00Z").value),
            "source_first_time_ns": int(pd.Timestamp("2025-12-31T23:55:00Z").value),
            "first_time_ns": int(pd.Timestamp("2026-01-01T00:00:00Z").value),
            "last_time_ns": int(pd.Timestamp("2026-01-01T00:00:00Z").value),
            "local_field_hashes": local_hashes,
            "active_mtf_field_hashes": active_mtf_hashes,
            **classify_active_duplicate_pairs(
                decision=decision,
                local_field_hashes=local_hashes,
                active_mtf_field_hashes=active_mtf_hashes,
            ),
        }
    cross_report.write_text(
        json.dumps(cross_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    cross_binding = {
        "path": str(cross_report.resolve()),
        "sha256": sha256_file(cross_report),
        "schema_version": CROSS_SURFACE_SCHEMA_VERSION,
        "entry_run_id": cross_run_id,
        "decision": "PASS",
        "row_counts": {"entry": 1, "exit": 1},
    }
    artifact = build_full_input_liveness_artifact(
        dataset_dir=dataset,
        contract_mode=MODEL_NATIVE_CONTRACT_MODE,
        field_order=order,
        stats_by_split=stats,
        manifest_bindings=manifest_bindings,
        scan_proof_by_split=scan_proof,
        multi_tf_liveness_contract=mtf_liveness,
        multi_tf_cache_binding={
            "manifest_path": str(mtf_manifest.resolve()),
            "manifest_sha256": sha256_file(mtf_manifest),
            "cache_identity_sha256": "a" * 64,
        },
        cross_surface_input_overlap=cross_binding,
        created_utc="2026-07-16T00:00:00+00:00",
    )
    path = tmp_path / "ENTRY_FULL_INPUT_LIVENESS_CONTRACT.json"
    path.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return path, artifact, manifest_bindings
