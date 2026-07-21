from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

from gx1.contracts.entry_full_input_liveness_v1 import (
    build_full_input_liveness_artifact,
    sha256_file,
)
from gx1.contracts.entry_model_native_signal_v1 import (
    MODEL_NATIVE_CONTRACT_MODE,
    MODEL_NATIVE_SIGNAL_DIM,
)


def full_input_field_order() -> dict[str, list[str]]:
    signal = [
        "smc_choch",
        "candle.pattern_outside_after_inside_bull_breakout_score",
        "candle.pattern_outside_after_inside_bear_breakout_score",
        "ctx_cont.d1_atr14_canon_v2",
        "ctx_cont._v1h4_atr",
        "chart.m5_ema50_200_cross_up",
        "chart.m5_ema50_200_cross_down",
    ]
    signal.extend(
        f"signal_feature_{idx:03d}"
        for idx in range(MODEL_NATIVE_SIGNAL_DIM - len(signal))
    )
    ctx_cont = [
        "d1_atr14_canon_v2",
        "_v1h4_atr",
        "d1_regime_changed_flag_v3",
    ]
    ctx_cont.extend(f"ctx_cont_feature_{idx:03d}" for idx in range(142 - len(ctx_cont)))
    ctx_cat = [f"ctx_cat_feature_{idx}" for idx in range(5)]
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
                surface_rows[field] = {
                    "row_count": rows,
                    "finite_count": rows,
                    "nonfinite_count": 0,
                    "mean": 1.0 + idx * 0.001,
                    "std": 0.5,
                    "min": 0.5 + idx * 0.001,
                    "max": 1.5 + idx * 0.001,
                    "value_range": 1.0,
                    "active_count": rows,
                    "active_rate": 1.0,
                }
            split_rows[surface] = surface_rows
        result[split] = split_rows
    return result


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
        manifest_bindings[split] = {"path": str(manifest), "sha256": sha256_file(manifest)}
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
    artifact = build_full_input_liveness_artifact(
        dataset_dir=dataset,
        contract_mode=MODEL_NATIVE_CONTRACT_MODE,
        field_order=order,
        stats_by_split=stats,
        manifest_bindings=manifest_bindings,
        scan_proof_by_split=scan_proof,
        created_utc="2026-07-16T00:00:00+00:00",
    )
    path = tmp_path / "ENTRY_FULL_INPUT_LIVENESS_CONTRACT.json"
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path, artifact, manifest_bindings
