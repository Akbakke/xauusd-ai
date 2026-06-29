#!/usr/bin/env python3
"""
Materialize promoted chart-structure features and a stable feature-order manifest.

The parquet emitted here is sample-aligned to the current V10 dataset rows. It is
useful for inspection, ranking, and feature order/provenance. For true temporal
Transformer training, build_entry_v10_ctx_training_dataset_v3.py should consume
the manifest with --seq-structure-compute-inline so the same features are
computed for every raw per-bar row in the 96-bar history.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from gx1.contracts.signal_bridge_v3 import ORDERED_SEQ_FIELDS_V3
from gx1.features.entry_foundation_structure_v1 import (
    FOUNDATION_STRUCTURE_FEATURE_NAMES,
    FOUNDATION_STRUCTURE_FEATURE_VERSION,
)
from gx1.scripts.analyze_entry_feature_interaction_tail_v1 import _build_all_features
from gx1.scripts.experiment_entry_chart_structure_ablation_v1 import (
    DEFAULT_DATASET_DIR,
    DEFAULT_SOURCE_PARQUET,
)


DEFAULT_PROMOTION_DIR = Path("/home/andre2/GX1_DATA/reports/sequence_feature_promotion_20260628_v1")
DEFAULT_MANIFEST = DEFAULT_PROMOTION_DIR / "sequence_feature_promotion_manifest.json"
DEFAULT_OUT_DIR = Path("/home/andre2/GX1_DATA/reports/sequence_structure_feature_layer_20260628_v1")


def _parse_csv(raw: str) -> list[str]:
    return [p.strip() for p in str(raw or "").split(",") if p.strip()]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_manifest(path: Path, top_n: int) -> list[str]:
    data = json.loads(path.read_text())
    features: list[str] = []
    for row in data.get("features", []):
        name = str(row.get("feature", "")).strip()
        if name and name not in features:
            features.append(name)
        if len(features) >= int(top_n):
            break
    return features


def _dedupe_preserve_order(features: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for raw in features:
        name = str(raw).strip()
        if not name or name in seen:
            continue
        seen.add(name)
        out.append(name)
    return out


def _requested_features(
    promotion_features: list[str],
    *,
    include_foundation_structure_features: bool,
) -> tuple[list[str], dict[str, Any]]:
    foundation_features = (
        list(FOUNDATION_STRUCTURE_FEATURE_NAMES)
        if include_foundation_structure_features
        else []
    )
    requested = _dedupe_preserve_order(list(promotion_features) + foundation_features)
    return requested, {
        "promotion_requested_feature_count": int(len(_dedupe_preserve_order(promotion_features))),
        "foundation_structure_feature_version": FOUNDATION_STRUCTURE_FEATURE_VERSION,
        "foundation_structure_features_required": bool(include_foundation_structure_features),
        "foundation_structure_feature_count": int(len(foundation_features)),
        "foundation_structure_features": foundation_features,
    }


def _clean_matrix(x: np.ndarray) -> np.ndarray:
    out = np.asarray(x, dtype=np.float32)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)


def run(args: argparse.Namespace) -> dict[str, Any]:
    manifest_path = Path(args.manifest).expanduser().resolve()
    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    source_parquet = Path(args.source_parquet).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    promotion_requested = _read_manifest(manifest_path, int(args.top_n))
    requested, request_meta = _requested_features(
        promotion_requested,
        include_foundation_structure_features=bool(args.include_foundation_structure_features),
    )
    foundation_required = list(request_meta["foundation_structure_features"])
    foundation_missing = [name for name in foundation_required if name not in requested]

    if bool(args.manifest_only):
        if foundation_missing:
            raise SystemExit(
                "foundation structure features missing from manifest-only request: "
                f"{foundation_missing[:30]} total={len(foundation_missing)}"
            )
        selected = list(requested)
        layer_manifest = {
            "schema_version": "sequence_structure_feature_layer_v1",
            "purpose": "manifest-only chart-structure feature order/contract for sequential Transformer/IQL training",
            "xgboost_primary_candidate": False,
            "manifest_only": True,
            "feature_availability_validated": False,
            "promotion_manifest": str(manifest_path),
            "dataset_dir": str(dataset_dir),
            "source_parquet": str(source_parquet),
            "data_splits": _parse_csv(args.data_splits),
            "include_price_ema_features": bool(args.include_price_ema_features),
            **request_meta,
            "n_rows": None,
            "requested_feature_count": int(len(requested)),
            "selected_feature_count": int(len(selected)),
            "missing_feature_count": None,
            "foundation_structure_missing_feature_count": int(len(foundation_missing)),
            "foundation_structure_missing_features": foundation_missing,
            "foundation_structure_all_required_selected": not foundation_missing,
            "selected_features": selected,
            "missing_features": [],
            "existing_seq_overlap": [name for name in selected if name in set(ORDERED_SEQ_FIELDS_V3)],
            "base_transformer_seq_dim_v3": int(len(ORDERED_SEQ_FIELDS_V3)),
            "proposed_seq_extension_dim": int(len(selected)),
            "proposed_seq_input_dim_with_extension": int(len(ORDERED_SEQ_FIELDS_V3) + len(selected)),
            "parquet_path": None,
            "parquet_sha256": None,
            "true_temporal_training_note": (
                "Manifest-only contract. Use build_entry_v10_ctx_training_dataset_v3.py "
                "--seq-structure-manifest <this manifest> --seq-structure-compute-inline. "
                "Run audit_entry_foundation_features_v1.py on the rebuilt dataset before training."
            ),
        }
        manifest_out = out_dir / "sequence_structure_feature_layer_manifest.json"
        manifest_out.write_text(json.dumps(layer_manifest, indent=2), encoding="utf-8")
        contract_path = out_dir / "ORDERED_SEQ_STRUCTURE_EXTENSION_V1.txt"
        contract_path.write_text("\n".join(selected) + "\n", encoding="utf-8")
        readme = [
            "# Sequence Structure Feature Layer",
            "",
            "This is a manifest-only feature-order contract for the next sequential rebuild.",
            "",
            f"- Selected features: {len(selected)}",
            f"- Foundation structure features: {len(foundation_required)} required / {len(foundation_missing)} missing",
            f"- Proposed sequence input dim when computed inline: {len(ORDERED_SEQ_FIELDS_V3)} + {len(selected)} = {len(ORDERED_SEQ_FIELDS_V3) + len(selected)}",
            "",
            "No sample-aligned parquet is emitted in manifest-only mode.",
            "Use this manifest with `--seq-structure-compute-inline`, then run the feature foundation audit.",
            "",
        ]
        (out_dir / "README.md").write_text("\n".join(readme), encoding="utf-8")
        summary = {
            "out_dir": str(out_dir),
            "manifest_path": str(manifest_out),
            "contract_path": str(contract_path),
            "manifest_only": True,
            "feature_availability_validated": False,
            "selected_feature_count": int(len(selected)),
            "foundation_structure_feature_count": int(len(foundation_required)),
            "foundation_structure_missing_feature_count": int(len(foundation_missing)),
            "proposed_seq_input_dim_with_extension": int(len(ORDERED_SEQ_FIELDS_V3) + len(selected)),
            "selected_features_head": selected[:20],
            "foundation_structure_missing_features": foundation_missing,
        }
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2))
        return summary

    x, feature_df, names = _build_all_features(
        dataset_dir,
        source_parquet,
        _parse_csv(args.data_splits),
        include_price_ema_features=bool(args.include_price_ema_features),
    )
    name_to_idx = {name: i for i, name in enumerate(names)}
    selected = [name for name in requested if name in name_to_idx]
    missing = [name for name in requested if name not in name_to_idx]
    foundation_missing = [name for name in foundation_required if name not in selected]
    existing_seq_overlap = [name for name in selected if name in set(ORDERED_SEQ_FIELDS_V3)]

    if not selected:
        raise SystemExit("no manifest features were available in the built feature matrix")
    if foundation_missing:
        raise SystemExit(
            "foundation structure features missing from generated matrix: "
            f"{foundation_missing[:30]} total={len(foundation_missing)}"
        )

    indices = [name_to_idx[name] for name in selected]
    feature_mat = _clean_matrix(x[:, indices])

    out = feature_df[["time", "session", "source_split"]].copy()
    out["time"] = pd.to_datetime(out["time"], utc=True)
    for j, name in enumerate(selected):
        out[name] = feature_mat[:, j].astype(np.float32, copy=False)

    parquet_path = out_dir / "sequence_structure_features.parquet"
    out.to_parquet(parquet_path, index=False)

    layer_manifest = {
        "schema_version": "sequence_structure_feature_layer_v1",
        "purpose": "sample-aligned chart-structure feature order/provenance for sequential Transformer/IQL training",
        "xgboost_primary_candidate": False,
        "promotion_manifest": str(manifest_path),
        "dataset_dir": str(dataset_dir),
        "source_parquet": str(source_parquet),
        "data_splits": _parse_csv(args.data_splits),
        "include_price_ema_features": bool(args.include_price_ema_features),
        **request_meta,
        "n_rows": int(len(out)),
        "requested_feature_count": int(len(requested)),
        "selected_feature_count": int(len(selected)),
        "missing_feature_count": int(len(missing)),
        "foundation_structure_missing_feature_count": int(len(foundation_missing)),
        "foundation_structure_missing_features": foundation_missing,
        "foundation_structure_all_required_selected": not foundation_missing,
        "selected_features": selected,
        "missing_features": missing,
        "existing_seq_overlap": existing_seq_overlap,
        "base_transformer_seq_dim_v3": int(len(ORDERED_SEQ_FIELDS_V3)),
        "proposed_seq_extension_dim": int(len(selected)),
        "proposed_seq_input_dim_with_extension": int(len(ORDERED_SEQ_FIELDS_V3) + len(selected)),
        "parquet_path": str(parquet_path),
        "parquet_sha256": _sha256(parquet_path),
        "true_temporal_training_note": (
            "Do not join this sample-aligned parquet directly into raw seq history. "
            "Use build_entry_v10_ctx_training_dataset_v3.py --seq-structure-manifest "
            "<this manifest> --seq-structure-compute-inline."
        ),
    }
    manifest_out = out_dir / "sequence_structure_feature_layer_manifest.json"
    manifest_out.write_text(json.dumps(layer_manifest, indent=2), encoding="utf-8")

    contract_path = out_dir / "ORDERED_SEQ_STRUCTURE_EXTENSION_V1.txt"
    contract_path.write_text("\n".join(selected) + "\n", encoding="utf-8")

    readme = [
        "# Sequence Structure Feature Layer",
        "",
        "This is a sample-aligned feature-order/provenance layer for the next sequential rebuild.",
        "",
        f"- Rows: {len(out)}",
        f"- Selected features: {len(selected)}",
        f"- Missing requested features: {len(missing)}",
        f"- Foundation structure features: {len(foundation_required)} required / {len(foundation_missing)} missing",
        f"- Proposed sequence input dim when computed inline: {len(ORDERED_SEQ_FIELDS_V3)} + {len(selected)} = {len(ORDERED_SEQ_FIELDS_V3) + len(selected)}",
        f"- Parquet: `{parquet_path}`",
        "",
        "Use the manifest with `--seq-structure-compute-inline` for true temporal seq history.",
        "Do not treat it as a promoted live policy by itself.",
        "",
    ]
    (out_dir / "README.md").write_text("\n".join(readme), encoding="utf-8")

    summary = {
        "out_dir": str(out_dir),
        "parquet_path": str(parquet_path),
        "manifest_path": str(manifest_out),
        "contract_path": str(contract_path),
        "n_rows": int(len(out)),
        "selected_feature_count": int(len(selected)),
        "missing_feature_count": int(len(missing)),
        "foundation_structure_feature_count": int(len(foundation_required)),
        "foundation_structure_missing_feature_count": int(len(foundation_missing)),
        "proposed_seq_input_dim_with_extension": int(len(ORDERED_SEQ_FIELDS_V3) + len(selected)),
        "selected_features_head": selected[:20],
        "missing_features": missing,
        "foundation_structure_missing_features": foundation_missing,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    ap.add_argument("--dataset-dir", default=str(DEFAULT_DATASET_DIR))
    ap.add_argument("--source-parquet", default=str(DEFAULT_SOURCE_PARQUET))
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--data-splits", default="train,val,test")
    ap.add_argument("--top-n", type=int, default=48)
    ap.add_argument("--include-price-ema-features", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--include-foundation-structure-features", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--manifest-only", action="store_true")
    return ap


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
