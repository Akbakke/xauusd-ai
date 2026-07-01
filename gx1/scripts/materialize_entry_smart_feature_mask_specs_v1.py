#!/usr/bin/env python3
"""Materialize post-hoc smart Entry feature-mask specs.

The specs are consumed by `evaluate_entry_candidate_selective_edge_v1
--feature-mask-json`. They describe explicit seq/snap feature names and
indices to zero during forward inference. This is a report-only ablation path:
it does not train, replay, distill, promote, shadow, or touch live paths.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from gx1.features.entry_candlestick_patterns_v1 import build_entry_candlestick_pattern_layer
from gx1.features.entry_chart_geometry_v1 import CHART_GEOMETRY_SOURCE_FIELDS, build_entry_chart_geometry_layer
from gx1.scripts.evaluate_entry_candidate_selective_edge_v1 import FEATURE_MASK_SCHEMA_VERSION
from gx1.scripts.materialize_entry_smart_ablation_replay_plan_gate_v1 import build_required_ablation_plan
from gx1.scripts.verify_entry_foundation_state_v1 import REPORTS_ROOT


SMART_DATASET_DIR = (
    Path("/home/andre2/GX1_DATA/runs/FASE2B_REGIME_V4_20260605")
    / "v10_6yr_rebuild_20260628_foundation_seq146/v10_dataset_smart_candidate_20260630"
)
DEFAULT_PLAN_JSON = (
    REPORTS_ROOT
    / "entry_smart_ablation_replay_plan_gate_20260630_v1/ENTRY_SMART_ABLATION_REPLAY_PLAN_GATE_latest.json"
)
DEFAULT_OUT_DIR = REPORTS_ROOT / "entry_smart_feature_mask_specs_20260701_v1"


def _json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


def _sha256_file(path: Path) -> str:
    if not path.exists() or not path.is_file():
        return ""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _signal_bridge(dataset_dir: Path) -> tuple[dict[str, Any], Path, list[dict[str, Any]]]:
    manifests = sorted(dataset_dir.glob("*_test.manifest.json")) or sorted(dataset_dir.glob("*.manifest.json"))
    if not manifests:
        raise RuntimeError(f"no dataset manifests found in {dataset_dir}")
    manifest_path = manifests[0]
    data = _read_json(manifest_path)
    extra = data.get("extra") if isinstance(data.get("extra"), dict) else {}
    signal_bridge = extra.get("signal_bridge") if isinstance(extra.get("signal_bridge"), dict) else {}
    fields = [str(value) for value in signal_bridge.get("fields", []) if str(value)]
    if not fields:
        raise RuntimeError(f"dataset manifest does not expose extra.signal_bridge.fields: {manifest_path}")

    split_reviews: list[dict[str, Any]] = []
    for split in ("train", "val", "test"):
        split_manifests = sorted(dataset_dir.glob(f"*_{split}.manifest.json"))
        if not split_manifests:
            split_reviews.append({"split": split, "manifest_path": "", "field_count": 0, "matches_reference": False})
            continue
        split_path = split_manifests[0]
        split_data = _read_json(split_path)
        split_extra = split_data.get("extra") if isinstance(split_data.get("extra"), dict) else {}
        split_bridge = split_extra.get("signal_bridge") if isinstance(split_extra.get("signal_bridge"), dict) else {}
        split_fields = [str(value) for value in split_bridge.get("fields", []) if str(value)]
        split_reviews.append(
            {
                "split": split,
                "manifest_path": str(split_path),
                "manifest_sha256": _sha256_file(split_path),
                "field_count": int(len(split_fields)),
                "matches_reference": split_fields == fields,
            }
        )
    return signal_bridge, manifest_path, split_reviews


def _required_arms(plan: dict[str, Any]) -> list[dict[str, Any]]:
    rows = plan.get("required_ablations")
    if isinstance(rows, list):
        return [row for row in rows if isinstance(row, dict)]
    required_plan = plan.get("required_ablation_plan") if isinstance(plan.get("required_ablation_plan"), dict) else {}
    rows = required_plan.get("required_ablations")
    if isinstance(rows, list):
        return [row for row in rows if isinstance(row, dict)]
    return build_required_ablation_plan().get("required_ablations", [])


def _builder_feature_names(label: str) -> list[str]:
    if label == "chart_geometry_smart2_layer":
        source_names = list(CHART_GEOMETRY_SOURCE_FIELDS)
        x = np.zeros((8, len(source_names)), dtype=np.float32)
        _, names = build_entry_chart_geometry_layer(x, source_names)
        return [str(name) for name in names[41:]]
    if label == "price_action_candle_smart3_layer":
        frame = {
            "open": np.linspace(100.0, 101.0, 8, dtype=np.float64),
            "high": np.linspace(100.5, 101.5, 8, dtype=np.float64),
            "low": np.linspace(99.5, 100.5, 8, dtype=np.float64),
            "close": np.linspace(100.1, 101.1, 8, dtype=np.float64),
        }
        _, names = build_entry_candlestick_pattern_layer(frame)
        return [str(name) for name in names[28:]]
    return []


def _write_spec(
    *,
    out_dir: Path,
    arm: dict[str, Any],
    zero_feature_names: list[str],
    fields: list[str],
    dataset_manifest_path: Path,
    plan_json: Path,
    overwrite: bool,
) -> dict[str, Any]:
    ablation_id = str(arm.get("ablation_id") or "")
    path = out_dir / f"{ablation_id}.feature_mask.json"
    if path.exists() and not overwrite:
        existing = _read_json(path)
        missing = existing.get("missing_feature_names")
        zero_features = existing.get("zero_feature_names")
        return {
            "ablation_id": ablation_id,
            "path": str(path),
            "sha256": _sha256_file(path),
            "zero_feature_count": int(len(zero_features)) if isinstance(zero_features, list) else int(existing.get("zero_feature_count") or 0),
            "missing_feature_count": int(len(missing)) if isinstance(missing, list) else 0,
            "expected_seq_snap_width": int(arm.get("expected_seq_snap_width") or 0),
            "mask_ablation_kind": str(existing.get("mask_ablation_kind") or ""),
            "preserved_existing_file": True,
        }
    field_to_idx = {name: idx for idx, name in enumerate(fields)}
    missing = [name for name in zero_feature_names if name not in field_to_idx]
    indices = sorted(field_to_idx[name] for name in zero_feature_names if name in field_to_idx)
    spec = {
        "schema_version": FEATURE_MASK_SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "ablation_id": ablation_id,
        "model_name": ablation_id,
        "mask_mode": "zero_seq_snap_features",
        "mask_ablation_kind": "posthoc_zero_mask_not_retrained_width",
        "zero_value": 0.0,
        "zero_indices": indices,
        "zero_feature_names": [fields[idx] for idx in indices],
        "zero_feature_count": int(len(indices)),
        "missing_feature_names": missing,
        "signal_field_count": int(len(fields)),
        "dataset_manifest": str(dataset_manifest_path),
        "dataset_manifest_sha256": _sha256_file(dataset_manifest_path),
        "plan_json": str(plan_json),
        "plan_json_sha256": _sha256_file(plan_json),
        "plan_arm": arm,
        "training_started": False,
        "replay_started": False,
        "iql_started": False,
        "shadow_live_promotion_allowed": False,
    }
    path.write_text(json.dumps(spec, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")
    return {
        "ablation_id": ablation_id,
        "path": str(path),
        "sha256": _sha256_file(path),
        "zero_feature_count": int(len(indices)),
        "missing_feature_count": int(len(missing)),
        "expected_seq_snap_width": int(arm.get("expected_seq_snap_width") or 0),
        "mask_ablation_kind": spec["mask_ablation_kind"],
        "preserved_existing_file": False,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    plan_json = Path(args.plan_json).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    plan = _read_json(plan_json) if plan_json.exists() else {"required_ablations": build_required_ablation_plan().get("required_ablations", [])}
    signal_bridge, dataset_manifest_path, split_reviews = _signal_bridge(dataset_dir)
    fields = [str(value) for value in signal_bridge.get("fields", []) if str(value)]
    extension = signal_bridge.get("seq_structure_extension_v1") if isinstance(signal_bridge.get("seq_structure_extension_v1"), dict) else {}
    smart_layers = [row for row in extension.get("smart_generated_layers", []) if isinstance(row, dict)]
    smart_layers_by_label = {str(row.get("label") or ""): row for row in smart_layers}
    smart_layers_by_label["chart_geometry_smart2_layer"] = {
        "label": "chart_geometry_smart2_layer",
        "features": _builder_feature_names("chart_geometry_smart2_layer"),
    }
    smart_layers_by_label["price_action_candle_smart3_layer"] = {
        "label": "price_action_candle_smart3_layer",
        "features": _builder_feature_names("price_action_candle_smart3_layer"),
    }
    smart_feature_names = [
        str(name)
        for layer in smart_layers_by_label.values()
        for name in layer.get("features", [])
        if str(name)
    ]
    arms = _required_arms(plan)
    spec_rows: list[dict[str, Any]] = []
    failures: list[str] = []
    if not all(bool(row.get("matches_reference")) for row in split_reviews):
        failures.append("train/val/test signal_bridge.fields do not all match the reference manifest")

    for arm in arms:
        ablation_id = str(arm.get("ablation_id") or "")
        if ablation_id in {"with_old_plus_smart", "no_xgb"}:
            continue
        if ablation_id == "old_only":
            zero_names = smart_feature_names
        elif ablation_id == "smart_only":
            zero_names = fields[41:215]
        elif str(arm.get("ablation_type") or "") == "drop_smart_family":
            dropped = arm.get("dropped_smart_family") if isinstance(arm.get("dropped_smart_family"), dict) else {}
            family_label = str(dropped.get("family_label") or "")
            layer = smart_layers_by_label.get(family_label, {})
            zero_names = [str(name) for name in layer.get("features", []) if str(name)]
        else:
            continue
        row = _write_spec(
            out_dir=out_dir,
            arm=arm,
            zero_feature_names=zero_names,
            fields=fields,
            dataset_manifest_path=dataset_manifest_path,
            plan_json=plan_json,
            overwrite=bool(args.overwrite),
        )
        spec_rows.append(row)
        if row["missing_feature_count"]:
            failures.append(f"{ablation_id}: missing {row['missing_feature_count']} requested feature names")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    json_path = out_dir / f"ENTRY_SMART_FEATURE_MASK_SPECS_{timestamp}.json"
    report = {
        "schema_version": "entry_smart_feature_mask_specs_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "decision": "PASS" if not failures else "FAIL",
        "dataset_dir": str(dataset_dir),
        "dataset_manifest": str(dataset_manifest_path),
        "dataset_manifest_sha256": _sha256_file(dataset_manifest_path),
        "split_signal_field_reviews": split_reviews,
        "plan_json": str(plan_json),
        "plan_json_sha256": _sha256_file(plan_json),
        "signal_field_count": int(len(fields)),
        "spec_count": int(len(spec_rows)),
        "specs": spec_rows,
        "training_started": False,
        "replay_started": False,
        "iql_started": False,
        "shadow_live_promotion_allowed": False,
        "failures": failures,
        "json_path": str(json_path),
    }
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")
    (out_dir / "ENTRY_SMART_FEATURE_MASK_SPECS_latest.json").write_text(
        json_path.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    if not args.quiet:
        print(json.dumps(report, indent=2, sort_keys=True, default=_json_default))
    if args.fail_on_audit_fail and failures:
        raise SystemExit(2)
    return report


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset-dir", default=str(SMART_DATASET_DIR))
    ap.add_argument("--plan-json", default=str(DEFAULT_PLAN_JSON))
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--fail-on-audit-fail", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--quiet", action="store_true")
    return ap


def main() -> int:
    run(build_parser().parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
