#!/usr/bin/env python3
"""Preflight the dormant smart Entry dataset rebuild.

Report-only. This gate binds the existing smart-layer manifest, Entry feature
AI inventory and active foundation split manifests into one concrete rebuild
contract before any capped dataset mutation can be reviewed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from gx1.runtime.entry_next_edge_legacy_guard import LEGACY_RESEARCH_ACK_ENV, LEGACY_RESEARCH_ACK_VALUE
from gx1.scripts.verify_entry_foundation_state_v1 import FOUNDATION_DATASET_DIR, REPORTS_ROOT


DEFAULT_SMART_REPORT = (
    REPORTS_ROOT
    / "entry_specialist_challenger_extension_manifest_20260630_v1"
    / "ENTRY_SPECIALIST_CHALLENGER_SMART_EXTENSION_REPORT_latest.json"
)
DEFAULT_INVENTORY_REPORT = (
    REPORTS_ROOT / "entry_feature_ai_inventory_20260630_v1" / "ENTRY_FEATURE_AI_INVENTORY_latest.json"
)
DEFAULT_OUT_DIR = REPORTS_ROOT / "entry_smart_seq_rebuild_preflight_20260630_v1"
DEFAULT_PLANNED_DATASET_DIR = (
    FOUNDATION_DATASET_DIR.parent / "v10_dataset_smart_candidate_20260630"
)
FIXED_BASE_COUNTS = {
    "base_signal_features": 41,
    "foundation_sequence_extension_features": 105,
    "chart_geometry_challenger_features": 41,
    "candlestick_challenger_features": 28,
}
EXPECTED_CTX_CAT_NAMES = (
    "session_id",
    "vol_regime_id",
    "atr_bucket",
    "spread_bucket",
    "H4_trend_sign_cat",
)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise RuntimeError(f"missing json: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256_file(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    text = str(value or "")
    return len(text) == 64 and all(ch in "0123456789abcdef" for ch in text.lower())


def _split_manifest_paths(dataset_dir: Path) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for split in ("train", "val", "test"):
        matches = sorted(dataset_dir.glob(f"*_{split}.manifest.json"))
        if len(matches) != 1:
            raise RuntimeError(f"expected exactly one {split} manifest under {dataset_dir}, got {matches}")
        out[split] = matches[0]
    return out


def _artifact_meta(path: Path, *, verify_hash: bool = True) -> dict[str, Any]:
    return {
        "path": str(path),
        "exists": path.exists(),
        "size_bytes": path.stat().st_size if path.exists() else None,
        "sha256": _sha256_file(path) if verify_hash else None,
    }


def _split_contract(path: Path, *, verify_large_hashes: bool) -> dict[str, Any]:
    data = _read_json(path)
    extra = data.get("extra") if isinstance(data.get("extra"), dict) else {}
    base28 = extra.get("base28_manifest") if isinstance(extra.get("base28_manifest"), dict) else {}
    signal_bridge = extra.get("signal_bridge") if isinstance(extra.get("signal_bridge"), dict) else {}
    ctx_contract = extra.get("ctx_contract") if isinstance(extra.get("ctx_contract"), dict) else {}
    extension = (
        signal_bridge.get("seq_structure_extension_v1")
        if isinstance(signal_bridge.get("seq_structure_extension_v1"), dict)
        else {}
    )
    output_data_path = Path(str(data.get("output_data_path") or ""))
    source_parquet = Path(str(base28.get("parquet_path") or extension.get("source_parquet_for_price_features") or ""))
    xgb_bundle = Path(str(extra.get("xgb_bundle") or ""))
    recorded_source_sha = str(base28.get("parquet_sha256") or "")
    observed_source_sha = _sha256_file(source_parquet) if verify_large_hashes and source_parquet.exists() else None
    return {
        "manifest": _artifact_meta(path),
        "output_data_path": str(output_data_path),
        "output_data_exists": output_data_path.exists(),
        "base28_manifest_path": str(base28.get("path") or ""),
        "source_parquet": str(source_parquet),
        "source_parquet_exists": source_parquet.exists(),
        "source_parquet_recorded_sha256": recorded_source_sha,
        "source_parquet_observed_sha256": observed_source_sha,
        "source_parquet_hash_verified": (
            observed_source_sha == recorded_source_sha if observed_source_sha is not None else None
        ),
        "xgb_bundle": str(xgb_bundle),
        "xgb_bundle_exists": xgb_bundle.exists(),
        "xgb_model_exists": (xgb_bundle / "xgb_universal_multihead_v2.joblib").exists(),
        "signal_bridge": {
            "id": signal_bridge.get("id"),
            "seq_input_dim": int(signal_bridge.get("seq_input_dim") or 0),
            "snap_input_dim": int(signal_bridge.get("snap_input_dim") or 0),
            "base_seq_input_dim": int(signal_bridge.get("base_seq_input_dim") or 0),
            "seq_structure_extension_dim": int(signal_bridge.get("seq_structure_extension_dim") or 0),
            "neutral_xgb_bridge": bool(signal_bridge.get("neutral_xgb_bridge")),
            "fields_count": len(signal_bridge.get("fields") or []),
        },
        "seq_structure_extension": {
            "enabled": bool(extension.get("enabled")),
            "feature_count": int(extension.get("feature_count") or 0),
            "manifest_path": extension.get("manifest_path"),
            "manifest_selected_feature_count": int(extension.get("manifest_selected_feature_count") or 0),
            "source_parquet_for_price_features": extension.get("source_parquet_for_price_features"),
        },
        "ctx_contract": {
            "tag": ctx_contract.get("tag"),
            "ctx_cont_dim": int(ctx_contract.get("ctx_cont_dim") or 0),
            "ctx_cat_dim": int(ctx_contract.get("ctx_cat_dim") or 0),
            "ctx_cat_names": list(ctx_contract.get("ctx_cat_names") or []),
            "allow_zero_ctx": bool(ctx_contract.get("allow_zero_ctx")),
        },
    }


def _check(checks: list[dict[str, Any]], name: str, ok: bool, details: Any = None) -> None:
    checks.append({"name": name, "ok": bool(ok), "details": details})


def _command_contract(
    *,
    source_parquet: str,
    xgb_bundle: str,
    smart_manifest: Path,
    planned_dataset_dir: Path,
    manifest_variant: str,
) -> dict[str, Any]:
    output = planned_dataset_dir / f"v10_{manifest_variant}.parquet"
    argv = [
        "scripts/gx1_capped_run.sh",
        "--mem",
        "4G",
        "--swap",
        "1G",
        "--",
        ".venv/bin/python",
        "-m",
        "gx1.scripts.build_entry_v10_ctx_training_dataset_v3",
        "--source-parquet-override",
        source_parquet,
        "--canonical_v2_parquet",
        source_parquet,
        "--xgb_bundle",
        xgb_bundle,
        "--seq-structure-manifest",
        str(smart_manifest),
        "--seq-structure-compute-inline",
        "--time_split",
        "--neutral-xgb-bridge",
        "--hold-bars",
        "3",
        "--output",
        str(output),
    ]
    return {
        "argv": argv,
        "allowed_without_vedtak": False,
        "requires_explicit_rebuild_vedtak": True,
        "requires_clean_git_before_execution": True,
        "uses_legacy_guarded_builder": True,
        "required_environment": {LEGACY_RESEARCH_ACK_ENV: LEGACY_RESEARCH_ACK_VALUE},
        "requires_ram_cap": True,
        "ram_cap_runner": "scripts/gx1_capped_run.sh",
        "memory_max": "4G",
        "swap_max": "1G",
        "num_workers": 0,
        "starts_trainer": False,
        "starts_replay": False,
        "starts_iql_distillation": False,
        "touches_shadow_or_live": False,
        "planned_dataset_dir": str(planned_dataset_dir),
        "planned_output_stem": output.stem,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    smart_report_path = Path(args.smart_report).expanduser().resolve()
    inventory_path = Path(args.inventory_report).expanduser().resolve()
    foundation_dataset_dir = Path(args.foundation_dataset_dir).expanduser().resolve()
    planned_dataset_dir = Path(args.planned_dataset_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    smart_report = _read_json(smart_report_path)
    inventory = _read_json(inventory_path)
    smart_manifest_path = Path(
        str(
            (smart_report.get("manifest") if isinstance(smart_report.get("manifest"), dict) else {}).get(
                "manifest_json_path"
            )
            or smart_report.get("manifest_json_path")
            or ""
        )
    ).expanduser()
    if smart_manifest_path:
        smart_manifest_path = smart_manifest_path.resolve()
    smart_manifest = _read_json(smart_manifest_path) if smart_manifest_path.exists() else {}

    split_contracts = {
        split: _split_contract(path, verify_large_hashes=bool(args.verify_large_input_hashes))
        for split, path in _split_manifest_paths(foundation_dataset_dir).items()
    }

    checks: list[dict[str, Any]] = []
    counts = smart_report.get("counts") if isinstance(smart_report.get("counts"), dict) else {}
    smart_candidate = inventory.get("smart_candidate") if isinstance(inventory.get("smart_candidate"), dict) else {}
    inventory_side_effects = (
        inventory.get("side_effects_started") if isinstance(inventory.get("side_effects_started"), dict) else {}
    )
    smart_selected = [str(x) for x in smart_manifest.get("selected_features", []) if str(x).strip()]
    duplicate_selected = len(smart_selected) - len(set(smart_selected))

    _check(checks, "smart report decision is rebuild-manifest ready", smart_report.get("decision") == "READY_FOR_SMART_CHALLENGER_DATASET_REBUILD_MANIFEST", smart_report_path)
    _check(checks, "smart manifest exists", smart_manifest_path.exists(), smart_manifest_path)
    manifest_variant = str(smart_manifest.get("manifest_variant") or "")
    expected_dim_from_variant = 0
    if manifest_variant.startswith("smart_seq") and manifest_variant.endswith("_candidate"):
        dim_text = manifest_variant.removeprefix("smart_seq").removesuffix("_candidate")
        expected_dim_from_variant = int(dim_text) if dim_text.isdigit() else 0
    smart_layer_features = int(smart_candidate.get("smart_layer_features") or counts.get("smart_candidate_features") or 0)
    expected_combined = (
        FIXED_BASE_COUNTS["foundation_sequence_extension_features"]
        + FIXED_BASE_COUNTS["chart_geometry_challenger_features"]
        + FIXED_BASE_COUNTS["candlestick_challenger_features"]
        + smart_layer_features
    )
    expected_seq_width = FIXED_BASE_COUNTS["base_signal_features"] + expected_combined
    _check(
        checks,
        "smart manifest variant matches expected signal dim",
        manifest_variant == f"smart_seq{expected_seq_width}_candidate" and expected_dim_from_variant == expected_seq_width,
        {"manifest_variant": manifest_variant, "expected_seq_width": expected_seq_width},
    )
    for key, expected in FIXED_BASE_COUNTS.items():
        _check(checks, f"smart fixed count {key} == {expected}", counts.get(key) == expected, {"observed": counts.get(key), "expected": expected})
    _check(checks, "smart report smart feature count matches inventory", counts.get("smart_candidate_features") == smart_layer_features, {"report": counts.get("smart_candidate_features"), "inventory": smart_layer_features})
    _check(checks, "smart combined selected count matches formula", counts.get("combined_selected_features", len(smart_selected)) == expected_combined, {"observed": counts.get("combined_selected_features"), "expected": expected_combined})
    _check(checks, "smart expected seq width matches formula", counts.get("expected_seq_snap_width") == expected_seq_width, {"observed": counts.get("expected_seq_snap_width"), "expected": expected_seq_width})
    _check(checks, "smart duplicate feature count is zero", counts.get("duplicate_feature_count", duplicate_selected) == 0, {"report": counts.get("duplicate_feature_count"), "manifest": duplicate_selected})
    _check(checks, "smart manifest selected feature count matches combined count", len(smart_selected) == expected_combined, {"selected": len(smart_selected), "expected": expected_combined})
    _check(checks, "smart manifest selected names are unique", duplicate_selected == 0, duplicate_selected)
    _check(checks, "smart report keeps training closed", smart_report.get("training_allowed") is False, smart_report.get("training_allowed"))
    _check(
        checks,
        "smart manifest requires rebuild before training",
        smart_manifest.get("dataset_rebuild_required_before_training") is True,
        smart_manifest.get("dataset_rebuild_required_before_training"),
    )

    _check(checks, "inventory decision is design-review ready", inventory.get("decision") == "READY_FOR_SPECIALIST_AI_DESIGN_REVIEW", inventory_path)
    _check(checks, "inventory smart variant matches manifest", smart_candidate.get("manifest_variant") == manifest_variant, smart_candidate)
    _check(checks, "inventory expected signal dim matches smart formula", smart_candidate.get("expected_signal_dim") == expected_seq_width, smart_candidate)
    _check(checks, "inventory smart layer feature count is positive", smart_layer_features > 0, smart_candidate)
    _check(
        checks,
        "inventory required source coverage is complete",
        smart_candidate.get("source_coverage_all_required_available") is True
        and not smart_candidate.get("missing_required_source_field_layers"),
        smart_candidate,
    )
    _check(checks, "inventory keeps training closed", inventory.get("training_allowed") is False, inventory.get("training_allowed"))
    _check(
        checks,
        "inventory side effects closed",
        all(value is False for value in inventory_side_effects.values()),
        inventory_side_effects,
    )

    source_parquets = {row["source_parquet"] for row in split_contracts.values()}
    recorded_source_hashes = {row["source_parquet_recorded_sha256"] for row in split_contracts.values()}
    xgb_bundles = {row["xgb_bundle"] for row in split_contracts.values()}
    _check(
        checks,
        "large source parquet hashes are explicitly verified",
        bool(args.verify_large_input_hashes),
        {"verify_large_input_hashes": bool(args.verify_large_input_hashes)},
    )
    for split, row in split_contracts.items():
        _check(checks, f"{split} active split output exists", row["output_data_exists"], row["output_data_path"])
        _check(checks, f"{split} source parquet exists", row["source_parquet_exists"], row["source_parquet"])
        _check(checks, f"{split} source parquet recorded sha256 present", _is_sha256(row["source_parquet_recorded_sha256"]), row["source_parquet_recorded_sha256"])
        _check(checks, f"{split} source parquet observed hash matches recorded", row["source_parquet_hash_verified"] is True, row)
        _check(checks, f"{split} xgb bundle exists", row["xgb_bundle_exists"], row["xgb_bundle"])
        _check(checks, f"{split} signal bridge is V3", row["signal_bridge"]["id"] == "XGB_SIGNAL_BRIDGE_V3", row["signal_bridge"])
        _check(checks, f"{split} active source seq dim is 146", row["signal_bridge"]["seq_input_dim"] == 146, row["signal_bridge"])
        _check(checks, f"{split} active foundation extension dim is 105", row["signal_bridge"]["seq_structure_extension_dim"] == 105, row["signal_bridge"])
        _check(checks, f"{split} ctx contract is CTX6CAT5", row["ctx_contract"]["tag"] == "CTX6CAT5", row["ctx_contract"])
        _check(checks, f"{split} ctx dims are 142/5", row["ctx_contract"]["ctx_cont_dim"] == 142 and row["ctx_contract"]["ctx_cat_dim"] == 5, row["ctx_contract"])
        _check(checks, f"{split} ctx cat names include spread_bucket", tuple(row["ctx_contract"]["ctx_cat_names"]) == EXPECTED_CTX_CAT_NAMES, row["ctx_contract"])

    _check(checks, "all splits use one source parquet", len(source_parquets) == 1, sorted(source_parquets))
    _check(checks, "all splits use one recorded source sha256", len(recorded_source_hashes) == 1 and all(_is_sha256(x) for x in recorded_source_hashes), sorted(recorded_source_hashes))
    _check(checks, "all splits use one xgb bundle", len(xgb_bundles) == 1, sorted(xgb_bundles))

    source_parquet = next(iter(source_parquets)) if len(source_parquets) == 1 else ""
    xgb_bundle = next(iter(xgb_bundles)) if len(xgb_bundles) == 1 else ""
    command_contract = _command_contract(
        source_parquet=source_parquet,
        xgb_bundle=xgb_bundle,
        smart_manifest=smart_manifest_path,
        planned_dataset_dir=planned_dataset_dir,
        manifest_variant=str(smart_manifest.get("manifest_variant") or "smart_seq_candidate"),
    )
    argv = command_contract["argv"]
    _check(checks, "rebuild command uses RAM cap runner", argv[:6] == ["scripts/gx1_capped_run.sh", "--mem", "4G", "--swap", "1G", "--"], argv[:8])
    _check(checks, "rebuild command uses source parquet override", "--source-parquet-override" in argv and source_parquet in argv, argv)
    _check(checks, "rebuild command pins canonical_v2 parquet", "--canonical_v2_parquet" in argv and source_parquet in argv, argv)
    _check(checks, "rebuild command computes smart extension inline", "--seq-structure-compute-inline" in argv, argv)
    _check(checks, "rebuild command keeps neutral xgb bridge", "--neutral-xgb-bridge" in argv, argv)
    _check(
        checks,
        "rebuild command declares legacy builder ack env",
        command_contract.get("required_environment", {}).get(LEGACY_RESEARCH_ACK_ENV)
        == LEGACY_RESEARCH_ACK_VALUE,
        command_contract.get("required_environment"),
    )
    _check(checks, "rebuild command does not start trainer", command_contract["starts_trainer"] is False, command_contract)

    failed = [row for row in checks if not row["ok"]]
    decision = (
        "READY_FOR_SMART_REBUILD_VEDTAK_REVIEW"
        if not failed
        else "BLOCKED_SMART_REBUILD_PREFLIGHT"
    )
    created = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report = {
        "schema_version": "entry_smart_seq_rebuild_preflight_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "decision": decision,
        "report_only": True,
        "training_allowed": False,
        "dataset_rebuild_allowed_without_vedtak": False,
        "dataset_rebuild_allowed_after_explicit_vedtak_review": not failed,
        "side_effects_started": {
            "dataset_rebuild": False,
            "training": False,
            "replay": False,
            "iql_distillation": False,
            "shadow": False,
            "live": False,
        },
        "counts": {
            "base_signal_features": 41,
            "foundation_sequence_extension_features": 105,
            "chart_geometry_challenger_features": 41,
            "candlestick_challenger_features": 28,
            "smart_layer_features": smart_layer_features,
            "combined_extension_features": expected_combined,
            "expected_seq_snap_width": expected_seq_width,
            "manifest_variant": manifest_variant,
            "source_active_seq_snap_width": 146,
        },
        "inputs": {
            "smart_report": _artifact_meta(smart_report_path),
            "smart_manifest": _artifact_meta(smart_manifest_path),
            "inventory_report": _artifact_meta(inventory_path),
            "foundation_dataset_dir": str(foundation_dataset_dir),
        },
        "split_contracts": split_contracts,
        "rebuild_command_contract": command_contract,
        "checks": checks,
        "failures": failed,
        "next_required_gate": (
            "explicit smart rebuild vedtak, run the capped dataset rebuild, then feature audit -> "
            "specialist audit -> liveness/non-collapse proof -> train-readiness; no smoke/candidate/replay/IQL "
            "from this preflight alone"
        ),
    }

    json_path = out_dir / f"ENTRY_SMART_REBUILD_PREFLIGHT_{created}.json"
    md_path = out_dir / f"ENTRY_SMART_REBUILD_PREFLIGHT_{created}.md"
    latest_json = out_dir / "ENTRY_SMART_REBUILD_PREFLIGHT_latest.json"
    latest_md = out_dir / "ENTRY_SMART_REBUILD_PREFLIGHT_latest.md"
    json_text = json.dumps(report, indent=2, sort_keys=True, default=_json_default) + "\n"
    json_path.write_text(json_text, encoding="utf-8")
    latest_json.write_text(json_text, encoding="utf-8")
    md = [
        "# Entry Smart Rebuild Preflight",
        "",
        f"- decision: `{decision}`",
        "- report_only: `true`",
        "- training_allowed: `false`",
        "- dataset_rebuild_allowed_without_vedtak: `false`",
        f"- expected_seq_snap_width: `{report['counts']['expected_seq_snap_width']}`",
        f"- smart_layer_features: `{report['counts']['smart_layer_features']}`",
        f"- failures: `{len(failed)}`",
        "",
        "Next: explicit smart rebuild vedtak, capped dataset rebuild, then audits and train-readiness.",
    ]
    md_text = "\n".join(md) + "\n"
    md_path.write_text(md_text, encoding="utf-8")
    latest_md.write_text(md_text, encoding="utf-8")
    report["json_path"] = str(json_path)
    report["md_path"] = str(md_path)
    latest_json.write_text(json.dumps(report, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description="Report-only smart sequence rebuild preflight.")
    ap.add_argument("--smart-report", default=str(DEFAULT_SMART_REPORT))
    ap.add_argument("--inventory-report", default=str(DEFAULT_INVENTORY_REPORT))
    ap.add_argument("--foundation-dataset-dir", default=str(FOUNDATION_DATASET_DIR))
    ap.add_argument("--planned-dataset-dir", default=str(DEFAULT_PLANNED_DATASET_DIR))
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--verify-large-input-hashes", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--no-fail-on-audit-fail", action="store_true")
    args = ap.parse_args()
    report = run(args)
    if not args.quiet:
        print(json.dumps(report, indent=2, sort_keys=True, default=_json_default))
    if report["decision"] != "READY_FOR_SMART_REBUILD_VEDTAK_REVIEW" and not args.no_fail_on_audit_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
