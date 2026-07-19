#!/usr/bin/env python3
"""Cheap report-only pre-rebuild diagnostic for model-native abstention.

Only explicit JSON/JSONL metadata and their caller-supplied hashes are read.
The report cannot authorize direction, rebuild, training, launch, or live use.
Missing historical benchmark bytes produce an immutable BLOCK report; label
balance remains a clearly marked probe diagnostic, never substitute evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from gx1.contracts.entry_model_native_abstention_probe_v1 import (
    HISTORICAL_ROLE,
    MODEL_NATIVE_ROLE,
    AbstentionProbeEvidenceError,
    compare_selection_evidence,
    sha256_file,
    validate_selection_evidence,
)
from gx1.contracts.immutable_event_authority_v1 import write_immutable_json_event
from gx1.scripts import materialize_entry_model_native_seq513_smoke_manifest_v1 as smoke_recipe_owner


SCHEMA_VERSION = "entry_model_native_pre_rebuild_abstention_probe_v1"
EVENT_PREFIX = "ENTRY_MODEL_NATIVE_PRE_REBUILD_ABSTENTION_PROBE"
PASS_DECISION = "PASS_ABSTENTION_EMPIRICAL_COMPARISON_ONLY"
BLOCK_DECISION = "BLOCK_ABSTENTION_EMPIRICAL_GATE"
HISTORICAL_MANIFEST_SCHEMA = "entry_smart_seq520_smoke_split_manifest_v1"
HISTORICAL_MANIFEST_VARIANT = "smart_seq520_candidate"
HISTORICAL_SIGNAL_DIM = 520
GX1_DATA_ROOT = Path("/home/andre2/GX1_DATA")
CLASS_NAMES = {"0": "LONG", "1": "SHORT", "2": "FLAT"}
REQUIRED_POSITIVE_ABSTENTION_RECIPE_FIELDS = (
    "direction_vs_flat_margin_weight",
    "direction_utility_margin_weight",
    "direction_utility_trade_conviction_weight",
    "direction_utility_triad_ce_weight",
    "hier_flat_logit_margin_weight",
    "hier_slice_flat_logit_margin_weight",
    "direction_flat_starvation_weight",
    "hier_trade_weight",
)


def _strict_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON token {token}")
            ),
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise RuntimeError(f"invalid strict JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON root must be an object: {path}")
    return value


def _bound_json(path_value: str, expected_sha256: str, *, context: str) -> tuple[Path, dict[str, Any]]:
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        raise RuntimeError(f"{context}: explicit path must be absolute")
    if any("latest" in part.lower() for part in path.parts):
        raise RuntimeError(f"{context}: mutable latest path is forbidden")
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"{context}: regular input file is missing")
    expected = str(expected_sha256 or "").lower()
    actual = sha256_file(path)
    if len(expected) != 64 or actual != expected:
        raise RuntimeError(
            f"{context}: sha256 mismatch expected={expected!r} actual={actual}"
        )
    resolved = path.resolve()
    return resolved, _strict_json(resolved)


def _manifest_probe(
    path_value: str,
    expected_sha256: str,
    *,
    expected_split: str,
) -> dict[str, Any]:
    path, payload = _bound_json(
        path_value,
        expected_sha256,
        context=f"historical {expected_split} smoke manifest",
    )
    if payload.get("schema_version") != HISTORICAL_MANIFEST_SCHEMA:
        raise RuntimeError(f"historical {expected_split} manifest schema is invalid")
    if payload.get("manifest_variant") != HISTORICAL_MANIFEST_VARIANT:
        raise RuntimeError(f"historical {expected_split} manifest variant is invalid")
    if payload.get("expected_seq_snap_width") != HISTORICAL_SIGNAL_DIM:
        raise RuntimeError(f"historical {expected_split} manifest width is invalid")
    summary = payload.get("foundation_smoke_dataset_v1")
    if not isinstance(summary, dict) or summary.get("split") != expected_split:
        raise RuntimeError(f"historical {expected_split} label summary is missing")
    counts_raw = summary.get("label_counts")
    if not isinstance(counts_raw, dict) or set(counts_raw) != set(CLASS_NAMES):
        raise RuntimeError(f"historical {expected_split} label counts are not exact")
    counts: dict[str, int] = {}
    for key in CLASS_NAMES:
        value = counts_raw[key]
        if type(value) is not int or value <= 0:
            raise RuntimeError(f"historical {expected_split} label count {key} is invalid")
        counts[key] = value
    sample_rows = summary.get("sample_rows")
    if type(sample_rows) is not int or sample_rows != sum(counts.values()):
        raise RuntimeError(f"historical {expected_split} sample row count is invalid")
    return {
        "path": str(path),
        "sha256": expected_sha256.lower(),
        "schema_version": HISTORICAL_MANIFEST_SCHEMA,
        "manifest_variant": HISTORICAL_MANIFEST_VARIANT,
        "expected_signal_dim": HISTORICAL_SIGNAL_DIM,
        "split": expected_split,
        "sample_rows": sample_rows,
        "label_counts": {
            CLASS_NAMES[key]: counts[key] for key in CLASS_NAMES
        },
        "label_rates": {
            CLASS_NAMES[key]: counts[key] / sample_rows for key in CLASS_NAMES
        },
        "probe_only": True,
        "accepted_model_native_dataset": False,
        "parquet_rows_read": 0,
    }


def _paired_optional(
    path_value: str | None,
    sha_value: str | None,
    *,
    label: str,
) -> tuple[Path, str] | None:
    if bool(path_value) != bool(sha_value):
        raise RuntimeError(f"{label}: path and sha256 must be supplied together")
    if not path_value:
        return None
    return Path(path_value), str(sha_value)


def _artifact_registry_probe(path_value: str, sha_value: str) -> tuple[dict[str, Any], bool]:
    path, payload = _bound_json(path_value, sha_value, context="artifact registry")
    if payload.get("schema_version") != "gx1_artifact_selection_v2":
        raise RuntimeError("artifact registry schema is invalid")
    entry_iql = (payload.get("active") or {}).get("entry_iql")
    if not isinstance(entry_iql, dict):
        raise RuntimeError("artifact registry lacks exact Entry-IQL state")
    registered_path = entry_iql.get("path")
    available = (
        entry_iql.get("artifact_present") is True
        and isinstance(registered_path, str)
        and Path(registered_path).is_absolute()
        and Path(registered_path).is_dir()
        and not Path(registered_path).is_symlink()
        and entry_iql.get("status") != "RETIRED_ARTIFACT_ABSENT"
    )
    return {
        "path": str(path),
        "sha256": sha_value.lower(),
        "entry_iql": {
            "path": entry_iql.get("path"),
            "status": entry_iql.get("status"),
            "artifact_present": entry_iql.get("artifact_present"),
        },
        "benchmark_bytes_registered_available": available,
    }, available


def _recipe_probe() -> dict[str, Any]:
    values = {
        key: smoke_recipe_owner.DIRECTION_BALANCE_RECIPE_CONTRACT.get(key)
        for key in REQUIRED_POSITIVE_ABSTENTION_RECIPE_FIELDS
    }
    positive = all(
        isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0
        for value in values.values()
    )
    source_path = Path(smoke_recipe_owner.__file__).resolve()
    return {
        "source_path": str(source_path),
        "source_sha256": sha256_file(source_path),
        "values": values,
        "all_required_weights_positive": positive,
        "code_contract_only": True,
        "empirical_abstention_edge_proven": False,
    }


def _out_dir(path_value: str) -> Path:
    path = Path(path_value).expanduser().resolve()
    if path == GX1_DATA_ROOT or path.is_relative_to(GX1_DATA_ROOT):
        raise RuntimeError("abstention probe output under GX1_DATA is forbidden")
    return path


def run(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = _out_dir(args.out_dir)
    manifests = {
        split: _manifest_probe(
            getattr(args, f"{split}_smoke_manifest"),
            getattr(args, f"{split}_smoke_manifest_sha256"),
            expected_split=split,
        )
        for split in ("train", "val", "test")
    }
    registry, registry_available = _artifact_registry_probe(
        args.artifact_registry_json,
        args.artifact_registry_sha256,
    )
    recipe = _recipe_probe()
    blockers: list[str] = []
    historical: dict[str, Any] | None = None
    learned: dict[str, Any] | None = None
    comparison: dict[str, Any] | None = None

    benchmark_args = _paired_optional(
        args.benchmark_evidence_json,
        args.benchmark_evidence_sha256,
        label="historical benchmark evidence",
    )
    learned_args = _paired_optional(
        args.learned_probe_evidence_json,
        args.learned_probe_evidence_sha256,
        label="learned probe evidence",
    )
    if not registry_available:
        blockers.append("HISTORICAL_ENTRY_IQL_BENCHMARK_BYTES_NOT_REGISTERED")
    if benchmark_args is None:
        blockers.append("EXACT_HISTORICAL_SELECTION_BENCHMARK_EVIDENCE_MISSING")
    else:
        try:
            historical = validate_selection_evidence(
                benchmark_args[0], benchmark_args[1], expected_role=HISTORICAL_ROLE
            )
        except AbstentionProbeEvidenceError as exc:
            blockers.append(f"HISTORICAL_SELECTION_BENCHMARK_INVALID: {exc}")
    if learned_args is None:
        blockers.append("EXACT_MODEL_NATIVE_LEARNED_ABSTENTION_PROBE_EVIDENCE_MISSING")
    else:
        try:
            learned = validate_selection_evidence(
                learned_args[0], learned_args[1], expected_role=MODEL_NATIVE_ROLE
            )
        except AbstentionProbeEvidenceError as exc:
            blockers.append(f"MODEL_NATIVE_LEARNED_ABSTENTION_PROBE_INVALID: {exc}")
    if not recipe["all_required_weights_positive"]:
        blockers.append("ACTIVE_ABSTENTION_RECIPE_WEIGHTS_NOT_POSITIVE")
    if historical is not None and learned is not None:
        comparison = compare_selection_evidence(historical, learned)
        if not comparison["passed"]:
            blockers.append("MODEL_NATIVE_ABSTENTION_DID_NOT_MATCH_OR_BEAT_BENCHMARK")

    empirical_gate_passed = not blockers and comparison is not None and comparison["passed"]
    report = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "decision": PASS_DECISION if empirical_gate_passed else BLOCK_DECISION,
        "report_only": True,
        "purpose": "pre_rebuild_cost_control_abstention_diagnostic",
        "inputs": {
            "historical_smoke_manifests": manifests,
            "artifact_registry": registry,
            "historical_benchmark_evidence": historical,
            "model_native_learned_probe_evidence": learned,
        },
        "probe_diagnostics": {
            "historical_rejected_seq520_label_balance": manifests,
            "active_recipe": recipe,
            "interpretation": (
                "Balanced labels show that source label starvation is not the observed zero-FLAT "
                "failure. They do not prove learned abstention or benchmark parity."
            ),
        },
        "empirical_comparison": comparison,
        "empirical_comparison_performed": comparison is not None,
        "empirical_gate_passed": empirical_gate_passed,
        "blockers": blockers,
        "direction_authority": False,
        "fallback_authority": False,
        "rebuild_authorized": False,
        "training_authorized": False,
        "launch_authorized": False,
        "live_authorized": False,
        "side_effects": {
            "parquet_read": False,
            "dataset_rebuild": False,
            "training": False,
            "replay": False,
            "shadow_or_live": False,
        },
        "next_required_evidence": (
            "separate explicit vedtak and downstream preflight"
            if empirical_gate_passed
            else "recover exact immutable aligned historical benchmark and learned-probe row evidence; otherwise stop"
        ),
    }
    report["evidence_binding_sha256"] = hashlib.sha256(
        json.dumps(report["inputs"], sort_keys=True, allow_nan=False).encode("utf-8")
    ).hexdigest()
    _, published = write_immutable_json_event(out_dir, EVENT_PREFIX, report)
    return published


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    for split in ("train", "val", "test"):
        parser.add_argument(f"--{split}-smoke-manifest", required=True)
        parser.add_argument(f"--{split}-smoke-manifest-sha256", required=True)
    parser.add_argument("--artifact-registry-json", required=True)
    parser.add_argument("--artifact-registry-sha256", required=True)
    parser.add_argument("--benchmark-evidence-json")
    parser.add_argument("--benchmark-evidence-sha256")
    parser.add_argument("--learned-probe-evidence-json")
    parser.add_argument("--learned-probe-evidence-sha256")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--quiet", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    report = run(args)
    if not args.quiet:
        print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["decision"] == PASS_DECISION else 2


if __name__ == "__main__":
    raise SystemExit(main())
