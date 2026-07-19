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

import numpy as np
import pandas as pd

from gx1.contracts.entry_model_native_abstention_probe_v1 import (
    HISTORICAL_ROLE,
    MODEL_NATIVE_ROLE,
    AbstentionProbeEvidenceError,
    compare_selection_evidence,
    sha256_file,
    validate_selection_evidence,
)
from gx1.contracts.immutable_event_authority_v1 import write_immutable_json_event
from gx1.contracts.model_native_serve_gate_v1 import (
    MODEL_NATIVE_REQUIRED_MODEL_NAME,
)
from gx1.models.entry_v10.direction_decision_contract import (
    MODEL_DIRECTION_SELECTION_MODE,
)
from gx1.scripts import materialize_entry_model_native_seq513_smoke_manifest_v1 as smoke_recipe_owner
from gx1.scripts.audit_model_native_direction_pockets_v1 import (
    _model_direction_contract_failures,
)
from gx1.scripts.entry_candidate_prediction_evidence_v1 import (
    resolve_and_validate_prediction_evidence,
)


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
_LEARNED_PREDICTION_ARGUMENTS = (
    "learned_predictions_parquet",
    "learned_predictions_sha256",
    "learned_prediction_report_json",
    "learned_prediction_report_sha256",
    "learned_bundle_dir",
    "learned_dataset_dir",
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


def _bound_file(path_value: str, expected_sha256: str, *, context: str) -> Path:
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        raise RuntimeError(f"{context}: explicit path must be absolute")
    if any("latest" in part.lower() for part in path.parts):
        raise RuntimeError(f"{context}: mutable latest path is forbidden")
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"{context}: regular input file is missing")
    resolved = path.resolve()
    if resolved != path:
        raise RuntimeError(f"{context}: explicit path must be canonical")
    expected = str(expected_sha256 or "").lower()
    actual = sha256_file(path)
    if len(expected) != 64 or actual != expected:
        raise RuntimeError(
            f"{context}: sha256 mismatch expected={expected!r} actual={actual}"
        )
    return resolved


def _bound_json(path_value: str, expected_sha256: str, *, context: str) -> tuple[Path, dict[str, Any]]:
    path = _bound_file(path_value, expected_sha256, context=context)
    return path, _strict_json(path)


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


def _learned_prediction_args(args: argparse.Namespace) -> dict[str, str] | None:
    values = {
        name: str(getattr(args, name, "") or "").strip()
        for name in _LEARNED_PREDICTION_ARGUMENTS
    }
    present = {name for name, value in values.items() if value}
    if present and present != set(_LEARNED_PREDICTION_ARGUMENTS):
        missing = sorted(set(_LEARNED_PREDICTION_ARGUMENTS) - present)
        raise RuntimeError(
            f"learned prediction lineage is partial; missing={missing}"
        )
    return values if present else None


def _canonical_dir(path_value: str, *, context: str) -> Path:
    path = Path(path_value).expanduser()
    if (
        not path.is_absolute()
        or path.resolve() != path
        or path.is_symlink()
        or not path.is_dir()
        or any("latest" in part.lower() for part in path.parts)
    ):
        raise RuntimeError(f"{context}: immutable directory identity is invalid")
    return path


def _learned_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            rows.append(json.loads(raw))
    return rows


def _validate_learned_prediction_lineage(
    learned: dict[str, Any],
    lineage_args: dict[str, str],
) -> dict[str, Any]:
    predictions_path = _bound_file(
        lineage_args["learned_predictions_parquet"],
        lineage_args["learned_predictions_sha256"],
        context="learned predictions",
    )
    prediction_report_path, expected_report = _bound_json(
        lineage_args["learned_prediction_report_json"],
        lineage_args["learned_prediction_report_sha256"],
        context="learned prediction report",
    )
    bundle_dir = _canonical_dir(
        lineage_args["learned_bundle_dir"],
        context="learned bundle",
    )
    dataset_dir = _canonical_dir(
        lineage_args["learned_dataset_dir"],
        context="learned dataset",
    )
    authoritative, report, declaration = resolve_and_validate_prediction_evidence(
        predictions_path,
        prediction_report_path=prediction_report_path,
        bundle_dir=bundle_dir,
        dataset_dir=dataset_dir,
        expected_split="test",
        expected_model=MODEL_NATIVE_REQUIRED_MODEL_NAME,
    )
    if authoritative != predictions_path or report != expected_report:
        raise RuntimeError("learned prediction resolver changed explicit identity")
    if str(declaration.get("sha256") or "").lower() != lineage_args[
        "learned_predictions_sha256"
    ].lower():
        raise RuntimeError("learned prediction declaration hash mismatch")

    frame = pd.read_parquet(predictions_path)
    failures = _model_direction_contract_failures(frame)
    if failures:
        raise RuntimeError(
            "learned prediction model-direction contract failed: "
            + " | ".join(failures)
        )
    scoped = frame.loc[
        (frame["split"].astype(str) == "test")
        & (frame["model"].astype(str) == MODEL_NATIVE_REQUIRED_MODEL_NAME)
    ].copy()
    if scoped.empty:
        raise RuntimeError("learned prediction event has no candidate TEST rows")
    if set(scoped["selection_score_mode"].astype(str)) != {
        MODEL_DIRECTION_SELECTION_MODE
    }:
        raise RuntimeError("learned TEST rows use a non-model direction selector")
    times = pd.to_datetime(scoped["time"], utc=True, errors="coerce")
    if times.isna().any() or times.duplicated().any():
        raise RuntimeError("learned TEST prediction times are invalid or duplicated")
    if not bool((times.dt.floor("s") == times).all()):
        raise RuntimeError("learned TEST prediction times are not exact UTC seconds")
    scoped["time_utc"] = times.dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    if scoped["time_utc"].duplicated().any():
        raise RuntimeError("learned TEST prediction UTC keys are duplicated")

    learned_rows_path = _bound_file(
        learned["row_evidence"]["path"],
        learned["row_evidence"]["sha256"],
        context="learned selection rows",
    )
    evidence_rows = _learned_rows(learned_rows_path)
    if len(evidence_rows) != len(scoped):
        raise RuntimeError(
            "learned evidence rows do not exactly cover candidate TEST predictions"
        )
    evidence_by_time = {str(row["time_utc"]): row for row in evidence_rows}
    if len(evidence_by_time) != len(evidence_rows) or set(evidence_by_time) != set(
        scoped["time_utc"]
    ):
        raise RuntimeError(
            "learned evidence UTC universe differs from candidate TEST predictions"
        )
    directions = pd.to_numeric(
        scoped["pred_direction"],
        errors="coerce",
    ).to_numpy(dtype=np.float64)
    if not np.isfinite(directions).all() or not np.equal(
        directions,
        np.floor(directions),
    ).all():
        raise RuntimeError("learned prediction direction is not an exact integer")
    for time_utc, direction in zip(
        scoped["time_utc"],
        directions.astype(np.int64),
        strict=True,
    ):
        row = evidence_by_time[str(time_utc)]
        if row["model_direction_index"] != int(direction):
            raise RuntimeError(
                "learned evidence direction differs from prediction event"
            )
    if sha256_file(predictions_path) != lineage_args[
        "learned_predictions_sha256"
    ].lower():
        raise RuntimeError("learned predictions changed during validation")
    if sha256_file(prediction_report_path) != lineage_args[
        "learned_prediction_report_sha256"
    ].lower():
        raise RuntimeError("learned prediction report changed during validation")
    if sha256_file(learned_rows_path) != learned["row_evidence"]["sha256"]:
        raise RuntimeError("learned selection rows changed during validation")
    return {
        "predictions_path": str(predictions_path),
        "predictions_sha256": lineage_args["learned_predictions_sha256"].lower(),
        "prediction_report_path": str(prediction_report_path),
        "prediction_report_sha256": lineage_args[
            "learned_prediction_report_sha256"
        ].lower(),
        "bundle_dir": str(bundle_dir),
        "dataset_dir": str(dataset_dir),
        "model": MODEL_NATIVE_REQUIRED_MODEL_NAME,
        "split": "test",
        "rows": int(len(scoped)),
        "direction_rows_exact": True,
    }


def _artifact_registry_probe(path_value: str, sha_value: str) -> tuple[dict[str, Any], bool]:
    path, payload = _bound_json(path_value, sha_value, context="artifact registry")
    if payload.get("schema_version") != "gx1_artifact_selection_v2":
        raise RuntimeError("artifact registry schema is invalid")
    active = payload.get("active")
    if not isinstance(active, dict):
        raise RuntimeError("artifact registry active inventory is invalid")
    stale_active = sorted({"v10_entry", "entry_iql"}.intersection(active))
    if stale_active:
        raise RuntimeError(
            f"artifact registry keeps non-active Entry records under active: {stale_active}"
        )
    retired = payload.get("retired")
    if not isinstance(retired, dict) or not isinstance(retired.get("v10_entry"), dict):
        raise RuntimeError("artifact registry lacks rejected v10_entry retirement state")
    entry_iql = retired.get("entry_iql")
    if not isinstance(entry_iql, dict):
        raise RuntimeError("artifact registry lacks exact retired Entry-IQL state")
    registered_path = entry_iql.get("path")
    registered_sha = entry_iql.get("sha256")
    artifact_present = entry_iql.get("artifact_present")
    status = entry_iql.get("status")
    available = False
    if artifact_present is True:
        benchmark_path = Path(str(registered_path or "")).expanduser()
        benchmark_sha = str(registered_sha or "").strip().lower()
        if (
            status != "HISTORICAL_COMPARISON_ONLY"
            or not benchmark_path.is_absolute()
            or benchmark_path.is_symlink()
            or not benchmark_path.is_file()
            or benchmark_path.resolve() != benchmark_path
            or any("latest" in part.lower() for part in benchmark_path.parts)
            or len(benchmark_sha) != 64
            or any(character not in "0123456789abcdef" for character in benchmark_sha)
            or sha256_file(benchmark_path) != benchmark_sha
        ):
            raise RuntimeError(
                "retired Entry-IQL benchmark registration lacks exact path/sha identity"
            )
        available = True
    elif not (
        artifact_present is False
        and status == "RETIRED_ARTIFACT_ABSENT"
        and registered_path is None
        and registered_sha is None
    ):
        raise RuntimeError("retired Entry-IQL absence state is not exact")
    return {
        "path": str(path),
        "sha256": sha_value.lower(),
        "active_entry_roles_absent": True,
        "entry_iql": {
            "path": entry_iql.get("path"),
            "sha256": entry_iql.get("sha256"),
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
    learned_prediction_lineage: dict[str, Any] | None = None
    comparison: dict[str, Any] | None = None
    prediction_parquet_read = False

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
    learned_prediction_args = _learned_prediction_args(args)
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
    if historical is not None and registry_available:
        registered = registry["entry_iql"]
        if (
            historical["path"] != registered["path"]
            or historical["sha256"] != registered["sha256"]
        ):
            blockers.append(
                "HISTORICAL_SELECTION_BENCHMARK_IS_NOT_EXACT_REGISTERED_ARTIFACT"
            )
            historical = None
    elif historical is not None:
        historical = None
    if learned_args is None:
        blockers.append("EXACT_MODEL_NATIVE_LEARNED_ABSTENTION_PROBE_EVIDENCE_MISSING")
    else:
        try:
            learned = validate_selection_evidence(
                learned_args[0], learned_args[1], expected_role=MODEL_NATIVE_ROLE
            )
        except AbstentionProbeEvidenceError as exc:
            blockers.append(f"MODEL_NATIVE_LEARNED_ABSTENTION_PROBE_INVALID: {exc}")
    if learned is not None:
        if learned_prediction_args is None:
            blockers.append("EXACT_MODEL_NATIVE_PREDICTION_LINEAGE_MISSING")
            learned = None
        else:
            prediction_parquet_read = True
            try:
                learned_prediction_lineage = _validate_learned_prediction_lineage(
                    learned,
                    learned_prediction_args,
                )
            except (RuntimeError, OSError, ValueError) as exc:
                blockers.append(f"MODEL_NATIVE_PREDICTION_LINEAGE_INVALID: {exc}")
                learned = None
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
            "model_native_prediction_lineage": learned_prediction_lineage,
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
            "parquet_read": prediction_parquet_read,
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
    parser.add_argument("--learned-predictions-parquet")
    parser.add_argument("--learned-predictions-sha256")
    parser.add_argument("--learned-prediction-report-json")
    parser.add_argument("--learned-prediction-report-sha256")
    parser.add_argument("--learned-bundle-dir")
    parser.add_argument("--learned-dataset-dir")
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
