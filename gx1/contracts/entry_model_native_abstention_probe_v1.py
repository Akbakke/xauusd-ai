"""Exact report-only evidence contract for the Entry abstention probe.

The historical Entry-IQL gate is comparison evidence only.  This module never
loads a model and never returns a LONG/SHORT/FLAT decision.  A model-native
probe row is a take exactly when the model's calibrated three-class argmax is
not FLAT; no threshold or secondary selector is accepted.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "entry_abstention_selection_evidence_v1"
ROW_SCHEMA_VERSION = "entry_abstention_selection_rows_v1"
HISTORICAL_ROLE = "historical_entry_iql_raw_adv_benchmark_comparison_only"
MODEL_NATIVE_ROLE = "model_native_calibrated_argmax_abstention_probe"
UTILITY_DEFINITION = "shared_realized_net_utility_bps_after_costs_v1"
MAX_COVERAGE_DELTA = 0.02

_BASE_ROW_KEYS = {
    "sample_id",
    "time_utc",
    "take",
    "realized_net_utility_bps",
    "costs_included",
}
_MODEL_ROW_KEYS = _BASE_ROW_KEYS | {
    "model_direction_index",
    "calibrated_argmax",
}


class AbstentionProbeEvidenceError(RuntimeError):
    """Raised when immutable abstention evidence is not exact."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _strict_json(text: str, *, context: str) -> Any:
    try:
        return json.loads(
            text,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON token {value}")
            ),
        )
    except (json.JSONDecodeError, ValueError) as exc:
        raise AbstentionProbeEvidenceError(f"{context}: invalid strict JSON: {exc}") from exc


def _exact_file(path_value: Any, sha_value: Any, *, context: str) -> Path:
    if not isinstance(path_value, (str, Path)) or not str(path_value).strip():
        raise AbstentionProbeEvidenceError(f"{context}: path is missing")
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        raise AbstentionProbeEvidenceError(f"{context}: path must be absolute")
    if any("latest" in part.lower() for part in path.parts):
        raise AbstentionProbeEvidenceError(f"{context}: mutable latest path is forbidden")
    if path.is_symlink() or not path.is_file():
        raise AbstentionProbeEvidenceError(f"{context}: regular immutable file is missing")
    expected_sha = str(sha_value or "").lower()
    if len(expected_sha) != 64 or any(c not in "0123456789abcdef" for c in expected_sha):
        raise AbstentionProbeEvidenceError(f"{context}: expected sha256 is invalid")
    actual_sha = sha256_file(path)
    if actual_sha != expected_sha:
        raise AbstentionProbeEvidenceError(
            f"{context}: sha256 mismatch expected={expected_sha} actual={actual_sha}"
        )
    return path.resolve()


def _finite(value: Any, *, context: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise AbstentionProbeEvidenceError(f"{context}: expected finite number") from exc
    if not math.isfinite(number):
        raise AbstentionProbeEvidenceError(f"{context}: expected finite number")
    return number


def _read_rows(path: Path, *, role: str) -> tuple[dict[str, Any], dict[str, float]]:
    expected_keys = _MODEL_ROW_KEYS if role == MODEL_NATIVE_ROLE else _BASE_ROW_KEYS
    seen: set[str] = set()
    take_values: list[float] = []
    skip_values: list[float] = []
    universe_rows: list[tuple[str, str, float]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.strip():
                raise AbstentionProbeEvidenceError(
                    f"row evidence line {line_number}: blank rows are forbidden"
                )
            row = _strict_json(raw, context=f"row evidence line {line_number}")
            if not isinstance(row, dict) or set(row) != expected_keys:
                raise AbstentionProbeEvidenceError(
                    f"row evidence line {line_number}: keys must be exact"
                )
            sample_id = row["sample_id"]
            if not isinstance(sample_id, str) or not sample_id.strip() or sample_id in seen:
                raise AbstentionProbeEvidenceError(
                    f"row evidence line {line_number}: sample_id missing or duplicated"
                )
            seen.add(sample_id)
            if not isinstance(row["time_utc"], str) or not row["time_utc"].endswith("Z"):
                raise AbstentionProbeEvidenceError(
                    f"row evidence line {line_number}: time_utc must be explicit UTC"
                )
            if type(row["take"]) is not bool or row["costs_included"] is not True:
                raise AbstentionProbeEvidenceError(
                    f"row evidence line {line_number}: take must be bool and costs included"
                )
            if role == MODEL_NATIVE_ROLE:
                direction = row["model_direction_index"]
                if type(direction) is not int or direction not in (0, 1, 2):
                    raise AbstentionProbeEvidenceError(
                        f"row evidence line {line_number}: model direction must be 0/1/2"
                    )
                if row["calibrated_argmax"] is not True or row["take"] != (direction != 2):
                    raise AbstentionProbeEvidenceError(
                        f"row evidence line {line_number}: take must equal calibrated argmax != FLAT"
                    )
            utility = _finite(
                row["realized_net_utility_bps"],
                context=f"row evidence line {line_number} utility",
            )
            universe_rows.append((sample_id, row["time_utc"], utility))
            (take_values if row["take"] else skip_values).append(utility)

    rows = len(seen)
    if rows < 2 or not take_values or not skip_values:
        raise AbstentionProbeEvidenceError(
            "row evidence must contain at least two rows and non-empty take/skip groups"
        )
    take_mean = sum(take_values) / len(take_values)
    skip_mean = sum(skip_values) / len(skip_values)
    metrics = {
        "rows": float(rows),
        "take_rows": float(len(take_values)),
        "skip_rows": float(len(skip_values)),
        "coverage": len(take_values) / rows,
        "take_ev_net_bps": take_mean,
        "skip_ev_net_bps": skip_mean,
        "take_skip_separation_net_bps": take_mean - skip_mean,
    }
    universe_sha256 = hashlib.sha256(
        json.dumps(
            sorted(universe_rows),
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    return {"evaluation_universe_sha256": universe_sha256, "rows": rows}, metrics


def validate_selection_evidence(
    evidence_path: Path,
    expected_sha256: str,
    *,
    expected_role: str,
) -> dict[str, Any]:
    """Validate one evidence declaration and recompute its row metrics."""

    path = _exact_file(evidence_path, expected_sha256, context="selection evidence")
    payload = _strict_json(path.read_text(encoding="utf-8"), context="selection evidence")
    if not isinstance(payload, dict):
        raise AbstentionProbeEvidenceError("selection evidence root must be an object")
    if set(payload) != {
        "schema_version",
        "role",
        "evaluation_scope",
        "split",
        "utility_definition",
        "evaluation_universe_sha256",
        "authority",
        "row_evidence",
        "metrics",
    }:
        raise AbstentionProbeEvidenceError("selection evidence keys must be exact")
    if payload.get("schema_version") != SCHEMA_VERSION or payload.get("role") != expected_role:
        raise AbstentionProbeEvidenceError("selection evidence schema or role is invalid")
    if payload.get("evaluation_scope") != "strict_oot" or payload.get("split") != "test":
        raise AbstentionProbeEvidenceError("selection evidence must be strict-OOT TEST")
    if payload.get("utility_definition") != UTILITY_DEFINITION:
        raise AbstentionProbeEvidenceError("selection evidence utility definition is invalid")
    authority = payload.get("authority")
    if authority != {
        "direction": False,
        "fallback": False,
        "launch": False,
        "live": False,
    }:
        raise AbstentionProbeEvidenceError("selection evidence authority must be exactly closed")
    universe = payload.get("evaluation_universe_sha256")
    if (
        not isinstance(universe, str)
        or len(universe) != 64
        or any(c not in "0123456789abcdef" for c in universe)
    ):
        raise AbstentionProbeEvidenceError("evaluation universe sha256 is invalid")
    row_evidence = payload.get("row_evidence")
    if not isinstance(row_evidence, dict) or row_evidence.get("schema_version") != ROW_SCHEMA_VERSION:
        raise AbstentionProbeEvidenceError("row evidence declaration is invalid")
    if set(row_evidence) != {"schema_version", "format", "path", "sha256"}:
        raise AbstentionProbeEvidenceError("row evidence declaration keys must be exact")
    if row_evidence.get("format") != "jsonl":
        raise AbstentionProbeEvidenceError("row evidence format must be jsonl")
    row_path = _exact_file(
        row_evidence.get("path"),
        row_evidence.get("sha256"),
        context="selection row evidence",
    )
    identity, recomputed = _read_rows(row_path, role=expected_role)
    if universe != identity["evaluation_universe_sha256"]:
        raise AbstentionProbeEvidenceError(
            "evaluation universe sha256 does not match the exact row universe"
        )
    declared = payload.get("metrics")
    if not isinstance(declared, dict) or set(declared) != set(recomputed):
        raise AbstentionProbeEvidenceError("selection evidence metric keys are invalid")
    for key, actual in recomputed.items():
        expected = _finite(declared[key], context=f"declared metric {key}")
        if not math.isclose(expected, actual, rel_tol=0.0, abs_tol=1e-12):
            raise AbstentionProbeEvidenceError(
                f"selection evidence metric mismatch {key}: declared={expected} actual={actual}"
            )
    return {
        "path": str(path),
        "sha256": expected_sha256.lower(),
        "role": expected_role,
        "evaluation_universe_sha256": universe,
        "utility_definition": UTILITY_DEFINITION,
        "row_evidence": {"path": str(row_path), "sha256": row_evidence["sha256"]},
        "metrics": {key: int(value) if key.endswith("rows") else value for key, value in recomputed.items()},
    }


def compare_selection_evidence(
    historical: dict[str, Any],
    model_native: dict[str, Any],
) -> dict[str, Any]:
    """Compare exact aligned OOT evidence at contracted comparable coverage."""

    same_universe = (
        historical["evaluation_universe_sha256"]
        == model_native["evaluation_universe_sha256"]
    )
    historical_metrics = historical["metrics"]
    model_metrics = model_native["metrics"]
    coverage_delta = abs(model_metrics["coverage"] - historical_metrics["coverage"])
    coverage_comparable = coverage_delta <= MAX_COVERAGE_DELTA
    take_ev_no_worse = (
        model_metrics["take_ev_net_bps"] >= historical_metrics["take_ev_net_bps"]
    )
    separation_no_worse = (
        model_metrics["take_skip_separation_net_bps"]
        >= historical_metrics["take_skip_separation_net_bps"]
    )
    passed = same_universe and coverage_comparable and take_ev_no_worse and separation_no_worse
    return {
        "passed": passed,
        "same_evaluation_universe": same_universe,
        "max_coverage_delta": MAX_COVERAGE_DELTA,
        "coverage_delta": coverage_delta,
        "coverage_comparable": coverage_comparable,
        "take_ev_no_worse": take_ev_no_worse,
        "take_skip_separation_no_worse": separation_no_worse,
    }
