"""Fail-closed cross-surface duplicate policy for the unified Entry/Exit model.

The model has two native decision surfaces (Entry M5 and Exit M1) plus the
immutable multi-timeframe V4 cache.  Individual surface liveness proves that a
column varies; it cannot prove that two *simultaneously consumed* surfaces do
not carry an unnoticed byte-identical feature.  This module owns that second
question.

Only a small set of MTF current-bar context values is intentionally also
available to the context-token path.  They are declared here from the sole
projection owner, reported as aliases, and are never silently ignored.  Every
other exact local-to-active-MTF duplicate fails the research dataset build.
"""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from gx1.contracts.entry_exit_feature_base_v1 import (
    ENTRY_MTF_CONTEXT_TIMEFRAMES,
    EXIT_MTF_CONTEXT_TIMEFRAMES,
)
from gx1.contracts.entry_model_native_signal_v1 import (
    MODEL_NATIVE_AVAILABLE_CANDIDATE_FIELDS,
    MODEL_NATIVE_CTX_CONT_FIELDS,
    MODEL_NATIVE_MANDATORY_SELECTED_FIELDS,
    MODEL_NATIVE_SIGNAL_DIM,
)
from gx1.features.htf_features import (
    MODEL_NATIVE_CONTEXT_MTF_PROJECTION,
    MODEL_NATIVE_MTF_SCALAR_PER_BAR_EXACT_ALIASES_V4,
    MULTI_TF_PER_BAR_FEATURES_V4,
)


SCHEMA_VERSION = "entry_cross_surface_input_overlap_v3"
POLICY_VERSION = "entry_cross_surface_input_overlap_policy_v3"
DECISION_ROUTES = {
    "entry": {
        "local_timeframe": "M5",
        "active_mtf_timeframes": tuple(ENTRY_MTF_CONTEXT_TIMEFRAMES),
    },
    "exit": {
        "local_timeframe": "M1",
        "active_mtf_timeframes": tuple(EXIT_MTF_CONTEXT_TIMEFRAMES),
    },
}


def declared_context_mtf_aliases(*, decision: str) -> frozenset[tuple[str, str]]:
    """Return the exact, intentional context-token / MTF-last-bar aliases.

    The values are duplicated across representation paths, not created by a
    second feature producer: the context path makes a current higher-TF value
    available to family gates while the MTF encoder reads its causal history.
    Deriving the list from ``MODEL_NATIVE_CONTEXT_MTF_PROJECTION`` ensures that
    adding a new context projection cannot silently escape duplicate reporting.
    """

    route = DECISION_ROUTES.get(str(decision))
    if route is None:
        raise RuntimeError(f"CROSS_SURFACE_DECISION_INVALID: {decision!r}")
    aliases: set[tuple[str, str]] = set()
    signal_fields = frozenset(
        (*MODEL_NATIVE_MANDATORY_SELECTED_FIELDS, *MODEL_NATIVE_AVAILABLE_CANDIDATE_FIELDS)
    )

    def add_local_paths(*, ctx_name: str, mtf_name: str) -> None:
        if ctx_name not in MODEL_NATIVE_CTX_CONT_FIELDS:
            raise RuntimeError(
                f"CROSS_SURFACE_DECLARED_ALIAS_CONTEXT_MISSING: {ctx_name}"
            )
        aliases.add((f"local.ctx_cont.{ctx_name}", mtf_name))
        signal_name = f"ctx_cont.{ctx_name}"
        if signal_name in signal_fields:
            aliases.add((f"local.signal.{signal_name}", mtf_name))

    for timeframe in route["active_mtf_timeframes"]:
        tf = str(timeframe).lower()
        for output_name, source_name in MODEL_NATIVE_CONTEXT_MTF_PROJECTION:
            ctx_name = f"{tf}_{output_name}_v2"
            if ctx_name in MODEL_NATIVE_CTX_CONT_FIELDS:
                add_local_paths(
                    ctx_name=ctx_name,
                    mtf_name=f"mtf.{tf}.{source_name}",
                )
        for scalar_name, per_bar_name in (
            MODEL_NATIVE_MTF_SCALAR_PER_BAR_EXACT_ALIASES_V4[str(timeframe)]
        ):
            add_local_paths(
                ctx_name=scalar_name,
                mtf_name=f"mtf.{tf}.{per_bar_name}",
            )
    return frozenset(aliases)


def _hash_groups(values: Mapping[str, str]) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for key, digest in values.items():
        name = str(key)
        value = str(digest).lower()
        if len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value):
            raise RuntimeError(f"CROSS_SURFACE_FIELD_HASH_INVALID: {name}")
        grouped[value].append(name)
    return {digest: sorted(names) for digest, names in grouped.items()}


def _require_hash_mapping(value: Any, *, expected_keys: set[str]) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != expected_keys:
        raise RuntimeError("CROSS_SURFACE_REPORT_FIELD_HASH_SET_INVALID")
    result = {str(key): str(digest).lower() for key, digest in value.items()}
    _hash_groups(result)
    return result


def classify_active_duplicate_pairs(
    *,
    decision: str,
    local_field_hashes: Mapping[str, str],
    active_mtf_field_hashes: Mapping[str, str],
) -> dict[str, list[dict[str, Any]]]:
    """Classify exact duplicates on a single actual decision-input population.

    ``local_field_hashes`` and ``active_mtf_field_hashes`` must be streaming
    SHA-256 values over the same chronological decision timestamps and their
    canonical float32 values.  Hash equality therefore represents an exact
    observed value sequence, not matching summary statistics.
    """

    aliases = declared_context_mtf_aliases(decision=decision)
    local_by_hash = _hash_groups(local_field_hashes)
    mtf_by_hash = _hash_groups(active_mtf_field_hashes)
    declared: list[dict[str, Any]] = []
    unexpected: list[dict[str, Any]] = []
    for digest in sorted(set(local_by_hash) & set(mtf_by_hash)):
        for local in local_by_hash[digest]:
            for mtf in mtf_by_hash[digest]:
                row = {"local_field": local, "mtf_field": mtf, "values_sha256": digest}
                if (local, mtf) in aliases:
                    declared.append(row)
                else:
                    unexpected.append(row)
    observed_declared = {
        (str(row["local_field"]), str(row["mtf_field"])) for row in declared
    }
    return {
        "declared_context_mtf_alias_pairs": declared,
        "missing_declared_context_mtf_alias_pairs": [
            {"local_field": local, "mtf_field": mtf}
            for local, mtf in sorted(aliases - observed_declared)
        ],
        "unexpected_active_exact_duplicate_pairs": unexpected,
    }


def require_eight_family_coverage(
    *,
    local_fields: Sequence[str],
    mtf_feature_names: Sequence[str],
) -> dict[str, dict[str, int]]:
    """Return non-empty eight-family coverage for both physical input planes."""

    from gx1.features.entry_specialist_feature_groups_v1 import (
        MODEL_NATIVE_TRAINING_SPECIALISTS,
        group_features_by_specialist,
        require_multi_tf_specialist_routing_v4,
    )

    local = group_features_by_specialist(local_fields)
    mtf_indices = require_multi_tf_specialist_routing_v4(mtf_feature_names)
    if local.get("unmapped") or local.get("retired_legacy_bridge"):
        raise RuntimeError("CROSS_SURFACE_LOCAL_FAMILY_ROUTING_INVALID")
    if tuple(mtf_indices) != tuple(MODEL_NATIVE_TRAINING_SPECIALISTS):
        raise RuntimeError("CROSS_SURFACE_MTF_FAMILY_ROUTING_INVALID")
    result: dict[str, dict[str, int]] = {}
    for family in MODEL_NATIVE_TRAINING_SPECIALISTS:
        local_count = len(local.get(family, ()))
        mtf_count = len(mtf_indices[family])
        if local_count <= 0 or mtf_count <= 0:
            raise RuntimeError(
                f"CROSS_SURFACE_FAMILY_EMPTY: {family}: "
                f"local={local_count} mtf={mtf_count}"
            )
        result[family] = {
            "local_field_count": local_count,
            "mtf_field_count": mtf_count,
        }
    return result


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_cross_surface_overlap_report(
    path: str | Path,
    *,
    expected_sha256: str | None = None,
    expected_entry_run_id: str | None = None,
    expected_input_bindings: Mapping[str, Mapping[str, str]] | None = None,
) -> dict[str, Any]:
    """Validate an immutable PASS report before it can bind a dataset.

    The report is an input-contract artifact, not a dashboard: all routes,
    family coverage and every declared alias must be present.  Optional input
    bindings let the dataset builder prove it audited these exact source bytes.
    """

    candidate = Path(path).expanduser()
    if (
        not candidate.is_absolute()
        or candidate.is_symlink()
        or not candidate.is_file()
        or candidate.resolve() != candidate
    ):
        raise RuntimeError("CROSS_SURFACE_REPORT_PATH_INVALID")
    observed_sha = _sha256_file(candidate)
    if expected_sha256 is not None and observed_sha != str(expected_sha256):
        raise RuntimeError("CROSS_SURFACE_REPORT_SHA256_MISMATCH")
    try:
        report = json.loads(candidate.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("CROSS_SURFACE_REPORT_JSON_INVALID") from exc
    if not isinstance(report, dict):
        raise RuntimeError("CROSS_SURFACE_REPORT_ROOT_INVALID")
    if (
        report.get("schema_version") != SCHEMA_VERSION
        or report.get("decision") != "PASS"
        or report.get("failures") != []
    ):
        raise RuntimeError("CROSS_SURFACE_REPORT_DECISION_INVALID")
    if (
        expected_entry_run_id is not None
        and report.get("entry_run_id") != expected_entry_run_id
    ):
        raise RuntimeError("CROSS_SURFACE_REPORT_RUN_ID_MISMATCH")
    policy = report.get("policy")
    if not isinstance(policy, Mapping) or policy.get("version") != POLICY_VERSION:
        raise RuntimeError("CROSS_SURFACE_REPORT_POLICY_INVALID")
    declared_routes = policy.get("decision_routes")
    expected_routes = {
        decision: {
            "local_timeframe": route["local_timeframe"],
            "active_mtf_timeframes": list(route["active_mtf_timeframes"]),
        }
        for decision, route in DECISION_ROUTES.items()
    }
    if (
        declared_routes != expected_routes
        or policy.get("decision_population")
        != "manifest_bound_history_start_through_surface_end"
    ):
        raise RuntimeError("CROSS_SURFACE_REPORT_ROUTE_INVALID")
    bindings = report.get("input_bindings")
    signal_binding = (
        bindings.get("signal_manifest") if isinstance(bindings, Mapping) else None
    )
    raw_history_start = (
        signal_binding.get("feature_history_start_utc")
        if isinstance(signal_binding, Mapping)
        else None
    )
    if not isinstance(raw_history_start, str):
        raise RuntimeError("CROSS_SURFACE_REPORT_POPULATION_INVALID")
    try:
        history_start = pd.Timestamp(raw_history_start)
    except Exception as exc:
        raise RuntimeError("CROSS_SURFACE_REPORT_POPULATION_INVALID") from exc
    if history_start.tzinfo is None:
        raise RuntimeError("CROSS_SURFACE_REPORT_POPULATION_INVALID")
    history_start_ns = int(history_start.tz_convert("UTC").value)
    coverage = report.get("eight_family_coverage")
    if not isinstance(coverage, Mapping) or len(coverage) != 8:
        raise RuntimeError("CROSS_SURFACE_REPORT_FAMILY_COVERAGE_INVALID")
    for family, row in coverage.items():
        if (
            not isinstance(family, str)
            or not isinstance(row, Mapping)
            or int(row.get("local_field_count") or 0) <= 0
            or int(row.get("mtf_field_count") or 0) <= 0
        ):
            raise RuntimeError("CROSS_SURFACE_REPORT_FAMILY_COVERAGE_INVALID")
    for decision, route in DECISION_ROUTES.items():
        row = report.get(decision)
        if not isinstance(row, Mapping):
            raise RuntimeError("CROSS_SURFACE_REPORT_DECISION_ROW_INVALID")
        if (
            row.get("local_timeframe") != route["local_timeframe"]
            or row.get("active_mtf_timeframes") != list(route["active_mtf_timeframes"])
            or int(row.get("row_count") or 0) <= 0
            or row.get("unexpected_active_exact_duplicate_pairs") != []
            or row.get("missing_declared_context_mtf_alias_pairs") != []
        ):
            raise RuntimeError("CROSS_SURFACE_REPORT_DUPLICATE_DECISION_INVALID")
        source_row_count = row.get("source_row_count")
        excluded_rows = row.get("excluded_pre_history_row_count")
        audit_start_ns = row.get("audit_start_time_ns")
        source_first_time_ns = row.get("source_first_time_ns")
        first_time_ns = row.get("first_time_ns")
        last_time_ns = row.get("last_time_ns")
        if (
            isinstance(source_row_count, bool)
            or not isinstance(source_row_count, int)
            or isinstance(excluded_rows, bool)
            or not isinstance(excluded_rows, int)
            or isinstance(audit_start_ns, bool)
            or not isinstance(audit_start_ns, int)
            or isinstance(source_first_time_ns, bool)
            or not isinstance(source_first_time_ns, int)
            or isinstance(first_time_ns, bool)
            or not isinstance(first_time_ns, int)
            or isinstance(last_time_ns, bool)
            or not isinstance(last_time_ns, int)
            or source_row_count != int(row["row_count"]) + excluded_rows
            or source_row_count < int(row["row_count"])
            or excluded_rows < 0
            or audit_start_ns != history_start_ns
            or source_first_time_ns > first_time_ns
            or first_time_ns < audit_start_ns
            or last_time_ns < first_time_ns
            or (excluded_rows == 0 and source_first_time_ns != first_time_ns)
            or (excluded_rows > 0 and source_first_time_ns >= audit_start_ns)
        ):
            raise RuntimeError("CROSS_SURFACE_REPORT_POPULATION_INVALID")
        local_hashes_raw = row.get("local_field_hashes")
        if not isinstance(local_hashes_raw, Mapping):
            raise RuntimeError("CROSS_SURFACE_REPORT_FIELD_HASH_SET_INVALID")
        local_keys = {str(key) for key in local_hashes_raw}
        expected_ctx_keys = {
            f"local.ctx_cont.{field}" for field in MODEL_NATIVE_CTX_CONT_FIELDS
        }
        signal_keys = {key for key in local_keys if key.startswith("local.signal.")}
        if (
            not expected_ctx_keys <= local_keys
            or len(signal_keys) != MODEL_NATIVE_SIGNAL_DIM
            or len(local_keys) != MODEL_NATIVE_SIGNAL_DIM + len(expected_ctx_keys)
        ):
            raise RuntimeError("CROSS_SURFACE_REPORT_FIELD_HASH_SET_INVALID")
        local_hashes = _require_hash_mapping(
            local_hashes_raw,
            expected_keys=local_keys,
        )
        active_mtf_keys = {
            f"mtf.{str(timeframe).lower()}.{field}"
            for timeframe in route["active_mtf_timeframes"]
            for field in MULTI_TF_PER_BAR_FEATURES_V4
        }
        active_mtf_hashes = _require_hash_mapping(
            row.get("active_mtf_field_hashes"),
            expected_keys=active_mtf_keys,
        )
        aliases = row.get("declared_context_mtf_alias_pairs")
        if not isinstance(aliases, list):
            raise RuntimeError("CROSS_SURFACE_REPORT_DECLARED_ALIAS_INVALID")
        expected_aliases = declared_context_mtf_aliases(decision=decision)
        if len(aliases) != len(expected_aliases):
            raise RuntimeError("CROSS_SURFACE_REPORT_DECLARED_ALIAS_INVALID")
        observed_aliases = {
            (str(item.get("local_field")), str(item.get("mtf_field")))
            for item in aliases
            if isinstance(item, Mapping)
            and set(item) == {"local_field", "mtf_field", "values_sha256"}
            and isinstance(item.get("values_sha256"), str)
            and len(str(item["values_sha256"])) == 64
            and all(
                ch in "0123456789abcdef" for ch in str(item["values_sha256"]).lower()
            )
        }
        if observed_aliases != expected_aliases:
            raise RuntimeError("CROSS_SURFACE_REPORT_DECLARED_ALIAS_INVALID")
        recomputed = classify_active_duplicate_pairs(
            decision=decision,
            local_field_hashes=local_hashes,
            active_mtf_field_hashes=active_mtf_hashes,
        )
        if (
            row.get("declared_context_mtf_alias_pairs")
            != recomputed["declared_context_mtf_alias_pairs"]
            or row.get("missing_declared_context_mtf_alias_pairs")
            != recomputed["missing_declared_context_mtf_alias_pairs"]
            or row.get("unexpected_active_exact_duplicate_pairs")
            != recomputed["unexpected_active_exact_duplicate_pairs"]
        ):
            raise RuntimeError("CROSS_SURFACE_REPORT_DUPLICATE_RECOMPUTE_INVALID")
    if not isinstance(bindings, Mapping):
        raise RuntimeError("CROSS_SURFACE_REPORT_BINDINGS_INVALID")
    if expected_input_bindings is not None:
        for binding_name, expected in expected_input_bindings.items():
            observed = bindings.get(binding_name)
            if not isinstance(observed, Mapping):
                raise RuntimeError("CROSS_SURFACE_REPORT_BINDINGS_INVALID")
            for key, value in expected.items():
                if observed.get(key) != value:
                    raise RuntimeError(
                        f"CROSS_SURFACE_REPORT_BINDING_MISMATCH: {binding_name}.{key}"
                    )
    return {
        "path": str(candidate),
        "sha256": observed_sha,
        "schema_version": SCHEMA_VERSION,
        "entry_run_id": str(report.get("entry_run_id") or ""),
        "decision": "PASS",
        "row_counts": {
            decision: int(report[decision]["row_count"]) for decision in DECISION_ROUTES
        },
    }
