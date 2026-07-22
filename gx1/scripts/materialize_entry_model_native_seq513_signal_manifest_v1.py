#!/usr/bin/env python3
"""Materialize one immutable model-native seq513 signal manifest.

The upstream feature-ranking JSON is a required offline TRAIN-only input, not
runtime authority and not a selected-field pass-through.  This consumer always
places every code-owned causal full-stack field first, in registry order, then
fills only the remaining positions from the deterministic ranking.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from gx1.contracts.entry_model_native_signal_v1 import (
    FORBIDDEN_LEGACY_BRIDGE_FIELDS,
    MODEL_NATIVE_BASE_FIELDS,
    MODEL_NATIVE_BASE_SIGNAL_DIM,
    MODEL_NATIVE_CONTRACT_MODE,
    MODEL_NATIVE_MANDATORY_FAMILY_FEATURES,
    MODEL_NATIVE_MANDATORY_FULL_STACK_SHA256,
    MODEL_NATIVE_MANDATORY_SELECTED_FEATURE_COUNT,
    MODEL_NATIVE_MANDATORY_SELECTED_FIELDS,
    MODEL_NATIVE_PRETRAIN_POLARITY_SIGNAL_CONTRACT,
    MODEL_NATIVE_RANKED_REMAINDER_FEATURE_COUNT,
    MODEL_NATIVE_SELECTED_FEATURE_COUNT,
    MODEL_NATIVE_SIGNAL_DIM,
    MODEL_NATIVE_STRUCTURAL_AUX_LABEL_SIGNAL_CONTRACT,
    model_native_mandatory_full_stack_metadata,
    model_native_signal_contract_metadata,
    require_model_native_manifest,
)
from gx1.features.entry_specialist_feature_groups_v1 import (
    MODEL_NATIVE_TRAINING_SPECIALISTS,
    classify_entry_specialist_feature,
    group_features_by_specialist,
)
from gx1.features.entry_foundation_structure_v1 import (
    FOUNDATION_STRUCTURE_FEATURE_NAMES,
    FOUNDATION_STRUCTURE_FEATURE_VERSION,
)
from gx1.contracts.entry_run_lineage_v1 import require_entry_run_id
from gx1.contracts.entry_model_native_state_v2 import (
    MODEL_NATIVE_RANK_TRANSFORM,
    MODEL_NATIVE_TRAIN_RANK_SCHEMA_VERSION,
    parse_utc as parse_state_utc,
    validate_train_rank_reference_lineage_v2,
)


SIGNAL_MANIFEST_SCHEMA_VERSION = "entry_model_native_seq513_signal_manifest_v6"
SIGNAL_MANIFEST_PRODUCER = (
    "gx1.scripts.materialize_entry_model_native_seq513_signal_manifest_v1"
)
SIGNAL_MANIFEST_PRODUCER_VERSION = "v6"
SIGNAL_MANIFEST_EVENT_PREFIX = "ENTRY_MODEL_NATIVE_SEQ513_SIGNAL_MANIFEST"
TRAIN_FEATURE_RANKING_SCHEMA_VERSION = "entry_model_native_train_feature_ranking_v6"
TRAIN_FEATURE_RANKING_PRODUCER = "entry_model_native_train_feature_ranker"
TRAIN_FEATURE_RANKING_PRODUCER_VERSION = "v6"
TRAIN_FEATURE_RANKING_ORDER = {
    "primary": "score_descending",
    "tie_break": "feature_name_ascending",
    "rank_base": 1,
}
TRAIN_FEATURE_CAUSALITY_CONTRACT = {
    "feature_values_from_closed_bars_only": True,
    "validation_rows_used": False,
    "test_rows_used": False,
    "future_feature_rows_used": False,
    "target_columns_in_feature_matrix": False,
    "leakage_columns": [],
}
MAX_OUTPUT_PAST_SKEW_SECONDS = 300.0
MAX_OUTPUT_FUTURE_SKEW_SECONDS = 30.0

_RANKING_TOP_LEVEL_KEYS = frozenset(
    {
        "schema_version",
        "created_utc",
        "entry_run_id",
        "producer",
        "producer_version",
        "fit_scope",
        "train_start_utc",
        "train_end_utc",
        "source_time_max_utc",
        "target_time_max_utc",
        "source_sha256",
        "target_sha256",
        "rank_reference",
        "ranking_order",
        "causality_contract",
        "ranked_features",
    }
)
_RANK_REFERENCE_KEYS = frozenset(
    {
        "path",
        "sha256",
        "sidecar_path",
        "sidecar_sha256",
        "schema_version",
        "entry_run_id",
        "source_parquet",
        "source_parquet_sha256",
        "history_start_utc",
        "fit_start_utc",
        "fit_end_utc",
        "fit_row_count",
        "fit_scope",
        "rank_transform",
    }
)
_RANKING_ROW_KEYS = frozenset({"rank", "name", "score"})
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_EVENT_NAME_RE = re.compile(
    rf"^{SIGNAL_MANIFEST_EVENT_PREFIX}_(\d{{8}}T\d{{6}}(?:\d{{6}})?Z)\.json$"
)
_RANKING_NAME_RE = re.compile(
    r"^ENTRY_MODEL_NATIVE_TRAIN_FEATURE_RANKING_(\d{8}T\d{6}(?:\d{6})?Z)\.json$"
)
_FORBIDDEN_LEAK_BARE_NAMES = frozenset(
    {
        "label_horizon_bars",
        "direction_label",
        "direction_target",
        "target_direction",
        "future_return",
        "future_mfe",
        "future_mae",
        "mfe_first_n",
        "mae_first_n",
        "net_pnl_bps",
        "direction_correct",
    }
)
_FORBIDDEN_LEAK_PREFIXES = (
    "y_",
    "label.",
    "target.",
    "future.",
    "future_",
    "forward.",
    "forward_",
    "post.",
    "post_",
    "exit.",
    "exit_",
    "outcome_",
    "outcome.",
    "realized_",
    "mfe_",
    "mae_",
    "net_pnl",
)
_FORBIDDEN_LEAK_SUBSTRINGS = (
    "mfe_first_n",
    "mae_first_n",
    "net_pnl_bps",
    "direction_correct",
    "label_horizon_bars",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _parse_utc(value: object, *, field: str) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise RuntimeError(f"FEATURE_RANKING_{field.upper()}_MISSING")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise RuntimeError(f"FEATURE_RANKING_{field.upper()}_INVALID") from exc
    if parsed.tzinfo is None:
        raise RuntimeError(f"FEATURE_RANKING_{field.upper()}_NOT_UTC")
    if parsed.utcoffset() is None or parsed.utcoffset().total_seconds() != 0:
        raise RuntimeError(f"FEATURE_RANKING_{field.upper()}_NOT_UTC")
    return parsed.astimezone(timezone.utc)


def _timestamp_from_name(path: Path, pattern: re.Pattern[str], *, field: str) -> datetime:
    match = pattern.fullmatch(path.name)
    if match is None:
        raise RuntimeError(f"{field}_IMMUTABLE_TIMESTAMPED_NAME_INVALID: {path.name}")
    stamp = match.group(1)
    fmt = "%Y%m%dT%H%M%S%fZ" if len(stamp) == 22 else "%Y%m%dT%H%M%SZ"
    return datetime.strptime(stamp, fmt).replace(tzinfo=timezone.utc)


def _require_timestamp_match(path: Path, created: datetime, *, ranking: bool) -> None:
    pattern = _RANKING_NAME_RE if ranking else _EVENT_NAME_RE
    label = "FEATURE_RANKING" if ranking else "SIGNAL_MANIFEST"
    match = pattern.fullmatch(path.name)
    if match is None:
        raise RuntimeError(
            f"{label}_IMMUTABLE_TIMESTAMPED_NAME_INVALID: {path.name}"
        )
    has_microseconds = len(match.group(1)) == 22
    filename_time = _timestamp_from_name(
        path,
        pattern,
        field=label,
    )
    comparable = created if has_microseconds else created.replace(microsecond=0)
    if comparable != filename_time:
        raise RuntimeError(
            f"{label}_FILENAME_CREATED_UTC_MISMATCH: "
            f"filename={filename_time.isoformat()} created={created.isoformat()}"
        )


def _require_sha256(value: object, *, field: str) -> str:
    if not isinstance(value, str):
        raise RuntimeError(f"FEATURE_RANKING_{field.upper()}_INVALID")
    if not _SHA256_RE.fullmatch(value) or value == "0" * 64:
        raise RuntimeError(f"FEATURE_RANKING_{field.upper()}_INVALID")
    return value


def _bare_feature_name(name: str) -> str:
    bare = name.lower()
    for prefix in ("chart.", "ctx_cont.", "ctx_cat.", "snap.", "seq."):
        if bare.startswith(prefix):
            return bare[len(prefix) :]
    return bare


def _is_forbidden_leak_name(name: str) -> bool:
    bare = _bare_feature_name(name)
    return (
        bare in _FORBIDDEN_LEAK_BARE_NAMES
        or bare.startswith(_FORBIDDEN_LEAK_PREFIXES)
        or any(token in bare for token in _FORBIDDEN_LEAK_SUBSTRINGS)
    )


def _absolute_without_resolving(path: Path) -> Path:
    expanded = path.expanduser()
    if not expanded.is_absolute():
        expanded = Path.cwd() / expanded
    return Path(os.path.abspath(expanded))


def _require_no_symlink_components(path: Path, *, field: str) -> None:
    for component in (path, *path.parents):
        if component.is_symlink():
            raise RuntimeError(f"{field}_SYMLINK_FORBIDDEN: {component}")


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def load_and_validate_train_feature_ranking(
    path: Path,
    *,
    entry_run_id: str,
) -> tuple[dict[str, Any], list[str], str]:
    if path.is_symlink() or not path.is_file() or "_latest" in path.name:
        raise RuntimeError(f"FEATURE_RANKING_NOT_IMMUTABLE: {path}")
    try:
        raw_bytes = path.read_bytes()
        ranking = json.loads(raw_bytes)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"FEATURE_RANKING_JSON_INVALID: {path}") from exc
    ranking_file_sha256 = hashlib.sha256(raw_bytes).hexdigest()
    if not isinstance(ranking, dict):
        raise RuntimeError("FEATURE_RANKING_ROOT_NOT_OBJECT")
    if frozenset(ranking) != _RANKING_TOP_LEVEL_KEYS:
        raise RuntimeError(
            "FEATURE_RANKING_SCHEMA_KEYS_INVALID: "
            f"missing={sorted(_RANKING_TOP_LEVEL_KEYS - frozenset(ranking))} "
            f"extra={sorted(frozenset(ranking) - _RANKING_TOP_LEVEL_KEYS)}"
        )
    exact = {
        "schema_version": TRAIN_FEATURE_RANKING_SCHEMA_VERSION,
        "producer": TRAIN_FEATURE_RANKING_PRODUCER,
        "producer_version": TRAIN_FEATURE_RANKING_PRODUCER_VERSION,
        "fit_scope": "train_only",
        "entry_run_id": entry_run_id,
    }
    for field, expected in exact.items():
        if ranking.get(field) != expected:
            raise RuntimeError(
                f"FEATURE_RANKING_{field.upper()}_INVALID: "
                f"observed={ranking.get(field)!r} expected={expected!r}"
            )
    if ranking.get("ranking_order") != TRAIN_FEATURE_RANKING_ORDER:
        raise RuntimeError("FEATURE_RANKING_DETERMINISTIC_ORDER_CONTRACT_INVALID")
    if ranking.get("causality_contract") != TRAIN_FEATURE_CAUSALITY_CONTRACT:
        raise RuntimeError("FEATURE_RANKING_CAUSALITY_CONTRACT_INVALID")

    created = _parse_utc(ranking.get("created_utc"), field="created_utc")
    _require_timestamp_match(path, created, ranking=True)
    train_start = _parse_utc(ranking.get("train_start_utc"), field="train_start_utc")
    train_end = _parse_utc(ranking.get("train_end_utc"), field="train_end_utc")
    source_max = _parse_utc(
        ranking.get("source_time_max_utc"), field="source_time_max_utc"
    )
    target_max = _parse_utc(
        ranking.get("target_time_max_utc"), field="target_time_max_utc"
    )
    if not train_start <= source_max <= train_end:
        raise RuntimeError("FEATURE_RANKING_SOURCE_TIME_OUTSIDE_TRAIN")
    if not train_start <= target_max <= train_end:
        raise RuntimeError("FEATURE_RANKING_TARGET_TIME_OUTSIDE_TRAIN")
    source_hash = _require_sha256(ranking.get("source_sha256"), field="source_sha256")
    target_hash = _require_sha256(ranking.get("target_sha256"), field="target_sha256")
    if source_hash == target_hash:
        raise RuntimeError("FEATURE_RANKING_SOURCE_TARGET_HASH_COLLISION")

    raw_rank_reference = ranking.get("rank_reference")
    if (
        not isinstance(raw_rank_reference, dict)
        or frozenset(raw_rank_reference) != _RANK_REFERENCE_KEYS
    ):
        raise RuntimeError("FEATURE_RANKING_RANK_REFERENCE_SCHEMA_INVALID")
    rank_path = _absolute_without_resolving(
        Path(str(raw_rank_reference.get("path") or ""))
    )
    _require_no_symlink_components(rank_path, field="FEATURE_RANKING_RANK_REFERENCE")
    history_start = parse_state_utc(
        raw_rank_reference.get("history_start_utc"), field="history_start_utc"
    )
    reference = validate_train_rank_reference_lineage_v2(
        rank_path,
        expected_run_id=entry_run_id,
        expected_source_parquet=Path(
            str(raw_rank_reference.get("source_parquet") or "")
        ),
        expected_source_sha256=source_hash,
        expected_history_start_utc=history_start,
        expected_fit_start_utc=train_start,
        expected_fit_end_utc=train_end,
    )
    expected_rank_reference = {
        "path": str(reference.path),
        "sha256": reference.sha256,
        "sidecar_path": str(
            reference.path.with_suffix(reference.path.suffix + ".json")
        ),
        "sidecar_sha256": reference.sidecar_sha256,
        "schema_version": MODEL_NATIVE_TRAIN_RANK_SCHEMA_VERSION,
        "entry_run_id": entry_run_id,
        "source_parquet": str(
            Path(str(reference.sidecar["source_parquet"])).expanduser().resolve()
        ),
        "source_parquet_sha256": source_hash,
        "history_start_utc": history_start.isoformat(),
        "fit_start_utc": reference.fit_start_utc.isoformat(),
        "fit_end_utc": reference.fit_end_utc.isoformat(),
        "fit_row_count": reference.fit_row_count,
        "fit_scope": "train_only",
        "rank_transform": MODEL_NATIVE_RANK_TRANSFORM,
    }
    if raw_rank_reference != expected_rank_reference:
        raise RuntimeError("FEATURE_RANKING_RANK_REFERENCE_METADATA_MISMATCH")

    raw_rows = ranking.get("ranked_features")
    if not isinstance(raw_rows, list) or not raw_rows:
        raise RuntimeError("FEATURE_RANKING_ROWS_MISSING")
    names: list[str] = []
    scored: list[tuple[float, str]] = []
    base = set(MODEL_NATIVE_BASE_FIELDS)
    forbidden = set(FORBIDDEN_LEGACY_BRIDGE_FIELDS)
    allowed_specialists = set(MODEL_NATIVE_TRAINING_SPECIALISTS)
    for index, row in enumerate(raw_rows, start=1):
        if not isinstance(row, dict) or frozenset(row) != _RANKING_ROW_KEYS:
            raise RuntimeError(f"FEATURE_RANKING_ROW_SCHEMA_INVALID: rank={index}")
        if row.get("rank") != index:
            raise RuntimeError(
                "FEATURE_RANKING_RANK_SEQUENCE_INVALID: "
                f"observed={row.get('rank')!r} expected={index}"
            )
        raw_name = row.get("name")
        if not isinstance(raw_name, str) or raw_name != raw_name.strip() or not raw_name:
            raise RuntimeError(f"FEATURE_RANKING_FEATURE_NAME_INVALID: rank={index}")
        name = raw_name
        raw_score = row.get("score")
        if isinstance(raw_score, bool) or not isinstance(raw_score, (int, float)):
            raise RuntimeError(f"FEATURE_RANKING_SCORE_INVALID: rank={index}")
        score = float(raw_score)
        if not math.isfinite(score):
            raise RuntimeError(f"FEATURE_RANKING_SCORE_NONFINITE: rank={index}")
        if name in base:
            raise RuntimeError(f"FEATURE_RANKING_BASE_FIELD_FORBIDDEN: {name}")
        if name in forbidden:
            raise RuntimeError(f"FEATURE_RANKING_LEGACY_BRIDGE_FIELD_FORBIDDEN: {name}")
        if _is_forbidden_leak_name(name):
            raise RuntimeError(f"FEATURE_RANKING_TARGET_OR_LEAK_FIELD_FORBIDDEN: {name}")
        specialist = classify_entry_specialist_feature(name)
        if specialist not in allowed_specialists:
            raise RuntimeError(
                f"FEATURE_RANKING_UNCLASSIFIABLE_FIELD: {name} owner={specialist}"
            )
        names.append(name)
        scored.append((score, name))
    if len(names) != len(set(names)):
        duplicates = sorted({name for name in names if names.count(name) > 1})
        raise RuntimeError(f"FEATURE_RANKING_DUPLICATE_NAMES: {duplicates[:20]}")
    expected_order = sorted(scored, key=lambda item: (-item[0], item[1]))
    if scored != expected_order:
        raise RuntimeError("FEATURE_RANKING_DECLARED_ORDER_OR_TIE_BREAK_INVALID")
    return ranking, names, ranking_file_sha256


def validate_signal_manifest_training_lineage(
    *,
    manifest_path: Path,
    feature_ranking_path: Path,
    expected_run_id: str,
    expected_source_sha256: str,
    expected_train_start_utc: object,
    expected_train_end_utc: object,
) -> dict[str, Any]:
    """Revalidate the immutable signal/ranking lineage at a consumer boundary."""

    run_id = require_entry_run_id(expected_run_id)
    raw_manifest_path = _absolute_without_resolving(Path(manifest_path))
    raw_ranking_path = _absolute_without_resolving(Path(feature_ranking_path))
    _require_no_symlink_components(raw_manifest_path, field="SIGNAL_MANIFEST")
    _require_no_symlink_components(raw_ranking_path, field="FEATURE_RANKING")
    manifest_path = raw_manifest_path.resolve()
    ranking_path = raw_ranking_path.resolve()
    if not manifest_path.is_file() or "_latest" in manifest_path.name:
        raise RuntimeError(f"SIGNAL_MANIFEST_NOT_IMMUTABLE: {manifest_path}")
    try:
        manifest_bytes = manifest_path.read_bytes()
        manifest = json.loads(manifest_bytes)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"SIGNAL_MANIFEST_JSON_INVALID: {manifest_path}") from exc
    if not isinstance(manifest, Mapping):
        raise RuntimeError("SIGNAL_MANIFEST_ROOT_NOT_OBJECT")

    exact_top_level = {
        "schema_version": SIGNAL_MANIFEST_SCHEMA_VERSION,
        "producer": SIGNAL_MANIFEST_PRODUCER,
        "producer_version": SIGNAL_MANIFEST_PRODUCER_VERSION,
        "entry_run_id": run_id,
        "decision": "READY_FOR_MODEL_NATIVE_SEQ513_DATASET_REBUILD_MANIFEST",
        "manifest_only": True,
        "training_allowed": False,
        "shadow_live_promotion_allowed": False,
        "foundation_structure_feature_version": FOUNDATION_STRUCTURE_FEATURE_VERSION,
        "foundation_structure_feature_count": len(FOUNDATION_STRUCTURE_FEATURE_NAMES),
        "foundation_structure_missing_feature_count": 0,
        "foundation_structure_all_required_selected": True,
        "structural_aux_label_signal_contract": (
            MODEL_NATIVE_STRUCTURAL_AUX_LABEL_SIGNAL_CONTRACT
        ),
        "pretrain_polarity_signal_contract": (
            MODEL_NATIVE_PRETRAIN_POLARITY_SIGNAL_CONTRACT
        ),
    }
    for field, expected in exact_top_level.items():
        if manifest.get(field) != expected:
            raise RuntimeError(
                f"SIGNAL_MANIFEST_{field.upper()}_INVALID: "
                f"observed={manifest.get(field)!r} expected={expected!r}"
            )
    if str(manifest.get("json_path") or "") != str(manifest_path):
        raise RuntimeError("SIGNAL_MANIFEST_SELF_PATH_MISMATCH")
    created = _parse_utc(manifest.get("created_utc"), field="manifest_created_utc")
    _require_timestamp_match(manifest_path, created, ranking=False)
    signal_contract = require_model_native_manifest(
        manifest, context="MODEL_NATIVE_SIGNAL_LINEAGE"
    )

    ranking, ranked_names, ranking_sha256 = load_and_validate_train_feature_ranking(
        ranking_path,
        entry_run_id=run_id,
    )
    expected_source_sha256 = _require_sha256(
        expected_source_sha256,
        field="expected_source_sha256",
    )
    expected_train_start = _parse_utc(
        expected_train_start_utc,
        field="expected_train_start_utc",
    )
    expected_train_end = _parse_utc(
        expected_train_end_utc,
        field="expected_train_end_utc",
    )
    ranking_train_start = _parse_utc(
        ranking["train_start_utc"], field="train_start_utc"
    )
    ranking_train_end = _parse_utc(ranking["train_end_utc"], field="train_end_utc")
    if ranking_train_start != expected_train_start or ranking_train_end != expected_train_end:
        raise RuntimeError(
            "FEATURE_RANKING_TRAIN_WINDOW_MISMATCH: "
            f"ranking={ranking_train_start.isoformat()}..{ranking_train_end.isoformat()} "
            f"expected={expected_train_start.isoformat()}..{expected_train_end.isoformat()}"
        )
    if ranking["source_sha256"] != expected_source_sha256:
        raise RuntimeError(
            "FEATURE_RANKING_SOURCE_SHA256_MISMATCH: "
            f"ranking={ranking['source_sha256']} expected={expected_source_sha256}"
        )
    ranking_created = _parse_utc(ranking["created_utc"], field="created_utc")
    if created <= ranking_created:
        raise RuntimeError("SIGNAL_MANIFEST_TIMESTAMP_NOT_AFTER_RANKING")

    mandatory_set = set(MODEL_NATIVE_MANDATORY_SELECTED_FIELDS)
    eligible_ranked_names = [name for name in ranked_names if name not in mandatory_set]
    ranked_remainder = eligible_ranked_names[:MODEL_NATIVE_RANKED_REMAINDER_FEATURE_COUNT]
    expected_selected = [*MODEL_NATIVE_MANDATORY_SELECTED_FIELDS, *ranked_remainder]
    expected_ranking_metadata = {
        "path": str(ranking_path),
        "sha256": ranking_sha256,
        "schema_version": ranking["schema_version"],
        "created_utc": ranking["created_utc"],
        "producer": ranking["producer"],
        "producer_version": ranking["producer_version"],
        "fit_scope": ranking["fit_scope"],
        "train_start_utc": ranking["train_start_utc"],
        "train_end_utc": ranking["train_end_utc"],
        "source_sha256": ranking["source_sha256"],
        "target_sha256": ranking["target_sha256"],
        "ranking_order": ranking["ranking_order"],
        "causality_contract": ranking["causality_contract"],
        "rank_reference": ranking["rank_reference"],
        "ranked_feature_count": len(ranked_names),
        "eligible_ranked_remainder_count": len(eligible_ranked_names),
        "mandatory_names_in_ranking_count": len(ranked_names)
        - len(eligible_ranked_names),
        "ranked_feature_order_sha256": _sha256_json(ranked_names),
    }
    if manifest.get("feature_ranking") != expected_ranking_metadata:
        raise RuntimeError("SIGNAL_MANIFEST_FEATURE_RANKING_METADATA_MISMATCH")
    if manifest.get("selected_features") != expected_selected:
        raise RuntimeError("SIGNAL_MANIFEST_SELECTED_FIELDS_NOT_DERIVED_FROM_RANKING")
    if manifest.get("ranked_remainder_features") != ranked_remainder:
        raise RuntimeError("SIGNAL_MANIFEST_RANKED_REMAINDER_MISMATCH")
    if manifest.get("selected_fields_sha256") != _sha256_json(expected_selected):
        raise RuntimeError("SIGNAL_MANIFEST_SELECTED_FIELDS_SHA256_MISMATCH")
    if manifest.get("ranked_remainder_fields_sha256") != _sha256_json(ranked_remainder):
        raise RuntimeError("SIGNAL_MANIFEST_RANKED_REMAINDER_SHA256_MISMATCH")
    if _sha256_file(ranking_path) != ranking_sha256:
        raise RuntimeError("FEATURE_RANKING_CHANGED_DURING_LINEAGE_VALIDATION")
    return {
        "manifest_path": str(manifest_path),
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "feature_ranking_path": str(ranking_path),
        "feature_ranking_sha256": ranking_sha256,
        "entry_run_id": run_id,
        "source_sha256": expected_source_sha256,
        "train_start_utc": expected_train_start.isoformat(),
        "train_end_utc": expected_train_end.isoformat(),
        "model_native_signal_contract": signal_contract,
    }


def _write_fresh_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags, 0o644)
    except FileExistsError as exc:
        raise RuntimeError(f"SIGNAL_MANIFEST_OUTPUT_NOT_FRESH: {path}") from exc
    try:
        view = memoryview(encoded)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError(f"short write: {path}")
            view = view[written:]
        os.fsync(descriptor)
    except Exception:
        path.unlink(missing_ok=True)
        raise
    finally:
        os.close(descriptor)


def run(args: argparse.Namespace) -> dict[str, Any]:
    entry_run_id = require_entry_run_id(getattr(args, "run_id", None))
    raw_ranking_path = _absolute_without_resolving(Path(args.feature_ranking_json))
    raw_out = _absolute_without_resolving(Path(args.out))
    _require_no_symlink_components(raw_ranking_path, field="FEATURE_RANKING")
    _require_no_symlink_components(raw_out, field="SIGNAL_MANIFEST_OUTPUT")
    ranking_path = raw_ranking_path.resolve()
    out = raw_out.resolve()
    if out.exists():
        raise RuntimeError(f"SIGNAL_MANIFEST_OUTPUT_NOT_FRESH: {out}")
    created = _timestamp_from_name(out, _EVENT_NAME_RE, field="SIGNAL_MANIFEST")

    ranking, ranked_names, ranking_file_sha256 = load_and_validate_train_feature_ranking(
        ranking_path,
        entry_run_id=entry_run_id,
    )
    ranking_created = _parse_utc(ranking["created_utc"], field="created_utc")
    now = _utc_now()
    if created <= ranking_created:
        raise RuntimeError(
            "SIGNAL_MANIFEST_TIMESTAMP_NOT_AFTER_RANKING: "
            f"manifest={created.isoformat()} ranking={ranking_created.isoformat()}"
        )
    past_skew = (now - created).total_seconds()
    future_skew = (created - now).total_seconds()
    if past_skew > MAX_OUTPUT_PAST_SKEW_SECONDS:
        raise RuntimeError(
            "SIGNAL_MANIFEST_TIMESTAMP_STALE: "
            f"manifest={created.isoformat()} now={now.isoformat()}"
        )
    if future_skew > MAX_OUTPUT_FUTURE_SKEW_SECONDS:
        raise RuntimeError(
            "SIGNAL_MANIFEST_TIMESTAMP_FUTURE: "
            f"manifest={created.isoformat()} now={now.isoformat()}"
        )
    mandatory_set = set(MODEL_NATIVE_MANDATORY_SELECTED_FIELDS)
    eligible_ranked_names = [name for name in ranked_names if name not in mandatory_set]
    if len(eligible_ranked_names) < MODEL_NATIVE_RANKED_REMAINDER_FEATURE_COUNT:
        raise RuntimeError(
            "FEATURE_RANKING_ELIGIBLE_REMAINDER_INSUFFICIENT: "
            f"observed={len(eligible_ranked_names)} "
            f"required={MODEL_NATIVE_RANKED_REMAINDER_FEATURE_COUNT}"
        )
    ranked_remainder = eligible_ranked_names[
        :MODEL_NATIVE_RANKED_REMAINDER_FEATURE_COUNT
    ]
    selected = [*MODEL_NATIVE_MANDATORY_SELECTED_FIELDS, *ranked_remainder]
    if len(selected) != MODEL_NATIVE_SELECTED_FEATURE_COUNT:
        raise RuntimeError("SIGNAL_MANIFEST_SELECTED_FEATURE_COUNT_INTERNAL_MISMATCH")
    foundation_missing = [
        name for name in FOUNDATION_STRUCTURE_FEATURE_NAMES if name not in selected
    ]
    if foundation_missing:
        raise RuntimeError(
            "SIGNAL_MANIFEST_FOUNDATION_MANDATORY_FIELDS_MISSING: "
            f"{foundation_missing[:20]} total={len(foundation_missing)}"
        )
    signal_contract = model_native_signal_contract_metadata(selected)
    grouped = group_features_by_specialist(selected)
    if grouped.get("unmapped") or grouped.get("forbidden_legacy_bridge"):
        raise RuntimeError(
            "SIGNAL_MANIFEST_SELECTED_ROUTING_INVALID: "
            f"unmapped={grouped.get('unmapped')} "
            f"forbidden={grouped.get('forbidden_legacy_bridge')}"
        )
    if any(not grouped.get(name) for name in MODEL_NATIVE_TRAINING_SPECIALISTS):
        raise RuntimeError("SIGNAL_MANIFEST_EIGHT_SPECIALIST_COVERAGE_INVALID")

    mandatory_metadata = model_native_mandatory_full_stack_metadata()
    family_counts = {
        family: len(features)
        for family, features in MODEL_NATIVE_MANDATORY_FAMILY_FEATURES
    }
    manifest = {
        "schema_version": SIGNAL_MANIFEST_SCHEMA_VERSION,
        "created_utc": created.isoformat(),
        "json_path": str(out),
        "producer": SIGNAL_MANIFEST_PRODUCER,
        "producer_version": SIGNAL_MANIFEST_PRODUCER_VERSION,
        "entry_run_id": entry_run_id,
        "manifest_variant": MODEL_NATIVE_CONTRACT_MODE,
        "decision": "READY_FOR_MODEL_NATIVE_SEQ513_DATASET_REBUILD_MANIFEST",
        "manifest_only": True,
        "ranking_artifact_is_upstream_prerequisite_not_runtime_authority": True,
        "selected_features": selected,
        "selected_feature_count": MODEL_NATIVE_SELECTED_FEATURE_COUNT,
        "base_signal_feature_count": MODEL_NATIVE_BASE_SIGNAL_DIM,
        "expected_seq_snap_width": MODEL_NATIVE_SIGNAL_DIM,
        "model_native_signal_contract": signal_contract,
        "structural_aux_label_signal_contract": (
            MODEL_NATIVE_STRUCTURAL_AUX_LABEL_SIGNAL_CONTRACT
        ),
        "pretrain_polarity_signal_contract": (
            MODEL_NATIVE_PRETRAIN_POLARITY_SIGNAL_CONTRACT
        ),
        "mandatory_full_stack": mandatory_metadata,
        "mandatory_full_stack_sha256": MODEL_NATIVE_MANDATORY_FULL_STACK_SHA256,
        "mandatory_selected_feature_count": MODEL_NATIVE_MANDATORY_SELECTED_FEATURE_COUNT,
        "foundation_structure_feature_version": FOUNDATION_STRUCTURE_FEATURE_VERSION,
        "foundation_structure_feature_count": len(FOUNDATION_STRUCTURE_FEATURE_NAMES),
        "foundation_structure_missing_feature_count": 0,
        "foundation_structure_all_required_selected": True,
        "ranked_remainder_feature_count": MODEL_NATIVE_RANKED_REMAINDER_FEATURE_COUNT,
        "ranked_remainder_features": ranked_remainder,
        "ranked_remainder_fields_sha256": _sha256_json(ranked_remainder),
        "selected_fields_sha256": _sha256_json(selected),
        "features_by_specialist": grouped,
        "required_training_specialists": list(MODEL_NATIVE_TRAINING_SPECIALISTS),
        "feature_counts_by_specialist": {
            specialist: len(grouped[specialist])
            for specialist in MODEL_NATIVE_TRAINING_SPECIALISTS
        },
        "smart_layers_included": True,
        "smart_layer_feature_counts": family_counts,
        "source_feature_counts": {
            "smart_candidate_layers": MODEL_NATIVE_MANDATORY_SELECTED_FEATURE_COUNT,
            "mandatory_full_stack": MODEL_NATIVE_MANDATORY_SELECTED_FEATURE_COUNT,
            "ranked_remainder": MODEL_NATIVE_RANKED_REMAINDER_FEATURE_COUNT,
        },
        "feature_ranking": {
            "path": str(ranking_path),
            "sha256": ranking_file_sha256,
            "schema_version": ranking["schema_version"],
            "created_utc": ranking["created_utc"],
            "producer": ranking["producer"],
            "producer_version": ranking["producer_version"],
            "fit_scope": ranking["fit_scope"],
            "train_start_utc": ranking["train_start_utc"],
            "train_end_utc": ranking["train_end_utc"],
            "source_sha256": ranking["source_sha256"],
            "target_sha256": ranking["target_sha256"],
            "ranking_order": ranking["ranking_order"],
            "causality_contract": ranking["causality_contract"],
            "rank_reference": ranking["rank_reference"],
            "ranked_feature_count": len(ranked_names),
            "eligible_ranked_remainder_count": len(eligible_ranked_names),
            "mandatory_names_in_ranking_count": len(ranked_names)
            - len(eligible_ranked_names),
            "ranked_feature_order_sha256": _sha256_json(ranked_names),
        },
        "dataset_rebuild_required_before_training": True,
        "training_allowed": False,
        "shadow_live_promotion_allowed": False,
    }
    if _sha256_file(ranking_path) != ranking_file_sha256:
        raise RuntimeError("FEATURE_RANKING_CHANGED_DURING_MANIFEST_BUILD")
    _write_fresh_json(out, manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-ranking-json", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    return parser


def main() -> int:
    manifest = run(build_parser().parse_args())
    print(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
