"""Offline-only, immutable champion/challenger comparison contract.

This owner deliberately does *not* train a model, collect data, schedule work,
place an order, promote a bundle, or alter any model weights.  It becomes
usable only after two independently materialized rolling-OOS result events
exist.  The comparison itself remains a review artifact: a human must decide
whether to start any later candidate-training workflow.

The contract is intentionally independent of the retired adaptation/shadow
stack.  In particular, it consumes current raw-Q candidate results rather than
retired probability fields, and it refuses the sealed TEST set as a repeated
tuning surface.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from gx1.contracts.gx1_scope_v1 import require_offline_scope
from gx1.contracts.immutable_event_authority_v1 import (
    ImmutableEventAuthorityError,
    require_newest_immutable_event,
    write_immutable_json_event,
)


OFFLINE_CHALLENGER_RESULT_SCHEMA_VERSION = "gx1_offline_walk_forward_candidate_result_v1"
OFFLINE_CHALLENGER_RESULT_EVENT_PREFIX = "GX1_OFFLINE_WALK_FORWARD_CANDIDATE_RESULT"
OFFLINE_CHALLENGER_COMPARISON_SCHEMA_VERSION = "gx1_offline_champion_challenger_v1"
OFFLINE_CHALLENGER_COMPARISON_EVENT_PREFIX = "GX1_OFFLINE_CHAMPION_CHALLENGER"
OFFLINE_CHALLENGER_CONTRACT = "immutable_raw_q_rolling_oos_review_only_v1"

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_CANDIDATE_ID_RE = re.compile(r"^[A-Z0-9][A-Z0-9_.-]{7,127}$")
_BINDING_KEYS = frozenset({"json_path", "sha256"})
_WINDOW_KEYS = frozenset({"start_utc", "end_utc"})
_METRIC_KEYS = frozenset(
    {
        "net_pnl_bps",
        "win_rate",
        "max_drawdown_loss_bps",
        "trade_count",
        "mean_mae_bps",
        "mean_mfe_bps",
        "mae_before_mfe_rate",
    }
)
_RESULT_KEYS = frozenset(
    {
        "schema_version",
        "created_utc",
        "json_path",
        "decision",
        "failures",
        "contract",
        "candidate_id",
        "bundle_sha256",
        "feature_contract_sha256",
        "target_contract_sha256",
        "decision_contract_sha256",
        "cost_model_sha256",
        "training_window",
        "evaluation_window",
        "evaluation_scope",
        "test_seal_used",
        "metrics",
        "activation_authority",
        "promotion_allowed",
        "online_weight_updates_allowed",
        "background_scheduler_allowed",
    }
)
_COMPARISON_KEYS = frozenset(
    {
        "schema_version",
        "created_utc",
        "json_path",
        "decision",
        "failures",
        "contract",
        "champion_result",
        "challenger_result",
        "evaluation_window",
        "metric_deltas",
        "review_required",
        "activation_authority",
        "promotion_allowed",
        "online_weight_updates_allowed",
        "background_scheduler_allowed",
    }
)


class OfflineChampionChallengerError(RuntimeError):
    """Raised when a candidate comparison would be non-causal or non-offline."""


def _fail(context: str, detail: str) -> None:
    raise OfflineChampionChallengerError(f"[{context}_INVALID] {detail}")


def _exact_mapping(
    value: Mapping[str, Any] | Any,
    expected: frozenset[str],
    *,
    context: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        _fail(context, "expected an object")
    observed = dict(value)
    missing = sorted(expected - set(observed))
    unexpected = sorted(set(observed) - expected)
    if missing or unexpected:
        _fail(context, f"exact keys mismatch: missing={missing} unexpected={unexpected}")
    return observed


def _sha256(value: Any, *, context: str) -> str:
    parsed = str(value or "").strip().lower()
    if _SHA256_RE.fullmatch(parsed) is None:
        _fail(context, "must be an exact SHA-256")
    return parsed


def _utc(value: Any, *, context: str) -> datetime:
    if not isinstance(value, str) or not value.strip():
        _fail(context, "must be a non-empty UTC timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise OfflineChampionChallengerError(
            f"[{context}_INVALID] invalid UTC timestamp"
        ) from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        _fail(context, "must be timezone-aware")
    return parsed.astimezone(timezone.utc)


def _read_regular_bytes(path: Path, *, context: str) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise OfflineChampionChallengerError(f"[{context}_INVALID] cannot open file") from exc
    try:
        opened = os.fstat(fd)
        if not stat.S_ISREG(opened.st_mode):
            _fail(context, "path is not a regular file")
        chunks: list[bytes] = []
        while True:
            block = os.read(fd, 1024 * 1024)
            if not block:
                break
            chunks.append(block)
        current = os.stat(path, follow_symlinks=False)
        if (
            opened.st_dev,
            opened.st_ino,
            opened.st_size,
            opened.st_mtime_ns,
        ) != (
            current.st_dev,
            current.st_ino,
            current.st_size,
            current.st_mtime_ns,
        ):
            _fail(context, "file changed while read")
        return b"".join(chunks)
    finally:
        os.close(fd)


def _window(value: Mapping[str, Any] | Any, *, context: str) -> tuple[dict[str, str], datetime, datetime]:
    raw = _exact_mapping(value, _WINDOW_KEYS, context=context)
    start = _utc(raw["start_utc"], context=f"{context}.start_utc")
    end = _utc(raw["end_utc"], context=f"{context}.end_utc")
    if start >= end:
        _fail(context, "start_utc must be before end_utc")
    return (
        {"start_utc": start.isoformat().replace("+00:00", "Z"), "end_utc": end.isoformat().replace("+00:00", "Z")},
        start,
        end,
    )


def _finite_number(value: Any, *, context: str) -> float:
    if isinstance(value, bool):
        _fail(context, "must be numeric, not boolean")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise OfflineChampionChallengerError(f"[{context}_INVALID] must be numeric") from exc
    if not math.isfinite(parsed):
        _fail(context, "must be finite")
    return parsed


def _metrics(value: Mapping[str, Any] | Any, *, context: str) -> dict[str, float | int]:
    raw = _exact_mapping(value, _METRIC_KEYS, context=context)
    metrics: dict[str, float | int] = {
        "net_pnl_bps": _finite_number(raw["net_pnl_bps"], context=f"{context}.net_pnl_bps"),
        "win_rate": _finite_number(raw["win_rate"], context=f"{context}.win_rate"),
        "max_drawdown_loss_bps": _finite_number(
            raw["max_drawdown_loss_bps"],
            context=f"{context}.max_drawdown_loss_bps",
        ),
        "mean_mae_bps": _finite_number(raw["mean_mae_bps"], context=f"{context}.mean_mae_bps"),
        "mean_mfe_bps": _finite_number(raw["mean_mfe_bps"], context=f"{context}.mean_mfe_bps"),
        "mae_before_mfe_rate": _finite_number(
            raw["mae_before_mfe_rate"],
            context=f"{context}.mae_before_mfe_rate",
        ),
    }
    if not 0.0 <= float(metrics["win_rate"]) <= 1.0:
        _fail(context, "win_rate must be in [0, 1]")
    if not 0.0 <= float(metrics["mae_before_mfe_rate"]) <= 1.0:
        _fail(context, "mae_before_mfe_rate must be in [0, 1]")
    count = raw["trade_count"]
    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        _fail(context, "trade_count must be a positive exact integer")
    metrics["trade_count"] = count
    if float(metrics["max_drawdown_loss_bps"]) < 0.0:
        _fail(context, "max_drawdown_loss_bps must be zero or positive")
    if float(metrics["mean_mae_bps"]) < 0.0:
        _fail(context, "mean_mae_bps must be zero or positive")
    if float(metrics["mean_mfe_bps"]) < 0.0:
        _fail(context, "mean_mfe_bps must be zero or positive")
    return metrics


def _binding(value: Mapping[str, Any] | Any, *, context: str) -> tuple[dict[str, str], dict[str, Any]]:
    raw = _exact_mapping(value, _BINDING_KEYS, context=context)
    path = Path(str(raw["json_path"] or "")).expanduser()
    if not path.is_absolute() or path.is_symlink() or path.suffix != ".json":
        _fail(context, "json_path must be an absolute non-symlink JSON file")
    path = path.resolve()
    declared_sha = _sha256(raw["sha256"], context=f"{context}.sha256")
    encoded = _read_regular_bytes(path, context=context)
    if hashlib.sha256(encoded).hexdigest() != declared_sha:
        _fail(context, "bound result SHA-256 mismatch")
    try:
        require_newest_immutable_event(path, OFFLINE_CHALLENGER_RESULT_EVENT_PREFIX)
        payload = json.loads(encoded.decode("utf-8"))
    except (ImmutableEventAuthorityError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise OfflineChampionChallengerError(
            f"[{context}_INVALID] result is not a newest immutable candidate event"
        ) from exc
    if not isinstance(payload, Mapping):
        _fail(context, "result root must be an object")
    return {"json_path": str(path), "sha256": declared_sha}, dict(payload)


def _validated_result(
    value: Mapping[str, Any] | Any,
    *,
    context: str,
) -> tuple[dict[str, str], dict[str, Any], datetime]:
    binding, payload = _binding(value, context=context)
    raw = _exact_mapping(payload, _RESULT_KEYS, context=f"{context}.payload")
    if (
        raw["schema_version"] != OFFLINE_CHALLENGER_RESULT_SCHEMA_VERSION
        or raw["contract"] != OFFLINE_CHALLENGER_CONTRACT
        or raw["decision"] != "PASS"
        or raw["failures"] != []
        or raw["json_path"] != binding["json_path"]
        or raw["evaluation_scope"] != "rolling_oos"
        or raw["test_seal_used"] is not False
        or raw["activation_authority"] is not False
        or raw["promotion_allowed"] is not False
        or raw["online_weight_updates_allowed"] is not False
        or raw["background_scheduler_allowed"] is not False
    ):
        _fail(context, "result must be a passing, review-only rolling-OOS event")
    candidate_id = str(raw["candidate_id"] or "")
    if _CANDIDATE_ID_RE.fullmatch(candidate_id) is None:
        _fail(context, "candidate_id is invalid")
    created = _utc(raw["created_utc"], context=f"{context}.created_utc")
    training_window, _training_start, training_end = _window(
        raw["training_window"], context=f"{context}.training_window"
    )
    evaluation_window, evaluation_start, _evaluation_end = _window(
        raw["evaluation_window"], context=f"{context}.evaluation_window"
    )
    if training_end > evaluation_start:
        _fail(context, "training window reaches into its evaluation window")
    canonical = {
        "candidate_id": candidate_id,
        "bundle_sha256": _sha256(raw["bundle_sha256"], context=f"{context}.bundle_sha256"),
        "feature_contract_sha256": _sha256(raw["feature_contract_sha256"], context=f"{context}.feature_contract_sha256"),
        "target_contract_sha256": _sha256(raw["target_contract_sha256"], context=f"{context}.target_contract_sha256"),
        "decision_contract_sha256": _sha256(raw["decision_contract_sha256"], context=f"{context}.decision_contract_sha256"),
        "cost_model_sha256": _sha256(raw["cost_model_sha256"], context=f"{context}.cost_model_sha256"),
        "training_window": training_window,
        "evaluation_window": evaluation_window,
        "metrics": _metrics(raw["metrics"], context=f"{context}.metrics"),
    }
    return binding, canonical, created


def build_offline_challenger_comparison(
    *,
    champion_result: Mapping[str, Any],
    challenger_result: Mapping[str, Any],
    created_utc: str,
) -> dict[str, Any]:
    """Build a non-authoritative comparison from two immutable OOS results.

    Neither model wins automatically.  The output only establishes that the two
    candidates used identical economics and an exactly identical unseen window,
    then records transparent metric deltas for human review.
    """

    require_offline_scope("offline_oos")
    champion_binding, champion, _champion_created = _validated_result(
        champion_result, context="champion_result"
    )
    challenger_binding, challenger, _challenger_created = _validated_result(
        challenger_result, context="challenger_result"
    )
    if champion["candidate_id"] == challenger["candidate_id"]:
        _fail("comparison", "champion and challenger candidate_id must differ")
    if champion["bundle_sha256"] == challenger["bundle_sha256"]:
        _fail("comparison", "champion and challenger bundle SHA-256 must differ")
    for field in (
        "feature_contract_sha256",
        "target_contract_sha256",
        "decision_contract_sha256",
        "cost_model_sha256",
        "evaluation_window",
    ):
        if champion[field] != challenger[field]:
            _fail("comparison", f"{field} differs between champion and challenger")

    created = _utc(created_utc, context="comparison.created_utc")
    if created < _champion_created or created < _challenger_created:
        _fail("comparison", "comparison predates a bound candidate result")
    deltas = {
        "net_pnl_bps": float(challenger["metrics"]["net_pnl_bps"]) - float(champion["metrics"]["net_pnl_bps"]),
        "win_rate": float(challenger["metrics"]["win_rate"]) - float(champion["metrics"]["win_rate"]),
        "max_drawdown_loss_bps": float(challenger["metrics"]["max_drawdown_loss_bps"]) - float(champion["metrics"]["max_drawdown_loss_bps"]),
        "mean_mae_bps": float(challenger["metrics"]["mean_mae_bps"]) - float(champion["metrics"]["mean_mae_bps"]),
        "mean_mfe_bps": float(challenger["metrics"]["mean_mfe_bps"]) - float(champion["metrics"]["mean_mfe_bps"]),
        "mae_before_mfe_rate": float(challenger["metrics"]["mae_before_mfe_rate"]) - float(champion["metrics"]["mae_before_mfe_rate"]),
        "trade_count": int(challenger["metrics"]["trade_count"]) - int(champion["metrics"]["trade_count"]),
    }
    return {
        "schema_version": OFFLINE_CHALLENGER_COMPARISON_SCHEMA_VERSION,
        "created_utc": created.isoformat().replace("+00:00", "Z"),
        "json_path": "",
        "decision": "READY_FOR_HUMAN_REVIEW",
        "failures": [],
        "contract": OFFLINE_CHALLENGER_CONTRACT,
        "champion_result": {**champion_binding, "candidate": champion},
        "challenger_result": {**challenger_binding, "candidate": challenger},
        "evaluation_window": champion["evaluation_window"],
        "metric_deltas": deltas,
        "review_required": True,
        "activation_authority": False,
        "promotion_allowed": False,
        "online_weight_updates_allowed": False,
        "background_scheduler_allowed": False,
    }


def publish_offline_challenger_comparison(
    *,
    out_dir: Path,
    champion_result: Mapping[str, Any],
    challenger_result: Mapping[str, Any],
    created_utc: str,
) -> tuple[Path, dict[str, Any]]:
    """Atomically publish one immutable, review-only comparison event."""

    report = build_offline_challenger_comparison(
        champion_result=champion_result,
        challenger_result=challenger_result,
        created_utc=created_utc,
    )
    if set(report) != _COMPARISON_KEYS:
        _fail("comparison", "internal report schema drift")
    try:
        return write_immutable_json_event(
            Path(out_dir), OFFLINE_CHALLENGER_COMPARISON_EVENT_PREFIX, report
        )
    except ImmutableEventAuthorityError as exc:
        raise OfflineChampionChallengerError(
            "[comparison_INVALID] immutable publication failed"
        ) from exc


__all__ = [
    "OFFLINE_CHALLENGER_COMPARISON_EVENT_PREFIX",
    "OFFLINE_CHALLENGER_COMPARISON_SCHEMA_VERSION",
    "OFFLINE_CHALLENGER_CONTRACT",
    "OFFLINE_CHALLENGER_RESULT_EVENT_PREFIX",
    "OFFLINE_CHALLENGER_RESULT_SCHEMA_VERSION",
    "OfflineChampionChallengerError",
    "build_offline_challenger_comparison",
    "publish_offline_challenger_comparison",
]
