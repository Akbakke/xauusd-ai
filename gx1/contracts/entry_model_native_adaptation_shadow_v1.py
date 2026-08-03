"""Paired zero-order incumbent/challenger shadow evidence for Entry promotion."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from gx1.contracts.entry_model_native_adaptation_drift_v1 import (
    MODEL_NATIVE_ADAPTATION_DRIFT_MAX_EVENT_AGE_SECONDS,
    MODEL_NATIVE_ADAPTATION_DRIFT_MAX_OBSERVATION_LAG_SECONDS,
    MODEL_NATIVE_ADAPTATION_DRIFT_MAX_OBSERVATION_WINDOW_SECONDS,
    MODEL_NATIVE_ADAPTATION_DRIFT_MIN_ROWS,
    MODEL_NATIVE_ADAPTATION_DRIFT_MIN_ROWS_PER_DIRECTION,
    MODEL_NATIVE_ADAPTATION_DRIFT_PNL_Z,
    ModelNativeAdaptationDriftError,
    adaptation_side_metrics,
    recompute_spread_aware_pnl_bps,
    require_adaptation_bundle_identity,
)
from gx1.contracts.entry_model_native_sizing_calibration_v1 import (
    ModelNativeSizingContractError,
    require_immutable_json_binding,
    sha256_file,
)


MODEL_NATIVE_ADAPTATION_SHADOW_SCHEMA_VERSION = (
    "entry_model_native_adaptation_shadow_v1"
)
MODEL_NATIVE_ADAPTATION_SHADOW_EVENT_PREFIX = (
    "ENTRY_MODEL_NATIVE_ADAPTATION_SHADOW"
)
MODEL_NATIVE_ADAPTATION_SHADOW_CONTRACT = (
    "paired_incumbent_challenger_zero_order_shadow_v1"
)
MODEL_NATIVE_ADAPTATION_SHADOW_MIN_CONTEXT_ROWS = 32

MODEL_NATIVE_ADAPTATION_SHADOW_COLUMNS = frozenset(
    {
        "time",
        "candidate_direction_index",
        "candidate_p_long",
        "candidate_p_short",
        "candidate_p_flat",
        "incumbent_direction_index",
        "incumbent_p_long",
        "incumbent_p_short",
        "incumbent_p_flat",
        "entry_bid",
        "entry_ask",
        "exit_bid",
        "exit_ask",
        "candidate_realized_pnl_bps",
        "incumbent_realized_pnl_bps",
        "session",
        "vol_regime",
        "candidate_bundle_metadata_sha256",
        "candidate_model_state_dict_sha256",
        "incumbent_bundle_metadata_sha256",
        "incumbent_model_state_dict_sha256",
        "outcome_source_id",
        "order_submitted",
    }
)

_EVENT_KEYS = frozenset(
    {
        "schema_version",
        "created_utc",
        "json_path",
        "decision",
        "failures",
        "shadow_contract",
        "incumbent_bundle",
        "candidate_bundle",
        "paired_rows",
        "coverage",
        "global_metrics",
        "context_metrics",
    }
)
_SOURCE_KEYS = frozenset({"path", "sha256"})
_COVERAGE_KEYS = frozenset(
    {
        "rows",
        "candidate_long_rows",
        "candidate_short_rows",
        "candidate_flat_rows",
        "incumbent_long_rows",
        "incumbent_short_rows",
        "incumbent_flat_rows",
        "first_utc",
        "last_utc",
        "utc_ns_sha256",
        "distinct_outcome_source_ids",
        "order_submission_count",
    }
)
_SIDE_KEYS = frozenset(
    {
        "rows",
        "losing_rows",
        "loss_rate",
        "loss_wilson_upper_95",
        "mean_pnl_bps",
        "mean_pnl_lower_95_bps",
        "decision",
    }
)
_GLOBAL_KEYS = frozenset(
    {
        "candidate_long",
        "candidate_short",
        "candidate_mean_pnl_bps",
        "incumbent_mean_pnl_bps",
        "paired_delta_mean_pnl_bps",
        "paired_delta_mean_lower_95_bps",
        "decision",
    }
)
_CONTEXT_KEYS = frozenset(
    {
        "field",
        "value",
        "candidate_direction_index",
        "rows",
        "paired_delta_mean_pnl_bps",
        "paired_delta_mean_lower_95_bps",
        "decision",
    }
)


class ModelNativeAdaptationShadowError(RuntimeError):
    """Paired shadow evidence is absent, stale, malformed or not superior."""


def _fail(context: str, detail: str) -> None:
    raise ModelNativeAdaptationShadowError(f"[{context}_INVALID] {detail}")


def _exact_keys(
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


def _utc(value: Any, *, context: str) -> pd.Timestamp:
    try:
        parsed = pd.Timestamp(value)
    except Exception as exc:
        raise ModelNativeAdaptationShadowError(
            f"[{context}_INVALID] invalid UTC timestamp"
        ) from exc
    if parsed.tz is None or parsed.utcoffset() is None:
        _fail(context, "timestamp must be timezone-aware UTC")
    return parsed.tz_convert("UTC")


def _source_binding(value: Any, *, context: str) -> dict[str, str]:
    observed = _exact_keys(value, _SOURCE_KEYS, context=context)
    raw = Path(str(observed["path"] or "")).expanduser()
    if (
        not raw.is_absolute()
        or raw.is_symlink()
        or raw.suffix != ".parquet"
        or "latest" in raw.name.lower()
    ):
        _fail(context, "paired rows path must be absolute immutable parquet")
    path = raw.resolve()
    expected_sha = str(observed["sha256"] or "").strip().lower()
    if len(expected_sha) != 64 or any(
        char not in "0123456789abcdef" for char in expected_sha
    ):
        _fail(context, "paired rows SHA-256 is invalid")
    if not path.is_file() or sha256_file(path) != expected_sha:
        _fail(context, "paired rows are missing or changed")
    return {"path": str(path), "sha256": expected_sha}


def _directions_and_probabilities(
    frame: pd.DataFrame,
    *,
    owner: str,
    context: str,
) -> tuple[np.ndarray, np.ndarray]:
    numeric = pd.to_numeric(
        frame[f"{owner}_direction_index"], errors="coerce"
    ).to_numpy(np.float64)
    if (
        not np.isfinite(numeric).all()
        or not np.array_equal(numeric, numeric.astype(np.int64))
        or not np.isin(numeric.astype(np.int64), [0, 1, 2]).all()
    ):
        _fail(context, f"{owner} direction is not exact LONG/SHORT/FLAT")
    directions = numeric.astype(np.int64)
    probabilities = frame[
        [f"{owner}_p_long", f"{owner}_p_short", f"{owner}_p_flat"]
    ].apply(pd.to_numeric, errors="coerce").to_numpy(np.float64)
    if (
        not np.isfinite(probabilities).all()
        or np.any(probabilities < 0.0)
        or np.any(probabilities > 1.0)
        or not np.allclose(probabilities.sum(axis=1), 1.0, rtol=0.0, atol=1e-6)
        or np.any(
            np.count_nonzero(
                probabilities == np.max(probabilities, axis=1, keepdims=True),
                axis=1,
            )
            != 1
        )
        or not np.array_equal(np.argmax(probabilities, axis=1), directions)
    ):
        _fail(context, f"{owner} probabilities do not prove exact unique model argmax")
    return directions, probabilities


def _mean_lower_95(values: np.ndarray) -> tuple[float, float]:
    mean = float(np.mean(values))
    lower = mean - MODEL_NATIVE_ADAPTATION_DRIFT_PNL_Z * float(
        np.std(values, ddof=1) / math.sqrt(len(values))
    )
    return mean, lower


def recompute_adaptation_shadow_evidence(
    *,
    paired_rows: pd.DataFrame,
    incumbent_bundle: Mapping[str, str],
    candidate_bundle: Mapping[str, str],
    event_created_utc: Any,
    context: str,
) -> dict[str, Any]:
    """Recompute absolute candidate edge and paired incumbent improvement."""

    if set(paired_rows.columns) != set(MODEL_NATIVE_ADAPTATION_SHADOW_COLUMNS):
        _fail(context, "paired shadow columns mismatch")
    if len(paired_rows) < MODEL_NATIVE_ADAPTATION_DRIFT_MIN_ROWS:
        _fail(context, "paired shadow support is insufficient")
    times = pd.to_datetime(paired_rows["time"], utc=True, errors="coerce")
    if times.isna().any() or times.duplicated().any() or not times.is_monotonic_increasing:
        _fail(context, "paired shadow times must be unique monotonic UTC")
    created = _utc(event_created_utc, context=f"{context}.created_utc")
    if times.iloc[-1] > created:
        _fail(context, "paired shadow rows cannot be newer than the event")
    if (times.iloc[-1] - times.iloc[0]).total_seconds() > MODEL_NATIVE_ADAPTATION_DRIFT_MAX_OBSERVATION_WINDOW_SECONDS:
        _fail(context, "paired shadow window is too wide")
    if (created - times.iloc[-1]).total_seconds() > MODEL_NATIVE_ADAPTATION_DRIFT_MAX_OBSERVATION_LAG_SECONDS:
        _fail(context, "paired shadow rows are stale")

    candidate_directions, _ = _directions_and_probabilities(
        paired_rows, owner="candidate", context=context
    )
    incumbent_directions, _ = _directions_and_probabilities(
        paired_rows, owner="incumbent", context=context
    )
    candidate_counts = [
        int(np.count_nonzero(candidate_directions == index)) for index in range(3)
    ]
    if min(candidate_counts) < MODEL_NATIVE_ADAPTATION_DRIFT_MIN_ROWS_PER_DIRECTION:
        _fail(context, "candidate LONG/SHORT/FLAT shadow support is insufficient")

    prices = paired_rows[["entry_bid", "entry_ask", "exit_bid", "exit_ask"]].apply(
        pd.to_numeric, errors="coerce"
    ).to_numpy(np.float64)
    try:
        candidate_expected = recompute_spread_aware_pnl_bps(
            candidate_directions, *prices.T, context=f"{context}.candidate_prices"
        )
        incumbent_expected = recompute_spread_aware_pnl_bps(
            incumbent_directions, *prices.T, context=f"{context}.incumbent_prices"
        )
    except ModelNativeAdaptationDriftError as exc:
        raise ModelNativeAdaptationShadowError(str(exc)) from exc
    candidate_pnl = pd.to_numeric(
        paired_rows["candidate_realized_pnl_bps"], errors="coerce"
    ).to_numpy(np.float64)
    incumbent_pnl = pd.to_numeric(
        paired_rows["incumbent_realized_pnl_bps"], errors="coerce"
    ).to_numpy(np.float64)
    if (
        not np.isfinite(candidate_pnl).all()
        or not np.isfinite(incumbent_pnl).all()
        or not np.allclose(candidate_pnl, candidate_expected, rtol=0.0, atol=1e-9)
        or not np.allclose(incumbent_pnl, incumbent_expected, rtol=0.0, atol=1e-9)
    ):
        _fail(context, "paired PnL differs from bid/ask recomputation")

    for field, expected in (
        ("candidate_bundle_metadata_sha256", candidate_bundle["bundle_metadata_sha256"]),
        ("candidate_model_state_dict_sha256", candidate_bundle["model_state_dict_sha256"]),
        ("incumbent_bundle_metadata_sha256", incumbent_bundle["bundle_metadata_sha256"]),
        ("incumbent_model_state_dict_sha256", incumbent_bundle["model_state_dict_sha256"]),
    ):
        if set(paired_rows[field].astype(str).str.lower()) != {expected}:
            _fail(context, f"{field} differs from exact bundle identity")
    for field in ("session", "vol_regime", "outcome_source_id"):
        values = paired_rows[field].astype(str)
        if values.str.strip().ne(values).any() or values.str.len().eq(0).any():
            _fail(context, f"{field} contains noncanonical values")
    if paired_rows["outcome_source_id"].astype(str).duplicated().any():
        _fail(context, "outcome_source_id must be unique")
    if not paired_rows["order_submitted"].map(
        lambda value: isinstance(value, (bool, np.bool_))
    ).all() or paired_rows["order_submitted"].astype(bool).any():
        _fail(context, "paired shadow evidence must submit zero orders")

    delta = candidate_pnl - incumbent_pnl
    delta_mean, delta_lower = _mean_lower_95(delta)
    candidate_long = adaptation_side_metrics(
        candidate_pnl, candidate_directions, 0
    )
    candidate_short = adaptation_side_metrics(
        candidate_pnl, candidate_directions, 1
    )
    global_failures: list[str] = []
    if candidate_long["decision"] != "PASS":
        global_failures.append("candidate_long_absolute_edge")
    if candidate_short["decision"] != "PASS":
        global_failures.append("candidate_short_absolute_edge")
    if delta_lower <= 0.0:
        global_failures.append("candidate_paired_delta_lower_95_not_positive")

    context_rows: list[dict[str, Any]] = []
    context_failures: list[str] = []
    for field in ("session", "vol_regime"):
        for value in sorted(set(paired_rows[field].astype(str))):
            for direction in (0, 1, 2):
                mask = (
                    (paired_rows[field].astype(str) == value).to_numpy()
                    & (candidate_directions == direction)
                )
                rows = int(np.count_nonzero(mask))
                if rows < MODEL_NATIVE_ADAPTATION_SHADOW_MIN_CONTEXT_ROWS:
                    continue
                mean, lower = _mean_lower_95(delta[mask])
                passed = lower > 0.0
                context_rows.append(
                    {
                        "field": field,
                        "value": value,
                        "candidate_direction_index": direction,
                        "rows": rows,
                        "paired_delta_mean_pnl_bps": mean,
                        "paired_delta_mean_lower_95_bps": lower,
                        "decision": "PASS" if passed else "FAIL",
                    }
                )
                if not passed:
                    context_failures.append(
                        f"paired_context:{field}={value}:direction={direction}"
                    )
    if {row["field"] for row in context_rows} != {"session", "vol_regime"}:
        context_failures.append("paired_context:support_missing")

    failures = global_failures + context_failures
    coverage = {
        "rows": int(len(paired_rows)),
        "candidate_long_rows": candidate_counts[0],
        "candidate_short_rows": candidate_counts[1],
        "candidate_flat_rows": candidate_counts[2],
        "incumbent_long_rows": int(np.count_nonzero(incumbent_directions == 0)),
        "incumbent_short_rows": int(np.count_nonzero(incumbent_directions == 1)),
        "incumbent_flat_rows": int(np.count_nonzero(incumbent_directions == 2)),
        "first_utc": times.iloc[0].isoformat(),
        "last_utc": times.iloc[-1].isoformat(),
        "utc_ns_sha256": hashlib.sha256(
            times.astype("int64").to_numpy(np.int64).tobytes()
        ).hexdigest(),
        "distinct_outcome_source_ids": int(
            paired_rows["outcome_source_id"].astype(str).nunique()
        ),
        "order_submission_count": 0,
    }
    global_metrics = {
        "candidate_long": candidate_long,
        "candidate_short": candidate_short,
        "candidate_mean_pnl_bps": float(np.mean(candidate_pnl)),
        "incumbent_mean_pnl_bps": float(np.mean(incumbent_pnl)),
        "paired_delta_mean_pnl_bps": delta_mean,
        "paired_delta_mean_lower_95_bps": delta_lower,
        "decision": "PASS" if not global_failures else "FAIL",
    }
    return {
        "decision": "PASS" if not failures else "FAIL",
        "failures": failures,
        "coverage": coverage,
        "global_metrics": global_metrics,
        "context_metrics": context_rows,
    }


def load_bound_adaptation_shadow_evidence(
    binding: Mapping[str, Any] | Any,
    *,
    incumbent_bundle: Mapping[str, str],
    candidate_bundle: Mapping[str, str],
    context: str,
    now_utc: Any | None = None,
) -> tuple[dict[str, Any], dict[str, str]]:
    """Load newest paired shadow evidence and independently recompute it."""

    try:
        canonical_binding = require_immutable_json_binding(
            binding,
            event_prefix=MODEL_NATIVE_ADAPTATION_SHADOW_EVENT_PREFIX,
            context=f"{context}.binding",
            verify_file=True,
        )
    except ModelNativeSizingContractError as exc:
        raise ModelNativeAdaptationShadowError(str(exc)) from exc
    path = Path(canonical_binding["json_path"])
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ModelNativeAdaptationShadowError(
            f"[{context}_INVALID] shadow event is unreadable"
        ) from exc
    event = _exact_keys(raw, _EVENT_KEYS, context=context)
    if (
        event["schema_version"] != MODEL_NATIVE_ADAPTATION_SHADOW_SCHEMA_VERSION
        or event["shadow_contract"] != MODEL_NATIVE_ADAPTATION_SHADOW_CONTRACT
    ):
        _fail(context, "shadow schema or contract mismatch")
    if event["decision"] != "PASS" or event["failures"] != []:
        _fail(context, "paired shadow must be zero-failure PASS")
    if Path(str(event["json_path"] or "")).expanduser().resolve() != path:
        _fail(context, "json_path self-reference mismatch")
    created = _utc(event["created_utc"], context=f"{context}.created_utc")
    now = _utc(
        pd.Timestamp.now(tz="UTC") if now_utc is None else now_utc,
        context=f"{context}.now_utc",
    )
    age = (now - created).total_seconds()
    if age < 0.0 or age > MODEL_NATIVE_ADAPTATION_DRIFT_MAX_EVENT_AGE_SECONDS:
        _fail(context, f"event age_seconds={age} is invalid")
    incumbent = require_adaptation_bundle_identity(
        event["incumbent_bundle"], context=f"{context}.incumbent_bundle"
    )
    candidate = require_adaptation_bundle_identity(
        event["candidate_bundle"], context=f"{context}.candidate_bundle"
    )
    if incumbent != incumbent_bundle or candidate != candidate_bundle:
        _fail(context, "paired shadow bundle identity mismatch")
    source = _source_binding(event["paired_rows"], context=f"{context}.paired_rows")
    rows = pd.read_parquet(source["path"])
    recomputed = recompute_adaptation_shadow_evidence(
        paired_rows=rows,
        incumbent_bundle=incumbent,
        candidate_bundle=candidate,
        event_created_utc=created,
        context=f"{context}.recompute",
    )
    if event["decision"] != recomputed["decision"] or event["failures"] != recomputed["failures"]:
        _fail(context, "reported shadow decision differs from recomputation")
    if _exact_keys(event["coverage"], _COVERAGE_KEYS, context=f"{context}.coverage") != recomputed["coverage"]:
        _fail(context, "reported shadow coverage differs from recomputation")
    global_metrics = _exact_keys(
        event["global_metrics"], _GLOBAL_KEYS, context=f"{context}.global"
    )
    _exact_keys(global_metrics["candidate_long"], _SIDE_KEYS, context=f"{context}.candidate_long")
    _exact_keys(global_metrics["candidate_short"], _SIDE_KEYS, context=f"{context}.candidate_short")
    if global_metrics != recomputed["global_metrics"]:
        _fail(context, "reported shadow metrics differ from recomputation")
    if not isinstance(event["context_metrics"], list):
        _fail(context, "context_metrics must be a list")
    for index, row in enumerate(event["context_metrics"]):
        _exact_keys(row, _CONTEXT_KEYS, context=f"{context}.context[{index}]")
    if event["context_metrics"] != recomputed["context_metrics"]:
        _fail(context, "reported shadow contexts differ from recomputation")
    return event, canonical_binding


__all__ = [
    "MODEL_NATIVE_ADAPTATION_SHADOW_COLUMNS",
    "MODEL_NATIVE_ADAPTATION_SHADOW_CONTRACT",
    "MODEL_NATIVE_ADAPTATION_SHADOW_EVENT_PREFIX",
    "MODEL_NATIVE_ADAPTATION_SHADOW_SCHEMA_VERSION",
    "ModelNativeAdaptationShadowError",
    "load_bound_adaptation_shadow_evidence",
    "recompute_adaptation_shadow_evidence",
]
