"""Row-recomputed market/edge drift evidence for model-native Entry.

This is an adaptation trigger, never a direction rule.  It compares one
accepted bundle's immutable TEST reference with settled broker-shadow rows for
the same bundle.  Missing support, stale observations, distribution movement,
or degraded spread-aware outcomes produces ``DRIFT`` and therefore cannot be
used to keep an incumbent launch-admissible.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from gx1.contracts.entry_model_native_sizing_calibration_v1 import (
    ModelNativeSizingContractError,
    require_immutable_json_binding,
    sha256_file,
)
from gx1.contracts.model_native_serve_gate_v1 import (
    direction_pocket_wilson_upper_95,
)


MODEL_NATIVE_ADAPTATION_DRIFT_SCHEMA_VERSION = (
    "entry_model_native_adaptation_drift_v1"
)
MODEL_NATIVE_ADAPTATION_DRIFT_EVENT_PREFIX = (
    "ENTRY_MODEL_NATIVE_ADAPTATION_DRIFT"
)
MODEL_NATIVE_ADAPTATION_DRIFT_CONTRACT = (
    "same_bundle_test_vs_settled_broker_shadow_v1"
)
MODEL_NATIVE_ADAPTATION_DRIFT_STABLE = "STABLE"
MODEL_NATIVE_ADAPTATION_DRIFT_RED = "DRIFT"
MODEL_NATIVE_ADAPTATION_DRIFT_MIN_ROWS = 360
MODEL_NATIVE_ADAPTATION_DRIFT_MIN_ROWS_PER_DIRECTION = 100
MODEL_NATIVE_ADAPTATION_DRIFT_MIN_REFERENCE_CONTEXT_ROWS = 32
MODEL_NATIVE_ADAPTATION_DRIFT_MIN_OBSERVATION_CONTEXT_ROWS = 24
MODEL_NATIVE_ADAPTATION_DRIFT_MAX_JS_DIVERGENCE = 0.10
MODEL_NATIVE_ADAPTATION_DRIFT_MAX_PROBABILITY_MEAN_SHIFT = 0.15
MODEL_NATIVE_ADAPTATION_DRIFT_MAX_OBSERVATION_WINDOW_SECONDS = 30 * 86_400
MODEL_NATIVE_ADAPTATION_DRIFT_MAX_OBSERVATION_LAG_SECONDS = 21_600
MODEL_NATIVE_ADAPTATION_DRIFT_MAX_EVENT_AGE_SECONDS = 86_400
MODEL_NATIVE_ADAPTATION_DRIFT_PNL_Z = 1.959963984540054
MODEL_NATIVE_ADAPTATION_DRIFT_MAX_REALIZED_LOSS_RATE = 0.10
MODEL_NATIVE_ADAPTATION_DRIFT_MAX_REALIZED_LOSS_WILSON_UPPER_95 = 0.15

MODEL_NATIVE_ADAPTATION_DRIFT_COLUMNS = frozenset(
    {
        "time",
        "evidence_scope",
        "model_direction_index",
        "p_long",
        "p_short",
        "p_flat",
        "entry_bid",
        "entry_ask",
        "exit_bid",
        "exit_ask",
        "realized_pnl_bps",
        "session",
        "vol_regime",
        "bundle_metadata_sha256",
        "model_state_dict_sha256",
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
        "drift_contract",
        "bundle_identity",
        "reference_rows",
        "observation_rows",
        "coverage",
        "global_metrics",
        "context_metrics",
    }
)
_BUNDLE_IDENTITY_KEYS = frozenset(
    {
        "bundle_dir",
        "bundle_metadata_path",
        "bundle_metadata_sha256",
        "model_state_dict_path",
        "model_state_dict_sha256",
    }
)
_SOURCE_KEYS = frozenset({"path", "sha256"})
_COVERAGE_KEYS = frozenset(
    {
        "reference_rows",
        "observation_rows",
        "reference_long_rows",
        "reference_short_rows",
        "reference_flat_rows",
        "observation_long_rows",
        "observation_short_rows",
        "observation_flat_rows",
        "reference_first_utc",
        "reference_last_utc",
        "observation_first_utc",
        "observation_last_utc",
        "reference_utc_ns_sha256",
        "observation_utc_ns_sha256",
        "distinct_observation_source_ids",
        "order_submission_count",
    }
)
_SIDE_METRIC_KEYS = frozenset(
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
        "reference_direction_distribution",
        "observation_direction_distribution",
        "direction_distribution_js_divergence",
        "reference_probability_means",
        "observation_probability_means",
        "max_probability_mean_shift",
        "reference_long",
        "reference_short",
        "observation_long",
        "observation_short",
        "decision",
    }
)
_CONTEXT_ROW_KEYS = frozenset(
    {
        "field",
        "value",
        "model_direction_index",
        "reference_rows",
        "observation_rows",
        "observation_losing_rows",
        "observation_loss_rate",
        "observation_loss_wilson_upper_95",
        "observation_mean_pnl_bps",
        "observation_mean_pnl_lower_95_bps",
        "decision",
    }
)


class ModelNativeAdaptationDriftError(RuntimeError):
    """Adaptation drift evidence is absent, stale, or malformed."""


def _fail(context: str, detail: str) -> None:
    raise ModelNativeAdaptationDriftError(f"[{context}_INVALID] {detail}")


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


def _sha(value: Any, *, context: str) -> str:
    parsed = str(value or "").strip().lower()
    if len(parsed) != 64 or any(ch not in "0123456789abcdef" for ch in parsed):
        _fail(context, "not an exact SHA-256")
    return parsed


def _utc(value: Any, *, context: str) -> pd.Timestamp:
    try:
        parsed = pd.Timestamp(value)
    except Exception as exc:
        raise ModelNativeAdaptationDriftError(
            f"[{context}_INVALID] invalid UTC timestamp"
        ) from exc
    if parsed.tz is None or parsed.utcoffset() is None:
        _fail(context, "timestamp must be timezone-aware UTC")
    return parsed.tz_convert("UTC")


def _source_binding(
    value: Mapping[str, Any] | Any,
    *,
    context: str,
) -> dict[str, str]:
    observed = _exact_keys(value, _SOURCE_KEYS, context=context)
    raw_path = Path(str(observed["path"] or "")).expanduser()
    if not raw_path.is_absolute() or raw_path.is_symlink():
        _fail(context, "path must be absolute and non-symlinked")
    if raw_path.suffix != ".parquet" or "latest" in raw_path.name.lower():
        _fail(context, "path must be an immutable parquet, never latest")
    path = raw_path.resolve()
    expected_sha = _sha(observed["sha256"], context=f"{context}.sha256")
    if not path.is_file():
        _fail(context, f"bound file missing: {path}")
    if sha256_file(path) != expected_sha:
        _fail(context, f"bound file hash mismatch: {path}")
    return {"path": str(path), "sha256": expected_sha}


def require_adaptation_bundle_identity(
    value: Mapping[str, Any] | Any,
    *,
    context: str,
) -> dict[str, str]:
    """Require exact bundle metadata/model bytes for drift evidence."""

    observed = _exact_keys(value, _BUNDLE_IDENTITY_KEYS, context=context)
    bundle_dir = Path(str(observed["bundle_dir"] or "")).expanduser()
    metadata_path = Path(str(observed["bundle_metadata_path"] or "")).expanduser()
    state_path = Path(str(observed["model_state_dict_path"] or "")).expanduser()
    if any(
        not path.is_absolute()
        or any(component.is_symlink() for component in (path, *path.parents))
        for path in (bundle_dir, metadata_path, state_path)
    ):
        _fail(context, "bundle identity paths must be absolute and non-symlinked")
    bundle_dir = bundle_dir.resolve()
    metadata_path = metadata_path.resolve()
    state_path = state_path.resolve()
    if metadata_path != bundle_dir / "bundle_metadata.json":
        _fail(context, "bundle metadata path is not canonical")
    if state_path != bundle_dir / "model_state_dict.pt":
        _fail(context, "model state path is not canonical")
    metadata_sha = _sha(
        observed["bundle_metadata_sha256"], context=f"{context}.metadata_sha"
    )
    state_sha = _sha(
        observed["model_state_dict_sha256"], context=f"{context}.state_sha"
    )
    if not metadata_path.is_file() or not state_path.is_file():
        _fail(context, "bundle metadata/model state is missing")
    if sha256_file(metadata_path) != metadata_sha or sha256_file(state_path) != state_sha:
        _fail(context, "bundle metadata/model state hash mismatch")
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ModelNativeAdaptationDriftError(
            f"[{context}_INVALID] bundle metadata is unreadable"
        ) from exc
    if not isinstance(metadata, dict) or str(metadata.get("state_dict_sha256") or "").lower() != state_sha:
        _fail(context, "bundle metadata does not bind model state")
    return {
        "bundle_dir": str(bundle_dir),
        "bundle_metadata_path": str(metadata_path),
        "bundle_metadata_sha256": metadata_sha,
        "model_state_dict_path": str(state_path),
        "model_state_dict_sha256": state_sha,
    }


def adaptation_bundle_identity_from_dir(
    bundle_dir: Path,
    *,
    context: str,
) -> dict[str, str]:
    """Build and verify the one canonical byte identity for an Entry bundle."""

    raw = bundle_dir.expanduser()
    absolute = raw if raw.is_absolute() else Path.cwd() / raw
    if any(component.is_symlink() for component in (absolute, *absolute.parents)):
        _fail(context, "bundle path must not traverse a symlink")
    bundle = raw.resolve()
    metadata_path = bundle / "bundle_metadata.json"
    state_path = bundle / "model_state_dict.pt"
    if not metadata_path.is_file() or not state_path.is_file():
        _fail(context, "bundle lacks canonical metadata/model state")
    return require_adaptation_bundle_identity(
        {
            "bundle_dir": str(bundle),
            "bundle_metadata_path": str(metadata_path),
            "bundle_metadata_sha256": sha256_file(metadata_path),
            "model_state_dict_path": str(state_path),
            "model_state_dict_sha256": sha256_file(state_path),
        },
        context=context,
    )


def recompute_spread_aware_pnl_bps(
    directions: np.ndarray,
    entry_bid: np.ndarray,
    entry_ask: np.ndarray,
    exit_bid: np.ndarray,
    exit_ask: np.ndarray,
    *,
    context: str,
) -> np.ndarray:
    """Recompute LONG/SHORT/FLAT PnL from executable bid/ask prices."""

    prices = np.column_stack((entry_bid, entry_ask, exit_bid, exit_ask)).astype(
        np.float64
    )
    entry_bid, entry_ask, exit_bid, exit_ask = prices.T
    if (
        not np.isfinite(prices).all()
        or np.any(prices <= 0.0)
        or np.any(entry_ask < entry_bid)
        or np.any(exit_ask < exit_bid)
    ):
        _fail(context, "bid/ask prices must be finite, positive and ordered")
    pnl = np.zeros(len(directions), dtype=np.float64)
    long_mask = directions == 0
    short_mask = directions == 1
    pnl[long_mask] = (
        (exit_bid[long_mask] - entry_ask[long_mask])
        / entry_ask[long_mask]
        * 10_000.0
    )
    pnl[short_mask] = (
        (entry_bid[short_mask] - exit_ask[short_mask])
        / entry_bid[short_mask]
        * 10_000.0
    )
    return pnl


def _validated_frame(
    frame: pd.DataFrame,
    *,
    scope: str,
    bundle_identity: Mapping[str, str],
    context: str,
) -> tuple[pd.DataFrame, pd.Series, np.ndarray, np.ndarray]:
    if set(frame.columns) != set(MODEL_NATIVE_ADAPTATION_DRIFT_COLUMNS):
        _fail(
            context,
            "columns mismatch: "
            f"missing={sorted(MODEL_NATIVE_ADAPTATION_DRIFT_COLUMNS - set(frame.columns))} "
            f"unexpected={sorted(set(frame.columns) - MODEL_NATIVE_ADAPTATION_DRIFT_COLUMNS)}",
        )
    if len(frame) < MODEL_NATIVE_ADAPTATION_DRIFT_MIN_ROWS:
        _fail(context, "insufficient rows")
    times = pd.to_datetime(frame["time"], utc=True, errors="coerce")
    if times.isna().any() or times.duplicated().any() or not times.is_monotonic_increasing:
        _fail(context, "times must be unique monotonic UTC")
    if set(frame["evidence_scope"].astype(str)) != {scope}:
        _fail(context, f"evidence_scope must be exact {scope}")
    numeric_direction = pd.to_numeric(
        frame["model_direction_index"], errors="coerce"
    ).to_numpy(dtype=np.float64)
    if (
        not np.isfinite(numeric_direction).all()
        or not np.array_equal(numeric_direction, numeric_direction.astype(np.int64))
        or not np.isin(numeric_direction.astype(np.int64), [0, 1, 2]).all()
    ):
        _fail(context, "direction must be exact LONG/SHORT/FLAT integers")
    directions = numeric_direction.astype(np.int64)
    probabilities = frame[["p_long", "p_short", "p_flat"]].apply(
        pd.to_numeric, errors="coerce"
    ).to_numpy(dtype=np.float64)
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
        _fail(
            context,
            "probabilities must be finite normalized unique model argmax evidence",
        )
    pnl = pd.to_numeric(frame["realized_pnl_bps"], errors="coerce").to_numpy(
        dtype=np.float64
    )
    prices = frame[["entry_bid", "entry_ask", "exit_bid", "exit_ask"]].apply(
        pd.to_numeric, errors="coerce"
    ).to_numpy(dtype=np.float64)
    entry_bid, entry_ask, exit_bid, exit_ask = prices.T
    recomputed_pnl = recompute_spread_aware_pnl_bps(
        directions,
        entry_bid,
        entry_ask,
        exit_bid,
        exit_ask,
        context=f"{context}.prices",
    )
    if (
        not np.isfinite(pnl).all()
        or np.any(pnl[directions == 2] != 0.0)
        or not np.allclose(pnl, recomputed_pnl, rtol=0.0, atol=1e-9)
    ):
        _fail(context, "realized PnL differs from spread-aware bid/ask recomputation")
    if not frame["order_submitted"].map(
        lambda value: isinstance(value, (bool, np.bool_))
    ).all():
        _fail(context, "order_submitted must contain exact booleans")
    if frame["order_submitted"].astype(bool).any():
        _fail(context, "adaptation evidence must be shadow-only and submit no order")
    for field in ("session", "vol_regime", "outcome_source_id"):
        values = frame[field].astype(str)
        if values.str.strip().ne(values).any() or values.str.len().eq(0).any():
            _fail(context, f"{field} contains empty or noncanonical values")
    if frame["outcome_source_id"].astype(str).duplicated().any():
        _fail(context, "outcome_source_id must be unique")
    for field, expected in (
        ("bundle_metadata_sha256", bundle_identity["bundle_metadata_sha256"]),
        ("model_state_dict_sha256", bundle_identity["model_state_dict_sha256"]),
    ):
        if set(frame[field].astype(str).str.lower()) != {expected}:
            _fail(context, f"{field} differs from bundle identity")
    counts = [int(np.count_nonzero(directions == index)) for index in range(3)]
    if min(counts) < MODEL_NATIVE_ADAPTATION_DRIFT_MIN_ROWS_PER_DIRECTION:
        _fail(context, "LONG/SHORT/FLAT support is insufficient")
    return frame.copy(), times, directions, probabilities


def adaptation_side_metrics(
    pnl: np.ndarray,
    directions: np.ndarray,
    direction: int,
) -> dict[str, Any]:
    values = pnl[directions == direction]
    losing = int(np.count_nonzero(values <= 0.0))
    rows = int(len(values))
    rate = losing / rows
    wilson = direction_pocket_wilson_upper_95(failures=losing, total=rows)
    mean = float(np.mean(values))
    lower = mean - MODEL_NATIVE_ADAPTATION_DRIFT_PNL_Z * float(
        np.std(values, ddof=1) / math.sqrt(rows)
    )
    passed = (
        rate <= MODEL_NATIVE_ADAPTATION_DRIFT_MAX_REALIZED_LOSS_RATE
        and wilson
        <= MODEL_NATIVE_ADAPTATION_DRIFT_MAX_REALIZED_LOSS_WILSON_UPPER_95
        and lower > 0.0
    )
    return {
        "rows": rows,
        "losing_rows": losing,
        "loss_rate": rate,
        "loss_wilson_upper_95": wilson,
        "mean_pnl_bps": mean,
        "mean_pnl_lower_95_bps": lower,
        "decision": "PASS" if passed else "FAIL",
    }


def _js_divergence(left: np.ndarray, right: np.ndarray) -> float:
    midpoint = 0.5 * (left + right)
    left_mask = left > 0.0
    right_mask = right > 0.0
    left_term = np.sum(left[left_mask] * np.log(left[left_mask] / midpoint[left_mask]))
    right_term = np.sum(
        right[right_mask] * np.log(right[right_mask] / midpoint[right_mask])
    )
    return float(0.5 * (left_term + right_term))


def _context_metrics(
    reference: pd.DataFrame,
    observations: pd.DataFrame,
) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    failures: list[str] = []
    covered_fields: set[str] = set()
    for field in ("session", "vol_regime"):
        for value in sorted(set(reference[field].astype(str))):
            for direction in (0, 1):
                ref = reference[
                    (reference[field].astype(str) == value)
                    & (reference["model_direction_index"].astype(int) == direction)
                ]
                if len(ref) < MODEL_NATIVE_ADAPTATION_DRIFT_MIN_REFERENCE_CONTEXT_ROWS:
                    continue
                obs = observations[
                    (observations[field].astype(str) == value)
                    & (observations["model_direction_index"].astype(int) == direction)
                ]
                obs_pnl = pd.to_numeric(obs["realized_pnl_bps"]).to_numpy(float)
                obs_rows = int(len(obs_pnl))
                losing = int(np.count_nonzero(obs_pnl <= 0.0)) if obs_rows else 0
                rate = losing / obs_rows if obs_rows else 1.0
                wilson = (
                    direction_pocket_wilson_upper_95(failures=losing, total=obs_rows)
                    if obs_rows
                    else 1.0
                )
                mean = float(np.mean(obs_pnl)) if obs_rows else 0.0
                lower = (
                    mean
                    - MODEL_NATIVE_ADAPTATION_DRIFT_PNL_Z
                    * float(np.std(obs_pnl, ddof=1) / math.sqrt(obs_rows))
                    if obs_rows > 1
                    else 0.0
                )
                passed = (
                    obs_rows >= MODEL_NATIVE_ADAPTATION_DRIFT_MIN_OBSERVATION_CONTEXT_ROWS
                    and rate
                    <= MODEL_NATIVE_ADAPTATION_DRIFT_MAX_REALIZED_LOSS_RATE
                    and wilson
                    <= MODEL_NATIVE_ADAPTATION_DRIFT_MAX_REALIZED_LOSS_WILSON_UPPER_95
                    and lower > 0.0
                )
                row = {
                    "field": field,
                    "value": value,
                    "model_direction_index": direction,
                    "reference_rows": int(len(ref)),
                    "observation_rows": obs_rows,
                    "observation_losing_rows": losing,
                    "observation_loss_rate": rate,
                    "observation_loss_wilson_upper_95": wilson,
                    "observation_mean_pnl_bps": mean,
                    "observation_mean_pnl_lower_95_bps": lower,
                    "decision": "PASS" if passed else "FAIL",
                }
                rows.append(row)
                covered_fields.add(field)
                if not passed:
                    failures.append(f"context:{field}={value}:direction={direction}")
    if covered_fields != {"session", "vol_regime"}:
        failures.append("context:reference_support_missing")
    return rows, failures


def recompute_adaptation_drift_evidence(
    *,
    reference_rows: pd.DataFrame,
    observation_rows: pd.DataFrame,
    bundle_identity: Mapping[str, str],
    event_created_utc: Any,
    context: str,
) -> dict[str, Any]:
    """Recompute the complete stable/drift decision from exact row evidence."""

    reference, ref_times, ref_dir, ref_probs = _validated_frame(
        reference_rows,
        scope="candidate_test",
        bundle_identity=bundle_identity,
        context=f"{context}.reference",
    )
    observations, obs_times, obs_dir, obs_probs = _validated_frame(
        observation_rows,
        scope="broker_shadow",
        bundle_identity=bundle_identity,
        context=f"{context}.observations",
    )
    created = _utc(event_created_utc, context=f"{context}.created_utc")
    if obs_times.iloc[0] <= ref_times.iloc[-1]:
        _fail(context, "broker-shadow window must begin after TEST reference")
    if obs_times.iloc[-1] > created:
        _fail(context, "broker-shadow rows cannot be newer than the event")
    if (obs_times.iloc[-1] - obs_times.iloc[0]).total_seconds() > MODEL_NATIVE_ADAPTATION_DRIFT_MAX_OBSERVATION_WINDOW_SECONDS:
        _fail(context, "broker-shadow observation window is too wide")
    if (created - obs_times.iloc[-1]).total_seconds() > MODEL_NATIVE_ADAPTATION_DRIFT_MAX_OBSERVATION_LAG_SECONDS:
        _fail(context, "broker-shadow observations are stale relative to the event")

    ref_distribution = np.bincount(ref_dir, minlength=3).astype(float) / len(ref_dir)
    obs_distribution = np.bincount(obs_dir, minlength=3).astype(float) / len(obs_dir)
    js = _js_divergence(ref_distribution, obs_distribution)
    ref_probability_means = ref_probs.mean(axis=0)
    obs_probability_means = obs_probs.mean(axis=0)
    max_probability_shift = float(
        np.max(np.abs(obs_probability_means - ref_probability_means))
    )
    ref_pnl = pd.to_numeric(reference["realized_pnl_bps"]).to_numpy(float)
    obs_pnl = pd.to_numeric(observations["realized_pnl_bps"]).to_numpy(float)
    side_sections = {
        "reference_long": adaptation_side_metrics(ref_pnl, ref_dir, 0),
        "reference_short": adaptation_side_metrics(ref_pnl, ref_dir, 1),
        "observation_long": adaptation_side_metrics(obs_pnl, obs_dir, 0),
        "observation_short": adaptation_side_metrics(obs_pnl, obs_dir, 1),
    }
    global_failures: list[str] = []
    if js > MODEL_NATIVE_ADAPTATION_DRIFT_MAX_JS_DIVERGENCE:
        global_failures.append("direction_distribution_js_divergence")
    if max_probability_shift > MODEL_NATIVE_ADAPTATION_DRIFT_MAX_PROBABILITY_MEAN_SHIFT:
        global_failures.append("direction_probability_mean_shift")
    global_failures.extend(
        name for name, section in side_sections.items() if section["decision"] != "PASS"
    )
    context_rows, context_failures = _context_metrics(reference, observations)
    failures = global_failures + context_failures
    coverage = {
        "reference_rows": int(len(reference)),
        "observation_rows": int(len(observations)),
        "reference_long_rows": int(np.count_nonzero(ref_dir == 0)),
        "reference_short_rows": int(np.count_nonzero(ref_dir == 1)),
        "reference_flat_rows": int(np.count_nonzero(ref_dir == 2)),
        "observation_long_rows": int(np.count_nonzero(obs_dir == 0)),
        "observation_short_rows": int(np.count_nonzero(obs_dir == 1)),
        "observation_flat_rows": int(np.count_nonzero(obs_dir == 2)),
        "reference_first_utc": ref_times.iloc[0].isoformat(),
        "reference_last_utc": ref_times.iloc[-1].isoformat(),
        "observation_first_utc": obs_times.iloc[0].isoformat(),
        "observation_last_utc": obs_times.iloc[-1].isoformat(),
        "reference_utc_ns_sha256": hashlib.sha256(
            ref_times.astype("int64").to_numpy(np.int64).tobytes()
        ).hexdigest(),
        "observation_utc_ns_sha256": hashlib.sha256(
            obs_times.astype("int64").to_numpy(np.int64).tobytes()
        ).hexdigest(),
        "distinct_observation_source_ids": int(
            observations["outcome_source_id"].astype(str).nunique()
        ),
        "order_submission_count": 0,
    }
    global_metrics = {
        "reference_direction_distribution": ref_distribution.tolist(),
        "observation_direction_distribution": obs_distribution.tolist(),
        "direction_distribution_js_divergence": js,
        "reference_probability_means": ref_probability_means.tolist(),
        "observation_probability_means": obs_probability_means.tolist(),
        "max_probability_mean_shift": max_probability_shift,
        **side_sections,
        "decision": "PASS" if not global_failures else "FAIL",
    }
    return {
        "decision": (
            MODEL_NATIVE_ADAPTATION_DRIFT_STABLE
            if not failures
            else MODEL_NATIVE_ADAPTATION_DRIFT_RED
        ),
        "failures": failures,
        "coverage": coverage,
        "global_metrics": global_metrics,
        "context_metrics": context_rows,
    }


def load_bound_adaptation_drift_evidence(
    binding: Mapping[str, Any] | Any,
    *,
    context: str,
    now_utc: Any | None = None,
) -> tuple[dict[str, Any], dict[str, str]]:
    """Load newest immutable drift evidence and independently recompute it."""

    try:
        canonical_binding = require_immutable_json_binding(
            binding,
            event_prefix=MODEL_NATIVE_ADAPTATION_DRIFT_EVENT_PREFIX,
            context=f"{context}.binding",
            verify_file=True,
        )
        path = Path(canonical_binding["json_path"])
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise ModelNativeAdaptationDriftError(
                f"[{context}_INVALID] drift event is unreadable"
            ) from exc
        event = _exact_keys(raw, _EVENT_KEYS, context=context)
        if event["schema_version"] != MODEL_NATIVE_ADAPTATION_DRIFT_SCHEMA_VERSION:
            _fail(context, "schema_version mismatch")
        if event["drift_contract"] != MODEL_NATIVE_ADAPTATION_DRIFT_CONTRACT:
            _fail(context, "drift contract mismatch")
        created = _utc(event["created_utc"], context=f"{context}.created_utc")
        if Path(str(event["json_path"] or "")).expanduser().resolve() != path:
            _fail(context, "json_path self-reference mismatch")
        now = _utc(
            pd.Timestamp.now(tz="UTC") if now_utc is None else now_utc,
            context=f"{context}.now_utc",
        )
        age = (now - created).total_seconds()
        if age < 0.0 or age > MODEL_NATIVE_ADAPTATION_DRIFT_MAX_EVENT_AGE_SECONDS:
            _fail(context, f"event age_seconds={age} is invalid")
        bundle_identity = require_adaptation_bundle_identity(
            event["bundle_identity"],
            context=f"{context}.bundle_identity",
        )
        if bundle_identity != event["bundle_identity"]:
            _fail(context, "bundle identity canonicalization mismatch")
        reference_binding = _source_binding(
            event["reference_rows"],
            context=f"{context}.reference_rows",
        )
        observation_binding = _source_binding(
            event["observation_rows"],
            context=f"{context}.observation_rows",
        )
        reference = pd.read_parquet(reference_binding["path"])
        observations = pd.read_parquet(observation_binding["path"])
        recomputed = recompute_adaptation_drift_evidence(
            reference_rows=reference,
            observation_rows=observations,
            bundle_identity=bundle_identity,
            event_created_utc=created,
            context=f"{context}.recompute",
        )
        if event["decision"] != recomputed["decision"] or event["failures"] != recomputed["failures"]:
            _fail(context, "reported drift decision differs from row recomputation")
        if _exact_keys(event["coverage"], _COVERAGE_KEYS, context=f"{context}.coverage") != recomputed["coverage"]:
            _fail(context, "reported coverage differs from row recomputation")
        reported_global = _exact_keys(
            event["global_metrics"], _GLOBAL_KEYS, context=f"{context}.global"
        )
        for name in ("reference_long", "reference_short", "observation_long", "observation_short"):
            _exact_keys(
                reported_global[name],
                _SIDE_METRIC_KEYS,
                context=f"{context}.global.{name}",
            )
        if reported_global != recomputed["global_metrics"]:
            _fail(context, "reported global metrics differ from row recomputation")
        if not isinstance(event["context_metrics"], list):
            _fail(context, "context_metrics must be a list")
        for index, row in enumerate(event["context_metrics"]):
            _exact_keys(row, _CONTEXT_ROW_KEYS, context=f"{context}.context[{index}]")
        if event["context_metrics"] != recomputed["context_metrics"]:
            _fail(context, "reported context metrics differ from row recomputation")
        return event, canonical_binding
    except ModelNativeSizingContractError as exc:
        raise ModelNativeAdaptationDriftError(str(exc)) from exc


__all__ = [
    "MODEL_NATIVE_ADAPTATION_DRIFT_COLUMNS",
    "MODEL_NATIVE_ADAPTATION_DRIFT_CONTRACT",
    "MODEL_NATIVE_ADAPTATION_DRIFT_EVENT_PREFIX",
    "MODEL_NATIVE_ADAPTATION_DRIFT_MAX_EVENT_AGE_SECONDS",
    "MODEL_NATIVE_ADAPTATION_DRIFT_RED",
    "MODEL_NATIVE_ADAPTATION_DRIFT_SCHEMA_VERSION",
    "MODEL_NATIVE_ADAPTATION_DRIFT_STABLE",
    "ModelNativeAdaptationDriftError",
    "adaptation_side_metrics",
    "adaptation_bundle_identity_from_dir",
    "load_bound_adaptation_drift_evidence",
    "recompute_adaptation_drift_evidence",
    "recompute_spread_aware_pnl_bps",
    "require_adaptation_bundle_identity",
]
