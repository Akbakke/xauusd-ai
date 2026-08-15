"""Canonical causal Bollinger squeeze/release state for every native clock.

The owner consumes already-closed OHLCV on exactly one declared timeframe.
It never resamples candles or previously computed indicators.  A two-state
Gaussian hidden-state model is fitted on the declared TRAIN population's
relative Bollinger bandwidth.  The lower fitted emission mean is the squeeze
state.  Its fitted transition matrix supplies persistence (hysteresis), so
there is no hand-picked bandwidth, percentile, duration or release threshold.

Runtime is a causal one-step MAP filter.  ``release_event`` is emitted exactly
once on a low-volatility -> high-volatility transition.  Durations and ages
are raw native-bar counts; TRAIN-only model-input normalization owns their
scale.  Before enough close history, or before the first observed release for
release-memory fields, absence of knowledge is NaN rather than a fabricated
zero/sentinel.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from gx1.time.session_detector import TRADING_SESSION_BOUNDARY_OFFSET


VOLATILITY_SQUEEZE_STATE_SCHEMA_VERSION = (
    "gx1_volatility_squeeze_state_train_fit_v2"
)
VOLATILITY_SQUEEZE_ARTIFACT_MANIFEST_SCHEMA_VERSION = (
    "gx1_volatility_squeeze_six_clock_manifest_v1"
)
VOLATILITY_SQUEEZE_STATE_OWNER = (
    "gx1.features.volatility_squeeze_state_v1"
)
VOLATILITY_SQUEEZE_FIT_METHOD = (
    "two_state_gaussian_viterbi_hard_em_on_relative_bb_bandwidth_v1"
)

# Existing Bollinger 20/2 identity used by both basic_v1 and htf_features.
# These are formula identity, not decision thresholds.
BOLLINGER_PERIOD_BARS = 20
BOLLINGER_STD_MULTIPLIER = 2.0
VOLATILITY_SQUEEZE_PREFIX_ROWS = BOLLINGER_PERIOD_BARS - 1

VOLATILITY_SQUEEZE_CLOCKS = ("M1", "M5", "M15", "H1", "H4", "D1")
VOLATILITY_SQUEEZE_CLOCK_CONTRACT = "closed_ohlcv_native_clock_v1"
_CLOCK_DURATION = {
    "M1": pd.Timedelta(minutes=1),
    "M5": pd.Timedelta(minutes=5),
    "M15": pd.Timedelta(minutes=15),
    "H1": pd.Timedelta(hours=1),
    "H4": pd.Timedelta(hours=4),
    "D1": pd.Timedelta(days=1),
}
VOLATILITY_SQUEEZE_FEATURE_NAMES = (
    "volatility.squeeze_active",
    "volatility.bars_in_squeeze",
    "volatility.squeeze_release_event",
    "volatility.duration_at_release",
    "volatility.squeeze_release_age_bars",
)

# The declared TRAIN window is carried by ``declared_train_window_start`` /
# ``declared_train_window_end`` and re-checked against the rebuild chain's own
# split authority at the chain's contract-validation step. A split-manifest
# pointer was additionally required here until 2026-08-15; it was removed
# because the chain PRODUCES its split manifests from the dataset this fit
# feeds, so the pointer could never be satisfied on a first build (circular
# lineage), and the invariant it named — "this fit saw exactly the declared
# TRAIN rows" — is owned by the window equality, not by a file pointer.
_SOURCE_PROVENANCE_KEYS = frozenset(
    {
        "source_artifact",
        "source_sha256",
        "source_schema_version",
        "source_lane",
        "tape_manifest_artifact",
        "tape_manifest_sha256",
        "pair_manifest_artifact",
        "pair_manifest_sha256",
        "pair_generation_id",
        "pair_symbol",
        "train_split_id",
        "clock_contract",
        "bar_grid",
    }
)
# Lineage keys retired on 2026-08-15. Nothing may reintroduce them: they are
# asserted absent from every payload key set by the owner's own tests.
RETIRED_TRAIN_LINEAGE_KEYS = frozenset(
    {"split_manifest_artifact", "split_manifest_sha256"}
)
_BAR_GRID_KEYS = frozenset(
    {"timeframe", "duration_ns", "origin_offset_ns", "clock_contract"}
)
_FIT_KEYS = frozenset(
    {
        "method",
        "low_state_index",
        "bandwidth_mean",
        "bandwidth_variance",
        "transition_probability",
        "initial_probability",
        "train_observation_count",
    }
)
_PARAM_KEYS = frozenset(
    {
        "schema_version",
        "owner",
        "timeframe",
        "bollinger_period_bars",
        "bollinger_std_multiplier",
        "declared_train_window_start",
        "declared_train_window_end",
        "train_ohlcv_rows",
        "input_ohlcv_sha256",
        "train_ohlcv_sha256",
        "source_provenance",
        "train_lineage_sha256",
        "fit",
        "contract_sha256",
    }
)
_COMMON_TRAIN_LINEAGE_KEYS = frozenset(
    {
        "declared_train_window_start",
        "declared_train_window_end",
        "train_split_id",
        "tape_manifest_artifact",
        "tape_manifest_sha256",
        "pair_manifest_artifact",
        "pair_manifest_sha256",
        "pair_generation_id",
        "pair_symbol",
        "clock_contract",
        "train_lineage_sha256",
    }
)
_ARTIFACT_ENTRY_KEYS = frozenset(
    {
        "params_artifact",
        "params_file_sha256",
        "params_payload_sha256",
        "source_artifact",
        "source_sha256",
        "source_schema_version",
        "source_lane",
        "bar_grid",
        "train_ohlcv_rows",
        "train_ohlcv_sha256",
    }
)
_ARTIFACT_MANIFEST_KEYS = frozenset(
    {
        "schema_version",
        "owner",
        "clocks",
        "feature_names",
        "common_train_lineage",
        "artifacts",
        "contract_sha256",
    }
)


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    value = dict(payload)
    value.pop("contract_sha256", None)
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _require_sha256(value: Any, *, label: str) -> str:
    observed = str(value)
    if len(observed) != 64 or any(ch not in "0123456789abcdef" for ch in observed):
        raise RuntimeError(f"{label}_SHA256_INVALID")
    return observed


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_timeframe(timeframe: str) -> str:
    if timeframe not in VOLATILITY_SQUEEZE_CLOCKS:
        raise RuntimeError(
            "VOLATILITY_SQUEEZE_TIMEFRAME_INVALID: "
            f"observed={timeframe!r} expected={VOLATILITY_SQUEEZE_CLOCKS}"
        )
    return timeframe


def volatility_squeeze_bar_grid(timeframe: str) -> dict[str, Any]:
    """Return the exact native closed-bar grid bound into fit artifacts."""

    clock = _require_timeframe(timeframe)
    duration = _CLOCK_DURATION[clock]
    offset = TRADING_SESSION_BOUNDARY_OFFSET % duration
    return {
        "timeframe": clock,
        "duration_ns": int(duration.value),
        "origin_offset_ns": int(offset.value),
        "clock_contract": VOLATILITY_SQUEEZE_CLOCK_CONTRACT,
    }


def _require_bar_grid(value: Any, *, timeframe: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != set(_BAR_GRID_KEYS):
        raise RuntimeError("VOLATILITY_SQUEEZE_BAR_GRID_INVALID")
    expected = volatility_squeeze_bar_grid(timeframe)
    if dict(value) != expected:
        raise RuntimeError("VOLATILITY_SQUEEZE_BAR_GRID_INVALID")
    return expected


def _require_utc_timestamp(value: Any, *, label: str) -> pd.Timestamp:
    try:
        observed = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"{label}_INVALID") from exc
    if observed.tzinfo is None or observed.utcoffset() != pd.Timedelta(0):
        raise RuntimeError(f"{label}_INVALID")
    return observed.as_unit("ns")


def _validate_closed_ohlcv(
    ohlcv: pd.DataFrame,
    *,
    timeframe: str,
    context: str,
) -> pd.DataFrame:
    if not isinstance(ohlcv, pd.DataFrame) or ohlcv.empty:
        raise RuntimeError(f"{context}_SOURCE_EMPTY")
    required = ("open", "high", "low", "close", "volume")
    if any(list(ohlcv.columns).count(name) != 1 for name in required):
        raise RuntimeError(f"{context}_SOURCE_COLUMNS_INVALID")
    if not isinstance(ohlcv.index, pd.DatetimeIndex):
        raise RuntimeError(f"{context}_TIME_INDEX_REQUIRED")
    index = ohlcv.index
    duration = _CLOCK_DURATION[_require_timeframe(timeframe)]
    offset = TRADING_SESSION_BOUNDARY_OFFSET % duration
    if (
        index.tz is None
        or index.hasnans
        or not index.is_unique
        or not index.is_monotonic_increasing
        or pd.Timestamp(index[0]).utcoffset() != pd.Timedelta(0)
        or np.any((index.asi8 - int(offset.value)) % int(duration.value) != 0)
    ):
        raise RuntimeError(f"{context}_TIME_INDEX_INVALID")
    try:
        numeric = ohlcv.loc[:, list(required)].apply(
            pd.to_numeric, errors="raise"
        ).astype(np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise RuntimeError(f"{context}_SOURCE_NOT_NUMERIC") from exc
    values = numeric.to_numpy(dtype=np.float64, copy=False)
    if not np.isfinite(values).all():
        raise RuntimeError(f"{context}_SOURCE_NONFINITE")
    open_ = numeric["open"].to_numpy(dtype=np.float64)
    high = numeric["high"].to_numpy(dtype=np.float64)
    low = numeric["low"].to_numpy(dtype=np.float64)
    close = numeric["close"].to_numpy(dtype=np.float64)
    volume = numeric["volume"].to_numpy(dtype=np.float64)
    if (
        np.any(open_ <= 0.0)
        or np.any(high <= 0.0)
        or np.any(low <= 0.0)
        or np.any(close <= 0.0)
        or np.any(high < np.maximum(open_, close))
        or np.any(low > np.minimum(open_, close))
        or np.any(high < low)
        or ohlcv["volume"].map(
            lambda value: isinstance(value, (bool, np.bool_))
        ).any()
        or np.any(volume <= 0.0)
        or not np.equal(volume, np.floor(volume)).all()
    ):
        raise RuntimeError(f"{context}_OHLCV_GEOMETRY_INVALID")
    numeric.index = index.as_unit("ns")
    return numeric


def _ohlcv_sha256(frame: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    digest.update(frame.index.asi8.astype("<i8", copy=False).tobytes(order="C"))
    digest.update(
        frame.loc[:, ["open", "high", "low", "close", "volume"]]
        .to_numpy(dtype="<f8", copy=True)
        .tobytes(order="C")
    )
    return digest.hexdigest()


def bollinger_relative_bandwidth(close: np.ndarray) -> np.ndarray:
    """Return ``4*std20/mean20`` with an honest 19-row NaN prefix."""

    values = np.asarray(close, dtype=np.float64)
    if values.ndim != 1 or len(values) == 0 or not np.isfinite(values).all():
        raise RuntimeError("VOLATILITY_SQUEEZE_CLOSE_SOURCE_INVALID")
    if np.any(values <= 0.0):
        raise RuntimeError("VOLATILITY_SQUEEZE_CLOSE_SOURCE_NONPOSITIVE")
    series = pd.Series(values, dtype=np.float64)
    mean = series.rolling(
        BOLLINGER_PERIOD_BARS,
        min_periods=BOLLINGER_PERIOD_BARS,
    ).mean()
    std = series.rolling(
        BOLLINGER_PERIOD_BARS,
        min_periods=BOLLINGER_PERIOD_BARS,
    ).std(ddof=0)
    width = (
        (2.0 * BOLLINGER_STD_MULTIPLIER) * std.to_numpy(dtype=np.float64)
        / mean.to_numpy(dtype=np.float64)
    )
    if not np.isnan(width[:VOLATILITY_SQUEEZE_PREFIX_ROWS]).all():
        raise RuntimeError("VOLATILITY_SQUEEZE_BANDWIDTH_PREFIX_INVALID")
    tail = width[VOLATILITY_SQUEEZE_PREFIX_ROWS:]
    if not np.isfinite(tail).all() or np.any(tail < 0.0):
        raise RuntimeError(
            "VOLATILITY_SQUEEZE_BANDWIDTH_UNIDENTIFIABLE: complete TRAIN "
            "windows must have nonnegative finite dispersion"
        )
    return width


def _log_normal_density(values: np.ndarray, means: np.ndarray, variances: np.ndarray) -> np.ndarray:
    return -0.5 * (
        np.log(2.0 * np.pi * variances)[None, :]
        + ((values[:, None] - means[None, :]) ** 2) / variances[None, :]
    )


def _estimate_hard_state_parameters(
    values: np.ndarray,
    states: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    means = np.asarray([values[states == state].mean() for state in (0, 1)])
    variances = np.asarray(
        [values[states == state].var(ddof=0) for state in (0, 1)]
    )
    if (
        not np.isfinite(means).all()
        or not np.isfinite(variances).all()
        or np.any(variances <= 0.0)
        or not means[0] < means[1]
    ):
        raise RuntimeError("VOLATILITY_SQUEEZE_TRAIN_STATES_UNIDENTIFIABLE")

    counts = np.zeros((2, 2), dtype=np.float64)
    np.add.at(counts, (states[:-1], states[1:]), 1.0)
    row_totals = counts.sum(axis=1)
    if np.any(row_totals <= 0.0) or np.any(counts <= 0.0):
        raise RuntimeError(
            "VOLATILITY_SQUEEZE_TRAIN_TRANSITIONS_UNIDENTIFIABLE: both "
            "states and both transition directions must be observed in TRAIN"
        )
    transition = counts / row_totals[:, None]
    if not (
        transition[0, 0] > transition[0, 1]
        and transition[1, 1] > transition[1, 0]
    ):
        raise RuntimeError(
            "VOLATILITY_SQUEEZE_TRAIN_HYSTERESIS_ABSENT: fitted states are "
            "not persistent on this TRAIN population"
        )
    # The stationary distribution is fully implied by the fitted transition
    # matrix; no separately guessed initial-state prior is introduced.
    cross_sum = transition[0, 1] + transition[1, 0]
    initial = np.asarray(
        [transition[1, 0] / cross_sum, transition[0, 1] / cross_sum],
        dtype=np.float64,
    )
    return means, variances, transition, initial


def _viterbi_path(
    values: np.ndarray,
    means: np.ndarray,
    variances: np.ndarray,
    transition: np.ndarray,
    initial: np.ndarray,
) -> np.ndarray:
    emissions = _log_normal_density(values, means, variances)
    log_transition = np.log(transition)
    scores = np.empty((len(values), 2), dtype=np.float64)
    parent = np.zeros((len(values), 2), dtype=np.int8)
    scores[0] = np.log(initial) + emissions[0]
    for row in range(1, len(values)):
        candidates = scores[row - 1][:, None] + log_transition
        parent[row] = np.argmax(candidates, axis=0).astype(np.int8)
        scores[row] = candidates[parent[row], np.arange(2)] + emissions[row]
    states = np.empty(len(values), dtype=np.int8)
    states[-1] = int(np.argmax(scores[-1]))
    for row in range(len(values) - 1, 0, -1):
        states[row - 1] = parent[row, states[row]]
    return states


def _fit_two_state_model(relative_bandwidth: np.ndarray) -> dict[str, Any]:
    values = np.asarray(relative_bandwidth, dtype=np.float64)
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all():
        raise RuntimeError("VOLATILITY_SQUEEZE_TRAIN_POPULATION_INVALID")
    median = float(np.median(values))
    states = (values > median).astype(np.int8)
    if len(np.unique(states)) != 2:
        raise RuntimeError("VOLATILITY_SQUEEZE_TRAIN_POPULATION_UNIDENTIFIABLE")

    # Deterministic hard-EM to a fixed point.  Repeated non-fixed assignments
    # are treated as an unidentifiable fit instead of accepting an arbitrary
    # iteration budget as hidden model state.
    seen: set[bytes] = set()
    while True:
        key = states.tobytes()
        if key in seen:
            raise RuntimeError("VOLATILITY_SQUEEZE_TRAIN_FIT_CYCLE")
        seen.add(key)
        means, variances, transition, initial = _estimate_hard_state_parameters(
            values,
            states,
        )
        updated = _viterbi_path(
            values,
            means,
            variances,
            transition,
            initial,
        )
        if np.array_equal(updated, states):
            break
        # Keep state 0 semantically bound to the lower emission mean.
        updated_means = np.asarray(
            [values[updated == state].mean() for state in (0, 1)]
        )
        if updated_means[0] > updated_means[1]:
            updated = (1 - updated).astype(np.int8)
        states = updated

    return {
        "method": VOLATILITY_SQUEEZE_FIT_METHOD,
        "low_state_index": 0,
        "bandwidth_mean": [float(value) for value in means],
        "bandwidth_variance": [float(value) for value in variances],
        "transition_probability": transition.tolist(),
        "initial_probability": initial.tolist(),
        "train_observation_count": int(len(values)),
    }


def _require_source_provenance(
    value: Mapping[str, Any],
    *,
    timeframe: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != set(
        _SOURCE_PROVENANCE_KEYS
    ):
        raise RuntimeError("VOLATILITY_SQUEEZE_SOURCE_PROVENANCE_INVALID")
    observed = {
        key: (
            _require_bar_grid(value[key], timeframe=timeframe)
            if key == "bar_grid"
            else str(value[key])
        )
        for key in _SOURCE_PROVENANCE_KEYS
    }
    artifact_bindings = (
        ("source_artifact", "source_sha256"),
        ("tape_manifest_artifact", "tape_manifest_sha256"),
        ("pair_manifest_artifact", "pair_manifest_sha256"),
    )
    if (
        any(not observed[key] for key in observed if key != "bar_grid")
        or observed["source_lane"] != timeframe
        or observed["pair_symbol"] != "XAUUSD"
        or observed["train_split_id"] != "TRAIN"
        or observed["clock_contract"] != VOLATILITY_SQUEEZE_CLOCK_CONTRACT
    ):
        raise RuntimeError("VOLATILITY_SQUEEZE_SOURCE_PROVENANCE_INVALID")
    for path_key, sha_key in artifact_bindings:
        artifact = Path(observed[path_key]).expanduser()
        expected = _require_sha256(
            observed[sha_key],
            label=f"VOLATILITY_SQUEEZE_{sha_key.upper()}",
        )
        if (
            not artifact.is_absolute()
            or artifact.is_symlink()
            or not artifact.is_file()
            or artifact.resolve(strict=True) != artifact
            or _sha256_file(artifact) != expected
        ):
            raise RuntimeError("VOLATILITY_SQUEEZE_SOURCE_PROVENANCE_INVALID")
    return observed


def _train_lineage_payload(
    provenance: Mapping[str, Any],
    *,
    declared_train_window_start: pd.Timestamp,
    declared_train_window_end: pd.Timestamp,
) -> dict[str, Any]:
    payload = {
        "declared_train_window_start": declared_train_window_start.isoformat(),
        "declared_train_window_end": declared_train_window_end.isoformat(),
        "train_split_id": provenance["train_split_id"],
        "tape_manifest_artifact": provenance["tape_manifest_artifact"],
        "tape_manifest_sha256": provenance["tape_manifest_sha256"],
        "pair_manifest_artifact": provenance["pair_manifest_artifact"],
        "pair_manifest_sha256": provenance["pair_manifest_sha256"],
        "pair_generation_id": provenance["pair_generation_id"],
        "pair_symbol": provenance["pair_symbol"],
        "clock_contract": provenance["clock_contract"],
    }
    payload["train_lineage_sha256"] = _canonical_sha256(payload)
    return payload


def fit_volatility_squeeze_params(
    ohlcv: pd.DataFrame,
    *,
    timeframe: str,
    declared_train_window_start: Any,
    declared_train_window_end: Any,
    source_provenance: Mapping[str, Any],
) -> dict[str, Any]:
    """Fit one immutable squeeze model on one clock's declared TRAIN rows."""

    clock = _require_timeframe(timeframe)
    source = _validate_closed_ohlcv(
        ohlcv,
        timeframe=clock,
        context="VOLATILITY_SQUEEZE_FIT",
    )
    window_start = _require_utc_timestamp(
        declared_train_window_start,
        label="VOLATILITY_SQUEEZE_TRAIN_WINDOW_START",
    )
    window_end = _require_utc_timestamp(
        declared_train_window_end,
        label="VOLATILITY_SQUEEZE_TRAIN_WINDOW_END",
    )
    if window_start > window_end:
        raise RuntimeError("VOLATILITY_SQUEEZE_TRAIN_WINDOW_INVALID")
    # This low-level fitter accepts TRAIN rows only.  Selection from a larger
    # tape is owned by the six-clock manifest fitter below and occurs before a
    # single observation reaches this function.
    if source.index[0] < window_start or source.index[-1] > window_end:
        raise RuntimeError("VOLATILITY_SQUEEZE_FIT_SOURCE_OUTSIDE_TRAIN")
    train = source
    if len(source) <= VOLATILITY_SQUEEZE_PREFIX_ROWS:
        raise RuntimeError("VOLATILITY_SQUEEZE_TRAIN_WINDOW_TOO_SHORT")
    provenance = _require_source_provenance(
        source_provenance,
        timeframe=clock,
    )

    bandwidth = bollinger_relative_bandwidth(
        train["close"].to_numpy(dtype=np.float64)
    )
    fit = _fit_two_state_model(bandwidth[VOLATILITY_SQUEEZE_PREFIX_ROWS:])
    lineage = _train_lineage_payload(
        provenance,
        declared_train_window_start=window_start,
        declared_train_window_end=window_end,
    )
    payload: dict[str, Any] = {
        "schema_version": VOLATILITY_SQUEEZE_STATE_SCHEMA_VERSION,
        "owner": VOLATILITY_SQUEEZE_STATE_OWNER,
        "timeframe": clock,
        "bollinger_period_bars": BOLLINGER_PERIOD_BARS,
        "bollinger_std_multiplier": BOLLINGER_STD_MULTIPLIER,
        "declared_train_window_start": window_start.isoformat(),
        "declared_train_window_end": window_end.isoformat(),
        "train_ohlcv_rows": int(len(train)),
        "input_ohlcv_sha256": _ohlcv_sha256(train),
        "train_ohlcv_sha256": _ohlcv_sha256(train),
        "source_provenance": provenance,
        "train_lineage_sha256": lineage["train_lineage_sha256"],
        "fit": fit,
    }
    payload["contract_sha256"] = _canonical_sha256(payload)
    return require_volatility_squeeze_params(payload, timeframe=clock)


def require_volatility_squeeze_params(
    params: Mapping[str, Any],
    *,
    timeframe: str,
) -> dict[str, Any]:
    """Validate exact schema, fit semantics, provenance and payload hash."""

    clock = _require_timeframe(timeframe)
    if not isinstance(params, Mapping) or set(params) != set(_PARAM_KEYS):
        raise RuntimeError("VOLATILITY_SQUEEZE_PARAMS_KEYS_INVALID")
    observed = dict(params)
    if (
        observed["schema_version"] != VOLATILITY_SQUEEZE_STATE_SCHEMA_VERSION
        or observed["owner"] != VOLATILITY_SQUEEZE_STATE_OWNER
        or observed["timeframe"] != clock
        or observed["bollinger_period_bars"] != BOLLINGER_PERIOD_BARS
        or observed["bollinger_std_multiplier"] != BOLLINGER_STD_MULTIPLIER
        or set(observed["fit"]) != set(_FIT_KEYS)
    ):
        raise RuntimeError("VOLATILITY_SQUEEZE_PARAMS_CONTRACT_INVALID")
    _require_source_provenance(
        observed["source_provenance"],
        timeframe=clock,
    )
    _require_sha256(
        observed["input_ohlcv_sha256"],
        label="VOLATILITY_SQUEEZE_INPUT_OHLCV",
    )
    _require_sha256(
        observed["train_ohlcv_sha256"],
        label="VOLATILITY_SQUEEZE_TRAIN_OHLCV",
    )
    _require_sha256(
        observed["contract_sha256"],
        label="VOLATILITY_SQUEEZE_CONTRACT",
    )
    if observed["contract_sha256"] != _canonical_sha256(observed):
        raise RuntimeError("VOLATILITY_SQUEEZE_PARAMS_HASH_MISMATCH")
    fit = observed["fit"]
    try:
        start = _require_utc_timestamp(
            observed["declared_train_window_start"],
            label="VOLATILITY_SQUEEZE_PARAMS_TRAIN_WINDOW",
        )
        end = _require_utc_timestamp(
            observed["declared_train_window_end"],
            label="VOLATILITY_SQUEEZE_PARAMS_TRAIN_WINDOW",
        )
    except (TypeError, ValueError) as exc:
        raise RuntimeError("VOLATILITY_SQUEEZE_PARAMS_TRAIN_WINDOW_INVALID") from exc
    if (
        start > end
        or isinstance(observed["train_ohlcv_rows"], bool)
    ):
        raise RuntimeError("VOLATILITY_SQUEEZE_PARAMS_TRAIN_WINDOW_INVALID")
    provenance = _require_source_provenance(
        observed["source_provenance"],
        timeframe=clock,
    )
    lineage = _train_lineage_payload(
        provenance,
        declared_train_window_start=start,
        declared_train_window_end=end,
    )
    _require_sha256(
        observed["train_lineage_sha256"],
        label="VOLATILITY_SQUEEZE_TRAIN_LINEAGE",
    )
    if observed["train_lineage_sha256"] != lineage["train_lineage_sha256"]:
        raise RuntimeError("VOLATILITY_SQUEEZE_TRAIN_LINEAGE_MISMATCH")
    if observed["input_ohlcv_sha256"] != observed["train_ohlcv_sha256"]:
        raise RuntimeError("VOLATILITY_SQUEEZE_NON_TRAIN_INPUT_FORBIDDEN")
    if fit["method"] != VOLATILITY_SQUEEZE_FIT_METHOD or fit["low_state_index"] != 0:
        raise RuntimeError("VOLATILITY_SQUEEZE_PARAMS_FIT_METHOD_INVALID")
    means = np.asarray(fit["bandwidth_mean"], dtype=np.float64)
    variances = np.asarray(fit["bandwidth_variance"], dtype=np.float64)
    transition = np.asarray(fit["transition_probability"], dtype=np.float64)
    initial = np.asarray(fit["initial_probability"], dtype=np.float64)
    if (
        means.shape != (2,)
        or variances.shape != (2,)
        or transition.shape != (2, 2)
        or initial.shape != (2,)
        or not np.isfinite(means).all()
        or not np.isfinite(variances).all()
        or not np.isfinite(transition).all()
        or not np.isfinite(initial).all()
        or not means[0] < means[1]
        or np.any(variances <= 0.0)
        or np.any(transition <= 0.0)
        or np.any(initial <= 0.0)
        or not np.allclose(transition.sum(axis=1), 1.0, rtol=0.0, atol=1e-12)
        or not math.isclose(float(initial.sum()), 1.0, rel_tol=0.0, abs_tol=1e-12)
        or not transition[0, 0] > transition[0, 1]
        or not transition[1, 1] > transition[1, 0]
        or isinstance(fit["train_observation_count"], bool)
        or int(fit["train_observation_count"]) <= 0
        or int(observed["train_ohlcv_rows"])
        != int(fit["train_observation_count"]) + VOLATILITY_SQUEEZE_PREFIX_ROWS
    ):
        raise RuntimeError("VOLATILITY_SQUEEZE_PARAMS_VALUES_INVALID")
    return observed


def write_volatility_squeeze_params(path: Path, params: Mapping[str, Any]) -> Path:
    """Create one immutable JSON artifact; an existing target is rejected."""

    target = Path(path).expanduser()
    if not target.is_absolute() or target.exists() or target.is_symlink():
        raise RuntimeError("VOLATILITY_SQUEEZE_ARTIFACT_TARGET_INVALID")
    validated = require_volatility_squeeze_params(
        params,
        timeframe=str(params.get("timeframe")),
    )
    encoded = (json.dumps(validated, indent=2, allow_nan=False) + "\n").encode("utf-8")
    with target.open("xb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    return target.resolve(strict=True)


def load_volatility_squeeze_params(
    path: Path,
    *,
    expected_sha256: str,
    timeframe: str,
) -> dict[str, Any]:
    """Load one explicit regular artifact and bind its exact file hash."""

    artifact = Path(path).expanduser()
    if artifact.is_symlink() or not artifact.is_file():
        raise RuntimeError("VOLATILITY_SQUEEZE_ARTIFACT_MISSING")
    expected = _require_sha256(
        expected_sha256,
        label="VOLATILITY_SQUEEZE_ARTIFACT",
    )
    data = artifact.read_bytes()
    if hashlib.sha256(data).hexdigest() != expected:
        raise RuntimeError("VOLATILITY_SQUEEZE_ARTIFACT_HASH_MISMATCH")
    try:
        payload = json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("VOLATILITY_SQUEEZE_ARTIFACT_JSON_INVALID") from exc
    return require_volatility_squeeze_params(payload, timeframe=timeframe)


@dataclass(frozen=True)
class VolatilitySqueezeArtifactSet:
    """Verified six-clock artifact set; the only admitted runtime carrier."""

    manifest_path: Path
    manifest_file_sha256: str
    manifest_payload_sha256: str
    train_lineage_sha256: str
    params_by_clock: Mapping[str, Mapping[str, Any]]
    params_file_sha256_by_clock: Mapping[str, str]

    def require_params(self, timeframe: str) -> dict[str, Any]:
        clock = _require_timeframe(timeframe)
        if tuple(self.params_by_clock) != VOLATILITY_SQUEEZE_CLOCKS:
            raise RuntimeError("VOLATILITY_SQUEEZE_ARTIFACT_SET_CLOCKS_INVALID")
        params = require_volatility_squeeze_params(
            self.params_by_clock[clock],
            timeframe=clock,
        )
        if params["train_lineage_sha256"] != self.train_lineage_sha256:
            raise RuntimeError("VOLATILITY_SQUEEZE_ARTIFACT_SET_LINEAGE_MISMATCH")
        return params

    def binding(self) -> dict[str, Any]:
        """Small exact manifest binding embedded in downstream artifacts."""

        return {
            "schema_version": VOLATILITY_SQUEEZE_ARTIFACT_MANIFEST_SCHEMA_VERSION,
            "manifest_path": str(self.manifest_path),
            "manifest_file_sha256": self.manifest_file_sha256,
            "manifest_payload_sha256": self.manifest_payload_sha256,
            "train_lineage_sha256": self.train_lineage_sha256,
            "params_payload_sha256_by_clock": {
                clock: self.require_params(clock)["contract_sha256"]
                for clock in VOLATILITY_SQUEEZE_CLOCKS
            },
            "params_file_sha256_by_clock": {
                clock: self.params_file_sha256_by_clock[clock]
                for clock in VOLATILITY_SQUEEZE_CLOCKS
            },
        }


_ARTIFACT_SET_BINDING_KEYS = frozenset(
    {
        "schema_version",
        "manifest_path",
        "manifest_file_sha256",
        "manifest_payload_sha256",
        "train_lineage_sha256",
        "params_payload_sha256_by_clock",
        "params_file_sha256_by_clock",
    }
)


def require_volatility_squeeze_artifact_set(
    value: Any,
) -> VolatilitySqueezeArtifactSet:
    """Reject loose mappings/defaults and validate every clock on each use."""

    if not isinstance(value, VolatilitySqueezeArtifactSet):
        raise RuntimeError(
            "VOLATILITY_SQUEEZE_ARTIFACT_SET_REQUIRED: load the exact "
            "six-clock manifest; bare params/defaults are forbidden"
        )
    _require_sha256(
        value.manifest_file_sha256,
        label="VOLATILITY_SQUEEZE_MANIFEST_FILE",
    )
    _require_sha256(
        value.manifest_payload_sha256,
        label="VOLATILITY_SQUEEZE_MANIFEST_PAYLOAD",
    )
    _require_sha256(
        value.train_lineage_sha256,
        label="VOLATILITY_SQUEEZE_TRAIN_LINEAGE",
    )
    if tuple(value.params_file_sha256_by_clock) != VOLATILITY_SQUEEZE_CLOCKS:
        raise RuntimeError("VOLATILITY_SQUEEZE_ARTIFACT_SET_CLOCKS_INVALID")
    for clock in VOLATILITY_SQUEEZE_CLOCKS:
        _require_sha256(
            value.params_file_sha256_by_clock[clock],
            label=f"VOLATILITY_SQUEEZE_{clock}_PARAMS_FILE",
        )
        value.require_params(clock)
    return value


def require_volatility_squeeze_artifact_binding(
    value: Any,
) -> VolatilitySqueezeArtifactSet:
    """Load and verify the exact manifest described by a downstream binding."""

    if not isinstance(value, Mapping) or set(value) != set(
        _ARTIFACT_SET_BINDING_KEYS
    ):
        raise RuntimeError("VOLATILITY_SQUEEZE_ARTIFACT_BINDING_INVALID")
    if value["schema_version"] != VOLATILITY_SQUEEZE_ARTIFACT_MANIFEST_SCHEMA_VERSION:
        raise RuntimeError("VOLATILITY_SQUEEZE_ARTIFACT_BINDING_INVALID")
    loaded = load_volatility_squeeze_artifact_manifest(
        Path(str(value["manifest_path"])),
        expected_sha256=str(value["manifest_file_sha256"]),
    )
    if loaded.binding() != dict(value):
        raise RuntimeError("VOLATILITY_SQUEEZE_ARTIFACT_BINDING_MISMATCH")
    return loaded


def _require_common_train_lineage(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != set(
        _COMMON_TRAIN_LINEAGE_KEYS
    ):
        raise RuntimeError("VOLATILITY_SQUEEZE_COMMON_TRAIN_LINEAGE_INVALID")
    observed = dict(value)
    start = _require_utc_timestamp(
        observed["declared_train_window_start"],
        label="VOLATILITY_SQUEEZE_COMMON_TRAIN_START",
    )
    end = _require_utc_timestamp(
        observed["declared_train_window_end"],
        label="VOLATILITY_SQUEEZE_COMMON_TRAIN_END",
    )
    if start > end:
        raise RuntimeError("VOLATILITY_SQUEEZE_COMMON_TRAIN_LINEAGE_INVALID")
    pseudo_provenance = {
        key: observed[key]
        for key in (
            "train_split_id",
            "tape_manifest_artifact",
            "tape_manifest_sha256",
            "pair_manifest_artifact",
            "pair_manifest_sha256",
            "pair_generation_id",
            "pair_symbol",
            "clock_contract",
        )
    }
    expected = _train_lineage_payload(
        pseudo_provenance,
        declared_train_window_start=start,
        declared_train_window_end=end,
    )
    if observed != expected:
        raise RuntimeError("VOLATILITY_SQUEEZE_COMMON_TRAIN_LINEAGE_INVALID")
    return observed


def _require_artifact_manifest_payload(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != set(
        _ARTIFACT_MANIFEST_KEYS
    ):
        raise RuntimeError("VOLATILITY_SQUEEZE_ARTIFACT_MANIFEST_INVALID")
    observed = dict(value)
    if (
        observed["schema_version"]
        != VOLATILITY_SQUEEZE_ARTIFACT_MANIFEST_SCHEMA_VERSION
        or observed["owner"] != VOLATILITY_SQUEEZE_STATE_OWNER
        or tuple(observed["clocks"]) != VOLATILITY_SQUEEZE_CLOCKS
        or tuple(observed["feature_names"]) != VOLATILITY_SQUEEZE_FEATURE_NAMES
        or not isinstance(observed["artifacts"], Mapping)
        or tuple(observed["artifacts"]) != VOLATILITY_SQUEEZE_CLOCKS
    ):
        raise RuntimeError("VOLATILITY_SQUEEZE_ARTIFACT_MANIFEST_INVALID")
    lineage = _require_common_train_lineage(observed["common_train_lineage"])
    _require_sha256(
        observed["contract_sha256"],
        label="VOLATILITY_SQUEEZE_MANIFEST_PAYLOAD",
    )
    if observed["contract_sha256"] != _canonical_sha256(observed):
        raise RuntimeError("VOLATILITY_SQUEEZE_ARTIFACT_MANIFEST_HASH_MISMATCH")
    for clock in VOLATILITY_SQUEEZE_CLOCKS:
        entry = observed["artifacts"][clock]
        if not isinstance(entry, Mapping) or set(entry) != set(_ARTIFACT_ENTRY_KEYS):
            raise RuntimeError("VOLATILITY_SQUEEZE_ARTIFACT_ENTRY_INVALID")
        if (
            entry["source_lane"] != clock
            or entry["params_payload_sha256"] is None
            or isinstance(entry["train_ohlcv_rows"], bool)
            or int(entry["train_ohlcv_rows"]) <= VOLATILITY_SQUEEZE_PREFIX_ROWS
        ):
            raise RuntimeError("VOLATILITY_SQUEEZE_ARTIFACT_ENTRY_INVALID")
        _require_bar_grid(entry["bar_grid"], timeframe=clock)
        for key in (
            "params_file_sha256",
            "params_payload_sha256",
            "source_sha256",
            "train_ohlcv_sha256",
        ):
            _require_sha256(
                entry[key],
                label=f"VOLATILITY_SQUEEZE_{clock}_{key.upper()}",
            )
        params_path = Path(str(entry["params_artifact"])).expanduser()
        source_path = Path(str(entry["source_artifact"])).expanduser()
        if (
            not params_path.is_absolute()
            or not source_path.is_absolute()
            or params_path.is_symlink()
            or source_path.is_symlink()
            or not params_path.is_file()
            or not source_path.is_file()
            or params_path.resolve(strict=True) != params_path
            or source_path.resolve(strict=True) != source_path
            or _sha256_file(source_path) != entry["source_sha256"]
        ):
            raise RuntimeError("VOLATILITY_SQUEEZE_ARTIFACT_ENTRY_INVALID")
        params = load_volatility_squeeze_params(
            params_path,
            expected_sha256=entry["params_file_sha256"],
            timeframe=clock,
        )
        provenance = params["source_provenance"]
        if (
            params["contract_sha256"] != entry["params_payload_sha256"]
            or params["train_lineage_sha256"] != lineage["train_lineage_sha256"]
            or params["train_ohlcv_rows"] != entry["train_ohlcv_rows"]
            or params["train_ohlcv_sha256"] != entry["train_ohlcv_sha256"]
            or provenance["source_artifact"] != entry["source_artifact"]
            or provenance["source_sha256"] != entry["source_sha256"]
            or provenance["source_schema_version"] != entry["source_schema_version"]
            or provenance["source_lane"] != clock
            or provenance["bar_grid"] != entry["bar_grid"]
        ):
            raise RuntimeError("VOLATILITY_SQUEEZE_ARTIFACT_ENTRY_MISMATCH")
    return observed


def fit_volatility_squeeze_artifact_manifest(
    closed_ohlcv_by_clock: Mapping[str, pd.DataFrame],
    *,
    declared_train_window_start: Any,
    declared_train_window_end: Any,
    source_provenance_by_clock: Mapping[str, Mapping[str, Any]],
    output_dir: Path,
) -> Path:
    """Fit and publish one TRAIN-only artifact for each of the six clocks."""

    if (
        not isinstance(closed_ohlcv_by_clock, Mapping)
        or tuple(closed_ohlcv_by_clock) != VOLATILITY_SQUEEZE_CLOCKS
        or not isinstance(source_provenance_by_clock, Mapping)
        or tuple(source_provenance_by_clock) != VOLATILITY_SQUEEZE_CLOCKS
    ):
        raise RuntimeError("VOLATILITY_SQUEEZE_SIX_CLOCK_INPUT_INVALID")
    start = _require_utc_timestamp(
        declared_train_window_start,
        label="VOLATILITY_SQUEEZE_TRAIN_WINDOW_START",
    )
    end = _require_utc_timestamp(
        declared_train_window_end,
        label="VOLATILITY_SQUEEZE_TRAIN_WINDOW_END",
    )
    if start > end:
        raise RuntimeError("VOLATILITY_SQUEEZE_TRAIN_WINDOW_INVALID")
    target = Path(output_dir).expanduser()
    if (
        not target.is_absolute()
        or target.is_symlink()
        or not target.is_dir()
        or target.resolve(strict=True) != target
    ):
        raise RuntimeError("VOLATILITY_SQUEEZE_OUTPUT_DIR_INVALID")
    manifest_path = target / "manifest.json"
    expected_names = {
        "manifest.json",
        *(f"{clock.lower()}_params.json" for clock in VOLATILITY_SQUEEZE_CLOCKS),
    }
    if any(target.iterdir()) or any(
        (target / name).exists() or (target / name).is_symlink()
        for name in expected_names
    ):
        raise RuntimeError("VOLATILITY_SQUEEZE_ARTIFACT_TARGET_EXISTS")

    artifacts: dict[str, Any] = {}
    common_lineage: dict[str, Any] | None = None
    for clock in VOLATILITY_SQUEEZE_CLOCKS:
        full_source = _validate_closed_ohlcv(
            closed_ohlcv_by_clock[clock],
            timeframe=clock,
            context=f"VOLATILITY_SQUEEZE_{clock}_FIT_SOURCE",
        )
        train = full_source.loc[(full_source.index >= start) & (full_source.index <= end)]
        if len(train) <= VOLATILITY_SQUEEZE_PREFIX_ROWS:
            raise RuntimeError(f"VOLATILITY_SQUEEZE_{clock}_TRAIN_WINDOW_TOO_SHORT")
        provenance = _require_source_provenance(
            source_provenance_by_clock[clock],
            timeframe=clock,
        )
        lineage = _train_lineage_payload(
            provenance,
            declared_train_window_start=start,
            declared_train_window_end=end,
        )
        if common_lineage is None:
            common_lineage = lineage
        elif lineage != common_lineage:
            raise RuntimeError("VOLATILITY_SQUEEZE_COMMON_TRAIN_LINEAGE_MISMATCH")
        params = fit_volatility_squeeze_params(
            train,
            timeframe=clock,
            declared_train_window_start=start,
            declared_train_window_end=end,
            source_provenance=provenance,
        )
        params_path = write_volatility_squeeze_params(
            target / f"{clock.lower()}_params.json",
            params,
        )
        artifacts[clock] = {
            "params_artifact": str(params_path),
            "params_file_sha256": _sha256_file(params_path),
            "params_payload_sha256": params["contract_sha256"],
            "source_artifact": provenance["source_artifact"],
            "source_sha256": provenance["source_sha256"],
            "source_schema_version": provenance["source_schema_version"],
            "source_lane": clock,
            "bar_grid": provenance["bar_grid"],
            "train_ohlcv_rows": params["train_ohlcv_rows"],
            "train_ohlcv_sha256": params["train_ohlcv_sha256"],
        }
    if common_lineage is None:
        raise RuntimeError("VOLATILITY_SQUEEZE_COMMON_TRAIN_LINEAGE_MISSING")
    manifest: dict[str, Any] = {
        "schema_version": VOLATILITY_SQUEEZE_ARTIFACT_MANIFEST_SCHEMA_VERSION,
        "owner": VOLATILITY_SQUEEZE_STATE_OWNER,
        "clocks": list(VOLATILITY_SQUEEZE_CLOCKS),
        "feature_names": list(VOLATILITY_SQUEEZE_FEATURE_NAMES),
        "common_train_lineage": common_lineage,
        "artifacts": artifacts,
    }
    manifest["contract_sha256"] = _canonical_sha256(manifest)
    _require_artifact_manifest_payload(manifest)
    encoded = (json.dumps(manifest, indent=2, allow_nan=False) + "\n").encode("utf-8")
    with manifest_path.open("xb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    return manifest_path.resolve(strict=True)


def load_volatility_squeeze_artifact_manifest(
    path: Path,
    *,
    expected_sha256: str,
) -> VolatilitySqueezeArtifactSet:
    """Load a complete six-clock manifest and all exact bound artifacts."""

    artifact = Path(path).expanduser()
    if (
        not artifact.is_absolute()
        or artifact.is_symlink()
        or not artifact.is_file()
        or artifact.resolve(strict=True) != artifact
    ):
        raise RuntimeError("VOLATILITY_SQUEEZE_ARTIFACT_MANIFEST_MISSING")
    expected_inventory = {
        "manifest.json",
        *(f"{clock.lower()}_params.json" for clock in VOLATILITY_SQUEEZE_CLOCKS),
    }
    if (
        artifact.name != "manifest.json"
        or {entry.name for entry in artifact.parent.iterdir()} != expected_inventory
    ):
        raise RuntimeError("VOLATILITY_SQUEEZE_ARTIFACT_INVENTORY_INVALID")
    expected = _require_sha256(
        expected_sha256,
        label="VOLATILITY_SQUEEZE_MANIFEST_FILE",
    )
    data = artifact.read_bytes()
    observed_file_sha = hashlib.sha256(data).hexdigest()
    if observed_file_sha != expected:
        raise RuntimeError("VOLATILITY_SQUEEZE_ARTIFACT_MANIFEST_FILE_HASH_MISMATCH")
    try:
        payload = json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("VOLATILITY_SQUEEZE_ARTIFACT_MANIFEST_JSON_INVALID") from exc
    manifest = _require_artifact_manifest_payload(payload)
    params_by_clock: dict[str, dict[str, Any]] = {}
    file_sha_by_clock: dict[str, str] = {}
    for clock in VOLATILITY_SQUEEZE_CLOCKS:
        entry = manifest["artifacts"][clock]
        params_by_clock[clock] = load_volatility_squeeze_params(
            Path(entry["params_artifact"]),
            expected_sha256=entry["params_file_sha256"],
            timeframe=clock,
        )
        file_sha_by_clock[clock] = entry["params_file_sha256"]
    result = VolatilitySqueezeArtifactSet(
        manifest_path=artifact,
        manifest_file_sha256=observed_file_sha,
        manifest_payload_sha256=manifest["contract_sha256"],
        train_lineage_sha256=manifest["common_train_lineage"]["train_lineage_sha256"],
        params_by_clock=params_by_clock,
        params_file_sha256_by_clock=file_sha_by_clock,
    )
    return require_volatility_squeeze_artifact_set(result)


@dataclass(frozen=True)
class VolatilitySqueezeCarryState:
    """Minimal exact carry for chunk-identical causal evaluation."""

    timeframe: str
    params_sha256: str
    trailing_closes: tuple[float, ...] = ()
    latent_state: int | None = None
    bars_in_squeeze: int = 0
    last_release_duration: int | None = None
    release_age_bars: int | None = None
    rows_seen: int = 0
    last_timestamp_ns: int | None = None


def _next_latent_state(
    relative_bandwidth: float,
    *,
    previous_state: int | None,
    means: np.ndarray,
    variances: np.ndarray,
    transition: np.ndarray,
    initial: np.ndarray,
) -> int:
    emission = _log_normal_density(
        np.asarray([relative_bandwidth], dtype=np.float64),
        means,
        variances,
    )[0]
    prior = initial if previous_state is None else transition[previous_state]
    return int(np.argmax(np.log(prior) + emission))


def compute_volatility_squeeze_state(
    ohlcv: pd.DataFrame,
    *,
    timeframe: str,
    params: Mapping[str, Any],
    carry: VolatilitySqueezeCarryState | None = None,
) -> tuple[pd.DataFrame, VolatilitySqueezeCarryState]:
    """Compute one clock's causal state and return exact continuation carry.

    ``duration_at_release`` is event-local: it is zero when no release occurs
    and the completed squeeze duration on the release row. ``release_age`` is
    a left-censored causal clock. It starts at zero on the first fully warmed
    state observation, advances until a release and resets to zero on release.
    This keeps absence-of-an-observed-release explicit and finite without
    fabricating a pre-history event or filling an unknown market value.
    """

    clock = _require_timeframe(timeframe)
    source = _validate_closed_ohlcv(
        ohlcv,
        timeframe=clock,
        context="VOLATILITY_SQUEEZE_RUNTIME",
    )
    frozen = require_volatility_squeeze_params(params, timeframe=clock)
    params_sha = str(frozen["contract_sha256"])
    if carry is None:
        state = VolatilitySqueezeCarryState(
            timeframe=clock,
            params_sha256=params_sha,
        )
    else:
        if (
            not isinstance(carry, VolatilitySqueezeCarryState)
            or carry.timeframe != clock
            or carry.params_sha256 != params_sha
            or len(carry.trailing_closes) > VOLATILITY_SQUEEZE_PREFIX_ROWS
            or carry.latent_state not in {None, 0, 1}
            or carry.rows_seen < len(carry.trailing_closes)
            or carry.bars_in_squeeze < 0
            or (carry.last_release_duration is not None and carry.last_release_duration <= 0)
            or (carry.release_age_bars is not None and carry.release_age_bars < 0)
        ):
            raise RuntimeError("VOLATILITY_SQUEEZE_CARRY_INVALID")
        if (
            carry.last_timestamp_ns is not None
            and int(source.index[0].value) <= carry.last_timestamp_ns
        ):
            raise RuntimeError("VOLATILITY_SQUEEZE_CARRY_TIME_OVERLAP")
        state = carry

    fit = frozen["fit"]
    means = np.asarray(fit["bandwidth_mean"], dtype=np.float64)
    variances = np.asarray(fit["bandwidth_variance"], dtype=np.float64)
    transition = np.asarray(fit["transition_probability"], dtype=np.float64)
    initial = np.asarray(fit["initial_probability"], dtype=np.float64)
    close = source["close"].to_numpy(dtype=np.float64)
    history = np.asarray((*state.trailing_closes, *close), dtype=np.float64)
    bandwidth = bollinger_relative_bandwidth(history) if len(history) >= BOLLINGER_PERIOD_BARS else np.full(len(history), np.nan)
    chunk_bandwidth = bandwidth[len(state.trailing_closes):]

    out = np.full(
        (len(source), len(VOLATILITY_SQUEEZE_FEATURE_NAMES)),
        np.nan,
        dtype=np.float64,
    )
    latent_state = state.latent_state
    bars_in_squeeze = int(state.bars_in_squeeze)
    last_release_duration = state.last_release_duration
    release_age = state.release_age_bars
    for row, width in enumerate(chunk_bandwidth):
        if not np.isfinite(width):
            continue
        previous_state = latent_state
        latent_state = _next_latent_state(
            float(width),
            previous_state=previous_state,
            means=means,
            variances=variances,
            transition=transition,
            initial=initial,
        )
        release_age = 0 if release_age is None else release_age + 1
        active = latent_state == 0
        released = previous_state == 0 and latent_state == 1
        if active:
            bars_in_squeeze = bars_in_squeeze + 1 if previous_state == 0 else 1
        else:
            if released:
                if bars_in_squeeze <= 0:
                    raise RuntimeError("VOLATILITY_SQUEEZE_RELEASE_DURATION_INVALID")
                last_release_duration = bars_in_squeeze
                release_age = 0
            bars_in_squeeze = 0
        out[row, 0] = float(active)
        out[row, 1] = float(bars_in_squeeze)
        out[row, 2] = float(released)
        out[row, 3] = float(last_release_duration) if released else 0.0
        out[row, 4] = float(release_age)

    result = pd.DataFrame(
        out,
        index=source.index,
        columns=list(VOLATILITY_SQUEEZE_FEATURE_NAMES),
        dtype=np.float64,
    )
    for column in result.columns:
        finite = np.isfinite(result[column].to_numpy(dtype=np.float64))
        if finite.any() and not finite[int(np.argmax(finite)):].all():
            raise RuntimeError(
                f"VOLATILITY_SQUEEZE_OUTPUT_WARMUP_NOT_PREFIX: {column}"
            )
    next_carry = VolatilitySqueezeCarryState(
        timeframe=clock,
        params_sha256=params_sha,
        trailing_closes=tuple(
            float(value) for value in history[-VOLATILITY_SQUEEZE_PREFIX_ROWS:]
        ),
        latent_state=latent_state,
        bars_in_squeeze=bars_in_squeeze,
        last_release_duration=last_release_duration,
        release_age_bars=release_age,
        rows_seen=state.rows_seen + len(source),
        last_timestamp_ns=int(source.index[-1].value),
    )
    return result.astype(np.float32), next_carry


__all__ = [
    "BOLLINGER_PERIOD_BARS",
    "BOLLINGER_STD_MULTIPLIER",
    "RETIRED_TRAIN_LINEAGE_KEYS",
    "VOLATILITY_SQUEEZE_CLOCKS",
    "VOLATILITY_SQUEEZE_CLOCK_CONTRACT",
    "VOLATILITY_SQUEEZE_ARTIFACT_MANIFEST_SCHEMA_VERSION",
    "VOLATILITY_SQUEEZE_FEATURE_NAMES",
    "VOLATILITY_SQUEEZE_PREFIX_ROWS",
    "VOLATILITY_SQUEEZE_STATE_SCHEMA_VERSION",
    "VolatilitySqueezeArtifactSet",
    "VolatilitySqueezeCarryState",
    "bollinger_relative_bandwidth",
    "compute_volatility_squeeze_state",
    "fit_volatility_squeeze_artifact_manifest",
    "fit_volatility_squeeze_params",
    "load_volatility_squeeze_artifact_manifest",
    "load_volatility_squeeze_params",
    "require_volatility_squeeze_artifact_binding",
    "require_volatility_squeeze_artifact_set",
    "require_volatility_squeeze_params",
    "volatility_squeeze_bar_grid",
    "write_volatility_squeeze_params",
]
