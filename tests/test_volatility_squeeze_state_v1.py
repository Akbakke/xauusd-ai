from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gx1.features.entry_specialist_feature_groups_v1 import (
    group_features_by_specialist,
)
from gx1.features.volatility_squeeze_state_v1 import (
    VOLATILITY_SQUEEZE_CLOCKS,
    VOLATILITY_SQUEEZE_CLOCK_CONTRACT,
    VOLATILITY_SQUEEZE_FEATURE_NAMES,
    VOLATILITY_SQUEEZE_PREFIX_ROWS,
    bollinger_relative_bandwidth,
    compute_volatility_squeeze_state,
    fit_volatility_squeeze_params,
    load_volatility_squeeze_params,
    require_volatility_squeeze_params,
    volatility_squeeze_bar_grid,
    write_volatility_squeeze_params,
)
from gx1.features.volatility_squeeze_state_v1 import _fit_two_state_model


_FREQ = {
    "M1": "1min",
    "M5": "5min",
    "M15": "15min",
    "H1": "1h",
    "H4": "4h",
    "D1": "1D",
}


def _closed_ohlcv(timeframe: str, rows: int = 1200) -> pd.DataFrame:
    rng = np.random.default_rng(20260814 + VOLATILITY_SQUEEZE_CLOCKS.index(timeframe))
    # Alternating long low/high-dispersion blocks make the two persistent
    # latent populations identifiable without encoding a decision threshold.
    block = (np.arange(rows) // 100) % 2
    sigma = np.where(block == 0, 0.00008, 0.0012)
    returns = rng.normal(0.0, sigma)
    close = 2_000.0 * np.exp(np.cumsum(returns))
    open_ = np.concatenate(([close[0]], close[:-1]))
    wick = np.maximum(np.abs(close - open_), close * sigma) + 0.01
    index = pd.date_range(
        "2022-01-02T22:00:00Z",
        periods=rows,
        freq=_FREQ[timeframe],
    )
    return pd.DataFrame(
        {
            "open": open_,
            "high": np.maximum(open_, close) + wick,
            "low": np.minimum(open_, close) - wick,
            "close": close,
            "volume": 100 + (np.arange(rows) % 37),
        },
        index=index,
    )


def _provenance(timeframe: str) -> dict[str, str]:
    immutable_test_source = Path(__file__).resolve(strict=True)
    source_sha = hashlib.sha256(immutable_test_source.read_bytes()).hexdigest()
    return {
        "source_artifact": str(immutable_test_source),
        "source_sha256": source_sha,
        "source_schema_version": "test_closed_ohlcv_v1",
        "source_lane": timeframe,
        "tape_manifest_artifact": str(immutable_test_source),
        "tape_manifest_sha256": source_sha,
        "pair_manifest_artifact": str(immutable_test_source),
        "pair_manifest_sha256": source_sha,
        "pair_generation_id": "test-pair-generation-v1",
        "pair_symbol": "XAUUSD",
        "train_split_id": "TRAIN",
        "clock_contract": VOLATILITY_SQUEEZE_CLOCK_CONTRACT,
        "bar_grid": volatility_squeeze_bar_grid(timeframe),
    }


def _fit(frame: pd.DataFrame, timeframe: str) -> dict:
    return fit_volatility_squeeze_params(
        frame,
        timeframe=timeframe,
        declared_train_window_start=frame.index[0],
        declared_train_window_end=frame.index[-1],
        source_provenance=_provenance(timeframe),
    )


def test_bandwidth_has_honest_full_window_prefix() -> None:
    close = _closed_ohlcv("M5", rows=100)["close"].to_numpy(dtype=np.float64)
    observed = bollinger_relative_bandwidth(close)
    expected = (
        4.0
        * pd.Series(close).rolling(20, min_periods=20).std(ddof=0)
        / pd.Series(close).rolling(20, min_periods=20).mean()
    ).to_numpy(dtype=np.float64)
    np.testing.assert_allclose(observed, expected, rtol=0.0, atol=0.0, equal_nan=True)
    assert np.isnan(observed[:VOLATILITY_SQUEEZE_PREFIX_ROWS]).all()
    assert np.isfinite(observed[VOLATILITY_SQUEEZE_PREFIX_ROWS:]).all()


def test_train_fit_is_clock_bound_hysteretic_and_hash_bound() -> None:
    frame = _closed_ohlcv("M5")
    params = _fit(frame, "M5")
    transition = np.asarray(params["fit"]["transition_probability"])
    assert transition[0, 0] > transition[0, 1]
    assert transition[1, 1] > transition[1, 0]
    assert params["fit"]["bandwidth_mean"][0] < params["fit"]["bandwidth_mean"][1]
    assert params["fit"]["train_observation_count"] == len(frame) - 19

    with pytest.raises(RuntimeError, match="TIMEFRAME|CONTRACT"):
        require_volatility_squeeze_params(params, timeframe="M1")
    tampered = json.loads(json.dumps(params))
    tampered["fit"]["bandwidth_mean"][0] *= 0.5
    with pytest.raises(RuntimeError, match="HASH_MISMATCH"):
        require_volatility_squeeze_params(tampered, timeframe="M5")


def test_fit_uses_only_declared_train_rows() -> None:
    frame = _closed_ohlcv("M5")
    cutoff = frame.index[899]
    changed_future = frame.copy()
    changed_future.loc[changed_future.index > cutoff, "close"] *= 1.25
    changed_future.loc[changed_future.index > cutoff, "open"] *= 1.25
    changed_future.loc[changed_future.index > cutoff, "high"] *= 1.25
    changed_future.loc[changed_future.index > cutoff, "low"] *= 1.25
    left = fit_volatility_squeeze_params(
        frame.loc[:cutoff],
        timeframe="M5",
        declared_train_window_start=frame.index[0],
        declared_train_window_end=cutoff,
        source_provenance=_provenance("M5"),
    )
    right = fit_volatility_squeeze_params(
        changed_future.loc[:cutoff],
        timeframe="M5",
        declared_train_window_start=frame.index[0],
        declared_train_window_end=cutoff,
        source_provenance=_provenance("M5"),
    )
    assert left["fit"] == right["fit"]
    assert left["train_ohlcv_sha256"] == right["train_ohlcv_sha256"]
    assert left["input_ohlcv_sha256"] == right["input_ohlcv_sha256"]
    assert left["contract_sha256"] == right["contract_sha256"]


def test_time_permutation_is_rejected_and_degenerate_fits_fail_closed() -> None:
    frame = _closed_ohlcv("M5")
    permutation = np.arange(len(frame))[::-1]
    with pytest.raises(RuntimeError, match="TIME_INDEX_INVALID"):
        fit_volatility_squeeze_params(
            frame.iloc[permutation],
            timeframe="M5",
            declared_train_window_start=frame.index[0],
            declared_train_window_end=frame.index[-1],
            source_provenance=_provenance("M5"),
        )
    with pytest.raises(RuntimeError, match="UNIDENTIFIABLE"):
        _fit_two_state_model(np.ones(200, dtype=np.float64))
    alternating = np.tile(np.asarray([0.1, 1.0], dtype=np.float64), 100)
    with pytest.raises(RuntimeError, match="UNIDENTIFIABLE|HYSTERESIS_ABSENT"):
        _fit_two_state_model(alternating)


def test_fit_rejects_wrong_native_clock_and_runtime_has_no_bare_payload_route() -> None:
    frame = _closed_ohlcv("M1")
    with pytest.raises(RuntimeError, match="TIME_INDEX_INVALID"):
        fit_volatility_squeeze_params(
            frame,
            timeframe="M5",
            declared_train_window_start=frame.index[0],
            declared_train_window_end=frame.index[-1],
            source_provenance=_provenance("M5"),
        )
    with pytest.raises(RuntimeError, match="PARAMS_KEYS_INVALID"):
        compute_volatility_squeeze_state(
            frame,
            timeframe="M1",
            params={},
        )


def test_runtime_events_are_genuine_edges_with_raw_duration_and_age() -> None:
    frame = _closed_ohlcv("M5")
    params = _fit(frame, "M5")
    out, _ = compute_volatility_squeeze_state(
        frame,
        timeframe="M5",
        params=params,
    )
    active = out["volatility.squeeze_active"].to_numpy(dtype=np.float64)
    duration = out["volatility.bars_in_squeeze"].to_numpy(dtype=np.float64)
    event = out["volatility.squeeze_release_event"].to_numpy(dtype=np.float64)
    at_release = out["volatility.duration_at_release"].to_numpy(dtype=np.float64)
    age = out["volatility.squeeze_release_age_bars"].to_numpy(dtype=np.float64)

    assert np.isnan(out.iloc[:19].to_numpy(dtype=np.float64)).all()
    release_rows = np.flatnonzero(event == 1.0)
    assert len(release_rows) >= 2
    for row in release_rows:
        assert active[row - 1] == 1.0
        assert active[row] == 0.0
        assert event[row - 1] == 0.0
        assert at_release[row] == duration[row - 1]
        assert age[row] == 0.0
        if row + 1 < len(age):
            assert age[row + 1] == 1.0
    first_release = int(release_rows[0])
    assert np.isnan(at_release[:VOLATILITY_SQUEEZE_PREFIX_ROWS]).all()
    assert np.isnan(age[:VOLATILITY_SQUEEZE_PREFIX_ROWS]).all()
    assert np.all(at_release[VOLATILITY_SQUEEZE_PREFIX_ROWS:first_release] == 0.0)
    np.testing.assert_array_equal(
        age[VOLATILITY_SQUEEZE_PREFIX_ROWS:first_release],
        np.arange(first_release - VOLATILITY_SQUEEZE_PREFIX_ROWS),
    )
    assert np.all(at_release[(event == 0.0) & np.isfinite(event)] == 0.0)


def test_chunk_carry_and_prefix_causality_are_exact() -> None:
    frame = _closed_ohlcv("M1")
    params = _fit(frame, "M1")
    full, _ = compute_volatility_squeeze_state(
        frame,
        timeframe="M1",
        params=params,
    )
    prefix, carry = compute_volatility_squeeze_state(
        frame.iloc[:437],
        timeframe="M1",
        params=params,
    )
    suffix, carry2 = compute_volatility_squeeze_state(
        frame.iloc[437:],
        timeframe="M1",
        params=params,
        carry=carry,
    )
    combined = pd.concat([prefix, suffix])
    np.testing.assert_array_equal(
        combined.to_numpy(dtype=np.float32),
        full.to_numpy(dtype=np.float32),
    )
    # First output row of a continued chunk is identical to the same row from
    # full-history computation, including its event/duration memory.
    np.testing.assert_array_equal(
        suffix.iloc[0].to_numpy(dtype=np.float32),
        full.iloc[437].to_numpy(dtype=np.float32),
    )
    assert carry2.rows_seen == len(frame)

    earlier = frame.iloc[:700]
    earlier_out, _ = compute_volatility_squeeze_state(
        earlier,
        timeframe="M1",
        params=params,
    )
    np.testing.assert_array_equal(
        earlier_out.to_numpy(dtype=np.float32),
        full.iloc[:700].to_numpy(dtype=np.float32),
    )


def test_release_on_first_suffix_row_is_exactly_once_at_chunk_boundary() -> None:
    frame = _closed_ohlcv("M5")
    params = _fit(frame, "M5")
    full, _ = compute_volatility_squeeze_state(
        frame,
        timeframe="M5",
        params=params,
    )
    event = full["volatility.squeeze_release_event"].to_numpy(dtype=np.float64)
    boundary = int(np.flatnonzero(event == 1.0)[1])
    left, carry = compute_volatility_squeeze_state(
        frame.iloc[:boundary],
        timeframe="M5",
        params=params,
    )
    right, _ = compute_volatility_squeeze_state(
        frame.iloc[boundary:],
        timeframe="M5",
        params=params,
        carry=carry,
    )
    assert left["volatility.squeeze_release_event"].iloc[-1] == 0.0
    assert right["volatility.squeeze_release_event"].iloc[0] == 1.0
    assert right["volatility.squeeze_release_event"].iloc[1] == 0.0
    np.testing.assert_array_equal(
        pd.concat([left, right]).to_numpy(dtype=np.float32),
        full.to_numpy(dtype=np.float32),
    )


@pytest.mark.parametrize("timeframe", VOLATILITY_SQUEEZE_CLOCKS)
def test_same_owner_runs_independently_on_all_six_native_clocks(timeframe: str) -> None:
    frame = _closed_ohlcv(timeframe)
    params = _fit(frame, timeframe)
    out, carry = compute_volatility_squeeze_state(
        frame,
        timeframe=timeframe,
        params=params,
    )
    assert tuple(out.columns) == VOLATILITY_SQUEEZE_FEATURE_NAMES
    assert out.index.equals(frame.index)
    assert carry.timeframe == timeframe
    assert carry.params_sha256 == params["contract_sha256"]
    assert (out["volatility.squeeze_release_event"].dropna().isin([0.0, 1.0])).all()
    live = out.iloc[VOLATILITY_SQUEEZE_PREFIX_ROWS:]
    assert np.isfinite(live.to_numpy(dtype=np.float64)).all()
    assert all(live[column].nunique() > 1 for column in VOLATILITY_SQUEEZE_FEATURE_NAMES)


def test_artifact_file_hash_is_required(tmp_path: Path) -> None:
    frame = _closed_ohlcv("H1")
    params = _fit(frame, "H1")
    path = write_volatility_squeeze_params(
        (tmp_path / "h1.squeeze.json").resolve(),
        params,
    )
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    loaded = load_volatility_squeeze_params(
        path,
        expected_sha256=sha,
        timeframe="H1",
    )
    assert loaded == params
    with pytest.raises(RuntimeError, match="HASH_MISMATCH"):
        load_volatility_squeeze_params(
            path,
            expected_sha256="0" * 64,
            timeframe="H1",
        )


def test_mutated_bound_source_manifest_invalidates_frozen_params(
    tmp_path: Path,
) -> None:
    bound = (tmp_path / "bound.json").resolve()
    bound.write_text("immutable-v1\n", encoding="utf-8")
    sha = hashlib.sha256(bound.read_bytes()).hexdigest()
    provenance = {
        "source_artifact": str(bound),
        "source_sha256": sha,
        "source_schema_version": "test_closed_ohlcv_v1",
        "source_lane": "M5",
        "tape_manifest_artifact": str(bound),
        "tape_manifest_sha256": sha,
        "pair_manifest_artifact": str(bound),
        "pair_manifest_sha256": sha,
        "pair_generation_id": "test-pair-generation-v1",
        "pair_symbol": "XAUUSD",
        "train_split_id": "TRAIN",
        "clock_contract": VOLATILITY_SQUEEZE_CLOCK_CONTRACT,
        "bar_grid": volatility_squeeze_bar_grid("M5"),
    }
    frame = _closed_ohlcv("M5")
    params = fit_volatility_squeeze_params(
        frame,
        timeframe="M5",
        declared_train_window_start=frame.index[0],
        declared_train_window_end=frame.index[-1],
        source_provenance=provenance,
    )
    bound.write_text("mutated-v2\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="SOURCE_PROVENANCE_INVALID"):
        require_volatility_squeeze_params(params, timeframe="M5")


def test_all_state_primitives_route_only_to_volatility_specialist() -> None:
    grouped = group_features_by_specialist(VOLATILITY_SQUEEZE_FEATURE_NAMES)
    assert grouped["vol_compression_encoder"] == list(
        VOLATILITY_SQUEEZE_FEATURE_NAMES
    )
    assert sum(len(values) for values in grouped.values()) == len(
        VOLATILITY_SQUEEZE_FEATURE_NAMES
    )


def test_routing_names_are_adopted_on_both_manifest_bound_surfaces() -> None:
    from gx1.contracts.entry_model_native_signal_v1 import MODEL_NATIVE_BASE_FIELDS
    from gx1.features.entry_model_native_feature_layers_v1 import (
        MODEL_NATIVE_MANDATORY_SELECTED_FIELDS,
    )
    from gx1.features.htf_features import MULTI_TF_PER_BAR_FEATURES_V4

    assert set(VOLATILITY_SQUEEZE_FEATURE_NAMES).issubset(
        set(MODEL_NATIVE_MANDATORY_SELECTED_FIELDS)
    )
    assert set(VOLATILITY_SQUEEZE_FEATURE_NAMES).issubset(
        set(MULTI_TF_PER_BAR_FEATURES_V4)
    )
    assert set(VOLATILITY_SQUEEZE_FEATURE_NAMES).isdisjoint(
        set(MODEL_NATIVE_BASE_FIELDS)
    )


def test_retired_split_manifest_keys_can_never_re_enter_the_lineage() -> None:
    """The squeeze fit is a chain PREREQUISITE, so it cannot bind a split manifest.

    Until 2026-08-15 the source provenance and the six-clock manifest's
    ``common_train_lineage`` both required a hash-bound
    ``split_manifest_artifact``/``split_manifest_sha256``. No split manifest can
    exist when this fit runs — the chain produces them from the dataset this
    fit feeds — so the binding was unsatisfiable. Both payload key sets must
    now reject those keys outright.
    """

    from gx1.features.volatility_squeeze_state_v1 import (
        RETIRED_TRAIN_LINEAGE_KEYS,
        _COMMON_TRAIN_LINEAGE_KEYS,
        _SOURCE_PROVENANCE_KEYS,
        _require_source_provenance,
    )

    assert RETIRED_TRAIN_LINEAGE_KEYS
    assert RETIRED_TRAIN_LINEAGE_KEYS.isdisjoint(_SOURCE_PROVENANCE_KEYS)
    assert RETIRED_TRAIN_LINEAGE_KEYS.isdisjoint(_COMMON_TRAIN_LINEAGE_KEYS)

    provenance = _provenance("M5")
    assert _require_source_provenance(provenance, timeframe="M5")

    for retired in sorted(RETIRED_TRAIN_LINEAGE_KEYS):
        polluted = dict(provenance)
        polluted[retired] = provenance["tape_manifest_artifact"]
        with pytest.raises(
            RuntimeError, match="VOLATILITY_SQUEEZE_SOURCE_PROVENANCE_INVALID"
        ):
            _require_source_provenance(polluted, timeframe="M5")


def test_common_train_lineage_rejects_a_reintroduced_split_pointer(tmp_path) -> None:
    from gx1.features.volatility_squeeze_state_v1 import (
        RETIRED_TRAIN_LINEAGE_KEYS,
        _require_common_train_lineage,
        _train_lineage_payload,
    )

    provenance = _provenance("M5")
    lineage = _train_lineage_payload(
        provenance,
        declared_train_window_start=pd.Timestamp("2020-01-01T00:00:00Z"),
        declared_train_window_end=pd.Timestamp("2020-06-01T00:00:00Z"),
    )
    assert _require_common_train_lineage(lineage) == lineage
    assert RETIRED_TRAIN_LINEAGE_KEYS.isdisjoint(lineage)

    for retired in sorted(RETIRED_TRAIN_LINEAGE_KEYS):
        polluted = dict(lineage)
        polluted[retired] = lineage["tape_manifest_artifact"]
        with pytest.raises(
            RuntimeError, match="VOLATILITY_SQUEEZE_COMMON_TRAIN_LINEAGE_INVALID"
        ):
            _require_common_train_lineage(polluted)


def test_fit_cli_has_no_split_manifest_flag_and_still_binds_tape_and_pair(
    tmp_path, monkeypatch
) -> None:
    """The producer CLI must expose exactly the lineage the fit can satisfy.

    A retired ``--split-manifest``/``--split-manifest-sha256`` pair must be
    rejected as unrecognized, while ``--tape-manifest``/``--pair-manifest``
    stay required: those two artifacts exist before the fit and are real source
    authority.
    """

    import sys

    from gx1.scripts import fit_volatility_squeeze_artifacts_v1 as producer

    bound = Path(__file__).resolve(strict=True)
    sha = hashlib.sha256(bound.read_bytes()).hexdigest()
    base = [
        "fit_volatility_squeeze_artifacts_v1",
        "--m1-source", str(bound), "--m1-source-sha256", sha,
        "--m5-source", str(bound), "--m5-source-sha256", sha,
        "--tape-manifest", str(bound), "--tape-manifest-sha256", sha,
        "--pair-manifest", str(bound), "--pair-manifest-sha256", sha,
        "--pair-generation-id", "test-pair-generation-v1",
        "--train-window-start", "2020-01-01T00:00:00+00:00",
        "--train-window-end", "2020-06-01T00:00:00+00:00",
        "--output-dir", str(tmp_path),
    ]

    monkeypatch.setattr(sys, "argv", [*base, "--split-manifest", str(bound)])
    with pytest.raises(SystemExit):
        producer.main()

    monkeypatch.setattr(sys, "argv", [*base, "--split-manifest-sha256", sha])
    with pytest.raises(SystemExit):
        producer.main()

    # Dropping a lineage flag that IS satisfiable must still fail closed.
    for retired_flag in ("--tape-manifest", "--pair-manifest"):
        pruned = list(base)
        index = pruned.index(retired_flag)
        del pruned[index : index + 2]
        monkeypatch.setattr(sys, "argv", pruned)
        with pytest.raises(SystemExit):
            producer.main()
