from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gx1.features import htf_features as htf
from gx1.features.entry_model_native_feature_layers_v1 import (
    MODEL_NATIVE_MANDATORY_SELECTED_FIELDS,
    VOLATILITY_SQUEEZE_LOCAL_LAYER_FEATURE_NAMES,
    build_volatility_squeeze_local_layer,
    v29_layer_first_complete_time,
)
from gx1.features.volatility_squeeze_state_v1 import (
    VOLATILITY_SQUEEZE_CLOCKS,
    VOLATILITY_SQUEEZE_FEATURE_NAMES,
    compute_volatility_squeeze_state,
    load_volatility_squeeze_artifact_manifest,
    require_volatility_squeeze_artifact_binding,
)
from tests.htf_v29_registry_test_support import (
    synthetic_v29_registry_constants,
)
from tests.volatility_squeeze_test_support import (
    make_volatility_squeeze_artifact_set,
    synthetic_closed_ohlcv,
)


def test_six_clock_manifest_is_exact_and_rejects_file_or_payload_mutation(
    tmp_path: Path,
) -> None:
    artifacts = make_volatility_squeeze_artifact_set(tmp_path)
    assert tuple(artifacts.params_by_clock) == VOLATILITY_SQUEEZE_CLOCKS
    assert require_volatility_squeeze_artifact_binding(
        artifacts.binding()
    ).binding() == artifacts.binding()
    manifest = json.loads(artifacts.manifest_path.read_text(encoding="utf-8"))
    assert tuple(manifest["artifacts"]) == VOLATILITY_SQUEEZE_CLOCKS
    assert {
        row["params_artifact"] for row in manifest["artifacts"].values()
    } == {
        str(artifacts.manifest_path.parent / f"{clock.lower()}_params.json")
        for clock in VOLATILITY_SQUEEZE_CLOCKS
    }

    payload = dict(manifest)
    payload["feature_names"] = list(payload["feature_names"][:-1])
    artifacts.manifest_path.write_text(
        json.dumps(payload) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="MANIFEST_HASH|MANIFEST_INVALID"):
        load_volatility_squeeze_artifact_manifest(
            artifacts.manifest_path,
            expected_sha256=hashlib.sha256(
                artifacts.manifest_path.read_bytes()
            ).hexdigest(),
        )


def test_six_clock_manifest_fits_only_rows_inside_frozen_train_window(
    tmp_path: Path,
) -> None:
    common_index = pd.date_range(
        "2022-01-02T22:00:00Z",
        periods=180,
        freq="1D",
    )
    frames = {}
    for clock in VOLATILITY_SQUEEZE_CLOCKS:
        frame = synthetic_closed_ohlcv(clock, rows=len(common_index))
        frame.index = common_index
        frames[clock] = frame
    train_end = common_index[119]
    left = make_volatility_squeeze_artifact_set(
        tmp_path / "left",
        frames_by_clock=frames,
        declared_train_window_start=common_index[0],
        declared_train_window_end=train_end,
    )
    changed_future = {clock: frame.copy() for clock, frame in frames.items()}
    for frame in changed_future.values():
        future = frame.index > train_end
        for column in ("open", "high", "low", "close"):
            frame.loc[future, column] *= 1.25
    right = make_volatility_squeeze_artifact_set(
        tmp_path / "right",
        frames_by_clock=changed_future,
        declared_train_window_start=common_index[0],
        declared_train_window_end=train_end,
    )

    for clock in VOLATILITY_SQUEEZE_CLOCKS:
        assert left.require_params(clock) == right.require_params(clock)


@pytest.mark.parametrize("timeframe", ("M1", "M5"))
def test_local_surface_is_exact_owner_projection_and_requires_manifest(
    tmp_path: Path,
    timeframe: str,
) -> None:
    artifacts = make_volatility_squeeze_artifact_set(
        tmp_path / timeframe.lower()
    )
    source = synthetic_closed_ohlcv(timeframe)
    parquet = (tmp_path / f"{timeframe.lower()}-source.parquet").resolve()
    source.reset_index(names="time").to_parquet(parquet, index=False)
    raw, names = build_volatility_squeeze_local_layer(
        None,
        parquet,
        timeframe=timeframe,
        artifact_set=artifacts,
        raw_frame=True,
    )
    expected, _ = compute_volatility_squeeze_state(
        source,
        timeframe=timeframe,
        params=artifacts.require_params(timeframe),
    )
    expected.index.name = "time"
    pd.testing.assert_frame_equal(raw, expected, check_freq=False)
    assert tuple(names) == VOLATILITY_SQUEEZE_FEATURE_NAMES
    first = v29_layer_first_complete_time(raw, context="TEST_SQUEEZE_LOCAL")
    assert first == raw.index[19]
    assert np.isfinite(raw.loc[first].to_numpy(dtype=np.float64)).all()
    assert np.isfinite(raw.loc[first:].to_numpy(dtype=np.float64)).all()
    sample_times = raw.index[raw.index >= first][::7]
    aligned, aligned_names = build_volatility_squeeze_local_layer(
        pd.DataFrame({"time": sample_times}),
        parquet,
        timeframe=timeframe,
        artifact_set=artifacts,
    )
    np.testing.assert_array_equal(
        aligned,
        raw.loc[sample_times].to_numpy(dtype=np.float32),
    )
    assert tuple(aligned_names) == VOLATILITY_SQUEEZE_FEATURE_NAMES
    with pytest.raises(RuntimeError, match="ARTIFACT_SET_REQUIRED"):
        build_volatility_squeeze_local_layer(
            pd.DataFrame({"time": sample_times}),
            parquet,
            timeframe=timeframe,
            artifact_set=None,  # type: ignore[arg-type]
        )


def test_mtf_m5_lane_equals_same_owner_and_surface_has_no_adoption_gap(
    tmp_path: Path,
) -> None:
    artifacts = make_volatility_squeeze_artifact_set(tmp_path)
    source = synthetic_closed_ohlcv("M5", rows=1200)
    observed = htf.compute_per_bar_features_v4(
        source,
        timeframe="M5",
        v29_registry_constants=synthetic_v29_registry_constants(),
        volatility_squeeze_artifacts=artifacts,
    )
    expected, _ = compute_volatility_squeeze_state(
        source,
        timeframe="M5",
        params=artifacts.require_params("M5"),
    )
    np.testing.assert_array_equal(
        observed.loc[:, list(VOLATILITY_SQUEEZE_FEATURE_NAMES)].to_numpy(
            dtype=np.float32
        ),
        expected.to_numpy(dtype=np.float32),
    )
    assert tuple(observed.columns) == htf.MULTI_TF_PER_BAR_FEATURES_V4
    assert set(VOLATILITY_SQUEEZE_FEATURE_NAMES).issubset(
        MODEL_NATIVE_MANDATORY_SELECTED_FIELDS
    )
    assert set(VOLATILITY_SQUEEZE_FEATURE_NAMES).issubset(
        htf.MULTI_TF_PER_BAR_FEATURES_V4
    )
    assert tuple(VOLATILITY_SQUEEZE_LOCAL_LAYER_FEATURE_NAMES) == (
        VOLATILITY_SQUEEZE_FEATURE_NAMES
    )


def test_all_mtf_lanes_equal_same_owner_on_native_closed_ohlcv(
    tmp_path: Path,
) -> None:
    artifacts = make_volatility_squeeze_artifact_set(tmp_path)
    source = synthetic_closed_ohlcv("M5", rows=8_000)
    observed = htf.build_multi_tf_per_bar_features_v4(
        source,
        v29_registry_constants=synthetic_v29_registry_constants(),
        volatility_squeeze_artifacts=artifacts,
    )

    for timeframe, surface in observed.items():
        native = (
            source
            if timeframe == "M5"
            else htf._resample_ohlcv(source, timeframe)
        ).loc[surface.index]
        expected, _ = compute_volatility_squeeze_state(
            native,
            timeframe=timeframe,
            params=artifacts.require_params(timeframe),
        )
        np.testing.assert_array_equal(
            surface.loc[
                :, list(VOLATILITY_SQUEEZE_FEATURE_NAMES)
            ].to_numpy(dtype=np.float32),
            expected.to_numpy(dtype=np.float32),
        )
