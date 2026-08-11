"""Contract tests for the V29 Exit M1-lane registry fit orchestration.

Mirrors the M5-side fit tests for the M1 lane: the one fit owner in
``gx1.features.htf_features`` fits the Exit local lane's level tolerance and
trendline band on the declared native-M1 TRAIN window (same underlying fit
owners ``fit_level_registry_tolerance`` / ``fit_trendline_tolerance``), the
payload carries the rule-2f provenance shape, and the M1 materializer
resolves it fail-closed (cross-lane payloads and provenance-free bare key
dicts rejected; no default exists).

All series here are synthetic: per rule 2c they prove code properties
(determinism, contract shapes, fail-closed behaviour), never production
behaviour.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gx1.contracts.entry_exit_feature_base_v1 import (
    EXIT_FEATURE_SEQUENCE_BARS,
)
from gx1.features.htf_features import (
    V29_REGISTRY_M1_LANE_MANIFEST_KEY,
    V29_REGISTRY_M1_LANE_PARAMS_SCHEMA_VERSION,
    fit_v29_registry_m1_lane_params_from_m1,
    require_v29_registry_m1_lane_params,
)
from gx1.scripts import materialize_entry_exit_m1_feature_base_v1 as feature_producer
from gx1.scripts import build_entry_exit_m1_enriched_frame_v1 as enriched_producer
from tests.htf_v29_registry_test_support import synthetic_v29_registry_constants


_TEST_Q = 0.5  # explicit test input mirroring the adopted recipe value


def _m1_frame(n: int = 720, seed: int = 7) -> pd.DataFrame:
    """Synthetic native-M1 OHLCV on the exact 1-minute UTC grid."""

    rng = np.random.default_rng(seed)
    close = 4000.0 + np.cumsum(rng.normal(0.0, 0.8, size=n))
    spread = np.abs(rng.normal(0.6, 0.25, size=n)) + 0.05
    high = close + spread
    low = close - spread
    open_ = np.concatenate(([close[0]], close[:-1]))
    index = pd.date_range("2026-01-05", periods=n, freq="1min", tz="UTC")
    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": rng.integers(1, 50, size=n).astype(np.int64),
        },
        index=index,
    )


def _fit(frame: pd.DataFrame, *, window_end=None, q: float = _TEST_Q) -> dict:
    return fit_v29_registry_m1_lane_params_from_m1(
        frame,
        level_tol_quantile_q=q,
        declared_train_window_end=(
            frame.index[-1] if window_end is None else window_end
        ),
        exit_m1_seq_len=EXIT_FEATURE_SEQUENCE_BARS,
    )


def test_m1_lane_fit_deterministic_with_rule2f_provenance() -> None:
    frame = _m1_frame()
    first = _fit(frame)
    second = _fit(frame)
    assert first == second
    assert require_v29_registry_m1_lane_params(first) == first

    assert first["schema_version"] == V29_REGISTRY_M1_LANE_PARAMS_SCHEMA_VERSION
    assert first["level_tol_quantile_q"] == _TEST_Q
    assert first["level_tol_atr"] > 0.0
    assert first["exit_m1"]["seq_len"] == EXIT_FEATURE_SEQUENCE_BARS
    assert first["exit_m1"]["trendline_band_atr"] > 0.0

    provenance = first["provenance"]
    assert provenance["fit_owner"] == (
        "gx1.features.htf_features.fit_v29_registry_m1_lane_params_from_m1"
    )
    assert provenance["n_train_m1_rows"] == len(frame)
    # rule 2f: the level fit states sample size and sampling bound on the
    # native M1 clock; the trendline fit states its complete population size.
    level_tol = provenance["level_tol"]
    assert level_tol["tf"] == "m1"
    assert level_tol["sample_size"] > 0
    assert level_tol["quantile_prob_se"] > 0.0
    assert level_tol["tol_level_atr"] == first["level_tol_atr"]
    band = provenance["trendline_band"]
    assert band["timeframe"] == "M1"
    assert band["seq_len"] == EXIT_FEATURE_SEQUENCE_BARS
    assert band["n_candidates_measured"] > 0
    assert band["band_atr"] == first["exit_m1"]["trendline_band_atr"]


def test_m1_lane_fit_uses_only_declared_train_window_rows() -> None:
    frame = _m1_frame()
    cut = len(frame) // 2
    window_end = frame.index[cut - 1]
    sliced = _fit(frame, window_end=window_end)
    truncated = _fit(frame.iloc[:cut], window_end=window_end)
    # rule 18/2g: only rows at or before the declared end participate, so the
    # fit on the full frame with an early window equals the truncated fit.
    assert sliced == truncated
    assert sliced["provenance"]["n_train_m1_rows"] == cut
    assert sliced != _fit(frame)


def test_m1_lane_fit_fail_closed() -> None:
    frame = _m1_frame()
    with pytest.raises(RuntimeError, match="V29_REGISTRY"):
        _fit(frame, q=0.0)
    with pytest.raises(RuntimeError, match="M1_FIT_Q_INVALID"):
        _fit(frame, q=1.0)
    with pytest.raises(RuntimeError, match="M1_FIT_WINDOW_INVALID"):
        _fit(frame, window_end=pd.Timestamp("2026-01-05 00:30:00"))
    with pytest.raises(RuntimeError, match="M1_FIT_WINDOW_EMPTY"):
        _fit(frame, window_end=pd.Timestamp("2020-01-01", tz="UTC"))
    with pytest.raises(RuntimeError, match="M1_FIT_EXIT_SEQ_LEN_INVALID"):
        fit_v29_registry_m1_lane_params_from_m1(
            frame,
            level_tol_quantile_q=_TEST_Q,
            declared_train_window_end=frame.index[-1],
            exit_m1_seq_len=True,
        )
    off_grid = frame.copy()
    off_grid.index = pd.date_range(
        "2026-01-05", periods=len(frame), freq="30s", tz="UTC"
    )
    with pytest.raises(RuntimeError, match="HTF_INPUT_FAIL"):
        _fit(off_grid)
    naive = frame.copy()
    naive.index = naive.index.tz_localize(None)
    with pytest.raises((RuntimeError, TypeError)):
        _fit(naive)


def test_m1_lane_params_validator_fail_closed() -> None:
    valid = _fit(_m1_frame(n=480))

    with pytest.raises(RuntimeError, match="M1_LANE_PARAMS_MISSING"):
        require_v29_registry_m1_lane_params(None)
    with pytest.raises(RuntimeError, match="exact keys differ"):
        require_v29_registry_m1_lane_params(
            {k: v for k, v in valid.items() if k != "provenance"}
        )
    with pytest.raises(RuntimeError, match="exact keys differ"):
        require_v29_registry_m1_lane_params({**valid, "extra": 1})
    with pytest.raises(RuntimeError, match="schema_version"):
        require_v29_registry_m1_lane_params(
            {**valid, "schema_version": "htf_v4_v29_registry_constants_v1"}
        )
    with pytest.raises(RuntimeError, match="level_tol_quantile_recipe_key"):
        require_v29_registry_m1_lane_params(
            {**valid, "level_tol_quantile_recipe_key": "guessed"}
        )
    with pytest.raises(RuntimeError, match="level_tol_atr"):
        require_v29_registry_m1_lane_params({**valid, "level_tol_atr": 0.0})
    with pytest.raises(RuntimeError, match="exit_m1"):
        require_v29_registry_m1_lane_params(
            {**valid, "exit_m1": {"seq_len": EXIT_FEATURE_SEQUENCE_BARS}}
        )
    with pytest.raises(RuntimeError, match="seq_len"):
        require_v29_registry_m1_lane_params(
            {
                **valid,
                "exit_m1": {**valid["exit_m1"], "seq_len": True},
            }
        )
    with pytest.raises(RuntimeError, match="provenance"):
        require_v29_registry_m1_lane_params({**valid, "provenance": {}})
    # The M5 constants payload is a different lane and a different shape; it
    # must never satisfy the M1-lane contract.
    with pytest.raises(RuntimeError, match="M1_LANE_PARAMS_INVALID"):
        require_v29_registry_m1_lane_params(synthetic_v29_registry_constants())


def test_materializer_resolves_each_lane_fail_closed(tmp_path: Path) -> None:
    resolve = feature_producer._resolve_v29_registry_layer_params
    constants = synthetic_v29_registry_constants()
    lane = _fit(_m1_frame(n=480))

    constants_json = tmp_path / "constants.json"
    constants_json.write_text(json.dumps(constants), encoding="utf-8")
    cache_manifest_json = tmp_path / "cache_manifest.json"
    cache_manifest_json.write_text(
        json.dumps({"v29_registry_constants": constants}), encoding="utf-8"
    )
    lane_json = tmp_path / "m1_lane.json"
    lane_json.write_text(json.dumps(lane), encoding="utf-8")
    m1_manifest_json = tmp_path / "m1_enriched.manifest.json"
    m1_manifest_json.write_text(
        json.dumps({V29_REGISTRY_M1_LANE_MANIFEST_KEY: lane}), encoding="utf-8"
    )
    bare_json = tmp_path / "bare.json"
    bare_json.write_text(
        json.dumps(
            {
                "level_tol_atr": 1.0,
                "trendline_band_atr": 0.5,
                "trendline_seq_len": 96,
            }
        ),
        encoding="utf-8",
    )

    expected_m5 = {
        "level_tol_atr": float(constants["level_tol_atr"]["M5"]),
        "trendline_band_atr": float(constants["entry_m5"]["trendline_band_atr"]),
        "trendline_seq_len": int(constants["entry_m5"]["seq_len"]),
    }
    assert resolve(constants_json, timeframe="M5") == expected_m5
    assert resolve(cache_manifest_json, timeframe="M5") == expected_m5

    expected_m1 = {
        "level_tol_atr": float(lane["level_tol_atr"]),
        "trendline_band_atr": float(lane["exit_m1"]["trendline_band_atr"]),
        "trendline_seq_len": int(lane["exit_m1"]["seq_len"]),
    }
    assert resolve(lane_json, timeframe="M1") == expected_m1
    assert resolve(m1_manifest_json, timeframe="M1") == expected_m1

    # Cross-lane payloads fail closed: the M5 constants cannot supply the M1
    # lane and the M1-lane params cannot supply the M5 lane.
    with pytest.raises(RuntimeError, match="V29_REGISTRY"):
        resolve(constants_json, timeframe="M1")
    with pytest.raises(RuntimeError, match="V29_REGISTRY"):
        resolve(lane_json, timeframe="M5")
    # A provenance-free bare key dict is hand-authored evidence; both lanes
    # reject it (rule 2a/14 — no default, no soft pass-through).
    with pytest.raises(RuntimeError, match="V29_REGISTRY"):
        resolve(bare_json, timeframe="M5")
    with pytest.raises(RuntimeError, match="V29_REGISTRY"):
        resolve(bare_json, timeframe="M1")


@pytest.mark.parametrize("root_flag", ["--native-m1-root", "--native-m5-root"])
def test_enriched_cli_requires_fit_inputs_on_both_routes(
    monkeypatch: pytest.MonkeyPatch, root_flag: str
) -> None:
    called: list[dict[str, object]] = []
    monkeypatch.setattr(
        enriched_producer,
        "_build_enriched_frame",
        lambda **kwargs: called.append(kwargs) or {"decision": "PASS"},
    )
    argv = [
        "producer", root_flag, "/tmp/native",
        "--rank-reference-npz", "/tmp/rank.npz",
        "--rank-reference-sha256", "a" * 64,
        "--pair-manifest", "/tmp/pair.json",
        "--multi-tf-cache-dir", "/tmp/cache",
        "--output-parquet", "/tmp/enriched.parquet",
        "--manifest-path", "/tmp/enriched.json",
        "--checkpoint-dir", "/tmp/checkpoint",
        "--dataset-run-id", "run",
        "--pair-generation-id", "b" * 64,
    ]
    monkeypatch.setattr(sys, "argv", list(argv))
    with pytest.raises(SystemExit):
        enriched_producer.main()
    assert called == []

    monkeypatch.setattr(
        sys,
        "argv",
        [
            *argv,
            "--level-tol-quantile-q", "0.5",
            "--registry-fit-train-end", "2026-01-05T00:00:00Z",
        ],
    )
    enriched_producer.main()
    assert [call["timeframe"] for call in called] == [
        "M1" if root_flag == "--native-m1-root" else "M5"
    ]
    assert called[0]["level_tol_quantile_q"] == 0.5
    assert called[0]["registry_fit_train_end"] == "2026-01-05T00:00:00Z"
