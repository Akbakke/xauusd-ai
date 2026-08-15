"""Registry artifact and M1-lane fail-closed integration tests."""
from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gx1.contracts.entry_exit_feature_base_v1 import EXIT_FEATURE_SEQUENCE_BARS
from gx1.features.htf_features import (
    MULTI_TF_RESAMPLE_RULES,
    V29_REGISTRY_M1_LANE_PARAMS_SCHEMA_VERSION,
    fit_v29_registry_constants_from_m5,
    fit_v29_registry_m1_lane_params_from_m1,
    require_v29_registry_constants,
    require_v29_registry_m1_lane_params,
)
from gx1.scripts import build_entry_exit_m1_enriched_frame_v1 as enriched_producer
from gx1.scripts import materialize_entry_exit_m1_feature_base_v1 as feature_producer
from gx1.utils.artifact_primitives_v1 import canonical_json_sha256
from tests.htf_v29_registry_test_support import (
    synthetic_v29_registry_constants,
    synthetic_v29_registry_m1_lane_params,
    write_synthetic_m1_registry_manifest,
)


def _rehash(payload: dict) -> None:
    payload["contract_sha256"] = canonical_json_sha256(
        {key: value for key, value in payload.items() if key != "contract_sha256"}
    )


# --- declared TRAIN window bounds ------------------------------------------
# The V29 registry fit owners took only an UPPER bound until 2026-08-15, so
# every row the source happened to carry before the declared TRAIN start also
# entered the fit and the result was frozen as a TRAIN fit (rule 2g: the
# measurement must be taken where the decision is made).  The fixtures below
# execute the real fit owners on a source that deliberately extends before the
# declared start.
_PRE_START = "2026-01-01T00:00:00+00:00"
_TRAIN_START = "2026-01-05T00:00:00+00:00"
_INNER_FIT_END = "2026-01-06T00:00:00+00:00"
_TRAIN_END = "2026-01-08T00:00:00+00:00"


def _ohlcv(start: str, end: str, *, freq: str, seed: int) -> pd.DataFrame:
    index = pd.date_range(start, end, freq=freq, tz="UTC")
    rows = len(index)
    rng = np.random.default_rng(seed)
    close = 2000.0 + np.cumsum(rng.normal(0.0, 0.35, rows))
    high = close + np.abs(rng.normal(0.0, 0.30, rows)) + 0.05
    low = close - np.abs(rng.normal(0.0, 0.30, rows)) - 0.05
    open_ = np.concatenate(([close[0]], close[:-1]))
    return pd.DataFrame(
        {
            "open": open_,
            "high": np.maximum.reduce([high, open_, close]),
            "low": np.minimum.reduce([low, open_, close]),
            "close": close,
            "volume": rng.integers(1, 500, rows).astype(float),
        },
        index=index,
    )


def _fit_lineage(root: Path, *, clock: str) -> dict:
    root.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for name in ("source", "tape", "pair"):
        path = (root / f"{name}.json").resolve()
        path.write_text(json.dumps({"name": name}) + "\n", encoding="utf-8")
        paths[name] = path

    def sha(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    return {
        "source_artifact": str(paths["source"]),
        "source_sha256": sha(paths["source"]),
        "source_schema_version": "synthetic_closed_ohlcv_v1",
        "source_lane": clock,
        "tape_manifest_artifact": str(paths["tape"]),
        "tape_manifest_sha256": sha(paths["tape"]),
        "pair_manifest_artifact": str(paths["pair"]),
        "pair_manifest_sha256": sha(paths["pair"]),
        "train_split_id": "synthetic:TRAIN",
        "declared_train_window_start": _TRAIN_START,
        "declared_train_window_end": _TRAIN_END,
    }


def test_m1_lane_fit_lower_bound_binds_the_declared_train_population(
    tmp_path: Path,
) -> None:
    """A source reaching before the declared start must not reach the fit.

    This is the regression that no gate could see: with only an upper bound
    the four decision-bearing constants were fitted on the whole source
    history while the payload recorded a TRAIN fit.  The proof is an exact
    payload equality between a fit on the extended source and a fit on the
    source already trimmed to the declared window.
    """

    lineage = _fit_lineage(tmp_path / "m1", clock="M1")
    extended = _ohlcv(_PRE_START, _TRAIN_END, freq="1min", seed=7)
    trimmed = extended[extended.index >= pd.Timestamp(_TRAIN_START)]
    assert len(trimmed) < len(extended)

    def _fit(frame: pd.DataFrame) -> dict:
        return fit_v29_registry_m1_lane_params_from_m1(
            frame,
            declared_train_window_start=_TRAIN_START,
            declared_train_window_end=_TRAIN_END,
            declared_inner_fit_window_end=_INNER_FIT_END,
            source_provenance=lineage,
            exit_m1_seq_len=60,
        )

    on_trimmed = _fit(trimmed)
    on_extended = _fit(extended)
    assert on_extended == on_trimmed
    assert on_extended["contract_sha256"] == on_trimmed["contract_sha256"]
    # The population is the declared window, not everything at or before the
    # declared end.
    assert on_extended["provenance"]["n_train_m1_rows"] == len(trimmed)
    assert on_extended["declared_train_window_start"] == _TRAIN_START
    assert on_extended["provenance"]["declared_train_window_start"] == _TRAIN_START


def test_registry_fit_owners_reject_a_source_entirely_before_the_declared_start(
    tmp_path: Path,
) -> None:
    """Both lanes fail closed when no row lies inside the declared window."""

    before_start = _ohlcv(
        _PRE_START, "2026-01-04T23:59:00+00:00", freq="1min", seed=11
    )
    with pytest.raises(RuntimeError, match="M1_FIT_WINDOW_EMPTY"):
        fit_v29_registry_m1_lane_params_from_m1(
            before_start,
            declared_train_window_start=_TRAIN_START,
            declared_train_window_end=_TRAIN_END,
            declared_inner_fit_window_end=_INNER_FIT_END,
            source_provenance=_fit_lineage(tmp_path / "m1", clock="M1"),
            exit_m1_seq_len=60,
        )

    m5_before_start = _ohlcv(
        _PRE_START, "2026-01-04T23:55:00+00:00", freq="5min", seed=13
    )
    with pytest.raises(RuntimeError, match="V29_REGISTRY_FIT_WINDOW_EMPTY"):
        fit_v29_registry_constants_from_m5(
            m5_before_start,
            declared_train_window_start=_TRAIN_START,
            declared_train_window_end=_TRAIN_END,
            declared_inner_fit_window_end=_INNER_FIT_END,
            source_provenance_by_clock={
                clock: _fit_lineage(tmp_path / f"m5-{clock}", clock=clock)
                for clock in MULTI_TF_RESAMPLE_RULES
            },
            per_tf_seq_lens={"M5": 8, "M15": 4, "H1": 4, "H4": 3, "D1": 2},
            entry_m5_seq_len=6,
        )


@pytest.mark.parametrize(
    ("start", "expected"),
    [
        ("2026-01-05T00:00:00", "must be timezone-aware UTC"),
        (_TRAIN_END, "must precede declared_train_window_end"),
        ("2026-01-07T00:00:00+00:00", "INNER_WINDOW_INVALID"),
    ],
)
def test_registry_fit_owners_require_an_ordered_utc_train_window(
    tmp_path: Path, start: str, expected: str
) -> None:
    """The lower bound is validated exactly as the upper bound is."""

    frame = _ohlcv(_TRAIN_START, _TRAIN_END, freq="1min", seed=17)
    with pytest.raises(RuntimeError, match=expected):
        fit_v29_registry_m1_lane_params_from_m1(
            frame,
            declared_train_window_start=start,
            declared_train_window_end=_TRAIN_END,
            declared_inner_fit_window_end=_INNER_FIT_END,
            source_provenance=_fit_lineage(tmp_path / "m1", clock="M1"),
            exit_m1_seq_len=60,
        )
    m5_frame = _ohlcv(_TRAIN_START, _TRAIN_END, freq="5min", seed=19)
    with pytest.raises(RuntimeError, match=expected):
        fit_v29_registry_constants_from_m5(
            m5_frame,
            declared_train_window_start=start,
            declared_train_window_end=_TRAIN_END,
            declared_inner_fit_window_end=_INNER_FIT_END,
            source_provenance_by_clock={
                clock: _fit_lineage(tmp_path / f"m5-{clock}", clock=clock)
                for clock in MULTI_TF_RESAMPLE_RULES
            },
            per_tf_seq_lens={"M5": 8, "M15": 4, "H1": 4, "H4": 3, "D1": 2},
            entry_m5_seq_len=6,
        )


def test_m1_lane_payload_binds_exact_hyperfit_and_lifetimes() -> None:
    payload = synthetic_v29_registry_m1_lane_params()
    assert require_v29_registry_m1_lane_params(payload) == payload
    assert payload["schema_version"] == V29_REGISTRY_M1_LANE_PARAMS_SCHEMA_VERSION
    assert payload["level_recurrence_threshold_atr"] > 0.0
    assert payload["level_expiry_bars"] > 0
    assert payload["exit_m1"]["seq_len"] == EXIT_FEATURE_SEQUENCE_BARS
    assert payload["exit_m1"]["trendline_band_atr"] > 0.0
    assert payload["exit_m1"]["trendline_expiry_bars"] > 0
    assert not any(
        "quantile" in key or "reaction_window" in key or "retest_window" in key
        for key in payload
    )
    for nested in (
        payload["provenance"]["level_recurrence_threshold"],
        payload["provenance"]["trendline_band"],
    ):
        assert nested["future_outcomes_usage"] == (
            "TRAIN_hyperparameter_fit_only_not_apply_features"
        )
        assert nested["candidate_count_total_empirical"] > 0
        assert nested["candidate_count_scoreable"] > 0


def test_m1_lane_validator_rejects_old_keys_and_nested_mutation() -> None:
    valid = synthetic_v29_registry_m1_lane_params()
    old_named = copy.deepcopy(valid)
    old_named["level_tol_atr"] = old_named.pop(
        "level_recurrence_threshold_atr"
    )
    _rehash(old_named)
    with pytest.raises(RuntimeError, match="exact keys differ"):
        require_v29_registry_m1_lane_params(old_named)
    with pytest.raises(RuntimeError, match="exact keys differ"):
        require_v29_registry_m1_lane_params({**valid, "level_tol_quantile_q": 0.5})
    with pytest.raises(RuntimeError, match="exact keys differ"):
        require_v29_registry_m1_lane_params({**valid, "reaction_window_bars": 12})

    mutated = copy.deepcopy(valid)
    source = Path(
        mutated["provenance"]["level_recurrence_threshold"]
        ["source_provenance"]["source_artifact"]
    )
    source.write_text("mutated\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="registry hyperfit provenance"):
        require_v29_registry_m1_lane_params(mutated)


def test_m5_registry_validator_rejects_resealed_population_tamper() -> None:
    valid = synthetic_v29_registry_constants()
    tampered = copy.deepcopy(valid)
    tampered["provenance"]["trendline_band"]["H4"][
        "population_configuration"
    ]["identity_expiry_bars"] += 1
    nested = tampered["provenance"]["trendline_band"]["H4"]
    _rehash(nested)
    _rehash(tampered)
    with pytest.raises(RuntimeError, match="registry hyperfit binding"):
        require_v29_registry_constants(tampered)


def test_materializer_resolves_m1_lifetime_params_fail_closed(tmp_path: Path) -> None:
    lane = synthetic_v29_registry_m1_lane_params()
    manifest = write_synthetic_m1_registry_manifest(tmp_path, params=lane)
    manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
    source = Path(manifest_payload["output_parquet"])
    resolved = feature_producer._resolve_v29_registry_layer_params(
        manifest,
        timeframe="M1",
        expected_source=source,
        expected_source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        expected_source_manifest=manifest,
    )
    assert resolved == {
        "level_recurrence_threshold_atr": lane[
            "level_recurrence_threshold_atr"
        ],
        "level_expiry_bars": lane["level_expiry_bars"],
        "trendline_band_atr": lane["exit_m1"]["trendline_band_atr"],
        "trendline_expiry_bars": lane["exit_m1"]["trendline_expiry_bars"],
        "trendline_seq_len": lane["exit_m1"]["seq_len"],
    }


@pytest.mark.parametrize("root_flag", ["--native-m1-root", "--native-m5-root"])
def test_enriched_cli_requires_complete_registry_fit_lineage(
    monkeypatch: pytest.MonkeyPatch, root_flag: str
) -> None:
    called: list[dict[str, object]] = []
    monkeypatch.setattr(
        enriched_producer,
        "_build_enriched_frame",
        lambda **kwargs: called.append(kwargs) or {"decision": "PASS"},
    )
    base = [
        "producer", root_flag, "/tmp/native",
        "--pair-manifest", "/tmp/pair.json",
        "--expected-pair-manifest-sha256", "a" * 64,
        "--multi-tf-cache-dir", "/tmp/cache",
        "--output-parquet", "/tmp/enriched.parquet",
        "--manifest-path", "/tmp/enriched.json",
        "--checkpoint-dir", "/tmp/checkpoint",
        "--dataset-run-id", "run",
        "--pair-generation-id", "b" * 64,
        "--registry-fit-train-start", "2026-01-01T00:00:00+00:00",
        "--registry-fit-train-end", "2026-01-31T00:00:00+00:00",
        "--registry-fit-tape-manifest", "/tmp/tape.json",
        "--expected-registry-fit-tape-manifest-sha256", "c" * 64,
        "--volatility-squeeze-manifest", "/tmp/squeeze.json",
        "--expected-volatility-squeeze-manifest-sha256", "d" * 64,
    ]
    monkeypatch.setattr(sys, "argv", list(base))
    with pytest.raises(SystemExit):
        enriched_producer.main()
    assert called == []

    monkeypatch.setattr(
        sys,
        "argv",
        [*base, "--registry-fit-inner-end", "2026-01-15T00:00:00+00:00"],
    )
    enriched_producer.main()
    assert len(called) == 1
    assert called[0]["registry_fit_inner_end"] == "2026-01-15T00:00:00+00:00"
    # The declared TRAIN window is an ordered PAIR on every route: without the
    # lower bound the registry operators are fitted on the whole source
    # history and recorded as a TRAIN fit (repaired 2026-08-15).
    assert called[0]["registry_fit_train_start"] == "2026-01-01T00:00:00+00:00"
    assert called[0]["registry_fit_train_end"] == "2026-01-31T00:00:00+00:00"
    # The hash-bound pair pointer the registry fit freezes into its source
    # provenance comes from the chain, never from a producer self-measurement.
    assert called[0]["expected_pair_manifest_sha256"] == "a" * 64
    assert "level_tol_quantile_q" not in called[0]
