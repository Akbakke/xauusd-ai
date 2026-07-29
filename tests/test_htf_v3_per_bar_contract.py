"""The V3 per-bar multi-timeframe contract.

The 513 signal fields are M5-only because their builders sit on 199 upstream
source fields, 194 of them derived, with dependencies between the families - so
producing them per timeframe means rebuilding the entire context pipeline. Two
owners have no such dependency: the candlestick family needs exactly
["open", "high", "low", "close", "time"] and swing structure is a pure function
of (high, low, close). V3 adds those to V2's 25 so the higher timeframes carry
real price geometry instead of a generic lens.
"""
from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd

from gx1.features import htf_features as htf


def _bars(n: int, *, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 2000.0 + np.cumsum(rng.normal(0.0, 1.5, size=n))
    high = close + np.abs(rng.normal(0.0, 1.0, size=n))
    low = close - np.abs(rng.normal(0.0, 1.0, size=n))
    open_ = np.concatenate([[close[0]], close[:-1]])
    index = pd.date_range("2021-01-04", periods=n, freq="5min", tz="UTC")
    return pd.DataFrame(
        {
            "open": open_,
            "high": np.maximum.reduce([high, open_, close]),
            "low": np.minimum.reduce([low, open_, close]),
            "close": close,
            "volume": np.abs(rng.normal(500.0, 50.0, size=n)),
        },
        index=index,
    )


def test_v3_extends_v2_without_redefining_it() -> None:
    assert htf.MULTI_TF_FEATURE_COUNT_V2 == 25
    assert htf.MULTI_TF_FEATURE_COUNT_V3 == 90
    # V2 is a bit-identical prefix, so every artifact built on V2 stays valid.
    assert htf.MULTI_TF_PER_BAR_FEATURES_V3[:25] == htf.MULTI_TF_PER_BAR_FEATURES_V2
    assert len(set(htf.MULTI_TF_PER_BAR_FEATURES_V3)) == 90
    assert htf.MULTI_TF_FEATURE_NAMES_SHA256_V3 != htf.MULTI_TF_FEATURE_NAMES_SHA256_V2
    assert htf.HTF_V3_MATRIX_CONTRACT != htf.HTF_V2_MATRIX_CONTRACT


def test_v3_candlestick_names_come_from_the_owner() -> None:
    """A rename in the candlestick owner must not silently desync this contract."""
    from gx1.features.entry_candlestick_patterns_v1 import (
        CANDLESTICK_PATTERN_FEATURE_NAMES,
    )

    assert len(htf.MULTI_TF_PER_BAR_CANDLESTICK_V3) == len(
        CANDLESTICK_PATTERN_FEATURE_NAMES
    )
    for declared, owned in zip(
        htf.MULTI_TF_PER_BAR_CANDLESTICK_V3,
        CANDLESTICK_PATTERN_FEATURE_NAMES,
        strict=True,
    ):
        assert declared.endswith(owned.split(".", 1)[1])


def test_v3_holds_its_contract_at_every_resolution() -> None:
    """Exact width, exact order, and a single causal warmup prefix per timeframe."""
    m5 = _bars(20000, seed=7)
    for timeframe, rule in htf.MULTI_TF_RESAMPLE_RULES.items():
        resampled = (
            m5.resample(rule)
            .agg(
                {
                    "open": "first",
                    "high": "max",
                    "low": "min",
                    "close": "last",
                    "volume": "sum",
                }
            )
            .dropna()
        )
        if len(resampled) < 300:
            continue
        built = htf.compute_per_bar_features_v3(resampled)
        assert tuple(built.columns) == htf.MULTI_TF_PER_BAR_FEATURES_V3, timeframe
        matrix = built.to_numpy(dtype=np.float32)
        warmup = htf.validate_causal_feature_matrix(
            matrix,
            expected_width=htf.MULTI_TF_FEATURE_COUNT_V3,
            context=f"HTF_V3_{timeframe}",
        )
        assert np.isfinite(matrix[warmup:]).all(), timeframe


def test_v3_first_25_columns_equal_v2_exactly() -> None:
    """Adding families may not perturb a single existing value."""
    bars = _bars(6000, seed=11)
    v2 = htf.compute_per_bar_features_v2(bars).to_numpy(dtype=np.float64)
    v3 = htf.compute_per_bar_features_v3(bars).to_numpy(dtype=np.float64)

    assert v3.shape[1] == 90
    both_finite = np.isfinite(v2) & np.isfinite(v3[:, :25])
    assert np.array_equal(v2[both_finite], v3[:, :25][both_finite])
    assert np.array_equal(np.isfinite(v2), np.isfinite(v3[:, :25]))


def test_dataset_rejects_undeclared_and_split_brain_contracts() -> None:
    """The Dataset reads the cache's declaration; it may not assume one.

    Pinning the width to MULTI_TF_FEATURE_COUNT_V2 is what held the higher
    timeframes at 25 generic features. Now the tables say what they are: an
    undeclared contract, an unknown one, or five timeframes that disagree all
    fail closed rather than silently taking V2.
    """
    from gx1.models.entry_v10 import entry_v10_ctx_train_v3 as trainer

    source = (
        pathlib.Path(trainer.__file__).read_text(encoding="utf-8")
    )
    for marker in (
        "MULTI_TF_CONTRACT_SPLIT_BRAIN",
        "MULTI_TF_CONTRACT_UNKNOWN",
        "MULTI_TF_CONTRACT_WIDTH_MISMATCH",
        "MULTI_TF_CONTRACT_ORDER_MISMATCH",
    ):
        assert marker in source, marker
    # and the width is no longer pinned to a constant
    assert "self._multi_tf_feature_count = int(MULTI_TF_FEATURE_COUNT_V2)" not in source


def test_bundle_and_normalization_read_the_declared_contract() -> None:
    """A lineage that records V2's names beside a V3 cache is train != serve."""
    import pathlib as _p

    repo = _p.Path(__file__).resolve().parents[1]

    normalization = (
        repo / "gx1/models/entry_v10/entry_v10_input_normalization.py"
    ).read_text()
    assert "def resolve_mtf_per_bar_contract(" in normalization
    assert "ENTRY_INPUT_NORMALIZATION_MTF_CONTRACT_UNKNOWN" in normalization
    # the lineage hash comes from the manifest, not from an imported constant
    assert '_mtf_manifest_declared["feature_names"]' in normalization

    bundle = (repo / "gx1/models/entry_v10/entry_v10_bundle.py").read_text()
    assert "ENTRY_BUNDLE_MODEL_NATIVE_MTF_CONTRACT_UNKNOWN" in bundle
    assert "_bundle_mtf_contracts" in bundle


def test_trainer_sweeps_orphaned_memmap_scratch() -> None:
    """Killed runs leaked 69 GB each; four were found holding 295 GB.

    TemporaryDirectory cleans up through a finalizer that a killed process never
    runs. The scratch name carries the creating PID, so an orphan is provable
    rather than guessed, and a live PID is left alone so a concurrent run is
    safe.
    """
    import os as _os
    import pathlib as _pathlib
    import tempfile

    from gx1.models.entry_v10 import entry_v10_ctx_train_v3 as trainer

    with tempfile.TemporaryDirectory() as raw_root:
        root = _pathlib.Path(raw_root)
        dead = root / "v10_seq513_dataset__HOLD_03B_train_999999999_abc"
        dead.mkdir()
        (dead / "seq.float32.mmap").write_bytes(b"x" * 32)
        alive = root / f"v10_seq513_dataset__HOLD_03B_train_{_os.getpid()}_xyz"
        alive.mkdir()
        (alive / "seq.float32.mmap").write_bytes(b"x" * 32)

        trainer._sweep_orphaned_memmap_scratch(root)

        assert not dead.exists(), "orphaned scratch survived the sweep"
        assert alive.exists(), "the sweep removed a live run's scratch"
