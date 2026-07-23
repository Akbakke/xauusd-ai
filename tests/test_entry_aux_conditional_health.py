from __future__ import annotations

import numpy as np
import pytest

from gx1.models.entry_v10.entry_v10_ctx_train_v3 import (
    _aux_head_diagnostics,
)


def _health_inputs() -> tuple[dict, dict, dict, dict]:
    rows = 180
    index = np.arange(rows, dtype=np.float64)
    long = index < 60
    short = (index >= 60) & (index < 120)
    trade = long | short
    bad = np.zeros(rows, dtype=np.float64)
    clean = np.zeros(rows, dtype=np.float64)
    survival = np.zeros(rows, dtype=np.float64)
    bad[trade] = (index[trade] % 2 == 0).astype(np.float64)
    clean[trade] = ((index[trade] // 2) % 2 == 0).astype(np.float64)
    survival[trade] = ((index[trade] // 3) % 2 == 0).astype(np.float64)
    jitter = ((index * 37) % 101) / 1000.0
    path_realized = (
        20.0
        - 14.0 * bad
        + 5.0 * clean
        + 2.0 * np.sin(index / 7.0)
        + jitter
    )
    mfe_realized = (
        15.0
        + 11.0 * survival
        - 3.0 * bad
        + 2.0 * np.cos(index / 9.0)
        + jitter
    )
    predictions = {
        "tradable": [0.15 + 0.7 * trade.astype(np.float64) + jitter],
        "bad_path": [0.1 + 0.8 * bad + jitter],
        "clean_edge": [0.1 + 0.8 * clean + jitter],
        "survival": [0.1 + 0.8 * survival + jitter],
        "path_quality": [path_realized / 20.0 + jitter],
        "mfe_first_n": [mfe_realized / 20.0 + jitter],
    }
    labels = {
        "tradable": [trade.astype(np.float64)],
        "bad_path": [bad],
        "clean_edge": [clean],
        "survival": [survival],
    }
    realized = {
        "mfe_first_n_bps": [mfe_realized],
        "path_quality_bps": [path_realized],
    }
    masks = {
        "trade": [trade.astype(np.float64)],
        "long": [long.astype(np.float64)],
        "short": [short.astype(np.float64)],
    }
    return predictions, labels, realized, masks


def test_aux_health_requires_trade_and_each_side_path_discrimination() -> None:
    predictions, labels, realized, masks = _health_inputs()

    metrics, failures = _aux_head_diagnostics(
        predictions,
        labels,
        realized,
        masks,
    )

    assert failures == []
    assert metrics["conditional_rows__trade"] == 120
    assert metrics["conditional_rows__long"] == 60
    assert metrics["conditional_rows__short"] == 60
    for head in ("bad_path", "clean_edge", "survival"):
        for scope in ("trade", "long", "short"):
            assert metrics[f"auc__{head}__{scope}"] > 0.99
            assert metrics[f"incremental_auc_lift__{head}__{scope}"] >= 0.01


def test_flat_recognition_cannot_pass_as_bad_path_edge() -> None:
    predictions, labels, realized, masks = _health_inputs()
    trade = masks["trade"][0].astype(bool)
    index = np.arange(trade.size, dtype=np.float64)
    predictions["bad_path"] = [
        0.1 + 0.8 * trade.astype(np.float64) + ((index * 17) % 53) / 10000.0
    ]

    metrics, failures = _aux_head_diagnostics(
        predictions,
        labels,
        realized,
        masks,
    )

    assert metrics["auc__bad_path__trade"] < 0.52
    assert any(
        "bad_path conditional AUC" in failure and "scope=trade" in failure
        for failure in failures
    )


def test_aux_health_rejects_non_partitioned_side_masks() -> None:
    predictions, labels, realized, masks = _health_inputs()
    masks["short"] = [masks["trade"][0].copy()]

    with pytest.raises(RuntimeError, match="SIDE_MASK_PARTITION_INVALID"):
        _aux_head_diagnostics(predictions, labels, realized, masks)
