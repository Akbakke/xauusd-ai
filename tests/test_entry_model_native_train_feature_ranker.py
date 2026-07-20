from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gx1.contracts.entry_model_native_signal_v1 import (
    FORBIDDEN_LEGACY_BRIDGE_FIELDS,
    MODEL_NATIVE_BASE_FIELDS,
    MODEL_NATIVE_MANDATORY_SELECTED_FIELDS,
    MODEL_NATIVE_RANKED_REMAINDER_FEATURE_COUNT,
    require_model_native_manifest,
)
from gx1.contracts.entry_model_native_signal_v1 import MODEL_NATIVE_CTX_CONT_FIELDS
from gx1.scripts import materialize_entry_model_native_seq513_signal_manifest_v1 as manifest_producer
from gx1.scripts import materialize_entry_model_native_train_feature_ranker_v1 as ranker


RUN_ID = "FEATURE_RANKER_UNIT_RUN_ID"
RANKING_CREATED = datetime(2026, 7, 18, 9, 0, 0, 1, tzinfo=timezone.utc)
MANIFEST_CREATED = datetime(2026, 7, 18, 9, 0, 1, 1, tzinfo=timezone.utc)


def _stamp(value: datetime) -> str:
    return value.strftime("%Y%m%dT%H%M%S%fZ")


def test_candidate_universe_is_clean_and_large_enough() -> None:
    universe = ranker._candidate_universe(list(MODEL_NATIVE_CTX_CONT_FIELDS))

    assert len(universe) >= MODEL_NATIVE_RANKED_REMAINDER_FEATURE_COUNT
    assert universe == sorted(universe)
    mandatory = set(MODEL_NATIVE_MANDATORY_SELECTED_FIELDS)
    base = set(MODEL_NATIVE_BASE_FIELDS)
    forbidden = set(FORBIDDEN_LEGACY_BRIDGE_FIELDS)
    for name in universe:
        assert name not in mandatory
        assert name not in base
        assert name not in forbidden
        assert not manifest_producer._is_forbidden_leak_name(name)


def test_spearman_scores_are_deterministic_with_support_floor() -> None:
    rows = 2000
    rng_free = np.linspace(-1.0, 1.0, rows)
    target = rng_free * 100.0
    noise = np.sin(np.arange(rows) * 12.9898)  # deterministic pseudo-noise
    sparse = np.full(rows, np.nan)
    sparse[: rows // 20] = rng_free[: rows // 20]

    names = [
        "session_regime.a_perfect",
        "session_regime.b_anti",
        "session_regime.c_noise",
        "session_regime.d_sparse",
    ]
    matrix = np.column_stack([rng_free, -rng_free, noise, sparse]).astype(np.float32)

    scores = ranker._spearman_scores(matrix, names, target)

    assert scores["session_regime.a_perfect"] == pytest.approx(1.0)
    assert scores["session_regime.b_anti"] == pytest.approx(1.0)
    assert scores["session_regime.c_noise"] < 0.2
    assert scores["session_regime.d_sparse"] == 0.0
    # Determinism: identical inputs give identical rounded scores.
    again = ranker._spearman_scores(matrix, names, target)
    assert again == scores


def test_forward_return_target_is_train_capped_and_correct() -> None:
    rows = 60
    times = pd.date_range("2026-01-01", periods=rows, freq="5min", tz="UTC")
    mid = np.full(rows, 100.0)
    mid[ranker.TARGET_HORIZON_BARS :] = 200.0  # first bars double over horizon
    frame = pd.DataFrame(
        {"time": times, "bid_close": mid - 0.01, "ask_close": mid + 0.01}
    )
    train_start = times[10]
    train_end = times[-1]

    out_times, target = ranker._forward_return_target(
        frame, train_start=train_start, train_end=train_end
    )

    assert len(out_times) == rows
    assert np.isnan(target[-ranker.TARGET_HORIZON_BARS :]).all()
    assert np.isnan(target[:10]).all()  # before train_start
    # Row 10 doubles over the horizon: log(2) * 1e4 bps.
    assert target[10] == pytest.approx(np.log(2.0) * 1e4, rel=1e-6)


def test_emit_ranking_round_trips_through_the_real_manifest_producer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidates = [
        f"session_regime.rank_candidate_{index:03d}"
        for index in range(MODEL_NATIVE_RANKED_REMAINDER_FEATURE_COUNT)
    ]
    scores = {
        name: float(MODEL_NATIVE_RANKED_REMAINDER_FEATURE_COUNT - index)
        for index, name in enumerate(candidates)
    }

    ranking_path = ranker.emit_ranking(
        out_dir=tmp_path,
        run_id=RUN_ID,
        train_start=pd.Timestamp("2020-11-09T00:00:00Z"),
        train_end=pd.Timestamp("2026-03-31T23:59:59Z"),
        source_time_max=pd.Timestamp("2026-03-31T23:55:00Z"),
        target_time_max=pd.Timestamp("2026-03-31T21:55:00Z"),
        source_sha256="1" * 64,
        target_sha256="2" * 64,
        scores=scores,
        created=RANKING_CREATED,
    )

    assert ranking_path.name == (
        f"{ranker.RANKING_EVENT_PREFIX}_{_stamp(RANKING_CREATED)}.json"
    )

    monkeypatch.setattr(
        manifest_producer,
        "_utc_now",
        lambda: datetime(2026, 7, 18, 9, 0, 1, 500_000, tzinfo=timezone.utc),
    )
    out = tmp_path / (
        f"{manifest_producer.SIGNAL_MANIFEST_EVENT_PREFIX}_{_stamp(MANIFEST_CREATED)}.json"
    )
    manifest = manifest_producer.run(
        argparse.Namespace(
            feature_ranking_json=str(ranking_path),
            out=str(out),
            run_id=RUN_ID,
        )
    )

    assert out.is_file()
    assert manifest["selected_features"][:305] == list(
        MODEL_NATIVE_MANDATORY_SELECTED_FIELDS
    )
    assert manifest["selected_features"][305:] == candidates
    require_model_native_manifest(manifest, context="RANKER_ROUND_TRIP_TEST")


def test_emit_ranking_orders_by_score_then_name(tmp_path: Path) -> None:
    scores = {
        "session_regime.bbb": 0.5,
        "session_regime.aaa": 0.5,
        "session_regime.ccc": 0.9,
    }
    path = ranker.emit_ranking(
        out_dir=tmp_path,
        run_id=RUN_ID,
        train_start=pd.Timestamp("2020-11-09T00:00:00Z"),
        train_end=pd.Timestamp("2026-03-31T23:59:59Z"),
        source_time_max=pd.Timestamp("2026-03-31T23:55:00Z"),
        target_time_max=pd.Timestamp("2026-03-31T21:55:00Z"),
        source_sha256="1" * 64,
        target_sha256="2" * 64,
        scores=scores,
        created=RANKING_CREATED,
    )
    import json

    rows = json.loads(path.read_text())["ranked_features"]
    assert [row["name"] for row in rows] == [
        "session_regime.ccc",
        "session_regime.aaa",
        "session_regime.bbb",
    ]
    assert [row["rank"] for row in rows] == [1, 2, 3]
