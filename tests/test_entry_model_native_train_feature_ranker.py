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
from gx1.contracts.xau_tape_provenance_v1 import (
    SEQ513_SOURCE_CASCADE_PAIR_PROOF_SCHEMA_VERSION,
)
from gx1.scripts import materialize_entry_model_native_seq513_signal_manifest_v1 as manifest_producer
from gx1.scripts import materialize_entry_model_native_train_feature_ranker_v1 as ranker
from tests.model_native_rank_reference_support import materialize_test_rank_reference


RUN_ID = "FEATURE_RANKER_UNIT_RUN_ID"
RANKING_CREATED = datetime(2026, 7, 18, 9, 0, 0, 1, tzinfo=timezone.utc)
MANIFEST_CREATED = datetime(2026, 7, 18, 9, 0, 1, 1, tzinfo=timezone.utc)


def _stamp(value: datetime) -> str:
    return value.strftime("%Y%m%dT%H%M%S%fZ")


def _rank_reference(tmp_path: Path):
    return materialize_test_rank_reference(
        tmp_path / "rank_reference",
        run_id=RUN_ID,
        history_start="2019-12-31T00:00:00Z",
        fit_start="2020-11-09T00:00:00Z",
        fit_end="2026-03-31T23:59:59Z",
    )[1]


def _source_cascade_metadata(
    tmp_path: Path,
    reference: object | None = None,
) -> dict[str, object]:
    event_root = tmp_path.resolve()
    canonical_path = (
        Path(str(reference.sidecar["source_parquet"])).resolve()
        if reference is not None
        else event_root / "canonical_v2.parquet"
    )
    source_path = event_root / "FULL_PLUS_CTX_v3src.parquet"
    if reference is not None and not source_path.exists():
        source_path.write_bytes(canonical_path.read_bytes())
    source_sha = (
        str(reference.sidecar["source_parquet_sha256"])
        if reference is not None
        else "1" * 64
    )
    return {
        "path": str((event_root / "SOURCE_CASCADE_PROOF.json").resolve()),
        "sha256": "9" * 64,
        "schema_version": SEQ513_SOURCE_CASCADE_PAIR_PROOF_SCHEMA_VERSION,
        "entry_run_id": RUN_ID,
        "event_root": str(event_root),
        "source_parquet_path": str(source_path),
        "source_parquet_sha256": source_sha,
        "canonical_v2_path": str(canonical_path),
        "canonical_v2_sha256": source_sha,
        "multi_tf_cache_dir": str(event_root / "MULTI_TF_V4_CACHE"),
        "multi_tf_manifest_sha256": "4" * 64,
        "multi_tf_cache_identity_sha256": "5" * 64,
        "pair_manifest_path": str(event_root / "PAIR_MANIFEST.json"),
        "pair_manifest_sha256": "6" * 64,
        "pair_generation_id": "7" * 64,
        "history_start_utc": "2019-12-31T00:00:00+00:00",
        "time_max_utc": "2026-07-24T20:55:00+00:00",
    }


def test_ranker_feature_attachment_is_single_worker() -> None:
    assert ranker.ATTACH_WORKERS == 1


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


def test_direction_utility_margin_target_is_train_capped_and_exact() -> None:
    rows = 60
    times = pd.date_range("2026-01-01", periods=rows, freq="5min", tz="UTC")
    mid = np.full(rows, 100.0)
    mid[ranker.DIRECTION_HORIZON_BARS :] = 200.0
    frame = pd.DataFrame(
        {"time": times, "bid_close": mid, "ask_close": mid}
    )
    train_start = times[10]
    train_end = times[-1]

    out_times, target = ranker._direction_utility_margin_target(
        frame, train_start=train_start, train_end=train_end
    )

    assert len(out_times) == rows
    assert np.isnan(target[-ranker.DIRECTION_HORIZON_BARS :]).all()
    assert np.isnan(target[:10]).all()  # before train_start
    # Row 10 doubles over H24 while its first-10 path is flat:
    # LONG utility=+10,000 bps, SHORT utility=-10,000 bps.
    assert target[10] == pytest.approx(20_000.0, rel=1e-6)


def test_direction_utility_margin_target_matches_spread_and_path_formula() -> None:
    rows = 40
    times = pd.date_range("2026-01-01", periods=rows, freq="5min", tz="UTC")
    bid = 100.0 + np.linspace(0.0, 1.2, rows)
    bid[1:11] += np.array([0.10, -0.08, 0.25, -0.15, 0.40, -0.20, 0.30, -0.05, 0.50, 0.15])
    ask = bid + 0.02
    frame = pd.DataFrame({"time": times, "bid_close": bid, "ask_close": ask})

    _, target = ranker._direction_utility_margin_target(
        frame,
        train_start=times[0],
        train_end=times[-1],
    )

    entry_bid = bid[0]
    entry_ask = ask[0]
    pnl_long = (bid[24] - entry_ask) / entry_ask * 1e4
    pnl_short = (entry_bid - ask[24]) / entry_bid * 1e4
    window_bid = bid[:11]
    window_ask = ask[:11]
    mfe_long = (window_bid.max() - entry_ask) / entry_ask * 1e4
    mae_long = (entry_ask - window_bid.min()) / entry_ask * 1e4
    mfe_short = (entry_bid - window_ask.min()) / entry_bid * 1e4
    mae_short = (window_ask.max() - entry_bid) / entry_bid * 1e4
    long_utility = (
        pnl_long
        + ranker.UTILITY_MFE_WEIGHT * mfe_long
        - ranker.UTILITY_MAE_WEIGHT * mae_long
        + ranker.UTILITY_PATH_WEIGHT * (mfe_long - mae_long)
    )
    short_utility = (
        pnl_short
        + ranker.UTILITY_MFE_WEIGHT * mfe_short
        - ranker.UTILITY_MAE_WEIGHT * mae_short
        + ranker.UTILITY_PATH_WEIGHT * (mfe_short - mae_short)
    )
    assert target[0] == pytest.approx(long_utility - short_utility, rel=1e-12)


def test_candidate_matrix_reads_ranked_common_history_close_and_atr(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from gx1.scripts import build_entry_v10_ctx_training_dataset_v3 as builder

    times = pd.date_range("2026-01-01", periods=4, freq="5min", tz="UTC")
    frame = pd.DataFrame(
        {
            "time": times,
            "close": [100.0, 101.0, 102.0, 103.0],
            "atr": [1.0, 1.1, 1.2, 1.3],
            "open": [99.5, 100.5, 101.5, 102.5],
            "high": [100.5, 101.5, 102.5, 103.5],
            "low": [99.0, 100.0, 101.0, 102.0],
        }
    )
    captured_path: Path | None = None
    # The price/candlestick layers read this parquet for causal EMA/derivative
    # history and then align to the sample rows, so it must carry the earlier
    # prefix as well as the ranked rows. One truth still holds: for every ranked
    # timestamp its close/atr must equal the ranked frame exactly.
    causal_source = tmp_path / "FULL_PLUS_CTX_v3src.parquet"
    warmup = pd.DataFrame(
        {
            "time": pd.date_range("2025-12-31", periods=3, freq="5min", tz="UTC"),
            "close": [97.0, 98.0, 99.0],
            "atr": [0.7, 0.8, 0.9],
            "open": [96.5, 97.5, 98.5],
            "high": [97.5, 98.5, 99.5],
            "low": [96.0, 97.0, 98.0],
        }
    )
    pd.concat([warmup, frame], ignore_index=True).to_parquet(causal_source, index=False)

    def fake_inline(
        observed_frame,
        *,
        requested_features,
        ctx_cont_names,
        ctx_cat_names,
        source_parquet,
        source_contract_label,
    ):
        nonlocal captured_path
        captured_path = Path(source_parquet)
        source = pd.read_parquet(captured_path)
        assert list(source.columns) == ["time", "close", "atr", "open", "high", "low"]
        assert len(source) > len(observed_frame)
        ranked = source.set_index("time").loc[observed_frame["time"]]
        np.testing.assert_array_equal(
            ranked["close"].to_numpy(), observed_frame["close"].to_numpy()
        )
        np.testing.assert_array_equal(
            ranked["atr"].to_numpy(), observed_frame["atr"].to_numpy()
        )
        assert source_contract_label == "train_feature_ranker_common_causal_history_v2"
        return (
            np.arange(len(observed_frame), dtype=np.float32).reshape(-1, 1),
            list(requested_features),
            {},
        )

    monkeypatch.setattr(builder, "_build_inline_seq_structure_extension", fake_inline)
    values, names = ranker._compute_candidate_matrix(
        frame,
        candidates=["trend.fixture_candidate"],
        source_ctx_cont=[],
        causal_source_parquet=causal_source,
    )

    assert names == ["trend.fixture_candidate"]
    assert values.shape == (4, 1)
    assert captured_path == causal_source


def test_emit_ranking_round_trips_through_the_real_manifest_producer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reference = _rank_reference(tmp_path)
    candidates = [
        f"session_regime.rank_candidate_{index:03d}"
        for index in range(MODEL_NATIVE_RANKED_REMAINDER_FEATURE_COUNT)
    ]
    scores = {
        name: float(MODEL_NATIVE_RANKED_REMAINDER_FEATURE_COUNT - index)
        for index, name in enumerate(candidates)
    }

    ranking_path = ranker.emit_ranking(
        out_path=tmp_path / f"{ranker.RANKING_EVENT_PREFIX}_{_stamp(RANKING_CREATED)}.json",
        run_id=RUN_ID,
        train_start=pd.Timestamp("2020-11-09T00:00:00Z"),
        train_end=pd.Timestamp("2026-03-31T23:59:59Z"),
        source_time_max=pd.Timestamp("2026-03-31T23:55:00Z"),
        target_time_max=pd.Timestamp("2026-03-31T21:55:00Z"),
        source_sha256=str(reference.sidecar["source_parquet_sha256"]),
        target_sha256="2" * 64,
        rank_reference=reference,
        source_cascade=_source_cascade_metadata(tmp_path, reference),
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
    source_cascade = _source_cascade_metadata(tmp_path, reference)
    monkeypatch.setattr(
        manifest_producer,
        "validate_seq513_source_cascade_proof",
        lambda *args, **kwargs: dict(source_cascade),
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
    mandatory_count = len(MODEL_NATIVE_MANDATORY_SELECTED_FIELDS)
    assert manifest["selected_features"][:mandatory_count] == list(
        MODEL_NATIVE_MANDATORY_SELECTED_FIELDS
    )
    assert manifest["selected_features"][mandatory_count:] == candidates
    assert manifest["feature_ranking"]["target_contract"] == (
        manifest_producer.TRAIN_FEATURE_RANKING_TARGET_CONTRACT
    )
    require_model_native_manifest(manifest, context="RANKER_ROUND_TRIP_TEST")


def test_emit_ranking_orders_by_score_then_name(tmp_path: Path) -> None:
    reference = _rank_reference(tmp_path)
    scores = {
        "session_regime.bbb": 0.5,
        "session_regime.aaa": 0.5,
        "session_regime.ccc": 0.9,
    }
    path = ranker.emit_ranking(
        out_path=tmp_path / f"{ranker.RANKING_EVENT_PREFIX}_{_stamp(RANKING_CREATED)}.json",
        run_id=RUN_ID,
        train_start=pd.Timestamp("2020-11-09T00:00:00Z"),
        train_end=pd.Timestamp("2026-03-31T23:59:59Z"),
        source_time_max=pd.Timestamp("2026-03-31T23:55:00Z"),
        target_time_max=pd.Timestamp("2026-03-31T21:55:00Z"),
        source_sha256=str(reference.sidecar["source_parquet_sha256"]),
        target_sha256="2" * 64,
        rank_reference=reference,
        source_cascade=_source_cascade_metadata(tmp_path, reference),
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


def test_ranker_checkpoint_key_binds_run_source_cache_and_window() -> None:
    base = {
        "run_id": RUN_ID,
        "source_sha256": "1" * 64,
        "mtf_cache_sha256": "2" * 64,
        "source_cascade_sha256": "9" * 64,
        "rank_reference_sha256": "5" * 64,
        "rank_reference_sidecar_sha256": "7" * 64,
        "history_start": pd.Timestamp("2021-01-05T00:00:00Z"),
        "train_start": pd.Timestamp("2021-03-16T00:00:00Z"),
        "train_end": pd.Timestamp("2026-03-31T23:59:59Z"),
    }
    expected = ranker._ranker_checkpoint_key(**base)
    assert len(expected) == 64
    for field, changed in (
        ("run_id", "FEATURE_RANKER_OTHER_RUN_ID"),
        ("source_sha256", "3" * 64),
        ("mtf_cache_sha256", "4" * 64),
        ("source_cascade_sha256", "0" * 64),
        ("rank_reference_sha256", "6" * 64),
        ("rank_reference_sidecar_sha256", "8" * 64),
        ("train_start", pd.Timestamp("2021-03-17T00:00:00Z")),
    ):
        variant = dict(base)
        variant[field] = changed
        assert ranker._ranker_checkpoint_key(**variant) != expected


def test_emit_ranking_rejects_filename_created_timestamp_mismatch(tmp_path: Path) -> None:
    reference = _rank_reference(tmp_path)
    out = tmp_path / f"{ranker.RANKING_EVENT_PREFIX}_20260718T090000000001Z.json"
    with pytest.raises(RuntimeError, match="OUTPUT_TIMESTAMP_MISMATCH"):
        ranker.emit_ranking(
            out_path=out,
            run_id=RUN_ID,
            train_start=pd.Timestamp("2020-11-09T00:00:00Z"),
            train_end=pd.Timestamp("2026-03-31T23:59:59Z"),
            source_time_max=pd.Timestamp("2026-03-31T23:55:00Z"),
            target_time_max=pd.Timestamp("2026-03-31T21:55:00Z"),
            source_sha256=str(reference.sidecar["source_parquet_sha256"]),
            target_sha256="2" * 64,
            rank_reference=reference,
            source_cascade=_source_cascade_metadata(tmp_path, reference),
            scores={"session_regime.test": 1.0},
            created=datetime(2026, 7, 18, 9, 0, 0, 2, tzinfo=timezone.utc),
        )
