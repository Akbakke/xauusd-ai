from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gx1.contracts.entry_model_native_signal_v1 import (
    MODEL_NATIVE_AVAILABLE_CANDIDATE_FIELDS,
    FORBIDDEN_LEGACY_BRIDGE_FIELDS,
    MODEL_NATIVE_BASE_FIELDS,
    MODEL_NATIVE_MANDATORY_SELECTED_FIELDS,
    require_model_native_manifest,
)
from gx1.contracts.entry_model_native_signal_v1 import MODEL_NATIVE_CTX_CONT_FIELDS
from gx1.contracts.xau_tape_provenance_v1 import (
    SEQ513_SOURCE_CASCADE_PAIR_PROOF_SCHEMA_VERSION,
)
from gx1.contracts.entry_causal_m1_outcomes_v1 import (
    build_entry_m1_fill_surface,
    causal_m1_terminal_outcomes_at_horizon,
)
from gx1.contracts.entry_causal_m1_target_policy_v1 import (
    causal_m1_direction_targets_from_policy,
)
from gx1.scripts import materialize_entry_model_native_seq513_signal_manifest_v1 as manifest_producer
from gx1.scripts import materialize_entry_model_native_train_feature_ranker_v1 as ranker
from tests.entry_direction_target_policy_support import (
    causal_m1_target_policy_fixture,
)


RUN_ID = "FEATURE_RANKER_UNIT_RUN_ID"
RANKING_CREATED = datetime(2026, 7, 18, 9, 0, 0, 1, tzinfo=timezone.utc)
MANIFEST_CREATED = datetime(2026, 7, 18, 9, 0, 1, 1, tzinfo=timezone.utc)
# The retired TRAIN rank reference used to supply this digest. The ranker no
# longer reads a rank artifact, so the cascade fixture owns it directly and the
# same value binds the target policy below.
SOURCE_PARQUET_SHA256 = "1" * 64


def _stamp(value: datetime) -> str:
    return value.strftime("%Y%m%dT%H%M%S%fZ")


def _source_cascade_metadata(tmp_path: Path) -> dict[str, object]:
    event_root = tmp_path.resolve()
    canonical_path = event_root / "canonical_v2.parquet"
    source_path = event_root / "FULL_PLUS_CTX_v3src.parquet"
    source_sha = SOURCE_PARQUET_SHA256
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
        "multi_tf_source_path": str(tmp_path / "m5_enriched.parquet"),
        "multi_tf_source_sha256": "6" * 64,
        "multi_tf_manifest_sha256": "4" * 64,
        "multi_tf_cache_identity_sha256": "5" * 64,
        "pair_manifest_path": str(event_root / "PAIR_MANIFEST.json"),
        "pair_manifest_sha256": "6" * 64,
        "pair_generation_id": "7" * 64,
        "history_start_utc": "2019-12-31T00:00:00+00:00",
        "time_max_utc": "2026-07-24T20:55:00+00:00",
    }


def _target_policy(
    *,
    source_sha256: str,
    train_start: str = "2020-11-09T00:00:00+00:00",
    train_end: str = "2026-03-31T23:59:59+00:00",
) -> dict[str, object]:
    return causal_m1_target_policy_fixture(
        source_parquet_sha256=source_sha256,
        tape_provenance_sha256="b" * 64,
        train_start_utc=train_start,
        train_end_utc=train_end,
    )


def _closed_m1_for_m5(times: pd.DatetimeIndex, *, bid: np.ndarray) -> pd.DataFrame:
    """Exact M1 fill/exit source that leaves the final fitted horizons absent."""

    minute_times = pd.date_range(
        times[0], periods=len(times) * 5 + 1, freq="min", tz="UTC"
    )
    minute_bid = np.interp(
        np.arange(len(minute_times), dtype=np.float64),
        np.linspace(0.0, len(minute_times) - 1, len(bid)),
        bid.astype(np.float64),
    )
    minute_ask = minute_bid + 0.02
    return pd.DataFrame(
        {
            "time": minute_times,
            "bid_open": minute_bid,
            "bid_high": minute_bid + 0.01,
            "bid_low": minute_bid - 0.01,
            "ask_open": minute_ask,
            "ask_high": minute_ask + 0.01,
            "ask_low": minute_ask - 0.01,
        }
    )


def _availability_fixture() -> tuple[list[str], dict[str, object], dict[str, float]]:
    names = list(MODEL_NATIVE_AVAILABLE_CANDIDATE_FIELDS)
    times = pd.date_range(
        end="2026-03-31T23:55:00Z",
        periods=256,
        freq="5min",
    )
    x = np.arange(len(times), dtype=np.float32)
    matrix = np.column_stack(
        [
            x * np.float32(index + 1) + np.float32(index * index + 0.125)
            for index in range(len(names))
        ]
    ).astype(np.float32)
    target = np.square(x.astype(np.float64))
    availability = ranker.fit_feature_availability_contract(
        matrix=matrix,
        names=names,
        times=times,
        train_start=pd.Timestamp("2020-11-09T00:00:00Z"),
        train_end=pd.Timestamp("2026-03-31T23:59:59Z"),
        diagnostic_target=target,
    )
    scores = {
        name: float(len(names) - index) / float(len(names))
        for index, name in enumerate(names)
    }
    return names, availability, scores


def test_ranker_feature_attachment_is_single_worker() -> None:
    assert ranker.ATTACH_WORKERS == 1


def test_main_rejects_invalid_output_identity_before_reading_sources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "ranker",
            "--run-id", RUN_ID,
            "--source-parquet", str(tmp_path / "missing_source.parquet"),
            "--canonical-v2-parquet", str(tmp_path / "missing_canonical.parquet"),
            "--mtf-cache-dir", str(tmp_path / "missing_cache"),
            "--source-cascade-proof", str(tmp_path / "missing_proof.json"),
            "--tape-root", str(tmp_path / "missing_tape"),
            "--m1-lifecycle-source", str(tmp_path / "missing_m1.parquet"),
            "--expected-source-time-max", "2026-06-30T23:55:00+00:00",
            "--history-start", "2021-06-01T00:00:00+00:00",
            "--train-start", "2021-06-01T00:00:00+00:00",
            "--train-end", "2025-05-31T23:55:00+00:00",
            "--out", str(tmp_path / "not_a_ranking_event.json"),
        ],
    )

    with pytest.raises(RuntimeError, match="FEATURE_RANKER_OUTPUT_IDENTITY_INVALID"):
        ranker.main()


def test_ranker_common_history_keeps_accepted_pre_2020_source_prefix(
    tmp_path: Path,
) -> None:
    source = tmp_path / "FULL_PLUS_CTX_v3src.parquet"
    pd.DataFrame(
        {
            "time": pd.to_datetime(
                [
                    "2019-11-08T21:55:00Z",
                    "2020-01-01T00:00:00Z",
                    "2020-01-01T00:05:00Z",
                ],
                utc=True,
            ),
            "open": [1.0, 2.0, 3.0],
            "high": [1.1, 2.1, 3.1],
            "low": [0.9, 1.9, 2.9],
            "close": [1.05, 2.05, 3.05],
        }
    ).to_parquet(source, index=False)

    observed = ranker._load_ranker_common_history_m5(
        train_end=pd.Timestamp("2020-01-01T00:00:00Z"),
        source_parquet=source,
        source_cascade={"schema_version": ranker.SEQ513_SOURCE_CASCADE_PAIR_PROOF_SCHEMA_VERSION},
    )

    assert list(observed.index) == list(
        pd.to_datetime(
            ["2019-11-08T21:55:00Z", "2020-01-01T00:00:00Z"], utc=True
        )
    )


def test_candidate_universe_is_clean_and_large_enough() -> None:
    universe = ranker._candidate_universe(list(MODEL_NATIVE_CTX_CONT_FIELDS))

    assert universe == list(MODEL_NATIVE_AVAILABLE_CANDIDATE_FIELDS)
    mandatory = set(MODEL_NATIVE_MANDATORY_SELECTED_FIELDS)
    base = set(MODEL_NATIVE_BASE_FIELDS)
    forbidden = set(FORBIDDEN_LEGACY_BRIDGE_FIELDS)
    for name in universe:
        assert name not in mandatory
        assert name not in base
        assert name not in forbidden
        assert not manifest_producer._is_forbidden_leak_name(name)


def test_candidate_availability_has_no_fixed_top_k_or_score_authority() -> None:
    ranker_source = Path(ranker.__file__).read_text(encoding="utf-8")
    manifest_source = Path(manifest_producer.__file__).read_text(encoding="utf-8")
    availability_source = Path(
        ranker.fit_feature_availability_contract.__code__.co_filename
    ).read_text(encoding="utf-8")

    for source in (ranker_source, manifest_source, availability_source):
        assert (
            "MODEL_NATIVE_" + "RANKED_REMAINDER_FEATURE_COUNT"
        ) not in source
        assert "MIN_SUPPORT_FRACTION" not in source
        assert "SCORE_DECIMALS" not in source
        assert "[:MODEL_NATIVE_RANKED" not in source
    assert '"target_scores_affect_selection": False' in availability_source
    assert "tuple(available_names) != MODEL_NATIVE_AVAILABLE_CANDIDATE_FIELDS" in (
        manifest_source
    )


def test_spearman_scores_are_diagnostic_without_support_cutoff() -> None:
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
    assert scores["session_regime.d_sparse"] > 0.0
    # Determinism: identical inputs give identical diagnostics.
    again = ranker._spearman_scores(matrix, names, target)
    assert again == scores


def test_direction_executable_pnl_margin_target_is_train_capped_and_exact() -> None:
    policy = causal_m1_target_policy_fixture()
    horizon = int(policy["selected_direction_horizon_bars"])
    rows = horizon + 40
    times = pd.date_range("2026-01-01", periods=rows, freq="5min", tz="UTC")
    mid = np.full(rows, 100.0)
    mid[horizon:] = 200.0
    frame = pd.DataFrame(
        {"time": times, "bid_close": mid, "ask_close": mid}
    )
    train_start = times[10]
    train_end = times[-1]

    closed_m1 = _closed_m1_for_m5(times, bid=np.full(rows, 100.0))
    row10_exit = (10 + 1 + horizon) * 5
    closed_m1.loc[row10_exit:, "bid_open"] = 200.0
    closed_m1.loc[row10_exit:, "bid_high"] = 200.01
    closed_m1.loc[row10_exit:, "bid_low"] = 199.99
    closed_m1.loc[row10_exit:, "ask_open"] = 200.02
    closed_m1.loc[row10_exit:, "ask_high"] = 200.03
    closed_m1.loc[row10_exit:, "ask_low"] = 200.01
    out_times, target = ranker._direction_executable_pnl_margin_target(
        frame,
        train_start=train_start,
        train_end=train_end,
        entry_direction_target_policy=policy,
        closed_m1=closed_m1,
    )

    assert len(out_times) == rows
    assert np.isnan(target[-horizon:]).all()
    assert np.isnan(target[:10]).all()  # before train_start
    # Row 10 doubles over the TRAIN-fitted horizon. The ranking target is the
    # exact executable long-minus-short PnL margin used by dataset labels.
    assert target[10] > 10_000.0


def test_direction_executable_pnl_margin_target_matches_policy_formula() -> None:
    policy = causal_m1_target_policy_fixture()
    horizon = int(policy["selected_direction_horizon_bars"])
    rows = horizon + 40
    times = pd.date_range("2026-01-01", periods=rows, freq="5min", tz="UTC")
    bid = 100.0 + np.linspace(0.0, 1.2, rows)
    bid[1:11] += np.array([0.10, -0.08, 0.25, -0.15, 0.40, -0.20, 0.30, -0.05, 0.50, 0.15])
    ask = bid + 0.02
    frame = pd.DataFrame({"time": times, "bid_close": bid, "ask_close": ask})
    closed_m1 = _closed_m1_for_m5(times, bid=bid)

    _, target = ranker._direction_executable_pnl_margin_target(
        frame,
        train_start=times[0],
        train_end=times[-1],
        entry_direction_target_policy=policy,
        closed_m1=closed_m1,
    )

    surface = build_entry_m1_fill_surface(
        m5_decision_times=frame["time"], closed_m1=closed_m1
    )
    terminal = causal_m1_terminal_outcomes_at_horizon(
        fill_surface=surface, closed_m1=closed_m1, horizon_m5_bars=horizon
    )
    expected = causal_m1_direction_targets_from_policy(
        policy=policy,
        long_executable_pnl_bps=terminal.loc[[0], "long_executable_pnl_bps"].to_numpy(),
        short_executable_pnl_bps=terminal.loc[[0], "short_executable_pnl_bps"].to_numpy(),
    )
    assert target[0] == pytest.approx(expected["side_margin_bps"][0], rel=1e-12)


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
            "volume": [10.0, 11.0, 12.0, 13.0],
            # The inline seq-structure extension reads these by name off the
            # temporary causal parquet, so the fixture must carry them exactly
            # as the real FULL_PLUS_CTX source does.
            "smc_bos_up": [0.0, 1.0, 0.0, 0.0],
            "smc_bos_down": [0.0, 0.0, 0.0, 1.0],
            "smc_choch": [0.0, 0.0, 1.0, 0.0],
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
            "volume": [7.0, 8.0, 9.0],
            "smc_bos_up": [0.0, 0.0, 1.0],
            "smc_bos_down": [1.0, 0.0, 0.0],
            "smc_choch": [0.0, 1.0, 0.0],
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
            local_timeframe,
            source_contract_label,
    ):
        nonlocal captured_path
        captured_path = Path(source_parquet)
        source = pd.read_parquet(captured_path)
        from gx1.features.entry_foundation_structure_v1 import (
            FOUNDATION_STRUCTURE_SOURCE_FIELDS,
        )

        assert list(source.columns) == [
            "time",
            "close",
            "atr",
            "open",
            "high",
            "low",
            "volume",
            *(
                name.removeprefix("snap.")
                for name in FOUNDATION_STRUCTURE_SOURCE_FIELDS
            ),
        ]
        assert len(source) > len(observed_frame)
        ranked = source.set_index("time").loc[observed_frame["time"]]
        np.testing.assert_array_equal(
            ranked["close"].to_numpy(), observed_frame["close"].to_numpy()
        )
        np.testing.assert_array_equal(
            ranked["atr"].to_numpy(), observed_frame["atr"].to_numpy()
        )
        assert source_contract_label == "train_feature_ranker_common_causal_history_v2"
        assert local_timeframe == "M5"
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
    # The layers receive a bounded seven-column materialization of the causal
    # source, published under a temporary path and removed afterwards.
    assert captured_path is not None
    assert captured_path != causal_source
    assert not captured_path.exists()


def test_emit_ranking_round_trips_through_the_real_manifest_producer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidates, availability, scores = _availability_fixture()

    ranking_path = ranker.emit_ranking(
        out_path=tmp_path / f"{ranker.RANKING_EVENT_PREFIX}_{_stamp(RANKING_CREATED)}.json",
        run_id=RUN_ID,
        train_start=pd.Timestamp("2020-11-09T00:00:00Z"),
        train_end=pd.Timestamp("2026-03-31T23:59:59Z"),
        source_time_max=pd.Timestamp("2026-03-31T23:55:00Z"),
        target_time_max=pd.Timestamp("2026-03-31T21:55:00Z"),
        source_sha256=SOURCE_PARQUET_SHA256,
        target_sha256="2" * 64,
        source_cascade=_source_cascade_metadata(tmp_path),
        entry_direction_target_policy=_target_policy(
            source_sha256=SOURCE_PARQUET_SHA256
        ),
        scores=scores,
        feature_availability=availability,
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
    source_cascade = _source_cascade_metadata(tmp_path)
    monkeypatch.setattr(
        manifest_producer,
        "validate_current_pair_source_cascade_proof",
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
    scores = {
        name: 0.1 for name in MODEL_NATIVE_AVAILABLE_CANDIDATE_FIELDS
    }
    availability_names, availability, _ = _availability_fixture()
    first, second, third = availability_names[:3]
    scores[first] = 0.5
    scores[second] = 0.5
    scores[third] = 0.9
    path = ranker.emit_ranking(
        out_path=tmp_path / f"{ranker.RANKING_EVENT_PREFIX}_{_stamp(RANKING_CREATED)}.json",
        run_id=RUN_ID,
        train_start=pd.Timestamp("2020-11-09T00:00:00Z"),
        train_end=pd.Timestamp("2026-03-31T23:59:59Z"),
        source_time_max=pd.Timestamp("2026-03-31T23:55:00Z"),
        target_time_max=pd.Timestamp("2026-03-31T21:55:00Z"),
        source_sha256=SOURCE_PARQUET_SHA256,
        target_sha256="2" * 64,
        source_cascade=_source_cascade_metadata(tmp_path),
        entry_direction_target_policy=_target_policy(
            source_sha256=SOURCE_PARQUET_SHA256
        ),
        scores=scores,
        feature_availability=availability,
        created=RANKING_CREATED,
    )
    import json

    rows = json.loads(path.read_text())["ranked_features"]
    assert [row["name"] for row in rows] == [
        third,
        *sorted((first, second)),
    ] + sorted(availability_names[3:])
    assert [row["rank"] for row in rows] == list(
        range(1, len(availability_names) + 1)
    )


def test_ranker_checkpoint_key_binds_run_source_cache_and_window() -> None:
    base = {
        "run_id": RUN_ID,
        "source_sha256": "1" * 64,
        "mtf_cache_sha256": "2" * 64,
        "source_cascade_sha256": "9" * 64,
        "entry_direction_target_policy_sha256": "a" * 64,
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
        ("entry_direction_target_policy_sha256", "b" * 64),
        ("train_start", pd.Timestamp("2021-03-17T00:00:00Z")),
    ):
        variant = dict(base)
        variant[field] = changed
        assert ranker._ranker_checkpoint_key(**variant) != expected


def test_emit_ranking_rejects_filename_created_timestamp_mismatch(tmp_path: Path) -> None:
    out = tmp_path / f"{ranker.RANKING_EVENT_PREFIX}_20260718T090000000001Z.json"
    with pytest.raises(RuntimeError, match="OUTPUT_TIMESTAMP_MISMATCH"):
        _names, availability, scores = _availability_fixture()
        ranker.emit_ranking(
            out_path=out,
            run_id=RUN_ID,
            train_start=pd.Timestamp("2020-11-09T00:00:00Z"),
            train_end=pd.Timestamp("2026-03-31T23:59:59Z"),
            source_time_max=pd.Timestamp("2026-03-31T23:55:00Z"),
            target_time_max=pd.Timestamp("2026-03-31T21:55:00Z"),
            source_sha256=SOURCE_PARQUET_SHA256,
            target_sha256="2" * 64,
            source_cascade=_source_cascade_metadata(tmp_path),
            entry_direction_target_policy=_target_policy(
                source_sha256=SOURCE_PARQUET_SHA256
            ),
            scores=scores,
            feature_availability=availability,
            created=datetime(2026, 7, 18, 9, 0, 0, 2, tzinfo=timezone.utc),
        )
