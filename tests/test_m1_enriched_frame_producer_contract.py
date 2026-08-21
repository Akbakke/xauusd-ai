from __future__ import annotations

import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def _load_builder():
    script = (
        Path(__file__).resolve().parents[1]
        / "gx1"
        / "scripts"
        / "build_entry_exit_m1_enriched_frame_v1.py"
    )
    spec = importlib.util.spec_from_file_location("m1_enriched_builder", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _source_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "time": pd.to_datetime(
                ["2026-01-01T00:00:00Z", "2026-01-01T00:01:00Z"],
                utc=True,
            ),
        }
    )


def _write_pair_manifest(
    path: Path,
    *,
    pair_generation_id: str,
    native_m1: dict[str, object],
) -> None:
    path.write_text(
        json.dumps(
            {
                "pair_generation_id": pair_generation_id,
                "lineage": {"native_sources": {"m1": native_m1}},
            }
        ),
        encoding="utf-8",
    )


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_m1_producer_requires_exact_pair_native_m1_binding(tmp_path: Path) -> None:
    builder = _load_builder()
    pair_id = "a" * 64
    frame = _source_frame()
    source = {
        "root": "/data/native/V3",
        "manifest_path": "/data/native/V3/MANIFEST.json",
        "manifest_sha256": "b" * 64,
    }
    expected = {
        **source,
        "row_count": len(frame),
        "time_min_utc": frame["time"].iloc[0].isoformat(),
        "time_max_utc": frame["time"].iloc[-1].isoformat(),
    }
    pair_manifest = tmp_path / "pair.json"
    _write_pair_manifest(
        pair_manifest,
        pair_generation_id=pair_id,
        native_m1=expected,
    )

    result = builder._require_pair_binding(
        pair_manifest_path=pair_manifest,
        expected_pair_manifest_sha256=_sha256_file(pair_manifest),
        pair_generation_id=pair_id,
        source_identity=source,
        native_summary={
            "row_count": len(frame),
            "time_min_utc": frame["time"].iloc[0].isoformat(),
            "time_max_utc": frame["time"].iloc[-1].isoformat(),
        },
    )
    assert result["pair_generation_id"] == pair_id
    assert result["native_m1"] == expected

    stale = dict(expected)
    stale["manifest_sha256"] = "c" * 64
    _write_pair_manifest(
        pair_manifest,
        pair_generation_id=pair_id,
        native_m1=stale,
    )
    with pytest.raises(RuntimeError, match="M1_ENRICHED_PAIR_NATIVE_M1_BINDING_MISMATCH"):
        builder._require_pair_binding(
            pair_manifest_path=pair_manifest,
            expected_pair_manifest_sha256=_sha256_file(pair_manifest),
            pair_generation_id=pair_id,
            source_identity=source,
            native_summary={
                "row_count": len(frame),
                "time_min_utc": frame["time"].iloc[0].isoformat(),
                "time_max_utc": frame["time"].iloc[-1].isoformat(),
            },
        )

    # The chain's validated pair hash is the authority; a producer that
    # measured its own would silently accept a swapped manifest.
    _write_pair_manifest(
        pair_manifest,
        pair_generation_id=pair_id,
        native_m1=expected,
    )
    with pytest.raises(
        RuntimeError, match="M1_ENRICHED_PAIR_MANIFEST_SHA256_MISMATCH"
    ):
        builder._require_pair_binding(
            pair_manifest_path=pair_manifest,
            expected_pair_manifest_sha256="d" * 64,
            pair_generation_id=pair_id,
            source_identity=source,
            native_summary={
                "row_count": len(frame),
                "time_min_utc": frame["time"].iloc[0].isoformat(),
                "time_max_utc": frame["time"].iloc[-1].isoformat(),
            },
        )


def test_enriched_producer_is_fixed_to_one_worker() -> None:
    builder = _load_builder()
    source = Path(builder.__file__).read_text(encoding="utf-8")
    assert 'parser.add_argument("--workers", type=int, default=1)' in source
    assert "or workers != 1" in source


def test_enriched_writer_publishes_exact_schema_in_bounded_row_groups(
    tmp_path: Path,
) -> None:
    builder = _load_builder()
    index = pd.date_range("2026-01-01", periods=5, freq="min", tz="UTC")
    frame = pd.DataFrame(
        {
            name: np.arange(len(index), dtype=np.float64)
            for name in builder.OUTPUT_COLUMNS[1:]
        },
        index=index,
    )
    output = tmp_path / "enriched.parquet"

    builder._write_output_parquet_bounded(frame, output, chunk_rows=2)

    observed = pd.read_parquet(output)
    assert tuple(observed.columns) == builder.OUTPUT_COLUMNS
    assert observed["time"].tolist() == list(index)
    assert len(observed) == len(frame)
    assert not list(tmp_path.glob(".*.partial-*"))


def _native_frame(times: pd.DatetimeIndex) -> pd.DataFrame:
    index = np.arange(len(times), dtype=np.float64)
    open_ = 2_000.0 + index
    close = open_ + 0.1
    high = close + 0.2
    low = open_ - 0.2
    frame = pd.DataFrame(
        {
            "time": times,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "bid_open": open_ - 0.05,
            "bid_high": high - 0.05,
            "bid_low": low - 0.05,
            "bid_close": close - 0.05,
            "ask_open": open_ + 0.05,
            "ask_high": high + 0.05,
            "ask_low": low + 0.05,
            "ask_close": close + 0.05,
            "volume": np.arange(1, len(times) + 1, dtype=np.int64),
        }
    )
    return frame


def test_native_source_is_spooled_in_bounded_batches_without_concat(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    builder = _load_builder()
    root = tmp_path / "native"
    first_dir = root / "year=2025"
    second_dir = root / "year=2026"
    first_dir.mkdir(parents=True)
    second_dir.mkdir()
    (root / "MANIFEST.json").write_text("{}\n", encoding="utf-8")
    first = _native_frame(
        pd.date_range("2025-12-31T23:56:00Z", periods=4, freq="min")
    )
    second = _native_frame(
        pd.date_range("2026-01-01T00:00:00Z", periods=5, freq="min")
    )
    first.to_parquet(first_dir / "part-000.parquet", index=False, row_group_size=2)
    second.to_parquet(second_dir / "part-000.parquet", index=False, row_group_size=2)

    def concat_forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError(f"read-all concat reached: {args!r} {kwargs!r}")

    monkeypatch.setattr(builder.pd, "concat", concat_forbidden)
    output = tmp_path / "native-stage.parquet"
    identity, summary, bounded = builder._materialize_native_source_bounded(
        root,
        timeframe="M1",
        output=output,
        batch_rows=2,
    )

    observed = pd.read_parquet(output)
    assert tuple(observed.columns) == builder.RAW_COLUMNS
    assert len(observed) == 9
    assert summary == {
        "row_count": 9,
        "time_min_utc": "2025-12-31T23:56:00+00:00",
        "time_max_utc": "2026-01-01T00:04:00+00:00",
    }
    assert len(identity["part_paths"]) == 2
    assert bounded["configured_batch_rows"] == 2
    assert 0 < bounded["maximum_observed_batch_rows"] <= 2
    source = Path(builder.__file__).read_text(encoding="utf-8")
    assert "pd.concat(" not in source
    assert "_load_native_frame" not in source


def test_generation_publication_rolls_back_parquet_and_cache_on_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    builder = _load_builder()
    output_stage = tmp_path / "staged.parquet"
    manifest_stage = tmp_path / "staged.manifest.json"
    cache_stage = tmp_path / ".cache.prepared"
    output_stage.write_bytes(b"parquet-bytes")
    manifest_stage.write_bytes(b"manifest-bytes")
    cache_stage.mkdir()
    (cache_stage / "manifest.json").write_text("{}\n", encoding="utf-8")
    output_final = tmp_path / "final.parquet"
    manifest_final = tmp_path / "final.parquet.manifest.json"
    cache_final = tmp_path / "cache"
    real_publish = builder._publish_file_noreplace

    def publish_then_fail(source: Path, destination: Path) -> None:
        if destination == manifest_final:
            raise RuntimeError("injected manifest failure")
        real_publish(source, destination)

    monkeypatch.setattr(builder, "_publish_file_noreplace", publish_then_fail)
    monkeypatch.setattr(builder, "_rename_dir_noreplace", os.rename)

    with pytest.raises(RuntimeError, match="injected manifest failure"):
        builder._publish_prepared_generation(
            output_stage=output_stage,
            output_final=output_final,
            manifest_stage=manifest_stage,
            manifest_final=manifest_final,
            cache_stage=cache_stage,
            cache_final=cache_final,
        )

    assert not output_final.exists()
    assert not manifest_final.exists()
    assert not cache_final.exists()
    assert cache_stage.is_dir()


def test_generation_manifest_is_the_last_published_commit_marker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    builder = _load_builder()
    output_stage = tmp_path / "staged.parquet"
    manifest_stage = tmp_path / "staged.manifest.json"
    output_stage.write_bytes(b"parquet-bytes")
    manifest_stage.write_bytes(b"manifest-bytes")
    output_final = tmp_path / "final.parquet"
    manifest_final = tmp_path / "final.parquet.manifest.json"
    real_publish = builder._publish_file_noreplace
    published: list[Path] = []

    def record_publish(source: Path, destination: Path) -> None:
        real_publish(source, destination)
        published.append(destination)

    monkeypatch.setattr(builder, "_publish_file_noreplace", record_publish)
    builder._publish_prepared_generation(
        output_stage=output_stage,
        output_final=output_final,
        manifest_stage=manifest_stage,
        manifest_final=manifest_final,
    )

    assert published == [output_final, manifest_final]
    assert output_final.read_bytes() == output_stage.read_bytes()
    assert manifest_final.read_bytes() == manifest_stage.read_bytes()


def test_full_frame_phases_run_in_disposable_child_processes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    builder = _load_builder()
    parent_pid = os.getpid()

    def fake_canonical_stage(**kwargs: object) -> dict[str, object]:
        return {"pid": os.getpid(), "kwargs": kwargs}

    monkeypatch.setattr(builder, "_build_canonical_stage", fake_canonical_stage)
    result = builder._run_isolated_stage(
        "canonical",
        kwargs={"sentinel": "exact"},
        report_path=tmp_path / "stage-report.json",
    )

    assert result["kwargs"] == {"sentinel": "exact"}
    assert isinstance(result["pid"], int)
    assert result["pid"] != parent_pid


def test_enriched_producer_can_satisfy_every_declared_ctx_cont_field() -> None:
    """The declared output contract must have a producer, not only a name.

    ``OUTPUT_COLUMNS`` requires every ``MODEL_NATIVE_CTX_CONT_FIELDS`` name on
    the emitted frame.  Ten of them --- ``{tf}_trend_state_age_bars_v2`` and
    ``{tf}_ema_stack_aligned_v2`` on the five declared context lanes --- are
    created by exactly one expression in the repository, the
    ``out[f"{tf_lower}_{output_name}_v2"]`` assignment inside
    ``htf_features.project_multi_tf_v4_scalars``.  Before this producer called
    it, the ten names were declared and unproducible, and
    ``_finish_model_native_surface`` could only have raised
    ``*_ENRICHED_OUTPUT_FIELDS_MISSING``; every chain run died earlier, on the
    V4 cache warmup, so the gate was never reached.
    """

    from gx1.contracts.entry_model_native_signal_v1 import (
        MODEL_NATIVE_CTX_CONT_FIELDS,
    )

    builder = _load_builder()
    assert set(MODEL_NATIVE_CTX_CONT_FIELDS).issubset(builder.OUTPUT_COLUMNS)

    stage_source = inspect.getsource(builder._build_enriched_stage)
    ordered_calls = (
        "_complete_v4_owned_context(",
        "_attach_ctx_cont_regime_projection(",
        "attach_group_a_ctx_columns_parallel(",
        "_trim_group_a_causal_warmup(",
        "_finish_model_native_surface(",
    )
    positions = [stage_source.index(call) for call in ordered_calls]
    assert positions == sorted(positions)
    assert stage_source.count("_attach_ctx_cont_regime_projection(") == 1

    projection_source = inspect.getsource(builder._attach_ctx_cont_regime_projection)
    assert "project_multi_tf_v4_scalars(" in projection_source


def test_enriched_group_a_warmup_is_trimmed_as_one_honest_prefix() -> None:
    """Long-memory Group-A absence is removed, never numerically filled."""

    from gx1.contracts.entry_model_native_signal_v1 import (
        MODEL_NATIVE_CTX_CONT_GROUP_A_FIELDS,
    )

    builder = _load_builder()
    index = pd.date_range("2026-01-01", periods=6, freq="1min", tz="UTC")
    frame = pd.DataFrame(
        {
            name: np.array([np.nan, np.nan, 1.0, 2.0, 3.0, 4.0])
            for name in MODEL_NATIVE_CTX_CONT_GROUP_A_FIELDS
        },
        index=index,
    )
    frame["close"] = np.arange(len(index), dtype=np.float64)

    out = builder._trim_group_a_causal_warmup(frame)

    assert out.index.equals(index[2:])
    assert out.attrs["causal_context_warmup_rows_trimmed"] == 2
    assert np.isfinite(
        out[list(MODEL_NATIVE_CTX_CONT_GROUP_A_FIELDS)].to_numpy()
    ).all()
    np.testing.assert_array_equal(out["close"], frame.loc[index[2]:, "close"])


def test_enriched_group_a_warmup_trim_rejects_an_interior_gap() -> None:
    from gx1.contracts.entry_model_native_signal_v1 import (
        MODEL_NATIVE_CTX_CONT_GROUP_A_FIELDS,
    )

    builder = _load_builder()
    frame = pd.DataFrame(
        {
            name: np.ones(4, dtype=np.float64)
            for name in MODEL_NATIVE_CTX_CONT_GROUP_A_FIELDS
        }
    )
    frame.loc[0, MODEL_NATIVE_CTX_CONT_GROUP_A_FIELDS[0]] = np.nan
    frame.loc[2, MODEL_NATIVE_CTX_CONT_GROUP_A_FIELDS[0]] = np.nan

    with pytest.raises(RuntimeError, match="not a contiguous warmup prefix"):
        builder._trim_group_a_causal_warmup(frame)


def test_enriched_regime_projection_binds_the_one_m5_source_owner() -> None:
    """One projection map, imported --- not a second tuple spelled the same.

    Rule 13: a consumer that restates the pairs is not an owner.  The producer
    must hold the *same objects* the M5 source materializer cross-checks its
    output against, or the two routes can drift into two formulas for one name.
    """

    from gx1.scripts import materialize_entry_model_native_m5_source_v1 as m5_source

    builder = _load_builder()
    assert builder.REGIME_COMPACT_PROJECTION is m5_source.REGIME_COMPACT_PROJECTION
    assert builder.REGIME_PROJECTED_FIELDS is m5_source.REGIME_PROJECTED_FIELDS
    assert set(builder.REGIME_PROJECTED_FIELDS).issubset(builder.OUTPUT_COLUMNS)
    assert set(builder.REGIME_PROJECTED_FIELDS).issubset(m5_source.ENRICHED_COLUMNS)


@pytest.mark.parametrize(
    "duration",
    [pd.Timedelta(minutes=1), pd.Timedelta(minutes=5)],
)
def test_enriched_regime_projection_forwards_the_contract_and_trims_warmup(
    monkeypatch: pytest.MonkeyPatch,
    duration: pd.Timedelta,
) -> None:
    builder = _load_builder()
    index = pd.date_range("2026-01-01", periods=6, freq=duration, tz="UTC")
    captured: dict[str, object] = {}

    def fake_projection(multi_tf, target_ts_ns, per_tf_map, tfs, skip, **kwargs):
        captured["multi_tf"] = multi_tf
        captured["target_ts_ns"] = np.asarray(target_ts_ns)
        captured["per_tf_map"] = per_tf_map
        captured["tfs"] = tfs
        captured["skip"] = skip
        captured["kwargs"] = kwargs
        column = np.arange(len(index), dtype=np.float64)
        column[:2] = np.nan
        return {name: column.copy() for name in builder.REGIME_PROJECTED_FIELDS}

    monkeypatch.setattr(builder, "project_multi_tf_v4_scalars", fake_projection)
    frame = pd.DataFrame({"close": np.arange(len(index), dtype=np.float64)}, index=index)

    out = builder._attach_ctx_cont_regime_projection(
        frame,
        multi_tf={"sentinel": "exact"},
        decision_bar_duration=duration,
    )

    # The exact declared objects reach the projection owner, with this route's
    # own decision clock and the empty skip set the M5 source route uses.
    assert captured["per_tf_map"] is builder.REGIME_COMPACT_PROJECTION
    assert captured["tfs"] is builder.MULTI_TF_TIMEFRAMES_LOWER_M5_LAST
    assert captured["skip"] == frozenset()
    assert captured["kwargs"] == {"decision_bar_duration": duration}
    assert np.array_equal(captured["target_ts_ns"], index.asi8)

    # Warmup absence is removed by the shared trim owner, never filled.
    assert set(builder.REGIME_PROJECTED_FIELDS).issubset(out.columns)
    assert len(out) == len(index) - 2
    assert out.index[0] == index[2]
    assert np.isfinite(out[list(builder.REGIME_PROJECTED_FIELDS)].to_numpy()).all()


def test_enriched_regime_projection_fails_closed_on_owner_and_field_conflicts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    builder = _load_builder()
    index = pd.date_range("2026-01-01", periods=4, freq="5min", tz="UTC")
    frame = pd.DataFrame({"close": np.arange(4, dtype=np.float64)}, index=index)

    duplicate = frame.copy()
    duplicate[builder.REGIME_PROJECTED_FIELDS[0]] = 0.0
    with pytest.raises(
        RuntimeError, match="ENTRY_EXIT_ENRICHED_DUPLICATE_REGIME_OWNER"
    ):
        builder._attach_ctx_cont_regime_projection(
            duplicate,
            multi_tf={},
            decision_bar_duration=pd.Timedelta(minutes=5),
        )

    def short_projection(multi_tf, target_ts_ns, per_tf_map, tfs, skip, **kwargs):
        return {
            name: np.zeros(len(index), dtype=np.float64)
            for name in builder.REGIME_PROJECTED_FIELDS[1:]
        }

    monkeypatch.setattr(builder, "project_multi_tf_v4_scalars", short_projection)
    with pytest.raises(
        RuntimeError, match="ENTRY_EXIT_ENRICHED_REGIME_PROJECTION_FIELDS_MISMATCH"
    ):
        builder._attach_ctx_cont_regime_projection(
            frame,
            multi_tf={},
            decision_bar_duration=pd.Timedelta(minutes=5),
        )
