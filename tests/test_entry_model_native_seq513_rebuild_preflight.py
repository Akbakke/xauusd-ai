import argparse
import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from tests.volatility_squeeze_test_support import (
    make_volatility_squeeze_artifact_set,
)


from gx1.contracts.entry_model_native_signal_v1 import (
    MODEL_NATIVE_AVAILABLE_CANDIDATE_FIELDS,
    MODEL_NATIVE_BASE_SIGNAL_DIM,
    MODEL_NATIVE_CONTRACT_MODE,
    MODEL_NATIVE_CTX_CAT_DIM,
    MODEL_NATIVE_CTX_CONT_DIM,
    MODEL_NATIVE_DIRECTION_LOGIT_MODE,
    MODEL_NATIVE_MANDATORY_SELECTED_FIELDS,
    MODEL_NATIVE_SELECTED_FEATURE_COUNT,
    MODEL_NATIVE_SEQ_LEN,
    MODEL_NATIVE_SIGNAL_DIM,
)
from gx1.contracts.entry_model_native_aux_targets_v3 import (
    MODEL_NATIVE_AUX_MAX_FUTURE_HORIZON_BARS,
    MODEL_NATIVE_AUX_TARGET_COLUMNS,
    MODEL_NATIVE_AUX_TARGET_SCHEMA_VERSION,
)
from gx1.contracts.entry_model_native_feature_availability_v1 import (
    fit_feature_availability_contract,
)
from gx1.features.entry_model_native_feature_layers_v1 import (
    PRICE_DERIVED_CAUSAL_WARMUP_ROWS,
)
from gx1.contracts.entry_exit_feature_base_v1 import (
    entry_exit_shared_feature_base_contract,
)
from gx1.contracts.entry_exit_feature_surface_v1 import (
    ENTRY_EXIT_FEATURE_SURFACE_SCHEMA_VERSION,
    ENTRY_EXIT_M5_FEATURE_SURFACE_SCHEMA_VERSION,
)
from gx1.contracts.entry_model_native_state_v2 import (
    MODEL_NATIVE_HISTORY_MODE,
    MODEL_NATIVE_STATE_SCHEMA_VERSION,
)
from gx1.contracts.unified_exit_lifecycle_v1 import (
    UNIFIED_EXIT_LIFECYCLE_EPISODE_SCHEMA_VERSION,
    UNIFIED_EXIT_M1_AUTHORITY_SCHEMA_VERSION,
)
from gx1.contracts.xau_tape_provenance_v1 import (
    SEQ513_SOURCE_CASCADE_PAIR_PROOF_SCHEMA_VERSION,
)
from gx1.features.entry_specialist_feature_groups_v1 import group_features_by_specialist
from gx1.features.htf_features import (
    HTF_V4_MATRIX_CONTRACT,
    MODEL_NATIVE_MTF_SCALAR_CONTRACT_V4,
    MODEL_NATIVE_MTF_SCALAR_FIELDS_BY_TIMEFRAME_V4,
    MULTI_TF_FEATURE_COUNT_V4,
    MULTI_TF_PER_BAR_FEATURES_V4,
    MULTI_TF_RESAMPLE_RULES,
    MULTI_TF_SHIFT,
    build_multi_tf_v4_closed_timestamp_indices,
    multi_tf_last_closed_label,
)
from gx1.scripts.build_entry_v10_ctx_training_dataset_v3 import (
    ENTRY_FITTED_Q_DATASET_STEM_SUFFIX,
)
from gx1.scripts import (
    materialize_entry_model_native_seq513_rebuild_preflight_v1 as preflight,
)
from gx1.scripts import (
    materialize_entry_model_native_seq513_signal_manifest_v1 as signal_manifest_producer,
)
from gx1.scripts.prebuild_multi_tf_cache_v4 import publish_multi_tf_v4_cache
from tests.model_native_signal_support import canonical_model_native_selected_fields
from tests.entry_direction_target_policy_support import (
    causal_m1_target_policy_fixture,
)

from tests.htf_v29_registry_test_support import (
    synthetic_v29_registry_constants,
)

_V29_TEST_REGISTRY_CONSTANTS = synthetic_v29_registry_constants()


STAMP = "20260716T120000123456Z"
CREATED = "2026-07-16T12:00:00.123456+00:00"
RUN_ID = "XAU_SEQ513_REBUILD_TEST_V1"
# The rebuild output name must carry the dataset stem suffix owned by the
# builder, never a restated literal.
DATASET_OUTPUT_NAME = f"model_native{ENTRY_FITTED_Q_DATASET_STEM_SUFFIX}.parquet"
SPLITS = {
    "history_start": "2020-11-08T00:00:00+00:00",
    "train_start": "2020-11-09T00:00:00+00:00",
    "train_end": "2025-09-30T23:59:59+00:00",
    "val_start": "2025-10-01T00:00:00+00:00",
    "val_end": "2025-12-31T23:59:59+00:00",
    "test_start": "2026-01-01T00:00:00+00:00",
    "test_end": "2026-06-26T03:25:00+00:00",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


@pytest.fixture(autouse=True)
def _stub_source_cascade_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        signal_manifest_producer,
        "validate_current_pair_source_cascade_proof",
        lambda path, **kwargs: json.loads(
            Path(path).read_text(encoding="utf-8")
        ),
    )

    def _stub_m1_authority(
        *,
        pair_manifest_path: Path,
        pair_generation_root: Path,
    ) -> tuple[Path, dict[str, object]]:
        source_path = (
            Path(pair_manifest_path).parent
            / "xauusd_m1_lifecycle.parquet"
        ).resolve(strict=True)
        authority = {
            "schema_version": UNIFIED_EXIT_M1_AUTHORITY_SCHEMA_VERSION,
            "pair_manifest_path": str(Path(pair_manifest_path)),
            "pair_generation_root": str(Path(pair_generation_root)),
            "m1_source_path": str(source_path),
            "m1_source_sha256": _sha256(source_path),
            "pair_generation_id": "fixture-pair-generation-v1",
        }
        return source_path, authority

    monkeypatch.setattr(
        preflight,
        "require_unified_exit_m1_pair_authority",
        _stub_m1_authority,
    )


def _write_parquet(path: Path, payload: dict[str, list]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table(payload), path)
    return path


def _selected_features() -> list[str]:
    return canonical_model_native_selected_fields()


def _stamp(value: datetime) -> str:
    return value.strftime("%Y%m%dT%H%M%S%fZ")


def test_manifest_filename_timestamp_accepts_second_resolution() -> None:
    path = Path("ENTRY_MODEL_NATIVE_SEQ513_SIGNAL_MANIFEST_20260720T182009Z.json")
    payload = {"created_utc": "2026-07-20T18:20:09+00:00"}

    assert preflight._manifest_timestamp_matches_created(path, payload)


def _build_fixture(
    tmp_path: Path,
    *,
    mutable_manifest: bool = False,
    break_signal_contract: bool = False,
    break_specialist_coverage: bool = False,
    break_source_manifest_hash: bool = False,
    break_ranking_run_id: bool = False,
    break_ranking_source_hash: bool = False,
    break_ranking_train_window: bool = False,
    break_m1_feature_identity: bool = False,
    break_m5_feature_identity: bool = False,
    break_mtf: bool = False,
    late_mtf_history_tf: str | None = None,
    missing_tape_year: int | None = None,
    missing_tape_column: str | None = None,
) -> argparse.Namespace:
    history_base = datetime.fromisoformat(SPLITS["history_start"])
    history_source_start = history_base - timedelta(days=2)
    history_times = [
        history_source_start + timedelta(minutes=5 * index)
        for index in range(3 * 288 + 1)
    ]
    test_end = datetime.fromisoformat(SPLITS["test_end"])
    test_tail_start = test_end.replace(
        hour=0,
        minute=0,
        second=0,
        microsecond=0,
    ) - timedelta(days=2)
    test_tail = [
        test_tail_start + timedelta(minutes=5 * index)
        for index in range(
            int((test_end - test_tail_start) / timedelta(minutes=5)) + 1
        )
    ]
    source_times = history_times + test_tail
    n_source = len(source_times)
    close = np.linspace(1800.0, 2300.0, n_source)
    source = _write_parquet(
        tmp_path / "inputs/source.parquet",
        {
            "time": source_times,
            "open": close.tolist(),
            "high": (close + 1.0).tolist(),
            "low": (close - 1.0).tolist(),
            "close": close.tolist(),
            "bid_close": (close - 0.05).tolist(),
            "ask_close": (close + 0.05).tolist(),
        },
    )
    canonical = _write_parquet(
        tmp_path / "inputs/canonical_v2.parquet",
        {
            "time": source_times,
            "high": (close + 1.0).tolist(),
            "low": (close - 1.0).tolist(),
            "close": close.tolist(),
            "bid_close": (close - 0.05).tolist(),
            "ask_close": (close + 0.05).tolist(),
            "canonical_feature": np.arange(n_source, dtype=np.float64).tolist(),
        },
    )
    _write_parquet(
        tmp_path / "inputs/xauusd_m1_lifecycle.parquet",
        {
            "time": [
                datetime(2026, 6, 26, 3, 24, tzinfo=timezone.utc),
                datetime(2026, 6, 26, 3, 25, tzinfo=timezone.utc),
            ],
            "open": [2000.0, 2000.1],
            "high": [2000.2, 2000.3],
            "low": [1999.8, 1999.9],
            "close": [2000.1, 2000.2],
            "bid_open": [1999.95, 2000.05],
            "bid_high": [2000.15, 2000.25],
            "bid_low": [1999.75, 1999.85],
            "bid_close": [2000.05, 2000.15],
            "ask_open": [2000.05, 2000.15],
            "ask_high": [2000.25, 2000.35],
            "ask_low": [1999.85, 1999.95],
            "ask_close": [2000.15, 2000.25],
            "volume": [10, 11],
        },
    )
    m1_feature_base = _write_parquet(
        tmp_path / "inputs/xauusd_m1_feature_base.parquet",
        {
            "time": [
                datetime(2026, 6, 26, 3, 24, tzinfo=timezone.utc),
                datetime(2026, 6, 26, 3, 25, tzinfo=timezone.utc),
            ],
        },
    )
    m1_feature_manifest_path = Path(
        str(m1_feature_base) + ".manifest.json"
    )
    _write_json(
        m1_feature_manifest_path,
        {
            "schema_version": ENTRY_EXIT_FEATURE_SURFACE_SCHEMA_VERSION,
            "decision": "PASS",
            "dataset_run_id": RUN_ID,
            "output_parquet": str(m1_feature_base),
            "output_parquet_sha256": _sha256(m1_feature_base),
            "pair_generation_id": "fixture-pair-generation-v1",
            "rows": 2,
            "shared_feature_base_contract": entry_exit_shared_feature_base_contract(),
        },
    )
    # The producer cannot emit the leading PRICE_DERIVED_CAUSAL_WARMUP_ROWS
    # rows, so the surface it publishes starts after that prefix.
    m5_surface_times = source_times[PRICE_DERIVED_CAUSAL_WARMUP_ROWS:]
    m5_feature_base = _write_parquet(
        tmp_path / "inputs/xauusd_m5_feature_base.parquet",
        {"time": m5_surface_times},
    )
    m5_feature_manifest_path = Path(
        str(m5_feature_base) + ".manifest.json"
    )
    _write_json(
        m5_feature_manifest_path,
        {
            "schema_version": ENTRY_EXIT_M5_FEATURE_SURFACE_SCHEMA_VERSION,
            "decision": "PASS",
            "dataset_run_id": RUN_ID,
            "output_parquet": str(m5_feature_base),
            "output_parquet_sha256": _sha256(m5_feature_base),
            "pair_generation_id": "fixture-pair-generation-v1",
            "rows": len(m5_surface_times),
            "shared_feature_base_contract": entry_exit_shared_feature_base_contract(),
        },
    )
    m1_pair_manifest = _write_json(
        tmp_path / "inputs/PAIR_MANIFEST.json",
        {"fixture": True},
    )
    m1_pair_generation_root = tmp_path / "inputs/pair_generations"
    m1_pair_generation_root.mkdir()

    source_layer = _write_json(
        tmp_path / f"inputs/MODEL_NATIVE_SIGNAL_LAYER_{STAMP}.json",
        {"created_utc": CREATED, "proof": True},
    )
    now = datetime.now(timezone.utc)
    # Keep event ordering stable even if the host clock is corrected backwards
    # by tens of seconds while this intentionally heavy fixture is built.
    ranking_created = now - timedelta(minutes=5)
    ranking_path = tmp_path / "inputs" / (
        "ENTRY_MODEL_NATIVE_TRAIN_FEATURE_RANKING_"
        f"{_stamp(ranking_created)}.json"
    )
    source_cascade = {
        "path": str((source.parent / "SOURCE_CASCADE_PROOF.json").resolve()),
        "sha256": "9" * 64,
        "schema_version": SEQ513_SOURCE_CASCADE_PAIR_PROOF_SCHEMA_VERSION,
        "entry_run_id": RUN_ID,
        "event_root": str(source.parent.resolve()),
        "source_parquet_path": str(source.resolve()),
        "source_parquet_sha256": _sha256(source),
        "canonical_v2_path": str(canonical.resolve()),
        "canonical_v2_sha256": _sha256(canonical),
        "multi_tf_cache_dir": str((tmp_path / "inputs/mtf_cache").resolve()),
        "multi_tf_source_path": str(tmp_path / "m5_enriched.parquet"),
        "multi_tf_source_sha256": "6" * 64,
        "multi_tf_manifest_sha256": "4" * 64,
        "multi_tf_cache_identity_sha256": "5" * 64,
        "pair_manifest_path": str(m1_pair_manifest.resolve()),
        "pair_manifest_sha256": _sha256(m1_pair_manifest),
        "pair_generation_id": "fixture-pair-generation-v1",
        "history_start_utc": SPLITS["history_start"],
        "time_max_utc": SPLITS["test_end"],
    }
    _write_json(Path(source_cascade["path"]), source_cascade)
    direction_target_policy = causal_m1_target_policy_fixture(
        source_parquet_sha256=_sha256(source),
        tape_provenance_sha256="b" * 64,
        train_start_utc=datetime.fromisoformat(SPLITS["train_start"]).isoformat(),
        train_end_utc=datetime.fromisoformat(SPLITS["train_end"]).isoformat(),
    )
    candidate_names = list(MODEL_NATIVE_AVAILABLE_CANDIDATE_FIELDS)
    train_time_max = pd.Timestamp(SPLITS["train_end"]).floor("5min")
    availability_times = pd.date_range(
        end=train_time_max,
        periods=256,
        freq="5min",
    )
    availability_x = np.arange(len(availability_times), dtype=np.float32)
    availability_matrix = np.column_stack(
        [
            availability_x * np.float32(index + 1)
            + np.float32(index * index + 0.125)
            for index in range(len(candidate_names))
        ]
    ).astype(np.float32)
    availability = fit_feature_availability_contract(
        matrix=availability_matrix,
        names=candidate_names,
        times=availability_times,
        train_start=pd.Timestamp(SPLITS["train_start"]),
        train_end=pd.Timestamp(SPLITS["train_end"]),
        diagnostic_target=np.square(availability_x.astype(np.float64)),
    )
    ranking = {
        "schema_version": signal_manifest_producer.TRAIN_FEATURE_RANKING_SCHEMA_VERSION,
        "created_utc": ranking_created.isoformat(),
        "entry_run_id": RUN_ID,
        "producer": signal_manifest_producer.TRAIN_FEATURE_RANKING_PRODUCER,
        "producer_version": signal_manifest_producer.TRAIN_FEATURE_RANKING_PRODUCER_VERSION,
        "fit_scope": "train_only",
        "train_start_utc": SPLITS["train_start"],
        "train_end_utc": SPLITS["train_end"],
        "source_time_max_utc": train_time_max.isoformat(),
        "target_time_max_utc": train_time_max.isoformat(),
        "source_sha256": _sha256(source),
        "target_sha256": "2" * 64,
        "target_contract": dict(
            signal_manifest_producer.TRAIN_FEATURE_RANKING_TARGET_CONTRACT
        ),
        "entry_direction_target_policy": direction_target_policy,
        "entry_direction_target_policy_sha256": direction_target_policy[
            "policy_sha256"
        ],
        "source_cascade": source_cascade,
        "ranking_order": dict(signal_manifest_producer.TRAIN_FEATURE_RANKING_ORDER),
        "causality_contract": dict(
            signal_manifest_producer.TRAIN_FEATURE_CAUSALITY_CONTRACT
        ),
        "feature_availability": availability,
        "ranked_features": [
            {
                "rank": index,
                "name": name,
                "score": float(len(candidate_names) - index + 1)
                / float(len(candidate_names)),
            }
            for index, name in enumerate(candidate_names, start=1)
        ],
    }
    _write_json(ranking_path, ranking)

    # Build time can be material for this large fixture.  Derive the immutable
    # output filename immediately before the producer call so the producer's
    # real 30-second future-skew gate does not make the test clock-sensitive.
    manifest_created = datetime.now(timezone.utc) - timedelta(seconds=1)
    immutable_manifest_path = tmp_path / "inputs" / (
        f"{signal_manifest_producer.SIGNAL_MANIFEST_EVENT_PREFIX}_"
        f"{_stamp(manifest_created)}.json"
    )
    manifest = signal_manifest_producer.run(
        argparse.Namespace(
            feature_ranking_json=str(ranking_path),
            out=str(immutable_manifest_path),
            run_id=RUN_ID,
        )
    )
    manifest["source_manifests"] = {
        "model_native_signal_layer": {
            "path": str(source_layer.resolve()),
            "sha256": "0" * 64 if break_source_manifest_hash else _sha256(source_layer),
        }
    }
    if break_signal_contract:
        manifest["model_native_signal_contract"]["seq_input_dim"] = 512
    if break_specialist_coverage:
        broken_groups = {
            name: list(features)
            for name, features in manifest["features_by_specialist"].items()
        }
        victim_group = next(
            name for name, features in broken_groups.items() if features
        )
        broken_groups[victim_group] = broken_groups[victim_group][1:]
        manifest["features_by_specialist"] = broken_groups

    manifest_path = immutable_manifest_path
    if mutable_manifest:
        manifest_path = tmp_path / "inputs/MODEL_NATIVE_SIGNAL_MANIFEST_latest.json"
        manifest["json_path"] = str(manifest_path.resolve())
        immutable_manifest_path.unlink()
    _write_json(manifest_path, manifest)
    m1_feature_manifest = json.loads(
        m1_feature_manifest_path.read_text(encoding="utf-8")
    )
    ordered_fields = list(manifest["model_native_signal_contract"]["fields"])
    m1_feature_manifest.update(
        {
            "anchor_timeframe": "M1",
            "feature_field_order": ordered_fields,
            "feature_field_order_sha256": _canonical_sha256(ordered_fields),
            "seq_structure_manifest": str(manifest_path.resolve()),
            "seq_structure_manifest_sha256": _sha256(manifest_path),
        }
    )
    if break_m1_feature_identity:
        broken_fields = list(ordered_fields)
        broken_fields[-1] = "chart.stale_exit_only_fixture"
        m1_feature_manifest["feature_field_order"] = broken_fields
        m1_feature_manifest["feature_field_order_sha256"] = (
            _canonical_sha256(broken_fields)
        )
    m1_feature_manifest["manifest_sha256"] = _canonical_sha256(
        m1_feature_manifest
    )
    _write_json(m1_feature_manifest_path, m1_feature_manifest)

    m5_feature_manifest = json.loads(
        m5_feature_manifest_path.read_text(encoding="utf-8")
    )
    m5_feature_manifest.update(
        {
            "anchor_timeframe": "M5",
            "feature_field_order": ordered_fields,
            "feature_field_order_sha256": _canonical_sha256(ordered_fields),
            "seq_structure_manifest": str(manifest_path.resolve()),
            "seq_structure_manifest_sha256": _sha256(manifest_path),
        }
    )
    if break_m5_feature_identity:
        m5_feature_manifest["seq_structure_manifest_sha256"] = "7" * 64
    m5_feature_manifest["manifest_sha256"] = _canonical_sha256(
        m5_feature_manifest
    )
    _write_json(m5_feature_manifest_path, m5_feature_manifest)

    if break_ranking_run_id:
        ranking["entry_run_id"] = "XAU_DIFFERENT_REBUILD_TEST_V1"
    if break_ranking_source_hash:
        ranking["source_sha256"] = "f" * 64
    if break_ranking_train_window:
        ranking["train_start_utc"] = "2020-11-10T00:00:00+00:00"
    if break_ranking_run_id or break_ranking_source_hash or break_ranking_train_window:
        _write_json(ranking_path, ranking)

    source_index = pd.DatetimeIndex(
        pd.to_datetime(source_times, utc=True)
    )
    expected_indices = build_multi_tf_v4_closed_timestamp_indices(source_index)
    mtf_cache = tmp_path / "inputs/mtf_cache"
    mtf_frames: dict[str, pd.DataFrame] = {}
    for tf_offset, tf in enumerate(MULTI_TF_RESAMPLE_RULES):
        timestamps = expected_indices[tf].asi8.astype(np.int64, copy=True)
        row = np.arange(len(timestamps), dtype=np.float32).reshape(-1, 1) + 1.0
        column = (
            np.arange(MULTI_TF_FEATURE_COUNT_V4, dtype=np.float32)
            .reshape(1, -1)
            + 1.0
        )
        values = row * column + np.float32(tf_offset)
        warmup_rows = 0
        if tf == late_mtf_history_tf:
            history_cutoff = multi_tf_last_closed_label(
                SPLITS["history_start"],
                tf,
            )
            warmup_rows = int(
                (expected_indices[tf] <= history_cutoff).sum()
            )
            values[:warmup_rows] = np.nan
        frame = pd.DataFrame(
            values,
            index=pd.to_datetime(timestamps, unit="ns", utc=True),
            columns=MULTI_TF_PER_BAR_FEATURES_V4,
        )
        frame.attrs["feats_np"] = values
        frame.attrs["ts_int64"] = timestamps
        frame.attrs["causal_warmup_rows"] = warmup_rows
        frame.attrs["htf_feature_contract"] = HTF_V4_MATRIX_CONTRACT
        scalar_fields = MODEL_NATIVE_MTF_SCALAR_FIELDS_BY_TIMEFRAME_V4[tf]
        scalar_values = np.ascontiguousarray(
            row * (np.arange(len(scalar_fields), dtype=np.float32) + 0.25)
            + np.float32(tf_offset * 100.0)
        )
        frame.attrs["model_native_mtf_scalar_fields_v4"] = scalar_fields
        frame.attrs["model_native_mtf_scalars_np_v4"] = scalar_values
        frame.attrs["model_native_mtf_scalar_warmup_rows_v4"] = 0
        frame.attrs["model_native_mtf_scalar_contract_v4"] = (
            MODEL_NATIVE_MTF_SCALAR_CONTRACT_V4
        )
        mtf_frames[tf] = frame
    publish_multi_tf_v4_cache(
        out_dir=mtf_cache,
        m5_prebuilt=source.resolve(),
        expected_source_sha256=_sha256(source),
        features=mtf_frames,
        v29_registry_constants=_V29_TEST_REGISTRY_CONSTANTS,
        volatility_squeeze_artifacts=(
            make_volatility_squeeze_artifact_set(tmp_path)
        ),
    )
    if break_mtf:
        mtf_manifest = json.loads(
            (mtf_cache / "manifest.json").read_text(encoding="utf-8")
        )
        mtf_manifest["feature_names"] = list(MULTI_TF_PER_BAR_FEATURES_V4[:-1])
        _write_json(mtf_cache / "manifest.json", mtf_manifest)

    tape_root = tmp_path / "inputs/tape"
    for year in range(2020, 2027):
        if year == missing_tape_year:
            continue
        tape_payload = {
                "time": [datetime(year, 1, 2, tzinfo=timezone.utc)],
                "open": [1800.05 + year],
                "high": [1800.25 + year],
                "low": [1799.85 + year],
                "close": [1800.05 + year],
                "bid_close": [1800.0 + year],
                "bid_high": [1800.2 + year],
                "bid_low": [1799.8 + year],
                "ask_close": [1800.1 + year],
                "ask_high": [1800.3 + year],
                "ask_low": [1799.9 + year],
        }
        if missing_tape_column and year == 2023:
            tape_payload.pop(missing_tape_column)
        _write_parquet(tape_root / f"year={year}/part-000.parquet", tape_payload)

    return argparse.Namespace(
        source_parquet=str(source),
        run_id=RUN_ID,
        canonical_v2_parquet=str(canonical),
        signal_manifest=str(manifest_path),
        feature_ranking_json=str(ranking_path),
        mtf_cache_dir=str(mtf_cache),
        tape_root=str(tape_root),
        m1_lifecycle_pair_manifest_json=str(m1_pair_manifest),
        m1_lifecycle_pair_generation_root=str(
            m1_pair_generation_root
        ),
        m1_feature_base_parquet=str(m1_feature_base),
        m5_feature_base_parquet=str(m5_feature_base),
        exit_lifecycle_dir=str(tmp_path / "run/exit_lifecycle"),
        output=str(tmp_path / "run/dataset" / DATASET_OUTPUT_NAME),
        audit_out_dir=str(tmp_path / "run/pretrain_audit"),
        out_dir=str(tmp_path / "reports"),
        **SPLITS,
    )


def test_preflight_binds_exact_run_lineage_and_wrapper_inputs(
    tmp_path: Path,
) -> None:
    args = _build_fixture(tmp_path)

    report = preflight.run(args)

    assert report["decision"] == preflight.READY_DECISION
    assert report["schema_version"] == "entry_model_native_seq513_rebuild_preflight_v13"
    assert not report["failures"]
    assert report["counts"] == {
        "base_signal_features": MODEL_NATIVE_BASE_SIGNAL_DIM,
        "selected_features": MODEL_NATIVE_SELECTED_FEATURE_COUNT,
        "expected_seq_snap_width": MODEL_NATIVE_SIGNAL_DIM,
        "seq_len": MODEL_NATIVE_SEQ_LEN,
        "ctx_cont_dim": MODEL_NATIVE_CTX_CONT_DIM,
        "ctx_cat_dim": MODEL_NATIVE_CTX_CAT_DIM,
        "required_specialist_count": 8,
        "manifest_variant": MODEL_NATIVE_CONTRACT_MODE,
    }
    assert report["required_model_native_contract"] == {
        "contract_mode": MODEL_NATIVE_CONTRACT_MODE,
        "direction_logit_mode": MODEL_NATIVE_DIRECTION_LOGIT_MODE,
        "seq_input_dim": MODEL_NATIVE_SIGNAL_DIM,
        "snap_input_dim": MODEL_NATIVE_SIGNAL_DIM,
        "seq_len": MODEL_NATIVE_SEQ_LEN,
        "ctx_cont_dim": MODEL_NATIVE_CTX_CONT_DIM,
        "ctx_cat_dim": MODEL_NATIVE_CTX_CAT_DIM,
        "base_signal_dim": MODEL_NATIVE_BASE_SIGNAL_DIM,
        "selected_feature_count": MODEL_NATIVE_SELECTED_FEATURE_COUNT,
        "bridge_dim": 0,
        "bridge_source": None,
        "anchor_source": None,
    }
    assert report["specialist_contract"]["all_eight_covered"] is True
    assert report["specialist_contract"]["mandatory_full_stack_exact"] is True
    assert report["inputs"]["multi_tf_cache"]["exact"] is True
    mtf_window_contract = report["inputs"]["multi_tf_cache"][
        "decision_window_coverage_contract"
    ]
    assert mtf_window_contract == {
        "scope": "cache_exact_source_closed_bar_geometry_v2",
        "target_availability_shift": str(MULTI_TF_SHIFT["M5"]),
        "source_timestamp_geometry_owner": (
            "gx1.features.htf_features."
            "build_multi_tf_v4_closed_timestamp_indices"
        ),
        "per_tf_seq_lens_declared_here": False,
        "equal_timeframe_seq_len_assumed": False,
        "progressive_resolution_pyramid_required": True,
        "resolution_pyramid_owner": (
            "gx1.features.htf_features.require_multi_tf_resolution_pyramid"
        ),
        "exact_split_window_coverage_owner": (
            "gx1.features.htf_features."
            "require_multi_tf_decision_window_coverage"
        ),
    }
    history_start = pd.Timestamp(SPLITS["history_start"])
    test_end = pd.Timestamp(SPLITS["test_end"])
    for tf, row in report["inputs"]["multi_tf_cache"]["tf_rows"].items():
        assert row["feature_history_closed_bar_cutoff_ns"] == int(
            multi_tf_last_closed_label(history_start, tf).value
        )
        assert row["test_end_closed_bar_cutoff_ns"] == int(
            multi_tf_last_closed_label(test_end, tf).value
        )
        assert row["causal_warmup_rows"] == 0
        assert row["first_complete_ts_ns"] == row["first_ts_ns"]
        assert row["source_timestamp_geometry_exact"] is True
        assert row["covers_feature_history_start"] is True
    assert (
        report["inputs"]["multi_tf_cache"]["source"][
            "timestamp_geometry_exact"
        ]
        is True
    )
    assert report["inputs"]["tape"]["exact"] is True
    assert report["inputs"]["source_parquet"]["sha256"] == _sha256(
        Path(args.source_parquet)
    )
    assert report["inputs"]["canonical_v2_parquet"]["sha256"] == _sha256(
        Path(args.canonical_v2_parquet)
    )

    command = report["rebuild_command_contract"]
    argv = command["argv_template"]
    assert argv[:3] == [
        "scripts/rebuild_entry_model_native_seq513_dataset.sh",
        "--run-id",
        RUN_ID,
    ]
    for flag in (
        "--source-parquet",
        "--canonical-v2-parquet",
        "--signal-manifest",
        "--feature-ranking-json",
        "--mtf-cache-dir",
        "--tape-root",
        "--m1-lifecycle-pair-manifest-json",
        "--m1-lifecycle-pair-generation-root",
        "--m1-feature-base-parquet",
        "--m5-feature-base-parquet",
        "--exit-lifecycle-dir",
        "--output",
        "--audit-out-dir",
        "--history-start",
        "--train-start",
        "--train-end",
        "--val-start",
        "--val-end",
        "--test-start",
        "--test-end",
    ):
        assert argv.count(flag) == 1
    assert command["run_lineage_required"] is True
    assert command["entry_run_id"] == RUN_ID
    assert command["unified_exit_lifecycle_contract"] == {
        "schema_version": UNIFIED_EXIT_LIFECYCLE_EPISODE_SCHEMA_VERSION,
        "m1_pair_manifest_json": str(
            Path(args.m1_lifecycle_pair_manifest_json)
        ),
        "m1_pair_generation_root": str(
            Path(args.m1_lifecycle_pair_generation_root)
        ),
        "output_dir": str(Path(args.exit_lifecycle_dir).resolve()),
        "target_policy": "fit_once_on_exact_train_native_m1_then_hash_bound",
        "val_test_rows_used_for_target_policy_fit": 0,
        "required_m1_columns": [
            "time",
            "open",
            "high",
            "low",
            "close",
            "bid_open",
            "bid_high",
            "bid_low",
            "bid_close",
            "ask_open",
            "ask_high",
            "ask_low",
            "ask_close",
            "volume",
        ],
        "both_sides_per_entry_snapshot": True,
        "starts_training": False,
    }
    assert command["run_id_validated"] is True
    assert "<EXPLICIT_RUN_ID_ID>" not in argv
    # V30 retired the TRAIN rank-reference subsystem: the rebuild command may
    # not declare or require it again.
    assert "--rank-reference-npz" not in argv
    assert "rank_reference_contract" not in command
    assert "rank_reference_run_id_match_required" not in (
        command["fixed_builder_contract"]
    )
    assert command["fixed_builder_contract"]["state_schema_version"] == MODEL_NATIVE_STATE_SCHEMA_VERSION
    assert command["fixed_builder_contract"]["entry_direction_target_policy"] == (
        "fit_once_on_exact_train_m5_decisions_and_pair_bound_m1_fills_then_hash_bound"
    )
    assert command["fixed_builder_contract"]["direction_target_mode"] == (
        "train_fitted_exact_m1_execution_pnl_v1"
    )
    assert command["fixed_builder_contract"]["run_lineage_required"] is True
    aux_contract = command["fixed_builder_contract"]["aux_head_target_contract"]
    assert aux_contract["schema_version"] == MODEL_NATIVE_AUX_TARGET_SCHEMA_VERSION
    assert len(aux_contract["columns"]) == len(MODEL_NATIVE_AUX_TARGET_COLUMNS)
    assert aux_contract["max_future_horizon_bars"] == int(
        MODEL_NATIVE_AUX_MAX_FUTURE_HORIZON_BARS
    )
    assert aux_contract["spread_aware_risk_magnitudes_required"] is True
    assert aux_contract["target_value_domains"]["dip_mfe"]["signed"] is True
    assert (
        aux_contract["target_value_domains"]["dip_mfe"]["negative_values_preserved"]
        is True
    )
    assert (
        aux_contract["target_value_domains"]["dip_mfe"]["lower_bound_bps"] is None
    )
    assert aux_contract["target_value_domains"]["dip_mae"]["signed"] is False
    assert (
        aux_contract["target_value_domains"]["dip_mae"]["lower_bound_bps"] == 0.0
    )
    assert aux_contract["incomplete_rows_may_be_emitted"] is False
    assert command["fixed_builder_contract"]["feature_history_mode"] == MODEL_NATIVE_HISTORY_MODE
    assert command["fixed_builder_contract"]["split_reset_allowed"] is False
    assert report["inputs"]["source_time_contract"]["exact"] is True
    assert report["entry_run_id"] == RUN_ID
    assert report["inputs"]["feature_ranking_json"]["sha256"] == _sha256(
        Path(args.feature_ranking_json)
    )
    assert report["signal_training_lineage"]["entry_run_id"] == RUN_ID
    assert report["training_allowed"] is False
    assert report["side_effects_started"] == {
        "dataset_rebuild": False,
        "training": False,
        "replay": False,
        "iql_distillation": False,
        "shadow": False,
        "live": False,
    }
    assert Path(report["json_path"]).name.startswith(
        "ENTRY_MODEL_NATIVE_SEQ513_REBUILD_PREFLIGHT_"
    )
    assert not list(Path(report["json_path"]).parent.glob("*_latest.json"))

    serialized = json.dumps(report, sort_keys=True).lower()
    for retired in (
        "smart_report",
        "inventory_report",
        "source_dataset_dir",
        "planned_dataset_dir",
        "allow_zero_ctx",
    ):
        assert retired not in serialized


def test_mtf_cache_contract_fails_closed_without_borrowing_seq513_length(
    tmp_path: Path,
) -> None:
    args = _build_fixture(tmp_path, late_mtf_history_tf="D1")

    contract = preflight._mtf_cache_contract(
        Path(args.mtf_cache_dir),
        history_start=datetime.fromisoformat(SPLITS["history_start"]),
        test_end=datetime.fromisoformat(SPLITS["test_end"]),
    )

    assert contract["verified_loader_error"] is None
    assert contract["exact"] is False
    assert contract["tf_rows"]["D1"]["first_complete_ts_ns"] > int(
        multi_tf_last_closed_label(SPLITS["history_start"], "D1").value
    )
    assert contract["tf_rows"]["D1"]["covers_feature_history_start"] is False
    assert all(
        contract["tf_rows"][tf]["covers_feature_history_start"]
        for tf in MULTI_TF_RESAMPLE_RULES
        if tf != "D1"
    )
    assert (
        "MODEL_NATIVE_SEQ_LEN"
        not in preflight._mtf_cache_contract.__code__.co_names
    )


@pytest.mark.parametrize(
    ("fixture_kwargs", "failure_name"),
    [
        (
            {"break_signal_contract": True},
            "signal manifest proves exact current ordered signal and 168/5 intent",
        ),
        (
            {"break_specialist_coverage": True},
            "declared specialist feature groups match derived routing",
        ),
        (
            {"break_source_manifest_hash": True},
            "declared signal source manifests are present and hash-bound",
        ),
        (
            {"break_m1_feature_identity": True},
            "explicit M1 feature base is the PASS artifact for this run and pair source",
        ),
        (
            {"break_m5_feature_identity": True},
            "explicit M5 feature base is the exact Entry surface for this run and pair source",
        ),
        (
            {"break_mtf": True},
            "explicit MTF cache has exact five-TF files/schema/source hash/coverage",
        ),
        (
            {"missing_tape_year": 2023},
            "explicit tape root has exact spread-aware OHLC coverage for every split year",
        ),
        (
            {"missing_tape_column": "ask_low"},
            "explicit tape root has exact spread-aware OHLC coverage for every split year",
        ),
    ],
)
def test_preflight_blocks_incomplete_contracts(
    tmp_path: Path, fixture_kwargs: dict, failure_name: str
) -> None:
    report = preflight.run(_build_fixture(tmp_path, **fixture_kwargs))

    assert report["decision"] == preflight.BLOCKED_DECISION
    assert failure_name in {row["name"] for row in report["failures"]}
    assert report["dataset_rebuild_allowed"] is False


@pytest.mark.parametrize(
    ("fixture_kwargs", "lineage_error"),
    [
        (
            {"break_ranking_run_id": True},
            "FEATURE_RANKING_ENTRY_RUN_ID_INVALID",
        ),
        (
            {"break_ranking_source_hash": True},
            "ENTRY_CAUSAL_M1_TARGET_POLICY_SOURCE_PARQUET_SHA256_MISMATCH",
        ),
        (
            {"break_ranking_train_window": True},
            "FEATURE_RANKING_ENTRY_TARGET_POLICY_MISMATCH",
        ),
    ],
)
def test_preflight_rejects_ranking_lineage_mismatch(
    tmp_path: Path,
    fixture_kwargs: dict,
    lineage_error: str,
) -> None:
    report = preflight.run(_build_fixture(tmp_path, **fixture_kwargs))

    assert report["decision"] == preflight.BLOCKED_DECISION
    failure = next(
        row
        for row in report["failures"]
        if row["name"]
        == "signal manifest binds the explicit ranking, run_id, source hash, and exact TRAIN window"
    )
    assert lineage_error in json.dumps(failure["details"], sort_keys=True)
    assert report["dataset_rebuild_allowed"] is False


def test_preflight_rejects_479_field_manifest_that_swaps_one_mandatory_member(
    tmp_path: Path,
) -> None:
    args = _build_fixture(tmp_path)
    manifest_path = Path(args.signal_manifest)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    selected = list(manifest["selected_features"])
    trend_fields = set(group_features_by_specialist(selected)["trend_ema_encoder"])
    victim = next(
        name
        for name in MODEL_NATIVE_MANDATORY_SELECTED_FIELDS
        if name in trend_fields
    )
    replacement = "trend.ema_adversarial_replacement_fixture"
    before = {
        name: len(fields)
        for name, fields in group_features_by_specialist(selected).items()
    }
    selected[selected.index(victim)] = replacement
    after_groups = group_features_by_specialist(selected)
    after = {name: len(fields) for name, fields in after_groups.items()}
    assert len(selected) == len(set(selected)) == MODEL_NATIVE_SELECTED_FEATURE_COUNT
    assert before == after
    manifest["selected_features"] = selected
    manifest["features_by_specialist"] = after_groups
    _write_json(manifest_path, manifest)

    report = preflight.run(args)

    assert report["decision"] == preflight.BLOCKED_DECISION
    assert report["specialist_contract"]["all_eight_covered"] is True
    assert report["specialist_contract"]["mandatory_full_stack_exact"] is False
    assert "all code-owned full-stack family fields are retained exactly" in {
        row["name"] for row in report["failures"]
    }


def test_preflight_rejects_stale_duplicate_family_count_metadata(
    tmp_path: Path,
) -> None:
    args = _build_fixture(tmp_path)
    manifest_path = Path(args.signal_manifest)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["smart_layer_feature_counts"]["raw_mtf_trend_layer"] -= 1
    _write_json(manifest_path, manifest)

    report = preflight.run(args)

    assert report["decision"] == preflight.BLOCKED_DECISION
    assert "declared mandatory family counts match recomputed selected names" in {
        row["name"] for row in report["failures"]
    }


def test_preflight_rejects_mutable_manifest_and_existing_outputs(tmp_path: Path) -> None:
    mutable = preflight.run(_build_fixture(tmp_path / "mutable", mutable_manifest=True))
    assert mutable["decision"] == preflight.BLOCKED_DECISION
    assert "signal manifest is an explicit timestamped immutable input" in {
        row["name"] for row in mutable["failures"]
    }

    args = _build_fixture(tmp_path / "existing")
    output = Path(args.output)
    output.parent.mkdir(parents=True)
    output.with_name(f"{output.stem}_train.manifest.json").write_text("{}\n")
    Path(args.audit_out_dir).mkdir(parents=True)
    report = preflight.run(args)
    failure_names = {row["name"] for row in report["failures"]}
    assert report["decision"] == preflight.BLOCKED_DECISION
    # V30 retired the rank-reference binding, so no such check may reappear.
    assert not any("rank-reference" in name for name in failure_names)
    assert "dataset output path and derived split artifacts are fresh" in failure_names
    assert "audit output directory is fresh" in failure_names


def test_preflight_blocks_overlapping_split_windows(tmp_path: Path) -> None:
    args = _build_fixture(tmp_path)
    args.val_start = args.train_end

    report = preflight.run(args)

    assert report["decision"] == preflight.BLOCKED_DECISION
    assert "explicit train/val/test split windows are ordered and non-overlapping" in {
        row["name"] for row in report["failures"]
    }
    assert report["rebuild_command_contract"] == {}


def _explicit_cli_args(tmp_path: Path) -> list[str]:
    args = [
        "--run-id",
        RUN_ID,
        "--source-parquet",
        str(tmp_path / "source.parquet"),
        "--canonical-v2-parquet",
        str(tmp_path / "canonical.parquet"),
        "--signal-manifest",
        str(tmp_path / f"MODEL_NATIVE_SIGNAL_MANIFEST_{STAMP}.json"),
        "--feature-ranking-json",
        str(
            tmp_path
            / f"ENTRY_MODEL_NATIVE_TRAIN_FEATURE_RANKING_{STAMP}.json"
        ),
        "--mtf-cache-dir",
        str(tmp_path / "mtf"),
        "--tape-root",
        str(tmp_path / "tape"),
        "--m1-lifecycle-pair-manifest-json",
        str(tmp_path / "pair_generation" / "PAIR_MANIFEST.json"),
        "--m1-lifecycle-pair-generation-root",
        str(tmp_path / "pair_generations"),
        "--m1-feature-base-parquet",
        str(tmp_path / "m1_feature_base.parquet"),
        "--m5-feature-base-parquet",
        str(tmp_path / "m5_feature_base.parquet"),
        "--exit-lifecycle-dir",
        str(tmp_path / "exit_lifecycle"),
        "--output",
        str(tmp_path / "dataset" / DATASET_OUTPUT_NAME),
        "--audit-out-dir",
        str(tmp_path / "audit"),
    ]
    for name, value in SPLITS.items():
        args.extend([f"--{name.replace('_', '-')}", value])
    args.extend(["--out-dir", str(tmp_path / "reports")])
    return args


def test_parser_requires_exact_rebuild_inputs_and_rejects_retired_arguments(
    tmp_path: Path,
) -> None:
    parser = preflight.build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args([])
    for retired in (
        ["--smart-report", str(tmp_path / "report.json")],
        ["--inventory-report", str(tmp_path / "inventory.json")],
        ["--source-dataset-dir", str(tmp_path / "old_dataset")],
        ["--gx1-data-root", str(tmp_path / "GX1_DATA")],
        ["--verify-large-input-hashes"],
        ["--neutral-external_tree_sidecar-bridge"],
        ["--allow-zero-ctx"],
    ):
        with pytest.raises(SystemExit):
            parser.parse_args(_explicit_cli_args(tmp_path) + retired)

    parsed = parser.parse_args(_explicit_cli_args(tmp_path))
    assert parsed.run_id == RUN_ID
    assert parsed.source_parquet == str(tmp_path / "source.parquet")
    assert parsed.feature_ranking_json == str(
        tmp_path / f"ENTRY_MODEL_NATIVE_TRAIN_FEATURE_RANKING_{STAMP}.json"
    )
    assert not hasattr(parsed, "early_move_threshold_bps")


def test_run_lineage_required(tmp_path: Path) -> None:
    args = _build_fixture(tmp_path)
    del args.run_id

    with pytest.raises(Exception, match="provide --run-id"):
        preflight.run(args)


def test_run_rejects_old_namespace_without_explicit_source(tmp_path: Path) -> None:
    old = argparse.Namespace(
        run_id=RUN_ID,
        smart_report=str(tmp_path / "old.json"),
        inventory_report=str(tmp_path / "old_inventory.json"),
        source_dataset_dir=str(tmp_path / "old_dataset"),
        out_dir=str(tmp_path / "reports"),
    )

    with pytest.raises(RuntimeError, match="explicit --source-parquet is required"):
        preflight.run(old)
