"""Genuine synthetic TRAIN-fit registry artifacts shared by tests.

Every fixture executes the canonical competing-risk fitter and binds real
temporary source/tape files. Production validators have no test bypass.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from gx1.contracts.entry_exit_feature_base_v1 import (
    ENTRY_EXIT_ENRICHED_CAUSAL_FRAME_SCHEMA_VERSION,
    EXIT_DECISION_BAR_SECONDS,
    entry_exit_shared_feature_base_contract,
)
from gx1.contracts.registry_hyperparameter_fit_v1 import (
    REGISTRY_OUTCOME_BREAK,
    REGISTRY_OUTCOME_REACTION,
    RegistryOutcomeStreamV1,
    fit_registry_competing_risk_threshold_v1,
)
from gx1.features.htf_features import (
    V29_REGISTRY_CONSTANTS_PROVENANCE_SCHEMA_VERSION,
    V29_REGISTRY_CONSTANTS_SCHEMA_VERSION,
    V29_REGISTRY_M1_LANE_MANIFEST_KEY,
    V29_REGISTRY_M1_LANE_PARAMS_SCHEMA_VERSION,
    V29_REGISTRY_M1_LANE_PROVENANCE_SCHEMA_VERSION,
    require_v29_registry_constants,
    require_v29_registry_m1_lane_params,
)
from gx1.features.level_registry_v1 import fit_level_registry_hyperparameters_v1
from gx1.features.smc_v1 import SWING_LOOKBACK
from gx1.utils.artifact_primitives_v1 import canonical_json_sha256, sha256_file


_WINDOW_START = "2026-01-01T00:00:00+00:00"
_INNER_END = "2026-01-15T00:00:00+00:00"
_WINDOW_END = "2026-01-31T23:55:00+00:00"
_OUTER_ROWS = 700
_INNER_ROW = 400


def _source(root: Path, *, clock: str) -> dict:
    root.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for name in ("source", "tape", "pair"):
        path = (root / f"{name}.json").resolve()
        path.write_text(json.dumps({"name": name}) + "\n", encoding="utf-8")
        paths[name] = path
    return {
        "source_artifact": str(paths["source"]),
        "source_sha256": sha256_file(paths["source"]),
        "source_schema_version": "synthetic_closed_ohlcv_v1",
        "source_lane": clock,
        "tape_manifest_artifact": str(paths["tape"]),
        "tape_manifest_sha256": sha256_file(paths["tape"]),
        "pair_manifest_artifact": str(paths["pair"]),
        "pair_manifest_sha256": sha256_file(paths["pair"]),
        "train_split_id": "synthetic_chronological_train_only",
        "declared_train_window_start": _WINDOW_START,
        "declared_train_window_end": _WINDOW_END,
    }


def _stream() -> RegistryOutcomeStreamV1:
    fit_origins = np.arange(1, 121, dtype=np.int64)
    selection_origins = np.arange(400, 520, dtype=np.int64)
    origins = np.concatenate((fit_origins, selection_origins))
    distances = np.asarray([0.1, 0.2, 0.3, 0.4, 0.5, 0.6] * 40)
    causes = tuple(
        REGISTRY_OUTCOME_REACTION if value <= 0.3 else REGISTRY_OUTCOME_BREAK
        for value in distances
    )
    events = origins + np.asarray([1, 2, 1, 2, 1, 2] * 40, dtype=np.int64)
    return RegistryOutcomeStreamV1(origins, distances, events, causes)


def _fit(
    *,
    source: dict,
    clock: str,
    registry_kind: str,
    population_configuration: dict,
) -> dict:
    index = pd.date_range(_WINDOW_START, periods=_OUTER_ROWS, freq="5min").asi8
    price = 2000.0 + np.arange(_OUTER_ROWS, dtype=np.float64)
    return fit_registry_competing_risk_threshold_v1(
        _stream(),
        registry_kind=registry_kind,
        clock=clock,
        n_rows=_OUTER_ROWS,
        inner_fit_end_exclusive=_INNER_ROW,
        index_ns=index,
        frame_columns=(price, price + 1.0, price - 1.0),
        source_provenance=source,
        population_configuration=population_configuration,
    )


def _level_fit(*, source: dict, clock: str) -> dict:
    n_rows = _OUTER_ROWS
    rng = np.random.default_rng(20260814)
    mid = 2000.0 + np.cumsum(rng.normal(0.0, 1.0, n_rows))
    high = mid + np.abs(rng.normal(0.0, 0.7, n_rows)) + 0.1
    low = mid - np.abs(rng.normal(0.0, 0.7, n_rows)) - 0.1
    close = low + rng.uniform(0.0, 1.0, n_rows) * (high - low)
    frame = pd.DataFrame(
        {
            "high": high,
            "low": low,
            "close": close,
            "atr": 1.0 + np.abs(rng.normal(0.0, 0.2, n_rows)),
        },
        index=pd.date_range(_WINDOW_START, periods=n_rows, freq="5min"),
    )
    return fit_level_registry_hyperparameters_v1(
        frame,
        tf=clock.lower(),
        inner_fit_end_exclusive=_INNER_ROW,
        source_provenance=source,
    )


def _trend_fit(*, source: dict, clock: str, seq_len: int) -> dict:
    # Mirrors fit_trendline_registry_hyperparameters_v1 exactly: the runtime
    # population is the declared receptive field plus the pivot look-around.
    # No identity_expiry_bars is injected — the registry stopped consuming a
    # fitted identity lifetime on 2026-08-15.
    return _fit(
        source=source,
        clock=clock,
        registry_kind="trendline",
        population_configuration={
            "owner": "trendline_exact_runtime_candidate_population_v1",
            "seq_len": int(seq_len),
            "swing_lookback": SWING_LOOKBACK,
        },
    )


def synthetic_v29_registry_constants() -> dict:
    from gx1.contracts.entry_exit_production_architecture_v1 import (
        PRODUCTION_MTF_PER_TF_WINDOW_BARS,
    )
    from gx1.contracts.entry_model_native_signal_v1 import MODEL_NATIVE_SEQ_LEN

    root = Path(tempfile.mkdtemp(prefix="gx1-registry-mtf-fixture-")).resolve()
    seq_lens = dict(PRODUCTION_MTF_PER_TF_WINDOW_BARS)
    level: dict[str, dict] = {}
    trend: dict[str, dict] = {}
    for clock in seq_lens:
        lineage = _source(root / clock, clock=clock)
        level[clock] = _level_fit(source=lineage, clock=clock)
        trend[clock] = _trend_fit(
            source=lineage, clock=clock, seq_len=int(seq_lens[clock])
        )
    entry = _trend_fit(
        source=level["M5"]["source_provenance"],
        clock="M5",
        seq_len=MODEL_NATIVE_SEQ_LEN,
    )
    payload = {
        "schema_version": V29_REGISTRY_CONSTANTS_SCHEMA_VERSION,
        "declared_train_window_start": _WINDOW_START,
        "declared_train_window_end": _WINDOW_END,
        "declared_inner_fit_window_end": _INNER_END,
        "level_recurrence_threshold_atr": {
            clock: value["selected_threshold_atr"] for clock, value in level.items()
        },
        "level_expiry_bars": {
            clock: value["learned_expiry_bars"] for clock, value in level.items()
        },
        "trendline_band_atr": {
            clock: value["selected_threshold_atr"] for clock, value in trend.items()
        },
        "per_tf_seq_lens": seq_lens,
        "entry_m5": {
            "seq_len": MODEL_NATIVE_SEQ_LEN,
            "trendline_band_atr": entry["selected_threshold_atr"],
        },
        "provenance": {
            "schema_version": V29_REGISTRY_CONSTANTS_PROVENANCE_SCHEMA_VERSION,
            "lane": "M5",
            "module": "gx1.features.htf_features",
            "payload_schema_version": V29_REGISTRY_CONSTANTS_SCHEMA_VERSION,
            "fit_owner": "gx1.features.htf_features.fit_v29_registry_constants_from_m5",
            "declared_train_window_start": _WINDOW_START,
            "declared_train_window_end": _WINDOW_END,
            "declared_inner_fit_window_end": _INNER_END,
            "n_train_m5_rows": int(level["M5"]["outer_train_rows"]),
            "inner_fit_end_exclusive_by_clock": {
                clock: int(level[clock]["inner_fit_end_exclusive"])
                for clock in seq_lens
            },
            "level_recurrence_threshold": level,
            "trendline_band": trend,
            "entry_m5_trendline_band": entry,
        },
    }
    payload["contract_sha256"] = canonical_json_sha256(payload)
    return require_v29_registry_constants(payload)


def synthetic_v29_registry_m1_lane_params() -> dict:
    from gx1.contracts.entry_exit_feature_base_v1 import EXIT_FEATURE_SEQUENCE_BARS

    root = Path(tempfile.mkdtemp(prefix="gx1-registry-m1-fixture-"))
    lineage = _source(root, clock="M1")
    level = _level_fit(source=lineage, clock="M1")
    trend = _trend_fit(
        source=lineage, clock="M1", seq_len=EXIT_FEATURE_SEQUENCE_BARS
    )
    payload = {
        "schema_version": V29_REGISTRY_M1_LANE_PARAMS_SCHEMA_VERSION,
        "declared_train_window_start": _WINDOW_START,
        "declared_train_window_end": _WINDOW_END,
        "declared_inner_fit_window_end": _INNER_END,
        "level_recurrence_threshold_atr": level["selected_threshold_atr"],
        "level_expiry_bars": level["learned_expiry_bars"],
        "exit_m1": {
            "seq_len": EXIT_FEATURE_SEQUENCE_BARS,
            "trendline_band_atr": trend["selected_threshold_atr"],
        },
        "provenance": {
            "schema_version": V29_REGISTRY_M1_LANE_PROVENANCE_SCHEMA_VERSION,
            "lane": "M1",
            "module": "gx1.features.htf_features",
            "payload_schema_version": V29_REGISTRY_M1_LANE_PARAMS_SCHEMA_VERSION,
            "fit_owner": "gx1.features.htf_features.fit_v29_registry_m1_lane_params_from_m1",
            "declared_train_window_start": _WINDOW_START,
            "declared_train_window_end": _WINDOW_END,
            "declared_inner_fit_window_end": _INNER_END,
            "n_train_m1_rows": int(level["outer_train_rows"]),
            "inner_fit_end_exclusive": int(level["inner_fit_end_exclusive"]),
            "level_recurrence_threshold": level,
            "trendline_band": trend,
        },
    }
    payload["contract_sha256"] = canonical_json_sha256(payload)
    return require_v29_registry_m1_lane_params(payload)


def write_synthetic_m1_registry_manifest(
    tmp_path: Path,
    *,
    params: dict,
    stem: str = "m1_enriched",
) -> Path:
    output = (Path(tmp_path) / f"{stem}.parquet").resolve()
    output.write_bytes(b"synthetic-m1-enriched-registry-container")
    manifest = Path(f"{output}.manifest.json")
    payload = {
        "schema_version": ENTRY_EXIT_ENRICHED_CAUSAL_FRAME_SCHEMA_VERSION,
        "decision": "PASS",
        "shared_feature_base_contract": entry_exit_shared_feature_base_contract(),
        "timeframe": "M1",
        "base_bar_seconds": EXIT_DECISION_BAR_SECONDS,
        "output_parquet": str(output),
        "output_parquet_sha256": sha256_file(output),
        V29_REGISTRY_M1_LANE_MANIFEST_KEY: params,
    }
    payload["manifest_sha256"] = canonical_json_sha256(payload)
    manifest.write_text(
        json.dumps(payload, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return manifest
