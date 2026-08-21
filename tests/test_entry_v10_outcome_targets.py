from __future__ import annotations

import json
from unittest import mock
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from gx1.contracts.xau_tape_provenance_v1 import (
    CANONICAL_NATIVE_CLOSURE_CONTRACT,
    canonical_xau_source_descriptor_v1,
)
from gx1.execution import v12_canonical_incremental as incremental
from gx1.scripts import backfill_xauusd_m5_from_oanda as native_publisher
from gx1.contracts.entry_model_native_aux_targets_v3 import (
    MODEL_NATIVE_AUX_RISK_HORIZONS,
    MODEL_NATIVE_AUX_TARGET_SCHEMA_VERSION,
    MODEL_NATIVE_DIP_MAE_TARGET_COLUMNS,
    MODEL_NATIVE_DIP_MFE_TARGET_COLUMNS,
    MODEL_NATIVE_DIP_MFE_UPPER_SAFETY_CAP_BPS,
    require_model_native_aux_target_contract,
    require_model_native_aux_target_emission_contract,
)
from gx1.contracts.unified_exit_lifecycle_v1 import (
    UNIFIED_EXIT_LIFECYCLE_EPISODE_SCHEMA_VERSION,
    UNIFIED_EXIT_LIFECYCLE_REQUIRED_M1_COLUMNS,
    UNIFIED_EXIT_STATE_SELECTION_SCHEMA_VERSION,
    UnifiedExitLifecycleCorpus,
    UnifiedExitLifecycleSplit,
    canonical_json_sha256,
    require_unified_exit_m1_pair_authority,
    sha256_file,
    unified_exit_state_population_arrays,
)
from gx1.contracts.entry_exit_feature_base_v1 import (
    ENTRY_EXIT_ENRICHED_CAUSAL_FRAME_SCHEMA_VERSION,
    entry_exit_shared_feature_base_contract,
    require_entry_exit_enriched_source_binding,
)
from gx1.contracts.entry_exit_feature_surface_v1 import (
    ENTRY_EXIT_FEATURE_SURFACE_COLUMNS,
    build_entry_exit_feature_surface_manifest,
    require_exact_m1_feature_surface_manifest,
)
from gx1.contracts.entry_model_native_signal_v1 import (
    MODEL_NATIVE_BASE_FIELDS,
    MODEL_NATIVE_CONTRACT_MODE,
    MODEL_NATIVE_CTX_CAT_DIM,
    MODEL_NATIVE_CTX_CONT_DIM,
    MODEL_NATIVE_SIGNAL_DIM,
    model_native_mandatory_full_stack_metadata,
    model_native_signal_contract_metadata,
)
from gx1.scripts.build_entry_v10_ctx_training_dataset_v3 import (
    MODEL_NATIVE_AUX_MAX_FUTURE_HORIZON_BARS,
    MODEL_NATIVE_AUX_TARGET_COLUMNS,
    MODEL_NATIVE_AUX_TARGET_HORIZON_BY_COLUMN,
    _build_model_native_aux_head_targets,
    _position_size_target_from_path,
    _selected_side_bad_path_target,
    _validate_model_native_aux_head_targets,
    build_unified_exit_lifecycle_episodes,
    representation_auxiliary_outcome_contract,
    model_native_aux_target_contract_metadata,
)
from gx1.features.htf_features import V29_REGISTRY_M1_LANE_MANIFEST_KEY
from tests.test_oanda_backfill_vedtak_gate import _FakeOandaClient
from tests.model_native_signal_support import canonical_model_native_selected_fields
from tests.htf_v29_registry_test_support import (
    synthetic_v29_registry_m1_lane_params,
)
from tests.entry_position_size_target_policy_support import (
    entry_position_size_target_policy_fixture,
)
from tests.volatility_squeeze_test_support import (
    make_volatility_squeeze_artifact_set,
)


BUILDER_PATH = Path("gx1/scripts/build_entry_v10_ctx_training_dataset_v3.py")


def _spread_tape(n_rows: int = 130) -> pd.DataFrame:
    close = 2000.0 + np.arange(n_rows, dtype=np.float64) * 0.10
    high = close + 0.20
    low = close - 0.20
    return pd.DataFrame(
        {
            "close": close,
            "high": high,
            "low": low,
            "bid_close": close - 0.05,
            "ask_close": close + 0.05,
            "bid_high": high - 0.05,
            "bid_low": low - 0.05,
            "ask_high": high + 0.05,
            "ask_low": low + 0.05,
        }
    )


def _monotonic_unfavorable_spread_tape(
    side: str,
    n_rows: int = 130,
) -> pd.DataFrame:
    if side not in {"long", "short"}:
        raise ValueError(f"unsupported side: {side}")
    step = -0.10 if side == "long" else 0.10
    close = 2000.0 + np.arange(n_rows, dtype=np.float64) * step
    high = close + 0.01
    low = close - 0.01
    return pd.DataFrame(
        {
            "close": close,
            "high": high,
            "low": low,
            "bid_close": close - 0.05,
            "ask_close": close + 0.05,
            "bid_high": high - 0.05,
            "bid_low": low - 0.05,
            "ask_high": high + 0.05,
            "ask_low": low + 0.05,
        }
    )


def test_aux_targets_have_exact_horizons_and_no_fake_tail_values() -> None:
    frame = _spread_tape()
    targets, complete = _build_model_native_aux_head_targets(frame)

    assert tuple(targets) == MODEL_NATIVE_AUX_TARGET_COLUMNS
    assert MODEL_NATIVE_AUX_MAX_FUTURE_HORIZON_BARS == 96
    assert complete.tolist() == ([True] * (len(frame) - 96) + [False] * 96)
    for name, horizon in MODEL_NATIVE_AUX_TARGET_HORIZON_BY_COLUMN.items():
        values = targets[name]
        assert np.isfinite(values[: len(frame) - horizon]).all()
        assert np.isnan(values[len(frame) - horizon :]).all()
        assert not np.isinf(values).any()


def test_aux_target_contract_is_exact_and_spread_aware() -> None:
    contract = model_native_aux_target_contract_metadata()

    assert contract["schema_version"] == MODEL_NATIVE_AUX_TARGET_SCHEMA_VERSION
    assert contract["columns"] == list(MODEL_NATIVE_AUX_TARGET_COLUMNS)
    assert contract["max_future_horizon_bars"] == 96
    assert contract["spread_aware_risk_magnitudes_required"] is True
    domains = contract["target_value_domains"]
    assert domains["dip_mfe"] == {
        "columns": list(MODEL_NATIVE_DIP_MFE_TARGET_COLUMNS),
        "unit": "bps",
        "finite_on_complete_rows": True,
        "signed": True,
        "negative_values_preserved": True,
        "lower_bound_bps": None,
        "upper_safety_cap_bps": MODEL_NATIVE_DIP_MFE_UPPER_SAFETY_CAP_BPS,
    }
    assert domains["dip_mae"]["columns"] == list(
        MODEL_NATIVE_DIP_MAE_TARGET_COLUMNS
    )
    assert domains["dip_mae"]["signed"] is False
    assert domains["dip_mae"]["lower_bound_bps"] == 0.0
    assert contract["mid_price_timing_reference_only"] is True
    assert contract["incomplete_rows_may_be_emitted"] is False
    # The offline-RL counterfactual action-value block is retired; the Entry
    # action value is the frozen fitted-Q teacher (gx1/contracts/entry_fitted_q_v1.py,
    # covered by tests/test_entry_fitted_q_v1.py).
    assert contract["offline_rl"] == "retired_replaced_by_entry_fitted_q"
    assert not [
        name for name in contract["columns"] if name.startswith("y_action_value_")
    ]
    timing = contract["turning_point_timing"]
    assert timing["output_dim"] == 12
    assert timing["layout"][0]["market_turn"] == "BOTTOM"
    assert timing["layout"][6]["market_turn"] == "TOP"
    assert timing["live_direction_rule_authority"] is False


def _emission_contract(*, candidates: int = 100) -> dict:
    return {
        **model_native_aux_target_contract_metadata(),
        "incomplete_tail_rows_total": 96,
        "candidate_rows_before_completeness": candidates,
        "incomplete_candidate_rows_excluded": 96,
        "complete_rows_emitted": candidates - 96,
    }


def test_aux_target_emission_contract_requires_and_normalizes_exact_row_proof() -> None:
    expected = model_native_aux_target_contract_metadata()

    assert require_model_native_aux_target_emission_contract(
        _emission_contract(),
        context="TEST",
    ) == expected
    with pytest.raises(RuntimeError, match="AUX_TARGET_CONTRACT_INVALID"):
        require_model_native_aux_target_contract(
            _emission_contract(),
            context="TEST_STATIC",
        )


@pytest.mark.parametrize(
    "mutation",
    ("missing", "extra", "bool", "tail", "excluded", "equation", "empty"),
)
def test_aux_target_emission_contract_rejects_non_exact_row_proof(
    mutation: str,
) -> None:
    candidate = _emission_contract()
    if mutation == "missing":
        candidate.pop("complete_rows_emitted")
    elif mutation == "extra":
        candidate["allow_incomplete"] = False
    elif mutation == "bool":
        candidate["complete_rows_emitted"] = True
    elif mutation == "tail":
        candidate["incomplete_tail_rows_total"] = 95
    elif mutation == "excluded":
        candidate["incomplete_candidate_rows_excluded"] = 95
    elif mutation == "equation":
        candidate["candidate_rows_before_completeness"] = 101
    else:
        candidate["complete_rows_emitted"] = 0
        candidate["candidate_rows_before_completeness"] = 96

    with pytest.raises(RuntimeError, match="AUX_TARGET_EMISSION_CONTRACT_INVALID"):
        require_model_native_aux_target_emission_contract(
            candidate,
            context="TEST",
        )


def test_model_native_group_a_recompute_is_memory_capped_and_explicit() -> None:
    source = BUILDER_PATH.read_text(encoding="utf-8")

    assert "_MODEL_NATIVE_GROUP_A_RECOMPUTE_WORKERS = 1" in source
    assert "workers=_MODEL_NATIVE_GROUP_A_RECOMPUTE_WORKERS" in source
    assert '"group_a_recompute_workers": _MODEL_NATIVE_GROUP_A_RECOMPUTE_WORKERS' in source


def test_aux_risk_magnitude_uses_executable_spread_path() -> None:
    frame = _spread_tape()
    targets, _ = _build_model_native_aux_head_targets(frame)

    expected_long_mfe = (
        (frame.loc[12, "bid_high"] - frame.loc[0, "ask_close"])
        / frame.loc[0, "ask_close"]
        * 1e4
    )
    expected_short_mfe = (
        (frame.loc[0, "bid_close"] - frame.loc[1, "ask_low"])
        / frame.loc[0, "bid_close"]
        * 1e4
    )
    assert targets["y_dip_mfe_long_K12"][0] == pytest.approx(expected_long_mfe)
    assert targets["y_dip_mfe_short_K12"][0] == pytest.approx(expected_short_mfe)


@pytest.mark.parametrize("side", ("long", "short"))
def test_signed_dip_mfe_preserves_negative_spread_excursion_on_unfavorable_tape(
    side: str,
) -> None:
    targets, _ = _build_model_native_aux_head_targets(
        _monotonic_unfavorable_spread_tape(side)
    )

    for horizon in MODEL_NATIVE_AUX_RISK_HORIZONS:
        mfe = targets[f"y_dip_mfe_{side}_K{horizon}"]
        mae = targets[f"y_dip_mae_{side}_K{horizon}"]
        complete_rows = len(mfe) - horizon
        assert np.all(mfe[:complete_rows] < 0.0)
        assert np.all(mae[:complete_rows] >= 0.0)


def test_aux_target_validator_rejects_negative_dip_mae_but_accepts_negative_mfe() -> None:
    targets, _ = _build_model_native_aux_head_targets(
        _monotonic_unfavorable_spread_tape("long")
    )
    assert targets["y_dip_mfe_long_K12"][0] < 0.0
    _validate_model_native_aux_head_targets(targets, n_rows=130)

    broken = {name: values.copy() for name, values in targets.items()}
    broken["y_dip_mae_long_K12"][0] = -1.0
    with pytest.raises(RuntimeError, match="AUX_TARGET_DOMAIN_INVALID"):
        _validate_model_native_aux_head_targets(broken, n_rows=130)


def test_aux_target_validator_rejects_finite_incomplete_tail() -> None:
    targets, _ = _build_model_native_aux_head_targets(_spread_tape())
    broken = {name: values.copy() for name, values in targets.items()}
    broken["y_forecast_ret_K1"][-1] = 0.0

    with pytest.raises(RuntimeError, match="AUX_TARGET_COMPLETENESS_INVALID"):
        _validate_model_native_aux_head_targets(broken, n_rows=130)


def _closed_m1_lifecycle_source(n_rows: int = 640) -> pd.DataFrame:
    time = pd.date_range(
        "2026-01-01T00:05:00Z",
        periods=n_rows,
        freq="min",
    )
    phase = np.arange(n_rows, dtype=np.float64)
    close = 2000.0 + np.sin(phase / 7.0) * 2.0 + phase * 0.001
    open_values = close - np.cos(phase / 11.0) * 0.05
    high = np.maximum(open_values, close) + 0.10
    low = np.minimum(open_values, close) - 0.10
    spread = 0.10
    return pd.DataFrame(
        {
            "time": time,
            "open": open_values,
            "high": high,
            "low": low,
            "close": close,
            "bid_open": open_values - spread / 2.0,
            "bid_high": high - spread / 2.0,
            "bid_low": low - spread / 2.0,
            "bid_close": close - spread / 2.0,
            "ask_open": open_values + spread / 2.0,
            "ask_high": high + spread / 2.0,
            "ask_low": low + spread / 2.0,
            "ask_close": close + spread / 2.0,
            "volume": np.arange(n_rows, dtype=np.int64) + 1,
        }
    )


def _in_memory_full_exit_split(
    *,
    source: pd.DataFrame | None = None,
    episodes: pd.DataFrame | None = None,
    proof: dict[str, object] | None = None,
    entry_row_count: int = 1,
) -> tuple[
    UnifiedExitLifecycleSplit,
    pd.DataFrame,
    dict[str, object],
    pd.DataFrame,
]:
    """Build a file-free full-population split for lifecycle unit tests."""

    frame = (
        _closed_m1_lifecycle_source(n_rows=1800)
        if source is None
        else source.copy()
    )
    if episodes is None or proof is None:
        entry_time = pd.Timestamp(frame["time"].iloc[600]) - pd.Timedelta(
            minutes=5
        )
        episodes, proof = build_unified_exit_lifecycle_episodes(
            min_m1_start_row=0,
            entry_rows=pd.DataFrame({"time": [entry_time]}),
            closed_m1=frame,
            split_end=pd.Timestamp(frame["time"].iloc[-1])
            + pd.Timedelta(minutes=1),
            market_closure_contract=CANONICAL_NATIVE_CLOSURE_CONTRACT,
        )
    times = pd.DatetimeIndex(
        pd.to_datetime(frame["time"], utc=True, errors="raise")
    ).as_unit("ns")
    m1_arrays = {
        name: pd.to_numeric(frame[name], errors="raise").to_numpy(
            dtype=np.float64
        )
        for name in UNIFIED_EXIT_LIFECYCLE_REQUIRED_M1_COLUMNS
        if name != "time"
    }
    row_value = np.arange(len(frame), dtype=np.float32)
    signal = np.zeros(
        (len(frame), MODEL_NATIVE_SIGNAL_DIM),
        dtype=np.float32,
    )
    signal[:, 0] = row_value
    ctx_cont = np.zeros(
        (len(frame), MODEL_NATIVE_CTX_CONT_DIM),
        dtype=np.float32,
    )
    ctx_cont[:, 0] = row_value
    ctx_cat = np.zeros(
        (len(frame), MODEL_NATIVE_CTX_CAT_DIM),
        dtype=np.int64,
    )
    split = UnifiedExitLifecycleSplit(
        split="val",
        entry_row_count=entry_row_count,
        feature_row_offset=0,
        episodes=episodes,
        split_manifest=proof,
        m1_times=times,
        m1_arrays=m1_arrays,
        m1_feature_times=times,
        m1_feature_arrays={
            "signal": signal,
            "ctx_cont": ctx_cont,
            "ctx_cat": ctx_cat,
        },
    )
    return split, episodes, proof, frame


class _LifecycleTrendOandaClient(_FakeOandaClient):
    """Timestamp-derived prices keep independently fetched M1/M5 fixtures aligned."""

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, object],
    ) -> dict[str, object]:
        response = super()._request(method, path, params=params)
        epoch = pd.Timestamp("2026-01-01T00:00:00Z")
        for candle in response["candles"]:
            timestamp = pd.Timestamp(candle["time"])
            bucket = int((timestamp - epoch) / pd.Timedelta(minutes=5))
            mid = 2000.0 + bucket * 0.01
            candle["mid"] = {
                "o": str(mid),
                "h": str(mid + 1.0),
                "l": str(mid - 1.0),
                "c": str(mid),
            }
            candle["bid"] = {
                "o": str(mid - 0.5),
                "h": str(mid + 0.5),
                "l": str(mid - 1.5),
                "c": str(mid - 0.5),
            }
            candle["ask"] = {
                "o": str(mid + 0.5),
                "h": str(mid + 1.5),
                "l": str(mid - 0.5),
                "c": str(mid + 0.5),
            }
        return response


def _strict_native_pair_fixture(
    tmp_path: Path,
    *,
    successor_end: str = "2026-01-01T09:25:00Z",
    trend_prices: bool = False,
) -> tuple[Path, Path, Path]:
    native_m1_parent_root = tmp_path / "native-m1-parent"
    native_m5_parent_root = tmp_path / "native-m5-parent"
    native_m1_root = tmp_path / "native-m1"
    native_m5_root = tmp_path / "native-m5"
    vedtak = "XAU_NATIVE_PAIR_LIFECYCLE_FIXTURE_V1"
    parent_end = "2026-01-01T09:20:00Z"
    client_type = _LifecycleTrendOandaClient if trend_prices else _FakeOandaClient
    with mock.patch.object(
        native_publisher,
        "_require_clean_repository",
        return_value="a" * 40,
    ):
        native_publisher.materialize_native_xau_snapshot(
            client=client_type(timeframe="M1"),
            timeframe="M1",
            vedtak_id=vedtak,
            start_utc="2026-01-01T00:00:00Z",
            end_utc=parent_end,
            out_root=native_m1_parent_root,
        )
        native_publisher.materialize_native_xau_snapshot(
            client=client_type(timeframe="M5"),
            timeframe="M5",
            vedtak_id=vedtak,
            start_utc="2026-01-01T00:00:00Z",
            end_utc=parent_end,
            out_root=native_m5_parent_root,
        )
        m1_parent = canonical_xau_source_descriptor_v1(
            native_m1_parent_root,
            timeframe="M1",
        )
        m5_parent = canonical_xau_source_descriptor_v1(
            native_m5_parent_root,
            timeframe="M5",
        )
        native_publisher.materialize_native_xau_successor(
            client=client_type(timeframe="M1"),
            timeframe="M1",
            vedtak_id=vedtak,
            end_utc=successor_end,
            out_root=native_m1_root,
            parent_root=native_m1_parent_root,
            expected_parent_manifest_sha256=m1_parent["manifest_sha256"],
        )
        native_publisher.materialize_native_xau_successor(
            client=client_type(timeframe="M5"),
            timeframe="M5",
            vedtak_id=vedtak,
            end_utc=successor_end,
            out_root=native_m5_root,
            parent_root=native_m5_parent_root,
            expected_parent_manifest_sha256=m5_parent["manifest_sha256"],
        )
    m1_descriptor = canonical_xau_source_descriptor_v1(
        native_m1_root,
        timeframe="M1",
    )
    m5_descriptor = canonical_xau_source_descriptor_v1(
        native_m5_root,
        timeframe="M5",
    )
    native_m1 = pd.read_parquet(
        native_m1_root / "year=2026" / "part-000.parquet"
    )
    native_m5 = pd.read_parquet(
        native_m5_root / "year=2026" / "part-000.parquet"
    )
    base28 = native_m1.set_index("time").loc[
        :, list(incremental.RAW_BASE28_COLUMNS)
    ]
    canonical = pd.DataFrame(
        {
            "open": native_m5["open"].to_numpy(dtype=np.float64),
            "fixture_signal": np.linspace(
                0.0,
                1.0,
                len(native_m5),
                dtype=np.float64,
            ),
        },
        index=pd.DatetimeIndex(native_m5["time"], name="time"),
    )
    source_inventory = [{"path": "fixture.py", "sha256": "b" * 64}]
    lineage = incremental._build_pair_lineage(
        vedtak=vedtak,
        commit="c" * 40,
        source_inventory=source_inventory,
        m1_descriptor=m1_descriptor,
        m5_descriptor=m5_descriptor,
        native_m1=native_m1,
        native_m5=native_m5,
        canonical=canonical,
        base28=base28,
        parent_pair_generation_id=None,
        parent_pair_manifest_sha256=None,
    )
    generation_root = tmp_path / "pair-generations"
    pointer = tmp_path / "CURRENT_PAIR_MANIFEST.json"
    stage = incremental._candidate_staging_path(generation_root)
    incremental._write_candidate_parquet(
        canonical.reset_index(),
        stage / incremental.PAIR_CANONICAL_FILENAME,
        index=False,
    )
    incremental._write_candidate_parquet(
        base28.reset_index(),
        stage / incremental.PAIR_BASE28_FILENAME,
        index=False,
    )
    generation_id = incremental._publish_prebuilt_pair_generation(
        stage,
        pair_manifest_path=pointer,
        generation_root=generation_root,
        expected_pair_generation_id=None,
        expected_manifest_sha256=None,
        lineage_contract=lineage,
        created_utc="2026-01-01T10:00:00Z",
    )
    generation_manifest = (
        generation_root
        / generation_id
        / incremental.PREBUILT_PAIR_GENERATION_MANIFEST_FILENAME
    )
    return generation_manifest, generation_root, pointer


def _write_exact_m1_feature_surface_fixture(
    tmp_path: Path,
    *,
    m1_source: Path,
    m1_frame: pd.DataFrame,
    dataset_run_id: str,
    pair_generation_id: str,
) -> tuple[Path, Path]:
    """Publish one small, fully bound M1 feature surface without bypasses."""

    enriched_source = (tmp_path / "xau_m1_enriched.parquet").resolve()
    m1_frame.to_parquet(enriched_source, index=False)
    rank_reference = (tmp_path / "train_rank_reference.npz").resolve()
    rank_reference.write_bytes(b"exact-m1-surface-rank-reference-v1")
    registry_params = synthetic_v29_registry_m1_lane_params()
    enriched_manifest = Path(f"{enriched_source}.manifest.json")
    enriched_payload = {
        "schema_version": ENTRY_EXIT_ENRICHED_CAUSAL_FRAME_SCHEMA_VERSION,
        "decision": "PASS",
        "shared_feature_base_contract": entry_exit_shared_feature_base_contract(),
        "dataset_run_id": dataset_run_id,
        "pair_generation_id": pair_generation_id,
        "timeframe": "M1",
        "base_bar_seconds": 60,
        "rank_reference_npz": str(rank_reference),
        "rank_reference_sha256": sha256_file(rank_reference),
        "output_parquet": str(enriched_source),
        "output_parquet_sha256": sha256_file(enriched_source),
        V29_REGISTRY_M1_LANE_MANIFEST_KEY: registry_params,
    }
    enriched_payload["manifest_sha256"] = canonical_json_sha256(
        enriched_payload
    )
    enriched_manifest.write_text(
        json.dumps(enriched_payload, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    source_binding = require_entry_exit_enriched_source_binding(
        enriched_source,
        dataset_run_id=dataset_run_id,
        pair_generation_id=pair_generation_id,
        timeframe="M1",
        context="TEST_UNIFIED_EXIT_M1_ENRICHED_SOURCE",
    )

    selected = canonical_model_native_selected_fields(
        remainder_prefix="session_regime.unified_exit_surface_fixture"
    )
    signal_contract = model_native_signal_contract_metadata(selected)
    signal_manifest = (tmp_path / "seq513_signal_manifest.json").resolve()
    signal_manifest.write_text(
        json.dumps(
            {
                "manifest_variant": MODEL_NATIVE_CONTRACT_MODE,
                "base_signal_feature_count": len(MODEL_NATIVE_BASE_FIELDS),
                "expected_seq_snap_width": MODEL_NATIVE_SIGNAL_DIM,
                "selected_features": selected,
                "mandatory_full_stack": (
                    model_native_mandatory_full_stack_metadata()
                ),
                "model_native_signal_contract": signal_contract,
            },
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )

    rows = len(m1_frame)
    surface = (tmp_path / "xau_m1_feature_base.parquet").resolve()
    table = pa.Table.from_arrays(
        [
            pa.Array.from_pandas(
                pd.to_datetime(m1_frame["time"], utc=True, errors="raise")
            ),
            pa.FixedSizeListArray.from_arrays(
                pa.array(
                    np.zeros(rows * MODEL_NATIVE_SIGNAL_DIM, dtype=np.float32),
                    type=pa.float32(),
                ),
                MODEL_NATIVE_SIGNAL_DIM,
            ),
            pa.FixedSizeListArray.from_arrays(
                pa.array(
                    np.zeros(rows * MODEL_NATIVE_CTX_CONT_DIM, dtype=np.float32),
                    type=pa.float32(),
                ),
                MODEL_NATIVE_CTX_CONT_DIM,
            ),
            pa.FixedSizeListArray.from_arrays(
                pa.array(
                    np.zeros(rows * MODEL_NATIVE_CTX_CAT_DIM, dtype=np.int64),
                    type=pa.int64(),
                ),
                MODEL_NATIVE_CTX_CAT_DIM,
            ),
        ],
        names=list(ENTRY_EXIT_FEATURE_SURFACE_COLUMNS),
    )
    pq.write_table(
        table,
        surface,
        compression="snappy",
        use_dictionary=False,
        row_group_size=480,
    )
    surface_manifest = Path(f"{surface}.manifest.json")
    registry_artifact = enriched_manifest
    squeeze_artifacts = make_volatility_squeeze_artifact_set(tmp_path)
    manifest = build_entry_exit_feature_surface_manifest(
        timeframe="M1",
        dataset_run_id=dataset_run_id,
        pair_generation_id=pair_generation_id,
        source=enriched_source,
        source_binding=source_binding,
        alignment=m1_source,
        seq_structure_manifest=signal_manifest,
        output=surface,
        rows=rows,
        signal_contract=signal_contract,
        extension={
            "mode": "exact_test_fixture_v1",
            "ordered_fields_sha256": signal_contract["ordered_fields_sha256"],
        },
        registry_fit_binding={
            "lane": "M1",
            "artifact_path": str(registry_artifact),
            "artifact_sha256": sha256_file(registry_artifact),
            "params_schema_version": registry_params["schema_version"],
            "params_module": registry_params["provenance"]["module"],
            "params_contract_sha256": registry_params["contract_sha256"],
        },
        volatility_squeeze_artifact_binding=squeeze_artifacts.binding(),
        materialization={
            "mode": "bounded_native_m1_owner_batches_v2_event_age_carry",
            "batch_rows": rows,
            "causal_overlap_rows": 0,
            "recursive_state_fields": [],
        },
    )
    surface_manifest.write_text(
        json.dumps(manifest, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return surface, surface_manifest


def test_exact_m1_feature_surface_rejects_resealed_row_tamper(
    tmp_path: Path,
) -> None:
    m1_frame = _closed_m1_lifecycle_source(n_rows=16)
    m1_source = (tmp_path / "authoritative_m1.parquet").resolve()
    m1_frame.to_parquet(m1_source, index=False)
    pair_generation_id = "b" * 64
    surface, manifest_path = _write_exact_m1_feature_surface_fixture(
        tmp_path,
        m1_source=m1_source,
        m1_frame=m1_frame,
        dataset_run_id="EXACT_M1_SURFACE_TAMPER_PYTEST_V1",
        pair_generation_id=pair_generation_id,
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["rows"] = len(m1_frame) - 1
    manifest.pop("manifest_sha256")
    manifest["manifest_sha256"] = canonical_json_sha256(manifest)
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="TEST_M1_SURFACE_IDENTITY_INVALID"):
        require_exact_m1_feature_surface_manifest(
            manifest_path=manifest_path,
            expected_manifest_sha256=sha256_file(manifest_path),
            expected_parquet_path=surface,
            expected_parquet_sha256=sha256_file(surface),
            expected_dataset_run_id="EXACT_M1_SURFACE_TAMPER_PYTEST_V1",
            expected_pair_generation_id=pair_generation_id,
            expected_rows=len(m1_frame),
            expected_m1_source_path=m1_source,
            expected_m1_source_sha256=sha256_file(m1_source),
            context="TEST_M1_SURFACE",
        )


def test_exact_m1_feature_surface_rejects_registry_source_params_mismatch(
    tmp_path: Path,
) -> None:
    m1_frame = _closed_m1_lifecycle_source(n_rows=16)
    m1_source = (tmp_path / "authoritative_m1.parquet").resolve()
    m1_frame.to_parquet(m1_source, index=False)
    dataset_run_id = "EXACT_M1_REGISTRY_BINDING_PYTEST_V1"
    pair_generation_id = "d" * 64
    surface, manifest_path = _write_exact_m1_feature_surface_fixture(
        tmp_path,
        m1_source=m1_source,
        m1_frame=m1_frame,
        dataset_run_id=dataset_run_id,
        pair_generation_id=pair_generation_id,
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["registry_fit_binding"]["params_contract_sha256"] = "0" * 64
    manifest.pop("manifest_sha256")
    manifest["manifest_sha256"] = canonical_json_sha256(manifest)
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        RuntimeError,
        match="ENTRY_EXIT_FEATURE_SURFACE_REGISTRY_BINDING_INVALID",
    ):
        require_exact_m1_feature_surface_manifest(
            manifest_path=manifest_path,
            expected_manifest_sha256=sha256_file(manifest_path),
            expected_parquet_path=surface,
            expected_parquet_sha256=sha256_file(surface),
            expected_dataset_run_id=dataset_run_id,
            expected_pair_generation_id=pair_generation_id,
            expected_rows=len(m1_frame),
            expected_m1_source_path=m1_source,
            expected_m1_source_sha256=sha256_file(m1_source),
            context="TEST_M1_REGISTRY_BINDING",
        )


def test_unified_exit_m1_authority_revalidates_native_complete_source(
    tmp_path: Path,
) -> None:
    generation_manifest, generation_root, pointer = (
        _strict_native_pair_fixture(tmp_path)
    )

    source_path, authority = require_unified_exit_m1_pair_authority(
        pair_manifest_path=generation_manifest,
        pair_generation_root=generation_root,
    )

    assert source_path.name == incremental.PAIR_BASE28_FILENAME
    assert authority["native_m1_completion_field"] == "complete"
    assert authority["native_m1_completion_value"] is True
    assert authority["base28_native_m1_subset_proof"]["rows"] == 565
    native_manifest = json.loads(
        Path(authority["native_m1_manifest_path"]).read_text(encoding="utf-8")
    )
    assert native_manifest["schema_version"] == (
        incremental.CANONICAL_NATIVE_SUCCESSOR_SOURCE_SCHEMA
    )
    with pytest.raises(
        RuntimeError,
        match="MUTABLE_PAIR_POINTER_FORBIDDEN",
    ):
        require_unified_exit_m1_pair_authority(
            pair_manifest_path=pointer,
            pair_generation_root=generation_root,
        )


def test_unified_exit_lifecycle_envelope_binds_both_sides_and_target_stream() -> None:
    entries = pd.DataFrame(
        {
            "time": pd.to_datetime(
                [
                    "2026-01-01T00:00:00Z",
                    "2026-01-01T00:05:00Z",
                ],
                utc=True,
            )
        }
    )
    source = _closed_m1_lifecycle_source()
    split_end = source["time"].iloc[-1] + pd.Timedelta(minutes=1)
    episodes, proof = build_unified_exit_lifecycle_episodes(
        min_m1_start_row=0,
        entry_rows=entries,
        closed_m1=source,
        split_end=split_end,
        market_closure_contract=CANONICAL_NATIVE_CLOSURE_CONTRACT,
    )
    repeated, repeated_proof = build_unified_exit_lifecycle_episodes(
        min_m1_start_row=0,
        entry_rows=entries,
        closed_m1=source,
        split_end=split_end,
        market_closure_contract=CANONICAL_NATIVE_CLOSURE_CONTRACT,
    )

    pd.testing.assert_frame_equal(episodes, repeated)
    assert len(episodes) == 4
    assert episodes.groupby("entry_row_index")["side"].apply(list).tolist() == [
        ["long", "short"],
        ["long", "short"],
    ]
    assert (episodes["entry_available_at"] == episodes["m1_start_time"]).all()
    assert (
        episodes["first_state_decision_time"]
        == episodes["entry_available_at"] + pd.Timedelta(minutes=1)
    ).all()
    assert (episodes["path_state_count"] == 512).all()
    assert proof["state_population_rows"] == len(episodes) * 512
    assert proof["state_population_per_episode"] == 512
    assert proof["state_population"] == (
        "all_authoritative_states_both_sides_every_complete_episode"
    )
    assert proof["exit_supervision_authority"] == (
        "executable_exit_now_reward_plus_train_fitted_q"
    )
    assert proof["extra_lookahead_beyond_trajectory"] == 0
    assert proof["decision"] == "PASS"
    assert proof["entry_side_selection"] == (
        "both_sides_for_every_causal_entry_snapshot"
    )
    assert proof["path_values_duplicated_into_episode_artifact"] is False
    assert proof["state_vectors_duplicated_into_episode_artifact"] is False
    assert not {
        "state_indices",
        "decision_row_indices",
        "state_row_time_ns",
        "decision_time_ns",
        "state_valid_mask",
    }.intersection(episodes.columns)
    assert len(proof["state_population_stream_sha256"]) == 64
    assert (
        proof["state_population_stream_sha256"]
        == repeated_proof["state_population_stream_sha256"]
    )


def test_unified_exit_builder_has_no_policy_horizon_or_hindsight_targets() -> None:
    builder_source = BUILDER_PATH.read_text(encoding="utf-8")
    assert "--exit-target-lookahead-m1-steps" not in builder_source
    assert "fit_unified_exit_target_policy(" not in builder_source
    assert "unified_exit_optimal_stopping_targets(" not in builder_source


def test_unified_exit_full_state_population_is_outcome_independent() -> None:
    times = pd.date_range("2026-01-01", periods=1400, freq="1min", tz="UTC")
    baseline = unified_exit_state_population_arrays(
        m1_times=times,
        m1_start_row=800,
    )

    assert UNIFIED_EXIT_STATE_SELECTION_SCHEMA_VERSION == (
        "gx1_unified_exit_full_authoritative_state_pointer_population_v2"
    )
    np.testing.assert_array_equal(baseline["state_indices"], np.arange(512))
    assert baseline["state_valid_mask"].all()

    appended = unified_exit_state_population_arrays(
        m1_times=times.append(
            pd.date_range(
                times[-1] + pd.Timedelta(minutes=1),
                periods=200,
                freq="1min",
                tz="UTC",
            )
        ),
        m1_start_row=800,
    )
    for name, values in baseline.items():
        np.testing.assert_array_equal(values, appended[name])
    with pytest.raises(RuntimeError, match="STATE_POPULATION_INPUT_INVALID"):
        unified_exit_state_population_arrays(
            m1_times=times,
            m1_start_row=800.5,
        )
    with pytest.raises(RuntimeError, match="STATE_POPULATION_INPUT_INVALID"):
        unified_exit_state_population_arrays(
            m1_times=times.tz_localize(None),
            m1_start_row=800,
        )


def test_unified_exit_population_rejects_omitted_state_and_pointer_tamper() -> None:
    _split, episodes, proof, source = _in_memory_full_exit_split()

    mutations: list[pd.DataFrame] = []
    omitted = episodes.copy()
    omitted.loc[0, "path_state_count"] = 511
    mutations.append(omitted)
    shifted = episodes.copy()
    shifted.loc[[0, 1], "m1_start_row"] += 1
    mutations.append(shifted)

    for mutated in mutations:
        with pytest.raises(
            RuntimeError,
            match="EPISODE_VALUE_INVALID|POINTER_TIME_MISMATCH",
        ):
            _in_memory_full_exit_split(
                source=source,
                episodes=mutated,
                proof=proof,
            )

    episode_reordered = episodes.iloc[::-1].reset_index(drop=True)
    with pytest.raises(RuntimeError, match="EPISODE_ORDER_INVALID"):
        _in_memory_full_exit_split(
            source=source,
            episodes=episode_reordered,
            proof=proof,
        )

    source_times = pd.DatetimeIndex(
        pd.to_datetime(source["time"], utc=True, errors="raise")
    )
    ordered_population = []
    for row in episodes.itertuples(index=False):
        population = unified_exit_state_population_arrays(
            m1_times=source_times,
            m1_start_row=int(row.m1_start_row),
        )
        ordered_population.extend(
            (int(row.side_index), int(state))
            for state in population["state_indices"]
        )
    assert len(ordered_population) == 1024
    assert ordered_population[511] == (0, 511)
    assert ordered_population[512] == (1, 0)
    assert all(state < 512 for _side, state in ordered_population)


def test_unified_exit_first_state_has_full_history_and_executable_pnl() -> None:
    split, episodes, _proof, source = _in_memory_full_exit_split()
    start = int(episodes.iloc[0]["m1_start_row"])
    episode = split.materialize_causal_episode_core(0)
    assert episode is not None

    assert start == 600
    assert episode["exit_local_history_x"].shape == (
        479 + 512,
        MODEL_NATIVE_SIGNAL_DIM,
    )
    np.testing.assert_array_equal(
        episode["exit_local_history_x"][:480, 0],
        np.arange(start - 479, start + 1, dtype=np.float32),
    )
    assert episode["exit_path_x"].shape[:2] == (2, 512)
    assert np.any(episode["exit_path_x"][:, 0] != 0.0)
    expected_state_time = pd.Timestamp(source["time"].iloc[start]).value
    assert episode["exit_state_row_time_ns"][0] == expected_state_time
    assert episode["exit_decision_time_ns"][0] == (
        expected_state_time + pd.Timedelta(minutes=1).value
    )

    entry_ask = float(episodes.iloc[0]["entry_ask"])
    long_exit = float(source["bid_close"].iloc[start])
    expected_long_pnl = (long_exit - entry_ask) / entry_ask * 10_000.0
    entry_bid = float(episodes.iloc[1]["entry_bid"])
    short_exit = float(source["ask_close"].iloc[start])
    expected_short_pnl = (entry_bid - short_exit) / entry_bid * 10_000.0
    assert episode["exit_entry_bid_ask"][0, 1] == pytest.approx(entry_ask)
    assert episode["exit_entry_bid_ask"][1, 0] == pytest.approx(entry_bid)
    assert episode["exit_now_reward_bps"][0, 0] == pytest.approx(expected_long_pnl)
    assert episode["exit_now_reward_bps"][1, 0] == pytest.approx(expected_short_pnl)
    assert episode["exit_action_valid_mask"][:, :-1].all()
    assert not episode["exit_action_valid_mask"][:, -1, 0].any()
    assert episode["exit_action_valid_mask"][:, -1, 1].all()


def test_unified_exit_ineligible_entry_is_empty_but_half_pair_fails() -> None:
    split, _episodes, _proof, _source = _in_memory_full_exit_split(
        entry_row_count=2
    )
    assert split.materialize_causal_episode_core(1) is None

    split._episode_pointers.pop((0, 1))
    with pytest.raises(RuntimeError, match="EPISODE_PAIR_MISSING"):
        split.materialize_causal_episode_core(0)


def test_unified_exit_lifecycle_uses_authoritative_rows_across_market_closure() -> None:
    entries = pd.DataFrame(
        {"time": pd.to_datetime(["2026-01-01T00:00:00Z"], utc=True)}
    )
    source = _closed_m1_lifecycle_source()
    gapped = source.drop(index=200).reset_index(drop=True)
    episodes, proof = build_unified_exit_lifecycle_episodes(
        min_m1_start_row=0,
        entry_rows=entries,
        closed_m1=gapped,
        split_end=gapped["time"].iloc[-1] + pd.Timedelta(minutes=1),
        market_closure_contract=CANONICAL_NATIVE_CLOSURE_CONTRACT,
    )
    assert len(episodes) == 2
    assert proof["required_observed_m1_rows_per_episode"] == 512
    assert proof["m1_row_clock"] == (
        "consecutive_authoritative_closed_m1_source_rows"
    )
    with pytest.raises(RuntimeError, match="MARKET_CLOSURE_PROOF_REQUIRED"):
        build_unified_exit_lifecycle_episodes(
        min_m1_start_row=0,
            entry_rows=entries,
            closed_m1=gapped,
            split_end=gapped["time"].iloc[-1] + pd.Timedelta(minutes=1),
            market_closure_contract="unproven",
        )
    with pytest.raises(
        RuntimeError,
        match="UNIFIED_EXIT_LIFECYCLE_NO_COMPLETE_EPISODES",
    ):
        build_unified_exit_lifecycle_episodes(
        min_m1_start_row=0,
            entry_rows=entries,
            closed_m1=source,
            split_end=source["time"].iloc[400],
                market_closure_contract=CANONICAL_NATIVE_CLOSURE_CONTRACT,
        )


def test_unified_exit_lifecycle_rejects_price_scale_corruption() -> None:
    entries = pd.DataFrame(
        {"time": pd.to_datetime(["2026-01-01T00:00:00Z"], utc=True)}
    )
    source = _closed_m1_lifecycle_source()
    price_columns = [
        name for name in source.columns if name not in {"time", "volume"}
    ]
    source.loc[200:300, price_columns] /= 10.0

    with pytest.raises(RuntimeError, match="PRICE_SCALE_GLITCH"):
        build_unified_exit_lifecycle_episodes(
        min_m1_start_row=0,
            entry_rows=entries,
            closed_m1=source,
            split_end=source["time"].iloc[-1] + pd.Timedelta(minutes=1),
            market_closure_contract=CANONICAL_NATIVE_CLOSURE_CONTRACT,
        )


def test_unified_exit_lifecycle_corpus_replays_only_causal_prefixes(
    tmp_path: Path,
) -> None:
    entries = pd.DataFrame(
        {
            "time": pd.to_datetime(
                [
                    "2026-01-01T08:00:00Z",
                    "2026-01-01T08:05:00Z",
                ],
                utc=True,
            )
        }
    )
    generation_manifest, generation_root, _pointer = _strict_native_pair_fixture(
        tmp_path,
        successor_end="2026-01-02T06:00:00Z",
        trend_prices=True,
    )
    m1_source, m1_authority = require_unified_exit_m1_pair_authority(
        pair_manifest_path=generation_manifest,
        pair_generation_root=generation_root,
    )
    source = pd.read_parquet(m1_source)
    assert len(source) == m1_authority["m1_source_rows"] == 1800
    m1_feature_base, m1_feature_manifest = (
        _write_exact_m1_feature_surface_fixture(
            tmp_path,
            m1_source=m1_source,
            m1_frame=source,
            dataset_run_id="EXIT_LIFECYCLE_PYTEST_V1",
            pair_generation_id=str(m1_authority["pair_generation_id"]),
        )
    )
    m1_authority_sha256 = canonical_json_sha256(m1_authority)
    lifecycle_dir = tmp_path / "exit_lifecycle_20260101T000000Z"
    lifecycle_dir.mkdir()
    bindings: dict[str, dict[str, object]] = {}
    entry_paths: dict[str, Path] = {}
    for split in ("train", "val", "test"):
        entry_path = tmp_path / f"entry_{split}_20260101T000000Z.parquet"
        entries.to_parquet(entry_path, index=False)
        entry_paths[split] = entry_path
        episodes, proof = build_unified_exit_lifecycle_episodes(
            min_m1_start_row=0,
            entry_rows=entries,
            closed_m1=source,
            split_end=source["time"].iloc[-1] + pd.Timedelta(minutes=1),
            market_closure_contract=CANONICAL_NATIVE_CLOSURE_CONTRACT,
        )
        lifecycle_path = (
            lifecycle_dir / f"{split}_unified_exit_lifecycle.parquet"
        )
        episodes.to_parquet(lifecycle_path, index=False)
        proof.update(
            {
                "entry_run_id": "EXIT_LIFECYCLE_PYTEST_V1",
                "split": split,
                "entry_dataset_path": str(entry_path),
                "entry_dataset_sha256": sha256_file(entry_path),
                "m1_source_path": str(m1_source),
                "m1_source_sha256": sha256_file(m1_source),
                "m1_feature_base_path": str(m1_feature_base),
                "m1_feature_base_sha256": sha256_file(m1_feature_base),
                "m1_feature_base_manifest_path": str(m1_feature_manifest),
                "m1_feature_base_manifest_sha256": sha256_file(
                    m1_feature_manifest
                ),
                "m1_authority_sha256": m1_authority_sha256,
                "lifecycle_parquet": lifecycle_path.name,
                "lifecycle_parquet_sha256": sha256_file(lifecycle_path),
                "lifecycle_parquet_rows": len(episodes),
            }
        )
        split_manifest = (
            lifecycle_dir
            / f"{split}_unified_exit_lifecycle.manifest.json"
        )
        split_manifest.write_text(
            json.dumps(proof, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        bindings[split] = {
            "entry_dataset_path": str(entry_path),
            "entry_dataset_sha256": sha256_file(entry_path),
            "lifecycle_parquet": lifecycle_path.name,
            "lifecycle_parquet_sha256": sha256_file(lifecycle_path),
            "lifecycle_manifest": split_manifest.name,
            "lifecycle_manifest_sha256": sha256_file(split_manifest),
            "episode_rows": len(episodes),
            "state_population_rows": proof["state_population_rows"],
            "state_population_stream_sha256": proof[
                "state_population_stream_sha256"
            ],
        }
    root_manifest = lifecycle_dir / "UNIFIED_EXIT_LIFECYCLE_MANIFEST.json"
    root_manifest.write_text(
        json.dumps(
            {
                "schema_version": UNIFIED_EXIT_LIFECYCLE_EPISODE_SCHEMA_VERSION,
                "decision": "PASS",
                "entry_run_id": "EXIT_LIFECYCLE_PYTEST_V1",
                "m1_source_path": str(m1_source),
                "m1_source_sha256": sha256_file(m1_source),
                "m1_feature_base_path": str(m1_feature_base),
                "m1_feature_base_sha256": sha256_file(m1_feature_base),
                "m1_feature_base_manifest_path": str(m1_feature_manifest),
                "m1_feature_base_manifest_sha256": sha256_file(
                    m1_feature_manifest
                ),
                "m1_authority": m1_authority,
                "m1_authority_sha256": m1_authority_sha256,
                "path_state_count": 512,
                "state_population_schema_version": (
                    UNIFIED_EXIT_STATE_SELECTION_SCHEMA_VERSION
                ),
                "state_population_per_episode": 512,
                "m1_row_clock": (
                    "consecutive_authoritative_closed_m1_source_rows"
                ),
                "shared_feature_base_contract": (
                    entry_exit_shared_feature_base_contract()
                ),
                "side_order": ["long", "short"],
                "action_order": ["HOLD", "EXIT_NOW"],
                "splits": bindings,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    corpus = UnifiedExitLifecycleCorpus(
        root_manifest_path=root_manifest,
        entry_parquets={name: entry_paths[name] for name in ("train", "val")},
        dataset_run_id="EXIT_LIFECYCLE_PYTEST_V1",
    )
    with pytest.raises(RuntimeError, match="STATE_SAMPLE_RETIRED"):
        corpus.splits["train"].sample(0)
    episode = corpus.splits["val"].materialize_causal_episode_core(0)
    assert episode is not None
    full_action_valid = episode["exit_action_valid_mask"]
    expected_full_rows = 2 * 512
    assert int(episode["exit_state_valid_mask"].sum()) == expected_full_rows
    assert episode["exit_local_history_x"].shape[0] == 479 + 512
    assert episode["exit_path_x"].shape[:2] == (2, 512)
    np.testing.assert_array_equal(
        episode["exit_decision_time_ns"],
        episode["exit_state_row_time_ns"] + pd.Timedelta(minutes=1).value,
    )
    assert np.isfinite(episode["exit_now_reward_bps"]).all()
    assert int(full_action_valid.sum()) == 2046
    assert full_action_valid[0, 511].tolist() == [False, True]
    assert full_action_valid[1, 511].tolist() == [False, True]
    assert corpus.evidence["future_outcomes_used_as_model_inputs"] is False
    assert corpus.evidence["sample_selection_depends_on_future_target"] is False
    assert corpus.evidence["training_population"] == (
        "gx1_unified_exit_full_authoritative_state_pointer_population_v2"
    )
    assert corpus.evidence["validation_population"] == "all_authoritative_states"
    assert corpus.evidence["test_population"] == "all_authoritative_states"
    assert corpus.evidence["splits"]["train"]["state_population_rows"] == (
        len(episodes) * 512
    )

    # TEST is sealed and semantically validated even though this consumer only
    # selected TRAIN/VAL. Re-sealing every ordinary file hash cannot bless an
    # source-invalid pointers against the source-recomputed population proof.
    test_lifecycle_path = (
        lifecycle_dir / "test_unified_exit_lifecycle.parquet"
    )
    tampered_test = pd.read_parquet(test_lifecycle_path)
    tampered_test.loc[[0, 1], "m1_start_row"] += 1
    tampered_test.to_parquet(test_lifecycle_path, index=False)
    test_manifest_path = (
        lifecycle_dir / "test_unified_exit_lifecycle.manifest.json"
    )
    test_manifest = json.loads(
        test_manifest_path.read_text(encoding="utf-8")
    )
    test_manifest["lifecycle_parquet_sha256"] = sha256_file(
        test_lifecycle_path
    )
    test_manifest_path.write_text(
        json.dumps(test_manifest, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    resealed_root = json.loads(root_manifest.read_text(encoding="utf-8"))
    resealed_root["splits"]["test"]["lifecycle_parquet_sha256"] = (
        sha256_file(test_lifecycle_path)
    )
    resealed_root["splits"]["test"]["lifecycle_manifest_sha256"] = (
        sha256_file(test_manifest_path)
    )
    root_manifest.write_text(
        json.dumps(resealed_root, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="POINTER_TIME_MISMATCH"):
        UnifiedExitLifecycleCorpus(
            root_manifest_path=root_manifest,
            entry_parquets={
                name: entry_paths[name] for name in ("train", "val")
            },
            dataset_run_id="EXIT_LIFECYCLE_PYTEST_V1",
        )


def test_aux_target_builder_requires_bid_ask_high_low() -> None:
    frame = _spread_tape().drop(columns=["ask_low"])

    with pytest.raises(RuntimeError, match="AUX_SPREAD_TAPE_MISSING.*ask_low"):
        _build_model_native_aux_head_targets(frame)


def test_selected_side_bad_path_is_copied_from_future_outcome() -> None:
    selected_side = np.array([0, 1, -1, 0], dtype=np.int8)
    long_bad = np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float32)
    short_bad = np.array([1.0, 0.0, 1.0, 0.0], dtype=np.float32)

    scalar_bad = _selected_side_bad_path_target(selected_side, long_bad, short_bad)

    assert scalar_bad.tolist() == [0.0, 0.0, 0.0, 1.0]


def test_position_size_target_uses_selected_future_path() -> None:
    size, mask = _position_size_target_from_path(
        mfe_first_n_bps=np.array([30.0, 5.0, 0.0, 1.0], dtype=np.float32),
        mae_first_n_bps=np.array([5.0, 20.0, 0.0, 1.0], dtype=np.float32),
        selected_side=np.array([0, 1, -1, 0], dtype=np.int8),
        trade_mask=np.array([1.0, 1.0, 0.0, 1.0], dtype=np.float32),
        target_policy=entry_position_size_target_policy_fixture(),
    )

    assert size[0] > size[3] > size[1]
    assert size[2] == 0.0
    assert mask.tolist() == [1.0, 1.0, 0.0, 1.0]


def test_position_size_target_decreases_monotonically_with_adverse_excursion() -> None:
    size, mask = _position_size_target_from_path(
        mfe_first_n_bps=np.full(4, 20.0, dtype=np.float32),
        mae_first_n_bps=np.array([0.0, 10.0, 20.0, 40.0], dtype=np.float32),
        selected_side=np.zeros(4, dtype=np.int8),
        trade_mask=np.ones(4, dtype=np.float32),
        target_policy=entry_position_size_target_policy_fixture(),
    )

    assert np.all(np.diff(size) <= 0.0)
    assert mask.tolist() == [1.0] * 4


@pytest.mark.parametrize(
    ("mfe", "side", "mask"),
    [
        (np.array([np.nan], dtype=np.float32), np.array([0]), np.array([1.0])),
        (np.array([3.0], dtype=np.float32), np.array([0]), np.array([0.5])),
        (np.array([3.0], dtype=np.float32), np.array([0]), np.array([0.0])),
    ],
)
def test_position_size_target_fails_closed_on_invalid_evidence(
    mfe, side, mask
) -> None:
    with pytest.raises(RuntimeError, match="POSITION_SIZE_TARGET"):
        _position_size_target_from_path(
            mfe_first_n_bps=mfe,
            mae_first_n_bps=np.array([1.0], dtype=np.float32),
            selected_side=side,
            trade_mask=mask,
            target_policy=entry_position_size_target_policy_fixture(),
        )


def test_position_size_target_rejects_signed_negative_mae() -> None:
    with pytest.raises(RuntimeError, match="POSITION_SIZE_TARGET_INPUT_INVALID"):
        _position_size_target_from_path(
            mfe_first_n_bps=np.array([3.0], dtype=np.float32),
            mae_first_n_bps=np.array([-1.0], dtype=np.float32),
            selected_side=np.array([0], dtype=np.int8),
            trade_mask=np.array([1.0], dtype=np.float32),
            target_policy=entry_position_size_target_policy_fixture(),
        )


def test_contract_forbids_feature_derived_core_target_rewrites() -> None:
    target = representation_auxiliary_outcome_contract()[
        "representation_auxiliary_outcomes"
    ]

    assert target["label_source"] == "future_executable_pnl_outcomes_only"
    assert target["feature_derived_core_rewrites_allowed"] is False
    assert target["utility_order_forcing_allowed"] is False
    assert target["structural_context_auxiliaries"]["may_change_core_targets"] is False


def test_builder_has_no_structural_side_or_utility_repair_primitive() -> None:
    source = BUILDER_PATH.read_text(encoding="utf-8")

    assert "def _apply_structural_side_repair" not in source
    assert "def _apply_structural_utility_repair" not in source
    assert "_structural_utility_repair_masks" not in source
    assert "structural_short_to_long" not in source
    assert "structural_long_to_short" not in source
    assert "np.nan_to_num(_side_score" not in source
    assert "LONG_WINDOW_TEACHER" not in source
    assert "long_window_teacher" not in source
    assert "y_teacher_bad_long" not in source
    assert "y_teacher_winner_long" not in source
    assert "GX1_V10_SPREAD_AWARE_RISK_TARGETS" not in source
    assert "GX1_ENTRY_DIRECTION_TARGET_MODE" not in source
    assert "GX1_ENTRY_DIRECTION_UTILITY_" not in source


def test_builder_has_one_model_native_signal_path_and_no_context_soft_pass() -> None:
    source = BUILDER_PATH.read_text(encoding="utf-8")
    retired_provider = "X" + "GB"

    forbidden = (
        f"{retired_provider}MultiheadModel",
        f"{retired_provider}InputSanitizer",
        "proba_to_signal_bridge_v1",
        "external_tree_sidecar_bundle_path",
        "external_tree_sidecar_model_sha256",
        f"GX1_{retired_provider}_BUNDLE_DIR",
        "--external_tree_sidecar",
        "--neutral-external_tree_sidecar-bridge",
        f"HARD_NEG_LONG_MIN_{retired_provider}_P_LONG",
        "hard_negative_uses_external_tree_sidecar_predictions",
        "hard_negative_long_external_tree_sidecar_p_long_min",
        "entry_runtime_gates",
        "flat_veto",
        "tradable_gate",
        "quality_gate",
        "allow_zero_ctx",
        "_feat.get(_k, 0.0)",
        'fillna("UNKNOWN")',
    )
    assert [token for token in forbidden if token in source] == []
    assert '"hard_negative_candidate_source": _hard_negative_candidate_source' in source
