from __future__ import annotations

import json
from unittest import mock
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gx1.contracts.xau_tape_provenance_v1 import (
    CANONICAL_NATIVE_CLOSURE_CONTRACT,
    canonical_xau_source_descriptor_v1,
)
from gx1.execution import v12_canonical_incremental as incremental
from gx1.scripts import backfill_xauusd_m5_from_oanda as native_publisher
import gx1.contracts.unified_exit_lifecycle_v1 as lifecycle_contract
from gx1.contracts.entry_model_native_aux_targets_v3 import (
    MODEL_NATIVE_AUX_RISK_HORIZONS,
    MODEL_NATIVE_DIP_MAE_TARGET_COLUMNS,
    MODEL_NATIVE_DIP_MFE_TARGET_COLUMNS,
    MODEL_NATIVE_DIP_MFE_UPPER_SAFETY_CAP_BPS,
    require_model_native_aux_target_contract,
    require_model_native_aux_target_emission_contract,
)
from gx1.contracts.unified_exit_lifecycle_v1 import (
    UNIFIED_EXIT_LIFECYCLE_EPISODE_SCHEMA_VERSION,
    UNIFIED_EXIT_M1_AUTHORITY_SCHEMA_VERSION,
    UnifiedExitLifecycleCorpus,
    canonical_json_sha256,
    require_unified_exit_m1_pair_authority,
    sha256_file,
)
from gx1.contracts.entry_exit_feature_base_v1 import (
    entry_exit_shared_feature_base_contract,
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
    hierarchical_direction_label_contract,
    model_native_aux_target_contract_metadata,
)
from tests.test_oanda_backfill_vedtak_gate import _FakeOandaClient


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

    assert contract["schema_version"] == "entry_model_native_aux_targets_v5"
    assert len(contract["columns"]) == 46
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
    assert contract["offline_rl"]["action_value_layout"] == "action_major_then_horizon"
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


def test_action_values_are_full_counterfactual_spread_aware_path_utilities() -> None:
    frame = _spread_tape()
    targets, _ = _build_model_native_aux_head_targets(frame)

    entry_ask = frame.loc[0, "ask_close"]
    long_pnl = (frame.loc[12, "bid_close"] - entry_ask) / entry_ask * 1e4
    long_mfe = (frame.loc[12, "bid_high"] - entry_ask) / entry_ask * 1e4
    long_mae = max(
        0.0,
        (entry_ask - frame.loc[1:12, "bid_low"].min()) / entry_ask * 1e4,
    )
    expected_long = long_pnl + 0.35 * long_mfe - 1.15 * long_mae + 0.25 * (
        long_mfe - long_mae
    )

    assert targets["y_action_value_long_K12"][0] == pytest.approx(expected_long)
    assert targets["y_action_value_short_K12"][0] < 0.0
    assert targets["y_action_value_flat_K12"][0] == 0.0


def test_aux_target_validator_rejects_finite_incomplete_tail() -> None:
    targets, _ = _build_model_native_aux_head_targets(_spread_tape())
    broken = {name: values.copy() for name, values in targets.items()}
    broken["y_forecast_ret_K1"][-1] = 0.0

    with pytest.raises(RuntimeError, match="AUX_TARGET_COMPLETENESS_INVALID"):
        _validate_model_native_aux_head_targets(broken, n_rows=130)


def _closed_m1_lifecycle_source(n_rows: int = 560) -> pd.DataFrame:
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


def _strict_native_pair_fixture(
    tmp_path: Path,
) -> tuple[Path, Path, Path]:
    native_m1_parent_root = tmp_path / "native-m1-parent"
    native_m5_parent_root = tmp_path / "native-m5-parent"
    native_m1_root = tmp_path / "native-m1"
    native_m5_root = tmp_path / "native-m5"
    vedtak = "XAU_NATIVE_PAIR_LIFECYCLE_FIXTURE_V1"
    parent_end = "2026-01-01T09:20:00Z"
    successor_end = "2026-01-01T09:25:00Z"
    with mock.patch.object(
        native_publisher,
        "_require_clean_repository",
        return_value="a" * 40,
    ):
        native_publisher.materialize_native_xau_snapshot(
            client=_FakeOandaClient(timeframe="M1"),
            timeframe="M1",
            vedtak_id=vedtak,
            start_utc="2026-01-01T00:00:00Z",
            end_utc=parent_end,
            out_root=native_m1_parent_root,
        )
        native_publisher.materialize_native_xau_snapshot(
            client=_FakeOandaClient(timeframe="M5"),
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
            client=_FakeOandaClient(timeframe="M1"),
            timeframe="M1",
            vedtak_id=vedtak,
            end_utc=successor_end,
            out_root=native_m1_root,
            parent_root=native_m1_parent_root,
            expected_parent_manifest_sha256=m1_parent["manifest_sha256"],
        )
        native_publisher.materialize_native_xau_successor(
            client=_FakeOandaClient(timeframe="M5"),
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
        entry_rows=entries,
        closed_m1=source,
        split_end=split_end,
        target_lookahead_m1_steps=3,
        market_closure_contract=CANONICAL_NATIVE_CLOSURE_CONTRACT,
    )
    repeated, repeated_proof = build_unified_exit_lifecycle_episodes(
        entry_rows=entries,
        closed_m1=source,
        split_end=split_end,
        target_lookahead_m1_steps=3,
        market_closure_contract=CANONICAL_NATIVE_CLOSURE_CONTRACT,
    )

    pd.testing.assert_frame_equal(episodes, repeated)
    assert len(episodes) == 4
    assert episodes.groupby("entry_row_index")["side"].apply(list).tolist() == [
        ["long", "short"],
        ["long", "short"],
    ]
    assert (episodes["entry_available_at"] == episodes["m1_start_time"]).all()
    assert (episodes["path_state_count"] == 512).all()
    assert (
        episodes["non_tied_target_count"]
        + episodes["tied_target_count"]
        == episodes["path_state_count"]
    ).all()
    assert proof["decision"] == "PASS"
    assert proof["entry_side_selection"] == (
        "both_sides_for_every_causal_entry_snapshot"
    )
    assert proof["path_values_duplicated_into_episode_artifact"] is False
    assert proof["target_counts"]["HOLD"] > 0
    assert proof["target_counts"]["EXIT_NOW"] > 0
    assert len(proof["target_stream_sha256"]) == 64
    assert (
        proof["target_stream_sha256"]
        == repeated_proof["target_stream_sha256"]
    )


def test_unified_exit_lifecycle_uses_authoritative_rows_across_market_closure() -> None:
    entries = pd.DataFrame(
        {"time": pd.to_datetime(["2026-01-01T00:00:00Z"], utc=True)}
    )
    source = _closed_m1_lifecycle_source()
    gapped = source.drop(index=200).reset_index(drop=True)

    episodes, proof = build_unified_exit_lifecycle_episodes(
        entry_rows=entries,
        closed_m1=gapped,
        split_end=gapped["time"].iloc[-1] + pd.Timedelta(minutes=1),
        target_lookahead_m1_steps=3,
        market_closure_contract=CANONICAL_NATIVE_CLOSURE_CONTRACT,
    )
    assert len(episodes) == 2
    assert proof["required_observed_m1_rows_per_episode"] == 515
    assert proof["m1_row_clock"] == (
        "consecutive_authoritative_closed_m1_source_rows"
    )
    with pytest.raises(RuntimeError, match="MARKET_CLOSURE_PROOF_REQUIRED"):
        build_unified_exit_lifecycle_episodes(
            entry_rows=entries,
            closed_m1=gapped,
            split_end=gapped["time"].iloc[-1] + pd.Timedelta(minutes=1),
            target_lookahead_m1_steps=3,
            market_closure_contract="unproven",
        )
    with pytest.raises(
        RuntimeError,
        match="UNIFIED_EXIT_LIFECYCLE_NO_COMPLETE_EPISODES",
    ):
        build_unified_exit_lifecycle_episodes(
            entry_rows=entries,
            closed_m1=source,
            split_end=source["time"].iloc[400],
            target_lookahead_m1_steps=3,
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
            entry_rows=entries,
            closed_m1=source,
            split_end=source["time"].iloc[-1] + pd.Timedelta(minutes=1),
            target_lookahead_m1_steps=3,
            market_closure_contract=CANONICAL_NATIVE_CLOSURE_CONTRACT,
        )


def test_unified_exit_lifecycle_corpus_replays_only_causal_prefixes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
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
    m1_source = tmp_path / "xau_m1_20260101T000000Z.parquet"
    source = _closed_m1_lifecycle_source(n_rows=1200)
    source.to_parquet(m1_source, index=False)
    m1_feature_base = tmp_path / "xau_m1_feature_base.parquet"
    zero_signal = np.zeros(513, dtype=np.float32).tolist()
    zero_ctx_cont = np.zeros(142, dtype=np.float32).tolist()
    zero_ctx_cat = np.zeros(5, dtype=np.int64).tolist()
    pd.DataFrame(
        {
            "time": source["time"],
            "signal": [zero_signal for _ in range(len(source))],
            "ctx_cont": [zero_ctx_cont for _ in range(len(source))],
            "ctx_cat": [zero_ctx_cat for _ in range(len(source))],
        }
    ).to_parquet(m1_feature_base, index=False)
    m1_feature_manifest = Path(str(m1_feature_base) + ".manifest.json")
    m1_feature_manifest.write_text(
        json.dumps(
            {
                "schema_version": "gx1_entry_exit_m1_feature_surface_v1",
                "decision": "PASS",
                "dataset_run_id": "EXIT_LIFECYCLE_PYTEST_V1",
                "output_parquet": str(m1_feature_base),
                "output_parquet_sha256": sha256_file(m1_feature_base),
                "shared_feature_base_contract": (
                    entry_exit_shared_feature_base_contract()
                ),
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    m1_authority = {
        "schema_version": UNIFIED_EXIT_M1_AUTHORITY_SCHEMA_VERSION,
        "pair_manifest_path": str(tmp_path / "PAIR_MANIFEST.json"),
        "pair_generation_root": str(tmp_path / "generations"),
        "m1_source_path": str(m1_source),
        "m1_source_sha256": sha256_file(m1_source),
    }
    m1_authority_sha256 = canonical_json_sha256(m1_authority)
    monkeypatch.setattr(
        lifecycle_contract,
        "require_unified_exit_m1_pair_authority",
        lambda **_kwargs: (m1_source, dict(m1_authority)),
    )
    lifecycle_dir = tmp_path / "exit_lifecycle_20260101T000000Z"
    lifecycle_dir.mkdir()
    bindings: dict[str, dict[str, object]] = {}
    entry_paths: dict[str, Path] = {}
    for split in ("train", "val", "test"):
        entry_path = tmp_path / f"entry_{split}_20260101T000000Z.parquet"
        entries.to_parquet(entry_path, index=False)
        entry_paths[split] = entry_path
        episodes, proof = build_unified_exit_lifecycle_episodes(
            entry_rows=entries,
            closed_m1=source,
            split_end=source["time"].iloc[-1] + pd.Timedelta(minutes=1),
            target_lookahead_m1_steps=3,
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
            "target_counts": proof["target_counts"],
            "target_stream_sha256": proof["target_stream_sha256"],
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
                    "target_lookahead_m1_steps": 3,
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
        entry_parquets=entry_paths,
        dataset_run_id="EXIT_LIFECYCLE_PYTEST_V1",
    )
    sample = corpus.splits["train"].sample(0)
    valid = sample["exit_sample_valid"]

    assert valid.any()
    assert set(sample["exit_action_target"][valid]) == {0, 1}
    assert set(sample["exit_side_index"][valid]) == {0, 1}
    for slot in np.flatnonzero(valid):
        length = int(sample["exit_path_lengths"][slot])
        assert 1 <= length <= 512
        assert np.any(sample["exit_path_x"][slot, :length] != 0.0)
        assert np.all(sample["exit_path_x"][slot, length:] == 0.0)
    assert corpus.evidence["future_outcomes_used_as_model_inputs"] is False
    assert corpus.evidence["splits"]["train"]["selected_target_counts"][
        "HOLD"
    ] > 0
    assert corpus.evidence["splits"]["train"]["selected_target_counts"][
        "EXIT_NOW"
    ] > 0


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
    size = _position_size_target_from_path(
        mfe_first_n_bps=np.array([30.0, 5.0, 0.0, 1.0], dtype=np.float32),
        mae_first_n_bps=np.array([5.0, 20.0, 0.0, 1.0], dtype=np.float32),
        atr_bps=np.array([10.0, 10.0, 10.0, 10.0], dtype=np.float32),
        trade_mask=np.array([1.0, 1.0, 0.0, 1.0], dtype=np.float32),
    )

    assert size[0] > 0.5
    assert size[1] < 0.5
    assert size[2] == 0.5
    assert size[3] == 0.5


def test_position_size_target_decreases_monotonically_with_adverse_excursion() -> None:
    size = _position_size_target_from_path(
        mfe_first_n_bps=np.full(4, 20.0, dtype=np.float32),
        mae_first_n_bps=np.array([0.0, 10.0, 20.0, 40.0], dtype=np.float32),
        atr_bps=np.full(4, 10.0, dtype=np.float32),
        trade_mask=np.ones(4, dtype=np.float32),
    )

    assert np.all(np.diff(size) < 0.0)


@pytest.mark.parametrize(
    ("atr", "mask"),
    [
        (np.array([0.0], dtype=np.float32), np.array([1.0], dtype=np.float32)),
        (np.array([np.nan], dtype=np.float32), np.array([1.0], dtype=np.float32)),
        (np.array([10.0], dtype=np.float32), np.array([0.5], dtype=np.float32)),
    ],
)
def test_position_size_target_fails_closed_on_invalid_evidence(atr, mask) -> None:
    with pytest.raises(ValueError, match="POSITION_SIZE_TARGET_INPUT_INVALID"):
        _position_size_target_from_path(
            mfe_first_n_bps=np.array([3.0], dtype=np.float32),
            mae_first_n_bps=np.array([1.0], dtype=np.float32),
            atr_bps=atr,
            trade_mask=mask,
        )


def test_position_size_target_rejects_signed_negative_mae() -> None:
    with pytest.raises(ValueError, match="mae_first_n_bps"):
        _position_size_target_from_path(
            mfe_first_n_bps=np.array([3.0], dtype=np.float32),
            mae_first_n_bps=np.array([-1.0], dtype=np.float32),
            atr_bps=np.array([10.0], dtype=np.float32),
            trade_mask=np.array([1.0], dtype=np.float32),
        )


def test_contract_forbids_feature_derived_core_target_rewrites() -> None:
    target = hierarchical_direction_label_contract()["hierarchical_direction_targets"]

    assert target["core_target_source"] == "future_path_and_utility_outcomes_only"
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
