"""Exit-IQL emitter-parity regression tests.

Guards the fail-closed coverage guard on the active Exit-IQL adapter. The
adapter raises if a REQUIRED feature (training
std >= STD_REQUIRED_THRESHOLD) is absent at serve, instead of silently 0-filling
it. These tests prove ``v12_exit_iql_live.build_bar_state`` emits every required
feature when given complete inputs.

The lightweight test uses ``object.__new__`` and a real canonical row; the
opt-in test uses the full live loader. Both skip cleanly when external Exit
artifacts are unavailable.
"""
from __future__ import annotations

import glob
import json
import math
import os
from pathlib import Path

import pandas as pd
import pytest

from tests.model_native_sizing_support import unverified_learned_sizing_authority
from gx1.features.entry_specialist_feature_groups_v1 import (
    MODEL_NATIVE_TRAINING_SPECIALISTS,
)

REPO = Path(__file__).resolve().parents[1]
PROJECT_STATE = REPO / "PROJECT_STATE_artifacts.json"

EXIT_BUNDLE = Path(
    "/home/andre2/GX1_DATA/reports/truth_e2e_sanity/"
    "BUILD_EXIT_IQL_V3_M1_COSTFIX_HEADS_FAST_20260528T205051Z"
)
# The live exit builder (v12_exit_iql_live.decide_for_trade -> build_bar_state)
# is passed an augmented canonical_v3 row (PrebuiltStateLoader output), NOT the V3
# per-bar scored frame. The canonical_v3 prebuilt carries the 54 base _v1_* columns
# that build_bar_state aliases to _v1_*_canon_v1 — so this is the correct serve-row
# source for the exit emitter-parity check.
EXIT_CV3_PREBUILTS = (
    "/home/andre2/GX1_DATA/data/data/prebuilt/xauusd_m5_CANONICAL_V3_2020_2026.parquet",
)


def _find_cv3_prebuilt() -> str | None:
    for p in EXIT_CV3_PREBUILTS:
        if Path(p).is_file():
            return p
    hits = glob.glob(
        "/home/andre2/GX1_DATA/**/xauusd_m5_CANONICAL_V3_2020_2026.parquet", recursive=True
    )
    return hits[0] if hits else None


def _project_state_entry(role: str) -> dict:
    if not PROJECT_STATE.exists():
        pytest.skip("PROJECT_STATE_artifacts.json not found")
    contract = json.loads(PROJECT_STATE.read_text())
    entry = (contract.get("active") or {}).get(role) or {}
    if entry.get("status") != "ACTIVE":
        pytest.skip(f"active/{role} missing or not ACTIVE in PROJECT_STATE_artifacts.json")
    path = Path(entry.get("path", ""))
    if not path.exists():
        pytest.skip(f"active/{role} path missing: {path}")
    return entry


def _complete_entry_snapshot() -> dict:
    """Model-entry snapshot fields carried into the Exit-IQL state."""
    direction_logits = [0.8, -0.2, 0.1]
    direction_exp = [math.exp(value - max(direction_logits)) for value in direction_logits]
    direction_probs = [value / sum(direction_exp) for value in direction_exp]
    public_logits = [max(direction_logits[0], direction_logits[1]), direction_logits[2]]
    public_exp = [math.exp(value - max(public_logits)) for value in public_logits]
    public_probs = [value / sum(public_exp) for value in public_exp]
    side_logits = [1.0, -0.5]
    side_exp = [math.exp(value - max(side_logits)) for value in side_logits]
    side_probs = [value / sum(side_exp) for value in side_exp]
    side_bad_logits = [-1.2, 0.7]
    side_validity_logits = [1.4, -0.3]
    mtf_logits = [0.8, -0.2, -0.6]
    mtf_exp = [math.exp(value - max(mtf_logits)) for value in mtf_logits]
    mtf_probs = [value / sum(mtf_exp) for value in mtf_exp]
    rail_logits = [-1.0, -0.5, 0.0, 0.5, 1.0, 1.5]
    tf_agreement_logit = 0.4
    position_size_logit = -0.25
    path_quality_log_var = -1.2
    return {
        "decision_ts": "2026-01-06T11:25:00+00:00",
        "runtime_evidence_schema_version": "entry_model_native_runtime_evidence_v1",
        "model_policy": "xau_seq513_model_native_direction_argmax_v1",
        "session_id": 1,
        "session": "EU",
        "decision_available_ts": "2026-01-06T11:30:00+00:00",
        "entry_signal_latency_sec": 0.0,
        "context_cutoff_ts": "2026-01-06T11:25:00+00:00",
        "context_age_m5_bars": 0,
        "raw_direction_logits": [0.77, -0.11, 0.11],
        "direction_logits": direction_logits,
        "direction_probs": direction_probs,
        "model_direction_index": 0,
        "model_direction": "LONG",
        "public_trade_flat_decision_logits": public_logits,
        "public_trade_flat_decision_probs": public_probs,
        "public_trade_flat_decision_index": 0,
        "public_trade_flat_decision": "TRADE",
        "selected_side": 0,
        "model_native_logits": [0.25, -0.1, 0.05],
        "path_quality_raw": 0.55,
        "path_quality_pred": 0.55,
        "tradable_prob": 0.6,
        "tradable_logit": math.log(0.6 / 0.4),
        "mfe_first_n": 12.0,
        "mfe_first_n_pred": 12.0,
        "path_quality": 0.55,
        "trade_logit": 0.3,
        "bad_path_prob": 0.2,
        "bad_path_logit_raw": math.log(0.2 / 0.8),
        "bad_path_logit": math.log(0.2 / 0.8),
        "clean_edge_logit": math.log(0.76 / 0.24),
        "clean_edge_prob": 0.76,
        "survival_logit": math.log(0.68 / 0.32),
        "survival_prob": 0.68,
        "dip_pred": [0.0] * 18,
        "forecast_pred": [0.0] * 4,
        "timing_pred": [0.0] * 12,
        "tail_risk_pred": [0.0] * 6,
        "vol_forecast_pred": [0.0] * 3,
        "p_trade": public_probs[0],
        "p_flat_hier": public_probs[1],
        "atr_bps": 13.0,
        "tf_agreement_logit": tf_agreement_logit,
        "tf_agreement_pred": 1.0 / (1.0 + math.exp(-tf_agreement_logit)),
        "path_quality_log_var": path_quality_log_var,
        "path_quality_std": math.exp(0.5 * path_quality_log_var),
        "position_size_logit": position_size_logit,
        "position_size_pred": 1.0 / (1.0 + math.exp(-position_size_logit)),
        "sizing_authority_contract": unverified_learned_sizing_authority(),
        "p_long_given_trade": side_probs[0],
        "p_short_given_trade": side_probs[1],
        "side_logits": side_logits,
        "side_probs": side_probs,
        "side_utility": [2.4, -0.8],
        "side_bad_path_logit": side_bad_logits,
        "long_bad_path_prob": 1.0 / (1.0 + math.exp(-side_bad_logits[0])),
        "short_bad_path_prob": 1.0 / (1.0 + math.exp(-side_bad_logits[1])),
        "side_validity_logit": side_validity_logits,
        "long_validity_prob": 1.0 / (1.0 + math.exp(-side_validity_logits[0])),
        "short_validity_prob": 1.0 / (1.0 + math.exp(-side_validity_logits[1])),
        "side_mae": [-3.2, -8.1],
        "mtf_dir_logits": mtf_logits,
        "mtf_dir_probs": mtf_probs,
        "mtf_trend_evidence": 0.69,
        "specialist_names": list(MODEL_NATIVE_TRAINING_SPECIALISTS),
        "specialist_gate": [0.125] * len(MODEL_NATIVE_TRAINING_SPECIALISTS),
        "trendline_rail_logits": rail_logits,
        "trendline_rail_probs": [1.0 / (1.0 + math.exp(-value)) for value in rail_logits],
        "geometry_channel_edge_pressure": 0.42,
        "geometry_rising_support_rail_long_pressure": 0.81,
        "geometry_rising_support_rail_short_trap_pressure": 0.07,
        "geometry_falling_resistance_rail_short_pressure": 0.02,
        "geometry_falling_resistance_rail_long_trap_pressure": 0.03,
        "calibration_version": "direction-cal-v1",
        "direction_calibration_enabled": True,
        "direction_calibration_temperature": 1.1,
        "direction_calibration_bias": [0.1, -0.1, 0.0],
        "path_calibration_enabled": True,
        "path_calibration": {
            "enabled": True,
            "version": "path-cal-v1",
            "path_quality_scale": 1.0,
            "path_quality_shift": 0.0,
            "bad_path_temperature": 1.0,
            "bad_path_bias": 0.0,
        },
    }


def _complete_v3_state() -> dict[str, float]:
    return {
        "v3_v8_should_exit_prob": 0.62,
        "v3_v8_profit_protect_prob": 0.41,
        "v3_v8_family_argmax": 2.0,
        "v3_v8_family_logit_max": 1.3,
    }


def _load_required(bundle: Path, variant: str, fold: str) -> tuple[set, list]:
    if not bundle.is_dir():
        pytest.skip(f"bundle missing: {bundle}")
    from gx1.runtime.exit_iql_v2_adapter import ExitIQLV2Adapter
    try:
        a = ExitIQLV2Adapter.load(
            bundle,
            variant=variant,
            fold_id=fold,
            aggregator="max",
            prefer_cuda=False,
        )
    except Exception as e:  # noqa: BLE001 — env without torch/ckpt → skip, not fail
        pytest.skip(f"cannot load exit adapter: {e}")
    return set(a.required_feature_names), list(a.feature_names)


def _unresolved_required(required: set, emitted: set) -> list:
    """REQUIRED features not covered by the emitted dict.

    A required one-hot ``cat__val`` (entry, double underscore) is considered
    covered if the underlying categorical column ``cat`` was emitted (the live
    builder always emits every one-hot slot once the column is present).
    """
    miss = []
    for f in sorted(required):
        if f in emitted:
            continue
        if "__" in f and f.split("__", 1)[0] in emitted:
            continue
        miss.append(f)
    return miss


def test_exit_builder_emits_all_required():
    required, _ = _load_required(EXIT_BUNDLE, "R_NET_REAL", "FOLD_1")
    assert required, "exit REQUIRED set should be non-empty"
    cv3 = _find_cv3_prebuilt()
    if cv3 is None:
        pytest.skip("canonical_v3 prebuilt not found")
    df = pd.read_parquet(cv3)
    if df.empty:
        pytest.skip(f"empty cv3 prebuilt: {cv3}")
    row = df.iloc[-1].copy()  # last closed M5 bar — what the live loader serves
    row.name = pd.Timestamp("2026-01-06 12:00:00")
    from gx1.execution.v12_exit_iql_live import ExitIQLLiveInference
    from gx1.execution.v12_trade_state import TradeState
    trade = TradeState.open_unit_normalized_research(
        entry_ts=pd.Timestamp("2026-01-06 11:30:00"),
        side="long",
        entry_bid=2000.0,
        entry_ask=2000.2,
        v10_snapshot=_complete_entry_snapshot(),
        normalization_contract="unit_normalized_direction_exit_research_v1",
    )
    inst = object.__new__(ExitIQLLiveInference)
    bs = ExitIQLLiveInference.build_bar_state(inst, trade, row, _complete_v3_state())
    missing = _unresolved_required(required, set(bs.keys()))
    assert not missing, (
        f"exit build_bar_state omits {len(missing)} REQUIRED feature(s) → strict "
        f"adapter would halt live: {missing[:20]}"
    )


def test_exit_builder_rejects_missing_v3_state_instead_of_zero_filling() -> None:
    from gx1.execution.v12_exit_iql_live import ExitIQLLiveInference
    from gx1.execution.v12_trade_state import TradeState

    trade = TradeState.open_unit_normalized_research(
        entry_ts=pd.Timestamp("2026-01-06 11:30:00", tz="UTC"),
        side="long",
        entry_bid=2000.0,
        entry_ask=2000.2,
        v10_snapshot=_complete_entry_snapshot(),
        normalization_contract="unit_normalized_direction_exit_research_v1",
    )
    inst = object.__new__(ExitIQLLiveInference)

    with pytest.raises(RuntimeError, match="EXIT_IQL_V3_STATE_MISSING"):
        ExitIQLLiveInference.build_bar_state(inst, trade, pd.Series(dtype=float), None)


@pytest.mark.skipif(
    os.environ.get("GX1_RUN_LIVE_LOADER_CONTRACT_TESTS") != "1",
    reason="full live-loader contract test is intentionally opt-in; it runs heavy prebuilt augmentation",
)
def test_active_exit_live_loader_emits_all_required():
    """Opt-in full-stack contract check for the ACTIVE Exit-IQL.

    A raw canonical_v3 parquet row is not enough for the current clean Exit-IQL:
    the ACTIVE bundle requires AUG64 columns that live adds via PrebuiltStateLoader.
    This test intentionally uses that same loader when explicitly enabled.
    """
    os.environ.setdefault("GX1_REGIME_V4", "1")
    os.environ.setdefault("GX1_TREND_REGIME_FROM_D1", "1")
    entry = _project_state_entry("exit_iql")
    from gx1.execution.v12_exit_iql_live import ExitIQLLiveInference
    from gx1.execution.v12_state_from_prebuilt import PrebuiltStateLoader
    from gx1.execution.v12_trade_state import TradeState

    loader = PrebuiltStateLoader()
    loader.load()
    if loader._cv3 is None or loader._cv3.empty:
        pytest.skip("live PrebuiltStateLoader produced no canonical_v3 rows")
    bundle = Path(entry["path"])
    exit_iql = ExitIQLLiveInference.load(
        bundle_dir=bundle,
        variant=entry.get("active_variant") or "R_NET_REAL",
        fold_id=(entry.get("active_folds") or ["FOLD_1"])[0],
        aggregator=entry.get("active_aggregator") or "max",
        prefer_cuda=False,
    )
    row = loader._cv3.iloc[-1].copy()
    now = pd.Timestamp(loader._cv3.index[-1])
    if now.tz is None:
        now = now.tz_localize("UTC")
    trade = TradeState.open_unit_normalized_research(
        entry_ts=now - pd.Timedelta(minutes=31),
        side="long",
        entry_bid=3300.0,
        entry_ask=3300.2,
        v10_snapshot=_complete_entry_snapshot(),
        normalization_contract="unit_normalized_direction_exit_research_v1",
    )
    trade.update_bar(
        bid=3301.0,
        ask=3301.2,
        m1_close=3301.1,
        bid_high=3301.5,
        bid_low=3300.7,
        ask_high=3301.7,
        ask_low=3300.9,
    )
    bar_state = exit_iql.build_bar_state(
        trade,
        row,
        _complete_v3_state(),
        current_m1_atr_bps_override=4.2,
        now_minute=now,
    )
    adapter = exit_iql.decider.iql_adapter
    adapter.build_state_vector(bar_state)
    missing = _unresolved_required(set(adapter.required_feature_names), set(bar_state.keys()))
    assert not missing, (
        f"ACTIVE exit live-loader bar_state omits {len(missing)} REQUIRED feature(s) "
        f"from {bundle.name}: {missing[:20]}"
    )
