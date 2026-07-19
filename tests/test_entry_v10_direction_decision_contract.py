import inspect
from pathlib import Path

import pytest

from gx1.models.entry_v10.direction_decision_contract import (
    MODEL_DIRECTION_SELECTION_MODE,
    model_direction_decision_contract_metadata,
    require_model_direction_decision_contract,
    require_model_direction_operating_point,
)


def test_model_direction_decision_contract_is_exact_and_rule_free() -> None:
    contract = model_direction_decision_contract_metadata()

    assert contract["selection_mode"] == MODEL_DIRECTION_SELECTION_MODE
    assert contract["direction_class_order"] == ["LONG", "SHORT", "FLAT"]
    assert contract["public_trade_flat_class_order"] == ["TRADE", "FLAT"]
    assert contract["auxiliary_heads_direction_authority"] == "none"
    assert contract["runtime_direction_overrides_allowed"] is False
    assert contract["sizing_authority"] == "separate_top_level_bundle_contract"
    assert contract["runtime_direction_thresholds_allowed"] is False
    assert require_model_direction_decision_contract(
        {"direction_decision_contract": contract},
        context="unit bundle",
    ) == contract


@pytest.mark.parametrize(
    "metadata",
    [
        {},
        {"direction_decision_contract": {}},
        {
            "direction_decision_contract": {
                **model_direction_decision_contract_metadata(),
                "selection_mode": "expected_utility",
            }
        },
        {
            "direction_decision_contract": {
                **model_direction_decision_contract_metadata(),
                "soft_compatibility": True,
            }
        },
    ],
)
def test_model_direction_decision_contract_rejects_missing_stale_or_soft_metadata(
    metadata: dict,
) -> None:
    with pytest.raises(RuntimeError, match="direction_decision_contract"):
        require_model_direction_decision_contract(metadata, context="unit bundle")


def test_direction_contract_rejects_nested_sizing_authority_reintroduction() -> None:
    contract = model_direction_decision_contract_metadata()
    contract["sizing_authority_contract"] = {"applied_size_multiplier": 1.0}

    with pytest.raises(RuntimeError, match="direction_decision_contract mismatch"):
        require_model_direction_decision_contract(
            {"direction_decision_contract": contract},
            context="unit bundle",
        )


def test_trainer_writes_contract_and_no_longer_advertises_utility_selection() -> None:
    from gx1.models.entry_v10 import entry_v10_ctx_train_v3 as trainer

    source = Path(trainer.__file__).read_text(encoding="utf-8")
    assert "direction_decision_contract = model_direction_decision_contract_metadata()" in source
    assert source.count('"direction_decision_contract": direction_decision_contract') >= 2
    assert "_direction_decision_contract_export_failures(lock, meta)" in source
    assert '"selection_score": MODEL_DIRECTION_SELECTION_MODE' in source
    assert '"selection_score": "expected_utility_side"' not in source


def test_model_direction_operating_point_is_exact_and_rule_free() -> None:
    operating_point = {
        "selection_score": MODEL_DIRECTION_SELECTION_MODE,
        "max_trades": 3,
    }

    assert require_model_direction_operating_point(
        operating_point,
        context="unit launch",
    ) == operating_point


@pytest.mark.parametrize(
    "operating_point",
    [
        {},
        {"selection_score": MODEL_DIRECTION_SELECTION_MODE},
        {"selection_score": MODEL_DIRECTION_SELECTION_MODE, "max_trades": 0},
        {"selection_score": "edge_score", "max_trades": 3},
        {
            "selection_score": MODEL_DIRECTION_SELECTION_MODE,
            "max_trades": 3,
            "edge_score_threshold": 0.1,
        },
        {
            "selection_score": MODEL_DIRECTION_SELECTION_MODE,
            "max_trades": 3,
            "sessions": ["US"],
        },
    ],
)
def test_model_direction_operating_point_rejects_missing_stale_or_soft_keys(
    operating_point: dict,
) -> None:
    with pytest.raises(RuntimeError, match="operating_point"):
        require_model_direction_operating_point(
            operating_point,
            context="unit launch",
        )


def test_active_pipeline_has_no_post_model_direction_rewrite() -> None:
    source = (
        Path(__file__).resolve().parents[1] / "gx1/execution/v12_pipeline.py"
    ).read_text(encoding="utf-8")
    start = source.index("decision = self.smart_entry.decide")
    end = source.index("    def make_exit_decision", start)
    active_entry_tail = source[start:end]

    assert "CLUSTER1" not in active_entry_tail
    assert "record_entry_for_cluster" not in active_entry_tail
    assert 'decision.update({"action"' not in active_entry_tail


def test_active_pipeline_never_synthesizes_flat_for_unavailable_model() -> None:
    source = (
        Path(__file__).resolve().parents[1] / "gx1/execution/v12_pipeline.py"
    ).read_text(encoding="utf-8")
    start = source.index("    def make_entry_decision(")
    end = source.index("    def make_exit_decision", start)
    active_entry = source[start:end]

    assert "_SKIP_BASE" not in active_entry
    assert '"advantage_over_skip"' not in active_entry
    assert 'decision["xgb"]' not in active_entry
    assert "EntryDecisionUnavailable" in active_entry
    assert "Operational no-data/stale/cadence states raise" in active_entry


def test_entry_unavailable_event_preserves_structured_evidence() -> None:
    from gx1.execution.v12_pipeline import EntryDecisionUnavailable

    exc = EntryDecisionUnavailable("entry_signal_stale", latency_sec=120.0)

    assert exc.reason == "entry_signal_stale"
    assert exc.evidence == {"latency_sec": 120.0}


def test_runner_market_gate_contains_no_session_direction_rule() -> None:
    source = (
        Path(__file__).resolve().parents[1] / "gx1/execution/v12_paper_runner.py"
    ).read_text(encoding="utf-8")
    start = source.index("def can_trade_now(")
    end = source.index("\n\n# ── ", start)
    market_gate = source[start:end]

    assert "get_session" not in market_gate
    assert "skip_asia" not in market_gate
    assert "session_detector" not in market_gate


def test_runner_has_no_legacy_post_model_direction_or_sizing_path() -> None:
    source = (
        Path(__file__).resolve().parents[1] / "gx1/execution/v12_paper_runner.py"
    ).read_text(encoding="utf-8")

    for retired_code in (
        "blocked_adaptive_min_adv",
        "blocked_regime_uptrend",
        "blocked_regime_downtrend",
        "blocked_low_confidence",
        "shadow_filters",
        "high_conviction_skip",
        "high_conviction_blocked",
        "def size_units(",
        "def units_from_position_size_pred(",
        "TIME_OF_DAY_EXIT",
        "expected_utility_side",
        "advantage_over_skip",
        "q_take_long",
        "q_take_short",
    ):
        assert retired_code not in source

    assert 'trade_units = args.units' not in source
    assert "apply_model_native_sizing(" in source
    assert "SIZING_UNAVAILABLE_NO_ORDER" in source
    assert 'p.add_argument("--units"' not in source
    assert 'event["order_status"] = "MODEL_DIRECTION_FLAT"' in source
    assert "runner action disagrees with model direction argmax" in source


def test_runner_rejects_presence_of_retired_entry_override_env(monkeypatch) -> None:
    from gx1.execution import v12_paper_runner as runner

    for name in runner.RETIRED_ENTRY_OVERRIDE_ENV:
        monkeypatch.delenv(name, raising=False)
    runner.assert_no_retired_entry_overrides()

    monkeypatch.setenv("GX1_SKIP_ASIA", "0")
    with pytest.raises(SystemExit, match="retired entry override"):
        runner.assert_no_retired_entry_overrides()

    monkeypatch.delenv("GX1_SKIP_ASIA")
    monkeypatch.setenv("GX1_SIZING_FUTURE_SOFT_TUNER", "0")
    with pytest.raises(SystemExit, match="GX1_SIZING_FUTURE_SOFT_TUNER"):
        runner.assert_no_retired_entry_overrides()


def test_model_native_mtf_splice_contract_has_no_runtime_tuning_surface() -> None:
    from gx1.execution import v12_model_native_state_live as state_live

    assert state_live.MODEL_NATIVE_MTF_SPLICE_TFS == ("M15", "H1")
    assert state_live.MODEL_NATIVE_MTF_SPLICE_WARMUP_M5 == 30000
    assert tuple(
        inspect.signature(state_live.append_multi_tf_incremental).parameters
    ) == ("cv3", "multi_tf")


def test_entry_latency_override_is_retired_from_runner_and_launcher(
    monkeypatch,
) -> None:
    from gx1.execution import v12_paper_runner as runner

    for name in runner.RETIRED_ENTRY_OVERRIDE_ENV:
        monkeypatch.delenv(name, raising=False)
    assert "GX1_MAX_ENTRY_DECISION_LATENCY_SEC" in runner.RETIRED_ENTRY_OVERRIDE_ENV
    assert "GX1_SMART_PARITY_GATE_MAX_AGE_HOURS" in runner.RETIRED_ENTRY_OVERRIDE_ENV
    assert (
        "GX1_SMART_PARITY_GATE_MAX_CUTOFF_LAG_HOURS"
        in runner.RETIRED_ENTRY_OVERRIDE_ENV
    )
    assert "GX1_SMART_DIRECTION_AUDIT_MAX_AGE_HOURS" in runner.RETIRED_ENTRY_OVERRIDE_ENV
    assert "GX1_SMART_CTX_MAX_STALENESS_M5" in runner.RETIRED_ENTRY_OVERRIDE_ENV
    assert "GX1_SMART_CTX_MTF_WARMUP_M5" in runner.RETIRED_ENTRY_OVERRIDE_ENV
    assert "GX1_MODEL_NATIVE_CTX_MTF_WARMUP_M5" in runner.RETIRED_ENTRY_OVERRIDE_ENV
    monkeypatch.setenv("GX1_MAX_ENTRY_DECISION_LATENCY_SEC", "-1")
    with pytest.raises(SystemExit, match="GX1_MAX_ENTRY_DECISION_LATENCY_SEC"):
        runner.assert_no_retired_entry_overrides()

    launcher = (
        Path(__file__).resolve().parents[1] / "scripts/launch_live_practice.sh"
    ).read_text(encoding="utf-8")
    assert "GX1_MAX_ENTRY_DECISION_LATENCY_SEC" not in launcher

    runner_source = (
        Path(__file__).resolve().parents[1] / "gx1/execution/v12_paper_runner.py"
    ).read_text(encoding="utf-8")
    assert "datetime.now(timezone.utc)," in runner_source
