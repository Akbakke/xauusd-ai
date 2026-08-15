import ast
import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gx1.models.entry_v10.direction_decision_contract import (
    MODEL_DIRECTION_ACTION_BY_INDEX,
    MODEL_DIRECTION_ACTION_ID_BY_INDEX,
    MODEL_DIRECTION_ACTION_ORDER,
    MODEL_DIRECTION_CLASS_ORDER,
    MODEL_DIRECTION_EXECUTION_SIDE_BY_INDEX,
    MODEL_DIRECTION_NAME_BY_INDEX,
    MODEL_DIRECTION_SELECTION_MODE,
    UNIFIED_EXIT_ACTION_ORDER,
    UNIFIED_EXIT_PATH_ENVELOPE_SCHEMA_VERSION,
    UNIFIED_EXIT_PATH_FEATURE_DIM,
    UNIFIED_EXIT_PATH_FEATURE_ORDER,
    canonical_closed_m1_bar,
    canonical_closed_m1_full_path_chain_sha256,
    canonical_closed_m1_path_sha256,
    canonical_unified_evidence_sha256,
    model_direction_decision_contract_metadata,
    require_model_direction_decision_contract,
    require_model_direction_operating_point,
    require_unified_exit_output,
    require_unified_exit_path_envelope,
    unified_exit_path_tensor,
    unified_entry_exit_contract_metadata,
)
from tests.unified_exit_input_support import (
    unified_exit_carry_fixture,
    unified_exit_input_fixture,
)


def test_model_direction_decision_contract_is_exact_and_rule_free() -> None:
    contract = model_direction_decision_contract_metadata()

    assert contract["selection_mode"] == MODEL_DIRECTION_SELECTION_MODE
    assert contract["action_order"] == ["LONG", "SHORT", "FLAT"]
    assert contract["target_unit"] == "raw_bps"
    assert contract["auxiliary_heads_direction_authority"] == "none"
    assert contract["runtime_direction_overrides_allowed"] is False
    assert contract["sizing_authority"] == "separate_top_level_bundle_contract"
    assert contract["runtime_direction_thresholds_allowed"] is False
    assert require_model_direction_decision_contract(
        {"direction_decision_contract": contract},
        context="unit bundle",
    ) == contract


def _unified_exit_output() -> dict[str, object]:
    input_envelope = unified_exit_input_fixture(
        entry_snapshot=_entry_snapshot(),
        exit_path_envelope=_path_envelope(),
        bundle_sha256="a" * 64,
        decision_identity="unit-exit-contract",
        side="long",
        entry_bid=100.0,
        entry_ask=100.04,
    )
    output = {
        "exit_action_q_bps": [-1.0, 2.0],
        "exit_action_valid_mask": [True, True],
        "exit_action_index": 1,
        "action": "EXIT_NOW",
        "decision_source": "unified_model",
        "exit_input_envelope": input_envelope,
        "exit_incremental_carry_envelope": unified_exit_carry_fixture(
            input_envelope=input_envelope,
            exit_path_envelope=_path_envelope(),
        ),
        "bundle_sha256": "a" * 64,
        "entry_snapshot_sha256": canonical_unified_evidence_sha256(
            _entry_snapshot()
        ),
        "exit_path_envelope_sha256": canonical_unified_evidence_sha256(
            _path_envelope()
        ),
        "exit_input_envelope_sha256": input_envelope[
            "input_envelope_sha256"
        ],
    }
    output["output_evidence_sha256"] = canonical_unified_evidence_sha256(
        output
    )
    return output


def _entry_snapshot() -> dict[str, object]:
    return {
        "schema_version": "entry-test-v2",
        "decision_ts": "2026-07-17T11:55:00+00:00",
        "model_direction_index": 0,
        "model_direction": "LONG",
        "entry_decision_representation": [
            float(index - 64) / 64.0 for index in range(128)
        ],
    }


def _path_envelope() -> dict[str, object]:
    rows = [_closed_m1_row("2026-07-17T12:00:00Z")]
    return require_unified_exit_path_envelope(
        {
            "schema_version": UNIFIED_EXIT_PATH_ENVELOPE_SCHEMA_VERSION,
            "entry_fill_ts": "2026-07-17T12:00:00+00:00",
            "first_full_m1_bar_ts": "2026-07-17T12:00:00+00:00",
            "last_closed_m1_bar_ts": "2026-07-17T12:00:00+00:00",
            "bars_in_trade": 1,
            "retained_path_length": 1,
            "path_rows": rows,
            "path_rows_sha256": canonical_closed_m1_path_sha256(rows),
            "full_path_chain_sha256": (
                canonical_closed_m1_full_path_chain_sha256(rows)
            ),
        },
        context="UNIT_EXIT_PATH",
    )


def _validate_exit(output: dict[str, object]) -> dict[str, object]:
    return require_unified_exit_output(
        output,
        context="UNIT_EXIT",
        expected_bundle_sha256="a" * 64,
        entry_snapshot=_entry_snapshot(),
        exit_path_envelope=_path_envelope(),
        exit_input_envelope=output["exit_input_envelope"],
    )


def test_unified_entry_exit_contract_has_one_rule_free_owner() -> None:
    contract = unified_entry_exit_contract_metadata()

    assert contract["single_model_bundle"] is True
    assert contract["shared_feature_encoder"] is True
    assert contract["exit_action_order"] == list(UNIFIED_EXIT_ACTION_ORDER)
    assert contract["exit_decision"] == (
        "unique_argmax(exit_action_q_bps over valid actions)"
    )
    assert contract["external_decision_models_allowed"] is False
    assert contract["runtime_entry_overrides_allowed"] is False
    assert contract["runtime_exit_overrides_allowed"] is False
    assert contract["exit_path_feature_order"] == list(
        UNIFIED_EXIT_PATH_FEATURE_ORDER
    )
    assert contract["exit_path_feature_dim"] == UNIFIED_EXIT_PATH_FEATURE_DIM
    assert contract["exit_frozen_entry_surface"] == (
        "entry_decision_representation"
    )


def _closed_m1_row(
    time: str,
    *,
    shift: float = 0.0,
) -> dict[str, object]:
    bid_open = 100.00 + shift
    bid_close = 100.02 + shift
    ask_open = 100.04 + shift
    ask_close = 100.06 + shift
    return canonical_closed_m1_bar(
        m1_bar_ts=pd.Timestamp(time),
        complete=True,
        source_path="/tmp/xau_m1.parquet",
        source_sha256="b" * 64,
        bid_open=bid_open,
        bid_high=bid_close + 0.02,
        bid_low=bid_open - 0.02,
        bid_close=bid_close,
        ask_open=ask_open,
        ask_high=ask_close + 0.02,
        ask_low=ask_open - 0.02,
        ask_close=ask_close,
        mid_open=(bid_open + ask_open) / 2.0,
        mid_high=(bid_close + ask_close) / 2.0 + 0.02,
        mid_low=(bid_open + ask_open) / 2.0 - 0.02,
        mid_close=(bid_close + ask_close) / 2.0,
        volume=100,
    )


def test_unified_exit_path_tensor_preserves_literal_mba_prefix_without_side_rule() -> None:
    rows = [
        _closed_m1_row("2026-07-29T12:00:00Z"),
        _closed_m1_row("2026-07-29T12:01:00Z", shift=0.05),
    ]
    tensor = unified_exit_path_tensor(
        path_rows=rows,
        bars_in_trade=2,
        entry_bid=99.98,
        entry_ask=100.02,
    )

    assert tensor.shape == (2, UNIFIED_EXIT_PATH_FEATURE_DIM)
    assert tensor.dtype == np.float32
    assert np.isfinite(tensor).all()
    assert tensor[0, 0] == pytest.approx(0.0)
    assert tensor[0, -3] == pytest.approx(np.log1p(100))
    assert tensor[:, -2].tolist() == pytest.approx(
        [np.log1p(1), np.log1p(2)]
    )
    assert tensor[0, -1] == pytest.approx(4.0)

    gapped = [rows[0], _closed_m1_row("2026-07-29T12:02:00Z")]
    gapped_tensor = unified_exit_path_tensor(
        path_rows=gapped,
        bars_in_trade=2,
        entry_bid=99.98,
        entry_ask=100.02,
    )
    assert gapped_tensor.shape == (2, UNIFIED_EXIT_PATH_FEATURE_DIM)

    reversed_rows = [gapped[1], gapped[0]]
    with pytest.raises(ValueError, match="row clock duplicate/reversal"):
        unified_exit_path_tensor(
            path_rows=reversed_rows,
            bars_in_trade=2,
            entry_bid=99.98,
            entry_ask=100.02,
        )

    noncanonical = [dict(rows[0])]
    noncanonical[0]["time"] = "2026-07-29 12:00:00+00:00"
    with pytest.raises(ValueError, match="not canonical"):
        unified_exit_path_tensor(
            path_rows=noncanonical,
            bars_in_trade=1,
            entry_bid=99.98,
            entry_ask=100.02,
        )


def test_unified_exit_output_requires_raw_q_validity_and_action_parity() -> None:
    output = _unified_exit_output()
    assert _validate_exit(output) == output

    for key, replacement in (
        ("exit_action_index", 0),
        ("action", "HOLD"),
        ("entry_snapshot_sha256", "not-a-hash"),
    ):
        malformed = dict(output)
        malformed[key] = replacement
        with pytest.raises(RuntimeError):
            _validate_exit(malformed)

    tied = dict(output)
    tied["exit_action_q_bps"] = [1.0, 1.0]
    with pytest.raises(RuntimeError, match="tied valid Exit Q"):
        _validate_exit(tied)

    terminal = dict(output)
    terminal["exit_action_q_bps"] = [1.0, 1.0]
    terminal["exit_action_valid_mask"] = [False, True]
    terminal["output_evidence_sha256"] = canonical_unified_evidence_sha256(
        {
            key: value
            for key, value in terminal.items()
            if key != "output_evidence_sha256"
        }
    )
    assert _validate_exit(terminal)["action"] == "EXIT_NOW"

    unexpected = dict(output)
    unexpected["exit_action_probs"] = [0.0, 1.0]
    with pytest.raises(RuntimeError, match="exact schema mismatch"):
        _validate_exit(unexpected)


def test_direction_class_and_action_layout_have_one_active_owner() -> None:
    from gx1.contracts.entry_fitted_q_v1 import (
        ENTRY_FITTED_Q_ACTION_ORDER,
    )
    from gx1.contracts.entry_model_native_runtime_evidence_v1 import (
        MODEL_DIRECTION_NAMES,
    )

    assert MODEL_DIRECTION_CLASS_ORDER is ENTRY_FITTED_Q_ACTION_ORDER
    assert tuple(MODEL_DIRECTION_NAMES) == ENTRY_FITTED_Q_ACTION_ORDER
    assert tuple(MODEL_DIRECTION_NAME_BY_INDEX.values()) == ENTRY_FITTED_Q_ACTION_ORDER
    assert tuple(MODEL_DIRECTION_ACTION_BY_INDEX.values()) == (
        MODEL_DIRECTION_ACTION_ORDER
    )
    assert tuple(MODEL_DIRECTION_ACTION_ID_BY_INDEX.values()) == (1, 2, 0)
    assert MODEL_DIRECTION_EXECUTION_SIDE_BY_INDEX == {
        0: "long",
        1: "short",
        2: None,
    }


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
    assert '"entry_action_q_loss": "masked_raw_bps_mean_squared_error"' in source
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


def _function_ast(path: Path, function_name: str, *, class_name: str | None = None) -> ast.AST:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    owner: ast.AST = tree
    if class_name is not None:
        owner = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == class_name
        )
    return next(
        node
        for node in owner.body  # type: ignore[attr-defined]
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == function_name
    )


def _decision_assignment_line(function: ast.AST) -> int:
    return min(
        node.lineno
        for node in ast.walk(function)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "decision" for target in node.targets)
    )


def test_downstream_entry_branches_cannot_promote_auxiliary_evidence_to_authority() -> None:
    root = Path(__file__).resolve().parents[1]
    functions = (
        _function_ast(
            root / "gx1/execution/v12_pipeline.py",
            "make_entry_decision",
            class_name="V12Pipeline",
        ),
        _function_ast(root / "gx1/execution/v12_paper_runner.py", "main"),
    )
    forbidden_branch_terms = (
        "edge_score",
        "selection_score",
        "p_long",
        "p_short",
        "p_flat",
        "p_trade",
        "tradable",
        "bad_path",
        "path_quality",
        "utility",
        "trend",
        "session",
        "specialist",
        "rail",
        "confidence",
        "threshold",
        "veto",
        "flip",
    )

    branch_expressions = [
        ast.unparse(node.test).lower()
        for function in functions
        for node in ast.walk(function)
        if isinstance(node, (ast.If, ast.IfExp, ast.While))
        and node.lineno >= _decision_assignment_line(function)
    ]
    offending = {
        term: expression
        for term in forbidden_branch_terms
        for expression in branch_expressions
        if term in expression
    }

    assert offending == {}


def test_downstream_entry_chain_never_assigns_a_replacement_direction_or_action() -> None:
    root = Path(__file__).resolve().parents[1]
    functions = (
        _function_ast(
            root / "gx1/execution/v12_pipeline.py",
            "make_entry_decision",
            class_name="V12Pipeline",
        ),
        _function_ast(root / "gx1/execution/v12_paper_runner.py", "main"),
    )
    authority_fields = {
        "action",
        "action_id",
        "model_direction",
        "model_direction_index",
        "direction_logits",
        "direction_probs",
    }
    replacements: list[tuple[int, str]] = []
    mapping_mutations: list[str] = []
    for function in functions:
        decision_line = _decision_assignment_line(function)
        for node in ast.walk(function):
            if (
                isinstance(node, ast.Call)
                and node.lineno >= decision_line
                and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "decision"
                and node.func.attr in {"update", "setdefault"}
            ):
                mapping_mutations.append(ast.unparse(node))
            if not isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                continue
            if node.lineno < decision_line:
                continue
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if not isinstance(target, ast.Subscript):
                    continue
                if not isinstance(target.value, ast.Name) or target.value.id != "decision":
                    continue
                key = target.slice.value if isinstance(target.slice, ast.Constant) else None
                if key in authority_fields:
                    replacements.append((node.lineno, str(key)))

    assert replacements == []
    assert mapping_mutations == ["decision.update(latency_fields)"]


def test_active_pipeline_never_synthesizes_flat_for_unavailable_model() -> None:
    source = (
        Path(__file__).resolve().parents[1] / "gx1/execution/v12_pipeline.py"
    ).read_text(encoding="utf-8")
    start = source.index("    def make_entry_decision(")
    end = source.index("    def make_exit_decision", start)
    active_entry = source[start:end]

    assert "_SKIP_BASE" not in active_entry
    assert '"advantage_over_skip"' not in active_entry
    assert 'decision["external_tree_sidecar"]' not in active_entry
    assert "EntryDecisionUnavailable" in active_entry
    assert "Operational no-data/stale/cadence states raise" in active_entry


def test_entry_unavailable_event_preserves_structured_evidence() -> None:
    from gx1.execution.v12_pipeline import EntryDecisionUnavailable

    exc = EntryDecisionUnavailable("entry_signal_stale", latency_sec=120.0)

    assert exc.reason == "entry_signal_stale"
    assert exc.evidence == {"latency_sec": 120.0}


def test_runner_has_no_hand_written_spread_or_session_entry_gate() -> None:
    source = (
        Path(__file__).resolve().parents[1] / "gx1/execution/v12_paper_runner.py"
    ).read_text(encoding="utf-8")

    assert "def can_trade_now(" not in source
    assert "spread_too_wide" not in source
    assert "BLOCKED_BY_EXECUTION_SPREAD" not in source
    assert "--max-spread-bps" not in source
    assert "literal_spread_supplied_to_model" in source


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


def test_model_native_context_gap_has_no_partial_mtf_splice_surface() -> None:
    from gx1.execution import v12_model_native_state_live as state_live
    from gx1.execution import v12_smart_entry_live as smart_live

    assert not hasattr(state_live, "append_multi_tf_incremental")
    assert not hasattr(state_live, "MODEL_NATIVE_MTF_SPLICE_TFS")
    assert smart_live.SMART_CTX_MAX_STALENESS_M5 == 0
    effective_source = inspect.getsource(
        smart_live.SmartEntryLiveInference._effective_context
    )
    assert "raise SmartContextStaleError" in effective_source
    assert "append_multi_tf_incremental" not in effective_source
