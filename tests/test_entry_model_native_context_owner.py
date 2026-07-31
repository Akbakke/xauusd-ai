from __future__ import annotations

import ast
from pathlib import Path

from gx1.models.entry_v10.direction_decision_contract import (
    UNIFIED_EXIT_ACTION_ORDER,
    unified_entry_exit_contract_metadata,
)


ROOT = Path(__file__).resolve().parents[1]
LEGACY_MODULES = {
    "gx1.contracts.signal_bridge_v1",
    "gx1.contracts.signal_bridge_v3",
    "gx1.execution.v12_v3_live",
    "gx1.execution.v12_exit_iql_live",
}


def _active_entry_sources() -> tuple[Path, ...]:
    paths = {
        ROOT / "gx1/contracts/entry_model_native_signal_v1.py",
        ROOT / "gx1/scripts/add_ctx_cont_columns_to_prebuilt.py",
        ROOT / "gx1/scripts/augment_forward_outcome_v2.py",
        ROOT / "gx1/execution/v12_model_native_state_live.py",
        ROOT / "gx1/execution/v12_smart_entry_live.py",
        ROOT / "gx1/execution/v12_state_from_prebuilt.py",
        ROOT / "gx1/execution/model_native_entry_replay_v1.py",
    }
    paths.update((ROOT / "gx1/models/entry_v10").glob("*.py"))
    paths.update((ROOT / "gx1/features").glob("entry_*.py"))
    paths.update((ROOT / "gx1/scripts").glob("*entry*.py"))
    paths.update((ROOT / "gx1/audit").glob("entry_*.py"))
    return tuple(sorted(path for path in paths if path.is_file()))


def test_active_entry_python_has_zero_legacy_signal_bridge_imports() -> None:
    offenders: list[str] = []
    for path in _active_entry_sources():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module in LEGACY_MODULES:
                offenders.append(f"{path.relative_to(ROOT)}:{node.lineno}:{node.module}")
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in LEGACY_MODULES:
                        offenders.append(
                            f"{path.relative_to(ROOT)}:{node.lineno}:{alias.name}"
                        )
    assert offenders == []


def test_retired_external_decision_owners_are_physically_absent() -> None:
    forbidden_paths = (
        ROOT / "gx1" / ("x" + "gb"),
        ROOT / "gx1/contracts/signal_bridge_v1.py",
        ROOT / "gx1/contracts/signal_bridge_v3.py",
        ROOT / "gx1/execution/v12_v3_live.py",
        ROOT / "gx1/execution/v12_exit_iql_live.py",
        ROOT / "gx1/execution" / ("v12_" + "x" + "gb" + "_live.py"),
    )
    assert [str(path.relative_to(ROOT)) for path in forbidden_paths if path.exists()] == []


def test_unified_contract_forbids_external_decision_authority() -> None:
    contract = unified_entry_exit_contract_metadata()

    assert contract["single_model_bundle"] is True
    assert contract["shared_feature_encoder"] is True
    assert contract["exit_bound_to_entry_snapshot"] is True
    assert contract["external_decision_models_allowed"] is False
    assert contract["runtime_entry_overrides_allowed"] is False
    assert contract["runtime_exit_overrides_allowed"] is False
    assert tuple(contract["exit_action_order"]) == UNIFIED_EXIT_ACTION_ORDER
