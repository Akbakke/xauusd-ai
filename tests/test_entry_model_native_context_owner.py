from __future__ import annotations

import ast
from pathlib import Path

from gx1.contracts import signal_bridge_v3
from gx1.contracts.entry_model_native_signal_v1 import (
    MODEL_NATIVE_BASE_FIELDS,
    MODEL_NATIVE_CTX_CAT_FIELDS,
    MODEL_NATIVE_CTX_CONT_DIP_STRUCT_FIELDS,
    MODEL_NATIVE_CTX_CONT_FIELDS,
    MODEL_NATIVE_CTX_CONT_GROUP_A_FIELDS,
)


ROOT = Path(__file__).resolve().parents[1]
LEGACY_MODULES = {
    "gx1.contracts.signal_bridge_v1",
    "gx1.contracts.signal_bridge_v3",
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


def test_retained_v3_bridge_reexports_entry_owned_market_fields_without_drift() -> None:
    assert tuple(signal_bridge_v3.PER_BAR_PRICE_STATE_FIELDS_V3) == (
        MODEL_NATIVE_BASE_FIELDS
    )
    assert tuple(signal_bridge_v3.ORDERED_CTX_CONT_NAMES_V3) == (
        MODEL_NATIVE_CTX_CONT_FIELDS
    )
    assert tuple(signal_bridge_v3.ORDERED_CTX_CAT_NAMES_V3) == (
        MODEL_NATIVE_CTX_CAT_FIELDS
    )
    assert tuple(signal_bridge_v3.ORDERED_CTX_CONT_GROUP_A_PARITY) == (
        MODEL_NATIVE_CTX_CONT_GROUP_A_FIELDS
    )
    assert tuple(signal_bridge_v3.ORDERED_CTX_CONT_DIP_STRUCT) == (
        MODEL_NATIVE_CTX_CONT_DIP_STRUCT_FIELDS
    )

    # The retained module still owns a real seven-field XGB/Exit-era bridge;
    # it is not an empty compatibility shell and is not an Entry authority.
    assert signal_bridge_v3.BRIDGE_DIM_V3 == 7
    assert signal_bridge_v3.SEQ_SIGNAL_DIM_V3 == 41
    assert callable(signal_bridge_v3.validate_seq_signal)
