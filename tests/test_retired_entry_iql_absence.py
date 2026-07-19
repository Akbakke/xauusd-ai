from __future__ import annotations

import ast
import subprocess
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]

RETIRED_ENTRY_IQL_EXECUTABLES = (
    "gx1/execution/v12_entry_iql_live.py",
    "gx1/execution/v12_v10_live.py",
    "gx1/runtime/entry_iql_v2_adapter.py",
    "gx1/research/step1_roundwall_ab.py",
    "gx1/scripts/audit_entry_iql_replay_slices_v1.py",
    "gx1/scripts/build_online_replay_buffer.py",
    "gx1/scripts/entry_iql_gpu_core_v1.py",
    "gx1/scripts/entry_iql_multi_head_gpu_core_v1.py",
    "gx1/scripts/materialize_build_entry_iql_v1.py",
    "gx1/scripts/materialize_build_entry_iql_v2.py",
    "gx1/scripts/materialize_entry_foundation_smart_selector_readiness_v1.py",
    "gx1/scripts/materialize_entry_iql_distillation_contract_v1.py",
    "gx1/scripts/materialize_entry_iql_replay_evidence_v1.py",
    "gx1/scripts/materialize_entry_iql_student_trade_log_v1.py",
    "gx1/scripts/nightly_op_comparison.py",
    "gx1/scripts/online_iql_warmstart.py",
    "gx1/scripts/v12_phase1_entry_iql_inference.py",
    "gx1/scripts/verify_entry_iql_replay_comparison_v1.py",
    "scripts/gx1_candidate_gate.sh",
    "scripts/gx1_volbal_baseline_oneshot.sh",
    "scripts/run_entry_foundation_iql_distill.sh",
)


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
            modules.update(f"{node.module}.{alias.name}" for alias in node.names)
    return modules


def test_retired_entry_iql_executables_are_physically_absent() -> None:
    present = [path for path in RETIRED_ENTRY_IQL_EXECUTABLES if (REPO / path).exists()]
    assert present == []


def test_retired_41_dim_v10_bridge_wrapper_is_physically_absent() -> None:
    assert not (REPO / "gx1/execution/v12_v10_live.py").exists()


def test_live_entry_stack_cannot_import_retired_entry_iql() -> None:
    forbidden = {
        "gx1.execution.v12_entry_iql_live",
        "gx1.execution.v12_v10_live",
        "gx1.runtime.entry_iql_v2_adapter",
    }
    live_modules = (
        REPO / "gx1/execution/v12_pipeline.py",
        REPO / "gx1/execution/v12_paper_runner.py",
        REPO / "gx1/execution/v12_smart_entry_live.py",
        REPO / "gx1/execution/v12_model_native_state_live.py",
        REPO / "gx1/execution/v12_exit_iql_live.py",
    )
    for module in live_modules:
        assert _imported_modules(module).isdisjoint(forbidden), module


def test_control_surface_has_no_entry_iql_reopening_route() -> None:
    control = REPO / "scripts/entry_next_edge_control.sh"
    text = control.read_text(encoding="utf-8")
    forbidden = (
        "iql-distill)",
        "iql-student-trade-log)",
        "iql-replay-evidence)",
        "iql-compare)",
        "iql-slice-audit)",
        "foundation-smart-selector-readiness)",
        "materialize_entry_iql_",
        "verify_entry_iql_",
        "audit_entry_iql_",
    )
    assert all(token not in text for token in forbidden)

    result = subprocess.run(
        ["bash", str(control), "iql-distill"],
        cwd=REPO,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert result.returncode == 2
    assert "unknown command: iql-distill" in result.stderr


def test_exit_iql_shared_core_and_adapter_remain_present() -> None:
    shared_core = REPO / "gx1/scripts/exit_iql_multi_head_gpu_core_v1.py"
    exit_adapter = REPO / "gx1/runtime/exit_iql_v2_adapter.py"
    assert shared_core.is_file()
    assert exit_adapter.is_file()
    assert "gx1.scripts.exit_iql_multi_head_gpu_core_v1" in _imported_modules(exit_adapter)
