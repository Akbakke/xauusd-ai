from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]

RETIRED_ZERO_REACHABILITY_HELPERS = (
    "gx1/execution/logging_transport_v1.py",
    "gx1/tools/verify_freeze.py",
    "gx1/utils/dt_module.py",
    "gx1/utils/empty_trade_outcomes.py",
    "gx1/rl/exit_critic/exit_snapshot_v1.py",
    "gx1/rl/exit_critic/integrate_exit_critic_runtime_v1.py",
    "gx1/rl/__init__.py",
    "gx1/rl/dataset_horizon_constants.py",
    "gx1/rl/reward_defs.py",
    "gx1/seq/sequence_features.py",
    "gx1/features/runtime_v10_ctx.py",
    "gx1/execution/prebuilt_features_loader.py",
    "gx1/regime/__init__.py",
    "gx1/tools/__init__.py",
    "gx1/tools/analysis/__init__.py",
    "gx1/tools/debug/__init__.py",
    "gx1/tuning/__init__.py",
    "gx1/execution/broker_client.py",
    "gx1/features/basic_v1_htf_timing_patch.py",
    "gx1/features/htf_align_state.py",
    "gx1/features/rolling_logger.py",
    "gx1/utils/canonical_prebuilt_rebuilder.py",
    "gx1/utils/perf_summary_io.py",
    "gx1/utils/replay_mode.py",
    "gx1/utils/run_identity.py",
    "gx1/utils/ts_utils.py",
    "gx1/portfolio/circuit_breaker_v1.py",
    "gx1/portfolio/__init__.py",
    "tests/test_circuit_breaker_parity.py",
    "gx1/execution/tests/test_entry_safety_invariant.py",
    "gx1/tests/test_checkpoint_guardrail.py",
    "gx1/execution/exit_logs.py",
    "gx1/tests/test_exits_jsonl_created.py",
    "tests/test_go_practice_live_env_validation.sh",
    "Makefile",
    "scripts/run_manifest.py",
    ".claude/skills/run-experiment/SKILL.md",
    ".github/workflows/gx1-ci.yml",
    "backup-exclude",
    "gx1/reports",
)

RETIRED_ZERO_REACHABILITY_MODULES = (
    "gx1.execution.logging_transport_v1",
    "gx1.tools.verify_freeze",
    "gx1.utils.dt_module",
    "gx1.utils.empty_trade_outcomes",
    "gx1.seq.sequence_features",
    "gx1.features.runtime_v10_ctx",
    "gx1.execution.prebuilt_features_loader",
    "gx1.regime",
    "gx1.execution.broker_client",
    "gx1.features.basic_v1_htf_timing_patch",
    "gx1.features.htf_align_state",
    "gx1.features.rolling_logger",
    "gx1.utils.canonical_prebuilt_rebuilder",
    "gx1.utils.perf_summary_io",
    "gx1.utils.replay_mode",
    "gx1.utils.run_identity",
    "gx1.utils.ts_utils",
    "gx1.portfolio.circuit_breaker_v1",
    "gx1.execution.exit_logs",
    # 2026-07-17 audit wave: zero-reachability deletions (user-approved).
    "gx1.execution.telemetry",
    "gx1.execution.v12_live_features",
    "gx1.features.feature_manifest",
    "gx1.utils.pnl",
    "gx1.contracts.signal_bridge_v2",
    "gx1.runtime.column_collision_guard",
    # Leftover empty package dirs removed so these fail closed as ModuleNotFound
    # instead of importing as silent empty namespace packages.
    "gx1.core",
    "gx1.inference",
    "gx1.portfolio",
    "gx1.sniper",
    "gx1.runtime.overlays",
)

RETIRED_ENTRY_REPORT_FAMILIES = (
    "gx1/reports/family_signal_report_v13_refined",
    "gx1/reports/family_signal_report_v13_refined2",
    "gx1/reports/family_signal_report_v13_refined3",
    "gx1/reports/feature_carpenter_audit_v13_refined3",
    "gx1/reports/prune_ab_eval_v13_refined3",
    "gx1/reports/prune_sets_v13_refined3",
)

RETIRED_ENTRY_REPORT_FILES = (
    "gx1/reports/BASELINE_VERIFICATION_GUIDE.md",
    "gx1/reports/GO_NO_GO_TYPEERROR_FIX.md",
    "gx1/reports/TRANSFORMER_READINESS_OVERVIEW_ENTRY_V10_CTX.md",
    "gx1/reports/TRUTH_AUDIT_SIGNAL_ONLY_POST_PIVOT_V1.json",
    "gx1/reports/TRUTH_AUDIT_SIGNAL_ONLY_POST_PIVOT_V1.md",
)


def test_retired_zero_reachability_helpers_remain_physically_absent() -> None:
    present = [path for path in RETIRED_ZERO_REACHABILITY_HELPERS if (REPO / path).exists()]
    assert present == []


def test_retired_zero_reachability_helpers_are_not_importable() -> None:
    sys.path.insert(0, str(REPO))
    try:
        importable = []
        for module in RETIRED_ZERO_REACHABILITY_MODULES:
            try:
                spec = importlib.util.find_spec(module)
            except ModuleNotFoundError:
                spec = None
            if spec is not None:
                importable.append(module)
    finally:
        sys.path.remove(str(REPO))
    assert importable == []


def test_retired_entry_xgb_v13_prune_report_families_remain_absent() -> None:
    retired = RETIRED_ENTRY_REPORT_FAMILIES + RETIRED_ENTRY_REPORT_FILES
    present = [path for path in retired if (REPO / path).exists()]
    assert present == []
