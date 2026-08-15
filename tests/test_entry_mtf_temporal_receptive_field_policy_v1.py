from __future__ import annotations

import copy
import math
from pathlib import Path

import pytest

from gx1.contracts.entry_mtf_temporal_receptive_field_policy_v1 import (
    PROPOSED_MINIMUM_WINDOW_BARS,
    require_temporal_receptive_field_policy,
    temporal_receptive_field_policy,
)
from gx1.scripts.audit_entry_mtf_temporal_receptive_field_v1 import (
    audit_temporal_receptive_field_migration_surface,
)
from gx1.scripts.benchmark_entry_mtf_temporal_receptive_field_v1 import (
    benchmark_temporal_profile,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def test_policy_binds_current_and_proposed_nominal_coverage() -> None:
    policy = require_temporal_receptive_field_policy(
        temporal_receptive_field_policy()
    )
    assert policy["current_window_bars"] == {
        "M5": 16,
        "M15": 64,
        "H1": 96,
        "H4": 96,
        "D1": 252,
    }
    assert policy["proposed_minimum_window_bars"] == {
        "M5": 96,
        "M15": 96,
        "H1": 168,
        "H4": 180,
        "D1": 252,
    }
    assert policy["nominal_coverage_seconds"] == {
        "M5": 8 * 60 * 60,
        "M15": 24 * 60 * 60,
        "H1": 7 * 24 * 60 * 60,
        "H4": 30 * 24 * 60 * 60,
        "D1": None,
    }
    assert policy["nominal_coverage_label"]["D1"] == (
        "252_daily_observations_approximately_one_trading_year"
    )
    assert policy["integration"]["integrated"] is False
    assert policy["integration"]["no_production_default_changes_in_this_wave"] is True


def test_policy_is_capacity_not_decision_or_forced_attention() -> None:
    semantics = temporal_receptive_field_policy()["capacity_semantics"]
    assert semantics == {
        "input_tensor_capacity_only": True,
        "decision_threshold": None,
        "trade_direction_authority": False,
        "entry_or_exit_timing_rule": False,
        "label_or_target_horizon": False,
        "forced_attention_or_usefulness": False,
        "model_learns_history_usefulness_from_train": True,
        "validation_or_test_selection_authority": False,
    }


def test_policy_hash_and_owner_observation_fail_closed() -> None:
    policy = temporal_receptive_field_policy()
    mutated = copy.deepcopy(policy)
    mutated["proposed_minimum_window_bars"]["H4"] = 179
    with pytest.raises(RuntimeError, match="POLICY_INVALID"):
        require_temporal_receptive_field_policy(mutated)
    mutated = copy.deepcopy(policy)
    mutated["contract_sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="POLICY_INVALID"):
        require_temporal_receptive_field_policy(mutated)


def test_migration_audit_is_deterministic_and_covers_every_owner_layer() -> None:
    first = audit_temporal_receptive_field_migration_surface(REPOSITORY_ROOT)
    second = audit_temporal_receptive_field_migration_surface(REPOSITORY_ROOT)
    assert first == second
    assert first["integrated"] is False
    assert first["matched_file_count"] >= 40
    assert first["hit_counts_by_kind"]["legacy_exact_literal"] >= 25
    expected_paths = {
        "gx1/contracts/entry_exit_production_architecture_v1.py",
        "gx1/contracts/entry_model_native_input_normalization_v1.py",
        "gx1/contracts/entry_model_native_train_launch_v1.py",
        "gx1/models/entry_v10/entry_v10_ctx_hybrid_transformer.py",
        "gx1/models/entry_v10/entry_v10_ctx_train_v3.py",
        "gx1/models/entry_v10/entry_v10_bundle.py",
        "gx1/scripts/prebuild_multi_tf_cache_v4.py",
        "gx1/execution/v12_smart_entry_live.py",
        "scripts/run_entry_model_native_seq513_train.sh",
    }
    assert expected_paths.issubset(first["matched_files"])
    observed_literals = {
        token
        for row in first["migration_hits"]
        for token in row["legacy_literal_tokens"]
    }
    assert observed_literals == {
        "legacy_literal_m5_16",
        "legacy_literal_m15_64",
        "legacy_literal_h1_96",
        "legacy_literal_h4_96",
        "legacy_literal_d1_252",
    }


def test_policy_has_no_production_integration_imports() -> None:
    import_token = "entry_mtf_temporal_receptive_field_policy_v1"
    allowed = {
        "gx1/contracts/entry_mtf_temporal_receptive_field_policy_v1.py",
        "gx1/scripts/audit_entry_mtf_temporal_receptive_field_v1.py",
        "gx1/scripts/benchmark_entry_mtf_temporal_receptive_field_v1.py",
        "tests/test_entry_mtf_temporal_receptive_field_policy_v1.py",
    }
    observed: set[str] = set()
    for source_root in ("gx1", "scripts", "tests"):
        for path in (REPOSITORY_ROOT / source_root).rglob("*"):
            if path.is_file() and path.suffix in {".py", ".sh"}:
                if import_token in path.read_text(encoding="utf-8"):
                    observed.add(path.relative_to(REPOSITORY_ROOT).as_posix())
    assert observed == allowed


def test_synthetic_benchmark_exposes_exact_workload_growth_and_training() -> None:
    common = {
        "batch_size": 1,
        "warmup_iterations": 0,
        "measured_iterations": 1,
        "d_model": 4,
        "n_heads": 1,
        "layers": 1,
    }
    current = benchmark_temporal_profile(
        profile="current", phase="forward", **common
    )
    proposed = benchmark_temporal_profile(
        profile="proposed", phase="forward", **common
    )
    training = benchmark_temporal_profile(
        profile="proposed", phase="training", **common
    )
    assert current["windows"] != proposed["windows"]
    assert proposed["windows"] == PROPOSED_MINIMUM_WINDOW_BARS
    for field in (
        "family_token_cells_per_iteration",
        "attention_score_cells_per_iteration",
    ):
        assert proposed["workload"][field] > current["workload"][field]
    assert training["phase"] == "training"
    for report in (current, proposed, training):
        measurement = report["measurement"]
        assert measurement["wall_seconds_mean"] > 0.0
        assert measurement["peak_rss_kib"] >= measurement["baseline_rss_kib"]
        assert math.isfinite(measurement["terminal_scalar"])
        assert report["authority"]["architecture_selection_authority"] is False
