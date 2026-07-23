from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np
import pytest

from gx1.runtime.exit_iql_v2_adapter import (
    ExitIQLV2Adapter,
    require_exit_iql_checkpoint_binding,
)
from gx1.runtime.exit_decider_v12_adapter import ExitDeciderV12Adapter
from gx1.scripts import exit_iql_artifact_primitives_v1 as primitives
from gx1_guards import artifacts


FEATURE_NAMES = ["trend_signal_v1", "support_distance_v1"]


def _feature_hash(feature_names: list[str] | None = None) -> str:
    return artifacts.exit_iql_ordered_feature_names_sha256(
        feature_names or FEATURE_NAMES
    )


def _production_summary(**updates: object) -> dict:
    summary = {
        "research_only_v1": False,
        "iql_production_allowed_v1": True,
        "n_features_v1": len(FEATURE_NAMES),
        "feature_names_v1": list(FEATURE_NAMES),
        "feature_names_sha256_v1": _feature_hash(),
    }
    summary.update(updates)
    return summary


def _checkpoint(**updates: object) -> dict:
    checkpoint = {
        "schema_v1": "MULTI_HEAD_EXIT_IQL_V2_CHECKPOINT",
        "variant": "R_NET_REAL",
        "fold_id": "FOLD_2",
        "state_dim": len(FEATURE_NAMES),
        "feature_names_v1": list(FEATURE_NAMES),
        "feature_names_sha256_v1": _feature_hash(),
        "feature_means": [0.0, 1.0],
        "feature_stds": [1.0, 0.000001],
    }
    checkpoint.update(updates)
    return checkpoint


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"research_only_v1": True}, "research_only_v1"),
        ({"research_only_v1": None}, "research_only_v1"),
        ({"iql_production_allowed_v1": False}, "iql_production_allowed_v1"),
        ({"iql_production_allowed_v1": 1}, "iql_production_allowed_v1"),
        ({"feature_names_sha256_v1": None}, "exact SHA-256"),
        ({"feature_names_sha256_v1": "0" * 64}, "SHA-256 mismatch"),
        (
            {
                "feature_names_v1": ["trend_signal_v1", "trend_signal_v1"],
                "n_features_v1": 2,
            },
            "duplicate",
        ),
    ],
)
def test_exit_iql_summary_contract_fails_closed(
    updates: dict,
    message: str,
) -> None:
    with pytest.raises(artifacts.ArtifactGuardError, match=message):
        artifacts.require_exit_iql_summary_contract(
            _production_summary(**updates),
            context="UNIT_EXIT_IQL",
        )


def test_exit_iql_summary_contract_accepts_only_canonical_ordered_hash() -> None:
    feature_names, feature_hash = artifacts.require_exit_iql_summary_contract(
        _production_summary(),
        context="UNIT_EXIT_IQL",
    )

    assert feature_names == FEATURE_NAMES
    assert feature_hash == _feature_hash()
    assert (
        primitives.ordered_feature_names_sha256(FEATURE_NAMES)
        == feature_hash
    )


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"feature_names_v1": None}, "lacks ordered"),
        (
            {"feature_names_v1": list(reversed(FEATURE_NAMES))},
            "differs from summary order",
        ),
        ({"feature_names_sha256_v1": None}, "not an exact SHA-256"),
        ({"feature_names_sha256_v1": "0" * 64}, "SHA-256 mismatch"),
        ({"fold_id": "FOLD_1"}, "serving fold"),
        ({"variant": "R_REGRET"}, "requested variant"),
        ({"state_dim": 3}, "state_dim"),
        ({"feature_means": [0.0]}, "feature_means shape"),
        ({"feature_stds": [1.0, 0.0]}, "strictly positive"),
        ({"feature_stds": [1.0, np.nan]}, "non-finite"),
    ],
)
def test_exit_iql_checkpoint_binding_rejects_unproven_or_mismatched_state(
    updates: dict,
    message: str,
) -> None:
    with pytest.raises(RuntimeError, match=message):
        require_exit_iql_checkpoint_binding(
            _checkpoint(**updates),
            feature_names=list(FEATURE_NAMES),
            feature_names_sha256=_feature_hash(),
            requested_variant="R_NET_REAL",
            requested_fold_id="FOLD_2",
        )


def test_exit_iql_checkpoint_binding_accepts_constant_but_bound_feature() -> None:
    means, stds = require_exit_iql_checkpoint_binding(
        _checkpoint(),
        feature_names=list(FEATURE_NAMES),
        feature_names_sha256=_feature_hash(),
        requested_variant="R_NET_REAL",
        requested_fold_id="FOLD_2",
    )

    assert means.shape == (2,)
    assert stds.tolist() == pytest.approx([1.0, 0.000001])


def _write_registry(
    tmp_path: Path,
    *,
    entry_updates: dict | None = None,
    summary_updates: dict | None = None,
) -> Path:
    bundle = tmp_path / "exit_bundle"
    bundle.mkdir()
    (bundle / "summary_v1.json").write_text(
        json.dumps(_production_summary(**(summary_updates or {}))),
        encoding="utf-8",
    )
    entry = {
        "path": str(bundle),
        "status": "ACTIVE",
        "in_sample_only": False,
        "active_variant": "R_NET_REAL",
        "active_aggregator": "max",
        "active_folds": ["FOLD_1", "FOLD_2"],
        "serving_fold": "FOLD_2",
    }
    entry.update(entry_updates or {})
    registry = tmp_path / "PROJECT_STATE_artifacts.json"
    registry.write_text(
        json.dumps(
            {
                "project": artifacts.THIS_PROJECT,
                "active": {"exit_iql": entry},
            }
        ),
        encoding="utf-8",
    )
    return registry


@pytest.mark.parametrize(
    ("entry_updates", "message"),
    [
        ({"active_variant": None}, "explicit active_variant"),
        ({"active_aggregator": None}, "active_aggregator"),
        ({"serving_fold": None}, "explicit serving_fold"),
        ({"serving_fold": "FOLD_3"}, "not present in active_folds"),
        ({"active_folds": ["FOLD_1", "FOLD_1"]}, "unique ordered"),
    ],
)
def test_exit_iql_registry_requires_one_explicit_serving_fold(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    entry_updates: dict,
    message: str,
) -> None:
    registry = _write_registry(tmp_path, entry_updates=entry_updates)
    monkeypatch.setattr(artifacts, "SELECTION_CONTRACT", registry)

    with pytest.raises(artifacts.ArtifactGuardError, match=message):
        artifacts.load_decision_entry("exit_iql")


def test_exit_iql_registry_rejects_research_only_active_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = _write_registry(
        tmp_path,
        summary_updates={
            "research_only_v1": True,
            "iql_production_allowed_v1": False,
        },
    )
    monkeypatch.setattr(artifacts, "SELECTION_CONTRACT", registry)

    with pytest.raises(artifacts.ArtifactGuardError, match="research_only_v1"):
        artifacts.load_decision_entry("exit_iql")


def test_exit_iql_registry_returns_the_explicit_serving_fold(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = _write_registry(tmp_path)
    monkeypatch.setattr(artifacts, "SELECTION_CONTRACT", registry)

    entry = artifacts.load_decision_entry("exit_iql")

    assert entry["serving_fold"] == "FOLD_2"


def test_exit_iql_adapters_have_no_implicit_fold_or_soft_coverage_escape() -> None:
    load_parameters = inspect.signature(ExitIQLV2Adapter.load).parameters
    decider_parameters = inspect.signature(ExitDeciderV12Adapter.load).parameters

    assert load_parameters["fold_id"].default is inspect.Parameter.empty
    assert decider_parameters["fold_id"].default is inspect.Parameter.empty
    assert "strict_failclosed" not in load_parameters
    assert "warmup_grace_features" not in load_parameters


@pytest.mark.parametrize("value", [None, "not-a-number", np.nan, np.inf])
def test_exit_iql_state_builder_rejects_missing_or_invalid_bound_value(
    value: object,
) -> None:
    adapter = ExitIQLV2Adapter(
        model=object(),  # build_state_vector does not touch the model
        feature_names=["small_scale_bound_feature_v1"],
        variant="R_NET_REAL",
        fold_id="FOLD_2",
        aggregator="max",
        beta=1.0,
        k_weights=None,
        artifact_root=Path("/unit"),
        required_feature_names=frozenset({"small_scale_bound_feature_v1"}),
    )

    with pytest.raises(RuntimeError, match="FEATURE_COVERAGE_FATAL"):
        adapter.build_state_vector({"small_scale_bound_feature_v1": value})


def test_exit_iql_state_builder_rejects_unknown_one_hot_category() -> None:
    adapter = ExitIQLV2Adapter(
        model=object(),
        feature_names=["session__EU", "session__US"],
        variant="R_NET_REAL",
        fold_id="FOLD_2",
        aggregator="max",
        beta=1.0,
        k_weights=None,
        artifact_root=Path("/unit"),
        required_feature_names=frozenset({"session__EU", "session__US"}),
    )

    with pytest.raises(RuntimeError, match="known-category-required"):
        adapter.build_state_vector({"session": "ASIA"})
