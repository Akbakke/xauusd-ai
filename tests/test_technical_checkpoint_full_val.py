from __future__ import annotations

from pathlib import Path

import pytest
import torch

from gx1.scripts import validate_entry_model_native_technical_checkpoint_v1 as technical_val
from gx1.scripts.validate_entry_model_native_technical_checkpoint_v1 import (
    TechnicalValidationError,
    _require_pretest_guard,
    _require_no_test_rows,
)


def test_technical_full_val_clock_guard_rejects_test_rows(tmp_path: Path) -> None:
    import pandas as pd

    path = tmp_path / "val.parquet"
    pd.DataFrame(
        {
            "time": pd.to_datetime(
                ["2026-06-30T23:55:00Z", "2026-07-01T00:00:00Z"], utc=True
            )
        }
    ).to_parquet(path, index=False)

    with pytest.raises(TechnicalValidationError, match="TEST_BOUNDARY_VIOLATION"):
        _require_no_test_rows(path.resolve())


def test_technical_full_val_clock_guard_accepts_strict_pretest_val(tmp_path: Path) -> None:
    import pandas as pd

    path = tmp_path / "val.parquet"
    pd.DataFrame(
        {
            "time": pd.to_datetime(
                ["2025-06-01T22:00:00Z", "2026-06-30T14:55:00Z"], utc=True
            )
        }
    ).to_parquet(path, index=False)

    observed = _require_no_test_rows(path.resolve())
    assert observed["rows"] == 2
    assert observed["timestamps_at_or_after_test_boundary"] == 0


def test_technical_full_val_binds_dataset_id_when_validating_test_guard(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Regression: the seal API rejects a call that omits dataset lineage."""

    captured: dict[str, object] = {}

    def fake_guard(*args: object, **kwargs: object) -> dict[str, object]:
        captured["args"] = args
        captured["kwargs"] = kwargs
        return {"access_proof": {"test_dataset_bytes_read": False}}

    monkeypatch.setattr(
        technical_val,
        "require_pretest_or_prefreeze_test_guard_lineage",
        fake_guard,
    )
    _require_pretest_guard(
        guard_json=tmp_path / "guard.json",
        guard_sha256="a" * 64,
        dataset_run_id="PRETEST_V3_20260829T173000Z",
        dataset_dir=tmp_path,
    )
    assert captured["kwargs"] == {
        "expected_dataset_run_id": "PRETEST_V3_20260829T173000Z",
        "expected_dataset_dir": tmp_path,
    }


def test_technical_full_val_session_is_hash_bound_and_monotonic(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A bounded VAL continuation cannot silently reuse or rewind evidence."""

    monkeypatch.setattr(
        technical_val,
        "_restore_candidate_validation_snapshot",
        lambda snapshot: dict(snapshot),
    )
    session = technical_val._TechnicalValidationSession(
        directory=(tmp_path / "session").resolve(),
        contract={"schema_version": "unit-test-v1", "input_sha256": "a" * 64},
    )
    assert session.load(expected_batches=10) is None

    session.save(
        expected_batches=10,
        next_batch_offset=3,
        validation_snapshot={"accumulator": torch.tensor([1, 2, 3])},
        elapsed_seconds_completed=12.5,
        complete=False,
    )
    restored = session.load(expected_batches=10)
    assert restored is not None
    assert restored["next_batch_offset"] == 3
    assert restored["elapsed_seconds_completed"] == 12.5

    with pytest.raises(TechnicalValidationError, match="NONMONOTONIC_PROGRESS"):
        session.save(
            expected_batches=10,
            next_batch_offset=3,
            validation_snapshot={"accumulator": torch.tensor([1, 2, 3])},
            elapsed_seconds_completed=13.0,
            complete=False,
        )


def test_direction_witness_cli_routes_before_legacy_val_parser(monkeypatch, tmp_path):
    from gx1.audit import dynamic_m5_direction_witness_v1 as direction

    observed = []
    monkeypatch.setattr(direction, "run", lambda **kwargs: observed.append(kwargs) or {"status": "source-only-test"})
    plan = tmp_path / "plan.json"
    out = (tmp_path / "out").resolve()
    result = technical_val.main(["--dynamic-m5-direction-witness-plan", str(plan),
        "--dynamic-m5-direction-witness-plan-sha256", "a" * 64,
        "--device", "cuda", "--out-dir", str(out)])
    assert result == 0
    assert observed == [{"plan_path": plan, "plan_sha256": "a" * 64, "device": "cuda", "out_dir": out}]


@pytest.mark.parametrize("extra", [
    ["--device", "cuda"],
    ["--dynamic-m5-direction-witness-plan", "another.json"],
    ["--bundle-dir", "legacy-bundle"],
    ["--dynamic-compact-input-influence-plan", "another-mode.json"],
])
def test_direction_witness_cli_rejects_duplicate_or_mixed_scope(monkeypatch, tmp_path, extra):
    from gx1.audit import dynamic_m5_direction_witness_v1 as direction

    calls = []
    monkeypatch.setattr(direction, "run", lambda **kwargs: calls.append(kwargs))
    with pytest.raises(SystemExit):
        technical_val.main(["--dynamic-m5-direction-witness-plan", str(tmp_path / "plan.json"),
            "--dynamic-m5-direction-witness-plan-sha256", "a" * 64,
            "--device", "cuda", "--out-dir", str((tmp_path / "out").resolve()), *extra])
    assert calls == []


@pytest.mark.parametrize("change", [
    {"training_enabled": True}, {"exit_evaluation": True}, {"test_data_used": True},
    {"max_forward_calls": 2}, {"max_reverse_calls": 23},
    {"headline_horizons": [1, 5, 15, 60, 120, 240, 481]},
])
def test_direction_witness_plan_rejects_scope_expansion(change):
    from gx1.audit import dynamic_m5_direction_witness_v1 as direction

    plan = {"schema_version": direction.PLAN_SCHEMA, "run_id": "fixed-train-witness",
        "policy_stage_key": "unit-scope", "input_scope": {"path": "/unused.json", "sha256": "a" * 64},
        "output_directory": "/unused-output", "one_shot_invocations": 1,
        "max_wall_seconds": 900, "max_forward_calls": 1, "max_reverse_calls": 22,
        "headline_horizons": [1, 5, 15, 60, 120, 240, 480],
        **{key: False for key in direction.FLAGS}}
    direction._require_plan(plan)
    with pytest.raises(RuntimeError, match="PLAN_BUDGET_OR_AUTHORITY"):
        direction._require_plan({**plan, **change})



def _direction_v2_reference_fixture():
    from gx1.audit import dynamic_m5_direction_witness_v1 as direction
    from gx1.contracts.entry_model_native_aux_targets_v3 import MODEL_NATIVE_FORECAST_OUTPUT_LAYOUT
    from gx1.contracts.entry_observed_market_horizons_v1 import (
        M5_FORECAST_VALID_BATCH_KEY, SUPPORT_FORECAST_TARGET_BATCH_KEY, SUPPORT_FORECAST_VALID_BATCH_KEY,
    )
    ref = lambda name: {"path": "/source-only/" + name, "sha256": "a" * 64}
    refs = {k: ref(k + ".json") for k in ("plan", "terminal", "actual_peer", "result")}
    scope = {"final_online_result": ref("final.json"), "loss_scales": ref("scales.json"),
        "parity_reference": ref("TRAIN_OBSERVATION.json"), "canonical_pointer_proof": refs,
        "session": {"online_model_state_sha256": "e" * 64}}
    parents, clocks = list(range(256)), list(range(1000, 1256))
    cohort = {"parent_entry_row_indices": parents, "entry_time_ns": clocks,
              "parent_entry_parquet": {"path": "/never-read/train.parquet"}}
    final = {"optimizer_steps": 7391, "model_state_sha256": "e" * 64,
        "target_model_state_sha256": "f" * 64, "target_model_role": "frozen_constructor_anchor_no_forward",
        "entry_observed_market": {"identity": "same"}, "dynamic_m5_loss_scales": {"identity": "111"},
        "observations": {"train": scope["parity_reference"]}}
    frozen = {"parent_entry_rows": parents, "entry_time_ns": clocks,
        "entry_supervision_identity": final["entry_observed_market"], "query_minutes": list(range(1, 481)),
        "dynamic_entry_m1_rows": list(range(5000, 5256)), "components": {}}
    metadata = {"parent_rows": parents[:16], "entry_time_ns": clocks[:16],
        "original_target_storage": {}, "target_sha256": "d" * 64}
    batch = {"entry_row_index": torch.tensor(parents[:16]),
        "dynamic_entry_parent_row_index": torch.tensor(parents[:16]),
        "dynamic_entry_m1_row": torch.tensor(frozen["dynamic_entry_m1_rows"][:16])}
    cells = {}
    for name in direction._COMPONENTS:
        shape = (10,) if name == "forecast_pred" else (480, 3 if name == direction._COMPONENTS[0] else 2)
        frozen["components"][name] = {"target_dtype": "float32", "mask_dtype": "bool",
            "shape": [256, *shape], "target_storage_sha256": "1" * 64, "observed_mask_sha256": "2" * 64}
        metadata["original_target_storage"][name] = {
            "original_target_storage_sha256": "1" * 64,
            "original_observed_mask_sha256": "2" * 64, "comparison_scope_rows": 256}
        cells[name] = {"prediction_bps": torch.zeros(shape).tolist(), "target_bps": torch.zeros(shape).tolist(),
                       "observed": torch.ones(shape, dtype=torch.bool).tolist()}
        if name != "forecast_pred":
            cells[name]["action_order"] = ["LONG", "SHORT", *(["FLAT"] if name == direction._COMPONENTS[0] else [])]
            batch["y_" + name] = torch.zeros(16, *shape)
    batch["dynamic_terminal_markout_observed"] = torch.ones(16, 480, 3, dtype=torch.bool)
    batch["dynamic_path_observed"] = torch.ones(16, 480, 2, dtype=torch.bool)
    batch["dynamic_closed_path_observed"] = torch.ones(16, 480, 2, dtype=torch.bool)
    for h in (1, 5, 12, 24): batch["y_forecast_ret_K" + str(h)] = torch.zeros(16)
    batch[M5_FORECAST_VALID_BATCH_KEY] = torch.ones(16, 4, dtype=torch.bool)
    batch[SUPPORT_FORECAST_TARGET_BATCH_KEY] = torch.zeros(16, 6)
    batch[SUPPORT_FORECAST_VALID_BATCH_KEY] = torch.ones(16, 6, dtype=torch.bool)
    traces = [{"entry_row_index": p, "entry_time_ns": t,
        "full_query_curves": {"query_minutes": list(range(1, 481)),
            "components": {n: cells[n] for n in direction._COMPONENTS[:-1]}},
        "forecast_support": {**cells["forecast_pred"], "output_layout": list(MODEL_NATIVE_FORECAST_OUTPUT_LAYOUT)}}
        for p, t in zip(parents, clocks)]
    observation = {"role": "train", **{k: final[k] for k in (
        "optimizer_steps", "model_state_sha256", "target_model_state_sha256", "target_model_role")},
        "exit_evaluation": False, "native_launch_authorized": False, "cohort": cohort,
        "diagnostics": {"dynamic_frozen_targets": frozen, "dynamic_entry_observations": traces,
            "dynamic_curve_statistics": {"dynamic_m5_loss_scales": final["dynamic_m5_loss_scales"],
                "dynamic_loss_residual_units": "TRAIN_target_RMS_units"}}}
    alignment = {k: True for k in (
        "actual_six_curve_references_equal_bound_three_version_two_role_metadata",
        "six_pointer_declarations_equal_PLAN",
        "full480_and_existing_canonical_153_152_target_storage_nullable_masks_clock_order_checks_passed_in_completed_worker",
        "cohort_nullable_target_mask_forecast_fields_exact_against_saved168_reference167_bridge",
        "original111_RMS_identity_same_all_versions")}
    alignment["role_entries"] = {"train": 256}
    plan = {"schema_version": "gx1_saved_direction_comparison_plan_v1",
        "results": {"current_final": scope["final_online_result"]}, "loss_scales": scope["loss_scales"],
        "pointer_projections": [{"variant": v, "role": r, "path": "/never-read/" + ("train" if r == "train" else "val") + ".parquet",
            "columns": ["dynamic_entry_m1_row"]} for v in ("own_initial", "current_final", "reference_final") for r in ("train", "control")]}
    terminal = {"schema_version": "gx1_saved_direction_comparison_terminal_v1", "status": "COMPLETE",
        "exit_code": 0, "source_unchanged": True, "test_data_used": False, "relaunch_allowed": False,
        "result": refs["result"], "canonical_parent_pointer_projection_count": 6}
    peer = {"schema_version": "gx1_actual_saved_direction_comparison_peer_review_v1",
        "status": "PASS_STORED_COMPARISON_INTEGRITY_WITH_PRIMARY_ENTRY_QUALITY_UNQUALIFIED",
        "unresolved_integrity_findings": [], "references": {"PLAN.json": refs["plan"],
            "TERMINAL.json": refs["terminal"], "RESULTS.json": refs["result"]},
        "execution": {"canonical_parent_pointer_projections": 6, "terminal_complete_exit0": True},
        "alignment_review": alignment}
    return observation, scope, metadata, batch, final, {"plan": plan, "terminal": terminal, "actual_peer": peer}


def test_direction_v2_reference_join_is_in_memory_and_preserves_original_storage(monkeypatch):
    from gx1.audit import dynamic_m5_direction_witness_v1 as direction
    observation, scope, metadata, batch, final, documents = _direction_v2_reference_fixture()
    def forbidden(*args, **kwargs):
        raise AssertionError("Reference join attempted file/Parquet access")
    monkeypatch.setattr(direction, "_read", forbidden)
    monkeypatch.setattr(direction, "_read_saved_train_reference", forbidden)
    monkeypatch.setattr(direction, "_reference", forbidden)
    references, proof = direction._join_saved_train_reference(observation, scope=scope,
        metadata=metadata, batch=batch, final=final, pointer_documents=documents)
    assert set(references) == set(direction._COMPONENTS)
    assert all(v.shape[0] == 16 and v.dtype == torch.float32 for v in references.values())
    assert proof["completed_canonical_projections_reused"] == 6
    assert proof["result_payload_read_or_rehashed"] is False
    assert proof["nullable_masked_backing_reconstructed"] is False
    assert proof["packet_target_sha256"] == metadata["target_sha256"]


@pytest.mark.parametrize("defect", [
    "role", "state", "cohort", "clock", "global_pointer", "storage", "target", "mask",
    "trace", "action_order", "forecast_layout", "peer", "pointer_plan",
])
def test_direction_v2_reference_join_rejects_mismatched_provenance(defect):
    import copy
    from gx1.audit import dynamic_m5_direction_witness_v1 as direction
    observation, scope, metadata, batch, final, documents = copy.deepcopy(_direction_v2_reference_fixture())
    if defect == "role": observation["role"] = "control"
    elif defect == "state": observation["model_state_sha256"] = "0" * 64
    elif defect == "cohort": metadata["parent_rows"][:2] = metadata["parent_rows"][1::-1]
    elif defect == "clock": metadata["entry_time_ns"][0] += 1
    elif defect == "global_pointer": observation["diagnostics"]["dynamic_frozen_targets"]["dynamic_entry_m1_rows"][0] += 1
    elif defect == "storage": metadata["original_target_storage"]["forecast_pred"]["original_target_storage_sha256"] = "0" * 64
    elif defect == "peer": documents["actual_peer"]["unresolved_integrity_findings"] = ["not admitted"]
    elif defect == "pointer_plan": documents["plan"]["pointer_projections"].pop()
    else:
        trace = observation["diagnostics"]["dynamic_entry_observations"][0]
        cell = trace["full_query_curves"]["components"]["dynamic_mfe_bps"]
        if defect == "target": cell["target_bps"][0][0] = .25
        elif defect == "mask":
            cell["observed"][0][0] = False
            cell["target_bps"][0][0] = None
        elif defect == "trace": trace["entry_time_ns"] += 1
        elif defect == "action_order": cell["action_order"].reverse()
        elif defect == "forecast_layout": trace["forecast_support"]["output_layout"] = []
    with pytest.raises(RuntimeError, match="DYNAMIC_M5_DIRECTION_WITNESS_"):
        direction._join_saved_train_reference(observation, scope=scope, metadata=metadata,
            batch=batch, final=final, pointer_documents=documents)


@pytest.mark.parametrize("bad", [{"max_reverse_calls": 22}, {"max_wall_seconds": 901}, {"max_forward_calls": 2}])
def test_direction_v2_budget_is_explicit_and_does_not_change_historic_mode(bad):
    from gx1.audit import dynamic_m5_direction_witness_v1 as direction
    plan = {"schema_version": direction.PLAN_SCHEMA_V2, "run_id": "unit", "policy_stage_key": "unit",
        "input_scope": {"path": "/unused.json", "sha256": "a" * 64}, "output_directory": "/unused",
        "one_shot_invocations": 1, "max_wall_seconds": 900, "max_forward_calls": 1,
        "max_reverse_calls": 16, "headline_horizons": [1, 5, 15, 60, 120, 240, 480],
        **{key: False for key in direction.FLAGS}}
    direction._require_plan(plan)
    with pytest.raises(RuntimeError, match="PLAN_BUDGET_OR_AUTHORITY"):
        direction._require_plan({**plan, **bad})
    direction._require_plan({**plan, "schema_version": direction.PLAN_SCHEMA, "max_reverse_calls": 22})


@pytest.mark.parametrize("contents", ['{"a":1,"a":2}', '{"a":NaN}'])
def test_direction_v2_saved_train_reader_keeps_strict_json_guards(tmp_path, contents):
    import hashlib
    from gx1.audit import dynamic_m5_direction_witness_v1 as direction
    path = tmp_path / "TRAIN_OBSERVATION.json"
    path.write_text(contents)
    ref = {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    with pytest.raises(RuntimeError, match="SAVED_TRAIN_"):
        direction._read_saved_train_reference(ref, proofs=[])



@pytest.mark.parametrize("missing", ["early", "later", "both"])
def test_direction_v2_physical_scope_requires_both_groups_before_any_forward(missing):
    from gx1.audit import dynamic_m5_direction_witness_v1 as direction
    _, _, _, batch, _, _ = _direction_v2_reference_fixture()
    counts = direction._require_native170_physical_batch(batch)
    assert counts == {"early_1_15": 16 * 15 * 3, "later_16_480": 16 * 465 * 3}
    # Other three task masks remain observed: four active tasks alone would not
    # reject this one-group packet. The physical audit must reject it pre-forward.
    if missing in ("early", "both"): batch["dynamic_terminal_markout_observed"][:, :15] = False
    if missing in ("later", "both"): batch["dynamic_terminal_markout_observed"][:, 15:] = False
    assert bool(batch["dynamic_path_observed"].all())
    assert bool(batch["dynamic_closed_path_observed"].all())
    with pytest.raises(RuntimeError, match="PHYSICAL_TERMINAL_GROUP_COVERAGE"):
        direction._require_native170_physical_batch(batch)
