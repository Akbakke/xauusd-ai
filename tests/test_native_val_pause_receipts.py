import copy
import json
from pathlib import Path

import numpy as np
import pytest

from gx1.models.entry_v10 import entry_v10_ctx_train_v3 as trainer
from gx1.contracts.unified_exit_random_access_val_evaluator_v1 import run_resumable_random_access_val_evaluation_v1
from tests.test_candidate_execution_staging import Harness
from tests.test_unified_exit_random_access_val_evaluator_v1 import _fixture, _with_route_outputs, _checkpoint_binding, _entry_policy


def test_successive_val_windows_with_same_train_pointer_preserve_each_receipt(tmp_path):
    h = Harness(tmp_path / "h")
    output = h.root / "STAGED"
    _, evidence = h.run(output, steps=4)
    evidence.update(phase="validation", epoch_index=0, reason="native_full_val_window_complete")
    progress = Path(evidence["session_directory"]) / "native_val/epoch_0001/ROLLOUT_PROGRESS.json"
    model, representations, adapter, contract = _fixture(
        thresholds=np.zeros((5508, 2), np.float32), counts=np.ones(5508, np.int64))
    model = _with_route_outputs(model)
    binding = _checkpoint_binding(contract, adapter, tmp_path)
    arguments = dict(model=model, entry_decision_representations=representations,
        adapter=adapter, checkpoint_binding=binding,
        entry_policy_decisions=_entry_policy(adapter, binding), entry_route_diagnostics={},
        progress_path=progress, result_path=progress.parent / "VAL_RESULT.json",
        policy_batch_size=128, progress_interval_forwards=64)
    receipts = []
    for expected_cursor in (128, 256, 384):
        pause = run_resumable_random_access_val_evaluation_v1(**arguments, max_forwards_this_invocation=1)
        assert pause["decision"] == "PAUSED_RESUMABLE"
        assert pause["next_entry_scan_position"] == expected_cursor
        evidence["native_val_pause"] = trainer._native_candidate_val_pause_binding(progress, pause)
        receipt = trainer._write_candidate_execution_pause_receipt(
            evidence, out_bundle_dir=output, gx1_data_override="")
        assert trainer._write_candidate_execution_pause_receipt(
            evidence, out_bundle_dir=output, gx1_data_override="") == receipt
        receipts.append((receipt, receipt.read_bytes()))
    assert len({p for p, _ in receipts}) == 3
    assert all(p.read_bytes() == content for p, content in receipts)
    assert len({json.loads(content)["execution_budget_sha256"] for _, content in receipts}) == 1
    changed = {**evidence, "reason": "invocation_wall_limit"}
    with pytest.raises(RuntimeError, match="RECEIPT_CONFLICT"):
        trainer._write_candidate_execution_pause_receipt(changed, out_bundle_dir=output, gx1_data_override="")
    for field, value in (("checkpoint_selection_advanced", True), ("progress_sha256", "f" * 64), ("progress_path", str(tmp_path))):
        changed = copy.deepcopy(evidence)
        changed["native_val_pause"][field] = value
        with pytest.raises(RuntimeError, match="VAL_RECEIPT_INVALID"):
            trainer._write_candidate_execution_pause_receipt(changed, out_bundle_dir=output, gx1_data_override="")
    changed = copy.deepcopy(evidence)
    changed["native_val_pause"]["pause"]["decision"] = "PASS_COMPLETE"
    with pytest.raises(RuntimeError, match="VAL_RECEIPT_INVALID"):
        trainer._write_candidate_execution_pause_receipt(changed, out_bundle_dir=output, gx1_data_override="")
    progress.write_text("changed after checkpoint")
    with pytest.raises(RuntimeError, match="VAL_RECEIPT_INVALID"):
        trainer._write_candidate_execution_pause_receipt(evidence, out_bundle_dir=output, gx1_data_override="")
