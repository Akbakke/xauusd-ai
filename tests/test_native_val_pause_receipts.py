import hashlib
import json
from pathlib import Path

import pytest

from gx1.models.entry_v10 import entry_v10_ctx_train_v3 as trainer
from tests.test_candidate_execution_staging import Harness


def test_successive_val_windows_with_same_train_pointer_preserve_each_receipt(tmp_path):
    h = Harness(tmp_path / "h")
    output = h.root / "STAGED"
    _, evidence = h.run(output, steps=4)
    evidence.update(phase="validation", epoch_index=0, reason="native_full_val_window_complete")
    progress = Path(evidence["session_directory"]) / "native_val/epoch_0001/ROLLOUT_PROGRESS.json"
    progress.parent.mkdir(parents=True)
    receipts = []
    for views in (128, 256, 384):
        progress.write_text(json.dumps({"materialized_state_view_count": views}))
        evidence["native_val_pause"] = {
            "decision": "PAUSED_RESUMABLE", "progress_path": str(progress),
            "progress_file_sha256": hashlib.sha256(progress.read_bytes()).hexdigest(),
        }
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
    progress.write_text("changed after checkpoint")
    with pytest.raises(RuntimeError, match="VAL_RECEIPT_INVALID"):
        trainer._write_candidate_execution_pause_receipt(evidence, out_bundle_dir=output, gx1_data_override="")
