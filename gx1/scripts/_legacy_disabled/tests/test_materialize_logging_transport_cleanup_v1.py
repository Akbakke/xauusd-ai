from __future__ import annotations

import json
from pathlib import Path

from gx1.scripts.materialize_logging_transport_cleanup_v1 import (
    CONSISTENCY_AUDIT,
    CONTRACT,
    COUNTER_SPEC,
    INVENTORY,
    MANIFEST_STATUS,
    NEXT_STEP,
    NO_SEMANTIC_CHANGE_GUARD,
    RATE_LIMIT_CONTRACT,
    REPORT,
    SPAM_PLAN,
    SUMMARY,
    materialize,
)


def test_materialize_logging_transport_cleanup_outputs_artifacts(tmp_path: Path) -> None:
    repo_root = Path("/home/andre2/src/GX1_ENGINE")
    reports_root = tmp_path / "reports"
    reports_root.mkdir()

    result = materialize(repo_root=repo_root, reports_root=reports_root)
    extension_dir = Path(result["extension_dir"])
    assert extension_dir.exists()

    expected = [
        CONTRACT,
        INVENTORY,
        SPAM_PLAN,
        COUNTER_SPEC,
        RATE_LIMIT_CONTRACT,
        NO_SEMANTIC_CHANGE_GUARD,
        NEXT_STEP,
        CONSISTENCY_AUDIT,
        SUMMARY,
        REPORT,
        MANIFEST_STATUS,
    ]
    for name in expected:
        assert (extension_dir / name).exists(), name

    summary = json.loads((extension_dir / SUMMARY).read_text(encoding="utf-8"))
    assert summary["inventory_row_count_v1"] >= 5
    assert summary["next_step_v1"]["decision_v1"] == "SAFE_TO_APPLY_WITHOUT_REPLAY_LOGIC_RISK"

    consistency = json.loads((extension_dir / CONSISTENCY_AUDIT).read_text(encoding="utf-8"))
    assert consistency["status_v1"] == "PASS"
