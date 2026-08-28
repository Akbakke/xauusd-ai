from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from gx1.scripts.materialize_entry_model_native_trade_path_metrics_v1 import (
    TRADE_PATH_METRICS_EVENT_PREFIX,
    TRADE_PATH_METRICS_EVENT_SCHEMA_VERSION,
    materialize_unified_exit_trade_path_metrics,
)
from tests.model_native_sizing_support import write_passing_unified_replay_fixture


def test_materialized_trade_path_event_binds_exact_inputs_and_outputs(
    tmp_path: Path,
) -> None:
    evidence = write_passing_unified_replay_fixture(tmp_path / "source")
    event_path, event = materialize_unified_exit_trade_path_metrics(
        replay_rows_path=evidence["joint_replay_rows_path"],
        exit_trace_rows_path=evidence["joint_exit_trace_rows_path"],
        candidate_bundle_sha256=(
            evidence["candidate_bundle_authority"]["bundle_commit_sha256"]
        ),
        output_dir=tmp_path / "published",
    )

    assert event_path.name.startswith(f"{TRADE_PATH_METRICS_EVENT_PREFIX}_")
    assert event["schema_version"] == TRADE_PATH_METRICS_EVENT_SCHEMA_VERSION
    assert event["decision"] == "RESEARCH_ONLY_BLOCKED_NET_ECONOMICS"
    assert event["production_authority_ready"] is False
    assert event["edge_claim_allowed"] is False
    assert event["report"]["authorized_independent_trade_metrics"][
        "completed_round_trips"
    ] == 128
    assert Path(event["replay_rows"]["path"]).is_file()
    assert Path(event["exit_trace_rows"]["path"]).is_file()
    trades_path = Path(event["trades"]["path"])
    assert trades_path.is_file()
    assert len(pd.read_parquet(trades_path)) == 128
    assert set(event["producer_source_files"]) == {
        "gx1/contracts/entry_model_native_trade_path_metrics_v1.py",
        "gx1/scripts/materialize_entry_model_native_trade_path_metrics_v1.py",
    }
    persisted = json.loads(event_path.read_text(encoding="utf-8"))
    assert persisted == event
