from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from gx1.contracts.entry_model_native_trade_path_metrics_v1 import (
    TRADE_PATH_METRICS_DECISION,
    TradePathMetricsError,
    derive_unified_exit_trade_path_metrics,
)
from tests.model_native_sizing_support import write_passing_unified_replay_fixture


def test_exact_unified_exit_trace_yields_research_only_path_metrics(
    tmp_path: Path,
) -> None:
    evidence = write_passing_unified_replay_fixture(tmp_path)
    replay_rows = pd.read_parquet(evidence["joint_replay_rows_path"])
    exit_trace_rows = pd.read_parquet(evidence["joint_exit_trace_rows_path"])

    report, trades = derive_unified_exit_trade_path_metrics(
        replay_rows=replay_rows,
        exit_trace_rows=exit_trace_rows,
        candidate_bundle_sha256=(
            evidence["candidate_bundle_authority"]["bundle_commit_sha256"]
        ),
    )

    assert report["decision"] == TRADE_PATH_METRICS_DECISION
    assert report["production_authority_ready"] is False
    assert report["edge_claim_allowed"] is False
    assert report["cost_policy"]["production_economics_bound"] is False
    assert len(trades) == 128
    assert set(trades["side"]) == {"LONG", "SHORT"}
    assert (trades["intrabar_mae_bps"] >= 0.0).all()
    assert (trades["intrabar_mfe_bps"] >= 0.0).all()
    assert (trades["holding_minutes"] > 0.0).all()
    assert int(trades["mae_before_mfe"].sum()) == len(trades)
    assert int(trades["same_m1_bar_order_unknown"].sum()) == 0
    independent = report["authorized_independent_trade_metrics"]
    assert independent["completed_round_trips"] == 128
    assert independent["one_way_turnover_events"] == 256
    serial = report["serial_one_position_ledger"]
    assert serial["selected_trade_rows"] == 128
    assert serial["skipped_overlapping_authorized_rows"] == 0


def test_trade_path_metrics_rejects_source_tape_byte_drift(
    tmp_path: Path,
) -> None:
    evidence = write_passing_unified_replay_fixture(tmp_path)
    replay_rows = pd.read_parquet(evidence["joint_replay_rows_path"])
    exit_trace_rows = pd.read_parquet(evidence["joint_exit_trace_rows_path"])
    source_path = Path(
        exit_trace_rows["closed_m1_source_path"].iloc[0]
    ).resolve()
    changed = pd.read_parquet(source_path)
    changed.loc[0, "volume"] = int(changed.loc[0, "volume"]) + 1
    changed.to_parquet(source_path, index=False)

    with pytest.raises(TradePathMetricsError, match="trace source hash differs"):
        derive_unified_exit_trade_path_metrics(
            replay_rows=replay_rows,
            exit_trace_rows=exit_trace_rows,
            candidate_bundle_sha256=(
                evidence["candidate_bundle_authority"]["bundle_commit_sha256"]
            ),
        )
