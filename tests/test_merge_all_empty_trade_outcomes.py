"""Test that merge with only empty trade_outcomes files produces MERGED + PROOF + metrics."""

import os
import tempfile
from pathlib import Path

import pytest


def test_merge_all_empty_trade_outcomes_produces_artifacts():
    """
    TRUTH 0-trades: Merge with all empty trade_outcomes files must produce
    trade_outcomes_*_MERGED.parquet, MERGE_PROOF.json, metrics_*_MERGED.json without error.
    """
    from gx1.utils.empty_trade_outcomes import write_empty_trade_outcomes_parquet

    with tempfile.TemporaryDirectory() as tmp:
        output_dir = Path(tmp)
        chunk_dir = output_dir / "chunk_0"
        chunk_dir.mkdir(parents=True)

        run_id = "TEST_RUN_EMPTY_20260215"
        outcomes_path = chunk_dir / f"trade_outcomes_{run_id}.parquet"
        write_empty_trade_outcomes_parquet(outcomes_path, run_id=run_id)
        assert outcomes_path.exists()

        chunk_results = [
            {
                "chunk_id": 0,
                "artifacts": {
                    "trade_outcomes": outcomes_path,
                },
            },
        ]

        # Must run in TRUTH mode for 0-trades merge path
        env_before = os.environ.get("GX1_RUN_MODE")
        os.environ["GX1_RUN_MODE"] = "TRUTH"
        try:
            from gx1.scripts.replay_eval_gated_parallel import merge_artifacts

            merged = merge_artifacts(chunk_results, run_id, output_dir)

            assert "trade_outcomes" in merged
            merged_path = merged["trade_outcomes"]
            assert merged_path.exists()
            assert "_MERGED.parquet" in str(merged_path)

            assert "merge_proof" in merged
            proof_path = merged["merge_proof"]
            assert proof_path.exists()
            import json
            proof = json.loads(proof_path.read_text())
            assert proof.get("all_empty") is True
            assert proof.get("n_trade_outcomes_files", 0) >= 1

            assert "metrics" in merged
            metrics_path = merged["metrics"]
            assert metrics_path.exists()
            metrics = json.loads(metrics_path.read_text())
            assert metrics.get("n_trades") == 0
            assert metrics.get("total_pnl_bps") == 0.0
            assert metrics.get("winrate") is None  # null for 0-trades
        finally:
            if env_before is not None:
                os.environ["GX1_RUN_MODE"] = env_before
            elif "GX1_RUN_MODE" in os.environ:
                os.environ.pop("GX1_RUN_MODE")
