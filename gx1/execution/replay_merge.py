"""
TRUTH 1W1C merge: produce MERGED artifacts and RUN_COMPLETED from a single chunk.

No dependency on legacy (quarantined) replay scripts.
Used by run_truth_e2e_sanity after process_chunk().
0-trades: when chunk has no trade_outcomes file but footer n_trades_closed==0, write empty MERGED parquet.
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from gx1.execution.chunk_failure import write_fatal_capsule
from gx1.utils.empty_trade_outcomes import write_empty_trade_outcomes_parquet

log = logging.getLogger(__name__)


def _load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def merge_artifacts_1w1c(run_dir: Path, run_id: str, output_dir: Optional[Path] = None) -> Dict[str, Any]:
    """
    Merge single chunk (chunk_0) into output: write *_MERGED.*, MERGE_PROOF, RUN_COMPLETED.

    run_dir must contain chunk_0/ with chunk_footer.json. If output_dir is given, MERGED artifacts
    are written there (root artifacts in run_root); otherwise written to run_dir.
    trade_outcomes: if chunk has trade_outcomes_{run_id}.parquet, copy to MERGED; if missing and
    n_trades_closed==0, write empty MERGED parquet with canonical schema; if missing and n_trades_closed>0, hard-fail.
    Returns dict of written paths and metadata (for tests/logging).
    """
    run_dir = run_dir.resolve()
    out_root = (output_dir.resolve() if output_dir else run_dir)
    chunk_dir = run_dir / "chunk_0"
    if not chunk_dir.is_dir():
        raise RuntimeError(f"[REPLAY_MERGE] chunk_0 not found in {run_dir}")

    footer_path = chunk_dir / "chunk_footer.json"
    if not footer_path.exists():
        raise RuntimeError(f"[REPLAY_MERGE] chunk_footer.json not found: {footer_path}")

    footer = _load_json(footer_path)
    if footer.get("status") != "ok":
        raise RuntimeError(f"[REPLAY_MERGE] chunk_footer status != ok: {footer.get('status')}")

    n_trades_closed = int(footer.get("n_trades_closed", 0) or 0)
    trade_src = chunk_dir / f"trade_outcomes_{run_id}.parquet"
    trade_dst = out_root / f"trade_outcomes_{run_id}_MERGED.parquet"

    if trade_src.exists():
        shutil.copy2(trade_src, trade_dst)
        log.info("[REPLAY_MERGE] Wrote %s", trade_dst.name)
    elif n_trades_closed == 0:
        write_empty_trade_outcomes_parquet(trade_dst, run_id=run_id)
        log.info("[REPLAY_MERGE] Wrote empty %s (0-trades contract)", trade_dst.name)
    else:
        write_fatal_capsule(
            chunk_output_dir=out_root,
            chunk_idx=0,
            run_id=run_id,
            fatal_reason="MERGE_TRADE_OUTCOMES_MISSING",
            error_message=(
                f"[REPLAY_MERGE] trade_outcomes not found: {trade_src} but n_trades_closed={n_trades_closed}>0. "
                "TRUTH requires chunk to write trade_outcomes parquet (or empty with 0 trades)."
            ),
            extra_fields={"trade_src": str(trade_src), "n_trades_closed": n_trades_closed},
        )
        raise RuntimeError(
            f"[REPLAY_MERGE] trade_outcomes not found: {trade_src} and n_trades_closed={n_trades_closed}. "
            "Hard-fail: chunk must produce trade_outcomes_{run_id}.parquet."
        )

    # canonical_economics_hash (optional): from trade_uid or row digest
    try:
        df = pd.read_parquet(trade_dst)
        if "trade_uid" in df.columns:
            canonical_economics_hash = hashlib.sha256(
                "".join(sorted(df["trade_uid"].astype(str))).encode("utf-8")
            ).hexdigest()
        else:
            canonical_economics_hash = hashlib.sha256(
                "".join(sorted(df.astype(str).sum(axis=1).tolist())).encode("utf-8")
            ).hexdigest()
    except Exception as e:
        log.warning("[REPLAY_MERGE] Could not compute canonical_economics_hash: %s", e)
        canonical_economics_hash = None

    # metrics_MERGED.json from chunk_footer
    n_trades = int(footer.get("n_trades_closed", 0) or 0)
    forward_calls = (
        footer.get("n_model_calls")
        or footer.get("bars_evaluated")
        or footer.get("transformer_forward_calls")
        or 0
    )
    try:
        forward_calls = int(forward_calls)
    except (TypeError, ValueError):
        forward_calls = 0

    metrics = {
        "run_id": run_id,
        "n_chunks": 1,
        "n_trades": n_trades,
        "transformer_forward_calls": forward_calls,
        "forward_calls_total": forward_calls,
        "n_model_calls": forward_calls,
        "source": "replay_merge_1w1c",
        "chunk_footer_status": footer.get("status"),
    }
    metrics_path = out_root / f"metrics_{run_id}_MERGED.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    log.info("[REPLAY_MERGE] Wrote %s", metrics_path.name)

    # MERGE_PROOF
    merge_proof = {
        "run_id": run_id,
        "n_chunks": 1,
        "n_trade_outcomes_files": 1,
        "canonical_economics_hash": canonical_economics_hash,
        "source": "replay_merge_1w1c",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    merge_proof_path = out_root / f"MERGE_PROOF_{run_id}.json"
    with open(merge_proof_path, "w", encoding="utf-8") as f:
        json.dump(merge_proof, f, indent=2)
    log.info("[REPLAY_MERGE] Wrote %s", merge_proof_path.name)

    # RUN_COMPLETED.json
    run_completed = {
        "status": "COMPLETED",
        "run_id": run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "chunks_submitted": 1,
        "chunks_completed": 1,
        "chunks_failed": [],
        "source": "replay_merge_1w1c",
    }
    run_completed_path = out_root / "RUN_COMPLETED.json"
    with open(run_completed_path, "w", encoding="utf-8") as f:
        json.dump(run_completed, f, indent=2)
    log.info("[REPLAY_MERGE] Wrote %s", run_completed_path.name)

    return {
        "trade_outcomes_merged": str(trade_dst),
        "metrics_merged": str(metrics_path),
        "merge_proof": str(merge_proof_path),
        "run_completed": str(run_completed_path),
    }
