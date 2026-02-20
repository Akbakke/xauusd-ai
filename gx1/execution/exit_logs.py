"""
Exit ML log path helpers.

TRUTH: when exit transformer is active, the exits jsonl file must exist
(even with 0 trades) so POSTRUN gate can pass. This module provides
a small helper to create the placeholder file early in replay.
"""

from __future__ import annotations

from pathlib import Path


def create_exit_jsonl_placeholder(run_dir: Path, run_id: str) -> Path:
    """
    Ensure logs/exits/exits_<run_id>.jsonl exists under run_dir (create empty if missing).

    Used at replay start when exit transformer is enabled so POSTRUN gate
    (exits jsonl required) passes even when n_trades_closed == 0.

    Parameters
    ----------
    run_dir : Path
        Chunk output directory (e.g. replay/chunk_0).
    run_id : str
        Run identifier for the filename.

    Returns
    -------
    Path
        Path to the exits jsonl file (existing or newly created).
    """
    log_path = Path(run_dir) / "logs" / "exits" / f"exits_{run_id}.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if not log_path.exists():
        log_path.write_text("", encoding="utf-8")
    return log_path
