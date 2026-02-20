"""
Re-export of legacy entry_v10_ctx build helpers.

Canonical API for tests and programmatic use. Implementation lives in
gx1.scripts.build_entry_v10_ctx_training_dataset_legacy. Prefer importing
from this module instead of the script (avoids "scripts as API").
"""

from __future__ import annotations

from gx1.scripts.build_entry_v10_ctx_training_dataset_legacy import (
    build_dataset,
    compute_session_histogram,
    write_manifest,
)

__all__ = ["compute_session_histogram", "build_dataset", "write_manifest"]
