from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from gx1.execution.entry_manager import EntryManager
from gx1.execution.exit_manager import ExitManager


class ReplayEngine:
    """Pure replay skeleton that will eventually decouple live plumbing."""

    def __init__(
        self,
        entry_manager: EntryManager,
        exit_manager: ExitManager,
    ) -> None:
        self.entry_manager = entry_manager
        self.exit_manager = exit_manager

    def run(self, price_data: pd.DataFrame, output_path: Path, *, chunk_id: Optional[int] = None) -> None:
        """Placeholder for the future replay loop (entry → exit → PnL)."""
        raise NotImplementedError("ReplayEngine.run is not implemented yet")
