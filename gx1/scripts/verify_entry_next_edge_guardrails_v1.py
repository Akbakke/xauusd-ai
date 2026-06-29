"""Compatibility wrapper for the active Entry foundation guardrails."""
from __future__ import annotations

from gx1.scripts.verify_entry_foundation_guardrails_v1 import main, run


__all__ = ["main", "run"]


if __name__ == "__main__":
    raise SystemExit(main())
