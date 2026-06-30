#!/usr/bin/env python3
"""Legacy compatibility wrapper for the smart seq520 rebuild preflight.

The active implementation is seq520. This seq495 entrypoint remains only so
older report references fail closed through the same current gate logic.
"""
from __future__ import annotations

from gx1.scripts.materialize_entry_smart_seq520_rebuild_preflight_v1 import main, run


if __name__ == "__main__":
    main()
