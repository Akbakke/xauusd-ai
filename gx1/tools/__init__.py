"""
GX1 Development Tools

This package contains development and debugging tools that are not part of
the core runtime but are useful for development, testing, and debugging.

Tools:
- verify_freeze: Verify PROD freeze structure
- debug_oanda_ping: Quick connectivity check against OANDA REST API

(exec_smoke_test parked to gx1/tools/_legacy_disabled/ 2026-07-07 cleanup wave -
its only consumers were a doc-only wrapper script and a broken test importing a
nonexistent module path.)
"""

__all__ = [
    "verify_freeze",
    "debug_oanda_ping",
]













