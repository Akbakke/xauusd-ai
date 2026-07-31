"""
GX1 package initializer.

This file ensures gx1 is treated as a regular package (not namespace-only),
so gx1.__file__ resolves inside the ENGINE repo.
"""

# ── UTC log timestamps (2026-06-18) ───────────────────────────────────────────
# Long-running GX1 processes such as collector and paper_runner log with
# datefmt="%Y-%m-%dT%H:%M:%SZ" — a literal 'Z'. Python's logging.Formatter defaults
# to time.localtime for %(asctime)s, so the 'Z' LIED by the local UTC offset (CEST
# = +2h: logs read "19:33Z" when real UTC was 17:33). The DATA clock (cutoff /
# decision_ts / file mtimes) was always correct UTC — this was purely a misleading
# log label — but a lying timestamp burns debugging time. Pin the converter to UTC
# here (runs on every `python -m gx1.…` before any basicConfig) so the 'Z' is honest
# for ALL gx1 loggers, one truth. No functional/data effect; cosmetic-but-correct.
import logging as _logging
import time as _time
_logging.Formatter.converter = _time.gmtime
