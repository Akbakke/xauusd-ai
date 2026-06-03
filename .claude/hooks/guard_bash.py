#!/usr/bin/env python3
"""
GX1 PreToolUse guard. Runs BEFORE any Bash command Claude tries to execute.
Blocks (exit 2) the irreversible / project-violating things from AGENTS.md.

Exit code 2 = block the action. Exit 0 = allow. (Exit 1 would only warn.)
Reads the tool-call JSON from stdin.
"""

import json
import re
import sys

# Patterns that must never run.
BLOCK_PATTERNS = [
    (r"git\s+push\s+.*--force", "force push is forbidden"),
    (r"git\s+push\s+.*\s-f(\s|$)", "force push (-f) is forbidden"),
    (r"git\s+commit\s+.*--amend", "amending commits is forbidden"),
    (r"\bgit\s+push\s+.*\+", "force-push refspec (+) is forbidden"),
    # Project isolation: no EURUSD anything inside this XAUUSD tree.
    # 2026-05-28: EURUSD was fully wiped per user vedtak. This guard stays as
    # defense-in-depth — no path containing "eurusd" should ever come back.
    (r"(?i)eurusd", "EURUSD reference inside XAUUSD project — never mix the two"),
    # Secrets: block adding/committing env or credential files.
    (r"git\s+add\s+.*\.env", "staging a .env file is forbidden (secrets)"),
    (r"git\s+commit\s+.*\.env", "committing a .env file is forbidden (secrets)"),
    (r"git\s+add\s+.*credentials", "staging a credentials file is forbidden"),
]


def main() -> int:
    try:
        event = json.load(sys.stdin)
    except Exception:
        # If we can't parse, fail open on parsing but don't crash the session.
        return 0

    cmd = event.get("tool_input", {}).get("command", "")
    if not cmd:
        return 0

    for pattern, reason in BLOCK_PATTERNS:
        if re.search(pattern, cmd):
            print(f"BLOCKED by GX1 guard: {reason}\n  command: {cmd}", file=sys.stderr)
            return 2  # <-- this is what actually stops Claude

    return 0


if __name__ == "__main__":
    sys.exit(main())
