#!/usr/bin/env python3
"""
GX1 PostToolUse Python breakage-check. Runs AFTER Claude writes/edits a .py file.

DELIBERATELY check-only:
  - NO `ruff format` (would rewrite the whole file → giant diffs on large legacy
    files, violating the "minimal change / never reorganize" guardrail).
  - NO `--fix` (auto-removing "unused" imports is dangerous in ML code with
    side-effect imports — registries, torch backends).
  - Only reports GENUINE breakage (syntax errors, undefined names, redefinition,
    return-outside-function), scoped to the file Claude just touched, so it can
    self-correct in the same turn before a multi-hour run is launched on it.

Falls back to `py_compile` (zero-dep, syntax-only) if ruff is unavailable.
"""

import json
import subprocess
import sys

VENV_PY = "/home/andre2/src/GX1_ENGINE/.venv/bin/python"
# Error classes only — no style. E9=syntax/indentation, F821=undefined name,
# F811=redefinition, F823=local used before assignment, F706=return outside fn,
# F704=yield outside fn, F501-2=%-format errors.
RUFF_SELECT = "E9,F821,F811,F823,F706,F704,F501,F502"


def main() -> int:
    try:
        event = json.load(sys.stdin)
    except Exception:
        return 0

    path = event.get("tool_input", {}).get("file_path", "")
    if not path.endswith(".py"):
        return 0

    res = subprocess.run(
        [VENV_PY, "-m", "ruff", "check", f"--select={RUFF_SELECT}", "--no-fix", path],
        capture_output=True,
        text=True,
    )
    # ruff not installed / failed to launch → fall back to syntax-only py_compile.
    if res.returncode not in (0, 1):
        res = subprocess.run(
            [VENV_PY, "-m", "py_compile", path], capture_output=True, text=True
        )
        if res.returncode != 0:
            print(f"Python syntax error in {path}:\n{res.stderr}", file=sys.stderr)
            return 2
        return 0

    if res.returncode == 1:  # ruff found real breakage
        print(
            f"ruff found genuine breakage in {path} (fix before running):\n"
            f"{res.stdout}{res.stderr}",
            file=sys.stderr,
        )
        return 2

    return 0


if __name__ == "__main__":
    sys.exit(main())
