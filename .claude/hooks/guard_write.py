#!/usr/bin/env python3
"""
GX1 PreToolUse write guard. Runs BEFORE Claude writes/edits a file.

Two recurring smells from AGENTS.md:
  1. EXIT-M5-COARSENING (HARD BLOCK, exit 2): the exit pipeline is M1-native.
     Coarsening/resampling its M1 grid to M5 is "fy-fy" (feedback_exit_m1_no_arch_reorg).
     Tight pattern — does NOT trip on the LEGIT M5 multi-TF branch
     (seq_m5 / m5_proj / m5_phase / load_multi_tf / htf), only on actual
     resample/downsample of the M1 stream inside exit-transformer code.
  2. NEW-SCRIPT (SOFT, non-blocking additionalContext): creating a brand-new
     .py in a core dir. Often a duplicate of something that already exists —
     surface it so Claude names the existing file it should extend instead.
"""

import json
import os
import re
import sys

# Files that ARE the exit-transformer M1 pipeline.
EXIT_FILE_RE = re.compile(
    r"(exit_transformer|exit_io|/exits/|materialize_build_v3_training|train_exit_)",
    re.IGNORECASE,
)
# Actual coarsening of the M1 grid (NOT the legit M5 multi-TF branch).
COARSEN_RES = [
    re.compile(r"\.resample\([^)]*['\"]\s*5\s*(min|t|m)\b", re.IGNORECASE),
    re.compile(r"\bdownsample\b", re.IGNORECASE),
    re.compile(r"\bcoarsen\b", re.IGNORECASE),
    re.compile(r"m5[^\n]{0,25}\.ffill|\.ffill[^\n]{0,25}m5", re.IGNORECASE),
]
# Legit M5 usages that must NOT be treated as coarsening.
LEGIT_M5 = re.compile(
    r"seq_m5|m5_proj|m5_phase|m5_encoder|multi_tf|load_multi_tf|htf|MULTI_TF",
    re.IGNORECASE,
)
CORE_DIR_RE = re.compile(r"/GX1_ENGINE/(gx1|gx1_guards)/")

# Protected LIVE chain + SACRED transformer contracts (CLAUDE.md rule 1). Edits are FROZEN
# unless the user drops a one-shot override marker. Matches the main checkout AND the
# _cleanup worktree so the discipline holds everywhere.
PROTECTED_CORE_RE = re.compile(
    r"/GX1_ENGINE(?:_cleanup)?/gx1/(?:execution|contracts|exits/contracts|models/entry_v10|core)/"
)
ALLOW_CORE_EDIT_MARKER = "/home/andre2/src/GX1_ENGINE/.claude/ALLOW_CORE_EDIT"


def _content(event: dict) -> str:
    ti = event.get("tool_input", {})
    return ti.get("content") or ti.get("new_string") or ""


def main() -> int:
    try:
        event = json.load(sys.stdin)
    except Exception:
        return 0

    ti = event.get("tool_input", {})
    path = ti.get("file_path", "") or ""
    tool = event.get("tool_name", "")
    content = _content(event)

    # 1) HARD BLOCK: coarsening the exit's M1 grid.
    if path.endswith(".py") and EXIT_FILE_RE.search(path):
        for line in content.splitlines():
            if LEGIT_M5.search(line):
                continue  # legit M5 multi-TF branch — ignore
            if any(rx.search(line) for rx in COARSEN_RES):
                print(
                    "BLOCKED by GX1 guard: exit is M1-NATIVE — this edit looks like "
                    "coarsening/resampling the M1 grid to M5 (fy-fy). The legit M5 "
                    "multi-TF branch (seq_m5/m5_phase/multi_tf) is fine; resampling "
                    "the exit's M1 price stream is not. See AGENTS.md + "
                    f"feedback_exit_m1_no_arch_reorg.\n  offending line: {line.strip()[:120]}",
                    file=sys.stderr,
                )
                return 2

    # 1b) Protected live core / SACRED contracts (CLAUDE.md rule 1). The one-shot MARKER GATE was
    # REMOVED 2026-06-05 (user vedtak — the per-edit `touch` friction was killing the rebuild workflow).
    # Protected-core edits are now ALLOWED, but LOUDLY logged so live-chain / contract changes stay
    # visible (never silent). The trust discipline still applies (CLAUDE.md: verify in-use, ONE truth,
    # minimal change, train==serve, git-clean-before-run) — this only drops the hard block + the marker.
    if PROTECTED_CORE_RE.search(path):
        print(json.dumps({
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "additionalContext": (
                    f"⚠ PROTECTED-CORE EDIT (no longer marker-gated, user vedtak 2026-06-05): {path} is part "
                    "of the LIVE chain / SACRED transformer contracts. Allowed — but be deliberate: verify "
                    "in-use, ONE truth, minimal change, train==serve. Logged for visibility."
                ),
            }
        }))
        return 0

    # 2) SOFT WARN: creating a brand-new .py in a core dir (possible duplicate).
    if tool == "Write" and path.endswith(".py") and CORE_DIR_RE.search(path) and not os.path.exists(path):
        warn = (
            f"GX1 soft guard: you are CREATING a new script ({os.path.basename(path)}). "
            "AGENTS.md says extend the existing component, don't fork a parallel script. "
            "Before writing: confirm this is a genuinely-new shared one-truth helper, "
            "not a duplicate — and name the existing file you considered extending."
        )
        print(json.dumps({
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "additionalContext": warn,
            }
        }))
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
