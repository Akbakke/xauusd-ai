#!/usr/bin/env python3
"""GX1 PreToolUse command-safety guard.

This hook is deliberately project-local.  It provides an early, fail-closed
floor for destructive Git operations and accidental secret staging.  Runtime
data identity is enforced by GX1's hash-bound XAU_USD contracts rather than by
directory-name heuristics in a shell-command parser.

Exit 2 blocks the command; exit 0 allows it.
"""
from __future__ import annotations

import json
import os
import re
import shlex
import sys


_GIT_OPTS_WITH_VALUE = {
    "-C",
    "-c",
    "--git-dir",
    "--work-tree",
    "--namespace",
    "--exec-path",
}
_GIT_MSG_OPTS = {
    "-m",
    "--message",
    "-F",
    "--file",
    "-C",
    "--reuse-message",
    "-c",
    "--reedit-message",
}


def _git_block(segment: str) -> str | None:
    """Return a reason when a segment violates the GX1 Git safety floor."""

    try:
        tokens = shlex.split(segment)
    except Exception:
        tokens = segment.split()
    git_index = None
    for index, token in enumerate(tokens):
        if re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", token):
            continue
        if os.path.basename(token) == "git":
            git_index = index
        break
    if git_index is None:
        return None
    remainder = tokens[git_index + 1 :]

    for index, token in enumerate(remainder):
        if (
            token == "-c"
            and index + 1 < len(remainder)
            and remainder[index + 1].startswith("alias.")
        ):
            return "inline git alias definition is forbidden"
        if token.startswith("-c") and "alias." in token:
            return "inline git alias definition is forbidden"

    cursor = 0
    subcommand = None
    while cursor < len(remainder):
        token = remainder[cursor]
        if token in _GIT_OPTS_WITH_VALUE:
            cursor += 2
            continue
        if token.startswith("-"):
            cursor += 1
            continue
        subcommand = token
        break
    arguments = remainder[cursor + 1 :] if subcommand is not None else []

    if subcommand == "push":
        if any(
            flag in ("--force", "--force-with-lease", "--force-if-includes")
            for flag in arguments
        ):
            return "force push is forbidden"
        if any(re.match(r"^-[A-Za-z]*f", token) for token in arguments):
            return "force push is forbidden"
        if any(token.startswith("+") for token in arguments):
            return "force push refspec is forbidden"
    if subcommand == "commit" and "--amend" in arguments:
        return "git commit --amend is forbidden"
    if subcommand in ("filter-branch", "filter-repo"):
        return f"git {subcommand} rewrites history"
    if subcommand == "rebase" and any(
        flag in arguments for flag in ("-i", "--interactive", "--root", "--autosquash")
    ):
        return "interactive/root rebase is forbidden"
    if subcommand == "reset" and "--hard" in arguments:
        return "git reset --hard is forbidden"
    if subcommand in ("add", "commit"):
        scanned: list[str] = []
        cursor = 0
        while cursor < len(arguments):
            token = arguments[cursor]
            if token in _GIT_MSG_OPTS:
                cursor += 2
                continue
            if token.startswith(("-m", "-F")) or token.startswith(
                ("--message=", "--file=")
            ):
                cursor += 1
                continue
            scanned.append(token)
            cursor += 1
        if re.search(r"(\.env\b|credentials)", " ".join(scanned)):
            return "staging a secret is forbidden"
    return None


def _segments(command: str) -> list[str]:
    segments: list[str] = []
    for part in re.split(r"&&|\|\||;|\n|\|", command):
        segments.extend(re.split(r"&(?!>)", part))
    return [segment for segment in segments if segment.strip()]


def main() -> int:
    try:
        event = json.load(sys.stdin)
    except Exception:
        return 0
    command = event.get("tool_input", {}).get("command", "")
    if not command:
        return 0
    for segment in _segments(command):
        reason = _git_block(segment)
        if reason:
            print(
                f"BLOCKED by GX1 guard: {reason}\n  command: {command}",
                file=sys.stderr,
            )
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
