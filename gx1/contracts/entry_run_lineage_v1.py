"""Immutable identity for Entry rebuild, training, and evaluation runs.

An Entry run ID is provenance, not authorization.  It binds all artifacts in
one causal lineage so mixed or stale outputs fail closed.  Evidence contracts,
not a human approval token, decide whether later stages may execute.
"""

from __future__ import annotations

import re


ENTRY_RUN_ID_SCHEMA_VERSION = "entry_run_lineage_v1"
_ENTRY_RUN_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{7,127}")
_PLACEHOLDERS = frozenset(
    {
        "TODO",
        "TBD",
        "PLACEHOLDER",
        "ENTRY_RUN_ID",
        "EXPLICIT_RUN_ID",
        "EXPLICIT_VEDTAK_ID",
    }
)


class EntryRunLineageError(RuntimeError):
    """Raised when an Entry artifact lineage ID is absent or malformed."""


def require_entry_run_id(value: object) -> str:
    """Return one normalized Entry run ID without granting any authority."""

    run_id = str(value or "").strip()
    if not run_id:
        raise EntryRunLineageError("Entry run identity missing: provide --run-id")
    if _ENTRY_RUN_ID_RE.fullmatch(run_id) is None:
        raise EntryRunLineageError(
            "--run-id must be 8-128 characters using only letters, digits, "
            "dot, underscore, colon, or hyphen"
        )
    if run_id.upper() in _PLACEHOLDERS:
        raise EntryRunLineageError("Placeholder --run-id values are forbidden")
    return run_id
