"""GX1 fail-closed guards (decision-artifact selection + gate enforcement).

Import these at the top of decisioning / retrain / gated entrypoints to turn
AGENTS.md guardrails into runtime guarantees that raise instead of hope.
"""
from pathlib import Path

# Repo root = parent of this package dir. All contract/gate paths resolve here,
# NOT relative to CWD, so guards work regardless of where a script is launched.
REPO_ROOT = Path(__file__).resolve().parent.parent
