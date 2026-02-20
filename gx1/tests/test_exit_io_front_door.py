#!/usr/bin/env python3
"""
TRUTH path must use gx1.contracts.exit_io (front door), not direct imports from
exit_transformer_io_v1 / _v2 / _v3. Datasets and legacy tests are allowed to keep
direct v1/v2/v3 imports for now.
"""

from __future__ import annotations

from pathlib import Path


# Direct imports from these modules in TRUTH-path code are forbidden (use exit_io).
FORBIDDEN_IMPORT_PATTERNS = [
    "from gx1.contracts.exit_transformer_io_v1",
    "from gx1.contracts.exit_transformer_io_v2",
    "from gx1.contracts.exit_transformer_io_v3",
    "import gx1.contracts.exit_transformer_io_v1",
    "import gx1.contracts.exit_transformer_io_v2",
    "import gx1.contracts.exit_transformer_io_v3",
]

# Directories that are TRUTH-path: must use exit_io for IOV3_CLEAN (no direct _v3).
# _v1/_v2 may still be used here for training/fallback until legacy is removed.
TRUTH_PATH_DIRS = ("execution", "policy", "scripts")

# Only forbid _v3 in TRUTH path (use exit_io); _v1/_v2 allowed for now in runner/training.
FORBIDDEN_V3_PATTERNS = [
    "from gx1.contracts.exit_transformer_io_v3",
    "import gx1.contracts.exit_transformer_io_v3",
]


def test_truth_path_must_not_import_exit_io_v3_directly():
    """TRUTH path (execution, policy, scripts) must import IOV3_CLEAN from exit_io, not from exit_transformer_io_v3."""
    repo_root = Path(__file__).resolve().parents[2]
    gx1_root = repo_root / "gx1"
    hits: list[tuple[str, int, str]] = []
    for dir_name in TRUTH_PATH_DIRS:
        dir_path = gx1_root / dir_name
        if not dir_path.is_dir():
            continue
        for py_path in dir_path.rglob("*.py"):
            try:
                text = py_path.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            rel = py_path.relative_to(repo_root)
            for pattern in FORBIDDEN_V3_PATTERNS:
                for i, line in enumerate(text.splitlines(), 1):
                    if pattern in line and not line.strip().startswith("#"):
                        hits.append((str(rel), i, line.strip()[:80]))
                        break
    assert not hits, (
        "TRUTH path must use gx1.contracts.exit_io, not exit_transformer_io_v3. "
        "Replace with: from gx1.contracts.exit_io import ... Hits: " + str(hits)
    )


if __name__ == "__main__":
    test_truth_path_must_not_import_exit_io_v3_directly()
    print("OK: exit_io front door check passed")
