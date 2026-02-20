#!/usr/bin/env python3
"""
ONE PATH: execution, policy, datasets must use gx1.contracts.exit_io only.

No direct imports from exit_transformer_io_v1, exit_transformer_io_v2, or
exit_transformer_io_v3 in gx1/execution, gx1/policy, gx1/datasets.
Exception: gx1/contracts/** and gx1/tests/** may reference legacy for
migration/contract tests.
"""

from __future__ import annotations

from pathlib import Path


FORBIDDEN_PATTERNS = [
    "from gx1.contracts.exit_transformer_io_v1",
    "from gx1.contracts.exit_transformer_io_v2",
    "from gx1.contracts.exit_transformer_io_v3",
    "import gx1.contracts.exit_transformer_io_v1",
    "import gx1.contracts.exit_transformer_io_v2",
    "import gx1.contracts.exit_transformer_io_v3",
]

ONE_PATH_DIRS = ("execution", "policy", "datasets")


def test_one_path_no_direct_legacy_imports():
    """execution, policy, datasets must import from exit_io; no direct v1/v2/v3."""
    repo_root = Path(__file__).resolve().parents[2]
    gx1_root = repo_root / "gx1"
    hits: list[tuple[str, int, str]] = []
    for dir_name in ONE_PATH_DIRS:
        dir_path = gx1_root / dir_name
        if not dir_path.is_dir():
            continue
        for py_path in dir_path.rglob("*.py"):
            try:
                text = py_path.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            rel = py_path.relative_to(repo_root)
            for pattern in FORBIDDEN_PATTERNS:
                for i, line in enumerate(text.splitlines(), 1):
                    if pattern in line and not line.strip().startswith("#"):
                        hits.append((str(rel), i, line.strip()[:80]))
                        break
    assert not hits, (
        "ONE PATH: gx1/execution, gx1/policy, gx1/datasets must use gx1.contracts.exit_io only. "
        "No direct imports from exit_transformer_io_v1/v2/v3. Hits: " + str(hits)
    )


if __name__ == "__main__":
    test_one_path_no_direct_legacy_imports()
    print("OK: exit_io one path check passed")
