#!/home/andre2/venvs/gx1/bin/python
# -*- coding: utf-8 -*-
"""
ENV Gate Policy Check (Anti-regression)

Scans `gx1/scripts/**` (including docs) to enforce the single-interpreter contract:
- No `#!/usr/bin/env python*` shebangs
- No `py3` mentions in scripts/docs under gx1/scripts/** (forbidden)
- Entrypoint scripts must use the absolute shebang:
    #!/home/andre2/venvs/gx1/bin/python

Exit code:
- 0: PASS (no violations)
- 2: FAIL (one or more violations)
"""

from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple


REQUIRED_PYTHON = "/home/andre2/venvs/gx1/bin/python"
REQUIRED_SHEBANG = f"#!{REQUIRED_PYTHON}"

if sys.executable != REQUIRED_PYTHON:
    raise RuntimeError(
        f"[ENV_IDENTITY_FAIL] Wrong python interpreter\n"
        f"Expected: {REQUIRED_PYTHON}\n"
        f"Actual:   {sys.executable}\n"
        f"Hint: source ~/venvs/gx1/bin/activate"
    )


@dataclass(frozen=True)
class Finding:
    path: str
    kind: str
    detail: str


def _iter_files(root: Path) -> Iterable[Path]:
    for p in root.rglob("*"):
        if p.is_dir():
            continue
        # Only scan scripts + docs
        if p.suffix.lower() in {".py", ".sh", ".md", ".json", ".txt"}:
            yield p


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        # Skip binary-ish files
        return ""


def _check_shebangs(py_path: Path, text: str) -> List[Finding]:
    findings: List[Finding] = []
    lines = text.splitlines()
    if not lines:
        return findings

    first = lines[0].strip()
    if first.startswith("#!"):
        env_py = "#!/usr/bin/env " + "python"
        env_py3 = "#!/usr/bin/env " + "python" + "3"
        if first in {env_py, env_py3}:
            findings.append(
                Finding(str(py_path), "FORBIDDEN_SHEBANG", f"line1={first!r}")
            )
        elif first != REQUIRED_SHEBANG:
            findings.append(
                Finding(
                    str(py_path),
                    "WRONG_SHEBANG",
                    f"Expected {REQUIRED_SHEBANG!r}, got {first!r}",
                )
            )
    else:
        findings.append(
            Finding(
                str(py_path),
                "MISSING_SHEBANG",
                f"Expected line1={REQUIRED_SHEBANG!r}",
            )
        )
    return findings


def _check_py3_mentions(path: Path, text: str) -> List[Finding]:
    findings: List[Finding] = []
    if not text:
        return findings

    token = "python" + "3"
    if token in text:
        # Find first 3 matches with crude context (line numbers)
        lines = text.splitlines()
        hits: List[Tuple[int, str]] = []
        for i, line in enumerate(lines, start=1):
            if token in line:
                hits.append((i, line.strip()))
                if len(hits) >= 3:
                    break
        detail = "; ".join([f"L{i}:{s}" for i, s in hits])
        findings.append(Finding(str(path), "FORBIDDEN_TOKEN", detail))

    bad_path = "/home/andre2/venvs/gx1/bin/" + "python" + "3"
    if bad_path in text:
        findings.append(Finding(str(path), "FORBIDDEN_PATH", "contains py3 path"))

    return findings


def main() -> int:
    repo_root = Path(__file__).resolve().parents[3]  # .../src/GX1_ENGINE
    scripts_root = repo_root / "gx1" / "scripts"

    if not scripts_root.exists():
        print(f"ERROR: scripts root not found: {scripts_root}", file=sys.stderr)
        return 2

    findings: List[Finding] = []

    for p in _iter_files(scripts_root):
        text = _read_text(p)
        # Global checks
        findings.extend(_check_py3_mentions(p, text))
        # Entrypoint check: all gx1/scripts/**/*.py must use required shebang
        if p.suffix.lower() == ".py":
            findings.extend(_check_shebangs(p, text))

    if not findings:
        print("[ENV_GATE_POLICY_CHECK] PASS")
        print(f"REQUIRED_PYTHON={REQUIRED_PYTHON}")
        return 0

    print("[ENV_GATE_POLICY_CHECK] FAIL")
    print(f"REQUIRED_PYTHON={REQUIRED_PYTHON}")
    print(f"FINDINGS={len(findings)}")
    for f in findings[:200]:
        print(f"- {f.kind}: {f.path} :: {f.detail}")
    if len(findings) > 200:
        print(f"... +{len(findings) - 200} more")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

