#!/home/andre2/venvs/gx1/bin/python
# -*- coding: utf-8 -*-
"""
ENV Gate Policy Check (Anti-regression)

Default: checks only the canonical wrapper set (ONE UNIVERSE). Use --full-scan to
scan all of gx1/scripts/** (legacy scripts may fail).

Rules:
- No `#!/usr/bin/env python*` shebangs (for .py entrypoints)
- No `py3` mentions in checked files (forbidden)
- Entrypoint .py scripts must use the absolute shebang:
    #!/home/andre2/venvs/gx1/bin/python

Exit code:
- 0: PASS (no violations)
- 2: FAIL (one or more violations)
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple


REQUIRED_PYTHON = "/home/andre2/venvs/gx1/bin/python"
REQUIRED_SHEBANG = f"#!{REQUIRED_PYTHON}"

# Canonical wrapper set (ONE UNIVERSE – run_env_gate_policy_check.sh WRAPPERS + this checker)
# Paths must exist; missing → hard-fail.
CANONICAL_WRAPPERS = [
    "gx1/scripts/tools/run_env_gate_policy_check.sh",
    "gx1/scripts/tools/env_gate_policy_check.py",
    "gx1/scripts/run_phase_a.sh",
    "gx1/scripts/run_phase_b.sh",
    "gx1/scripts/run_phase_c.sh",
    "gx1/scripts/run_build_year_metrics.sh",
    "gx1/scripts/run_replay_eval_chain_compute.sh",
]

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


def _iter_files_full(root: Path) -> Iterable[Path]:
    """Scan entire gx1/scripts/** (legacy --full-scan)."""
    for p in root.rglob("*"):
        if p.is_dir():
            continue
        if p.suffix.lower() in {".py", ".sh", ".md", ".json", ".txt"}:
            yield p


def _iter_files_allowlist(repo_root: Path) -> List[Path]:
    """Only canonical wrapper paths. Hard-fail if any missing."""
    missing: List[str] = []
    out: List[Path] = []
    for rel in CANONICAL_WRAPPERS:
        p = repo_root / rel
        if not p.exists():
            missing.append(rel)
        else:
            out.append(p)
    if missing:
        print("[ENV_GATE_POLICY_CHECK] FAIL", file=sys.stderr)
        print(f"REQUIRED_PYTHON={REQUIRED_PYTHON}", file=sys.stderr)
        print("Missing canonical wrappers:", file=sys.stderr)
        for m in missing:
            print(f"  - {m}", file=sys.stderr)
        sys.exit(2)
    return out


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
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
    parser = argparse.ArgumentParser(description="ENV gate policy check")
    parser.add_argument(
        "--full-scan",
        action="store_true",
        help="Scan all gx1/scripts/** (default: only canonical wrappers)",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[3]
    scripts_root = repo_root / "gx1" / "scripts"

    if not scripts_root.exists():
        print(f"ERROR: scripts root not found: {scripts_root}", file=sys.stderr)
        return 2

    if args.full_scan:
        files_to_check = list(_iter_files_full(scripts_root))
    else:
        files_to_check = _iter_files_allowlist(repo_root)

    findings: List[Finding] = []

    for p in files_to_check:
        text = _read_text(p)
        findings.extend(_check_py3_mentions(p, text))
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
