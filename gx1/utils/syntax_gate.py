#!/usr/bin/env python3
"""
Syntax gate for GX1.

Goal:
- Deterministic syntax/indent check over the gx1 package directory only (not the whole repo).
- Always write SYNTAX_AUDIT_REPORT.md (to output_dir when possible; /tmp fallback when not).
- In TRUTH/SMOKE: hard-fail early with SYNTAX_FATAL.json + exit code 2 if ANY .py has syntax errors.

This is designed to prevent "start replay, crash later" failures caused by latent syntax errors.
"""

from __future__ import annotations

import compileall
import json
import os
import py_compile
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class SyntaxFailure:
    """A single syntax/indent failure record."""

    file: str
    line: Optional[int]
    message: str


def _now_utc_compact() -> str:
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S")


def _safe_mkdir(path: Path) -> bool:
    try:
        path.mkdir(parents=True, exist_ok=True)
        return True
    except Exception:
        return False


def _write_text_atomic(path: Path, text: str) -> None:
    tmp_path = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with open(tmp_path, "w", encoding="utf-8") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)


def _write_json_atomic(path: Path, obj: Dict[str, Any]) -> None:
    tmp_path = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)


def _fallback_dir() -> Path:
    root = Path(tempfile.gettempdir()) / "gx1_syntax_gate"
    # Always try to create the fallback root; if that fails, use /tmp directly.
    if not _safe_mkdir(root):
        root = Path("/tmp")
    run_dir = root / f"run_{_now_utc_compact()}_{os.getpid()}"
    _safe_mkdir(run_dir)
    return run_dir


def _choose_report_dir(output_dir: Path) -> Path:
    """
    Prefer output_dir if we can create it; otherwise fallback to /tmp.
    """
    try:
        if output_dir is not None:
            if _safe_mkdir(output_dir):
                return output_dir
    except Exception:
        pass
    return _fallback_dir()


def _collect_failures_py_compile(target_dir: Path) -> List[SyntaxFailure]:
    """
    Deterministic syntax-check across all .py in target_dir using py_compile.

    We intentionally avoid importing gx1 modules here.
    """
    failures: List[SyntaxFailure] = []
    py_files = sorted(p for p in target_dir.rglob("*.py") if p.is_file())

    with tempfile.TemporaryDirectory(prefix="gx1_syntax_gate_pyc_") as td:
        tmp_dir = Path(td)
        for i, p in enumerate(py_files):
            try:
                # Write compiled output into a temp directory (avoid touching source tree).
                cfile = tmp_dir / f"{p.name}.{i}.pyc"
                py_compile.compile(str(p), cfile=str(cfile), doraise=True)
            except py_compile.PyCompileError as exc:
                e = getattr(exc, "exc_value", exc)
                if isinstance(e, SyntaxError):
                    msg = getattr(e, "msg", "SyntaxError")
                    text = getattr(e, "text", None)
                    if text:
                        msg = f"{msg}: {text.strip()}"
                    failures.append(SyntaxFailure(file=str(p), line=getattr(e, "lineno", None), message=msg))
                else:
                    failures.append(SyntaxFailure(file=str(p), line=None, message=str(exc)))
            except Exception as e:
                # Unexpected failure during read/compile; treat as fatal for the gate.
                failures.append(SyntaxFailure(file=str(p), line=None, message=f"{type(e).__name__}: {e}"))

    return failures


def _render_audit_md(
    *,
    status: str,
    target_dir: Path,
    failures: List[SyntaxFailure],
) -> str:
    lines: List[str] = []
    lines.append("# SYNTAX_AUDIT_REPORT")
    lines.append("")
    lines.append(f"STATUS: **{status}**")
    lines.append("")
    lines.append("## Environment")
    lines.append("")
    lines.append(f"- sys.executable: `{sys.executable}`")
    lines.append(f"- sys.version: `{sys.version.replace(os.linesep, ' ')}`")
    lines.append(f"- cwd: `{Path.cwd()}`")
    lines.append(f"- target_dir: `{target_dir}`")
    lines.append("")
    lines.append("## Failures")
    lines.append("")
    lines.append("| file | line | error |")
    lines.append("|---|---:|---|")
    if failures:
        for f in failures:
            line = "" if f.line is None else str(f.line)
            err = f.message.replace("\n", " ").replace("\r", " ")
            lines.append(f"| `{f.file}` | {line} | {err} |")
    else:
        lines.append("| *(none)* |  |  |")
    lines.append("")
    return "\n".join(lines)


def run_syntax_gate_or_fatal(output_dir: Path, truth_or_smoke: bool) -> None:
    """
    Run syntax gate over gx1/ and hard-fail (exit code 2) in TRUTH/SMOKE if any failures.

    Args:
        output_dir: Intended run output directory (preferred report/capsule location)
        truth_or_smoke: If False, gate is a no-op (does not write).
    """
    if not truth_or_smoke:
        return

    target_dir = Path(__file__).resolve().parents[1]  # .../gx1
    report_dir = _choose_report_dir(output_dir)
    report_path = report_dir / "SYNTAX_AUDIT_REPORT.md"
    fatal_path = report_dir / "SYNTAX_FATAL.json"

    # 1) compileall pass (boolean) - prints only errors with quiet=1.
    #    Note: compileall may emit .pyc; this is acceptable for a compile gate.
    compileall_ok = compileall.compile_dir(str(target_dir), quiet=1)

    # 2) deterministic failure collection for exact file/line/message
    failures = _collect_failures_py_compile(target_dir)
    failures_sorted = sorted(failures, key=lambda x: (x.file, x.line or -1, x.message))

    status = "PASS" if (compileall_ok and not failures_sorted) else "FAIL"
    md = _render_audit_md(status=status, target_dir=target_dir, failures=failures_sorted)
    _write_text_atomic(report_path, md)

    if status == "PASS":
        return

    capsule = {
        "status": "FAIL",
        "failing_files": [
            {"file": f.file, "line": f.line, "message": f.message} for f in failures_sorted
        ],
        "sys.executable": sys.executable,
        "sys.version": sys.version,
        "cwd": str(Path.cwd()),
        "target_dir": str(target_dir),
        "report_path": str(report_path),
    }
    _write_json_atomic(fatal_path, capsule)

    # Exit code 2 as required.
    raise SystemExit(2)


if __name__ == "__main__":
    # Smoke test: run gate against gx1/ and write report to /tmp unless OUTPUT_DIR is provided.
    out = Path(os.getenv("OUTPUT_DIR", str(_fallback_dir())))
    try:
        run_syntax_gate_or_fatal(output_dir=out, truth_or_smoke=True)
        print(f"[SYNTAX_GATE] PASS report={out / 'SYNTAX_AUDIT_REPORT.md'}")
    except SystemExit as e:
        print(f"[SYNTAX_GATE] FAIL exit_code={e.code} report={out / 'SYNTAX_AUDIT_REPORT.md'} fatal={out / 'SYNTAX_FATAL.json'}")
