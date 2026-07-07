#!/usr/bin/env python3
"""
Environment identity gate for GX1 (TRUTH/SMOKE).

Purpose:
- Prove which Python / venv is being used.
- Fail early with a clear capsule if the wrong interpreter is active (e.g. pandas missing).

Contract (when truth_or_smoke=True):
- Always write ENV_IDENTITY.md atomically to output_dir (or /tmp fallback if output_dir not usable yet).
- Import-test: pandas, numpy, pyarrow (at minimum pandas + numpy) and write versions if OK.
- If pandas import fails: write ENV_IDENTITY_FATAL.json capsule and exit code 2 with message:
  "Wrong Python/venv. Do NOT install packages blindly. Activate correct env."
"""

from __future__ import annotations

import json
import os
import platform
import site
import subprocess
import sys
import tempfile
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


@dataclass(frozen=True)
class ImportProbe:
    """Result of an import probe."""

    name: str
    ok: bool
    version: Optional[str]
    error: Optional[str]
    traceback: Optional[str]


def _now_utc_compact() -> str:
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S")


def _safe_mkdir(path: Path) -> bool:
    try:
        path.mkdir(parents=True, exist_ok=True)
        return True
    except Exception:
        return False


def _fallback_dir() -> Path:
    root = Path(tempfile.gettempdir()) / "gx1_env_identity_gate"
    if not _safe_mkdir(root):
        root = Path("/tmp")
    run_dir = root / f"run_{_now_utc_compact()}_{os.getpid()}"
    _safe_mkdir(run_dir)
    return run_dir


def _choose_report_dir(output_dir: Path) -> Path:
    try:
        if output_dir is not None and _safe_mkdir(output_dir):
            return output_dir
    except Exception:
        pass
    return _fallback_dir()


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


def _run_pip_version() -> Dict[str, Any]:
    """
    Run "<python> -m pip -V" under the current interpreter and capture output.
    """
    cmd = [sys.executable, "-m", "pip", "-V"]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=False)
        return {
            "cmd": " ".join(cmd),
            "returncode": res.returncode,
            "stdout": (res.stdout or "").strip(),
            "stderr": (res.stderr or "").strip(),
        }
    except Exception as e:
        return {
            "cmd": " ".join(cmd),
            "returncode": None,
            "stdout": "",
            "stderr": f"{type(e).__name__}: {e}",
        }


def _probe_import(name: str) -> ImportProbe:
    try:
        mod = __import__(name)
        version = getattr(mod, "__version__", None)
        return ImportProbe(name=name, ok=True, version=str(version) if version is not None else None, error=None, traceback=None)
    except Exception as e:
        return ImportProbe(
            name=name,
            ok=False,
            version=None,
            error=f"{type(e).__name__}: {e}",
            traceback=traceback.format_exc(),
        )


def _render_md(
    *,
    report_dir: Path,
    pip_info: Dict[str, Any],
    probes: List[ImportProbe],
) -> str:
    lines: List[str] = []
    lines.append("# ENV_IDENTITY")
    lines.append("")
    lines.append("## Core")
    lines.append("")
    lines.append(f"- sys.executable: `{sys.executable}`")
    lines.append(f"- sys.version: `{sys.version.replace(os.linesep, ' ')}`")
    lines.append(f"- platform: `{platform.platform()}`")
    try:
        uname = platform.uname()
        lines.append(f"- uname: `{uname}`")
    except Exception as e:
        lines.append(f"- uname: `ERROR {type(e).__name__}: {e}`")
    lines.append(f"- cwd: `{Path.cwd()}`")
    lines.append(f"- report_dir: `{report_dir}`")
    lines.append("")
    lines.append("## site / sys.path")
    lines.append("")
    try:
        pkgs = site.getsitepackages()
        lines.append("- site.getsitepackages():")
        for p in pkgs:
            lines.append(f"  - `{p}`")
    except Exception as e:
        lines.append(f"- site.getsitepackages(): `ERROR {type(e).__name__}: {e}`")
    lines.append("")
    lines.append("- sys.path head (15):")
    for p in sys.path[:15]:
        lines.append(f"  - `{p}`")
    lines.append("")
    lines.append("## pip")
    lines.append("")
    lines.append(f"- cmd: `{pip_info.get('cmd')}`")
    lines.append(f"- returncode: `{pip_info.get('returncode')}`")
    if pip_info.get("stdout"):
        lines.append(f"- stdout: `{pip_info['stdout']}`")
    if pip_info.get("stderr"):
        lines.append(f"- stderr: `{pip_info['stderr']}`")
    lines.append("")
    lines.append("## Import probes")
    lines.append("")
    lines.append("| module | ok | version | error |")
    lines.append("|---|---|---|---|")
    for pr in probes:
        ok = "YES" if pr.ok else "NO"
        ver = pr.version or ""
        err = (pr.error or "").replace("\n", " ").replace("\r", " ")
        lines.append(f"| `{pr.name}` | {ok} | {ver} | {err} |")
    lines.append("")
    # Append tracebacks for failures (high signal)
    failed = [p for p in probes if not p.ok and p.traceback]
    if failed:
        lines.append("## Tracebacks (failed imports)")
        lines.append("")
        for pr in failed:
            lines.append(f"### {pr.name}")
            lines.append("")
            lines.append("```")
            lines.append(pr.traceback.rstrip())
            lines.append("```")
            lines.append("")
    return "\n".join(lines)


def run_env_identity_gate_or_fatal(output_dir: Path, truth_or_smoke: bool) -> None:
    """
    TRUTH/SMOKE gate: prove Python identity and fail fast if the wrong interpreter is active.
    """
    if not truth_or_smoke:
        return

    report_dir = _choose_report_dir(output_dir)
    md_path = report_dir / "ENV_IDENTITY.md"
    fatal_path = report_dir / "ENV_IDENTITY_FATAL.json"

    pip_info = _run_pip_version()
    probes = [
        _probe_import("pandas"),
        _probe_import("numpy"),
        _probe_import("pyarrow"),
    ]

    md = _render_md(report_dir=report_dir, pip_info=pip_info, probes=probes)
    _write_text_atomic(md_path, md)

    pandas_probe = next((p for p in probes if p.name == "pandas"), None)
    if pandas_probe is not None and not pandas_probe.ok:
        msg = "Wrong Python/venv. Do NOT install packages blindly. Activate correct env."
        capsule = {
            "status": "FAIL",
            "message": msg,
            "sys.executable": sys.executable,
            "sys.version": sys.version,
            "platform": platform.platform(),
            "cwd": str(Path.cwd()),
            "report_dir": str(report_dir),
            "report_path": str(md_path),
            "pip": pip_info,
            "import_probes": [
                {
                    "name": p.name,
                    "ok": p.ok,
                    "version": p.version,
                    "error": p.error,
                    "traceback": p.traceback,
                }
                for p in probes
            ],
        }
        _write_json_atomic(fatal_path, capsule)
        raise SystemExit(2)


if __name__ == "__main__":
    out = Path(os.getenv("OUTPUT_DIR", str(_fallback_dir())))
    try:
        run_env_identity_gate_or_fatal(output_dir=out, truth_or_smoke=True)
        print(f"[ENV_IDENTITY_GATE] PASS report={out / 'ENV_IDENTITY.md'}")
    except SystemExit as e:
        print(f"[ENV_IDENTITY_GATE] FAIL exit_code={e.code} report={out / 'ENV_IDENTITY.md'} fatal={out / 'ENV_IDENTITY_FATAL.json'}")
