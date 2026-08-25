"""Fail-closed dependency readiness for the immutable dataset rebuild chain.

The rebuild is intentionally expensive.  It must not discover after a feature
producer starts that the virtual environment has a missing, incompatible or
unimportable direct dependency.  ``requirements.txt`` is the only version
owner; this contract reads its exact pins rather than copying them into Python.
"""
from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import json
import platform
import sys
from pathlib import Path
from typing import Any, Callable


REBUILD_DEPENDENCY_READINESS_SCHEMA_VERSION = "gx1_rebuild_dependency_readiness_v1"
REBUILD_DEPENDENCY_REQUIREMENTS_RELATIVE_PATH = "requirements.txt"
REBUILD_DEPENDENCY_MINIMUM_PYTHON = (3, 10)

# Direct runtime distributions that the offline data/feature/rebuild chain
# needs to import.  Test-only tools remain pinned and version-checked from
# requirements, but do not make this readiness path import a linter.
_RUNTIME_IMPORT_MODULES = {
    "numba": "numba",
    "numpy": "numpy",
    "pandas": "pandas",
    "pyarrow": "pyarrow",
    "PyYAML": "yaml",
    "requests": "requests",
    "scikit-learn": "sklearn",
    "scipy": "scipy",
    "torch": "torch",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _requirements_path(repo: str | Path) -> Path:
    root = Path(repo).expanduser().resolve()
    if root.is_symlink() or not root.is_dir():
        raise RuntimeError("REBUILD_DEPENDENCY_READINESS_REPO_INVALID")
    path = root / REBUILD_DEPENDENCY_REQUIREMENTS_RELATIVE_PATH
    if path.is_symlink() or not path.is_file() or path.resolve() != path:
        raise RuntimeError("REBUILD_DEPENDENCY_READINESS_REQUIREMENTS_MISSING")
    return path


def _parse_exact_requirement_pins(path: Path) -> dict[str, str]:
    """Parse the deliberately small direct-dependency file, fail closed otherwise."""

    pins: dict[str, str] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise RuntimeError("REBUILD_DEPENDENCY_READINESS_REQUIREMENTS_UNREADABLE") from exc
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#") or line.startswith("--"):
            continue
        if line.count("==") != 1:
            raise RuntimeError("REBUILD_DEPENDENCY_READINESS_REQUIREMENT_NOT_EXACT")
        name, version = (part.strip() for part in line.split("==", 1))
        if not name or not version or name in pins:
            raise RuntimeError("REBUILD_DEPENDENCY_READINESS_REQUIREMENTS_INVALID")
        pins[name] = version
    if set(_RUNTIME_IMPORT_MODULES) - set(pins):
        raise RuntimeError("REBUILD_DEPENDENCY_READINESS_RUNTIME_PIN_MISSING")
    return pins


def build_rebuild_dependency_readiness(
    *,
    repo: str | Path,
    version_lookup: Callable[[str], str] = importlib.metadata.version,
    importer: Callable[[str], Any] = importlib.import_module,
    python_version: tuple[int, int, int] | None = None,
    implementation: str | None = None,
) -> dict[str, Any]:
    """Verify exact installed pins and actual imports without loading data or CUDA."""

    requirements = _requirements_path(repo)
    pins = _parse_exact_requirement_pins(requirements)
    observed_version = python_version or tuple(sys.version_info[:3])
    observed_implementation = implementation or platform.python_implementation()
    failures: list[str] = []
    if (
        len(observed_version) != 3
        or any(not isinstance(part, int) for part in observed_version)
        or observed_version[:2] != REBUILD_DEPENDENCY_MINIMUM_PYTHON
    ):
        failures.append("python_version_not_cpython_3_10")
    if observed_implementation != "CPython":
        failures.append("python_implementation_not_cpython")

    packages: dict[str, dict[str, Any]] = {}
    for distribution, expected in pins.items():
        try:
            installed = str(version_lookup(distribution))
        except Exception as exc:
            installed = ""
            version_error = f"{type(exc).__name__}: {exc}"
        else:
            version_error = None
        import_module = _RUNTIME_IMPORT_MODULES.get(distribution)
        import_error = None
        if import_module is not None:
            try:
                importer(import_module)
            except Exception as exc:
                import_error = f"{type(exc).__name__}: {exc}"
        package_ok = installed == expected and version_error is None and import_error is None
        packages[distribution] = {
            "expected_version": expected,
            "installed_version": installed or None,
            "import_module": import_module,
            "import_ok": import_error is None if import_module is not None else None,
            "version_error": version_error,
            "import_error": import_error,
            "ok": package_ok,
        }
        if not package_ok:
            failures.append(f"dependency_not_ready:{distribution}")

    return {
        "schema_version": REBUILD_DEPENDENCY_READINESS_SCHEMA_VERSION,
        "decision": "PASS" if not failures else "FAIL",
        "requirements_path": str(requirements),
        "requirements_sha256": sha256_file(requirements),
        "python": {
            "implementation": observed_implementation,
            "version": ".".join(str(part) for part in observed_version),
            "required_major_minor": ".".join(
                str(part) for part in REBUILD_DEPENDENCY_MINIMUM_PYTHON
            ),
            "executable": str(Path(sys.executable).resolve()),
        },
        "packages": packages,
        "failures": failures,
    }


def require_rebuild_dependency_readiness(**kwargs: Any) -> dict[str, Any]:
    report = build_rebuild_dependency_readiness(**kwargs)
    if report["decision"] != "PASS":
        raise RuntimeError(
            "REBUILD_DEPENDENCY_READINESS_FAILED: "
            + json.dumps(report["failures"], sort_keys=True)
        )
    return report
