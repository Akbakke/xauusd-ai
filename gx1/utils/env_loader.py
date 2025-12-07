"\"\"\"Utilities for loading .env files into os.environ.\"\"\""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)


def _strip_quotes(value: str) -> str:
    if value.startswith(("\"", "'")) and value.endswith(("\"", "'")):
        return value[1:-1]
    return value


def load_dotenv_if_present(dotenv_path: Optional[str] = None) -> None:
    """
    If a .env file exists, load KEY=VALUE pairs into os.environ.

    Expected format (one per line):
        KEY=VALUE
        OTHER_KEY=other

    Also tolerates legacy lines like:
        export KEY="VALUE"

    This function is idempotent and silently ignores missing files.
    """

    if dotenv_path is None:
        candidate = Path(".env")
    else:
        candidate = Path(dotenv_path)

    if not candidate.exists() or not candidate.is_file():
        log.info("No .env file found (looked for %s)", candidate.resolve())
        return

    log.info("Loading .env from %s", candidate.resolve())

    for raw_line in candidate.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        if line.startswith("export "):
            line = line[len("export ") :].strip()

        if "=" not in line:
            continue  # skip malformed lines

        key, value = line.split("=", 1)
        key = key.strip()
        value = _strip_quotes(value.strip())
        if not key:
            continue
        os.environ[key] = value

