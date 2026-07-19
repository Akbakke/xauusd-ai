#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$ROOT/.venv/bin/python"

if [[ ! -x "$PY" ]]; then
  echo "FATAL: repository GX1 python missing: $PY" >&2
  echo "Restore the repository environment before running tests." >&2
  exit 2
fi

cd "$ROOT"
exec "$PY" -m pytest "$@"
