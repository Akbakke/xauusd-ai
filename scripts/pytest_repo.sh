#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${GX1_CANONICAL_PYTHON:-/home/andre2/venvs/gx1/bin/python}"

if [[ ! -x "$PY" ]]; then
  echo "FATAL: canonical GX1 python missing: $PY" >&2
  echo "Use one shared env instead of per-repo installs: /home/andre2/venvs/gx1" >&2
  exit 2
fi

cd "$ROOT"
exec "$PY" -m pytest "$@"
