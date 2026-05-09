#!/usr/bin/env bash
# Head-to-head V5 vs V6 bundle eval on the test split.
#
# Usage:
#   compare_v5_v6_bundles.sh <V5_BUNDLE_DIR> <V6_BUNDLE_DIR> [SPLIT]
#
# Default split is "test". Both evals run sequentially (16 GB host can't fit two).
# Output JSON paths are printed at the end so a follow-up diff is easy.

set -euo pipefail

V5_BUNDLE="${1:-}"
V6_BUNDLE="${2:-}"
SPLIT="${3:-test}"

if [[ -z "$V5_BUNDLE" || -z "$V6_BUNDLE" ]]; then
    echo "usage: $0 <V5_BUNDLE_DIR> <V6_BUNDLE_DIR> [SPLIT]" >&2
    exit 2
fi

PYTHON=/home/andre2/venvs/gx1/bin/python3
export GX1_DATA=/home/andre2/GX1_DATA
TS=$(date -u +%Y%m%dT%H%M%SZ)
V5_OUT=/tmp/eval_v5_${SPLIT}_${TS}.json
V6_OUT=/tmp/eval_v6_${SPLIT}_${TS}.json

echo "[compare] V5 bundle: $V5_BUNDLE"
echo "[compare] V6 bundle: $V6_BUNDLE"
echo "[compare] split: $SPLIT"

echo "[compare] === V5 eval ==="
"$PYTHON" -u -m gx1.scripts.evaluate_exit_v5_bundle \
    --bundle-dir "$V5_BUNDLE" --split "$SPLIT" \
    --device cuda --batch-size 256 --out-json "$V5_OUT"

echo "[compare] === V6 eval ==="
"$PYTHON" -u -m gx1.scripts.evaluate_exit_v5_bundle \
    --bundle-dir "$V6_BUNDLE" --split "$SPLIT" \
    --device cuda --batch-size 256 --out-json "$V6_OUT"

echo
echo "[compare] === side-by-side summary ==="
"$PYTHON" - <<PYEOF
import json
v5 = json.load(open("$V5_OUT"))
v6 = json.load(open("$V6_OUT"))

def keymap(d, prefix=""):
    out = {}
    for k, v in d.items():
        kk = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(keymap(v, prefix=kk + "."))
        elif isinstance(v, (int, float, str, bool)) or v is None:
            out[kk] = v
    return out

flat_v5 = keymap(v5)
flat_v6 = keymap(v6)
keys = sorted(set(flat_v5) | set(flat_v6))
print(f"{'metric':<60s} {'v5':>14s} {'v6':>14s} {'delta':>14s}")
print("-" * 105)
for k in keys:
    a, b = flat_v5.get(k), flat_v6.get(k)
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        try:
            delta = b - a
            print(f"{k:<60s} {a:>14.4f} {b:>14.4f} {delta:>+14.4f}")
        except Exception:
            print(f"{k:<60s} {str(a):>14s} {str(b):>14s} {'-':>14s}")
    else:
        if a != b:
            print(f"{k:<60s} {str(a):>14s} {str(b):>14s} {'(diff)':>14s}")
PYEOF

echo
echo "[compare] V5 JSON: $V5_OUT"
echo "[compare] V6 JSON: $V6_OUT"
