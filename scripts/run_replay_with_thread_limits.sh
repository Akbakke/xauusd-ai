#!/usr/bin/env bash
# Del 2: Replay script with thread limits to reduce segfault risk
#
# Sets recommended thread limits for BLAS/OpenMP libraries and runs replay.
# These limits reduce the risk of segfaults in pandas/numpy operations.

ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"
test "$(pwd)" = "$ROOT" || { echo "❌ FAIL: Not in repo-root (pwd=$(pwd), root=$ROOT)"; exit 1; }
echo "[RUN_CTX] root=$ROOT"
echo "[RUN_CTX] head=$(git rev-parse --short HEAD)"
echo "[RUN_CTX] whoami=$(whoami) host=$(hostname)"

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# Del 2: Set thread limits to reduce segfault risk
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_MAX_THREADS=1

echo "=================================================================================="
echo "REPLAY WITH THREAD LIMITS"
echo "=================================================================================="
echo ""
echo "Thread limits set:"
echo "  OMP_NUM_THREADS=1"
echo "  MKL_NUM_THREADS=1"
echo "  OPENBLAS_NUM_THREADS=1"
echo "  VECLIB_MAXIMUM_THREADS=1"
echo "  NUMEXPR_MAX_THREADS=1"
echo ""
echo "This reduces the risk of segfaults in pandas/numpy operations."
echo ""

# Check if arguments provided, otherwise show usage
if [ $# -eq 0 ]; then
    echo "Usage: $0 <replay_script> [script_args...]"
    echo ""
    echo "Example:"
    echo "  $0 scripts/run_mini_replay_perf.py <policy> <data_file> <output_dir>"
    echo ""
    exit 1
fi

REPLAY_SCRIPT="$1"
shift

# Run the replay script with remaining arguments
exec python3 "$REPLAY_SCRIPT" "$@"

