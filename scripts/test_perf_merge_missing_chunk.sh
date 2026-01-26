#!/bin/bash
# Test for perf summary merge: missing chunk summary case
#
# Runs smoke test, then deletes one chunk summary and verifies merge fails.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

echo "=================================================================================="
echo "PERF MERGE MISSING CHUNK TEST"
echo "=================================================================================="
echo ""

# Step 1: Run smoke test first
echo "[1/3] Running smoke test to generate summaries..."
bash scripts/test_perf_merge_smoke.sh "$@" > /tmp/test_perf_merge_missing_smoke.log 2>&1 || {
    echo "ERROR: Smoke test failed (prerequisite)"
    cat /tmp/test_perf_merge_missing_smoke.log
    exit 1
}

# Extract test output directory from smoke test log
TEST_OUTPUT_DIR=$(grep "^Output:" /tmp/test_perf_merge_missing_smoke.log | tail -1 | awk '{print $2}')
if [[ -z "$TEST_OUTPUT_DIR" ]] || [[ ! -d "$TEST_OUTPUT_DIR" ]]; then
    echo "ERROR: Could not find test output directory from smoke test"
    exit 1
fi

echo "Using test output: $TEST_OUTPUT_DIR"

# Step 2: Delete one chunk summary
echo ""
echo "[2/3] Deleting chunk 0 summary to simulate missing chunk..."

CHUNK_0_SUMMARY="$TEST_OUTPUT_DIR/parallel_chunks/chunk_0/REPLAY_PERF_SUMMARY.json"
if [[ ! -f "$CHUNK_0_SUMMARY" ]]; then
    echo "ERROR: Chunk 0 summary not found: $CHUNK_0_SUMMARY"
    exit 1
fi

# Backup for cleanup
BACKUP_DIR="$TEST_OUTPUT_DIR/backup_before_delete"
mkdir -p "$BACKUP_DIR"
cp "$CHUNK_0_SUMMARY" "$BACKUP_DIR/"

# Delete the summary
rm "$CHUNK_0_SUMMARY"
echo "✅ Deleted: $CHUNK_0_SUMMARY"

# Step 3: Run merge and verify it fails
echo ""
echo "[3/3] Running merge (should fail)..."

set +e  # Allow merge to fail
python3 scripts/merge_perf_summaries.py "$TEST_OUTPUT_DIR" > "$TEST_OUTPUT_DIR/merge_missing_chunk.log" 2>&1
MERGE_EXIT=$?
set -e

# U: After deletion of one chunk summary: merge exit code != 0
if [[ $MERGE_EXIT -eq 0 ]]; then
    echo "❌ TEST FAILED (U): Merge should have failed but returned exit code 0"
    echo "Merge output:"
    cat "$TEST_OUTPUT_DIR/merge_missing_chunk.log"
    exit 1
fi

# V: stderr/stdout must contain exact file name + chunk_id
if ! grep -q "Missing perf summary" "$TEST_OUTPUT_DIR/merge_missing_chunk.log"; then
    echo "❌ TEST FAILED (V): Error message doesn't mention 'Missing perf summary'"
    echo "Merge output:"
    cat "$TEST_OUTPUT_DIR/merge_missing_chunk.log"
    exit 1
fi

if ! grep -q "chunk_0" "$TEST_OUTPUT_DIR/merge_missing_chunk.log"; then
    echo "❌ TEST FAILED (V): Error message doesn't mention 'chunk_0'"
    echo "Merge output:"
    cat "$TEST_OUTPUT_DIR/merge_missing_chunk.log"
    exit 1
fi

echo "✅ Merge correctly failed with exit code $MERGE_EXIT (U)"
echo "✅ Error message contains file name and chunk_id (V)"
echo "Error message:"
grep "Missing perf summary" "$TEST_OUTPUT_DIR/merge_missing_chunk.log" | head -1

# Restore for cleanup
cp "$BACKUP_DIR/REPLAY_PERF_SUMMARY.json" "$CHUNK_0_SUMMARY"

echo ""
echo "=================================================================================="
echo "✅ MISSING CHUNK TEST PASSED"
echo "=================================================================================="
echo ""

